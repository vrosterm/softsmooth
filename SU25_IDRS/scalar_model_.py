import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from scipy.stats import chi2
from scalar_smooth import scalar_smoothing
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Bias(nn.Module):
    '''https://discuss.pytorch.org/t/bias-only-layer/167523
    https://discuss.pytorch.org/t/learnable-bias-layer/4221
    Currently just adds a constant bias. Linked above are potentially helpful
    forum posts to make the bias learnable.'''
    
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1)*0.1)
    def forward(self, x):
        return x.add(self.bias)

    
class ReLUBias(nn.Module):
    def __init__(self, min_sig_value=0.65):
        super().__init__()
        self.bias = min_sig_value
        self.min_sig_value = min_sig_value

    def forward(self, x):
        x = F.relu(x) + self.bias   # Applying ReLU and bias
        return x  #(Batch Size, 1)

model_mu_sig = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(),
    nn.Linear(200,1), ReLUBias()
).to(device)

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# Loading the pretrained model.
pretrained = nn.Sequential(
    nn.Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

pretrained.load_state_dict(torch.load("softsmooth/SU25_BASECODE/models/dnn_2_l2_pgd_epsilon_1.pt", map_location=device, weights_only=True))

#Chi-square PDF
def chi2_pdf(x, dof):
    return chi2.pdf(x, df=dof)

#Chi-square CDF
def chi2_cdf(x, dof):
    return chi2.cdf(x, df=dof)

#Chi-squre CDF inverse
def chi2_cdf_inv(p, dof):
    return chi2.ppf(p, df=dof)

#Derivative of phi wrt x (constant mu)
def phi_derivative(x, mu):
    temp = torch.zeros(1)
    temp[0] = 2 * np.pi
    return -x * torch.exp(-((x - mu) ** 2) / 2) / torch.sqrt(temp)

# Inverse of phi (constant mu)
def phi_inv(x, mu):
    if type(x) is float:
        temp = torch.zeros(1)
        temp[0] = 2 * x - 1
    elif type(x) is torch.Tensor:
        temp = 2 * x - 1
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(temp)

def epoch_params(pretrained, model_params, loader, lam=0.01, L=0.5, beta=1, p_min=10**(-7), dof=784):
    '''Learns the combined sigma and mu neural net.'''
    total_loss, total_err = 0.,0.
    acr = []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        batch_size = X.shape[0]
        mu = torch.zeros(batch_size, 784).to(device)  # Zero mean per image
        
        sigma = model_params(X)  # (batch_size, 1)
        sigma_vals = sigma.squeeze(-1) #(batch_size,)
        dim = 784
        #print(sigma)

        # Batch of sigmas for each image
        sigma_diag = sigma_vals.squeeze(-1)  # shape: (batch_size,)

        # Pass to randomized smoothing
        g, yp = scalar_smoothing(pretrained, sigma_diag, X, n_samples=1000, beta=beta, p_min=p_min)
        yp_tensor = torch.tensor(yp, device=y.device)
        
        #Computing L_final using section 4 math
        sigma_min = model_params[-1].min_sig_value
        C = chi2_cdf_inv(1-p_min, dof) * chi2_pdf(chi2_cdf_inv(1-p_min, dof), 1) / phi_derivative(phi_inv(p_min, 0), 0)
        L_final = (1 / sigma_min + 2 * L * C / sigma_min).to(device)

        # Computing certified radii for each image
        radii = torch.zeros((len(X)))
        radii = (phi_inv(g.values[:,0], 0) - phi_inv(g.values[:,1], 0)) / (2 * L_final)
        radii = radii*(yp_tensor==y)

        spec_reg = 0.0
        
        for layer in model_params:
            if isinstance(layer, nn.Linear):
                spec_norm = torch.linalg.matrix_norm(layer.weight)
                spec_reg += spec_norm
        
        # Computing ACR/loss.
        acr.append((sum(radii)/len(radii)).detach().cpu().item())
        loss = torch.zeros(1)
        loss = -acr[-1] + lam * spec_reg

        num_linear_layers = sum(1 for layer in model_params if isinstance(layer, nn.Linear)) 
        L_const = L ** (1 / num_linear_layers) 

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            #New weight normalization step without parsing through parameter names
            for layer in model_params:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    spec_norm = torch.linalg.matrix_norm(weight)
                    norm_weight = L_const * weight / spec_norm
                    layer.weight.data.copy_(norm_weight)
        
        total_err += (yp_tensor != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    cert_rad = sum(acr)/len(acr)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), cert_rad

print("Begin training")
t = time.time()
# Train and save models if not already saved
if not os.path.exists("model_IDRS.pt"):
    opt = optim.SGD(model_mu_sig.parameters(), lr=0.1)
    t1 = time.time()
    for n in range(10):
        t0 = t1
        err, loss, acr = epoch_params(pretrained, model_mu_sig, train_loader, lam=0.01, L=0.5, beta=1, p_min=10**(-7), dof=784)
        t1 = time.time()
        print(f"Epoch {n+1}:\tTime: {(t1-t0)/60} minutes")
        print(f"Epoch {n+1}:\tAccuracy: {1-err}\tLoss: {loss}\tACR: {acr}")
    print(f"Total time: {(t1-t)/60} minutes, {(t1-t)/3600} hours")
    torch.save(model_mu_sig.state_dict(), "model_IDRS.pt")
