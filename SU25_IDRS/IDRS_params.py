import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from IDRS_smooth import IDRS_matrices
import time

t = time.time()

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
    '''ReLU and Bias layer applied to only the second half of the neural net
    (the half with sigma)
    
    Learnable bias layer modified from https://discuss.pytorch.org/t/learnable-bias-layer/4221'''
    
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1)*0.1)
        self.lin1 = nn.Linear(784,1)
    def forward(self, x):
        # In our current mu/sigma network, x is a FloatTensor of shape (100,1568).
        L = len(x[0])//2
        sigma_vals = x[:,L:]    # (100, 784) tensor
        sigma_vals = F.relu(sigma_vals) # ReLU
        #bias = self.lin1(sigma_vals)    # (100, 1) tensor
        sigma_vals += 0.1              # Bias. Constant for now
        x[:,L:] = sigma_vals
        return x

model_sigma = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,784), nn.ReLU(), Bias()
).to(device)

model_mu = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,784)
).to(device)

model_mu_sig = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,1568), ReLUBias()
).to(device)

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# Loading the pretrained model. Currently the  layer NN.
pretrained = nn.Sequential(nn.Flatten(), nn.Linear(784,200), nn.ReLU(), 
                            nn.Linear(200,100), nn.ReLU(),
                            nn.Linear(100,100), nn.ReLU(),
                            nn.Linear(100,10)).to(device)
pretrained.load_state_dict(torch.load("softsmooth/SU25_BASECODE/models/dnn_4_l2_pgd_epsilon_1.pt", map_location=device, weights_only=True))

def epoch_params(pretrained, model_params, loader, lam=0.01, L=10.0,):
    '''Learns the sigma and mu neural nets.'''
    total_loss, total_err = 0.,0.
    acr = []

    #Computing the Lipschitz constant for the pretrained model
    lip_g = 1.0
    for layer in pretrained:
        if isinstance(layer, nn.Linear):
            weight_norm = torch.linalg.matrix_norm(layer.weight)
            lip_g *= weight_norm
    print(f"lip_g: {lip_g}")

    L_max = lip_g * (1 + ((L ** 2 + L ** 2)**(1/2))) # Lipschitz constant for the mu/sigma model
    for X,y in loader:
        # Getting initial X, y, output tensors
        X,y = X.to(device), y.to(device) # X is shape (100,1,28,28)- batch of images

        outputs = model_params(X)  # (100, 1568) tensor- X is (100,1,28,28)
        
        # Organizing mu from outputs. Means are the first 784 values.
        mu = outputs[:,:(len(outputs[0])//2)] # (100, 784) tensor
        mu = mu.detach().cpu().numpy()
        
        #Organizing sigma from outputs. Sigma is the second 784 values.
        sigma = outputs[:,(len(outputs[0])//2):] # (100, 784) tensor
        sigma_diag = np.zeros((len(sigma),len(sigma[0]),len(sigma[0])))
        sigma_diag[np.arange(len(sigma))[:, None], np.arange(len(sigma[0])), np.arange(len(sigma[0]))] = sigma.detach().cpu().numpy()
        # Squaring the given sigma^1/2 matrices
        sigma_diag = np.matmul(sigma_diag,sigma_diag)
        
        # Calling new randomized smoothing function. g is the top 2 items, yp is predicted labels.
        g, yp = IDRS_matrices(pretrained, mu, sigma_diag, X, n_samples=50)
        yp_tensor = torch.tensor(yp, device=y.device)

        # Computing certified radii for each image
        radii = torch.zeros((len(X)))
        radii = (g.values[:,0] - g.values[:,1]) / (2 * L_max)
        radii = radii*(yp_tensor==y)

        spec_reg = 0.0
        
        for layer in model_params:
            if isinstance(layer, nn.Linear):
                spec_norm = torch.linalg.matrix_norm(layer.weight) #matrix_norm is more than 10x faster than SVD
                spec_reg += spec_norm
        
        # Computing ACR/loss.
        acr.append((sum(radii)/len(radii)).detach().cpu().item())
        loss = torch.zeros(1)
        loss = -acr[-1] + lam * spec_reg

        num_linear_layers = sum(1 for layer in model_params if isinstance(layer, nn.Linear)) # Will want to make this the layers in the combined mu/sigma model
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
        err, loss, acr = epoch_params(pretrained, model_mu_sig, train_loader, lam=0.01, L=10000.0)
        t1 = time.time()
        print(f"Epoch {n+1}:\tTime: {(t1-t0)/60} minutes")
        print(f"Epoch {n+1}:\tAccuracy: {1-err}\tLoss: {loss}\tACR: {acr}")
    print(f"Total time: {(t1-t)/60} minutes, {(t1-t)/3600} hours")
    torch.save(model_mu_sig.state_dict(), "model_IDRS.pt")