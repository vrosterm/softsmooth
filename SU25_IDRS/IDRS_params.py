import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from IDRS_smooth import IDRS_matrices

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

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# Loading the pretrained model. Currently the 2 layer NN.
model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(), 
                                nn.Linear(200,10)).to(device)
model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt", map_location=device, weights_only=True))

def epoch_params(pretrained, model_sigma, model_mu, loader, *args, lam=0.3, L=1.0,):
    '''Learns the sigma and mu neural nets. Incomplete.'''
    total_loss, total_err = 0.,0.
    for X,y in loader:
        # Getting initial X, y, sigma, mu tensors
        X,y = X.to(device), y.to(device) # X is shape (100,1,28,28)- batch of images
        sigma = model_sigma(X)  # (100, 784) tensor- X is (100,1,28,28)
        sigma_diag = np.zeros((len(sigma),len(sigma[0]),len(sigma[0])))
        sigma_diag[np.arange(len(sigma))[:, None], np.arange(len(sigma[0])), np.arange(len(sigma[0]))] = sigma.detach().cpu().numpy()
        # Squaring the given sigma^1/2 matrices
        sigma_diag = np.matmul(sigma_diag,sigma_diag)

        mu = model_mu(X)        # (100, 784) tensor
        mu = mu.detach().cpu().numpy()

        # Calling new randomized smoothing function. g is the top 2 items, yp is predicted labels.
        g, yp = IDRS_matrices(pretrained, mu, sigma_diag, X, n_samples=50)
               
        # Computing certified radii for each image
        radii = torch.zeros((len(X)))
        for n in range(len(X)):
            # Only overwrite the 0 if the predicted class equals the actual class
            if yp[n] == y[n]:
                radii[n] = (g.values[n][0].item() - g.values[n][1].item()) / 2  # Still need to incorporate Lipschitz constant
        
        # Computing ACR/loss. Could combine into one line if we want.
        acr = (sum(radii)/len(radii)).detach().cpu().item()
        loss = -acr
       
        spec_reg = 0.0
        for layer in pretrained:
            if isinstance(layer, nn.Linear):
                spec_norm = torch.linalg.matrix_norm(layer.weight) #matrix_norm is more than 10x faster than SVD
                spec_reg += spec_norm

        loss += lam * spec_reg
        num_linear_layers = sum(1 for layer in model_sigma if isinstance(layer, nn.Linear)) # Will want to make this the layers in the combined mu/sigma model
        L_const = L ** (1 / num_linear_layers) # We might want to feed our chosen L into the function parameters
        
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Rewrite below once mu and sigma models are a single model
            spectral_sigma = []
            spectral_mu = []
            
            #New weight normalization step without parsing through parameter names
            for layer in model_sigma:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    spec_norm = torch.linalg.matrix_norm(weight)
                    spectral_sigma.append(spec_norm.item())
                    norm_weight = L_const * weight / spec_norm
                    layer.weight.data.copy_(norm_weight)

            for layer in model_mu:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    spec_norm = torch.linalg.matrix_norm(weight)
                    spectral_mu.append(spec_norm.item())
                    norm_weight = L_const * weight / spec_norm
                    layer.weight.data.copy_(norm_weight)

        
        yp_tensor = torch.tensor(yp, device=y.device)
        total_err += (yp_tensor != y).sum().item()
        total_loss += loss.item() * X.shape[0] # Add on lambda*sum(spectralnorms). We have norms for both mu and sigma.
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Copied from train_save_smooth
training_epsilon = 0.05  # Maximum perturbation
epsilon = 0.1  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

# Train and save models if not already saved
if not os.path.exists("model_IDRS.pt"):
    opt = optim.SGD(model_dnn_2.parameters(), lr=0.1)
    for _ in range(10):
        epoch_params(model_dnn_2, model_sigma, model_mu, train_loader, training_epsilon, alpha, num_iter, lam=0.3, L=1.0)
    torch.save(model_dnn_2.state_dict(), "model_IDRS.pt")
