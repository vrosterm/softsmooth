import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np


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
model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))

def epoch_params(pretrained, model_sigma, model_mu, loader, *args):
    '''Learns the sigma and mu neural nets. Currently similar code below to epoch_adversarial'''
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device) # X is size (100,1,28,28)- batch of images
        # Need to implement a smoothing function including model_sigma and model_mu outputs
        sigma = model_sigma(X)  # 100x784 tensor- X is (100,1,28,28)
        sigma_diag = torch.zeros((len(sigma),len(sigma[0]),len(sigma[0]))).to(device)
        sigma_diag[torch.arange(len(sigma))[:, None], torch.arange(len(sigma[0])), torch.arange(len(sigma[0]))] = sigma
        sigma_diag = sigma_diag.detach().cpu().numpy()

        mu = model_mu(X)        # 100x784 tensor
        mu = mu.detach().cpu().numpy()

        rng = np.random.default_rng()
        n_samples = 100
        epsilon = np.ndarray((len(X),n_samples,28,28))
        for n in range(len(X)):
            temp = rng.multivariate_normal(mu[n],sigma_diag[n],size=(n_samples))   # mu must be 1D
            temp = np.reshape(temp,(n_samples,28,28))
            epsilon[n] = temp

        epsilon_torch = torch.from_numpy(epsilon).double().to(device)

        X = X.expand(-1, n_samples, -1, -1) # n_samples of each image (second dimension)
        scores = pretrained(X+epsilon_torch)

        raise KeyboardInterrupt # ending loop for testing purposes
        yp = pretrained(X)

        # Use current mu, sigma to get current radii w/ smoothing function
        # Use current radii to compute next mu, sigma

        # Computing loss- sum of radii
        num_radii = 10 # arbitrary, num of radii to average over (N)
        for n in range(num_radii):
        
            loss = nn.CrossEntropyLoss()(yp,y) # Make this custom

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Copied from train_save_smooth
epsilon = 0.1  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

# Train and save models if not already saved
if not os.path.exists("model_IDRS.pt"):
    opt = optim.SGD(model_dnn_2.parameters(), lr=0.1)
    for _ in range(10):
        epoch_params(model_dnn_2, model_sigma, model_mu, train_loader)
    torch.save(model_dnn_2.state_dict(), "model_IDRS.pt")