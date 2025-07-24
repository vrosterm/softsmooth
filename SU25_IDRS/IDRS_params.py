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
    '''Learns the sigma and mu neural nets. Incomplete.'''
    total_loss, total_err = 0.,0.
    for X,y in loader:
        # Getting initial X, y, sigma, mu tensors
        X,y = X.to(device), y.to(device) # X is shape (100,1,28,28)- batch of images
        sigma = model_sigma(X)  # (100, 784) tensor- X is (100,1,28,28)
        sigma_diag = np.zeros((len(sigma),len(sigma[0]),len(sigma[0])))
        sigma_diag[np.arange(len(sigma))[:, None], np.arange(len(sigma[0])), np.arange(len(sigma[0]))] = sigma.detach().cpu().numpy()

        mu = model_mu(X)        # (100, 784) tensor
        mu = mu.detach().cpu().numpy()

        # Using sigma and mu tensors to create random noise with those values
        rng = np.random.default_rng()
        n_samples = 50 # Number of each image to create random noise for. Make this an input argument?
        epsilon = np.ndarray((len(X),n_samples,28,28))
        for n in range(len(X)):
            # Creating random noise with custom mean vector and covariance matrix
            temp = rng.multivariate_normal(mu[n],sigma_diag[n],size=(n_samples))   # mu must be 1D, temp is (n_samples, 784) tensor
            temp = np.reshape(temp,(n_samples,28,28))
            epsilon[n] = temp # Epsilon ends up being (len(X), n_samples, 28, 28)

        # Getting the scores of the images with random noise added to images
        epsilon_torch = torch.from_numpy(epsilon).float().to(device)
        X = X.expand(-1, n_samples, -1, -1) # n_samples of each image (second dimension), (len(X), n_samples, 28, 28) shape
        scores = torch.zeros((len(X),n_samples,10)).to(device) # shape is (images in batch, n_samples, number of classes)
        for n in range(len(X)):
            scores[n] = pretrained(X[n]+epsilon_torch[n])
        
        # Getting probabilities of each class, and top 2 likely images based on smoothing
        probs = torch.softmax(scores, dim=2)    # Softmax each set of scores
        avg_probs = probs.mean(dim=1)
        best_scores = torch.topk(avg_probs, 2)
        
        # Computing certified radii for each image
        radii = torch.zeros((len(X)))
        for n in range(len(avg_probs)):
            radii[n] = (best_scores.values[n][0].item() - best_scores.values[n][1].item()) / 2  # Still need to incorporate Lipschitz constant
        
        # Computing ACR
        acr = sum(radii)/len(radii)
       
        # Computing losses and optimizing
        spectral_sigma = []
        spectral_mu = []
        lam = 0.3   # lambda
        n_layers = 4    # number of layers in neural net. Maybe make this an input argument?
        L_const = 1.5**(1/n_layers) # Modified Lipschitz constant. Maybe make the 1.5 stand-in an input argument?

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # New lines- reweighting layers based on spectral weight calculation
            for name, param in model_sigma.named_parameters():    # mu and sigma models
                name=name.lstrip("1234567890.")
                if name == "weight":
                    # Weight layers only. L_const is currently taken to be Lipschitz constant ^ (1/n_layers)
                    spectral_sigma.append(torch.linalg.matrix_norm(param.data,ord=2).item())
                    param.data =  L_const*(param.data/spectral_sigma[-1])

            for name, param in model_mu.named_parameters():    # mu and sigma models
                name=name.lstrip("1234567890.")
                if name == "weight":
                    # Weight layers only. L_const is currently taken to be Lipschitz constant ^ (1/n_layers)
                    spectral_mu.append(torch.linalg.matrix_norm(param.data,ord=2).item())
                    param.data =  L_const*(param.data/spectral_mu[-1])

        
        total_err += (yp.max(dim=1)[1] != y).sum().item()   # Modify since we no longer have yp in the current version.
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
        epoch_params(model_dnn_2, model_sigma, model_mu, train_loader, training_epsilon, alpha, num_iter)
    torch.save(model_dnn_2.state_dict(), "model_IDRS.pt")