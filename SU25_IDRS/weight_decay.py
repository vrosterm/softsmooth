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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model definitions
model_dnn_2 = nn.Sequential(
    nn.Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

model_dnn_4 = nn.Sequential(
    nn.Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,10)
).to(device)

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# PGD attack parameters
training_epsilon = 0.05  # Maximum perturbation
epsilon = 0.1  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

# Define separate optimizers with 
opt_dnn2 = optim.SGD(model_dnn_2.parameters(), lr=0.1)
opt_dnn4 = optim.SGD(model_dnn_4.parameters(), lr=0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Training functions
def epoch_adversarial(model, loader, attack, opt, *args):
    total_loss, total_err = 0., 0.
    L = 25.0  # Lipschitz constant for the model
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X + delta)

        loss = nn.CrossEntropyLoss()(yp, y)

        num_linear_layers = sum(1 for layer in model if isinstance(layer, nn.Linear))
        L_const = L ** (1 / num_linear_layers)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Weight normalization
            for layer in model:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    spec_norm = torch.linalg.matrix_norm(weight)
                    norm_weight = L_const * weight / spec_norm
                    layer.weight.data.copy_(norm_weight)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_params(pretrained, model_params, loader, lam=0.01, L=0.1, opt=None):
    total_loss, total_err = 0., 0.
    acr = []

    # Computing the Lipschitz constant for the pretrained model
    lip_g = 1.0
    for layer in pretrained:
        if isinstance(layer, nn.Linear):
            weight_norm = torch.linalg.matrix_norm(layer.weight)
            lip_g *= weight_norm
    L_max = lip_g * (1 + ((L ** 2 + L ** 2)**(1/2)))  # Lipschitz const for mu/sigma net
    print(f"L_max (denominator for radii): {L_max:.4f}")
    print(f"lip_g: {lip_g:.4f}")

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        outputs = model_params(X)  # (batch_size, 1568)

        # Separate mu and sigma from outputs
        half_len = outputs.shape[1] // 2
        mu = outputs[:, :half_len].detach().cpu().numpy()
        sigma = outputs[:, half_len:].detach().cpu().numpy()

        # Construct diagonal covariance matrices and square them
        sigma_diag = np.zeros((len(sigma), half_len, half_len))
        sigma_diag[np.arange(len(sigma))[:, None], np.arange(half_len), np.arange(half_len)] = sigma
        sigma_diag = np.matmul(sigma_diag, sigma_diag)

        # Call your randomized smoothing function
        g, yp = IDRS_matrices(pretrained, mu, sigma_diag, X, n_samples=50)
        yp_tensor = torch.tensor(yp, device=y.device)

        numerator = (g.values[:, 0] - g.values[:, 1])
        avg_numerator = numerator.mean().item()
        print(f"Avg numerator (margin) for batch: {avg_numerator:.6f}")
        

        radii = numerator / (2 * L_max)
        radii = radii * (yp_tensor == y)

        spec_reg = 0.0
        for layer in model_params:
            if isinstance(layer, nn.Linear):
                spec_norm = torch.linalg.matrix_norm(layer.weight)
                spec_reg += spec_norm

        acr.append(radii.mean().detach().cpu().item())

        loss = -acr[-1] + lam * spec_reg

        num_linear_layers = sum(1 for layer in model_params if isinstance(layer, nn.Linear))
        L_const = L ** (1 / num_linear_layers)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Weight normalization
            for layer in model_params:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    spec_norm = torch.linalg.matrix_norm(weight)
                    norm_weight = L_const * weight / spec_norm
                    layer.weight.data.copy_(norm_weight)

        total_err += (yp_tensor != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    cert_rad = sum(acr) / len(acr)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), cert_rad

# PGF L_2 Norm Attack
def pgd_l2(model, X, y, epsilon=2.4, alpha=0.01, num_iter=20):
    delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        
        # Update delta
        delta.data = delta.data + alpha * delta.grad.detach()
        
        # Project onto L2 ball with radius epsilon
        delta_norms = torch.norm(delta.data.view(delta.shape[0], -1), dim=1, keepdim=True)
        delta.data = delta.data / delta_norms.view(-1, 1, 1, 1) * torch.min(delta_norms, torch.tensor(epsilon).to(delta.device)).view(-1, 1, 1, 1)
        
        delta.grad.zero_()
    return delta.detach()

# Model evaluation on clean data
def evaluate_clean(model, loader):
    model.eval()
    total_err = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

# Model evaluation under PGD attack
def evaluate_under_attack(model, loader, epsilon, alpha, num_iter):
    model.eval()
    total_err = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = pgd_l2(model, X, y, epsilon, alpha, num_iter)
        yp = model(X + delta)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

if __name__ == '__main__':
    # Train and save DNN2 if not already saved
    if not os.path.exists("model_dnn_2.pt"):
        for epoch in range(10):
            train_err, train_loss = epoch_adversarial(model_dnn_2, train_loader, pgd_l2, opt_dnn2, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[DNN_2] Epoch {epoch+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")

    # Train and save DNN4 if not already saved
    if not os.path.exists("model_dnn_4.pt"):
        for epoch in range(10):
            train_err, train_loss = epoch_adversarial(model_dnn_4, train_loader, pgd_l2, opt_dnn4, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[DNN_4] Epoch {epoch+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_dnn_4.state_dict(), "model_dnn_4.pt")

    # Loading save states
    model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt", map_location=device, weights_only=True))
    model_dnn_4.load_state_dict(torch.load("model_dnn_4.pt", map_location=device, weights_only=True))

    # Evaluating and printing results
    for model, name in [
        (model_dnn_2, "DNN_2"),
        (model_dnn_4, "DNN_4")
    ]:
        clean_acc = evaluate_clean(model, test_loader)
        adv_acc = evaluate_under_attack(model, test_loader, epsilon, alpha, num_iter)
        print(f"Accuracy of {name} on clean data: {clean_acc:.4f}")
        print(f"Accuracy of {name} under PGD attack: {adv_acc:.4f}")
