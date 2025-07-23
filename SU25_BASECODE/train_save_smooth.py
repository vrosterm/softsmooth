import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as pyplot
from numpy import linspace
from tqdm import tqdm

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

def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for X, y in tqdm(loader, desc="Epoch Progress"):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Training functions
def epoch_adversarial(model, loader, attack, opt, *args):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# PGD L_inf Attack
def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

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
        delta = pgd_linf(model, X, y, epsilon, alpha, num_iter)
        yp = model(X + delta)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

# Inverse of the Gaussian CDF
def phi_inverse(x, mu):
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(2 * x - 1)

# Smooth function for certified radius. Uses softmax to obtain vectors with entries in [0,1] that sum to 1 so they can be inputted into erfinv.
def smooth(X, model, sigma, n_samples=1000):
    X = X.expand(n_samples, -1, -1, -1)
    epsilon = sigma * torch.randn_like(X)
    scores = model(X + epsilon) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    best_scores = torch.topk(avg_probs, 2)          
    radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item(), radius.item()

if __name__ == '__main__':
    # Train and save DNN2 if not already saved
    if not os.path.exists("model_dnn_2.pt"):
        for epoch in range(10):
            train_err, train_loss = epoch_adversarial(model_dnn_2, train_loader, pgd_linf, opt_dnn2, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[DNN_2] Epoch {epoch+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")

    # Train and save DNN4 if not already saved
    if not os.path.exists("model_dnn_4.pt"):
        for epoch in range(10):
            train_err, train_loss = epoch_adversarial(model_dnn_4, train_loader, pgd_linf, opt_dnn4, training_epsilon, alpha, num_iter)
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
