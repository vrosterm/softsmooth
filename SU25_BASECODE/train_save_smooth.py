import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Model definitions
model_dnn_2 = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

model_dnn_4 = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,10)
).to(device)

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# Training function
def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        
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

# PGD attack parameters
epsilon = 0.082  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

# Train and save models if not already saved
if not os.path.exists("model_dnn_2.pt"):
    opt = optim.SGD(model_dnn_2.parameters(), lr=0.1)
    for _ in range(10):
        epoch_adversarial(model_dnn_2, train_loader, pgd_linf, opt, epsilon, alpha, num_iter)
    torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")

if not os.path.exists("model_dnn_4.pt"):
    opt = optim.SGD(model_dnn_4.parameters(), lr=0.1)
    for _ in range(10):
        epoch_adversarial(model_dnn_4, train_loader, pgd_linf, opt, epsilon, alpha, num_iter)
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

# Inverse of the Gaussian CDF
def phi_inverse(x, mu):
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(2 * x - 1)

# Smooth function for certified radius. Uses softmax to obtain vectors with entries in [0,1] that sum to 1 so they can be inputted into erfinv.
def smooth(x, model, sigma, n_samples=1000):
    x = x.expand(n_samples, -1, -1, -1)
    epsilon = sigma * torch.randn_like(x)
    scores = model(x + epsilon) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    best_scores = torch.topk(avg_probs, 2)          
    radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item(), radius.item()

# Example of the smooth function on a random test image
'''
idx = random.randint(0, len(mnist_test) - 1)
x, y = mnist_test[idx]
x = x.unsqueeze(0).to(device) 
y = torch.tensor([y]).to(device)

# Compute the L2 norm of the image vector
x_norm = torch.norm(x.view(-1)).item()

# Get smoothed classifier prediction and radius
label, radius = smooth(x, model_dnn_2, sigma=0.2, n_samples=1000)

print(f"True label: {y.item()}, Predicted: {label}, Certified radius: {radius:.4f}, Image L2 norm: {x_norm:.4f}")
'''
