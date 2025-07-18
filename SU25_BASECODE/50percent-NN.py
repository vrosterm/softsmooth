import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

mnist_train = datasets.MNIST("C:/Users/walke/data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("C:/Users/walke/data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

def pgd_linf(model, X, y, epsilon=0.086, alpha=0.01, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

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
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_adversarial(loader, model, attack_fn):
    total_err = 0
    total_loss = 0
    for X, y in tqdm(loader, desc="Adversarial Epoch Progress"):
        X, y = X.to(device), y.to(device)
        delta = attack_fn(model, X, y)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

opt = optim.SGD(model_cnn.parameters(), lr=1e-1)

# Run a single trial
train_err, train_loss = epoch(train_loader, model_cnn, opt)
test_err, test_loss = epoch(test_loader, model_cnn)
adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn, pgd_linf)
print("Trial 1:")
print("| Train Accuracy:     {:.2f}%".format(100 - train_err * 100))
print("| Test Accuracy:      {:.2f}%".format(100 - test_err * 100))
print("| Adversarial Accuracy:  {:.2f}%".format(100 - adv_err * 100))

torch.save(model_cnn.state_dict(), "model_cnn.pt")