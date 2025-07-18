import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from SU25_BASECODE.LWRS_model import LWRS  # Import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LWRS_model = LWRS().to(device)  # Instantiate model

# The following is all code from the adversarial training tutorial. https://adversarial-ml-tutorial.org/adversarial_examples/

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

# So far the only way to make models is to manually copy/paste like below. 
# Maybe in the future we can add something that enables us to make custom models.
# Better yet, load in models that we choose from outside.


def epoch(loader, model, opt=None):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

### Returns average of all scores for one input. I'm not sure what x should be in this case, either the batch or one of the inputs.
def smooth(x: torch.tensor, model: torch.nn.Module, sigma: float, n_samples: int): 
    batch = x.repeat((batch_size, 1)) #1 channel, since B&W
    epsilon = sigma * torch.randn(batch, n_samples)
    scores = model.forward(batch + epsilon)
    avg_scores = torch.mean(scores, dim = 1)
    return torch.argmax(avg_scores)

if __name__ == "__main__":
    batch_size = 100
    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
    opt = optim.SGD(LWRS_model.parameters(), lr=1e-1)
    print("Model Results for {}:".format(type(LWRS_model).__name__))
    print('\t\ttrain_err\ttrain_loss\ttrain_acc\ttest_err\ttest_loss\ttest_acc') # Including accuracy in the printout
    for _epoch_ in range(30): # Number of epochs, at 30 right now
        train_err, train_loss = epoch(train_loader, LWRS_model, opt)
        test_err, test_loss = epoch(test_loader, LWRS_model)
        train_acc = (1 - train_err) * 100  # Convert to percentage
        test_acc = (1 - test_err) * 100  # Convert to percentage
        print(
            "Epoch {}:    \t{:.6f}\t{:.6f}\t{:.4f} %\t{:.6f}\t{:.6f}\t{:.4f} %".format(
                _epoch_ + 1, train_err, train_loss, train_acc, test_err, test_loss, test_acc))
    torch.save(LWRS_model.state_dict(), "LWRS_model.pt")

