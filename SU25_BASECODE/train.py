import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The following is all code from the adversarial training tutorial. https://adversarial-ml-tutorial.org/adversarial_examples/

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

"""So far the only way to make models is to manually copy/paste like below. 
Maybe in the future we can add something that enables us to make custom models. 
Better yet, load in models that we choose from outside."""
model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(), 
                            nn.Linear(200,10)).to(device)

model_dnn_4 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(), 
                            nn.Linear(200,100), nn.ReLU(),
                            nn.Linear(100,100), nn.ReLU(),
                            nn.Linear(100,10)).to(device)

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
    # The following only uses model_dnn_2. At later points we can add an arg that loads a model for us.
    batch_size = 100
    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
    opt = optim.SGD(model_dnn_2.parameters(), lr=1e-1)
    print('train_err\ttrain_loss\ttest_err\ttest_loss')
    for _ in range(10):
        train_err, train_loss = epoch(train_loader, model_dnn_2, opt)
        test_err, test_loss = epoch(test_loader, model_dnn_2)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")

