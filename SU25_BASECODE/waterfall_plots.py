import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot
from numpy import linspace
from train_save_smooth import smooth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)  

try:
    model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(), 
                                nn.Linear(200,10)).to(device)
    model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))
    print("model loaded!")
except: 
    print("model not found- training model")
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

    opt = optim.SGD(model_dnn_2.parameters(), lr=1e-1)

    print("TrainError\tTrainLoss\tTestError\tTestLoss")
    for _ in range(10):
        train_err, train_loss = epoch(train_loader, model_dnn_2, opt)
        test_err, test_loss = epoch(test_loader, model_dnn_2)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")

    torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")
    print("trained!")

# Function to create a waterfall plot
def waterfall_plot(model,sigma,n_test_images=500):
    labels=[0 for n in range(n_test_images*10)]
    radii=[0 for n in range(n_test_images*10)]
    classes = [0 for n in range(10)]
    n_total = 0
    test_idx = 0
    while n_total < n_test_images*10:
        x, y = mnist_test[test_idx]
        x = x.unsqueeze(0).to(device) 
        y = torch.tensor([y]).to(device)
        test_idx+=1
        if classes[y.item()-1]<500:
            classes[y.item()-1]+=1
            labels[n_total],radii[n_total]=smooth(x,model,sigma)
            if labels[n_total] != y:
                radii[n_total]=0
            n_total+=1

    radius_domain = linspace(0,2,1000)
    wf_radii = [0 for n in range(len(radius_domain))]
    for i in range(len(radius_domain)): # for every radius in the domain
        for j in range(len(radii)):     # for every radius computed
            if radii[j]>= radius_domain[i]:  # check if computed radius is greater than current radius
                wf_radii[i] += 1/len(radii) # adds a proportional cumulative data point to the y axis data

    # Plotting data
    pyplot.figure()
    pyplot.plot(radius_domain,wf_radii)
    pyplot.xlabel("radius")
    pyplot.ylabel("certified accuracy")
    pyplot.xlim(0,2)
    pyplot.ylim(0,1)
    pyplot.title(f"sigma = {sigma}")
    pyplot.show()

#testing waterfall_plot

waterfall_plot(model_dnn_2,0.2)
