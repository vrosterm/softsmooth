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
 

# Load existing model- edit this to change which model is loaded and its structure
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
model_dnn_2 = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

model_dnn_2.load_state_dict(torch.load("models/model_dnn_2.pt"))

def waterfall_limited(model,sigma=[0.25,0.5,0.75,1],n_test_images=500):
    '''Function to create a waterfall plot based on a limited/partial set of the
    overall test data.
    model = the nn to make the waterfall plots for. This is smoothed within the
            function.
    sigma = list of sigma values to create the waterfall plot for
    n_test_images = number of test images per class'''
    labels=[[0 for n in range(n_test_images*10)]for m in range(len(sigma))]
    radii=[[0 for n in range(n_test_images*10)] for m in range(len(sigma))]
    for i in range(len(sigma)):
        classes = [0 for n in range(10)]
        test_idx = 0
        n_total = 0
        while n_total < n_test_images*10:
            x, y = mnist_test[test_idx]
            x = x.unsqueeze(0).to(device) 
            y = torch.tensor([y]).to(device)
            test_idx+=1
            if classes[y.item()-1]<500:
                classes[y.item()-1]+=1
                labels[i][n_total],radii[i][n_total]=smooth(x,model,sigma[i])
                if labels[i][n_total] != y:
                    radii[i][n_total]=0
                n_total+=1
    radius_domain = linspace(0,2,1000)
    wf_radii = [[0 for n in range(len(radius_domain))] for m in range(len(sigma))]
    for i in range(len(sigma)):
        for j in range(len(radius_domain)): # for every radius in the domain
            for k in range(len(radii[i])):     # for every radius computed
                if radii[i][k]>= radius_domain[j]:  # check if computed radius is greater than current radius
                    wf_radii[i][j] += 1/len(radii[i]) # adds a proportional cumulative data point to the y axis data
    # Plotting data
    fig = pyplot.figure()
    for i in range(len(sigma)):
        pyplot.plot(radius_domain,wf_radii[i],label = f"sigma = {sigma[i]}")
    pyplot.xlabel("radius")
    pyplot.ylabel("certified accuracy")
    pyplot.xlim(0,2)
    pyplot.ylim(0,1)
    pyplot.title(f"partial data set, n_test_images = {n_test_images}, sigma = {sigma}")
    pyplot.legend()
    pyplot.show()

#calling the plot/function. default settings takes 28.6s to run on Faith's laptop.
#waterfall_limited(model_dnn_2)


def waterfall_full(model,sigma=[0.25,0.5,0.75,1]):
    '''Function to create a waterfall plot for the full mnist test set
    model = the nn to make the waterfall plots for. This is smoothed within the
            function.
    sigma = list of sigma values to create the waterfall plot for'''

    labels=[[0 for n in range(len(mnist_test))]for m in range(len(sigma))]
    radii= [[0 for n in range(len(mnist_test))] for m in range(len(sigma))]
    for i in range(len(sigma)):
        for j in range(len(mnist_test)):
            x, y = mnist_test[j]
            x = x.unsqueeze(0).to(device) 
            y = torch.tensor([y]).to(device)
            labels[i][j],radii[i][j]=smooth(x,model,sigma[i])
            if labels[i][j] != y:
                radii[i][j]=0

    radius_domain = linspace(0,2,1000)
    wf_radii = [[0 for n in range(len(radius_domain))] for m in range(len(sigma))]
    for i in range(len(sigma)):
        for j in range(len(radius_domain)): # for every radius in the domain
            for k in range(len(radii[i])):     # for every radius computed
                if radii[i][k]>= radius_domain[j]:  # check if computed radius is greater than current radius
                    wf_radii[i][j] += 1/len(radii[i]) # adds a proportional cumulative data point to the y axis data

    # Plotting data
    fig = pyplot.figure()
    for i in range(len(sigma)):
        pyplot.plot(radius_domain,wf_radii[i],label = f"sigma = {sigma[i]}")
    pyplot.xlabel("radius")
    pyplot.ylabel("certified accuracy")
    pyplot.xlim(0,2)
    pyplot.ylim(0,1)
    pyplot.title(f"full data set, sigma = {sigma}")
    pyplot.legend()
    pyplot.show()

#testing the full waterfall plot- default settings takes 56.2s to run on Faith's laptop.
#waterfall_full(model_dnn_2)