import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot
from numpy import linspace
from SU25_BASECODE.LWRS_model import LWRS  # Import model
from tqdm import tqdm

# Don't need the FutureWarning message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Apply CPU or cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)  

# Load LWRS model from file
LWRS_model = LWRS().to(device)
LWRS_model.load_state_dict(torch.load("LWRS_model.pt"))
print("model loaded!")

# Function to create a waterfall plot
def waterfall_plot(model, sigma=[0.25,0.5,0.75,1], n_test_images=500):
    labels = [[0 for n in range(n_test_images*10)] for m in range(len(sigma))]
    radii = [[0 for n in range(n_test_images*10)] for m in range(len(sigma))]
    for i in range(len(sigma)):
        classes = [0 for n in range(10)]
        test_idx = 0
        n_total = 0
        # Add tqdm progress bar here
        with tqdm(total=n_test_images*10, desc=f"sigma={sigma[i]}") as pbar:
            while n_total < n_test_images*10:
                x, y = mnist_test[test_idx]
                x = x.unsqueeze(0).to(device) 
                y = torch.tensor([y]).to(device)
                test_idx += 1
                if classes[y.item()-1] < 500:
                    classes[y.item()-1] += 1
                    labels[i][n_total], radii[i][n_total] = smooth(x, model, sigma[i])
                    if labels[i][n_total] != y:
                        radii[i][n_total] = 0
                    n_total += 1
                    pbar.update(1)
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
    pyplot.title(f"sigma = {sigma}")
    pyplot.legend()
    pyplot.show()

#testing waterfall_plot

waterfall_plot(LWRS_model)