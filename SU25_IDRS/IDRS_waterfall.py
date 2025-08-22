import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot
import numpy as np
from scipy.stats import chi2
from numpy import linspace
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class ReLUBias(nn.Module):
    def __init__(self, min_sig_value=0.5):
        super().__init__()
        self.bias = min_sig_value
        self.min_sig_value = min_sig_value

    def forward(self, x):
        x = F.relu(x) + self.bias   # Applying ReLU and bias
        return x  #(Batch Size, 1)
    
model_base = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

model_base.load_state_dict(torch.load("softsmooth/SU25_BASECODE/models/dnn_2_l2_pgd_epsilon_1.pt"))

model_smooth = nn.Sequential(
    Flatten(), nn.Linear(784,200), nn.ReLU(),
    nn.Linear(200,1), ReLUBias()
).to(device)

model_smooth.load_state_dict(torch.load("model_IDRS.pt"))

#Chi-square PDF
def chi2_pdf(x, dof):
    return chi2.pdf(x, df=dof)

#Chi-square CDF
def chi2_cdf(x, dof):
    return chi2.cdf(x, df=dof)

#Chi-squre CDF inverse
def chi2_cdf_inv(p, dof):
    return chi2.ppf(p, df=dof)

#Derivative of phi wrt x (constant mu)
def phi_derivative(x, mu):
    temp = torch.zeros(1)
    temp[0] = 2 * np.pi
    return -x * torch.exp(-((x - mu) ** 2) / 2) / torch.sqrt(temp)

# Inverse of phi (constant mu)
def phi_inv(x, mu):
    if type(x) is float:
        temp = torch.zeros(1)
        temp[0] = 2 * x - 1
    elif type(x) is torch.Tensor:
        temp = 2 * x - 1
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(temp)

def smooth(X, y, model, sigma, n_samples=1000):
    X = X.expand(n_samples, -1, -1, -1)
    epsilon = sigma * torch.randn_like(X)
    scores = model(X + epsilon) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    if label != y.item():
        radius = 0.0
        return label.item(), radius
    best_scores = torch.topk(avg_probs, 2)          
    radius = sigma * (phi_inv(torch.tensor(best_scores.values[0].item()), 0) - phi_inv(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item(), radius.item()

def scalar_smoothing(pretrained, sigma, X, n_samples=50, beta=10, p_min=1e-5):
    device = X.device
    num_classes = 10

    scores = torch.zeros(n_samples, num_classes, device=device)
    probs = torch.zeros_like(scores)
    yp = []

    epsilon = torch.randn(n_samples, 784, device=device) 
    epsilon = epsilon * torch.sqrt(sigma)  # Elementwise multipying by square root of diagonal elements
    epsilon = epsilon.view(n_samples, 1, 28, 28)
    current_img = X.expand(n_samples, -1, -1, -1)
    
    scores = torch.softmax(beta * pretrained(current_img + epsilon), dim=1)
    probs = (1 - 10 * p_min) * scores + p_min

    avg_probs = probs.mean(dim=1)

    yp.append(torch.argmax(avg_probs).item())
    g = torch.topk(avg_probs, 2)
    
    return g, yp

def waterfall_sig_list(model,sigma=[0.25,0.5,1]):
    '''Returns data for a waterfall plot using a constant/scalar sigma value'''
    labels=[[0 for n in range(len(mnist_test))]for m in range(len(sigma))]
    radii= [[0 for n in range(len(mnist_test))] for m in range(len(sigma))]
    for i in range(len(sigma)):
        for j in range(len(mnist_test)):
            x, y = mnist_test[j]
            x = x.unsqueeze(0).to(device) 
            y = torch.tensor([y]).to(device)
            labels[i][j],radii[i][j]=smooth(x,y,model,sigma[i])
            if labels[i][j] != y:
                radii[i][j]=0

    radius_domain = linspace(0,3,2000)
    wf_radii = [[0 for n in range(len(radius_domain))] for m in range(len(sigma))]
    for i in range(len(sigma)):
        for j in range(len(radius_domain)): # for every radius in the domain
            for k in range(len(radii[i])):     # for every radius computed
                if radii[i][k]>= radius_domain[j]:  # check if computed radius is greater than current radius
                    wf_radii[i][j] += 1/len(radii[i]) # adds a proportional cumulative data point to the y axis data
    
    return radius_domain,wf_radii

def waterfall_sig_model(model,sigma_model):
    '''Returns data for a waterfall plot using a model for sigma.'''
    labels=[0 for n in range(len(mnist_test))]
    radii= [0 for n in range(len(mnist_test))]
    p_min = 10**(-7)
    beta = 100
    dof = 784
    L = 0.025

    for i in range(len(mnist_test)):
        x, y = mnist_test[i]
        x = x.unsqueeze(0).to(device) 
        y = torch.tensor([y]).to(device)
                
        sigma = sigma_model(x)  # (batch_size, 1)
        sigma_vals = sigma.squeeze(-1) #(batch_size,)

        # Batch of sigmas for each image
        sigma_diag = sigma_vals.squeeze(-1)  # shape: (batch_size,)

        # Pass to randomized smoothing
        g, yp = scalar_smoothing(model, sigma_diag, x, n_samples=50, beta=beta, p_min=p_min)
        yp_tensor = torch.tensor(yp, device=y.device)
        
        #Computing L_final using section 4 math
        sigma_min = sigma_model[-1].min_sig_value
        C = chi2_cdf_inv(1-p_min, dof) * chi2_pdf(chi2_cdf_inv(1-p_min, dof), 1) / phi_derivative(phi_inv(p_min, 0), 0)
        L_final = (1 / sigma_min + 2 * L * C / sigma_min).to(device)
        
        if yp == y:
            radii[i]  = (phi_inv(g.values[0], 0) - phi_inv(g.values[1], 0)) / (2 * L_final)

    radius_domain = linspace(0,3,2000)
    wf_radii = [0 for n in range(len(radius_domain))]
    for i in range(len(radius_domain)):
        for j in range(len(radii)):
            if radii[j] >= radius_domain[i]:
                wf_radii[i] += 1/len(radii)

    return radius_domain, wf_radii


# Making and plotting graphs

sigma = [0.25,0.5,1]

x_const, y_const = waterfall_sig_list(model_base,sigma=sigma)
x_model, y_model = waterfall_sig_model(model_base,model_smooth)

fig = pyplot.figure()
pyplot.plot(x_model,y_model,label = f"Scalar sigma_min = 0.5")
for i in range(len(sigma)):
    pyplot.plot(x_const,y_const[i],label = f"Constant sigma = {sigma[i]}")
pyplot.xlabel("l2 Radius")
pyplot.ylabel("Certified Accuracy")
pyplot.xlim(0,3)
pyplot.ylim(0,1)
pyplot.title(f"Title Here")
pyplot.legend()
pyplot.show()
