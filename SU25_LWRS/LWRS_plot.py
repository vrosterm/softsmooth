import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot
from numpy import linspace
from LWRS_model import LWRS, LWRS_tilde  # Import both models
from tqdm import tqdm
from SU25_BASECODE.train_save_smooth import phi_inverse, smooth

# Don't need the FutureWarning message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Apply CPU or cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

# Load both models
LWRS_model = LWRS().to(device)
LWRS_tilde_model = LWRS_tilde().to(device)

# Load trained weights for both models
LWRS_model.load_state_dict(torch.load("models/LWRS_model.pt", map_location=device))
print("LWRS model loaded!")

# Transfer same weights to LWRS_tilde 
LWRS_tilde_model.load_state_dict(torch.load("models/LWRS_model.pt", map_location=device))
print("LWRS_tilde model loaded!")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)  

# Function to create a waterfall plot for a single sigma
def waterfall_plot(model, model_name, sigma=1.0, n_test_images=500):
    print(f"Creating waterfall plot for {model_name} with sigma = {sigma}")
    
    labels = [0 for n in range(n_test_images*10)]
    radii = [0 for n in range(n_test_images*10)]
    
    classes = [0 for n in range(10)]
    test_idx = 0
    n_total = 0
    
    # Add tqdm progress bar
    with tqdm(total=n_test_images*10, desc=f"{model_name} sigma={sigma}") as pbar:
        while n_total < n_test_images*10 and test_idx < len(mnist_test):
            x, y = mnist_test[test_idx]
            x = x.unsqueeze(0).to(device) 
            y_tensor = torch.tensor([y]).to(device)
            test_idx += 1
            
            if classes[y] < n_test_images:
                classes[y] += 1
                # Use correct smooth function signature
                labels[n_total], radii[n_total] = smooth(x, y_tensor, model, sigma, n_samples=500)
                if labels[n_total] != y:
                    radii[n_total] = 0
                n_total += 1
                pbar.update(1)
    
    return labels, radii

# Function to compare both models with single sigma
def compare_models(sigma=1.0, n_test_images=500):
    # Get data for both models
    labels_lwrs, radii_lwrs = waterfall_plot(LWRS_model, "LWRS (h(x))", sigma, n_test_images)
    labels_lwrs_tilde, radii_lwrs_tilde = waterfall_plot(LWRS_tilde_model, "LWRS_tilde (h~(x))", sigma, n_test_images)
    
    # Create radius domain
    radius_domain = linspace(0, 2, 1000)
    
    # Calculate waterfall data for both models
    wf_radii_lwrs = [0 for n in range(len(radius_domain))]
    wf_radii_lwrs_tilde = [0 for n in range(len(radius_domain))]
    
    for j in range(len(radius_domain)):
        # LWRS model
        for k in range(len(radii_lwrs)):
            if radii_lwrs[k] >= radius_domain[j]:
                wf_radii_lwrs[j] += 1/len(radii_lwrs)
        
        # LWRS_tilde model
        for k in range(len(radii_lwrs_tilde)):
            if radii_lwrs_tilde[k] >= radius_domain[j]:
                wf_radii_lwrs_tilde[j] += 1/len(radii_lwrs_tilde)
    
    # Create the plot
    fig = pyplot.figure(figsize=(10, 6))
    pyplot.plot(radius_domain, wf_radii_lwrs, label="LWRS (h(x)) - ReLU", linewidth=2)
    pyplot.plot(radius_domain, wf_radii_lwrs_tilde, label="LWRS_tilde (h~(x)) - ReGU", linewidth=2)
    pyplot.xlabel("Certified Radius")
    pyplot.ylabel("Certified Accuracy")
    pyplot.xlim(0, 1.5)
    pyplot.ylim(0, 1)
    pyplot.title(f"Waterfall Plot Comparison (σ = {sigma})")
    pyplot.legend()
    pyplot.grid(True, alpha=0.3)
    
    # Save as PNG file
    pyplot.savefig(f"LWRS_waterfall_comparison_sigma_{sigma}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved as LWRS_waterfall_comparison_sigma_{sigma}.png")
    
    pyplot.show()
    
    # Print summary statistics
    print(f"\nSummary for σ = {sigma}:")
    print(f"LWRS (h(x)) - Average radius: {sum(r for r in radii_lwrs if r > 0)/max(1, sum(1 for r in radii_lwrs if r > 0)):.4f}")
    print(f"LWRS_tilde (h~(x)) - Average radius: {sum(r for r in radii_lwrs_tilde if r > 0)/max(1, sum(1 for r in radii_lwrs_tilde if r > 0)):.4f}")
    print(f"LWRS (h(x)) - Certified accuracy: {sum(1 for r in radii_lwrs if r > 0)/len(radii_lwrs):.4f}")
    print(f"LWRS_tilde (h~(x)) - Certified accuracy: {sum(1 for r in radii_lwrs_tilde if r > 0)/len(radii_lwrs_tilde):.4f}")

if __name__ == "__main__":
    # Create comparison waterfall plot with sigma = 1.0
    print("Creating waterfall plot comparison with sigma = 1.0...")
    compare_models(sigma=1.0, n_test_images=200)  # Reduced for faster execution