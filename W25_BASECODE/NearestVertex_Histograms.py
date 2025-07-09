import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from tqdm import tqdm

# Load CIFAR-10 and MNIST datasets
def load_datasets():
    transform = transforms.Compose([transforms.ToTensor()])
    
    # CIFAR-10 dataset
    CIFAR10_data = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_data, batch_size=len(CIFAR10_data), shuffle=False)
    CIFAR10_images, _ = next(iter(CIFAR10_loader))  # Get all images
    CIFAR10_images = CIFAR10_images.view(CIFAR10_images.size(0), -1)  # Flatten images
    
    # MNIST dataset
    MNIST_data = datasets.MNIST("data", train=False, download=True, transform=transform)
    MNIST_loader = torch.utils.data.DataLoader(MNIST_data, batch_size=len(MNIST_data), shuffle=False)
    MNIST_images, _ = next(iter(MNIST_loader))  # Get all images
    MNIST_images = MNIST_images.view(MNIST_images.size(0), -1)  # Flatten images
    
    return CIFAR10_images, MNIST_images

# Generate random uniform distributions
def random_distribution(CIFAR10_images, MNIST_images):
    CIFAR10_random = torch.rand_like(CIFAR10_images)  # Random uniform distribution for CIFAR-10
    MNIST_random = torch.rand_like(MNIST_images)  # Random uniform distribution for MNIST
    return CIFAR10_random, MNIST_random

# Calculate L2 norm distance to the nearest vertex
def vertex_L2(data):
    nearest_vertex = torch.round(data)  # Scale and round to nearest vertex
    distances = torch.norm(data - nearest_vertex, dim=1)  # L2 norm distance
    return distances

# Plot histograms
def plot_histograms(CIFAR10_distances, CIFAR10_random_distances, MNIST_distances, MNIST_random_distances):
    plt.figure(figsize=(12, 8))
    
    # Determine the domain for CIFAR-10 histograms
    cifar10_min = min(CIFAR10_distances.min(), CIFAR10_random_distances.min())
    cifar10_max = max(CIFAR10_distances.max(), CIFAR10_random_distances.max())
    cifar10_bins = torch.linspace(cifar10_min, cifar10_max, steps=50).numpy()
    
    # Determine the domain for MNIST histograms
    mnist_min = min(MNIST_distances.min(), MNIST_random_distances.min())
    mnist_max = max(MNIST_distances.max(), MNIST_random_distances.max())
    mnist_bins = torch.linspace(mnist_min, mnist_max, steps=50).numpy()
    
    # CIFAR-10 histogram
    plt.subplot(2, 2, 1)
    plt.hist(CIFAR10_distances.numpy(), bins=cifar10_bins, color='blue', alpha=0.7)
    plt.title("CIFAR-10 L2 Vertex Distances")
    plt.xlabel("L2 Distance")
    plt.ylabel("Frequency")
    
    # CIFAR-10 random histogram
    plt.subplot(2, 2, 3)
    plt.hist(CIFAR10_random_distances.numpy(), bins=cifar10_bins, color='green', alpha=0.7)
    plt.title("CIFAR-10 Random L2 Vertex Distances")
    plt.xlabel("L2 Distance")
    plt.ylabel("Frequency")
    
    # MNIST histogram
    plt.subplot(2, 2, 2)
    plt.hist(MNIST_distances.numpy(), bins=mnist_bins, color='red', alpha=0.7)
    plt.title("MNIST L2 Vertex Distances")
    plt.xlabel("L2 Distance")
    plt.ylabel("Frequency")
    
    # MNIST random histogram
    plt.subplot(2, 2, 4)
    plt.hist(MNIST_random_distances.numpy(), bins=mnist_bins, color='purple', alpha=0.7)
    plt.title("MNIST Random L2 Vertex Distances")
    plt.xlabel("L2 Distance")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load datasets
    CIFAR10_images, MNIST_images = load_datasets()
    
    # Output dimensions
    print(f"CIFAR-10 Dimensions: {CIFAR10_images.shape}")  # Output CIFAR-10 dimensions
    print(f"MNIST Dimensions: {MNIST_images.shape}")  # Output MNIST dimensions
    
    # Generate random uniform distributions
    CIFAR10_random, MNIST_random = random_distribution(CIFAR10_images, MNIST_images)
    
    # Calculate distances for CIFAR-10
    print("Calculating distances for CIFAR-10...")
    CIFAR10_distances = vertex_L2(CIFAR10_images)
    CIFAR10_random_distances = vertex_L2(CIFAR10_random)
    
    # Calculate distances for MNIST
    print("Calculating distances for MNIST...")
    MNIST_distances = vertex_L2(MNIST_images)
    MNIST_random_distances = vertex_L2(MNIST_random)
    
    # Plot histograms
    plot_histograms(CIFAR10_distances, CIFAR10_random_distances, MNIST_distances, MNIST_random_distances)

# Run the main function
if __name__ == "__main__":
    main()