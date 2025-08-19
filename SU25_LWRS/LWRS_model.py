# We need to train h(x) ALONE, and then apply its weights and biases to h~(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define ReGU activation function
def regu(x, sigma=0.1):
    
    # Standard normal PDF: φ(z) = exp(-z²/2) / sqrt(2π)
    phi = lambda z: torch.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
    
    # Standard normal CDF: Φ(z) = (1 + erf(z/sqrt(2))) / 2
    Phi = lambda z: (1 + torch.erf(z / math.sqrt(2))) / 2
    
    regu_output = (x * Phi(x / sigma) + 
                   sigma * phi(x / sigma))
    
    return regu_output

# Model h(x)
class LWRS(nn.Module):
    def __init__(self, sigma=0.1, n_samples=20):
        super(LWRS, self).__init__()
        self.sigma = sigma
        self.n_samples = n_samples
        
        # Layerwise structure
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 200)  # First layer
        self.fc1 = nn.Linear(200, 100)  # Second layer
        self.fc2 = nn.Linear(100, 50)   # Hidden2 → Hidden3 (new intermediate layer)
        self.fc3 = nn.Linear(50, 10)    # Hidden3 → Output (10 classes)

    def add_noise(self, x):
        if self.sigma > 0:
            epsilon = self.sigma * torch.randn_like(x)
            return x + epsilon
        return x

    def forward(self, x):
        batch_size = x.size(0)
        
        # Repeat each input n_samples times
        x = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Flatten input
        x = self.flatten(x)

        # Layer 1 (no noise)
        x = self.fc0(x)
        x = F.relu(x)

        # Layer 2
        x = self.add_noise(x)
        x = self.fc1(x)
        x = F.relu(x)

        # Layer 3
        x = self.add_noise(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Apply linear transformation ALONE
        x = self.fc3(x)

        # Reshape to [batch_size, n_samples, num_classes]
        x = x.view(batch_size, self.n_samples, -1)
        
        # Average logits
        x = x.mean(dim=1) ## Only aggregate AFTER SOFTMAX!! Push changes to github to document progress. --------
                        ## use "feedthrough" variables to apply the noise at specific layers ON THEIR OWN.
        return x

# Model h~(x)
class LWRS_tilde(nn.Module):
    def __init__(self, sigma=0.1, n_samples=20):
        super(LWRS_tilde, self).__init__()
        self.sigma = sigma
        self.n_samples = n_samples
        
        # Layerwise structure
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 200)  # First layer (same as LWRS)
        self.fc1 = nn.Linear(200, 100)  # Second layer (same as LWRS)
        self.fc2 = nn.Linear(100, 50)   # Hidden2 → Hidden3 (same as LWRS)
        self.fc3 = nn.Linear(50, 10)    # Hidden3 → Output (same as LWRS)

    def add_noise(self, x):
        if self.sigma > 0:
            epsilon = self.sigma * torch.randn_like(x)
            return x + epsilon
        return x

    def forward(self, x):
        batch_size = x.size(0)
        
        # Repeat each input n_samples times
        x = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Flatten input
        x = self.flatten(x)

        # Layer 1 (no noise) - ReLU like LWRS
        x = self.fc0(x)
        x = F.relu(x)

        # Layer 2 (no noise) - ReGU instead of ReLU
        x = self.fc1(x)
        x = regu(x, self.sigma)

        # Layer 3 (no noise) - ReGU instead of ReLU
        x = self.fc2(x)
        x = regu(x, self.sigma)

        # Apply linear transformation ALONE (no activation like LWRS)
        x = self.fc3(x)

        # Reshape to [batch_size, n_samples, num_classes]
        x = x.view(batch_size, self.n_samples, -1)
        
        # Average logits (NO softmax like LWRS)
        x = x.mean(dim=1)
        
        return x