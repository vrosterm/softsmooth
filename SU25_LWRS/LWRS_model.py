import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Model h(x) with layer-specific noise control
class LWRS(nn.Module):
    def __init__(self, sigma, n_samples):
        super(LWRS, self).__init__()
        self.sigma = sigma
        self.n_samples = n_samples
        # Layerwise structure with matching dimensions: 784 -> 784 -> 784 -> 10
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 784)  # Input layer: 784 -> 784
        self.fc1 = nn.Linear(784, 784)  # Hidden layer 1: 784 -> 784
        self.fc2 = nn.Linear(784, 784)  # Hidden layer 2: 784 -> 784
        self.fc3 = nn.Linear(784, 10)   # Output layer: 784 -> 10
        
        # Noise control variables - which layers to apply noise to
        self.z = [0, 0, 0]  # [z^(0), z^(1), z^(2)] - binary flags for noise at each layer

    def set_noise_layers(self, z_list):
        """Set which layers should have noise applied.
        z_list: [z0, z1, z2] where 1 = apply noise, 0 = no noise
        """
        self.z = z_list

    def add_noise(self, x, layer_idx):
        """Add noise only if the corresponding z flag is set"""
        if self.sigma[layer_idx] > 0 and self.z[layer_idx] == 1:
            epsilon = self.sigma[layer_idx] * torch.randn_like(x)
            return x + epsilon
        return x

    def forward(self, x):
        batch_size = x.size(0)
        # Repeat each input n_samples times
        x = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Flatten input
        x = self.flatten(x)

        # Layer 0: 784 -> 784
        x = self.add_noise(x, 0)  # Apply noise if z^(0) = 1
        x = self.fc0(x)
        x = F.relu(x)
        
        # Layer 1: 784 -> 784
        x = self.add_noise(x, 1)  # Apply noise if z^(1) = 1
        x = self.fc1(x)
        x = F.relu(x)
        
        # Layer 2: 784 -> 784
        x = self.add_noise(x, 2)  # Apply noise if z^(2) = 1
        x = self.fc2(x)
        x = F.relu(x)
        
        # Output layer: 784 -> 10 (no activation, no noise)
        x = self.fc3(x)

        # Reshape to [batch_size, n_samples, num_classes]
        x = x.view(batch_size, self.n_samples, -1)

        # Apply Softmax to the samples 
        x = torch.softmax(x,dim=1)

        # Average logits
        x = x.mean(dim=1)        # Then aggregate

        return x


'''
model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)
'''