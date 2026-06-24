import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Output-averaging reference copy of the original LWRS model.
class LWRS(nn.Module):
    def __init__(self, sigma, n_samples):
        super(LWRS, self).__init__()
        self.sigma = sigma
        self.n_samples = n_samples
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 784)
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 10)
        self.z = [0, 0, 0]

    def set_noise_layers(self, z_list):
        """Set which layers should have noise applied."""
        self.z = z_list

    def add_noise(self, x, layer_idx):
        """Add noise only if the corresponding z flag is set."""
        if self.sigma[layer_idx] > 0 and self.z[layer_idx] == 1:
            epsilon = self.sigma[layer_idx] * torch.randn_like(x)
            return x + epsilon
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.flatten(x)

        x = self.add_noise(x, 0)
        x = self.fc0(x)
        x = F.relu(x)

        x = self.add_noise(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.add_noise(x, 2)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.view(batch_size, self.n_samples, -1)
        x = torch.softmax(x, dim=1)
        x = x.mean(dim=1)
        return x
