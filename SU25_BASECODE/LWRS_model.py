import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class LWRS(nn.Module):
    def __init__(self, noise_std=0.1, n_samples=2):
        super(LWRS, self).__init__()
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.flatten = Flatten()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def add_noise(self, x):
        """
        if self.noise_std > 0:
            noise = self.noise_std * torch.randn_like(x)
            return x + noise
        """
        x = x.expand(self.n_samples, -1, -1, -1)
        epsilon = self.noise_std * torch.randn_like(x)
        return x + epsilon

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.add_noise(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.add_noise(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.add_noise(x)
        return x