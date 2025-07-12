import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class RSLW(nn.Module):
    def __init__(self, noise_std=0.1):
        super(RSLW, self).__init__()
        self.noise_std = noise_std
        self.flatten = Flatten()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def add_noise(self, x):
        if self.noise_std > 0:
            noise = self.noise_std * torch.randn_like(x)
            return x + noise
        return x

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