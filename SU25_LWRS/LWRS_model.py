import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class LWRS(nn.Module):
    def __init__(self, noise_std=0.03, n_samples=2):
        super(LWRS, self).__init__()
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.conv_stack = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*64, 100)
        self.fc2 = nn.Linear(100, 10)


    def add_noise(self, x):
        if self.noise_std > 0:
            epsilon = self.noise_std * torch.randn_like(x)
            return x + epsilon

    def forward(self, x):
        batch_size = x.size(0)
        # Repeat each input n_samples times
        x = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)  # [B, n_samples, C, H, W]
        x = x.view(-1, x.size(2), x.size(3), x.size(4))        # [B*n_samples, C, H, W]
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.add_noise(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.add_noise(x)
        x = self.fc2(x)
        # Reshape back to [batch_size, n_samples, num_classes]
        x = x.view(batch_size, self.n_samples, -1)
        x = x.mean(dim=1)  # Average logits, not softmax
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