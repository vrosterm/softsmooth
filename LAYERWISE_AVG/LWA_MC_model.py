import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .LWA_model import _as_sigma_list
except ImportError:
    from LWA_model import _as_sigma_list


class LWMC(nn.Module):
    """Monte Carlo layerwise averaging reference model for validating analytic LWA."""

    def __init__(self, sigma=(0.5, 0.1, 0.1), n_samples=1000):
        super().__init__()
        self.n_samples = int(n_samples)
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 784)
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 10)
        self.set_sigma(sigma)

    def set_sigma(self, sigma):
        self.base_sigma = _as_sigma_list(sigma)
        self.sigma = list(self.base_sigma)
        self.z = [1 if sigma_l > 0 else 0 for sigma_l in self.sigma]

    def set_noise_layers(self, z_list):
        if len(z_list) != len(self.base_sigma):
            raise ValueError(f"Expected {len(self.base_sigma)} layer flags, got {len(z_list)}")
        self.z = [1 if int(flag) else 0 for flag in z_list]
        self.sigma = [
            sigma_l if enabled else 0.0
            for sigma_l, enabled in zip(self.base_sigma, self.z)
        ]

    def linear_layers(self):
        return [self.fc0, self.fc1, self.fc2, self.fc3]

    def regu_layers(self):
        return [self.fc0, self.fc1, self.fc2]

    def _mc_activation(self, x, layer, layer_idx):
        sigma_l = self.sigma[layer_idx]
        if sigma_l <= 0:
            return F.relu(layer(x))

        batch_size = x.size(0)
        n_samples = max(self.n_samples, 1)
        x_samples = x.unsqueeze(1).repeat(1, n_samples, 1)
        x_samples = x_samples.view(batch_size * n_samples, -1)
        x_samples = x_samples + float(sigma_l) * torch.randn_like(x_samples)

        activations = F.relu(layer(x_samples))
        activations = activations.view(batch_size, n_samples, -1)
        return activations.mean(dim=1)

    def features_to_last_regu_input(self, x):
        x = self.flatten(x)
        x = self._mc_activation(x, self.fc0, 0)
        x = self._mc_activation(x, self.fc1, 1)
        return x

    def forward(self, x):
        x = self.features_to_last_regu_input(x)
        x = self._mc_activation(x, self.fc2, 2)
        return self.fc3(x)
