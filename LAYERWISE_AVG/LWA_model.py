import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_sigma_list(sigma, n_layers=3):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().tolist()
    if isinstance(sigma, (int, float)):
        return [float(sigma)] * n_layers
    sigma = [float(s) for s in sigma]
    if len(sigma) == 1:
        return sigma * n_layers
    if len(sigma) != n_layers:
        raise ValueError(f"Expected {n_layers} sigma values, got {len(sigma)}")
    return sigma


class LWRS(nn.Module):
    """LWA model using analytic ReGU hidden activations."""

    def __init__(self, sigma=(0.5, 0.1, 0.1), n_samples=1000):
        super().__init__()
        self.n_samples = n_samples
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, 784)
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 10)
        self.set_sigma(sigma)

    def set_sigma(self, sigma):
        """Set analytic ReGU sigmas. A zero sigma gives plain ReLU."""
        self.base_sigma = _as_sigma_list(sigma)
        self.sigma = list(self.base_sigma)
        self.z = [1 if sigma_l > 0 else 0 for sigma_l in self.sigma]

    def set_noise_layers(self, z_list):
        """
        Compatibility shim for older code.

        A flag of 1 keeps that layer's configured sigma; 0 sets that layer
        to sigma=0, which is the ReLU limit of ReGU.
        """
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

    @staticmethod
    def _normal_cdf(x):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _normal_pdf(x):
        return torch.exp(-0.5 * x.pow(2)) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def row_noise_std(layer, sigma_l):
        """sqrt(diag(sigma_l^2 W W^T)) = sigma_l * rowwise ||W||_2."""
        return float(sigma_l) * torch.linalg.vector_norm(layer.weight, ord=2, dim=1)

    def _regu_activation(self, x, layer, layer_idx, eps=1e-12):
        z = layer(x)
        sigma_l = self.sigma[layer_idx]
        if sigma_l <= 0:
            return F.relu(z)

        std = self.row_noise_std(layer, sigma_l).to(device=z.device, dtype=z.dtype)
        std = std.unsqueeze(0)
        safe_std = std.clamp_min(eps)
        u = -z / safe_std

        regu = z * (1.0 - self._normal_cdf(u)) + safe_std * self._normal_pdf(u)
        return torch.where(std > eps, regu, F.relu(z))

    def features_to_last_regu_input(self, x):
        x = self.flatten(x)
        x = self._regu_activation(x, self.fc0, 0)
        x = self._regu_activation(x, self.fc1, 1)
        return x

    def forward(self, x):
        x = self.features_to_last_regu_input(x)
        x = self._regu_activation(x, self.fc2, 2)
        return self.fc3(x)
