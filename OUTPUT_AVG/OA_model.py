import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_sigma(sigma):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().flatten().tolist()
    if isinstance(sigma, (list, tuple)):
        if len(sigma) == 0:
            return 0.0
        return float(sigma[0])
    return float(sigma)


class OutputAveragingRS(nn.Module):
    """Standard input randomized smoothing with one ReLU hidden layer."""

    def __init__(self, sigma=2.0, n_samples=1000, average="logits", hidden_dim=784):
        super().__init__()
        self.n_samples = int(n_samples)
        self.average = average
        self.hidden_dim = int(hidden_dim)
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(784, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, 10)
        self.input_noise = True
        self.set_sigma(sigma)

    def set_sigma(self, sigma):
        self.sigma = _input_sigma(sigma)

    def set_noise_layers(self, z_list):
        if len(z_list) == 0:
            raise ValueError("Expected at least one input-noise flag")
        self.input_noise = bool(int(z_list[0]))

    def linear_layers(self):
        return [self.fc0, self.fc1]

    def regu_layers(self):
        return [self.fc0]

    def output_layer(self):
        return self.fc1

    def _relu_logits(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc0(x))
        return self.fc1(x)

    def sample_logits(self, x, n_samples=None):
        batch_size = x.size(0)
        n_samples = max(int(n_samples or self.n_samples), 1)
        x_samples = x.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
        x_samples = x_samples.view(batch_size * n_samples, *x.shape[1:])

        if self.input_noise and self.sigma > 0:
            x_samples = x_samples + self.sigma * torch.randn_like(x_samples)

        logits = self._relu_logits(x_samples)
        return logits.view(batch_size, n_samples, -1)

    def forward(self, x):
        sample_logits = self.sample_logits(x)
        if self.average == "logits":
            return sample_logits.mean(dim=1)
        return torch.softmax(sample_logits, dim=-1).mean(dim=1)

LWRS = OutputAveragingRS
