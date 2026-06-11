import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

def scalar_smoothing(pretrained, sigma, X, n_samples=1000, beta=10, p_min=10**(-7)):
    device = X.device
    batch_size = len(X)
    num_classes = 10

    scores = torch.zeros(batch_size, n_samples, num_classes, device=device)
    probs = torch.zeros_like(scores)
    yp = []

    for n in range(batch_size): 
        epsilon = torch.randn(n_samples, 784, device=device) 
        epsilon = epsilon * sigma[n] 
        epsilon = epsilon.view(n_samples, 28, 28)

        current_img = X[n].expand(n_samples, -1, -1)
        scores[n] = torch.softmax(beta * pretrained(current_img + epsilon), dim=1)
        probs[n] = (1 - 10 * p_min) * scores[n] + p_min

    avg_probs = probs.mean(dim=1)

    for n in range(batch_size):
        yp.append(torch.argmax(avg_probs[n]).item())

    g = torch.topk(avg_probs, 2)
    return g, yp
