import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

def IDRS_softmax(pretrained, mu, sigma, X, n_samples=50, beta=10, p_min=10**(-5)):
    '''Takes mu and sigma matrices, generates random noise, and applies it to 
    the given images to create new predicted outputs.
    
    pretrained- The pretrained neural net to be smoothed
    mu- A 1xd numpy array of means
    sigma- A dxd positive, semidefinite numpy covariance matrix
    X- The batch of images to be smoothed
    y- The correct labels for the images
    n_samples- the number of samples of each image

    Returns:
    g- The top 2 probabilities for images as topk
    yp- the top predicted score'''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Using sigma and mu tensors to create random noise with those values
    rng = np.random.default_rng()
    scores = torch.zeros((len(X),n_samples,10)).to(device) # shape is (images in batch, n_samples, number of classes)
    yp = []
    probs = torch.zeros((len(X),n_samples,10)).to(device)

    #Below loop works image by image
    for n in range(len(X)):
        # Creating random noise with custom mean vector and covariance matrix
        epsilon = rng.standard_normal(size=(n_samples, 784))
        epsilon = mu[n] + epsilon @ np.sqrt(sigma[n])   # current mu and sigma for this image
        epsilon = np.reshape(epsilon,(n_samples,28,28))
        
        # Getting the scores of the images with random noise added to images
        epsilon_torch = torch.from_numpy(epsilon).float().to(device)

        current_img = X[n].expand(n_samples, -1, -1) # n_samples of each image (second dimension), (n_samples, 28, 28) shape

        scores[n] = torch.softmax(beta * pretrained(current_img+epsilon_torch), dim = 1)

        probs[n] = (1 - (10 * p_min)) * (scores[n]) + p_min
        
    # Getting probabilities of each class, top 2 likely classes based on smoothing, and predicted image labels
    avg_probs = probs.mean(dim=1)

    min_prob = avg_probs.min().item() # Checks if minimum probabilty of all 10 classes is less than p_min
    if min_prob < p_min:
        print(f"Error: Minimum probability {min_prob} is less than p_min {p_min}.")
    
    for n in range(len(X)):
        yp.append(np.argmax(avg_probs[n].detach().cpu().numpy()).item())
    g = torch.topk(avg_probs, 2)

    return g, yp


def IDRS_raw(pretrained, mu, sigma, X, n_samples=50):
    '''Takes mu and sigma matrices, generates random noise, and applies it to 
    the given images to create new predicted outputs.
    
    pretrained- The pretrained neural net to be smoothed
    mu- A 1xd numpy array of means
    sigma- A dxd positive, semidefinite numpy covariance matrix
    X- The batch of images to be smoothed
    y- The correct labels for the images
    n_samples- the number of samples of each image

    Returns:
    g- The top 2 raw scores for images as topk
    yp- the top predicted score'''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Using sigma and mu tensors to create random noise with those values
    rng = np.random.default_rng()
    scores = torch.zeros((len(X),n_samples,10)).to(device) # shape is (images in batch, n_samples, number of classes)
    yp = []

    #Below loop works image by image
    for n in range(len(X)):
        # Creating random noise with custom mean vector and covariance matrix
        epsilon = rng.standard_normal(size=(n_samples, 784))
        epsilon = mu[n] + epsilon @ np.sqrt(sigma[n])   # current mu and sigma for this image
        epsilon = np.reshape(epsilon,(n_samples,28,28))
        
        # Getting the scores of the images with random noise added to images
        epsilon_torch = torch.from_numpy(epsilon).float().to(device)

        current_img = X[n].expand(n_samples, -1, -1) # n_samples of each image (second dimension), (n_samples, 28, 28) shape

        scores[n] = pretrained(current_img+epsilon_torch)
        
    # Getting probabilities of each class, top 2 likely classes based on smoothing, and predicted image labels
    avg_scores = scores.mean(dim=1)
    for n in range(len(X)):
        yp.append(np.argmax(avg_scores[n].detach().cpu().numpy()).item())
    g = torch.topk(avg_scores, 2)

    return g, yp
