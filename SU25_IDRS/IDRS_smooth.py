import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def IDRS_matrices(pretrained, mu, sigma, X, n_samples=50):
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
    epsilon = np.ndarray((len(X),n_samples,28,28))
    for n in range(len(X)):
        # Creating random noise with custom mean vector and covariance matrix
        temp = rng.multivariate_normal(mu[n],sigma[n],size=(n_samples))   # mu must be 1D, temp is (n_samples, 784) tensor
        temp = np.reshape(temp,(n_samples,28,28))
        epsilon[n] = temp # Epsilon ends up being (len(X), n_samples, 28, 28)

    # Getting the scores of the images with random noise added to images
    epsilon_torch = torch.from_numpy(epsilon).float().to(device)
    X = X.expand(-1, n_samples, -1, -1) # n_samples of each image (second dimension), (len(X), n_samples, 28, 28) shape
    scores = torch.zeros((len(X),n_samples,10)).to(device) # shape is (images in batch, n_samples, number of classes)
    for n in range(len(X)):
        scores[n] = pretrained(X[n]+epsilon_torch[n])

    # Getting probabilities of each class, top 2 likely classes based on smoothing, and predicted image labels
    probs = torch.softmax(scores, dim=2)    # Softmax each set of scores
    avg_probs = probs.mean(dim=1)
    g = torch.topk(avg_probs, 2)
    yp = []
    for n in range(len(scores)):
        yp.append(np.argmax(scores[n]).item())

    return g, yp
