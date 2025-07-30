import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from LWRS_model import LWRS, DNN_4
from tqdm import tqdm
from SU25_BASECODE.train_save_smooth import epoch, epoch_adversarial, pgd_linf, pgd_l2, evaluate_clean, evaluate_under_l2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# PGD attack parameters
epsilon = 1  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

LWRS_model = LWRS().to(device)
opt_LWRS = optim.SGD(LWRS_model.parameters(), lr=0.1)

dnn4 = DNN_4().to(device)

dnn4_clean = DNN_4(noise_std=0).to(device)
opt_dnn4_clean = optim.SGD(dnn4_clean.parameters(), lr=0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Training and Evaluation
if not os.path.exists("SU25_LWRS/models/dnn_4_l2e1_test.pt"):
    for epoch_num in range(10):
        print(f"Epoch {epoch_num+1}:")
        train_err, train_loss = epoch_adversarial(dnn4_clean, train_loader, pgd_l2, opt_dnn4_clean, epsilon, alpha, num_iter)
        # test_err, test_loss = epoch(test_loader, LWRS_model)
        # adv_err, adv_loss = epoch_adversarial(test_loader, LWRS_model, pgd_linf, epsilon, alpha, num_iter)
        print("| Train Accuracy:     {:.2f}%".format(100 - train_err * 100))
        # print("| Test Accuracy:      {:.2f}%".format(100 - test_err * 100))
        # print("| Adversarial Accuracy:  {:.2f}%".format(100 - adv_err * 100))
    torch.save(dnn4_clean.state_dict(), "SU25_LWRS/models/dnn_4_l2e1_test.pt")

# Load saved LWRS model
dnn4.load_state_dict(torch.load("SU25_LWRS/models/dnn_4_l2e1_test.pt", map_location=device))

# Evaluating and printing results for LWRS
clean_acc = evaluate_clean(dnn4, test_loader)
adv_acc = evaluate_under_l2(dnn4, test_loader, epsilon, alpha, num_iter)
print(f"Accuracy of model on clean data: {clean_acc:.4f}")
print(f"Accuracy of model under PGD attack: {adv_acc:.4f}")