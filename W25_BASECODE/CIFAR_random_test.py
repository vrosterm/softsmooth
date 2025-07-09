# Import data and libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import warnings

# Suppress the FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
transform = transforms.Compose([transforms.ToTensor()])
CIFAR10_data = datasets.CIFAR10("C:/Users/walke/data", train=False, download=True, transform=transform)

# Randomly partition the dataset into two subsets of equal size
subset_size = len(CIFAR10_data) // 2
subset1, subset2 = random_split(CIFAR10_data, [subset_size, subset_size])

# Convert subsets to TensorDatasets
X_1 = torch.stack([subset1[i][0] for i in range(len(subset1))]).numpy()
X_2 = torch.stack([subset2[i][0] for i in range(len(subset2))]).numpy()

# Check separability using the convex hull method
def check_sep(X_1, X_2):
    A = np.array(X_1)
    A = np.reshape(A, (A.shape[0], -1)).T

    num_robust = X_1.shape[0]
    one_vec = np.ones(num_robust)

    B = np.array(X_2)
    B = np.reshape(B, (B.shape[0], -1))

    test_failed = False

    for y in tqdm(B, desc="Checking separability"):
        lam = cp.Variable(num_robust)
        obj = cp.Minimize(1)  # Dummy objective
        const = [lam >= 0, one_vec @ lam == 1, A @ lam == y]
        prob = cp.Problem(obj, const)
        try:
            result = prob.solve(solver=cp.SCS, verbose=False)  # Use SCS solver
            if result is not None and result != float('inf'):
                test_failed = True
                break
        except cp.error.SolverError:
            continue
    
    if test_failed:
        print("Test unsuccessful: There exists a y inside the convex hull of X_1.")
    else:
        print("Test successful: No y is inside the convex hull of X_1.")

# Test the separability of the subsets
check_sep(X_1, X_2)