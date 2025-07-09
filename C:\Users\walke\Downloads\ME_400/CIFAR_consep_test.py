# Import data and libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import warnings

# Suppress the FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Load the saved robust and non-robust datasets
robust_dataset = torch.load("robust_dataset.pt")
non_robust_dataset = torch.load("non_robust_dataset.pt")

# Extract the data part from the robust and non-robust examples
X_robust = robust_dataset.tensors[0].numpy()
X_non_robust = non_robust_dataset.tensors[0].numpy()

# Check separability using the convex hull method
def check_sep(X_robust, X_non_robust):
    A = np.array(X_robust)
    A = np.reshape(A, (A.shape[0], -1)).T

    num_robust = X_robust.shape[0]
    one_vec = np.ones(num_robust)

    B = np.array(X_non_robust)
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
        print("Test unsuccessful: There exists a y inside the convex hull of X_robust.")
    else:
        print("Test successful: No y is inside the convex hull of X_robust.")

# Test the separability of the subsets
check_sep(X_robust, X_non_robust)