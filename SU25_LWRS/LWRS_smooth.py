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
import argparse  # Add this import
from LWRS_model import LWRS, LWRS_tilde
from tqdm import tqdm
from SU25_BASECODE.train_save_smooth import epoch, epoch_adversarial, pgd_l2, evaluate_clean, evaluate_under_attack, phi_inverse, smooth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# PGD attack parameters
epsilon = 0.1  # 10% of pixel range
alpha = 0.01
num_iter = 40  # Number of iterations

# Initialize models using correct class names
LWRS_model = LWRS(sigma=1).to(device)  # h(x) model, sigma is the INPUT noise!
LWRS_tilde_model = LWRS_tilde(sigma=0.1).to(device)  # h~(x) model
opt_LWRS = optim.SGD(LWRS_model.parameters(), lr=0.1)

if __name__ == '__main__': 
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate LWRS models')
    parser.add_argument('--train', action='store_true', help='Force retraining of models even if saved model exists')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train LWRS (h(x)) model
    print("=== Training LWRS (h(x)) Model ===")
    
    # Check if we should train (either no saved model exists OR --train flag is used)
    should_train = not os.path.exists("models/LWRS_model.pt") or args.train
    
    if should_train:
        if args.train:
            print("--train flag detected. Retraining model...")
        else:
            print("No saved model found. Training new model...")
            
        for epoch_num in range(args.epochs):
            print(f"Epoch {epoch_num+1}/{args.epochs}:")
            # Fix: Add the missing attack function and parameters
            train_err, train_loss = epoch_adversarial(train_loader, LWRS_model, pgd_l2, opt_LWRS, epsilon, alpha, 10)  # Use 10 iterations
            test_err, test_loss = epoch(test_loader, LWRS_model, None)  
            # Keep evaluation at 40 iterations for stronger attacks
            adv_err, adv_loss = epoch_adversarial(test_loader, LWRS_model, pgd_l2, None, epsilon, alpha, num_iter)
            print("| Train Accuracy:     {:.2f}%".format(100 - train_err * 100))
            print("| Test Accuracy:      {:.2f}%".format(100 - test_err * 100))
            print("| Adversarial Accuracy:  {:.2f}%".format(100 - adv_err * 100))
        torch.save(LWRS_model.state_dict(), "models/LWRS_model.pt")
        print("LWRS model saved.")
    else:
        # Load saved LWRS model
        LWRS_model.load_state_dict(torch.load("models/LWRS_model.pt", map_location=device))
        print("LWRS model loaded from saved file.")

    # Transfer weights from LWRS to LWRS_tilde
    print("\n=== Transferring weights from LWRS to LWRS_tilde ===")
    LWRS_tilde_model.load_state_dict(LWRS_model.state_dict())
    print("Weights transferred.")

    # Evaluate both models
    print("\n=== Evaluating Models ===")
    
    # Evaluate LWRS (h(x))
    clean_acc_h = evaluate_clean(LWRS_model, test_loader)
    adv_acc_h = evaluate_under_attack(LWRS_model, test_loader, epsilon, alpha, num_iter)
    print(f"LWRS (h(x)) - Clean Accuracy: {clean_acc_h:.4f}")
    print(f"LWRS (h(x)) - Adversarial Accuracy: {adv_acc_h:.4f}")
    
    # Evaluate LWRS_tilde (h~(x))
    clean_acc_h_tilde = evaluate_clean(LWRS_tilde_model, test_loader)
    adv_acc_h_tilde = evaluate_under_attack(LWRS_tilde_model, test_loader, epsilon, alpha, num_iter)
    print(f"LWRS_tilde (h~(x)) - Clean Accuracy: {clean_acc_h_tilde:.4f}")
    print(f"LWRS_tilde (h~(x)) - Adversarial Accuracy: {adv_acc_h_tilde:.4f}")

    # Smoothing evaluation
    print("\n=== Randomized Smoothing Results ===")
    
    # Test smoothing on both models
    for model_name, model in [("LWRS (h(x))", LWRS_model), ("LWRS_tilde (h~(x))", LWRS_tilde_model)]:
        print(f"\n--- {model_name} Smoothing ---")
        
        # Test on entire test dataset
        total_radius = 0
        correct_smooth = 0
        total_samples = 0
        
        # Go through entire test dataset
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc=f"Smoothing {model_name}")):
            for i in range(x_batch.size(0)):  # Process each sample in the batch
                x = x_batch[i].unsqueeze(0).to(device)
                y = y_batch[i].to(device)
                y_tensor = y.unsqueeze(0)
                
                # Get smoothed classifier prediction and radius
                label, radius = smooth(x, y_tensor, model, sigma=1.0, n_samples=500)
                
                total_samples += 1
                if label == y.item():
                    correct_smooth += 1
                    total_radius += radius
                
                # Print progress every 1000 samples
                if total_samples % 1000 == 0:
                    print(f"Processed {total_samples}/10000 samples...")
        
        print(f"{model_name} - Smooth Accuracy: {correct_smooth}/{total_samples} ({100*correct_smooth/total_samples:.1f}%)")
        if correct_smooth > 0:
            print(f"{model_name} - Average Certified Radius: {total_radius/correct_smooth:.4f}")
        else:
            print(f"{model_name} - No correctly classified samples for radius calculation")