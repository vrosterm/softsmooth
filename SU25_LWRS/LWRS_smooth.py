
''' ARGUMENT PARSERS FOR SMOOTHING

parser.add_argument('--train', action='store_true', help='Force retraining of model even if saved model exists')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
parser.add_argument('--sigma', type=float, default=1.0, help='Layerwise smoothing sigma (default: 1.0)')
parser.add_argument('--samples', type=int, default=500, help='Number of samples for smoothing (default: 500)')

'''

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
import argparse
from LWRS_model import LWRS
from tqdm import tqdm
import math
from SU25_BASECODE.train_save_smooth import epoch, epoch_adversarial, pgd_l2, evaluate_clean, evaluate_l2, phi_inverse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def certify_radius(X, y, model, sigma, z_config):
    """
    Apply layerwise smoothing based on z_config.
    z_config: [z0, z1, z2] where 1 = apply smoothing at that layer, 0 = no smoothing
    """
    # Set the model's noise configuration
    model.set_noise_layers(z_config)
    
    # For layerwise smoothing, we don't add input noise - the model handles internal noise
    scores = model(X)[0]
    label = torch.argmax(scores)
    
    if label != y.item():
        radius = 0.0
        return label.item(), radius
    
    best_scores = torch.topk(scores, 2)    
    radius = sigma * (phi_inverse(best_scores.values[0], 0) - phi_inverse(best_scores.values[1], 0)) / 2
    return label.item(), radius.item()

def certify_layerwise_radius(X, y, model, sigma):
    """
    Apply layerwise smoothing based on z_config.
    z_config: [z0, z1, z2] where 1 = apply smoothing at that layer, 0 = no smoothing
    """
    #Initializing list for all radii values
    radii = []
    sv = torch.load('SU25_LWRS/models/sv_min.pt')

    # Set the model's noise configuration
    model.set_noise_layers([1,1,1])
    
    # For layerwise smoothing, we don't add input noise - the model handles internal noise
    scores = model(X)[0]
    label = torch.argmax(scores)
    
    if label != y.item():
        radius = 0.0
        return label.item(), radius
    
    best_scores = torch.topk(scores, 2)    
    for i in range(len(sv)):
        radius = sigma[-(1+i)] * (phi_inverse(best_scores.values[0], 0) - phi_inverse(best_scores.values[1], 0)) / 2
        radii.append(radius * math.prod(sv[0:i+1]))
    return label.item(), max(radii)

def evaluate_layer_smoothing(model, test_loader, layer_name, sigma, z_config):
    """Evaluate a specific layer smoothing configuration"""
    print(f"\n=== Evaluating {layer_name} ===")
    
    total_radius = 0
    correct_smooth = 0
    total_samples = 0
    
    # Go through test dataset -- ADD LIMIT IF THIS TAKES TOO LONG
    max_samples = 1000 # Total sample size of 10,000, using 1000.
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc=f"Smoothing {layer_name}")):
        for i in range(x_batch.size(0)):
            if total_samples >= max_samples:
                break
                
            x = x_batch[i].unsqueeze(0).to(device)
            y = y_batch[i].to(device)
            y_tensor = y.unsqueeze(0)
            
            # Apply layerwise smoothing with z_config
            if z_config == [1,1,1]:
                label, radius = certify_layerwise_radius(x, y_tensor, model, sigma)
            elif z_config != [1,1,1]:
                label, radius = certify_radius(x, y_tensor, model, sigma, z_config)

            
            total_samples += 1
            if label == y.item() and not math.isinf(radius):
                correct_smooth += 1
                total_radius += radius
        
        if total_samples >= max_samples:
            break
    
    # Results
    smooth_accuracy = 100 * correct_smooth / total_samples
    avg_radius = total_radius / correct_smooth if correct_smooth > 0 else 0
    
    print(f"{layer_name} - Smooth Accuracy: {correct_smooth}/{total_samples} ({smooth_accuracy:.1f}%)")
    if correct_smooth > 0:
        print(f"{layer_name} - Average Certified Radius: {avg_radius:.4f}")
        print(f"{layer_name} - Total Certified Samples: {correct_smooth}")
    else:
        print(f"{layer_name} - No correctly classified samples for radius calculation")
    
    return smooth_accuracy, avg_radius, correct_smooth

def compute_sv(model):
    sv_min = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] == layer.weight.data.shape[1]:
            W_inv = torch.linalg.inv(layer.weight.data)
            # Compute singular values
            S = torch.linalg.svdvals(W_inv)
            sv_min.append(S[-1])
    torch.save(sv_min, "SU25_LWRS/models/sv_min.pt")
    return sv_min

# Data
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# PGD attack parameters
epsilon = 0.1  # 10% of pixel range
alpha = 0.01
num_iter = 40  # Number of iterations

if __name__ == '__main__': 
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train LWRS model and evaluate layerwise smoothing')
    parser.add_argument('--train', action='store_true', help='Force retraining of model even if saved model exists')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--sigma', type=float, default=[0.5, 0.1, 0.1], help='Layerwise smoothing sigma (default: [1, 0.1, 0.1])')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for smoothing (default: 500)')
    parser.add_argument('--sv', action='store_true', help='Calculates and saves Singular values')
    args = parser.parse_args()

    # Initialize base LWRS model h(x)
    LWRS_model = LWRS(sigma=args.sigma, n_samples=args.samples).to(device)  # sigma for internal layerwise noise
    opt_LWRS = optim.SGD(LWRS_model.parameters(), lr=0.1)
    # Singular Values
    
    # Create models directory if it doesn't exist
    os.makedirs("SU25_LWRS/models", exist_ok=True)

    # Check if SVs need to be recalculated
    if args.sv or not os.path.exists("SU25_LWRS/models/sv_min.pt"):
        compute_sv(LWRS_model)
        print("Singular values computed.")
    
    # Check if we should train
    if not os.path.exists("SU25_LWRS/models/LWRS_layerwise_model.pt") or args.train:
        # Train base LWRS (h(x)) model
        print("=== Training Base LWRS h(x) Model ===")
        if args.train:
            print("--train flag detected. Retraining model...")
        else:
            print("No saved model found. Training new model...")
            
        # Set no internal noise for training (z = [0,0,0])
        LWRS_model.set_noise_layers([0, 0, 0])
        
        for epoch_num in range(args.epochs):
            print(f"Epoch {epoch_num+1}/{args.epochs}:")
            sv = compute_sv(LWRS_model)
            train_err, train_loss = epoch_adversarial(train_loader, LWRS_model, pgd_l2, opt_LWRS, sv, epsilon, alpha)
            print(compute_sv(LWRS_model))
            test_err, test_loss = epoch(test_loader, LWRS_model, opt=None, sv=sv)
            adv_err, adv_loss = epoch_adversarial(test_loader, LWRS_model, pgd_l2, None, sv, epsilon, alpha)
            print("| Train Accuracy:     {:.2f}%".format(100 - train_err * 100))
            print("| Test Accuracy:      {:.2f}%".format(100 - test_err * 100))
            print("| Adversarial Accuracy:  {:.2f}%".format(100 - adv_err * 100))
        
        torch.save(LWRS_model.state_dict(), "SU25_LWRS/models/LWRS_layerwise_model.pt")
        print("Base LWRS h(x) model saved.")

    # Load saved LWRS model 
    LWRS_model.load_state_dict(torch.load("SU25_LWRS/models/LWRS_layerwise_model.pt", map_location=device))
    print("Base LWRS h(x) model loaded from saved file.")


    # Evaluate base model
    print("\n=== Evaluating Base LWRS h(x) Model ===")
    LWRS_model.set_noise_layers([0, 0, 0])  # No internal noise for evaluation
    clean_acc_h = evaluate_clean(LWRS_model, test_loader)
    adv_acc_h = evaluate_l2(LWRS_model, test_loader, epsilon, alpha, num_iter)
    print(f"Base h(x) - Clean Accuracy: {clean_acc_h:.4f}")
    print(f"Base h(x) - Adversarial Accuracy: {adv_acc_h:.4f}")
    
    # Evaluate the three layerwise smoothing configurations
    print(f"\n=== Evaluating Layerwise Smoothing (σ = {args.sigma}) ===")
    
    results = {}
    
    # Configuration 1: g_tail(z^(0)) - Smoothing at layer 0
    results['layer_0'] = evaluate_layer_smoothing(
        LWRS_model, test_loader, f"g_tail(z^(0)) - Layer 0 Smoothing - {args.sigma[0]}", 
        sigma=args.sigma[0], z_config=[1, 0, 0]
    )
    
    # Configuration 2: g_tail(z^(1)) - Smoothing at layer 1
    results['layer_1'] = evaluate_layer_smoothing(
        LWRS_model, test_loader, f"g_tail(z^(1)) - Layer 1 Smoothing - {args.sigma[1]}", 
        sigma=args.sigma[1], z_config=[0, 1, 0]
    )
    
    # Configuration 3: g_tail(z^(2)) - Smoothing at layer 2
    results['layer_2'] = evaluate_layer_smoothing(
        LWRS_model, test_loader, f"g_tail(z^(2)) - Layer 2 Smoothing - {args.sigma[2]}", 
        sigma=args.sigma[2], z_config=[0, 0, 1]
    )

    # Configuration 3: g_tail(z^(2)) - Smoothing at layer 2
    results['all_enabled'] = evaluate_layer_smoothing(
        LWRS_model, test_loader, "All layers enabled", 
        sigma=args.sigma, z_config=[1, 1, 1]
    )


    
    # Summary comparison
    print(f"\n=== Summary Comparison (σ = {args.sigma}) ===")
    print("Configuration               | Smooth Acc | Avg Radius | Certified")
    print("-" * 65)
    
    for config, (smooth_acc, avg_radius, certified) in results.items():
        config_name = {
            'layer_0': 'g_tail(z^(0)) - Layer 0',
            'layer_1': 'g_tail(z^(1)) - Layer 1', 
            'layer_2': 'g_tail(z^(2)) - Layer 2',
            'all_enabled': 'All layers enabled'
        }[config]
        
        print(f"{config_name:<26} | {smooth_acc:>9.1f}% | {avg_radius:>10.4f} | {certified:>9d}")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k][1])  # Best by avg radius
    best_radius = results[best_config][1]
    
    print(f"\nBest Configuration: {best_config} with average radius {best_radius:.4f}")
    
    # Save results
    torch.save(results, f"models/layerwise_smoothing_results_sigma_{args.sigma}.pt")
    print(f"Results saved to models/layerwise_smoothing_results_sigma_{args.sigma}.pt")