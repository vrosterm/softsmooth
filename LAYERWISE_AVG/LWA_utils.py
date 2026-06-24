from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYERWISE_AVG_DIR = Path(__file__).resolve().parent
MODEL_DIR = LAYERWISE_AVG_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "LWA_layerwise_model.pt"
DEFAULT_MC_MODEL_PATH = MODEL_DIR / "LWA_MC_layerwise_model.pt"
DEFAULT_SV_PATH = MODEL_DIR / "sv_min.pt"
DEFAULT_MC_SV_PATH = MODEL_DIR / "mc_sv_min.pt"
DEFAULT_CERTIFICATE_PATH = MODEL_DIR / "lwa_certificate_results.pt"
DEFAULT_MC_CERTIFICATE_PATH = MODEL_DIR / "lwa_mc_certificate_results.pt"
DEFAULT_PLOT_PATH = MODEL_DIR / "lwa_certificate_waterfall.png"
LEGACY_MODEL_PATH = PROJECT_ROOT / "SU25_LWRS" / "models" / "LWRS_layerwise_model.pt"


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_mnist_loaders(data_dir=None, batch_size=100, download=True):
    if data_dir is None:
        data_dir = PROJECT_ROOT.parent / "data"
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(str(data_dir), train=True, download=download, transform=transform)
    mnist_test = datasets.MNIST(str(data_dir), train=False, download=download, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def compute_sv(model, output_path=DEFAULT_SV_PATH):
    sv_min = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] == layer.weight.data.shape[1]:
            weight_inv = torch.linalg.pinv(layer.weight.data)
            singular_values = torch.linalg.svdvals(weight_inv)
            sv_min.append(singular_values[-1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sv_min, output_path)
    return sv_min


def pgd_l2(model, X, y, epsilon=0.1, alpha=0.01, num_iter=40):
    delta = torch.zeros_like(X)

    for _ in range(num_iter):
        delta.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta.detach() + alpha * grad.detach()

        delta_flat = delta.view(delta.shape[0], -1)
        delta_norms = torch.linalg.vector_norm(delta_flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(delta_norms, torch.tensor(epsilon, device=delta.device)) / delta_norms
        delta = (delta_flat * factors).view_as(delta).detach()

    return delta.detach()


def _sv_penalty(sv, device):
    if sv is None:
        return 0.0
    if len(sv) == 0:
        return 0.0
    return sum(float(value.detach().to(device).item()) for value in sv)


def epoch(loader, model, opt=None, sv=None, device=None):
    if device is None:
        device = get_device()
    total_loss, total_err = 0.0, 0.0
    model.train(opt is not None)

    for X, y in tqdm(loader, desc="Epoch Progress"):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y) - 0.15 * _sv_penalty(sv, device)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, opt=None, sv=None, *attack_args, device=None):
    if device is None:
        device = get_device()
    total_loss, total_err = 0.0, 0.0
    model.train(opt is not None)

    for X, y in tqdm(loader, desc="Adversarial Training"):
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *attack_args)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y) - 0.15 * _sv_penalty(sv, device)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def evaluate_clean(model, loader, device=None):
    if device is None:
        device = get_device()
    model.eval()
    total_err = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            total_err += (yp.max(dim=1)[1] != y).sum().item()

    return 1 - total_err / len(loader.dataset)


def evaluate_l2(model, loader, epsilon=0.1, alpha=0.01, num_iter=40, device=None):
    if device is None:
        device = get_device()
    model.eval()
    total_err = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = pgd_l2(model, X, y, epsilon, alpha, num_iter)
        yp = model(X + delta)
        total_err += (yp.max(dim=1)[1] != y).sum().item()

    return 1 - total_err / len(loader.dataset)


def load_or_initialize_checkpoint(model, model_path=DEFAULT_MODEL_PATH, device=None, legacy_path=LEGACY_MODEL_PATH):
    if device is None:
        device = get_device()
    model_path = Path(model_path)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model_path

    if legacy_path is not None and Path(legacy_path).exists():
        model.load_state_dict(torch.load(legacy_path, map_location=device))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return Path(legacy_path)

    return None
