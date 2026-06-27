import argparse
import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
LAYERWISE_AVG_DIR = PROJECT_ROOT / "LAYERWISE_AVG"
OUTPUT_AVG_DIR = PROJECT_ROOT / "OUTPUT_AVG"

for path in (SCRIPT_DIR, PROJECT_ROOT, LAYERWISE_AVG_DIR, OUTPUT_AVG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from LAYERWISE_AVG.LWA_certification import (  # noqa: E402
    _norm_label,
    _randomized_smoothing_radius_from_logits,
    certify_all_radius_results,
    certify_lwa_kp,
)
# LWA_MC integration is intentionally disabled in MAIN for now.
# from LAYERWISE_AVG.LWA_MC_model import LWMC  # noqa: E402
from LAYERWISE_AVG.LWA_model import LWRS  # noqa: E402
from LAYERWISE_AVG.LWA_utils import (  # noqa: E402
    DEFAULT_CERTIFICATE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SV_PATH,
    LEGACY_MODEL_PATH,
    compute_sv,
    ensure_model_dir,
    epoch,
    epoch_adversarial,
    evaluate_clean,
    evaluate_l2,
    get_device,
    get_mnist_loaders,
    load_or_initialize_checkpoint,
    pgd_l2,
)
from OUTPUT_AVG.OA_model import OutputAveragingRS  # noqa: E402


OUTPUT_AVG_MODEL_DIR = OUTPUT_AVG_DIR / "models"
DEFAULT_OUTPUT_AVG_MODEL_PATH = OUTPUT_AVG_MODEL_DIR / "OUTPUT_AVG_2regu_width256_model.pt"
DEFAULT_OUTPUT_AVG_SV_PATH = OUTPUT_AVG_MODEL_DIR / "sv_min_2regu_width256.pt"
DEFAULT_OUTPUT_AVG_CERTIFICATE_PATH = OUTPUT_AVG_MODEL_DIR / "output_avg_2regu_width256_certificate_results.pt"
OUTPUT_AVG_FAMILY = "output_avg"

FAMILIES = ["lwa_k_p", "lwa_k_abs", "affine_deviation", "affine"]
FAMILY_NAMES = {
    "lwa_k_p": "max(R_LWA_1,R_LWA_2) with K_p",
    "lwa_k_abs": "max(R_LWA_1,R_LWA_2) with K_abs",
    "affine_deviation": "Affine surrogate deviation r_p",
    "affine": "Affine surrogate r_aff,p",
}


def _progress_total(loader, max_samples):
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else max_samples
    return min(int(max_samples), int(dataset_size))


def _sample_progress(name, loader, max_samples):
    return tqdm(
        total=_progress_total(loader, max_samples),
        desc=name,
        unit="sample",
        dynamic_ncols=True,
    )


def _is_finite_radius(value):
    return value is not None and not math.isnan(float(value)) and not math.isinf(float(value))


def evaluate_sigma_configuration(model, test_loader, sigma, name, max_samples, n_samples, device):
    print(f"\n=== Evaluating {name} ===")
    total_radius = 0.0
    correct = 0
    total = 0

    model.eval()
    with _sample_progress(name, test_loader, max_samples) as progress:
        for x_batch, y_batch in test_loader:
            for idx in range(x_batch.size(0)):
                if total >= max_samples:
                    break

                x = x_batch[idx].unsqueeze(0).to(device)
                y = y_batch[idx].to(device)
                label, radius, _ = certify_lwa_kp(
                    x,
                    y.unsqueeze(0),
                    model,
                    sigma=sigma,
                    p=2,
                    n_samples=n_samples,
                )

                total += 1
                progress.update(1)
                if label == y.item() and _is_finite_radius(radius):
                    correct += 1
                    total_radius += float(radius)

            if total >= max_samples:
                break

    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_radius = total_radius / correct if correct > 0 else 0.0
    print(f"{name} - Smooth Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{name} - Average Certified Radius: {avg_radius:.4f}" if correct > 0 else f"{name} - No certified samples")
    return accuracy, avg_radius, correct


def evaluate_certificate_suite(
    model,
    test_loader,
    sigma,
    norms,
    max_samples,
    n_samples,
    output_path,
    device,
    name="Requested Certificate Suite",
    seed=None,
):
    print(f"\n=== Evaluating {name} ===")
    norm_labels = [_norm_label(p) for p in norms]
    summary = {
        norm: {family: {"sum": 0.0, "count": 0, "avg": None} for family in FAMILIES}
        for norm in norm_labels
    }
    per_sample = []
    total = 0

    model.eval()
    with _sample_progress(name, test_loader, max_samples) as progress:
        for x_batch, y_batch in test_loader:
            for idx in range(x_batch.size(0)):
                if total >= max_samples:
                    break

                x = x_batch[idx].unsqueeze(0).to(device)
                y = y_batch[idx].to(device)
                if seed is not None:
                    torch.manual_seed(seed + total)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + total)
                radii = certify_all_radius_results(
                    x,
                    y.unsqueeze(0),
                    model,
                    sigma=sigma,
                    norms=norms,
                    n_samples=n_samples,
                )
                per_sample.append({"index": total, "label": int(y.item()), "radii": radii})

                for norm, norm_results in radii.items():
                    for family in FAMILIES:
                        value = norm_results[family]
                        if not _is_finite_radius(value):
                            continue
                        summary[norm][family]["sum"] += float(value)
                        summary[norm][family]["count"] += 1

                total += 1
                progress.update(1)

            if total >= max_samples:
                break

    averages = {}
    for norm in norm_labels:
        averages[norm] = {}
        for family in FAMILIES:
            count = summary[norm][family]["count"]
            avg = summary[norm][family]["sum"] / count if count > 0 else None
            summary[norm][family]["avg"] = avg
            averages[norm][family] = avg

    print("Norm | K_p LWA | K_abs LWA | Affine deviation | Affine")
    print("-" * 70)
    for norm in norm_labels:
        row = ["n/a" if averages[norm][family] is None else f"{averages[norm][family]:.6f}" for family in FAMILIES]
        print(f"{norm:>4} | {row[0]:>8} | {row[1]:>9} | {row[2]:>16} | {row[3]:>8}")

    payload = {
        "sigma": list(sigma),
        "model_name": name,
        "norms": norm_labels,
        "families": FAMILIES,
        "family_names": FAMILY_NAMES,
        "max_samples": max_samples,
        "total_samples": total,
        "averages": averages,
        "summary": summary,
        "samples": per_sample,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Certificate suite saved to {output_path}")
    return payload


def train_or_load_model(
    model,
    optimizer,
    model_path,
    sv_path,
    args,
    device,
    name,
    train_loader,
    test_loader,
    legacy_path=None,
):
    if args.sv or not Path(sv_path).exists():
        compute_sv(model, sv_path)
        print(f"{name} singular values computed.")

    loaded_path = None if args.train else load_or_initialize_checkpoint(
        model,
        model_path=model_path,
        device=device,
        legacy_path=legacy_path,
    )
    if loaded_path is not None:
        print(f"{name} loaded from {loaded_path}")
        return loaded_path

    print(f"=== Training {name} ===")
    print(f"Optimizer weight decay: {args.weight_decay:g}")
    for epoch_num in range(args.epochs):
        print(f"{name} Epoch {epoch_num + 1}/{args.epochs}:")
        sv = compute_sv(model, sv_path)
        train_err, _ = epoch_adversarial(
            train_loader,
            model,
            pgd_l2,
            optimizer,
            sv,
            args.epsilon,
            args.alpha,
            args.num_iter,
            device=device,
        )
        test_err, _ = epoch(test_loader, model, opt=None, sv=sv, device=device)
        adv_err, _ = epoch_adversarial(
            test_loader,
            model,
            pgd_l2,
            None,
            sv,
            args.epsilon,
            args.alpha,
            args.num_iter,
            device=device,
        )
        print(f"| Train Accuracy:        {100 - train_err * 100:.2f}%")
        print(f"| Test Accuracy:         {100 - test_err * 100:.2f}%")
        print(f"| Adversarial Accuracy:  {100 - adv_err * 100:.2f}%")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"{name} saved to {model_path}")
    return None


def evaluate_output_avg_rs(
    model,
    test_loader,
    sigma,
    max_samples,
    n_samples,
    output_path,
    device,
    seed=None,
    name="OUTPUT_AVG Standard RS",
):
    print(f"\n=== Evaluating {name} ===")
    norm = "2"
    summary = {norm: {OUTPUT_AVG_FAMILY: {"sum": 0.0, "count": 0, "avg": None}}}
    per_sample = []
    correct = 0
    total = 0
    total_radius = 0.0

    model.eval()
    with _sample_progress(name, test_loader, max_samples) as progress:
        for x_batch, y_batch in test_loader:
            for idx in range(x_batch.size(0)):
                if total >= max_samples:
                    break

                x = x_batch[idx].unsqueeze(0).to(device)
                y = y_batch[idx].to(device)
                if seed is not None:
                    torch.manual_seed(seed + total)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + total)

                with torch.no_grad():
                    sample_logits = model.sample_logits(x, n_samples=n_samples)
                label, radius = _randomized_smoothing_radius_from_logits(sample_logits, sigma, y.unsqueeze(0))
                radius = float(radius) if _is_finite_radius(radius) else 0.0
                radii = {norm: {OUTPUT_AVG_FAMILY: radius}}
                per_sample.append({"index": total, "label": int(y.item()), "radii": radii})

                if label == y.item() and radius > 0:
                    correct += 1
                    total_radius += radius
                    summary[norm][OUTPUT_AVG_FAMILY]["sum"] += radius
                    summary[norm][OUTPUT_AVG_FAMILY]["count"] += 1

                total += 1
                progress.update(1)

            if total >= max_samples:
                break

    count = summary[norm][OUTPUT_AVG_FAMILY]["count"]
    avg = summary[norm][OUTPUT_AVG_FAMILY]["sum"] / count if count > 0 else None
    summary[norm][OUTPUT_AVG_FAMILY]["avg"] = avg
    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_radius = total_radius / correct if correct > 0 else 0.0
    print(f"{name} - Smooth Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{name} - Average Certified Radius: {avg_radius:.4f}" if correct > 0 else f"{name} - No certified samples")

    payload = {
        "sigma": float(sigma),
        "model_name": name,
        "norms": [norm],
        "families": [OUTPUT_AVG_FAMILY],
        "family_names": {OUTPUT_AVG_FAMILY: "OUTPUT_AVG standard RS"},
        "max_samples": max_samples,
        "total_samples": total,
        "averages": {norm: {OUTPUT_AVG_FAMILY: avg}},
        "summary": summary,
        "samples": per_sample,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"OUTPUT_AVG certificate suite saved to {output_path}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Train and certify analytic layerwise averaging ReGU smoothing")
    parser.add_argument("--train", action="store_true", help="Force retraining even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--sigma", nargs=2, type=float, default=[5.0, 5.0], help="Layerwise averaging ReGU sigmas")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width for the two ReGU/ReLU layers")
    parser.add_argument("--samples", type=int, default=1000, help="Certification-only samples for R_LWA_2")
    parser.add_argument("--verification-seed", type=int, default=0, help="Seed for randomized certificate sampling")
    parser.add_argument("--skip-layer-eval", action="store_true", help="Skip the per-layer sigma diagnostic certification sweep")
    parser.add_argument("--cert-samples", type=int, default=1000, help="Number of test examples to certify")
    parser.add_argument("--norms", nargs="+", default=["1", "2", "inf"], help="Norms to certify")
    parser.add_argument("--batch-size", type=int, default=100, help="MNIST batch size")
    parser.add_argument("--data-dir", default=None, help="MNIST data directory")
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST if missing")
    parser.add_argument("--sv", action="store_true", help="Recompute singular-value regularizer cache")
    parser.add_argument("--lr", type=float, default=0.1, help="SGD learning rate")
    parser.add_argument("--weight-decay", type=float, default=3e-3, help="SGD L2 weight decay for real weight regularization")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PGD L2 epsilon")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--num-iter", type=int, default=40, help="PGD iterations")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Model checkpoint path")
    parser.add_argument("--certificate-path", default=str(DEFAULT_CERTIFICATE_PATH), help="Saved certificate payload")
    parser.add_argument("--skip-output-avg", action="store_true", help="Skip OUTPUT_AVG standard input RS certification")
    parser.add_argument("--output-avg-sigma", type=float, default=5.0, help="Input-noise sigma for OUTPUT_AVG standard RS")
    parser.add_argument("--output-avg-train-samples", type=int, default=1, help="Input-noise samples used while training OUTPUT_AVG")
    parser.add_argument("--output-avg-samples", type=int, default=1000, help="Samples for OUTPUT_AVG standard RS")
    parser.add_argument("--output-avg-model-path", default=str(DEFAULT_OUTPUT_AVG_MODEL_PATH), help="OUTPUT_AVG checkpoint path")
    parser.add_argument("--output-avg-sv-path", default=str(DEFAULT_OUTPUT_AVG_SV_PATH), help="OUTPUT_AVG singular-value cache path")
    parser.add_argument("--output-avg-certificate-path", default=str(DEFAULT_OUTPUT_AVG_CERTIFICATE_PATH), help="Saved OUTPUT_AVG certificate payload")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_dir()
    OUTPUT_AVG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=not args.no_download,
    )

    model = LWRS(sigma=args.sigma, n_samples=args.samples, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_or_load_model(
        model,
        optimizer,
        Path(args.model_path),
        DEFAULT_SV_PATH,
        args,
        device,
        "Analytic LWA ReGU Model",
        train_loader,
        test_loader,
        legacy_path=LEGACY_MODEL_PATH,
    )

    print("\n=== Evaluating Analytic LWA ReGU Model ===")
    clean_acc = evaluate_clean(model, test_loader, device=device)
    adv_acc = evaluate_l2(model, test_loader, args.epsilon, args.alpha, args.num_iter, device=device)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Adversarial Accuracy: {adv_acc:.4f}")

    sigma = list(args.sigma)
    if args.skip_layer_eval:
        print("\n=== Skipping per-layer sigma diagnostic sweep ===")
        layer_results = {}
    else:
        layer_results = {}
        for layer_idx, sigma_l in enumerate(sigma):
            sigma_config = [0.0] * len(sigma)
            sigma_config[layer_idx] = sigma_l
            layer_results[f"layer_{layer_idx}"] = evaluate_sigma_configuration(
                model,
                test_loader,
                sigma_config,
                f"Layer {layer_idx} ReGU - sigma {sigma_l}",
                args.cert_samples,
                args.samples,
                device,
            )
        layer_results["all_enabled"] = evaluate_sigma_configuration(
            model, test_loader, sigma, "All ReGU layers enabled", args.cert_samples, args.samples, device
        )

    certificate_results = evaluate_certificate_suite(
        model,
        test_loader,
        sigma=sigma,
        norms=args.norms,
        max_samples=args.cert_samples,
        n_samples=args.samples,
        output_path=args.certificate_path,
        device=device,
        name="LWA Certificate Suite",
        seed=args.verification_seed,
    )

    output_avg_results = None
    if not args.skip_output_avg:
        output_avg_model = OutputAveragingRS(
            sigma=args.output_avg_sigma,
            n_samples=args.output_avg_train_samples,
        ).to(device)
        output_avg_optimizer = optim.SGD(output_avg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_or_load_model(
            output_avg_model,
            output_avg_optimizer,
            Path(args.output_avg_model_path),
            Path(args.output_avg_sv_path),
            args,
            device,
            "OUTPUT_AVG Standard RS Model",
            train_loader,
            test_loader,
            legacy_path=None,
        )
        output_avg_model.n_samples = args.output_avg_samples
        output_avg_results = evaluate_output_avg_rs(
            output_avg_model,
            test_loader,
            sigma=args.output_avg_sigma,
            max_samples=args.cert_samples,
            n_samples=args.output_avg_samples,
            output_path=args.output_avg_certificate_path,
            device=device,
            seed=args.verification_seed,
        )

    combined_path = Path(args.certificate_path).with_name(f"{Path(args.certificate_path).stem}_combined.pt")
    torch.save(
        {
            "layer_results": layer_results,
            "certificate_results": certificate_results,
            "output_avg_results": output_avg_results,
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
        },
        combined_path,
    )
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
