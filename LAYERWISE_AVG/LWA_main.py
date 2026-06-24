import argparse
import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from LWA_certification import certify_all_radius_results, certify_lwa_kp, _norm_label
from LWA_MC_model import LWMC
from LWA_model import LWRS
from LWA_utils import (
    DEFAULT_CERTIFICATE_PATH,
    DEFAULT_MC_CERTIFICATE_PATH,
    DEFAULT_MC_MODEL_PATH,
    DEFAULT_MC_SV_PATH,
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
                if label == y.item() and radius is not None and not math.isinf(radius):
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
                        if value is None or math.isnan(value) or math.isinf(value):
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
    for epoch_num in range(args.epochs):
        print(f"{name} Epoch {epoch_num + 1}/{args.epochs}:")
        sv = compute_sv(model, sv_path)
        train_err, train_loss = epoch_adversarial(
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
        test_err, test_loss = epoch(test_loader, model, opt=None, sv=sv, device=device)
        adv_err, adv_loss = epoch_adversarial(
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


def compare_regu_to_mc(analytic_model, test_loader, sigma, n_samples, max_batches, seed, device):
    print("\n=== ReGU vs Layerwise Averaging MC Verification ===")
    mc_model = LWMC(sigma=sigma, n_samples=n_samples).to(device)
    mc_model.load_state_dict(analytic_model.state_dict())
    analytic_model.eval()
    mc_model.eval()

    diffs = []
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            torch.manual_seed(seed + batch_idx)
            x = x_batch.to(device)
            analytic_scores = analytic_model(x)
            mc_scores = mc_model(x)
            diffs.append((analytic_scores - mc_scores).abs().detach().cpu())

    if not diffs:
        return {"mean_abs": None, "max_abs": None, "n_samples": n_samples, "batches": 0}

    all_diffs = torch.cat([diff.reshape(-1) for diff in diffs])
    result = {
        "mean_abs": float(all_diffs.mean().item()),
        "max_abs": float(all_diffs.max().item()),
        "n_samples": int(n_samples),
        "batches": int(len(diffs)),
    }
    print(
        "ReGU vs MC | "
        f"samples={result['n_samples']} | batches={result['batches']} | "
        f"mean_abs={result['mean_abs']:.6f} | max_abs={result['max_abs']:.6f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train and certify analytic layerwise averaging ReGU smoothing")
    parser.add_argument("--train", action="store_true", help="Force retraining even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--sigma", nargs=3, type=float, default=[0.5, 0.1, 0.1], help="Layerwise averaging ReGU sigmas")
    parser.add_argument("--samples", type=int, default=1000, help="Certification-only MC samples for R_LWA_2")
    parser.add_argument("--mc-samples", type=int, default=10, help="Layerwise averaging samples used by the trainable MC reference model")
    parser.add_argument("--verify-mc-samples", type=int, default=200, help="Samples used for analytic ReGU vs MC verification")
    parser.add_argument("--mc-certificate-model-samples", type=int, default=None, help="Layerwise MC samples for the MC certificate overlay; defaults to --verify-mc-samples")
    parser.add_argument("--verify-batches", type=int, default=1, help="Number of test batches for ReGU vs MC verification")
    parser.add_argument("--verification-seed", type=int, default=0, help="Seed for ReGU vs MC verification sampling")
    parser.add_argument("--skip-mc", action="store_true", help="Skip training/loading the parallel MC reference model")
    parser.add_argument("--include-mc-certificates", action="store_true", help="Also certify an LWA_MC reference initialized with the analytic LWA weights")
    parser.add_argument("--skip-layer-eval", action="store_true", help="Skip the per-layer sigma diagnostic certification sweep")
    parser.add_argument("--cert-samples", type=int, default=1000, help="Number of test examples to certify")
    parser.add_argument("--norms", nargs="+", default=["1", "2", "inf"], help="Norms to certify")
    parser.add_argument("--batch-size", type=int, default=100, help="MNIST batch size")
    parser.add_argument("--data-dir", default=None, help="MNIST data directory")
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST if missing")
    parser.add_argument("--sv", action="store_true", help="Recompute singular-value regularizer cache")
    parser.add_argument("--lr", type=float, default=0.1, help="SGD learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PGD L2 epsilon")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--num-iter", type=int, default=40, help="PGD iterations")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Model checkpoint path")
    parser.add_argument("--mc-model-path", default=str(DEFAULT_MC_MODEL_PATH), help="MC reference model checkpoint path")
    parser.add_argument("--certificate-path", default=str(DEFAULT_CERTIFICATE_PATH), help="Saved certificate payload")
    parser.add_argument("--mc-certificate-path", default=str(DEFAULT_MC_CERTIFICATE_PATH), help="Saved MC reference certificate payload")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_dir()
    device = get_device()
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=not args.no_download,
    )

    model = LWRS(sigma=args.sigma, n_samples=args.samples).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
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

    mc_model = None
    mc_clean_acc = None
    mc_adv_acc = None
    if not args.skip_mc:
        mc_model = LWMC(sigma=args.sigma, n_samples=args.mc_samples).to(device)
        mc_optimizer = optim.SGD(mc_model.parameters(), lr=args.lr)
        train_or_load_model(
            mc_model,
            mc_optimizer,
            Path(args.mc_model_path),
            DEFAULT_MC_SV_PATH,
            args,
            device,
            "LWA MC Reference Model",
            train_loader,
            test_loader,
            legacy_path=None,
        )

    print("\n=== Evaluating Analytic LWA ReGU Model ===")
    clean_acc = evaluate_clean(model, test_loader, device=device)
    adv_acc = evaluate_l2(model, test_loader, args.epsilon, args.alpha, args.num_iter, device=device)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Adversarial Accuracy: {adv_acc:.4f}")

    if mc_model is not None:
        print("\n=== Evaluating LWA MC Reference Model ===")
        mc_clean_acc = evaluate_clean(mc_model, test_loader, device=device)
        mc_adv_acc = evaluate_l2(mc_model, test_loader, args.epsilon, args.alpha, args.num_iter, device=device)
        print(f"MC Clean Accuracy: {mc_clean_acc:.4f}")
        print(f"MC Adversarial Accuracy: {mc_adv_acc:.4f}")

    regu_mc_comparison = compare_regu_to_mc(
        model,
        test_loader,
        sigma=args.sigma,
        n_samples=args.verify_mc_samples,
        max_batches=args.verify_batches,
        seed=args.verification_seed,
        device=device,
    )

    sigma = list(args.sigma)
    if args.skip_layer_eval:
        print("\n=== Skipping per-layer sigma diagnostic sweep ===")
        layer_results = {}
    else:
        layer_results = {
            "layer_0": evaluate_sigma_configuration(
                model, test_loader, [sigma[0], 0.0, 0.0], f"Layer 0 ReGU - sigma {sigma[0]}", args.cert_samples, args.samples, device
            ),
            "layer_1": evaluate_sigma_configuration(
                model, test_loader, [0.0, sigma[1], 0.0], f"Layer 1 ReGU - sigma {sigma[1]}", args.cert_samples, args.samples, device
            ),
            "layer_2": evaluate_sigma_configuration(
                model, test_loader, [0.0, 0.0, sigma[2]], f"Layer 2 ReGU - sigma {sigma[2]}", args.cert_samples, args.samples, device
            ),
            "all_enabled": evaluate_sigma_configuration(
                model, test_loader, sigma, "All ReGU layers enabled", args.cert_samples, args.samples, device
            ),
        }

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

    mc_certificate_results = None
    if args.include_mc_certificates:
        mc_certificate_model_samples = args.mc_certificate_model_samples or args.verify_mc_samples
        torch.manual_seed(args.verification_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.verification_seed)
        mc_certificate_model = LWMC(sigma=args.sigma, n_samples=mc_certificate_model_samples).to(device)
        mc_certificate_model.load_state_dict(model.state_dict())
        mc_certificate_results = evaluate_certificate_suite(
            mc_certificate_model,
            test_loader,
            sigma=sigma,
            norms=args.norms,
            max_samples=args.cert_samples,
            n_samples=args.samples,
            output_path=args.mc_certificate_path,
            device=device,
            name="LWA_MC Certificate Suite",
            seed=args.verification_seed,
        )

    combined_path = CURRENT_DIR / "models" / f"lwa_results_sigma_{sigma}.pt"
    torch.save(
        {
            "layer_results": layer_results,
            "certificate_results": certificate_results,
            "mc_certificate_results": mc_certificate_results,
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "mc_clean_accuracy": mc_clean_acc,
            "mc_adversarial_accuracy": mc_adv_acc,
            "regu_mc_comparison": regu_mc_comparison,
        },
        combined_path,
    )
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
