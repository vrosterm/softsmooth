import argparse
import math
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from LWA_certification import (  # noqa: E402
    _as_sigma_list,
    _dual_norm_ord,
    _norm_label,
    _norm_ord,
    _output_layer,
    _regu_layers,
    _row_noise_std,
    _row_norm_factor,
    _torch_norm_ord,
    _temporary_sigma,
    _vector_norm,
    affine_deviation_bound,
    affine_surrogate_parameters,
)
from LWA_model import LWRS  # noqa: E402
from LWA_utils import DEFAULT_MODEL_PATH, get_device, get_mnist_loaders  # noqa: E402


def _normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_pdf(x):
    return torch.exp(-0.5 * x.pow(2)) / math.sqrt(2.0 * math.pi)


def _stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"n": 0}
    values = sorted(values)

    def quantile(frac):
        idx = min(len(values) - 1, max(0, int(frac * (len(values) - 1))))
        return values[idx]

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "max": values[-1],
    }


def _fmt_stats(stats):
    if stats.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={stats['n']} mean={stats['mean']:.6g} min={stats['min']:.6g} "
        f"p10={stats['p10']:.6g} p50={stats['p50']:.6g} "
        f"p90={stats['p90']:.6g} max={stats['max']:.6g}"
    )


def current_formula_layer_contributions(model, X, sigma, p=2, eps=1e-12):
    """Mirror affine_deviation_bound, but keep each layer's contribution."""
    sigma = _as_sigma_list(sigma)
    p = _norm_ord(p)
    regu_layers = _regu_layers(model)
    output = _output_layer(model)
    x_aff = model.flatten(X)
    all_contribs = []

    for sample_idx in range(x_aff.size(0)):
        activation = x_aff[sample_idx]
        contribs = []

        for layer_idx, layer in enumerate(regu_layers):
            z_aff = layer(activation)
            std = _row_noise_std(layer, sigma[layer_idx]).clamp_min(eps)
            delta_l = z_aff.pow(2) / (2 * math.sqrt(2 * math.pi) * std)
            delta_norm = _vector_norm(delta_l, p)

            propagation = activation.new_tensor(1.0)
            for later_layer in regu_layers[layer_idx + 1:]:
                propagation = propagation * _row_norm_factor(later_layer, p)
            propagation = propagation * _row_norm_factor(output, p)

            contribs.append(float((propagation * delta_norm).item()))
            activation = 0.5 * z_aff + _row_noise_std(layer, sigma[layer_idx]) / math.sqrt(2 * math.pi)

        all_contribs.append(contribs)

    return all_contribs


def finite_sigma_slope_stats(model, X, sigma, eps=1e-12):
    """Stats for the implemented analytic ReGU derivative Phi(z/std)."""
    sigma = _as_sigma_list(sigma)
    activation = model.flatten(X)
    layer_stats = []

    for layer_idx, layer in enumerate(_regu_layers(model)):
        z = layer(activation)
        std = _row_noise_std(layer, sigma[layer_idx]).to(device=z.device, dtype=z.dtype).clamp_min(eps)
        std = std.unsqueeze(0)
        slope = _normal_cdf(z / std)
        layer_stats.append(_stats(slope.detach().cpu().reshape(-1).tolist()))
        activation = z * slope + std * _normal_pdf(z / std)

    return layer_stats


def analyze_sample(model, X, y, sigma, p):
    p = _norm_ord(p)
    q = _dual_norm_ord(p)
    label_name = _norm_label(p)

    with _temporary_sigma(model, sigma), torch.no_grad():
        model_scores = model(X)[0]

    W_aff, b_aff = affine_surrogate_parameters(model, sigma)
    scores_aff = (model.flatten(X) @ W_aff.T + b_aff)[0]
    model_label = int(torch.argmax(model_scores).item())
    affine_label = int(torch.argmax(scores_aff).item())
    true_label = int(y.item())

    e_p = float(affine_deviation_bound(model, X, sigma, p=p)[0].item())
    contribs = current_formula_layer_contributions(model, X, sigma, p=p)[0]

    target_label = model_label
    margins = []
    adjusted = []
    denoms = []
    for class_idx in range(scores_aff.numel()):
        if class_idx == target_label:
            continue
        margin = scores_aff[target_label] - scores_aff[class_idx]
        denom = _vector_norm(W_aff[target_label] - W_aff[class_idx], q).clamp_min(1e-12)
        margins.append(float(margin.item()))
        adjusted.append(float((margin - 2 * e_p).item()))
        denoms.append(float(denom.item()))

    min_margin = min(margins)
    min_adjusted = min(adjusted)
    min_denom = min(denoms)
    current_radius = max(0.0, min(adj / den for adj, den in zip(adjusted, denoms)))
    unadjusted_radius = max(0.0, min(m / den for m, den in zip(margins, denoms)))

    return {
        "norm": label_name,
        "true_label": true_label,
        "model_label": model_label,
        "affine_label": affine_label,
        "model_correct": model_label == true_label,
        "affine_correct": affine_label == true_label,
        "model_affine_agree": model_label == affine_label,
        "e_p": e_p,
        "min_affine_margin_for_model_label": min_margin,
        "min_adjusted_margin": min_adjusted,
        "min_denom": min_denom,
        "two_e_over_min_margin": (2 * e_p / min_margin) if min_margin > 0 else float("inf"),
        "current_deviation_radius": current_radius,
        "unadjusted_affine_radius_same_label": unadjusted_radius,
        "layer_contribs": contribs,
    }


def print_summary(rows, slope_rows):
    norms = []
    for row in rows:
        if row["norm"] not in norms:
            norms.append(row["norm"])

    print("\n=== Deviation Diagnostic Summary ===")
    print(f"Samples analyzed: {len(rows) // max(len(norms), 1)}")

    for norm in norms:
        norm_rows = [row for row in rows if row["norm"] == norm]
        print(f"\nNorm {norm}")
        print(f"  model correct:       {sum(row['model_correct'] for row in norm_rows)}/{len(norm_rows)}")
        print(f"  affine correct:      {sum(row['affine_correct'] for row in norm_rows)}/{len(norm_rows)}")
        print(f"  model/affine agree:  {sum(row['model_affine_agree'] for row in norm_rows)}/{len(norm_rows)}")
        print(f"  e_p:                 {_fmt_stats(_stats(row['e_p'] for row in norm_rows))}")
        print(f"  min affine margin:   {_fmt_stats(_stats(row['min_affine_margin_for_model_label'] for row in norm_rows))}")
        print(f"  min adjusted margin: {_fmt_stats(_stats(row['min_adjusted_margin'] for row in norm_rows))}")
        print(f"  2e_p / margin:       {_fmt_stats(_stats(row['two_e_over_min_margin'] for row in norm_rows))}")
        print(f"  dev radius:          {_fmt_stats(_stats(row['current_deviation_radius'] for row in norm_rows))}")
        print(f"  affine radius same label: {_fmt_stats(_stats(row['unadjusted_affine_radius_same_label'] for row in norm_rows))}")

        for layer_idx in range(3):
            print(
                f"  layer {layer_idx} contrib: "
                f"{_fmt_stats(_stats(row['layer_contribs'][layer_idx] for row in norm_rows))}"
            )

    print("\n=== Implemented Finite-Sigma ReGU Slope Stats: Phi(z/std) ===")
    for layer_idx in range(3):
        values = []
        for sample_stats in slope_rows:
            # Expand approximately by using summary means across batches. This is enough
            # to reveal whether slopes are near the high-sigma 1/2 tangent.
            if sample_stats[layer_idx].get("n", 0):
                values.append(sample_stats[layer_idx]["mean"])
        print(f"Layer {layer_idx} mean-slope across batches: {_fmt_stats(_stats(values))}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose current affine-deviation certificate collapse")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Analytic LWA model checkpoint")
    parser.add_argument("--sigma", nargs=3, type=float, default=[0.5, 0.1, 0.1], help="Layerwise ReGU sigmas")
    parser.add_argument("--norms", nargs="+", default=["1", "2", "inf"], help="Norms to diagnose")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples to inspect")
    parser.add_argument("--batch-size", type=int, default=100, help="MNIST batch size")
    parser.add_argument("--data-dir", default=None, help="MNIST data directory")
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST if missing")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    _, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=not args.no_download,
    )

    model = LWRS(sigma=args.sigma).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    rows = []
    slope_rows = []
    total = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        slope_rows.append(finite_sigma_slope_stats(model, x_batch, args.sigma))

        for idx in range(x_batch.size(0)):
            if total >= args.samples:
                break
            X = x_batch[idx].unsqueeze(0)
            y = y_batch[idx]
            for norm in args.norms:
                rows.append(analyze_sample(model, X, y, args.sigma, norm))
            total += 1

        if total >= args.samples:
            break

    print_summary(rows, slope_rows)


if __name__ == "__main__":
    main()
