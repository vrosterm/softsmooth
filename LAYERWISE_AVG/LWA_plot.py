import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = CURRENT_DIR / "models" / "lwa_certificate_results.pt"
DEFAULT_MC_INPUT = CURRENT_DIR / "models" / "lwa_mc_certificate_results.pt"
DEFAULT_OUTPUT = CURRENT_DIR / "models" / "lwa_certificate_waterfall.png"
CERTIFICATE_ORDER = ["lwa_k_p", "lwa_k_abs", "affine_deviation", "affine"]
DEFAULT_LABELS = {
    "lwa_k_p": "LWA K_p",
    "lwa_k_abs": "LWA K_abs",
    "affine_deviation": "Affine dev.",
    "affine": "Affine",
}
COLORS = {
    "lwa_k_p": "#3b82f6",
    "lwa_k_abs": "#14b8a6",
    "affine_deviation": "#f59e0b",
    "affine": "#111827",
}
LINE_STYLES = {
    "lwa_k_p": "-",
    "lwa_k_abs": "--",
    "affine_deviation": "-.",
    "affine": ":",
}
MODEL_LINE_STYLES = ["-", "--", "-.", ":"]
NORM_LABELS = {
    "1": r"$\ell_1$-Radius",
    "2": r"$\ell_2$-Radius",
    "inf": r"$\ell_{\infty}$-Radius",
}
FIG_WIDTH_IN = 2.925
FIG_HEIGHT_IN = 2.486
PLOT_LINEWIDTH = 1.5

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 6.3,
    "lines.linewidth": PLOT_LINEWIDTH,
})


def _load_payload(path, result_key=None):
    payload = torch.load(path, map_location="cpu")
    if result_key is not None and result_key in payload:
        payload = payload[result_key]
    elif "certificate_results" in payload:
        payload = payload["certificate_results"]
    elif "mc_certificate_results" in payload:
        payload = payload["mc_certificate_results"]
    if payload is None or "samples" not in payload:
        raise ValueError(f"{path} does not contain per-sample LWA certificate results")
    return payload


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _linspace(start, stop, count):
    if count <= 1:
        return [float(stop)]
    step = (stop - start) / (count - 1)
    return [start + step * idx for idx in range(count)]


def _sample_radii(samples, norm, family):
    radii = []
    has_supported_values = False

    for sample in samples:
        norm_results = sample.get("radii", {}).get(norm, {})
        value = _finite_or_none(norm_results.get(family))
        if value is None:
            radii.append(-1.0)
            continue

        has_supported_values = True
        # Failed or incorrect certificates are stored as 0.0 by LWA_main.py.
        radii.append(value if value > 0 else -1.0)

    return radii if has_supported_values else []


def _certified_accuracies(radii, plot_radii):
    if not radii:
        return []

    total = len(radii)
    return [sum(radius >= threshold for radius in radii) / total for threshold in plot_radii]


def _payload_norms(payload):
    samples = payload["samples"]
    norms = list(payload.get("norms", []))
    if not norms and samples:
        norms = list(samples[0].get("radii", {}).keys())
    return norms


def _max_positive_radius(named_payloads, norms, families):
    max_radius = 0.0
    for _, payload in named_payloads:
        samples = payload["samples"]
        for norm in norms:
            for family in families:
                for radius in _sample_radii(samples, norm, family):
                    if radius > max_radius:
                        max_radius = radius
    return max_radius


def _norm_file_label(norm):
    return f"p{str(norm).replace('.', '_')}"


def _norm_axis_label(norm):
    return NORM_LABELS.get(str(norm), f"p={norm} Radius")


def _norm_output_path(output_path, norm):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_{_norm_file_label(norm)}{output_path.suffix}")


def _line_label(model_name, family, family_count, model_count):
    if model_count == 1:
        return DEFAULT_LABELS[family]
    if family_count == 1:
        return model_name
    return f"{model_name} {DEFAULT_LABELS[family]}"


def plot_waterfall(named_payloads, output_path, points=1000, max_radius=None, families=None):
    families = list(families or CERTIFICATE_ORDER)
    norms = []
    for _, payload in named_payloads:
        for norm in _payload_norms(payload):
            if norm not in norms:
                norms.append(norm)

    if not norms:
        raise ValueError("No norms found in the certificate payload")

    output_paths = []
    model_count = len(named_payloads)
    family_count = len(families)

    for norm in norms:
        norm_max_radius = max_radius
        if norm_max_radius is None:
            norm_max_radius = _max_positive_radius(named_payloads, [norm], families) * 1.1
        norm_max_radius = max(float(norm_max_radius), 1e-12)
        plot_radii = _linspace(0.0, norm_max_radius, points)

        fig = plt.figure(dpi=72, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        ax = plt.gca()
        plotted_any = False

        for model_idx, (model_name, payload) in enumerate(named_payloads):
            samples = payload["samples"]
            for family in families:
                radii = _sample_radii(samples, norm, family)
                if not radii:
                    continue

                line_style = MODEL_LINE_STYLES[model_idx % len(MODEL_LINE_STYLES)] if model_count > 1 else LINE_STYLES[family]
                ax.plot(
                    plot_radii,
                    _certified_accuracies(radii, plot_radii),
                    color=COLORS[family],
                    linestyle=line_style,
                    linewidth=PLOT_LINEWIDTH,
                    label=_line_label(model_name, family, family_count, model_count),
                )
                plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No supported certificates", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlim(0, norm_max_radius)
        ax.set_ylim(0, 1)
        ax.set_xlabel(_norm_axis_label(norm))
        ax.set_ylabel("Certified Accuracy")
        ax.legend(
            loc="upper right",
            handlelength=2.0,
            handletextpad=0.6,
            labelspacing=0.4,
            borderpad=0.3,
            framealpha=0.95,
        )

        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)
        norm_output_path = _norm_output_path(output_path, norm)
        norm_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(norm_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(norm_output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Plot LWA ReGU certified-accuracy waterfall curves")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Certificate result .pt file from LWA_main.py")
    parser.add_argument("--mc-input", default="auto", help="Optional LWA_MC certificate result .pt file to overlay; use 'none' to disable auto-overlay")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path")
    parser.add_argument("--families", nargs="+", choices=CERTIFICATE_ORDER, default=CERTIFICATE_ORDER, help="Certificate families to plot")
    parser.add_argument("--points", type=int, default=1000, help="Number of radius thresholds to plot")
    parser.add_argument("--max-radius", type=float, default=None, help="Maximum input perturbation radius on the x-axis")
    args = parser.parse_args()

    named_payloads = [("LWA", _load_payload(args.input, result_key="certificate_results"))]
    mc_input_arg = args.mc_input.lower()
    mc_input = None if mc_input_arg == "none" else args.mc_input
    if mc_input_arg == "auto":
        mc_input = str(DEFAULT_MC_INPUT) if DEFAULT_MC_INPUT.exists() else None
    if mc_input is not None:
        named_payloads.append(("LWA_MC", _load_payload(mc_input, result_key="mc_certificate_results")))

    output_paths = plot_waterfall(
        named_payloads,
        args.output,
        points=args.points,
        max_radius=args.max_radius,
        families=args.families,
    )
    for output_path in output_paths:
        print(f"Saved waterfall plot to {output_path}")


if __name__ == "__main__":
    main()
