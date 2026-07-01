import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
LAYERWISE_AVG_DIR = PROJECT_ROOT / "LAYERWISE_AVG"
OUTPUT_AVG_DIR = PROJECT_ROOT / "OUTPUT_AVG"
for path in (SCRIPT_DIR, PROJECT_ROOT, LAYERWISE_AVG_DIR, OUTPUT_AVG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from LAYERWISE_AVG.LWA_utils import DEFAULT_CERTIFICATE_PATH, DEFAULT_PLOT_PATH  # noqa: E402


DEFAULT_INPUT = DEFAULT_CERTIFICATE_PATH
DEFAULT_OUTPUT_AVG_INPUT = OUTPUT_AVG_DIR / "models" / "output_avg_standard_rs_width784_certificate_results.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "figs" / "results" / DEFAULT_PLOT_PATH.name
CERTIFICATE_ORDER = ["lwa_k_p", "lwa_k_abs", "affine", "output_avg"]
OUTPUT_AVG_FAMILY = "output_avg"
STANDARD_RS_NORMS = ["1", "2", "inf"]
STANDARD_RS_INPUT_DIM = 784
DEFAULT_LABELS = {
    "lwa_k_p": r"LWA ($K_p$)",
    "lwa_k_abs": r"LWA ($K_{\mathrm{abs}}$)",
    # "affine_deviation": "Affine dev.",
    "affine": "Affine Surrogate",
    "output_avg": "Standard RS",
}
PGF_LABELS = {
    "lwa_k_p": r"LWA ($K_p$)",
    "lwa_k_abs": r"LWA ($K_{\textup{abs}}$)",
    "affine": "Affine Surrogate",
    "output_avg": "Standard RS",
}
COLORS = {
    "lwa_k_p": "#3b82f6",
    "lwa_k_abs": "#14b8a6",
    # "affine_deviation": "#f59e0b",
    "affine": "#111827",
    "output_avg": "#dc2626",
}
LINE_STYLES = {
    "lwa_k_p": "-",
    "lwa_k_abs": "-",
    # "affine_deviation": "-.",
    "affine": "-",
    "output_avg": "--",
}
NORM_LABELS = {
    "1": r"$\ell_1$-Radius",
    "2": r"$\ell_2$-Radius",
    "inf": r"$\ell_{\infty}$-Radius",
}
NORM_AXIS_LIMITS = {
    "1": 5.0,
    "2": 3.0,
    "inf": 0.2,
}
NORM_X_TICKS = {
    "1": [0, 1, 2, 3, 4, 5],
    "2": [0, 1, 2, 3],
    "inf": [0, 0.05, 0.10, 0.15, 0.20],
}
Y_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
    if result_key is not None:
        if result_key in payload:
            payload = payload[result_key]
        elif "samples" not in payload:
            raise ValueError(f"{path} does not contain {result_key}")
    elif "certificate_results" in payload:
        payload = payload["certificate_results"]
    elif "output_avg_results" in payload:
        payload = payload["output_avg_results"]
    if payload is None or "samples" not in payload:
        raise ValueError(f"{path} does not contain per-sample certificate results")
    return payload


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _standard_rs_radii_from_l2(radius_l2, input_dim):
    radius_l2 = float(radius_l2)
    return {
        "1": radius_l2,
        "2": radius_l2,
        "inf": radius_l2 / math.sqrt(float(input_dim)),
    }


def _augment_output_avg_norm_conversions(payload):
    input_dim = payload.get("input_dim") or STANDARD_RS_INPUT_DIM
    norms = list(payload.get("norms", []))

    for sample in payload["samples"]:
        radii = sample.setdefault("radii", {})
        l2_result = radii.get("2", {})
        radius_l2 = _finite_or_none(l2_result.get(OUTPUT_AVG_FAMILY))
        if radius_l2 is None:
            continue

        for norm, radius in _standard_rs_radii_from_l2(radius_l2, input_dim).items():
            radii.setdefault(norm, {}).setdefault(OUTPUT_AVG_FAMILY, radius)
            if norm not in norms:
                norms.append(norm)

    payload["norms"] = [norm for norm in STANDARD_RS_NORMS if norm in norms]
    payload["input_dim"] = input_dim
    return payload


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
        # Failed or incorrect certificates are stored as 0.0 by MAIN/main.py.
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


def _pgf_output_path(output_path):
    return Path(output_path).with_suffix(".pgf")


def _pgfplots_line_style(line_style):
    return {
        "-": "solid",
        "--": "dashed",
        "-.": "dash dot",
        ":": "dotted",
    }.get(line_style, "solid")


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[idx : idx + 2], 16) for idx in (0, 2, 4))


def _escape_pgf_text(text):
    text = str(text)
    if "$" in text:
        return text
    return text.replace("_", r"\_")


def _format_tick(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _pgf_tick_list(ticks):
    return "{" + ",".join(_format_tick(tick) for tick in ticks) + "}"


def _norm_axis_limit(norm):
    return NORM_AXIS_LIMITS.get(str(norm))


def _norm_x_ticks(norm):
    return NORM_X_TICKS.get(str(norm))


def _include_family_for_norm(family, norm):
    return not (family == "lwa_k_abs" and str(norm) == "2")


def _write_pgfplots(path, series, norm, max_radius):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axis_label = _norm_axis_label(norm)

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"  width={FIG_WIDTH_IN}in,",
        f"  height={FIG_HEIGHT_IN}in,",
        "  xmin=0,",
        f"  xmax={max_radius:.12g},",
        "  ymin=0,",
        "  ymax=1,",
        f"  xlabel={{{axis_label}}},",
        r"  ylabel={Certified Accuracy},",
        r"  legend pos=north east,",
        r"  legend cell align=left,",
        r"  legend style={font=\scriptsize, fill=white, fill opacity=0.95, draw=black},",
        r"  tick align=outside,",
        f"  ytick={_pgf_tick_list(Y_TICKS)},",
    ]
    x_ticks = _norm_x_ticks(norm)
    if x_ticks is not None:
        lines.append(f"  xtick={_pgf_tick_list(x_ticks)},")
    lines.append(r"]")

    for idx, item in enumerate(series):
        color_name = f"certcolor{idx}"
        red, green, blue = _hex_to_rgb(item["color"])
        lines.append(f"\\definecolor{{{color_name}}}{{RGB}}{{{red},{green},{blue}}}")
        lines.append(
            "\\addplot+["
            f"no markers, color={color_name}, {_pgfplots_line_style(item['line_style'])}, "
            f"line width={PLOT_LINEWIDTH}pt"
            "] coordinates {"
        )
        lines.extend(f"({x:.12g},{y:.12g})" for x, y in zip(item["x"], item["y"]))
        lines.append("};")
        lines.append(f"\\addlegendentry{{{_escape_pgf_text(item['pgf_label'])}}}")

    lines.extend([r"\end{axis}", r"\end{tikzpicture}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _line_label(family):
    return DEFAULT_LABELS[family]


def _pgf_line_label(family):
    return PGF_LABELS[family]


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
    for norm in norms:
        norm_max_radius = max_radius
        if norm_max_radius is None:
            norm_max_radius = _norm_axis_limit(norm)
        if norm_max_radius is None:
            norm_max_radius = _max_positive_radius(named_payloads, [norm], families) * 1.1
        norm_max_radius = max(float(norm_max_radius), 1e-12)
        plot_radii = _linspace(0.0, norm_max_radius, points)

        fig = plt.figure(dpi=72, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        ax = plt.gca()
        plotted_any = False
        pgf_series = []

        for _, payload in named_payloads:
            samples = payload["samples"]
            for family in families:
                if not _include_family_for_norm(family, norm):
                    continue

                radii = _sample_radii(samples, norm, family)
                if not radii:
                    continue

                line_style = LINE_STYLES[family]
                label = _line_label(family)
                accuracies = _certified_accuracies(radii, plot_radii)
                ax.plot(
                    plot_radii,
                    accuracies,
                    color=COLORS[family],
                    linestyle=line_style,
                    linewidth=PLOT_LINEWIDTH,
                    label=label,
                )
                pgf_series.append(
                    {
                        "x": plot_radii,
                        "y": accuracies,
                        "color": COLORS[family],
                        "line_style": line_style,
                        "label": label,
                        "pgf_label": _pgf_line_label(family),
                    }
                )
                plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No supported certificates", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlim(0, norm_max_radius)
        ax.set_ylim(0, 1)
        x_ticks = _norm_x_ticks(norm)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        ax.set_yticks(Y_TICKS)
        ax.set_xlabel(_norm_axis_label(norm))
        ax.set_ylabel("Certified Accuracy")
        legend = ax.legend(
            loc="upper right",
            handlelength=2.0,
            handletextpad=0.6,
            labelspacing=0.4,
            borderpad=0.3,
            framealpha=0.95,
        )
        if legend is not None:
            legend._legend_box.align = "left"
            for text in legend.get_texts():
                text.set_ha("left")

        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)
        norm_output_path = _norm_output_path(output_path, norm)
        norm_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(norm_output_path, dpi=300, bbox_inches="tight")
        pgf_output_path = _pgf_output_path(norm_output_path)
        _write_pgfplots(pgf_output_path, pgf_series, norm, norm_max_radius)
        plt.close(fig)
        output_paths.append(norm_output_path)
        output_paths.append(pgf_output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Plot LWA and OUTPUT_AVG certified-accuracy waterfall curves")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="LWA certificate result .pt file from MAIN/main.py")
    parser.add_argument("--output-avg-input", default="auto", help="Optional OUTPUT_AVG result .pt file to overlay; use 'none' to disable auto-overlay")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path")
    parser.add_argument("--families", nargs="+", choices=CERTIFICATE_ORDER, default=CERTIFICATE_ORDER, help="Certificate families to plot")
    parser.add_argument("--points", type=int, default=1000, help="Number of radius thresholds to plot")
    parser.add_argument("--max-radius", type=float, default=None, help="Maximum input perturbation radius on the x-axis")
    args = parser.parse_args()

    named_payloads = [("LWA", _load_payload(args.input, result_key="certificate_results"))]
    output_avg_arg = args.output_avg_input.lower()
    output_avg_input = None if output_avg_arg == "none" else args.output_avg_input
    if output_avg_arg == "auto":
        output_avg_input = str(DEFAULT_OUTPUT_AVG_INPUT) if DEFAULT_OUTPUT_AVG_INPUT.exists() else None
    if output_avg_input is not None:
        output_avg_payload = _augment_output_avg_norm_conversions(
            _load_payload(output_avg_input, result_key="output_avg_results")
        )
        named_payloads.append(("OUTPUT_AVG", output_avg_payload))

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
