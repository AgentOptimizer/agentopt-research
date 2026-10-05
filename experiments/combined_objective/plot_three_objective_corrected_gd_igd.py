#!/usr/bin/env python3
"""Recompute and render corrected three-objective GD/IGD paper curves.

The plot uses the same method colors, serif typography, line weights, grid,
and mean +/- 2 standard-error convention as the ablation branch.  It reads one
already validated saved run at a time and never changes experiment results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    / "gittins_g2_main_figures/corrected_gd_igd_complete_returned_set_20260929"
    / "three_objective"
)
SOURCE = (
    ROOT.parent
    / "agentopt-research-three-objective/experiments/combined_objective/results"
)
SEEDS = tuple(range(42, 62))
DATASETS = ("hotpotqa", "mathqa")
DATASET_LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
GRID = np.arange(1, 61, dtype=np.float64) / 200.0
CURVE_LINEWIDTH = 2.4
HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

# Match the method subset, colors, and ordering used by the earlier paper
# figures.  The axes-only CC-Gittins variant and EGE-SR are intentionally
# excluded from both the panels and the exported plotting table.
METHODS = (
    "cc_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
STYLES: dict[str, dict[str, str | float]] = {
    "cc_gittins": {
        "color": "tab:orange", "label": "CC-Gittins",
        "linewidth": HIGHLIGHT_LINEWIDTH, "linestyle": "-",
    },
    "ege_sh": {
        "color": "tab:green", "label": "EGE-SH",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "ape_k": {
        "color": "tab:purple", "label": "APE-k",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "qnehvi": {
        "color": "tab:pink", "label": "qNEHVI",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "random_configurations": {
        "color": "tab:brown", "label": "Random configurations",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "random_questions": {
        "color": "tab:cyan", "label": "Random questions",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
}
METRICS = (
    ("hv_regret", "HV Regret", 1.0),
    ("generational_distance", "GD", 1.0),
    ("inverted_generational_distance", "IGD", 1.0),
)
NATURAL_STOPS = {
    "direction_eta_numerical_floor",
    "all_arms_completed",
    "all_cells_observed",
    "question_budget",
    "all_arms_exhausted",
}


def nondominated_indices(points: np.ndarray) -> np.ndarray:
    """Indices not dominated when all three scoring coordinates are maximized."""
    array = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)


def front_distance(obtained: np.ndarray, reference: np.ndarray) -> float:
    """Mean nearest-point distance without filtering either supplied set."""
    if not len(obtained):
        return math.inf if len(reference) else 0.0
    if not len(reference):
        return math.inf
    delta = obtained[:, None, :] - reference[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1).mean())


def recompute_corrected_distances(run: dict[str, Any]) -> None:
    """Recompute GD/IGD from selections while preserving every returned arm."""
    raw = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
    positive_latency = raw[:, 1][raw[:, 1] > 0.0]
    positive_cost = raw[:, 2][raw[:, 2] > 0.0]
    latency_reference = float(np.median(positive_latency)) if len(positive_latency) else 1.0
    cost_reference = float(np.median(positive_cost)) if len(positive_cost) else 1.0
    metric_truth = np.column_stack((
        raw[:, 0],
        latency_reference / (latency_reference + raw[:, 1]),
        cost_reference / (cost_reference + raw[:, 2]),
    ))
    truth_front = metric_truth[nondominated_indices(metric_truth)]
    for point in run["points"]:
        selected = np.asarray(point["selected_arm_indices"], dtype=int)
        obtained = metric_truth[selected]
        gd = front_distance(obtained, truth_front)
        igd = front_distance(truth_front, obtained)
        point["generational_distance"] = None if not math.isfinite(gd) else gd
        point["inverted_generational_distance"] = None if not math.isfinite(igd) else igd


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def latest_value(run: dict[str, Any], target: float, field: str) -> float:
    terminal = float(run["cost_fraction"])
    if terminal + 1e-12 < target and run["stop_reason"] not in NATURAL_STOPS:
        return math.nan
    eligible = [
        point for point in run["points"]
        if float(point["cost_fraction"]) <= target + 1e-12
    ]
    if not eligible:
        return math.nan
    value = eligible[-1].get(field)
    return math.nan if value is None else float(value)


def run_path(source: Path, dataset: str, method: str, seed: int) -> Path:
    if method == "cc_gittins":
        return source / "three_objective_20seed" / f"seed_{seed}" / dataset / "result.json"
    if method == "cc_gittins_axes":
        return source / "three_objective_axes_20seed" / f"seed_{seed}" / dataset / "result.json"
    if method in {"ege_sh", "ege_sr", "ape_k", "qnehvi"}:
        return (
            source / "three_objective_pareto_baselines" / f"seed_{seed}"
            / dataset / method / "result.json"
        )
    return (
        source / "three_objective_random_20seed" / f"seed_{seed}"
        / dataset / method / "result.json"
    )


def summarize_aligned(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(aligned), axis=0)
    means = np.full(len(GRID), np.nan, dtype=np.float64)
    two_se = np.full(len(GRID), np.nan, dtype=np.float64)
    for column, count in enumerate(counts):
        values = aligned[:, column]
        values = values[np.isfinite(values)]
        if count:
            means[column] = float(np.mean(values))
            two_se[column] = (
                2.0 * float(np.std(values, ddof=1)) / math.sqrt(int(count))
                if count > 1 else 0.0
            )
    return means, two_se, counts


def aggregate_method(
    source: Path, dataset: str, method: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    expected_selector = {
        "cc_gittins": "three_objective_cc_gittins",
        "cc_gittins_axes": "cc_gittins_axes",
    }.get(method, method)
    rows = {field: [] for field, _, _ in METRICS}
    for seed in SEEDS:
        path = run_path(source, dataset, method, seed)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        run = payload["run"]
        saved_seed = int(payload["config"]["seed"])
        if saved_seed != seed or run["selector"] != expected_selector:
            raise ValueError(f"run identity mismatch: {path}")
        recompute_corrected_distances(run)
        for field, _, scale in METRICS:
            rows[field].append([
                latest_value(run, float(target), field) * scale for target in GRID
            ])
        del payload, run
    return {
        field: summarize_aligned(np.asarray(values, dtype=np.float64))
        for field, values in rows.items()
    }


def build_summary(source: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            print(f"Aggregating {dataset} / {method}", flush=True)
            aggregated = aggregate_method(source, dataset, method)
            for field, label, scale in METRICS:
                means, two_se, counts = aggregated[field]
                for x, mean, spread, count in zip(GRID, means, two_se, counts):
                    rows.append({
                        "dataset": dataset,
                        "method": method,
                        "metric": field,
                        "metric_label": label,
                        "cost_fraction": float(x),
                        "mean": None if not np.isfinite(mean) else float(mean),
                        "two_se": None if not np.isfinite(spread) else float(spread),
                        "n_runs": int(count),
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def load_summary(path: Path) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["mean"] or not row["two_se"]:
                continue
            key = (row["dataset"], row["method"], row["metric"])
            grouped.setdefault(key, []).append((
                float(row["cost_fraction"]),
                float(row["mean"]),
                float(row["two_se"]),
                int(row["n_runs"]),
            ))
    output = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        output[key] = (array[:, 0], array[:, 1], array[:, 2], array[:, 3])
    return output


def plot(summary_path: Path, output_stem: Path) -> None:
    curves = load_summary(summary_path)
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    compact = FuncFormatter(lambda value, _: f"{value:g}")

    for row, dataset in enumerate(DATASETS):
        for column, (field, title, _) in enumerate(METRICS):
            axis = axes[row, column]
            for method in METHODS:
                xs, means, two_se, _ = curves[(dataset, method, field)]
                style = STYLES[method]
                is_random = method in {"random_configurations", "random_questions"}
                axis.plot(
                    xs,
                    means,
                    color=str(style["color"]),
                    linewidth=float(style["linewidth"]),
                    linestyle=str(style["linestyle"]),
                    zorder=5 if method == "cc_gittins" else 3,
                    drawstyle="steps-post" if is_random else "default",
                )
                axis.fill_between(
                    xs,
                    np.maximum(0.0, means - two_se),
                    means + two_se,
                    color=str(style["color"]),
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                    step="post" if is_random else None,
                )
            if row == 0:
                axis.set_title(title, fontsize=28, pad=13)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis="both", labelsize=26, width=1.0, length=5)
            axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)

    subplot_bottom = 0.32
    subplot_top = 0.925
    figure.text(
        0.535, 0.225, "Percentage of Exhaustive Evaluation Cost",
        ha="center", va="center", fontsize=30,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, top=subplot_top, bottom=subplot_bottom,
        wspace=0.30, hspace=0.34,
    )
    row_centers = [
        (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0
        for row in range(2)
    ]
    figure.text(
        0.020, row_centers[0], DATASET_LABELS[DATASETS[0]],
        ha="center", va="center", rotation=90, fontsize=31,
    )
    figure.text(
        0.020, row_centers[1], DATASET_LABELS[DATASETS[1]],
        ha="center", va="center", rotation=90, fontsize=31,
    )
    handles = [
        Line2D(
            [], [], color=str(STYLES[method]["color"]),
            linewidth=float(STYLES[method]["linewidth"]),
            linestyle=str(STYLES[method]["linestyle"]),
            label=str(STYLES[method]["label"]),
        )
        for method in METHODS
    ]
    handles.append(Patch(
        facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE",
    ))
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=28,
        handlelength=2.7,
        columnspacing=1.15,
        handletextpad=0.65,
        labelspacing=0.65,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.12}
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, **save_options)
    figure.savefig(output_stem.with_suffix(".pdf"), **save_options)
    figure.savefig(output_stem.with_suffix(".svg"), **save_options)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--summary", type=Path, default=HERE / "corrected_curve_summary.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "three_objective_hv_gd_igd_corrected_complete_returned_set",
    )
    parser.add_argument("--reuse-summary", action="store_true")
    args = parser.parse_args()
    configure_matplotlib()
    if not args.reuse_summary or not args.summary.is_file():
        build_summary(args.source, args.summary)
    plot(args.summary, args.output)
    metadata = {
        "source": str(args.source),
        "seeds": list(SEEDS),
        "cost_range": [0.005, 0.30],
        "grid_step": 0.005,
        "uncertainty": "mean +/- 2 sample standard errors over available matched seeds",
        "hv_metric": "absolute hypervolume regret (not relative and not percentage)",
        "distance_metric": (
            "GD/IGD recomputed from every returned configuration, including "
            "configurations dominated in true evaluation space"
        ),
        "styles": STYLES,
        "template": (
            "experiments/combined_objective/plot_frontier_source_hv_curves.py "
            "and its method_styles.json"
        ),
    }
    (args.output.parent / f"{args.output.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
