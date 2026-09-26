#!/usr/bin/env python3
"""Plot 0--100% curves using the paper frontier's hybrid data sources."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
LATEST_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
LEGACY_FIGURE_ROOT = ROOT / "analysis/20seed_results/figures/eight_dataset_curves"
DEFAULT_OUTPUT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed"
    / "figures_preview/hv_frontier_source"
)

DATASETS = (
    "hotpotqa",
    "restaurant_valid",
    "bird_mini_dev",
    "bing_querylogs",
    "mathqa",
    "restaurant_test",
    "bird_dev",
    "stackoverflow",
)
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "restaurant_test": "Restaurant Test",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD Dev",
    "restaurant_valid": "Restaurant Valid",
    "bing_querylogs": "Bing Query Logs",
    "bird_mini_dev": "BIRD Mini Dev",
}
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METRICS = {
    "hv_regret": "Hypervolume Regret",
    "generational_distance": "Generational Distance (GD)",
    "inverted_generational_distance": "Inverted Generational Distance (IGD)",
}
LEGACY_QNEHVI_DATASETS = {
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
}
EXPECTED_SEEDS = tuple(range(42, 62))
GRID = np.linspace(0.0, 1.0, 201)


@dataclass
class MeanCurve:
    x: np.ndarray
    mean: np.ndarray
    two_se: np.ndarray
    count: np.ndarray


def load_styles() -> dict[str, dict[str, str | float]]:
    path = LEGACY_FIGURE_ROOT / "method_styles.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {method: payload[method] for method in METHODS}


def load_legacy_curves() -> dict[tuple[str, str, str], MeanCurve]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = (
        defaultdict(list)
    )
    path = LEGACY_FIGURE_ROOT / "curve_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"], row["metric"])
            grouped[key].append(
                (
                    float(row["cost_fraction"]),
                    float(row["mean"]),
                    float(row["two_se"]),
                    int(row["n_runs"]),
                )
            )
    output: dict[tuple[str, str, str], MeanCurve] = {}
    for key, rows in grouped.items():
        rows.sort()
        output[key] = MeanCurve(
            x=np.asarray([row[0] for row in rows]),
            mean=np.asarray([row[1] for row in rows]),
            two_se=np.asarray([row[2] for row in rows]),
            count=np.asarray([row[3] for row in rows], dtype=int),
        )
    return output


def load_numeric_trajectory(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split(",")
        positions = {name: index for index, name in enumerate(header)}
        needed = ("budget_fraction", *METRICS)
        last_needed = max(positions[name] for name in needed)
        for line in handle:
            fields = line.rstrip("\n").split(",", last_needed + 1)
            rows.append(tuple(float(fields[positions[name]]) for name in needed))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return values[:, 0], {
        metric: values[:, index + 1] for index, metric in enumerate(METRICS)
    }


def aggregate_latest(dataset: str, method: str) -> dict[str, MeanCurve]:
    runs = []
    for seed in EXPECTED_SEEDS:
        path = LATEST_ROOT / dataset / method / f"seed-{seed}" / "cost_trajectory.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing latest frontier-source trajectory: {path}")
        runs.append(load_numeric_trajectory(path))

    # A random-search recommendation is constant between saved events.  Use
    # the union of all run event locations so its mean is an exact step
    # function rather than a coarse interpolation on the display grid.
    grid = (
        np.unique(np.concatenate([xs for xs, _ in runs]))
        if method in {"random_configurations", "random_questions"}
        else GRID
    )
    aligned = {
        metric: np.full((len(EXPECTED_SEEDS), len(grid)), np.nan, dtype=np.float64)
        for metric in METRICS
    }
    for row_index, (xs, metric_values) in enumerate(runs):
        positions = np.searchsorted(xs, grid, side="right") - 1
        available = (positions >= 0) & (grid <= xs[-1] + 1e-12)
        for metric in METRICS:
            aligned[metric][row_index, available] = metric_values[metric][
                positions[available]
            ]

    output: dict[str, MeanCurve] = {}
    for metric, values in aligned.items():
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.full(len(grid), np.nan)
        two_se = np.full(len(grid), np.nan)
        valid = count > 0
        mean[valid] = np.nanmean(values[:, valid], axis=0)
        for column in np.flatnonzero(valid):
            finite = values[:, column][np.isfinite(values[:, column])]
            two_se[column] = (
                2.0 * np.std(finite, ddof=1) / math.sqrt(len(finite))
                if len(finite) > 1
                else 0.0
            )
        output[metric] = MeanCurve(grid.copy(), mean, two_se, count)
    return output


def build_hybrid_curves() -> tuple[
    dict[tuple[str, str, str], MeanCurve], dict[str, dict[str, str]]
]:
    legacy = load_legacy_curves()
    curves: dict[tuple[str, str, str], MeanCurve] = {}
    sources: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        sources[dataset] = {}
        for method in METHODS:
            use_legacy = method == "radial_gittins" or (
                method == "qnehvi" and dataset in LEGACY_QNEHVI_DATASETS
            )
            if use_legacy:
                sources[dataset][method] = "legacy_0_100_curve_summary"
                for metric in METRICS:
                    curves[(dataset, method, metric)] = legacy[(dataset, method, metric)]
                continue
            sources[dataset][method] = "latest_usd_checkpoint_run_trajectory"
            aggregated = aggregate_latest(dataset, method)
            for metric, curve in aggregated.items():
                curves[(dataset, method, metric)] = curve
    return curves, sources


def write_summary(path: Path, curves: dict[tuple[str, str, str], MeanCurve]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "method",
                "metric",
                "cost_fraction",
                "mean",
                "two_se",
                "n_runs",
            ),
        )
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                for metric in METRICS:
                    curve = curves[(dataset, method, metric)]
                    for x, mean, two_se, count in zip(
                        curve.x, curve.mean, curve.two_se, curve.count
                    ):
                        if np.isfinite(mean) and np.isfinite(two_se) and count > 0:
                            writer.writerow(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "metric": metric,
                                    "cost_fraction": x,
                                    "mean": mean,
                                    "two_se": two_se,
                                    "n_runs": int(count),
                                }
                            )


def plot_metric(
    metric: str,
    curves: dict[tuple[str, str, str], MeanCurve],
    styles: dict[str, dict[str, str | float]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    for axis, dataset in zip(axes.flat, DATASETS):
        for method in METHODS:
            curve = curves[(dataset, method, metric)]
            shown = np.isfinite(curve.mean) & np.isfinite(curve.two_se)
            style = styles[method]
            is_random = method in {"random_configurations", "random_questions"}
            axis.plot(
                curve.x[shown],
                curve.mean[shown],
                color=str(style["color"]),
                linewidth=float(style["linewidth"]),
                linestyle=str(style["linestyle"]),
                label=str(style["label"]),
                zorder=4 if method == "radial_gittins" else 3,
                drawstyle="steps-post" if is_random else "default",
            )
            axis.fill_between(
                curve.x[shown],
                np.maximum(0.0, curve.mean[shown] - curve.two_se[shown]),
                curve.mean[shown] + curve.two_se[shown],
                color=str(style["color"]),
                alpha=0.10,
                linewidth=0,
                zorder=1,
                step="post" if is_random else None,
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=23, pad=10)
        axis.set_xlim(-0.015, 1.015)
        y_top = axis.get_ylim()[1]
        axis.set_ylim(-0.02 * y_top, y_top)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.xaxis.set_major_formatter(percent)
        labels = axis.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.tick_params(axis="both", labelsize=23, width=1.1, length=6)
        axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
        axis.set_axisbelow(True)

    subplot_bottom = 0.265
    subplot_top = 0.945
    figure.text(
        0.5,
        0.168,
        "Percentage of Exhaustive Evaluation Cost",
        ha="center",
        va="center",
        fontsize=28,
    )
    figure.text(
        0.023,
        (subplot_bottom + subplot_top) / 2.0,
        METRICS[metric],
        ha="center",
        va="center",
        rotation=90,
        fontsize=28,
    )
    handles = [
        Line2D(
            [],
            [],
            color=str(styles[method]["color"]),
            linewidth=float(styles[method]["linewidth"]),
            linestyle=str(styles[method]["linestyle"]),
            label=str(styles[method]["label"]),
        )
        for method in METHODS
    ]
    handles.append(
        Patch(facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE")
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.001),
        ncol=4,
        frameon=False,
        fontsize=22,
        handlelength=3.15,
        columnspacing=1.55,
        handletextpad=0.75,
        labelspacing=0.70,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=subplot_top,
        bottom=subplot_bottom,
        wspace=0.34,
        hspace=0.34,
    )
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    styles = load_styles()
    for method in METHODS:
        styles[method]["linewidth"] = 2.4
    styles["radial_gittins"]["linewidth"] = 2.4 * 1.2
    curves, sources = build_hybrid_curves()
    write_summary(DEFAULT_OUTPUT / "curve_summary.csv", curves)
    (DEFAULT_OUTPUT / "method_styles.json").write_text(
        json.dumps(styles, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "source_plot_branch": "plot_paper",
        "source_plot_commit": "d4b08a5138675f352cd9bcc2417b7e1d11dca280",
        "source_policy": sources,
        "panel_order": list(DATASETS),
        "panel_columns_by_role_count": [2, 3, 4, 5],
        "random_aggregation": (
            "For each seed with a recommendation available at x, use its latest "
            "saved recommendation at or before x; then compute mean and 2 SE at "
            "the union of all per-seed event x values. Render as steps-post. The "
            "n_runs column records the contributing count, which can be below 20 "
            "before every seed has produced its first recommendation."
        ),
        "line_widths": {
            "other_methods": 2.4,
            "radial_gittins": 2.88,
            "radial_multiplier": 1.2,
        },
        "expected_seeds": list(EXPECTED_SEEDS),
        "legacy_qnehvi_seed_counts": {
            "restaurant_valid": 20,
            "bing_querylogs": 18,
            "bird_mini_dev": 19,
        },
        "metric_space": "reciprocal-cost baseline evaluation space",
        "uncertainty": "mean +/- 2 standard errors",
    }
    (DEFAULT_OUTPUT / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for metric in METRICS:
        plot_metric(metric, curves, styles, DEFAULT_OUTPUT / f"eight_datasets_{metric}")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
