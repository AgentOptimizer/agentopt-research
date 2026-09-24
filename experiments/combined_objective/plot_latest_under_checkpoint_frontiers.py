#!/usr/bin/env python3
"""Plot the latest-at-or-below USD checkpoint recommendation frontiers.

The figures intentionally read the fresh schema-v3 replay directory directly.
Radial Gittins is read from its membership-change trajectory because those runs
already contain the complete recommendation intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
GITTINS_ROOT = ROOT / "analysis/20seed_results/radial_gittins"
LEGACY_QNEHVI_ROOT = ROOT / "analysis/final_run_baselines"
OUTPUT_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/figures_preview/frontier"
)
DATASETS = (
    "hotpotqa",
    "mathqa",
    "restaurant_test",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
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
    "ege_sr",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
PAPER_COMPARISON_METHODS = (
    "radial_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METHOD_LABELS = {
    "radial_gittins": "Radial Gittins",
    "ege_sh": "EGE-SH",
    "ege_sr": "EGE-SR",
    "ape_k": "APE-k",
    "qnehvi": "qNEHVI",
    "random_configurations": "Random configurations",
    "random_questions": "Random questions",
}
METHOD_COLORS = {
    "radial_gittins": "tab:orange",
    "ege_sh": "tab:green",
    "ege_sr": "tab:blue",
    "ape_k": "tab:purple",
    "qnehvi": "tab:pink",
    "random_configurations": "tab:brown",
    "random_questions": "tab:cyan",
}
SEEDS = tuple(range(42, 62))
LEGACY_QNEHVI_DATASETS = frozenset(
    {"bird_mini_dev", "restaurant_valid", "bing_querylogs"}
)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
    }
)


@dataclass(frozen=True)
class Landscape:
    truth: np.ndarray
    frontier: np.ndarray
    full_search_cost: float


@dataclass(frozen=True)
class FrontierSummary:
    counts: Counter[int]
    available: int
    mean_start_usd: float
    mean_end_usd: float
    mean_start_fraction: float
    mean_end_fraction: float


def _indices(value: str | list[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(";") if item.strip())
    return tuple(int(item) for item in value)


def _load_landscape(dataset: str) -> Landscape:
    path = GITTINS_ROOT / dataset / "seed-42/result.json"
    run = json.loads(path.read_text(encoding="utf-8"))["run"]
    truth = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
    return Landscape(
        truth=truth,
        frontier=np.asarray(run["full_data_pareto_arm_indices"], dtype=int),
        full_search_cost=float(run["bruteforce_search_cost_usd"]),
    )


def _baseline_interval(
    method: str, dataset: str, seed: int, target: float
) -> tuple[tuple[int, ...], float, float] | None:
    directory = RUN_ROOT / dataset / method / f"seed-{seed}"
    summary_path = directory / "summary.json"
    trajectory_path = directory / "cost_trajectory.csv"
    if not summary_path.is_file() or not trajectory_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("params", {}).get("recommendation_checkpoint_schema_version", 0)) < 3:
        raise ValueError(f"old checkpoint schema: {summary_path}")
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    event = f"cost_checkpoint_{int(round(100 * target))}pct"
    positions = [index for index, row in enumerate(rows) if row["event"] == event]
    if len(positions) != 1:
        raise ValueError(f"expected one {event} in {trajectory_path}, got {len(positions)}")
    position = positions[0]
    checkpoint = rows[position]
    selected = _indices(checkpoint["selected_arm_indices"])
    fraction = float(checkpoint["budget_fraction"])
    if fraction > target + 1e-12:
        raise ValueError(f"over-budget {event} in {trajectory_path}: {fraction}")

    membership_events = {
        "initial_empty",
        "recommendation_initial",
        "recommendation_changed",
    }
    starts = [row for row in rows[: position + 1] if row["event"] in membership_events]
    if not starts and not selected:
        # The retained checkpoint can itself be the zero-cost empty snapshot;
        # its original ``initial_empty`` row is intentionally not duplicated.
        start = checkpoint
    elif starts:
        start = starts[-1]
    else:
        raise ValueError(f"missing recommendation start in {trajectory_path}")
    if _indices(start["selected_arm_indices"]) != selected:
        raise ValueError(f"checkpoint membership mismatch in {trajectory_path}")
    end = next(
        (
            row
            for row in rows[position + 1 :]
            if row["event"] in membership_events
            and _indices(row["selected_arm_indices"]) != selected
        ),
        None,
    )
    if end is None:
        terminals = [row for row in rows if row["event"] == "terminal"]
        if len(terminals) != 1:
            raise ValueError(f"missing unique terminal in {trajectory_path}")
        end = terminals[0]
    return (
        selected,
        float(start["cumulative_search_cost_usd"]),
        float(end["cumulative_search_cost_usd"]),
    )


def _gittins_interval(
    dataset: str, seed: int, target: float
) -> tuple[tuple[int, ...], float, float] | None:
    path = GITTINS_ROOT / dataset / f"seed-{seed}" / "result.json"
    if not path.is_file():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))["run"]
    points = list(run["points"])
    eligible = [point for point in points if float(point["cost_fraction"]) <= target + 1e-12]
    if not eligible:
        return None
    start = eligible[-1]
    selected = _indices(start["selected_arm_indices"])
    end = next(
        (
            point
            for point in points[len(eligible) :]
            if _indices(point["selected_arm_indices"]) != selected
        ),
        points[-1],
    )
    return selected, float(start["cost_usd"]), float(end["cost_usd"])


def _legacy_qnehvi_interval(
    dataset: str, seed: int, target: float
) -> tuple[tuple[int, ...], float, float] | None:
    """Return the latest saved legacy qNEHVI state at or below ``target``.

    Legacy trajectories did not retain every recommendation membership change,
    so the returned endpoints delimit the contiguous run of matching *saved*
    states.  ``recommendation_changed`` rows from the oldest schema are omitted:
    those rows describe a completed-arm deployable front, whereas the qNEHVI
    checkpoint rows describe the empirical Pareto recommendation.
    """
    path = (
        LEGACY_QNEHVI_ROOT
        / dataset
        / "qnehvi"
        / f"seed-{seed}"
        / "cost_trajectory.csv"
    )
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["event"] != "recommendation_changed"
        ]
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if float(row["budget_fraction"]) <= target + 1e-12
    ]
    if not eligible:
        return None
    position, checkpoint = eligible[-1]
    selected = _indices(checkpoint["selected_arm_indices"])

    start_position = position
    while (
        start_position > 0
        and _indices(rows[start_position - 1]["selected_arm_indices"]) == selected
    ):
        start_position -= 1
    end = next(
        (
            row
            for row in rows[position + 1 :]
            if _indices(row["selected_arm_indices"]) != selected
        ),
        rows[-1],
    )
    return (
        selected,
        float(rows[start_position]["cumulative_search_cost_usd"]),
        float(end["cumulative_search_cost_usd"]),
    )


def _summary(
    method: str,
    dataset: str,
    target: float,
    landscape: Landscape,
    *,
    legacy_qnehvi: bool = False,
) -> FrontierSummary:
    counts: Counter[int] = Counter()
    starts: list[float] = []
    ends: list[float] = []
    for seed in SEEDS:
        if legacy_qnehvi:
            if method != "qnehvi":
                raise ValueError("legacy_qnehvi is only valid for qNEHVI")
            interval = _legacy_qnehvi_interval(dataset, seed, target)
        elif method == "radial_gittins":
            interval = _gittins_interval(dataset, seed, target)
        else:
            interval = _baseline_interval(method, dataset, seed, target)
        if interval is None:
            continue
        selected, start, end = interval
        if any(index < 0 or index >= len(landscape.truth) for index in selected):
            raise ValueError(f"invalid arm index in {method}/{dataset}/seed-{seed}")
        counts.update(selected)
        starts.append(start)
        ends.append(end)
    available = len(starts)
    if not available:
        return FrontierSummary(counts, 0, math.nan, math.nan, math.nan, math.nan)
    mean_start = float(np.mean(starts))
    mean_end = float(np.mean(ends))
    return FrontierSummary(
        counts,
        available,
        mean_start,
        mean_end,
        mean_start / landscape.full_search_cost,
        mean_end / landscape.full_search_cost,
    )


def _mix(color: tuple[float, float, float], other: tuple[float, float, float], amount: float):
    return tuple((1.0 - amount) * value + amount * target for value, target in zip(color, other))


def _colormap(method: str) -> LinearSegmentedColormap:
    base = to_rgb(METHOD_COLORS[method])
    return LinearSegmentedColormap.from_list(
        f"{method}_frequency",
        (_mix(base, (1, 1, 1), 0.82), _mix(base, (1, 1, 1), 0.45), base, _mix(base, (0, 0, 0), 0.38)),
        N=256,
    )


def _usd(value: float) -> str:
    if value < 1.0:
        return f"${value:.2f}"
    if value < 100.0:
        return f"${value:.1f}"
    return f"${value:,.0f}"


def _draw(
    ax: mpl.axes.Axes,
    landscape: Landscape,
    summary: FrontierSummary,
    title: str,
    cmap: mpl.colors.Colormap,
    *,
    show_sample_count: bool = True,
) -> None:
    truth = landscape.truth
    frontier = landscape.frontier[np.argsort(truth[landscape.frontier, 1])]
    ax.scatter(truth[:, 1], truth[:, 0], s=48, color="#c0c5cc", alpha=0.64, edgecolors="none", zorder=1)
    ax.plot(truth[frontier, 1], truth[frontier, 0], color="#25282c", linewidth=1.7, zorder=2)
    ax.scatter(truth[frontier, 1], truth[frontier, 0], s=72, facecolors="white", edgecolors="#25282c", linewidths=1.25, zorder=3)
    shown = np.asarray(sorted(summary.counts), dtype=int)
    if len(shown):
        frequencies = np.asarray([summary.counts[int(index)] for index in shown])
        ax.scatter(
            truth[shown, 1], truth[shown, 0], c=frequencies, cmap=cmap,
            norm=mpl.colors.Normalize(vmin=1, vmax=20), s=175,
            edgecolors="#6f4a22", linewidths=1.1, zorder=4,
        )
    shown_title = (
        title
        if summary.available == 20 or not show_sample_count
        else f"{title}  [n={summary.available}/20]"
    )
    ax.text(0.5, 1.22, shown_title, transform=ax.transAxes, ha="center", va="bottom", fontsize=27)
    if summary.available:
        start_usd = _usd(summary.mean_start_usd).replace("$", r"\$")
        end_usd = _usd(summary.mean_end_usd).replace("$", r"\$")
        subtitle = (
            f"{start_usd}–{end_usd} "
            f"({summary.mean_start_fraction:.1%}–{summary.mean_end_fraction:.1%})"
        )
    else:
        subtitle = "pending"
    ax.text(0.5, 1.17, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=26)
    if np.all(truth[:, 1] > 0):
        ax.set_xscale("log")
    ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=23)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _finish(
    fig: mpl.figure.Figure,
    axes: np.ndarray,
    *,
    suptitle: str,
    cmap: mpl.colors.Colormap,
    stem: Path,
) -> tuple[Path, Path]:
    fig.supxlabel("Mean deployment cost (USD per query, log scale)", fontsize=30, y=0.115)
    fig.supylabel("Mean accuracy", fontsize=30, x=0.022)
    fig.suptitle(suptitle, fontsize=34, y=0.955)
    fig.legend(
        handles=(
            Line2D([], [], linestyle="none", marker="o", markersize=13, markerfacecolor="#c0c5cc", markeredgecolor="none", label="All configurations"),
            Line2D([], [], color="#25282c", linewidth=2, marker="o", markersize=10, markerfacecolor="white", markeredgewidth=1.25, label="Full-data Pareto frontier"),
            Line2D([], [], linestyle="none", marker="o", markersize=15, markerfacecolor=cmap(0.6), markeredgecolor="#6f4a22", label="Recommendations (color = frequency)"),
        ),
        loc="lower center", bbox_to_anchor=(0.48, 0.035), ncol=3,
        frameon=False, fontsize=27, columnspacing=2.2, handletextpad=0.65,
    )
    fig.subplots_adjust(left=0.080, right=0.865, bottom=0.205, top=0.775, wspace=0.26, hspace=0.58)
    colorbar_ax = fig.add_axes([0.900, 0.245, 0.021, 0.52])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, 20), cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)", fontsize=28, labelpad=19)
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=24)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_method(method: str, target: float, output: Path) -> tuple[Path, Path]:
    cmap = _colormap(method)
    fig, axes = plt.subplots(2, 4, figsize=(24, 14.5))
    for ax, dataset in zip(axes.flat, DATASETS):
        landscape = _load_landscape(dataset)
        summary = _summary(method, dataset, target, landscape)
        _draw(ax, landscape, summary, DATASET_LABELS[dataset], cmap)
    stem = output / "by_method" / method / f"{method}_20seed_frequency_frontiers_{int(target * 100)}pct"
    return _finish(
        fig,
        axes,
        suptitle=(
            f"{METHOD_LABELS[method]} recommendations at {target:.0%} "
            "of brute-force search cost"
        ),
        cmap=cmap,
        stem=stem,
    )


def plot_hybrid_qnehvi(target: float, output: Path) -> tuple[Path, Path]:
    """Plot exact qNEHVI runs plus legacy fallback data for three datasets."""
    method = "qnehvi"
    cmap = _colormap(method)
    fig, axes = plt.subplots(2, 4, figsize=(24, 14.5))
    for ax, dataset in zip(axes.flat, DATASETS):
        legacy = dataset in LEGACY_QNEHVI_DATASETS
        landscape = _load_landscape(dataset)
        summary = _summary(
            method,
            dataset,
            target,
            landscape,
            legacy_qnehvi=legacy,
        )
        _draw(
            ax,
            landscape,
            summary,
            DATASET_LABELS[dataset],
            cmap,
            show_sample_count=False,
        )
    stem = (
        output
        / "by_method"
        / method
        / "old&new"
        / f"{method}_20seed_frequency_frontiers_{int(target * 100)}pct"
    )
    return _finish(
        fig,
        axes,
        suptitle=(
            f"{METHOD_LABELS[method]} recommendations at {target:.0%} "
            "of brute-force search cost"
        ),
        cmap=cmap,
        stem=stem,
    )


def plot_dataset_comparison(dataset: str, target: float, output: Path) -> tuple[Path, Path]:
    landscape = _load_landscape(dataset)
    cmap = LinearSegmentedColormap.from_list("comparison", ("#fff3d8", "#f6bf73", "#ff7f0e", "#8a4b08"), N=256)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14.5), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, PAPER_COMPARISON_METHODS):
        summary = _summary(method, dataset, target, landscape)
        _draw(ax, landscape, summary, METHOD_LABELS[method], cmap)
    stem = output / "by_dataset" / f"{dataset}_2x3_method_frontiers_{int(target * 100)}pct"
    return _finish(
        fig,
        axes,
        suptitle=(
            f"{DATASET_LABELS[dataset]}: 20-seed recommendation frontiers at "
            f"{target:.0%} of brute-force search cost"
        ),
        cmap=cmap,
        stem=stem,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=("stackoverflow",))
    parser.add_argument(
        "--hybrid-qnehvi",
        action="store_true",
        help=(
            "plot qNEHVI with exact runs for five datasets and legacy latest-"
            "saved checkpoints for bird_mini_dev, restaurant_valid, and "
            "bing_querylogs"
        ),
    )
    args = parser.parse_args()
    if args.hybrid_qnehvi:
        for target in (0.10, 0.30):
            for path in plot_hybrid_qnehvi(target, args.output_dir):
                print(f"wrote {path}", flush=True)
        return
    for method in args.methods:
        for target in (0.10, 0.30):
            for path in plot_method(method, target, args.output_dir):
                print(f"wrote {path}", flush=True)
    for dataset in args.datasets:
        for target in (0.10, 0.30):
            for path in plot_dataset_comparison(dataset, target, args.output_dir):
                print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
