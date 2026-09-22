#!/usr/bin/env python3
"""Plot corrected online-estimate versus full-data-actual Gittins results.

The selected configuration IDs are held fixed within each comparison.  The
estimated point is the finite-target posterior mean available to Gittins at
the latest recorded recommendation event not exceeding the requested search
budget.  The actual point is the full-data value of that same configuration.
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from plot_gittins_20seed_frequency_frontiers import (
    DATASET_LABELS,
    DATASETS,
    EXPECTED_SEEDS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    ROOT
    / "experiments/combined_objective/results/gauss_radau_8bench_20seed_independent"
)
DEFAULT_OUTPUT = (
    ROOT / "analysis/20seed_results/figures/gittins_estimated_vs_actual"
)
PAIR_NAME = "gauss_radau_accuracy_endpoint"


@dataclass(frozen=True)
class Snapshot:
    source_fraction: float
    arm_indices: np.ndarray
    estimated: np.ndarray
    actual: np.ndarray
    truth: np.ndarray
    truth_frontier: np.ndarray


def _result_path(run_root: Path, dataset: str, seed: int) -> Path:
    return run_root / f"seed-{seed}" / PAIR_NAME / dataset / "result.json"


def _snapshot(run_root: Path, dataset: str, seed: int, target: float) -> Snapshot:
    path = _result_path(run_root, dataset, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    run = json.loads(path.read_text(encoding="utf-8"))["run"]
    points = run["points"]
    fractions = [float(point["cost_fraction"]) for point in points]
    position = bisect.bisect_right(fractions, target + 1e-12) - 1
    if position < 0:
        raise ValueError(
            f"{dataset} seed {seed} has no checkpoint at or before {target:.1%}"
        )
    point = points[position]
    arms = np.asarray(point["selected_arm_indices"], dtype=int)
    estimated = np.asarray(point["finite_target_mean_vectors"], dtype=float).reshape((-1, 2))
    actual = np.asarray(point["offline_raw_selected_vectors"], dtype=float).reshape((-1, 2))
    if len(arms) != len(estimated) or len(arms) != len(actual):
        raise ValueError(f"unaligned estimate/actual arrays in {path}")
    truth = np.asarray(run["raw_truth_vectors"], dtype=float)
    frontier = np.asarray(run["full_data_pareto_arm_indices"], dtype=int)
    if len(arms) and not np.allclose(actual, truth[arms], rtol=1e-10, atol=1e-12):
        raise ValueError(f"offline actual values do not match truth vectors in {path}")
    return Snapshot(
        source_fraction=float(point["cost_fraction"]),
        arm_indices=arms,
        estimated=estimated,
        actual=actual,
        truth=truth,
        truth_frontier=frontier,
    )


def _format_axes(ax: mpl.axes.Axes, *, log_cost: bool) -> None:
    if log_cost:
        ax.set_xscale("log")
    ax.grid(color="#d9dde3", linewidth=0.55, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _errors(snapshot: Snapshot) -> tuple[float, float]:
    accuracy_mae_pp = float(
        100.0 * np.median(np.abs(snapshot.estimated[:, 0] - snapshot.actual[:, 0]))
    )
    positive = snapshot.actual[:, 1] > 0.0
    cost_mape = float(
        100.0
        * np.median(
            np.abs(snapshot.estimated[positive, 1] - snapshot.actual[positive, 1])
            / snapshot.actual[positive, 1]
        )
    )
    return accuracy_mae_pp, cost_mape


def plot_frontier_overlay(
    *, run_root: Path, output_dir: Path, target: float, seed: int
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.4))
    for ax, dataset in zip(axes.flat, DATASETS):
        snapshot = _snapshot(run_root, dataset, seed, target)
        truth = snapshot.truth
        frontier = snapshot.truth_frontier
        frontier = frontier[np.argsort(truth[frontier, 1])]
        ax.scatter(
            truth[:, 1], truth[:, 0], s=12, color="#cfd3d8", alpha=0.58,
            edgecolors="none", zorder=1,
        )
        ax.plot(
            truth[frontier, 1], truth[frontier, 0], color="#25282c",
            linewidth=1.5, zorder=2,
        )
        ax.scatter(
            truth[frontier, 1], truth[frontier, 0], s=34, facecolors="white",
            edgecolors="#25282c", linewidths=1.0, zorder=3,
        )
        for estimate, actual in zip(snapshot.estimated, snapshot.actual):
            ax.annotate(
                "", xy=(actual[1], actual[0]), xytext=(estimate[1], estimate[0]),
                arrowprops={
                    "arrowstyle": "->", "color": "#6688a8", "alpha": 0.72,
                    "linewidth": 1.0, "shrinkA": 3, "shrinkB": 3,
                },
                zorder=4,
            )
        ax.scatter(
            snapshot.actual[:, 1], snapshot.actual[:, 0], s=68, marker="D",
            color="#e07a22", edgecolors="#73431e", linewidths=0.8, zorder=5,
        )
        ax.scatter(
            snapshot.estimated[:, 1], snapshot.estimated[:, 0], s=82, marker="o",
            facecolors="none", edgecolors="#2468a2", linewidths=1.6, zorder=6,
        )
        accuracy_mae_pp, cost_mape = _errors(snapshot)
        ax.text(
            0.025, 0.025,
            f"n={len(snapshot.arm_indices)}  |  Acc. MAE={accuracy_mae_pp:.2f} pp"
            f"  |  Cost MdAPE={cost_mape:.1f}%",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8,
            color="#292d32",
            bbox={
                "boxstyle": "round,pad=0.22", "facecolor": "white",
                "edgecolor": "none", "alpha": 0.84,
            },
            zorder=8,
        )
        ax.set_title(
            f"{DATASET_LABELS[dataset]}\n"
            f"latest event at {snapshot.source_fraction:.1%}",
            fontsize=14, pad=8, fontweight="semibold",
        )
        _format_axes(ax, log_cost=bool(np.all(truth[:, 1] > 0.0)))

    fig.supxlabel("Mean deployment cost (USD per query, log scale)", fontsize=17, y=0.035)
    fig.supylabel("Mean accuracy", fontsize=17, x=0.025)
    fig.suptitle(
        "Gittins posterior estimates vs full-data actual values "
        f"(seed {seed}, {target:.0%} search budget)",
        fontsize=21, y=0.985,
    )
    handles = (
        Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor="none", markeredgecolor="#2468a2",
               markeredgewidth=1.6, label="Posterior mean estimate"),
        Line2D([], [], linestyle="none", marker="D", markersize=7,
               markerfacecolor="#e07a22", markeredgecolor="#73431e",
               label="Same configuration: full-data actual"),
        Line2D([], [], color="#6688a8", linewidth=1.2, marker=">",
               markevery=[1], label="Estimate → actual"),
        Line2D([], [], color="#25282c", linewidth=1.5, marker="o",
               markerfacecolor="white", label="True Pareto frontier"),
    )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.002),
        ncol=4, frameon=False, fontsize=11,
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.12, top=0.89, wspace=0.24, hspace=0.34,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"gittins_estimated_vs_actual_frontiers_seed{seed}_{round(100 * target):02d}pct"
    )
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def _parity_axis(
    ax: mpl.axes.Axes,
    estimated: np.ndarray,
    actual: np.ndarray,
    *,
    objective: int,
) -> None:
    x = estimated[:, objective]
    y = actual[:, objective]
    positive = (x > 0.0) & (y > 0.0) if objective == 1 else np.ones(len(x), dtype=bool)
    x = x[positive]
    y = y[positive]
    ax.scatter(
        x, y, s=20, color="#7251b5", alpha=0.38, edgecolors="none", zorder=2,
    )
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    if objective == 1:
        low *= 0.82
        high *= 1.22
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        padding = max(0.01, 0.06 * (high - low))
        low = max(0.0, low - padding)
        high = min(1.0, high + padding)
    ax.plot([low, high], [low, high], linestyle="--", color="#25282c",
            linewidth=1.1, zorder=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.grid(color="#d9dde3", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    if objective == 0:
        mae = 100.0 * float(np.median(np.abs(x - y)))
        metric = f"Median AE: {mae:.2f} pp"
    else:
        mdape = 100.0 * float(np.median(np.abs(x - y) / y))
        metric = f"Median APE: {mdape:.1f}%"
        compact = FuncFormatter(lambda value, _: f"{value:g}")
        ax.xaxis.set_major_formatter(compact)
        ax.yaxis.set_major_formatter(compact)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(
        0.04, 0.94, f"n={len(x)}\n{metric}", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )


def plot_parity(
    *, run_root: Path, output_dir: Path, target: float
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(4, 4, figsize=(15.5, 14.0))
    for dataset_index, dataset in enumerate(DATASETS):
        block = 0 if dataset_index < 4 else 2
        column = dataset_index % 4
        snapshots = [
            _snapshot(run_root, dataset, seed, target)
            for seed in sorted(EXPECTED_SEEDS)
        ]
        estimated = np.concatenate([snapshot.estimated for snapshot in snapshots], axis=0)
        actual = np.concatenate([snapshot.actual for snapshot in snapshots], axis=0)
        accuracy_ax = axes[block, column]
        cost_ax = axes[block + 1, column]
        _parity_axis(accuracy_ax, estimated, actual, objective=0)
        _parity_axis(cost_ax, estimated, actual, objective=1)
        accuracy_ax.set_title(DATASET_LABELS[dataset], fontsize=14, pad=8, fontweight="semibold")
        accuracy_ax.set_xlabel("Estimated accuracy", fontsize=10)
        accuracy_ax.set_ylabel("Actual accuracy", fontsize=10)
        cost_ax.set_xlabel("Estimated deployment cost", fontsize=10)
        cost_ax.set_ylabel("Actual deployment cost", fontsize=10)

    fig.suptitle(
        "Gittins estimate calibration across 20 seeds "
        f"({target:.0%} search budget; each point is one recommended configuration)",
        fontsize=20, y=0.992,
    )
    fig.legend(
        handles=(
            Line2D([], [], linestyle="none", marker="o", markersize=7,
                   markerfacecolor="#7251b5", markeredgecolor="none",
                   alpha=0.55, label="Recommended configuration"),
            Line2D([], [], linestyle="--", color="#25282c", linewidth=1.2,
                   label="Perfect calibration (estimated = actual)"),
        ),
        loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=2,
        frameon=False, fontsize=11,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.075, top=0.955,
        wspace=0.30, hspace=0.36,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"gittins_estimated_vs_actual_parity_20seed_{round(100 * target):02d}pct"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42, choices=sorted(EXPECTED_SEEDS))
    parser.add_argument("--targets", type=float, nargs="+", default=(0.10, 0.30))
    args = parser.parse_args()
    for target in args.targets:
        if not 0.0 < target <= 1.0:
            parser.error("targets must lie in (0, 1]")
        for path in plot_frontier_overlay(
            run_root=args.run_root,
            output_dir=args.output_dir,
            target=target,
            seed=args.seed,
        ):
            print(f"wrote {path}")
        for path in plot_parity(
            run_root=args.run_root,
            output_dir=args.output_dir,
            target=target,
        ):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
