#!/usr/bin/env python3
"""Draw old-style separated estimated/actual Gittins appendix figures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from plot_gittins_20seed_frequency_frontiers import _frequency_colormap
from plot_gittins_estimated_vs_actual import (
    DATASET_LABELS,
    DATASETS,
    DEFAULT_OUTPUT,
    DEFAULT_RUN_ROOT,
    EXPECTED_SEEDS,
    Snapshot,
    _snapshot,
)


def _draw_reference(ax: mpl.axes.Axes, snapshot: Snapshot) -> None:
    truth = snapshot.truth
    frontier = snapshot.truth_frontier
    frontier = frontier[np.argsort(truth[frontier, 1])]
    ax.scatter(
        truth[:, 1], truth[:, 0], s=24, color="#c7ccd3", alpha=0.78,
        edgecolors="none", zorder=1,
    )
    ax.plot(
        truth[frontier, 1], truth[frontier, 0], color="#30343a",
        linewidth=1.8, marker="o", markersize=5.5, markerfacecolor="white",
        markeredgewidth=1.1, zorder=3,
    )
    if np.all(truth[:, 1] > 0.0):
        ax.set_xscale("log")
    ax.grid(color="#d9dde3", linewidth=0.65, alpha=0.62)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _shared_limits(snapshot: Snapshot) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate((snapshot.truth[:, 1], snapshot.estimated[:, 1]))
    y = np.concatenate((snapshot.truth[:, 0], snapshot.estimated[:, 0]))
    if np.all(x > 0.0):
        low, high = np.log10(x.min()), np.log10(x.max())
        padding = max(0.08, 0.06 * (high - low))
        x_limits = (10 ** (low - padding), 10 ** (high + padding))
    else:
        padding = max(1e-9, 0.05 * (x.max() - x.min()))
        x_limits = (float(x.min() - padding), float(x.max() + padding))
    padding = max(0.01, 0.06 * (y.max() - y.min()))
    y_limits = (max(0.0, float(y.min() - padding)), min(1.02, float(y.max() + padding)))
    return x_limits, y_limits


def _draw_page(
    *,
    run_root: Path,
    output_dir: Path,
    datasets: tuple[str, ...],
    page_number: int,
    target: float,
    seed: int,
) -> tuple[Path, Path]:
    cmap = _frequency_colormap()
    norm = mpl.colors.Normalize(vmin=1, vmax=len(EXPECTED_SEEDS))
    fig, axes = plt.subplots(3, 4, figsize=(19.2, 13.0), squeeze=False)

    for column, dataset in enumerate(datasets):
        snapshot = _snapshot(run_root, dataset, seed, target)
        all_seed_snapshots = [
            _snapshot(run_root, dataset, other_seed, target)
            for other_seed in sorted(EXPECTED_SEEDS)
        ]
        counts: Counter[int] = Counter()
        for item in all_seed_snapshots:
            counts.update(int(index) for index in item.arm_indices)
        frequency_indices = np.asarray(sorted(counts), dtype=int)
        frequencies = np.asarray(
            [counts[int(index)] for index in frequency_indices], dtype=float
        )

        for row in range(3):
            _draw_reference(axes[row, column], snapshot)

        axes[0, column].scatter(
            snapshot.estimated[:, 1], snapshot.estimated[:, 0], s=125,
            color="#d55e00", edgecolors="#6f4228", linewidths=1.0, zorder=5,
        )
        axes[1, column].scatter(
            snapshot.actual[:, 1], snapshot.actual[:, 0], s=125,
            color="#d55e00", edgecolors="#6f4228", linewidths=1.0, zorder=5,
        )
        axes[2, column].scatter(
            snapshot.truth[frequency_indices, 1],
            snapshot.truth[frequency_indices, 0],
            c=frequencies, cmap=cmap, norm=norm, s=125,
            edgecolors="#6f4228", linewidths=1.0, zorder=5,
        )

        axes[0, column].set_title(
            DATASET_LABELS[dataset], fontsize=20, pad=10, fontweight="semibold",
        )
        panel_labels = (
            f"{len(snapshot.arm_indices)} recommended",
            f"same {len(snapshot.arm_indices)} configurations",
            f"{len(frequency_indices)} unique across 20 seeds",
        )
        for row, label in enumerate(panel_labels):
            axes[row, column].text(
                0.96, 0.05, label, transform=axes[row, column].transAxes,
                ha="right", va="bottom", fontsize=12, color="#292d32",
                bbox={
                    "boxstyle": "round,pad=0.20", "facecolor": "white",
                    "edgecolor": "none", "alpha": 0.86,
                },
                zorder=7,
            )

        x_limits, y_limits = _shared_limits(snapshot)
        for row in range(3):
            axes[row, column].set_xlim(*x_limits)
            axes[row, column].set_ylim(*y_limits)

    fig.subplots_adjust(
        left=0.090, right=0.895, bottom=0.105, top=0.865,
        wspace=0.24, hspace=0.29,
    )
    row_labels = (
        ("Estimated", f"Seed {seed}"),
        ("Actual", f"Seed {seed}"),
        ("Actual", "Seeds 42–61"),
    )
    for row, (value_label, seed_label) in enumerate(row_labels):
        box = axes[row, 0].get_position()
        center = (box.y0 + box.y1) / 2
        fig.text(
            0.024, center, value_label, rotation=90, ha="center", va="center",
            fontsize=20, fontweight="semibold",
        )
        fig.text(
            0.043, center, seed_label, rotation=90, ha="center", va="center",
            fontsize=15, color="#4e5359",
        )

    fig.supxlabel(
        "Mean deployment cost (USD per query, log scale)",
        fontsize=20, x=0.49, y=0.038,
    )
    fig.supylabel("Mean accuracy", fontsize=20, x=0.062, y=0.50)
    fig.suptitle(
        "Gauss–Radau Gittins: estimated and actual Pareto recommendations "
        f"({target:.0%} search budget)",
        fontsize=25, y=0.982,
    )
    fig.legend(
        handles=(
            Line2D([], [], linestyle="none", marker="o", markersize=12,
                   markerfacecolor="#d55e00", markeredgecolor="#6f4228",
                   label=f"Recommended configuration (seed {seed})"),
            Line2D([], [], linestyle="none", marker="o", markersize=10,
                   markerfacecolor="#c7ccd3", markeredgecolor="none",
                   label="All configurations (full-data reference)"),
            Line2D([], [], color="#30343a", linewidth=1.8, marker="o",
                   markersize=7, markerfacecolor="white",
                   label="Full-data Pareto frontier"),
        ),
        loc="upper center", bbox_to_anchor=(0.49, 0.928), ncol=3,
        frameon=False, fontsize=14, columnspacing=2.2,
    )
    colorbar_ax = fig.add_axes([0.922, 0.16, 0.015, 0.60])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax,
    )
    colorbar.set_label(
        "Recommendation frequency (out of 20 seeds)", fontsize=15, labelpad=11,
    )
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"gittins_estimated_vs_actual_separated_seed{seed}_"
        f"{round(100 * target):02d}pct_page{page_number}"
    )
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
    pages = (DATASETS[:4], DATASETS[4:])
    for target in args.targets:
        if not 0.0 < target <= 1.0:
            parser.error("targets must lie in (0, 1]")
        for page_number, datasets in enumerate(pages, start=1):
            png, pdf = _draw_page(
                run_root=args.run_root,
                output_dir=args.output_dir,
                datasets=datasets,
                page_number=page_number,
                target=target,
                seed=args.seed,
            )
            print(f"wrote {png}")
            print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
