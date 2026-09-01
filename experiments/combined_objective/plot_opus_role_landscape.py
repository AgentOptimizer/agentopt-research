#!/usr/bin/env python3
"""Highlight configurations containing Claude Opus by pipeline role."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
    pareto_min_cost_indices,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


PICKLES = {
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
    "gpqa": ROOT / "experiments/data/lookup/gpqa_lookup.pkl",
    "bfcl": ROOT / "experiments/data/lookup/bfcl_lookup.pkl",
}
LABELS = {
    "hotpotqa": "HotpotQA", "mathqa": "MathQA", "gpqa": "GPQA", "bfcl": "BFCL",
}
ROLE_LABELS = {
    "hotpotqa": ("Opus as planner", "Opus as solver"),
    "mathqa": ("Opus as answer model", "Opus as critic"),
}
FIRST_COLOR = "tab:blue"
SECOND_COLOR = "tab:red"
AGENT_COLOR = "tab:purple"
BACKGROUND_COLOR = "#c4cad2"
POINT_EDGE_COLOR = "#725b46"


def _opus_roles(configuration: str) -> tuple[bool, bool]:
    """Return whether Opus appears before and after the role separator."""
    first, second = configuration.split(" + ", maxsplit=1)
    return "opus" in first.lower(), "opus" in second.lower()


def _draw_bicolor_point(ax: plt.Axes, x: float, y: float,
                        point_scale: float) -> None:
    ax.plot(
        x, y, linestyle="none", marker="o", markersize=np.sqrt(110),
        markerfacecolor=FIRST_COLOR, markerfacecoloralt=SECOND_COLOR,
        fillstyle="left", markeredgecolor=POINT_EDGE_COLOR, markeredgewidth=0.85,
        zorder=6,
    )


def _plot_dataset(ax: plt.Axes, benchmark: str, point_scale: float = 1.0) -> None:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    truth = mean_raw_vectors(
        models, common_question_ids(models, datapoints, table), table,
    )
    if benchmark in ("gpqa", "bfcl"):
        opus = np.asarray(["opus" in model.lower() for model in models])
        ax.scatter(
            truth[~opus, 1], truth[~opus, 0], s=26, color=BACKGROUND_COLOR,
            alpha=0.85, edgecolors="none", zorder=1,
        )
        frontier_indices = np.asarray(pareto_min_cost_indices(truth), dtype=int)
        frontier_indices = frontier_indices[np.argsort(truth[frontier_indices, 1])]
        ax.plot(
            truth[frontier_indices, 1], truth[frontier_indices, 0],
            color="#3f4854", linewidth=1.7, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor="#3f4854",
            markeredgewidth=1.2, zorder=3,
        )
        ax.scatter(
            truth[opus, 1], truth[opus, 0], s=90 * point_scale, color=AGENT_COLOR,
            edgecolors="#343a40", linewidths=0.7, zorder=6,
        )
        # BFCL's Opus point is over 30x costlier than its upper-right Pareto
        # point. Keep the full range so that fact remains visible.
        ax.margins(x=0.06)
        ax.set_title(LABELS[benchmark], fontsize=22, pad=10)
        ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=16)
        ax.set_axisbelow(True)
        return

    roles = [_opus_roles(model) for model in models]
    first_only = np.asarray([a and not b for a, b in roles])
    second_only = np.asarray([b and not a for a, b in roles])
    neither = np.asarray([not a and not b for a, b in roles])
    both = np.asarray([a and b for a, b in roles])

    ax.scatter(
        truth[neither, 1], truth[neither, 0], s=26, color=BACKGROUND_COLOR,
        alpha=0.85, edgecolors="none", zorder=1,
    )
    frontier_indices = np.asarray(pareto_min_cost_indices(truth), dtype=int)
    frontier_indices = frontier_indices[np.argsort(truth[frontier_indices, 1])]
    ax.plot(
        truth[frontier_indices, 1], truth[frontier_indices, 0],
        color="#3f4854", linewidth=1.7, marker="o", markersize=5,
        markerfacecolor="white", markeredgecolor="#3f4854",
        markeredgewidth=1.2, zorder=3,
    )
    ax.scatter(
        truth[first_only, 1], truth[first_only, 0], s=110,
        color=FIRST_COLOR, edgecolors=POINT_EDGE_COLOR, linewidths=0.85, zorder=5,
    )
    ax.scatter(
        truth[second_only, 1], truth[second_only, 0], s=110,
        color=SECOND_COLOR, edgecolors=POINT_EDGE_COLOR, linewidths=0.85, zorder=5,
    )
    for x, y in truth[both][:, [1, 0]]:
        _draw_bicolor_point(ax, float(x), float(y), point_scale)

    ax.set_title(LABELS[benchmark], fontsize=22, pad=10)
    ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-benchmarks", action="store_true",
        help="Draw HotpotQA, MathQA, GPQA, and BFCL in a 2x2 appendix figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path without extension.",
    )
    args = parser.parse_args()
    default_name = "opus_role_landscape_4benchmarks" if args.all_benchmarks else "opus_role_landscape"
    output = args.output or (
        ROOT / "analysis/paper_20seed_method_comparison/figures" / default_name
    )
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    benchmarks = (
        ("hotpotqa", "mathqa", "gpqa", "bfcl")
        if args.all_benchmarks else ("hotpotqa", "mathqa")
    )
    if args.all_benchmarks:
        fig, axes_grid = plt.subplots(2, 2, figsize=(11.2, 8.2))
        axes = axes_grid.ravel()
        bottom, top, y_center = 0.20, 0.95, 0.575
        xlabel_y, legend_y, legend_fontsize = bottom - 0.12, -0.015, 15
    else:
        fig, axes_grid = plt.subplots(1, 2, figsize=(11.2, 5.6))
        axes = np.asarray(axes_grid).ravel()
        bottom, top, y_center = 0.32, 0.90, 0.61
        xlabel_y, legend_y, legend_fontsize = 0.17, 0.0, 17
    point_scale = 1.0 if args.all_benchmarks else 1.5
    legend_marker_size = 8 if args.all_benchmarks else 10
    for ax, benchmark in zip(axes, benchmarks):
        _plot_dataset(ax, benchmark, point_scale)

    fig.supxlabel("Mean deployment cost (USD)", fontsize=20, y=xlabel_y)
    # Centre the shared y label on the plotting area, excluding the legend.
    fig.supylabel("Mean accuracy", fontsize=20, x=0.015, y=y_center)
    legend_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=legend_marker_size,
               markerfacecolor=FIRST_COLOR, markeredgecolor=POINT_EDGE_COLOR,
               label="Opus in first role (planner / answer)"),
        Line2D([], [], linestyle="none", marker="o", markersize=legend_marker_size,
               markerfacecolor=SECOND_COLOR, markeredgecolor=POINT_EDGE_COLOR,
               label="Opus in second role (solver / critic)"),
        Line2D([], [], linestyle="none", marker="o", markersize=legend_marker_size,
               markerfacecolor=BACKGROUND_COLOR, markeredgecolor="none",
               label="No Opus"),
        Line2D([], [], linestyle="none", marker="o", markersize=legend_marker_size,
               markerfacecolor=AGENT_COLOR, markeredgecolor="#343a40",
               label="Opus as the sole agent"),
        Line2D([], [], color="#3f4854", linewidth=1.7, marker="o",
               markerfacecolor="white", markersize=4,
               label="True Pareto frontier"),
    ]
    fig.legend(
        handles=(legend_handles if args.all_benchmarks else [
            legend_handles[0], legend_handles[1], legend_handles[2], legend_handles[4]
        ]),
        loc="lower center", ncol=(3 if args.all_benchmarks else 2), frameon=False,
        fontsize=legend_fontsize, bbox_to_anchor=(0.5, legend_y), columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, top=top, bottom=bottom,
        wspace=0.20, hspace=(0.32 if args.all_benchmarks else 0.20),
    )
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
