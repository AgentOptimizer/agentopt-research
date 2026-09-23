#!/usr/bin/env python3
"""Compare six 20-seed recommendation frontiers for one dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "analysis/20seed_results"
DEFAULT_AGGREGATE = (
    ROOT
    / "experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/aggregate"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures/dataset_method_frontiers"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.plot_baseline_20seed_frequency_frontiers import (  # noqa: E402
    _summary as _baseline_summary,
)
from experiments.combined_objective.plot_gittins_20seed_frequency_frontiers import (  # noqa: E402
    DATASETS,
    DATASET_LABELS,
    PANEL_SUBTITLE_FONTSIZE,
    PANEL_TITLE_FONTSIZE,
    SHARED_AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    _frequency_colormap,
    _load_landscape,
    _load_targets,
    _target_summary,
    _usd,
)


# EGE-SR is intentionally not part of this comparison.  The five entries after
# Radial Gittins are the baseline set used in the paper figure.
METHODS = (
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
    "ape_k": "APE-k",
    "qnehvi": "qNEHVI",
    "random_configurations": "Random configurations",
    "random_questions": "Random questions",
}
EXPECTED_SEED_COUNT = 20


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


def _method_summaries(
    *,
    dataset: str,
    target: float,
    results_root: Path,
    aggregate_dir: Path,
    landscape: dict[str, object],
) -> dict[str, dict[str, object]]:
    summaries = {
        "radial_gittins": _target_summary(
            _load_targets(aggregate_dir),
            dataset=dataset,
            target=target,
            model_count=len(np.asarray(landscape["truth"])),
            full_search_cost=float(landscape["full_search_cost"]),
        )
    }
    for method in METHODS[1:]:
        summaries[method] = _baseline_summary(
            results_root=results_root,
            method=method,
            dataset=dataset,
            target=target,
            landscape=landscape,
        )

    incomplete = {
        method: int(summary["available_seed_count"])
        for method, summary in summaries.items()
        if int(summary["available_seed_count"]) != EXPECTED_SEED_COUNT
    }
    if incomplete:
        raise ValueError(
            f"expected {EXPECTED_SEED_COUNT} seeds for every method; got {incomplete}"
        )
    return summaries


def _draw_panel(
    ax: mpl.axes.Axes,
    *,
    method: str,
    landscape: dict[str, object],
    summary: dict[str, object],
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    truth = np.asarray(landscape["truth"])
    frontier = np.asarray(landscape["frontier"], dtype=int)
    frontier = frontier[np.argsort(truth[frontier, 1])]

    ax.scatter(
        truth[:, 1],
        truth[:, 0],
        s=48,
        color="#c0c5cc",
        alpha=0.64,
        edgecolors="none",
        zorder=1,
    )
    ax.plot(
        truth[frontier, 1],
        truth[frontier, 0],
        color="#25282c",
        linewidth=1.7,
        zorder=2,
    )
    ax.scatter(
        truth[frontier, 1],
        truth[frontier, 0],
        s=72,
        facecolors="white",
        edgecolors="#25282c",
        linewidths=1.25,
        zorder=3,
    )

    counts = summary["counts"]
    shown_indices = np.asarray(sorted(counts), dtype=int)
    shown_counts = np.asarray([counts[int(index)] for index in shown_indices], dtype=int)
    ax.scatter(
        truth[shown_indices, 1],
        truth[shown_indices, 0],
        c=shown_counts,
        cmap=cmap,
        norm=norm,
        s=175,
        edgecolors="#6f4a22",
        linewidths=1.1,
        zorder=4,
    )

    mean_spend = float(summary["mean_spend_usd"])
    actual_fraction = mean_spend / float(landscape["full_search_cost"])
    ax.text(
        0.5,
        1.36,
        METHOD_LABELS[method],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    ax.text(
        0.5,
        1.35,
        f"Mean cost: {_usd(mean_spend)}\nActual: {actual_fraction:.1%}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_SUBTITLE_FONTSIZE,
        linespacing=1.15,
    )

    positive_costs = truth[:, 1][truth[:, 1] > 0.0]
    if len(positive_costs) == len(truth):
        ax.set_xscale("log")
    ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def make_figure(
    *,
    dataset: str,
    target: float,
    results_root: Path,
    aggregate_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    landscape = _load_landscape(results_root, dataset)
    summaries = _method_summaries(
        dataset=dataset,
        target=target,
        results_root=results_root,
        aggregate_dir=aggregate_dir,
        landscape=landscape,
    )

    cmap = _frequency_colormap()
    norm = mpl.colors.Normalize(vmin=1, vmax=EXPECTED_SEED_COUNT)
    # Match the 2x4 20-seed figures' paper canvas as well as their typography.
    # Keeping the full width also prevents the shared 31-point legend from
    # being clipped in the three-column comparison.
    fig, axes = plt.subplots(2, 3, figsize=(24.0, 14.5), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        _draw_panel(
            ax,
            method=method,
            landscape=landscape,
            summary=summaries[method],
            cmap=cmap,
            norm=norm,
        )

    fig.supxlabel(
        "Mean deployment cost (USD per query, log scale)",
        fontsize=SHARED_AXIS_LABEL_FONTSIZE,
        y=0.115,
    )
    fig.supylabel(
        "Mean accuracy", fontsize=SHARED_AXIS_LABEL_FONTSIZE, x=0.022
    )
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: 20-seed recommendation frontiers "
        f"(total search-cost budget: {target:.0%})",
        fontsize=35,
        y=0.950,
    )
    fig.legend(
        handles=(
            Line2D(
                [], [], linestyle="none", marker="o", markersize=13,
                markerfacecolor="#c0c5cc", markeredgecolor="none",
                label="All configurations",
            ),
            Line2D(
                [], [], color="#25282c", linewidth=2.0, marker="o",
                markersize=10, markerfacecolor="white", markeredgewidth=1.25,
                label="Full-data Pareto frontier",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor="#ff7f0e", markeredgecolor="#6f4a22",
                label="Recommendations (color = frequency)",
            ),
        ),
        loc="lower center",
        bbox_to_anchor=(0.48, 0.035),
        ncol=3,
        frameon=False,
        fontsize=SHARED_AXIS_LABEL_FONTSIZE - 1,
        columnspacing=2.2,
        handletextpad=0.65,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.865,
        bottom=0.205,
        top=0.790,
        wspace=0.26,
        hspace=0.82,
    )
    colorbar_ax = fig.add_axes([0.900, 0.245, 0.021, 0.52])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label(
        "Recommendation frequency (out of 20 seeds)", fontsize=30, labelpad=19
    )
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=26)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"{dataset}_2x3_method_frontiers_{int(round(100 * target))}pct"
    )
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="stackoverflow")
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 < args.target <= 1.0:
        parser.error("--target must be in (0, 1]")

    png_path, pdf_path = make_figure(
        dataset=args.dataset,
        target=args.target,
        results_root=args.results_root,
        aggregate_dir=args.aggregate_dir,
        output_dir=args.output_dir,
    )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
