#!/usr/bin/env python3
"""Plot 20-seed Radial Gittins recommendation frequencies.

The two paper-preview figures use the same 2x4 layout.  One summarizes the
recommendations saved for the nominal 10% checkpoint and the other summarizes
the nominal 30% checkpoint.  Every colored configuration therefore represents
between one and twenty independent seeds, rather than a singled-out run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "analysis/20seed_results"
DEFAULT_AGGREGATE = (
    ROOT
    / "experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/aggregate"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures/gittins_frequency_frontiers"

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
EXPECTED_SEEDS = set(range(42, 62))
PANEL_TITLE_FONTSIZE = 30
PANEL_SUBTITLE_FONTSIZE = PANEL_TITLE_FONTSIZE - 1
TICK_LABEL_FONTSIZE = 25
SHARED_AXIS_LABEL_FONTSIZE = 32


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


def _frequency_colormap() -> LinearSegmentedColormap:
    """Single-hue Gittins palette without the red end of YlOrRd."""

    return LinearSegmentedColormap.from_list(
        "gittins_orange_frequency",
        ("#fff3d8", "#f6bf73", "#ff7f0e", "#8a4b08"),
        N=256,
    )


def _load_landscape(results_root: Path, dataset: str) -> dict[str, object]:
    result_path = results_root / "radial_gittins" / dataset / "seed-42" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing Radial Gittins result: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    run = payload["run"]
    truth = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError(f"unexpected raw_truth_vectors shape for {dataset}: {truth.shape}")
    return {
        "truth": truth,
        "frontier": np.asarray(run["full_data_pareto_arm_indices"], dtype=int),
        "full_search_cost": float(run["bruteforce_search_cost_usd"]),
        "model_names": list(run["model_names"]),
    }


def _load_targets(aggregate_dir: Path) -> list[dict[str, str]]:
    path = aggregate_dir / "plot_targets.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing aggregate target table: {path}")
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _target_summary(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    target: float,
    model_count: int,
    full_search_cost: float,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["benchmark"] == dataset
        and np.isclose(float(row["target_cost_fraction"]), target)
        and row["available"].lower() == "true"
    ]
    seeds = {int(row["seed"]) for row in selected}
    if seeds != EXPECTED_SEEDS:
        missing = sorted(EXPECTED_SEEDS - seeds)
        extra = sorted(seeds - EXPECTED_SEEDS)
        raise ValueError(
            f"{dataset} at {target:.0%} does not have exactly seeds 42--61; "
            f"missing={missing}, extra={extra}"
        )

    counts: Counter[int] = Counter()
    spend_usd: list[float] = []
    recalls: list[float] = []
    exact = 0
    for row in selected:
        indices = [int(value) for value in json.loads(row["selected_arm_indices"])]
        if any(index < 0 or index >= model_count for index in indices):
            raise ValueError(f"invalid selected arm index for {dataset}, seed {row['seed']}")
        counts.update(indices)
        # Recommendations are materialized only when membership changes.  Use
        # the actual source snapshot cost rather than relabeling it with the
        # nominal 10%/30% target.
        spend_usd.append(float(row["source_snapshot_cost_fraction"]) * full_search_cost)
        recalls.append(float(row["pareto_recall"]))
        exact += int(
            int(row["pareto_false_positive_count"]) == 0
            and int(row["pareto_false_negative_count"]) == 0
        )

    return {
        "counts": counts,
        "available_seed_count": len(selected),
        "mean_spend_usd": float(np.mean(spend_usd)),
        "spend_iqr_usd": tuple(float(value) for value in np.quantile(spend_usd, (0.25, 0.75))),
        "mean_recall": float(np.mean(recalls)),
        "exact": exact,
    }


def _usd(value: float) -> str:
    if value < 1.0:
        return f"${value:.2f}"
    if value < 100.0:
        return f"${value:.1f}"
    return f"${value:,.0f}"


def _draw_panel(
    ax: mpl.axes.Axes,
    *,
    dataset: str,
    landscape: dict[str, object],
    summary: dict[str, object],
    target: float,
    cmap: LinearSegmentedColormap,
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
    if len(shown_indices):
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

    available = int(summary.get("available_seed_count", 20))
    actual_cost_fraction = (
        float(summary["mean_spend_usd"]) / float(landscape["full_search_cost"])
        if available
        else float("nan")
    )
    ax.text(
        0.5,
        1.36,
        DATASET_LABELS[dataset],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="normal",
    )
    ax.text(
        0.5,
        1.35,
        (
            f"Mean cost: {_usd(summary['mean_spend_usd'])}\n"
            f"Actual: {actual_cost_fraction:.1%}"
            if available
            else "Recommendation sets unavailable"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_SUBTITLE_FONTSIZE,
        linespacing=1.15,
        color="#000000",
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
    target: float,
    results_root: Path,
    aggregate_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    rows = _load_targets(aggregate_dir)
    cmap = _frequency_colormap()
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    fig, axes = plt.subplots(2, 4, figsize=(24.0, 14.5))

    for ax, dataset in zip(axes.flat, DATASETS):
        landscape = _load_landscape(results_root, dataset)
        summary = _target_summary(
            rows,
            dataset=dataset,
            target=target,
            model_count=len(np.asarray(landscape["truth"])),
            full_search_cost=float(landscape["full_search_cost"]),
        )
        _draw_panel(
            ax,
            dataset=dataset,
            landscape=landscape,
            summary=summary,
            target=target,
            cmap=cmap,
            norm=norm,
        )

    fig.supxlabel(
        "Mean deployment cost (USD per query, log scale)",
        fontsize=SHARED_AXIS_LABEL_FONTSIZE,
        fontweight="normal",
        y=0.115,
    )
    fig.supylabel(
        "Mean accuracy",
        fontsize=SHARED_AXIS_LABEL_FONTSIZE,
        fontweight="normal",
        x=0.022,
    )
    fig.suptitle(
        "Radial Gittins recommendations across 20 seeds "
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
        left=0.080,
        right=0.865,
        bottom=0.205,
        top=0.780,
        wspace=0.26,
        hspace=0.82,
    )
    colorbar_ax = fig.add_axes([0.900, 0.245, 0.021, 0.52])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label(
        "Recommendation frequency (out of 20 seeds)",
        fontsize=30,
        labelpad=19,
    )
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=26)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"gittins_20seed_frequency_frontiers_{int(round(100 * target))}pct"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for target in (0.10, 0.30):
        png_path, pdf_path = make_figure(
            target=target,
            results_root=args.results_root,
            aggregate_dir=args.aggregate_dir,
            output_dir=args.output_dir,
        )
        print(f"wrote {png_path}")
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
