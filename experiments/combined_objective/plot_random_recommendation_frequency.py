#!/usr/bin/env python3
"""Create paper-ready single-run and multi-seed random-search frontier plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
    pareto_min_cost_indices,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


BENCHMARK_PICKLES = {
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}
BENCHMARK_LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}


@dataclass(frozen=True)
class SavedRecommendation:
    seed: int
    budget_fraction: float
    selected_models: tuple[str, ...]
    hypervolume_ratio: float


def load_saved_recommendations(
    path: Path, *, version: str, budget_fraction: float,
) -> list[SavedRecommendation]:
    """Load one protocol/budget slice from the saved multi-seed CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    selected = []
    for record in records:
        if record["version"] != version:
            continue
        if not np.isclose(float(record["budget_fraction"]), budget_fraction):
            continue
        ground_truth_hv = float(record["ground_truth_hypervolume"])
        selected.append(
            SavedRecommendation(
                seed=int(record["seed"]),
                budget_fraction=float(record["budget_fraction"]),
                selected_models=tuple(json.loads(record["selected_models"])),
                hypervolume_ratio=float(record["hypervolume"]) / ground_truth_hv,
            )
        )
    if not selected:
        raise ValueError(
            f"no rows for version={version!r}, budget={budget_fraction:g} in {path}"
        )
    seeds = [result.seed for result in selected]
    if len(seeds) != len(set(seeds)):
        raise ValueError("result slice contains duplicate seeds")
    return sorted(selected, key=lambda result: result.seed)


def recommendation_counts(
    recommendations: Sequence[SavedRecommendation], models: Sequence[str],
) -> np.ndarray:
    """Count in how many seed-level recommendation sets each model occurs."""
    model_index = {model: index for index, model in enumerate(models)}
    counts: Counter[str] = Counter()
    for result in recommendations:
        counts.update(set(result.selected_models))
    unknown = sorted(set(counts) - set(model_index))
    if unknown:
        raise ValueError(f"saved results contain unknown configurations: {unknown}")
    return np.asarray([counts[model] for model in models], dtype=int)


def choose_representative_run(
    recommendations: Sequence[SavedRecommendation],
) -> SavedRecommendation:
    """Choose a median-quality run, breaking close ties toward a sparse set.

    Runs in the middle half of the HV-ratio distribution are considered
    representative. Within that band, the run with the fewest recommendations
    is preferred, followed by proximity to the median HV ratio.
    """
    ratios = np.asarray([result.hypervolume_ratio for result in recommendations])
    lower, median, upper = np.quantile(ratios, [0.25, 0.5, 0.75])
    middle = [
        result
        for result in recommendations
        if lower <= result.hypervolume_ratio <= upper
    ]
    return min(
        middle,
        key=lambda result: (
            len(result.selected_models),
            abs(result.hypervolume_ratio - median),
            result.seed,
        ),
    )


def _frontier(truth_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(pareto_min_cost_indices(truth_vectors), dtype=int)
    indices = indices[np.argsort(truth_vectors[indices, 1])]
    return indices, truth_vectors[indices]


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.set_xlabel("Mean deployment cost (USD; lower is better)")
    ax.set_ylabel("Mean accuracy (higher is better)")
    ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def _draw_landscape(ax: mpl.axes.Axes, truth_vectors: np.ndarray) -> np.ndarray:
    front_indices, front = _frontier(truth_vectors)
    ax.scatter(
        truth_vectors[:, 1], truth_vectors[:, 0], s=24, color="#d8dce2",
        edgecolors="none", alpha=0.72, label="Candidate configurations", zorder=1,
    )
    ax.plot(
        front[:, 1], front[:, 0], color="#3f4854", linewidth=1.7,
        marker="o", markersize=3.8, markerfacecolor="white",
        markeredgewidth=0.9, label="True Pareto frontier", zorder=3,
    )
    return front_indices


def plot_single_run(
    truth_vectors: np.ndarray,
    models: Sequence[str],
    result: SavedRecommendation,
    *,
    benchmark: str,
    version_label: str,
    output_stem: Path,
    budget_label: str | None = None,
) -> None:
    """Plot all candidates, the true frontier, and one run's recommendations."""
    model_index = {model: index for index, model in enumerate(models)}
    selected = np.asarray([model_index[name] for name in result.selected_models])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    _draw_landscape(ax, truth_vectors)
    points = truth_vectors[selected]
    ax.scatter(
        points[:, 1], points[:, 0], s=76, marker="o", color="#d55e00",
        edgecolors="white", linewidths=1.0,
        label=f"Recommended ({len(selected)})", zorder=5,
    )
    _style_axis(ax)
    ax.set_title(
        f"{BENCHMARK_LABELS.get(benchmark, benchmark)}: single random-search run\n"
        f"{version_label}, {budget_label or f'budget {result.budget_fraction:.0%}'}, "
        f"seed {result.seed}"
    )
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_recommendation_frequency(
    truth_vectors: np.ndarray,
    models: Sequence[str],
    recommendations: Sequence[SavedRecommendation],
    *,
    benchmark: str,
    version_label: str,
    output_stem: Path,
    budget_label: str | None = None,
) -> None:
    """Plot per-configuration recommendation frequency across random seeds."""
    counts = recommendation_counts(recommendations, models)
    recommended = counts > 0
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    _draw_landscape(ax, truth_vectors)

    # The palest end of YlOrRd is almost white and disappears among the gray
    # candidates.  Truncate it and add a restrained warm-gray outline so that
    # even a configuration recommended once remains legible in print.
    base_cmap = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "YlOrRd_paper", base_cmap(np.linspace(0.18, 1.0, 256)),
    )
    # Scale to the observed maximum so that the most frequently recommended
    # configuration is visually dark even when no model appears in every run.
    max_count = int(counts.max())
    norm = mpl.colors.Normalize(vmin=1, vmax=max_count)
    scatter = ax.scatter(
        truth_vectors[recommended, 1], truth_vectors[recommended, 0],
        c=counts[recommended], cmap=cmap, norm=norm, s=62,
        edgecolors="#725b46", linewidths=0.55, zorder=5,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.055)
    colorbar.set_label(f"Recommendation frequency (out of {len(recommendations)} seeds)")
    colorbar.set_ticks(np.unique(np.linspace(1, max_count, 5, dtype=int)))
    _style_axis(ax)
    ax.set_title(
        f"{BENCHMARK_LABELS.get(benchmark, benchmark)}: aggregate recommendation frequency\n"
        f"{version_label}, "
        f"{budget_label or f'budget {recommendations[0].budget_fraction:.0%}'}"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_single_and_frequency_panel(
    truth_vectors: np.ndarray,
    models: Sequence[str],
    representative: SavedRecommendation,
    recommendations: Sequence[SavedRecommendation],
    *,
    benchmark: str,
    version_label: str,
    output_stem: Path,
    budget_label: str | None = None,
) -> None:
    """Place the representative run and aggregate frequency side by side."""
    model_index = {model: index for index, model in enumerate(models)}
    selected = np.asarray(
        [model_index[name] for name in representative.selected_models], dtype=int,
    )
    counts = recommendation_counts(recommendations, models)
    recommended = counts > 0
    base_cmap = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "YlOrRd_paper_panel", base_cmap(np.linspace(0.18, 1.0, 256)),
    )
    max_count = int(counts.max())
    norm = mpl.colors.Normalize(vmin=1, vmax=max_count)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.75), sharex=True, sharey=True)
    for ax in axes:
        _draw_landscape(ax, truth_vectors)
        _style_axis(ax)

    single_points = truth_vectors[selected]
    axes[0].scatter(
        single_points[:, 1], single_points[:, 0], s=76, marker="o",
        color="#d55e00", edgecolors="white", linewidths=1.0,
        label=f"Recommended ({len(selected)})", zorder=5,
    )
    axes[0].set_title(
        f"(a) Single run (seed {representative.seed})\n"
        f"{len(selected)} recommendations"
    )
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")

    scatter = axes[1].scatter(
        truth_vectors[recommended, 1], truth_vectors[recommended, 0],
        c=counts[recommended], cmap=cmap, norm=norm, s=62,
        edgecolors="#725b46", linewidths=0.55, zorder=5,
    )
    axes[1].set_title(
        f"(b) Aggregate across {len(recommendations)} seeds\n"
        "Recommendation frequency"
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], labels[:2], frameon=False, fontsize=8.5, loc="best")
    colorbar = fig.colorbar(scatter, ax=axes[1], pad=0.025, fraction=0.055)
    colorbar.set_label(f"Recommendation frequency (out of {len(recommendations)} seeds)")
    colorbar.set_ticks(np.unique(np.linspace(1, max_count, 5, dtype=int)))

    # Only the left panel needs the shared y-axis label.
    axes[1].set_ylabel("")
    fig.suptitle(
        f"{BENCHMARK_LABELS.get(benchmark, benchmark)} — {version_label}, "
        f"{budget_label or f'budget {representative.budget_fraction:.0%}'}",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _save_figure(fig: mpl.figure.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_PICKLES), default="hotpotqa")
    parser.add_argument(
        "--version", choices=("random_configurations", "random_questions"),
        default="random_configurations",
    )
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument(
        "--single-seed", type=int, default=42,
        help="Seed used in the single-run figure and left panel (default: 42).",
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=ROOT / "analysis/vs/random_search_20seeds",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "analysis/vs/random_search_20seeds/paper_frequency_frontier",
    )
    args = parser.parse_args()

    models, datapoints, table = load_pickle(str(BENCHMARK_PICKLES[args.benchmark]))
    questions = common_question_ids(models, datapoints, table)
    truth_vectors = mean_raw_vectors(models, questions, table)
    csv_path = args.results_root / args.benchmark / "multi_seed_results.csv"
    recommendations = load_saved_recommendations(
        csv_path, version=args.version, budget_fraction=args.budget,
    )
    try:
        representative = next(
            result for result in recommendations if result.seed == args.single_seed
        )
    except StopIteration as error:
        available = ", ".join(str(result.seed) for result in recommendations)
        raise ValueError(
            f"single-run seed {args.single_seed} is unavailable; available seeds: {available}"
        ) from error
    version_label = {
        "random_configurations": "random configurations",
        "random_questions": "random shared questions",
    }[args.version]
    prefix = f"{args.benchmark}_{args.version}_{round(100 * args.budget):03d}pct"
    plot_single_run(
        truth_vectors, models, representative, benchmark=args.benchmark,
        version_label=version_label, output_stem=args.outdir / f"{prefix}_single_run",
    )
    plot_recommendation_frequency(
        truth_vectors, models, recommendations, benchmark=args.benchmark,
        version_label=version_label, output_stem=args.outdir / f"{prefix}_aggregate_frequency",
    )
    plot_single_and_frequency_panel(
        truth_vectors, models, representative, recommendations,
        benchmark=args.benchmark, version_label=version_label,
        output_stem=args.outdir / f"{prefix}_single_and_aggregate_panel",
    )
    print(
        f"single-run seed={representative.seed}, "
        f"recommendations={len(representative.selected_models)}, "
        f"HV ratio={representative.hypervolume_ratio:.4f}"
    )
    print(f"wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
