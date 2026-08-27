#!/usr/bin/env python3
"""Plot 10%-through-100% Pareto snapshots for two random-search baselines."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    DEFAULT_BUDGET_FRACTIONS,
    VERSIONS,
    MultiObjectiveRandomSearchResult,
    pareto_min_cost_indices,
    run_budget_sweep,
    write_results_csv,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


BENCHMARK_PICKLES = {
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}


def _version_title(version: str) -> str:
    return {
        "random_configurations": "Random configurations, all questions",
        "random_questions": "All configurations, random shared questions",
    }[version]


def plot_snapshot(
    result: MultiObjectiveRandomSearchResult,
    *,
    benchmark: str,
    output_path: Path,
) -> None:
    truth = result.truth_vectors
    true_front_indices = set(pareto_min_cost_indices(truth))
    true_front = truth[sorted(true_front_indices)]
    true_front = true_front[np.argsort(true_front[:, 1])]
    selected_indices = set(result.selected_arm_indices)
    correct = sorted(selected_indices & true_front_indices)
    false_positive = sorted(selected_indices - true_front_indices)
    missed = sorted(true_front_indices - selected_indices)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        truth[:, 1], truth[:, 0], s=18, color="#b8c0cc", alpha=0.6,
        label="all configurations", zorder=1,
    )
    ax.plot(
        true_front[:, 1], true_front[:, 0], color="#5b6472", marker="o",
        markersize=4, linewidth=1.4, label="brute-force Pareto frontier", zorder=2,
    )
    groups = (
        (correct, "#2f855a", "*", 105, "recommended and on true front"),
        (false_positive, "#1f4e79", "s", 62, "recommended but dominated"),
        (missed, "#c45c26", "D", 58, "true-front point not recommended"),
    )
    for indices, color, marker, size, label in groups:
        if indices:
            points = truth[indices]
            ax.scatter(points[:, 1], points[:, 0], color=color, marker=marker,
                       s=size, edgecolors="white", linewidths=0.7,
                       label=label, zorder=4)
    ax.set_xlabel("Mean deployment cost (USD, lower is better)")
    ax.set_ylabel("Mean accuracy (higher is better)")
    ax.set_title(
        f"{benchmark} — {_version_title(result.version)}\n"
        f"budget={result.budget_fraction:.0%}, seed={result.seed}, "
        f"HV regret={result.hypervolume_regret:.4f}, "
        f"GD={result.generational_distance:.4f}, "
        f"IGD={result.inverted_generational_distance:.4f}, "
        f"precision={result.recommendation_precision:.2f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(
    results: Sequence[MultiObjectiveRandomSearchResult],
    *,
    benchmark: str,
    version: str,
    output_path: Path,
) -> None:
    ordered = sorted(results, key=lambda result: result.budget_fraction)
    truth = ordered[0].truth_vectors
    true_front_indices = set(pareto_min_cost_indices(truth))
    true_front = truth[sorted(true_front_indices)]
    true_front = true_front[np.argsort(true_front[:, 1])]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for ax, result in zip(axes.flat, ordered):
        selected_indices = set(result.selected_arm_indices)
        correct = sorted(selected_indices & true_front_indices)
        false_positive = sorted(selected_indices - true_front_indices)
        missed = sorted(true_front_indices - selected_indices)
        ax.scatter(truth[:, 1], truth[:, 0], s=8, color="#c5cad3", alpha=0.55)
        ax.plot(true_front[:, 1], true_front[:, 0], color="#626b78", linewidth=1.0)
        for indices, color, marker, size in (
            (correct, "#2f855a", "*", 38),
            (false_positive, "#1f4e79", "s", 24),
            (missed, "#c45c26", "D", 22),
        ):
            if indices:
                points = truth[indices]
                ax.scatter(points[:, 1], points[:, 0], color=color,
                           marker=marker, s=size, edgecolors="white",
                           linewidths=0.4, zorder=4)
        ax.set_title(
            f"{result.budget_fraction:.0%}\n"
            f"HV={result.hypervolume_regret:.3f}, "
            f"GD={result.generational_distance:.3f}, "
            f"IGD={result.inverted_generational_distance:.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
    fig.supxlabel("Mean deployment cost (USD)")
    fig.supylabel("Mean accuracy")
    fig.suptitle(f"{benchmark} — {_version_title(version)}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _metric_series(
    results: Sequence[MultiObjectiveRandomSearchResult],
    version: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[float, list[float]] = defaultdict(list)
    for result in results:
        if result.version == version:
            grouped[result.budget_fraction].append(float(getattr(result, field)))
    fractions = np.asarray(sorted(grouped), dtype=np.float64)
    means = np.asarray([np.mean(grouped[f]) for f in fractions], dtype=np.float64)
    ci95 = np.asarray(
        [
            1.96 * np.std(grouped[f], ddof=1) / np.sqrt(len(grouped[f]))
            if len(grouped[f]) > 1
            else 0.0
            for f in fractions
        ],
        dtype=np.float64,
    )
    return fractions, means, ci95


def _draw_metric_axis(
    ax,
    results: Sequence[MultiObjectiveRandomSearchResult],
    benchmark: str,
    *,
    field: str,
    ylabel: str,
) -> None:
    styles = {
        "random_configurations": ("#c45c26", "Random configurations"),
        "random_questions": ("#1f4e79", "Random shared questions"),
    }
    for version in VERSIONS:
        fractions, means, ci95 = _metric_series(results, version, field)
        color, label = styles[version]
        ax.plot(fractions, means, marker="o", linewidth=1.8, color=color, label=label)
        ax.fill_between(
            fractions,
            np.maximum(0.0, means - ci95),
            means + ci95,
            color=color,
            alpha=0.16,
            linewidth=0,
            label="95% CI" if version == VERSIONS[0] else None,
        )
    ax.set_title(benchmark)
    ax.set_xlabel("Observed cell-budget fraction")
    ax.set_ylabel(ylabel)
    ax.set_xticks(DEFAULT_BUDGET_FRACTIONS)
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _draw_regret_axis(
    ax, results: Sequence[MultiObjectiveRandomSearchResult], benchmark: str,
) -> None:
    _draw_metric_axis(
        ax,
        results,
        benchmark,
        field="hypervolume_regret",
        ylabel="Hypervolume regret (lower is better)",
    )


def plot_regret_curve(
    results: Sequence[MultiObjectiveRandomSearchResult],
    *,
    benchmark: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _draw_regret_axis(ax, results, benchmark)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curve(
    results: Sequence[MultiObjectiveRandomSearchResult],
    *,
    benchmark: str,
    field: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _draw_metric_axis(ax, results, benchmark, field=field, ylabel=ylabel)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined_regret_curves(
    results_by_benchmark: Dict[str, Sequence[MultiObjectiveRandomSearchResult]],
    output_path: Path,
) -> None:
    plot_combined_metric_curves(
        results_by_benchmark,
        output_path,
        field="hypervolume_regret",
        ylabel="Hypervolume regret (lower is better)",
        title="Multi-objective Random Search vs observed budget",
    )


def plot_combined_metric_curves(
    results_by_benchmark: Dict[str, Sequence[MultiObjectiveRandomSearchResult]],
    output_path: Path,
    *,
    field: str,
    ylabel: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(results_by_benchmark), figsize=(12.5, 4.6))
    if len(results_by_benchmark) == 1:
        axes = [axes]
    for ax, (benchmark, results) in zip(axes, results_by_benchmark.items()):
        _draw_metric_axis(ax, results, benchmark, field=field, ylabel=ylabel)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/random_search_pareto"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-seeds", type=int, default=50)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(BENCHMARK_PICKLES),
        default=list(BENCHMARK_PICKLES),
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    summary_by_benchmark = {}

    for benchmark in args.benchmarks:
        models, datapoints, table = load_pickle(str(BENCHMARK_PICKLES[benchmark]))
        plot_results = run_budget_sweep(
            models, datapoints, table, seeds=(args.seed,),
        )
        benchmark_dir = outdir / benchmark
        for result in plot_results:
            percent = int(round(100 * result.budget_fraction))
            plot_snapshot(
                result,
                benchmark=benchmark,
                output_path=(
                    benchmark_dir / result.version / f"pareto_{percent:03d}pct.png"
                ),
            )
        for version in VERSIONS:
            version_results = [r for r in plot_results if r.version == version]
            plot_contact_sheet(
                version_results,
                benchmark=benchmark,
                version=version,
                output_path=benchmark_dir / f"{version}_contact_sheet.png",
            )

        summary_results = run_budget_sweep(
            models,
            datapoints,
            table,
            seeds=range(args.seed, args.seed + args.summary_seeds),
        )
        write_results_csv(summary_results, benchmark_dir / "multi_seed_results.csv")
        plot_regret_curve(
            summary_results,
            benchmark=benchmark,
            output_path=benchmark_dir / "hv_regret_curve.png",
        )
        plot_metric_curve(
            summary_results,
            benchmark=benchmark,
            field="generational_distance",
            ylabel="Generational distance (GD, lower is better)",
            output_path=benchmark_dir / "gd_curve.png",
        )
        plot_metric_curve(
            summary_results,
            benchmark=benchmark,
            field="inverted_generational_distance",
            ylabel="Inverted generational distance (IGD, lower is better)",
            output_path=benchmark_dir / "igd_curve.png",
        )
        summary_by_benchmark[benchmark] = summary_results
        print(f"wrote {benchmark_dir}")

    plot_combined_regret_curves(
        summary_by_benchmark,
        outdir / "hv_regret_curves.png",
    )
    print(f"wrote {outdir / 'hv_regret_curves.png'}")
    plot_combined_metric_curves(
        summary_by_benchmark,
        outdir / "gd_curves.png",
        field="generational_distance",
        ylabel="Generational distance (GD, lower is better)",
        title="Multi-objective Random Search GD vs observed budget",
    )
    print(f"wrote {outdir / 'gd_curves.png'}")
    plot_combined_metric_curves(
        summary_by_benchmark,
        outdir / "igd_curves.png",
        field="inverted_generational_distance",
        ylabel="Inverted generational distance (IGD, lower is better)",
        title="Multi-objective Random Search IGD vs observed budget",
    )
    print(f"wrote {outdir / 'igd_curves.png'}")


if __name__ == "__main__":
    main()
