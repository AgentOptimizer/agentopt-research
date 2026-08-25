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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


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
    full_search_cost_usd: float,
    output_path: Path,
) -> None:
    ordered = sorted(results, key=lambda result: result.budget_fraction)
    truth = ordered[0].truth_vectors
    true_front_indices = set(pareto_min_cost_indices(truth))
    true_front = truth[sorted(true_front_indices)]
    true_front = true_front[np.argsort(true_front[:, 1])]
    ncols = min(5, len(ordered))
    nrows = int(np.ceil(len(ordered) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax, result in zip(axes.flat, ordered):
        selected_indices = set(result.selected_arm_indices)
        correct = sorted(selected_indices & true_front_indices)
        false_positive = sorted(selected_indices - true_front_indices)
        missed = sorted(true_front_indices - selected_indices)
        ax.scatter(truth[:, 1], truth[:, 0], s=8, color="#c5cad3", alpha=0.55)
        ax.plot(true_front[:, 1], true_front[:, 0], color="#626b78", linewidth=1.0)
        for indices, color, marker, size in (
            (correct, "#2f855a", "*", 105),
            (false_positive, "#1f4e79", "s", 24),
            (missed, "#c45c26", "D", 22),
        ):
            if indices:
                points = truth[indices]
                ax.scatter(points[:, 1], points[:, 0], color=color,
                           marker=marker, s=size, edgecolors="white",
                           linewidths=0.4, zorder=4)
        ax.set_title(
            f"{result.budget_fraction:.0%} "
            f"(cost={result.total_search_cost_usd / full_search_cost_usd:.1%})\n"
            f"regret={result.hypervolume_regret:.4f}, "
            f"FP={result.false_positive_count}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(ordered):]:
        ax.set_visible(False)
    fig.supxlabel("Mean deployment cost (USD)")
    fig.supylabel("Mean accuracy", x=0.002)
    fig.suptitle(f"{benchmark} — {_version_title(version)}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined_contact_sheet(
    results_by_benchmark: Dict[str, Sequence[MultiObjectiveRandomSearchResult]],
    *,
    version: str,
    full_search_costs_usd: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot budget checkpoints as rows and benchmarks as columns."""
    benchmark_items = list(results_by_benchmark.items())
    ordered_by_benchmark = {
        benchmark: sorted(results, key=lambda result: result.budget_fraction)
        for benchmark, results in benchmark_items
    }
    n_checkpoints = len(next(iter(ordered_by_benchmark.values())))
    fig, axes = plt.subplots(
        n_checkpoints, len(benchmark_items), figsize=(13, 4.6 * n_checkpoints),
        sharex="col", sharey="col", squeeze=False,
    )
    for col, (benchmark, _) in enumerate(benchmark_items):
        ordered = ordered_by_benchmark[benchmark]
        truth = ordered[0].truth_vectors
        true_front_indices = set(pareto_min_cost_indices(truth))
        true_front = truth[sorted(true_front_indices)]
        true_front = true_front[np.argsort(true_front[:, 1])]
        for row, result in enumerate(ordered):
            ax = axes[row, col]
            selected_indices = set(result.selected_arm_indices)
            correct = sorted(selected_indices & true_front_indices)
            false_positive = sorted(selected_indices - true_front_indices)
            missed = sorted(true_front_indices - selected_indices)
            ax.scatter(truth[:, 1], truth[:, 0], s=18, color="#c5cad3", alpha=0.55)
            ax.plot(
                true_front[:, 1], true_front[:, 0], color="#626b78",
                marker="o", markersize=5, linewidth=1.0,
            )
            for indices, color, marker, size in (
                (correct, "#2f855a", "*", 155),
                (false_positive, "#1f4e79", "s", 42),
                (missed, "#c45c26", "D", 40),
            ):
                if indices:
                    points = truth[indices]
                    ax.scatter(
                        points[:, 1], points[:, 0], color=color, marker=marker,
                        s=size, edgecolors="white", linewidths=0.4, zorder=4,
                    )
            ax.set_title(
                f"{result.budget_fraction:.0%} "
                f"(cost={result.total_search_cost_usd / full_search_costs_usd[benchmark]:.1%})\n"
                f"regret={result.hypervolume_regret:.4f}, "
                f"FP={result.false_positive_count}",
                fontsize=18,
            )
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=16)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        display_name = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}.get(
            benchmark, benchmark,
        )
        axes[0, col].annotate(
            display_name, xy=(0.5, 1.28), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=24,
        )
    for row in range(n_checkpoints):
        axes[row, 0].set_ylabel("Mean accuracy", labelpad=22, fontsize=19)
    neutral_legend_handles = [
        Line2D([], [], linestyle="", marker="o", markersize=6,
               color="#c5cad3", label="all configurations"),
        Line2D([], [], color="#626b78", marker="o", markersize=4,
               label="true full-data Pareto front"),
    ]
    colored_legend_handles = [
        Line2D([], [], linestyle="", marker="*", markersize=12,
               color="#2f855a", label="correctly recommended"),
        Line2D([], [], linestyle="", marker="D", markersize=7,
               color="#c45c26", label="missed Pareto arm"),
        Line2D([], [], linestyle="", marker="s", markersize=7,
               color="#1f4e79", label="false-positive recommendation"),
    ]
    fig.supxlabel("Mean deployment cost (USD)", fontsize=21, y=0.072)
    fig.suptitle(_version_title(version), fontsize=27, y=0.978)
    fig.legend(
        handles=colored_legend_handles, loc="lower center", ncol=3,
        frameon=False, fontsize=17, bbox_to_anchor=(0.5, 0.028),
        handletextpad=0.7, columnspacing=1.8,
    )
    fig.legend(
        handles=neutral_legend_handles, loc="lower center", ncol=2,
        frameon=False, fontsize=17, bbox_to_anchor=(0.5, 0.003),
        handletextpad=0.7, columnspacing=2.2,
    )
    fig.tight_layout(rect=(0.065, 0.105, 0.935, 0.945), h_pad=2.4, w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _regret_series(
    results: Sequence[MultiObjectiveRandomSearchResult], version: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[float, list[float]] = defaultdict(list)
    for result in results:
        if result.version == version:
            grouped[result.budget_fraction].append(result.hypervolume_regret)
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


def _draw_regret_axis(
    ax, results: Sequence[MultiObjectiveRandomSearchResult], benchmark: str,
) -> None:
    styles = {
        "random_configurations": ("#c45c26", "Random configurations"),
        "random_questions": ("#1f4e79", "Random shared questions"),
    }
    for version in VERSIONS:
        fractions, means, ci95 = _regret_series(results, version)
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
    ax.set_ylabel("Hypervolume regret (lower is better)")
    ax.set_xticks(DEFAULT_BUDGET_FRACTIONS)
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


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


def plot_combined_regret_curves(
    results_by_benchmark: Dict[str, Sequence[MultiObjectiveRandomSearchResult]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(results_by_benchmark), figsize=(12.5, 4.6))
    if len(results_by_benchmark) == 1:
        axes = [axes]
    for ax, (benchmark, results) in zip(axes, results_by_benchmark.items()):
        _draw_regret_axis(ax, results, benchmark)
    fig.suptitle("Multi-objective Random Search vs observed budget", fontsize=13)
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
        "--plot-budget-percentages", nargs="+", type=float,
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Cell-budget percentages to include in snapshots/contact sheets.",
    )
    parser.add_argument(
        "--contact-sheets-only", action="store_true",
        help="Skip individual snapshots and multi-seed regret summaries.",
    )
    parser.add_argument(
        "--contact-sheet-suffix", default="",
        help="Optional filename suffix for contact-sheet outputs.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(BENCHMARK_PICKLES),
        default=list(BENCHMARK_PICKLES),
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    summary_by_benchmark = {}
    plot_results_by_benchmark = {}
    full_search_costs_usd = {}
    plot_budget_fractions = tuple(value / 100.0 for value in args.plot_budget_percentages)

    for benchmark in args.benchmarks:
        models, datapoints, table = load_pickle(str(BENCHMARK_PICKLES[benchmark]))
        plot_results = run_budget_sweep(
            models, datapoints, table, seeds=(args.seed,),
            budget_fractions=plot_budget_fractions,
        )
        benchmark_dir = outdir / benchmark
        questions = tuple(
            question_id for question_id in datapoints
            if all(question_id in table.get(model, {}) for model in models)
        )
        full_search_cost_usd = sum(
            table[model][question_id].cost
            for model in models
            for question_id in questions
        )
        plot_results_by_benchmark[benchmark] = plot_results
        full_search_costs_usd[benchmark] = full_search_cost_usd
        if not args.contact_sheets_only:
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
            contact_sheet_path = (
                outdir / f"{benchmark}_{version}{args.contact_sheet_suffix}.png"
                if args.contact_sheets_only
                else benchmark_dir / f"{version}_contact_sheet.png"
            )
            plot_contact_sheet(
                version_results,
                benchmark=benchmark,
                version=version,
                full_search_cost_usd=full_search_cost_usd,
                output_path=contact_sheet_path,
            )

        if args.contact_sheets_only:
            print(f"wrote {outdir / f'{benchmark}_*.png'}")
            continue

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
        summary_by_benchmark[benchmark] = summary_results
        print(f"wrote {benchmark_dir}")

    if args.contact_sheets_only:
        for version in VERSIONS:
            plot_combined_contact_sheet(
                {
                    benchmark: [
                        result for result in plot_results_by_benchmark[benchmark]
                        if result.version == version
                    ]
                    for benchmark in args.benchmarks
                },
                version=version,
                full_search_costs_usd=full_search_costs_usd,
                output_path=(
                    outdir / f"combined_{version}{args.contact_sheet_suffix}.png"
                ),
            )
        return

    plot_combined_regret_curves(
        summary_by_benchmark,
        outdir / "hv_regret_curves.png",
    )
    print(f"wrote {outdir / 'hv_regret_curves.png'}")


if __name__ == "__main__":
    main()
