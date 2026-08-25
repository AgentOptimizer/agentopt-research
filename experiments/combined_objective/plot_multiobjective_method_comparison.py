#!/usr/bin/env python3
"""Compare Radial-Gittins and both random-search Pareto baselines in one figure."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
)
from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    MultiObjectiveRandomSearchResult,
    normalized_truth_vectors,
    pareto_min_cost_indices,
    run_budget_sweep,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    hypervolume_2d,
)
from experiments.combined_objective.plot_radial_gittins_trajectories import (  # noqa: E402
    run_benchmark,
)


PICKLES = {
    "HotpotQA": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "MathQA": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}


def _common_hv_context(raw_vectors: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive_costs = raw_vectors[:, 1][raw_vectors[:, 1] > 0.0]
    cost_reference = float(np.median(positive_costs))
    normalized = normalized_truth_vectors(raw_vectors, cost_reference)
    true_front = pareto_min_cost_indices(raw_vectors)
    ground_truth_hv = hypervolume_2d(normalized[true_front])
    return normalized, cost_reference, float(ground_truth_hv)


def _selected_regret(
    normalized_truth: np.ndarray,
    selected_indices: Sequence[int],
    ground_truth_hv: float,
) -> float:
    selected_hv = (
        hypervolume_2d(normalized_truth[list(selected_indices)])
        if selected_indices
        else 0.0
    )
    return float(max(0.0, ground_truth_hv - selected_hv))


def _random_regret_series(
    results: Sequence[MultiObjectiveRandomSearchResult],
    *,
    version: str,
    normalized_truth: np.ndarray,
    ground_truth_hv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[float, list[float]] = defaultdict(list)
    for result in results:
        if result.version != version:
            continue
        grouped[result.budget_fraction].append(
            _selected_regret(
                normalized_truth,
                result.selected_arm_indices,
                ground_truth_hv,
            )
        )
    xs = np.asarray(sorted(grouped), dtype=np.float64)
    means = np.asarray([np.mean(grouped[x]) for x in xs], dtype=np.float64)
    ci95 = np.asarray(
        [
            1.96 * np.std(grouped[x], ddof=1) / np.sqrt(len(grouped[x]))
            if len(grouped[x]) > 1
            else 0.0
            for x in xs
        ]
    )
    return xs, means, ci95


def _radial_regret_series(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate irregular trajectories without reducing them to 10% checkpoints.

    At each checkpoint in the union of all runs, a run contributes its most
    recent recommendation. Runs that have not reached their first checkpoint do
    not contribute yet.
    """
    xs = np.unique(np.concatenate([x for x, _ in trajectories]))
    aligned = np.full((len(trajectories), len(xs)), np.nan, dtype=np.float64)
    for row, (run_x, run_y) in enumerate(trajectories):
        positions = np.searchsorted(run_x, xs, side="right") - 1
        available = positions >= 0
        aligned[row, available] = run_y[positions[available]]

    counts = np.sum(np.isfinite(aligned), axis=0)
    means = np.nanmean(aligned, axis=0)
    ci95 = np.zeros_like(means)
    for column, count in enumerate(counts):
        if count > 1:
            ci95[column] = (
                1.96
                * np.nanstd(aligned[:, column], ddof=1)
                / np.sqrt(count)
            )
    return xs, means, ci95, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/method_comparison"))
    parser.add_argument("--seed", type=int, default=42, help="First run seed.")
    parser.add_argument(
        "--seeds",
        "--random-seeds",
        dest="seeds",
        type=int,
        default=1,
        help="Number of identical run seeds used by every method.",
    )
    parser.add_argument("--batch-size", type=int, default=4, choices=(4, 8))
    parser.add_argument("--grid-size", type=int, default=129)
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cache = RadialGittinsBoundaryCache()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    rows = []

    for ax, (benchmark, pickle_path) in zip(axes, PICKLES.items()):
        run_seeds = range(args.seed, args.seed + args.seeds)
        radial_runs = []
        raw_vectors = None
        evaluation_questions = None
        for run_seed in run_seeds:
            radial, run_vectors, run_questions = run_benchmark(
                pickle_path=pickle_path,
                name=benchmark,
                batch_size=args.batch_size,
                seed=run_seed,
                grid_size=args.grid_size,
                cache=cache,
            )
            radial_runs.append(radial)
            if raw_vectors is None:
                raw_vectors = run_vectors
                evaluation_questions = run_questions
            elif not np.allclose(raw_vectors, run_vectors):
                raise ValueError("Ground-truth vectors changed across run seeds")
        assert raw_vectors is not None and evaluation_questions is not None
        normalized_truth, cost_reference, ground_truth_hv = _common_hv_context(
            raw_vectors
        )

        radial_trajectories = []
        for radial in radial_runs:
            run_x = np.asarray(
                [point.budget_fraction for point in radial.recommendation_trajectory]
            )
            run_y = np.asarray(
                [
                    _selected_regret(
                        normalized_truth,
                        point.selected_arm_indices,
                        ground_truth_hv,
                    )
                    for point in radial.recommendation_trajectory
                ]
            )
            radial_trajectories.append((run_x, run_y))
        radial_x, radial_y, radial_ci95, radial_counts = _radial_regret_series(
            radial_trajectories
        )
        ax.plot(
            radial_x,
            radial_y,
            color="#1f4e79",
            linewidth=1.8,
            label=(
                f"Radial-Gittins (seed={args.seed})"
                if args.seeds == 1
                else f"Radial-Gittins (mean, n={args.seeds})"
            ),
            zorder=3,
        )
        ax.fill_between(
            radial_x,
            np.maximum(0.0, radial_y - radial_ci95),
            radial_y + radial_ci95,
            color="#1f4e79",
            alpha=0.13,
            linewidth=0,
        )
        stop_fractions = np.asarray(
            [
                run.gittins_stop_budget_fraction
                for run in radial_runs
                if run.gittins_stop_budget_fraction is not None
            ],
            dtype=np.float64,
        )
        if len(stop_fractions):
            stop_mean = float(np.mean(stop_fractions))
            stop_ci95 = (
                1.96 * float(np.std(stop_fractions, ddof=1)) / np.sqrt(len(stop_fractions))
                if len(stop_fractions) > 1
                else 0.0
            )
            ax.axvline(
                stop_mean,
                color="#c45c26",
                linestyle="--",
                linewidth=1.4,
                label=(
                    f"Gittins stop ({stop_mean:.1%})"
                    if args.seeds == 1
                    else f"Mean Gittins stop ({stop_mean:.1%})"
                ),
            )
            ax.axvspan(
                max(0.0, stop_mean - stop_ci95),
                min(1.0, stop_mean + stop_ci95),
                color="#c45c26",
                alpha=0.10,
                linewidth=0,
            )

        for x, mean, ci, count in zip(
            radial_x, radial_y, radial_ci95, radial_counts
        ):
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": "radial_gittins",
                    "budget_fraction": x,
                    "mean_hv_regret": mean,
                    "ci95_half_width": ci,
                    "n_runs": int(count),
                    "cost_reference_usd": cost_reference,
                }
            )

        random_results = run_budget_sweep(
            list(radial_runs[0].model_results[i].model_name for i in range(len(raw_vectors))),
            list(evaluation_questions),
            load_table_from_pickle(pickle_path),
            seeds=run_seeds,
        )
        random_styles = {
            "random_configurations": ("#7b61a8", "Random configurations"),
            "random_questions": ("#2a9d8f", "Random shared questions"),
        }
        for version, (color, label) in random_styles.items():
            xs, means, ci95 = _random_regret_series(
                random_results,
                version=version,
                normalized_truth=normalized_truth,
                ground_truth_hv=ground_truth_hv,
            )
            plot_label = (
                f"{label} (seed={args.seed})" if args.seeds == 1 else label
            )
            ax.plot(
                xs,
                means,
                color=color,
                marker="o",
                linewidth=1.5,
                label=plot_label,
            )
            ax.fill_between(
                xs,
                np.maximum(0.0, means - ci95),
                means + ci95,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            for x, mean, ci in zip(xs, means, ci95):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": version,
                        "budget_fraction": x,
                        "mean_hv_regret": mean,
                        "ci95_half_width": ci,
                        "n_runs": args.seeds,
                        "cost_reference_usd": cost_reference,
                    }
                )

        ax.set_title(benchmark)
        ax.set_xlabel("Fraction of brute-force search cost")
        ax.set_ylabel("Hypervolume regret")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)

    figure.suptitle("Multi-objective recommendation quality vs search budget", fontsize=13)
    figure.tight_layout()
    output_path = outdir / "radial_gittins_vs_random_search_hv_regret.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    csv_path = outdir / "method_hv_regret_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")


def load_table_from_pickle(pickle_path: Path):
    """Load only the lookup table while keeping the plotting loop readable."""
    from experiments.single_objective.offline_selector_sim import load_pickle

    _, _, table = load_pickle(str(pickle_path))
    return table


if __name__ == "__main__":
    main()
