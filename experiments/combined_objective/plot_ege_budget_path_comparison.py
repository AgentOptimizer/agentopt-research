#!/usr/bin/env python3
"""Compare EGE-SH's 100%-budget prefix with a direct 10%-budget run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    mean_raw_vectors,
)
from experiments.combined_objective.offline_pareto_baselines import (
    EGE_SH,
    simulate_pareto_baseline,
)
from experiments.combined_objective.plot_paper_20seed_comparison import (
    LABELS,
    PICKLES,
    _draw_landscape,
)
from experiments.single_objective.offline_selector_sim import load_pickle


DEFAULT_BENCHMARKS = ("hotpotqa", "mathqa")


def _plot(
    output: Path,
    panels: list[tuple[str, np.ndarray, tuple[int, ...], int, float]],
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(11.5, 4.8), squeeze=False)
    for ax, (name, truth, selected, evaluations, cost_fraction) in zip(axes[0], panels):
        _draw_landscape(ax, truth)
        points = truth[list(selected)]
        ax.scatter(
            points[:, 1], points[:, 0], s=115, color="#d55e00",
            edgecolors="#725b46", linewidths=0.9, zorder=5,
        )
        ax.set_title(name, fontsize=18)
        ax.tick_params(axis="both", labelsize=12)
        ax.text(
            0.97, 0.04,
            f"{len(selected)} recommended\n{evaluations:,} observations\n"
            f"actual search cost: {cost_fraction:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        )
    figure.suptitle(title, fontsize=19, y=0.98)
    figure.supxlabel("Mean deployment cost (USD)", fontsize=16, y=0.03)
    figure.supylabel("Mean accuracy", fontsize=16, x=0.015)
    figure.tight_layout(rect=(0.04, 0.08, 0.995, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"wrote {output}")


def _plot_benchmark_comparison(
    output: Path,
    *,
    benchmark: str,
    prefix: tuple[str, np.ndarray, tuple[int, ...], int, float],
    direct: tuple[str, np.ndarray, tuple[int, ...], int, float],
    seed: int,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True, squeeze=False,
    )
    subtitles = (
        "100%-budget run at 10% observations",
        "Direct 10%-budget run",
    )
    for ax, panel, subtitle in zip(axes[0], (prefix, direct), subtitles):
        _, truth, selected, evaluations, cost_fraction = panel
        _draw_landscape(ax, truth)
        points = truth[list(selected)]
        ax.scatter(
            points[:, 1], points[:, 0], s=120, color="#d55e00",
            edgecolors="#725b46", linewidths=0.9, zorder=5,
        )
        ax.set_title(subtitle, fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.text(
            0.97, 0.04,
            f"{len(selected)} recommended\n{evaluations:,} observations\n"
            f"actual search cost: {cost_fraction:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        )
    figure.suptitle(f"{benchmark}: EGE-SH budget-path comparison (seed {seed})",
                    fontsize=19, y=0.98)
    figure.supxlabel("Mean deployment cost (USD)", fontsize=16, y=0.03)
    figure.supylabel("Mean accuracy", fontsize=16, x=0.015)
    figure.tight_layout(rect=(0.04, 0.08, 0.995, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "analysis/paper_20seed_method_comparison/figures",
    )
    args = parser.parse_args()

    prefix_panels = []
    direct_panels = []
    for benchmark in DEFAULT_BENCHMARKS:
        models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
        questions = common_question_ids(models, datapoints, table)
        truth = mean_raw_vectors(models, questions, table)
        full = simulate_pareto_baseline(
            models, datapoints, table, method=EGE_SH, seed=args.seed,
            batch_size=args.batch_size, observation_budget_fraction=1.0,
        )
        direct = simulate_pareto_baseline(
            models, datapoints, table, method=EGE_SH, seed=args.seed,
            batch_size=args.batch_size, observation_budget_fraction=0.1,
        )
        target = int(np.floor(0.1 * len(models) * len(questions)))
        prefix = max(
            (point for point in full.recommendation_trajectory
             if point.cumulative_evaluations <= target),
            key=lambda point: point.cumulative_evaluations,
        )
        full_cost = float(full.params["bruteforce_search_cost_usd"])
        prefix_panels.append((
            LABELS[benchmark], truth, prefix.selected_arm_indices,
            prefix.cumulative_evaluations,
            prefix.cumulative_search_cost_usd / full_cost,
        ))
        direct_panels.append((
            LABELS[benchmark], truth, direct.selected_arm_indices,
            direct.total_evaluations,
            direct.total_search_cost_usd / full_cost,
        ))
        prefix_set = set(prefix.selected_models)
        direct_set = set(direct.selected_models)
        print(
            f"{LABELS[benchmark]}: prefix={len(prefix_set)}, direct={len(direct_set)}, "
            f"shared={len(prefix_set & direct_set)}, "
            f"prefix_only={len(prefix_set - direct_set)}, "
            f"direct_only={len(direct_set - prefix_set)}"
        )

    _plot(
        args.outdir / "ege_sh_100pct_run_at_10pct_frontier.png",
        prefix_panels,
        title=f"EGE-SH: 10% checkpoint from a 100%-budget run (seed {args.seed})",
    )
    _plot(
        args.outdir / "ege_sh_direct_10pct_budget_frontier.png",
        direct_panels,
        title=f"EGE-SH: direct 10%-budget run (seed {args.seed})",
    )
    for prefix, direct in zip(prefix_panels, direct_panels):
        slug = prefix[0].lower()
        _plot_benchmark_comparison(
            args.outdir / f"{slug}_ege_sh_10pct_budget_path_comparison.png",
            benchmark=prefix[0], prefix=prefix, direct=direct, seed=args.seed,
        )


if __name__ == "__main__":
    main()
