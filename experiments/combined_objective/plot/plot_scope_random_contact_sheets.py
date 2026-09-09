#!/usr/bin/env python3
"""Plot six SCOPE datasets in one 6-by-10 random-search contact sheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    VERSIONS,
    pareto_min_cost_indices,
    run_budget_sweep,
)
from experiments.combined_objective.plot.plot_multiobjective_random_search import (  # noqa: E402
    _version_title,
)
from experiments.single_objective.offline_selector_sim import load_scope  # noqa: E402


BENCHMARKS = (
    ("bing_querylogs", "Bing Query Logs"),
    ("bird_dev", "BIRD Dev"),
    ("bird_mini_dev", "BIRD Mini Dev"),
    ("restaurant_test", "Restaurant Test"),
    ("restaurant_valid", "Restaurant Valid"),
    ("stackoverflow", "Stack Overflow"),
)
PLOT_BUDGET_FRACTIONS = (0.05, 0.10, 0.30, 0.50, 0.70, 0.90)


def _legend_handles():
    return [
        Line2D([], [], linestyle="", marker="o", markersize=7,
               color="#c5cad3", label="all configurations"),
        Line2D([], [], color="#626b78", label="brute-force Pareto frontier"),
        Line2D([], [], linestyle="", marker="*", markersize=15,
               color="#2f855a", label="recommended and on true front"),
        Line2D([], [], linestyle="", marker="s", markersize=9,
               color="#1f4e79", label="recommended but dominated"),
        Line2D([], [], linestyle="", marker="D", markersize=9,
               color="#c45c26", label="true-front point not recommended"),
    ]


def write_contact_sheet(all_results, *, version, output_path, seed):
    fig, axes = plt.subplots(6, 6, figsize=(24, 23), squeeze=False)
    for row, (_, title) in enumerate(BENCHMARKS):
        results = sorted(
            all_results[title], key=lambda result: result.budget_fraction
        )
        truth = results[0].truth_vectors
        true_front_indices = set(pareto_min_cost_indices(truth))
        ordered_front = sorted(
            true_front_indices, key=lambda index: truth[index, 1]
        )
        for column, result in enumerate(results):
            ax = axes[row, column]
            selected = set(result.selected_arm_indices)
            correct = sorted(selected & true_front_indices)
            false_positive = sorted(selected - true_front_indices)
            missed = sorted(true_front_indices - selected)
            ax.scatter(
                truth[:, 1], truth[:, 0], s=18, color="#c5cad3",
                alpha=0.55, edgecolors="none", zorder=1,
            )
            ax.plot(
                truth[ordered_front, 1], truth[ordered_front, 0],
                color="#626b78", linewidth=1.0, zorder=2,
            )
            for indices, color, marker, size in (
                (correct, "#2f855a", "*", 180),
                (false_positive, "#1f4e79", "s", 75),
                (missed, "#c45c26", "D", 70),
            ):
                if indices:
                    points = truth[indices]
                    ax.scatter(
                        points[:, 1], points[:, 0], color=color,
                        marker=marker, s=size, edgecolors="white",
                        linewidths=0.5, zorder=4,
                    )
            ax.set_title(
                f"{result.budget_fraction:.0%}\n"
                f"regret={result.hypervolume_regret:.4f}, "
                f"FP={result.false_positive_count}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.25)
            if column == 0:
                ax.set_ylabel(f"{title}\nMean accuracy", fontsize=10)
            if row == len(BENCHMARKS) - 1:
                ax.set_xlabel("Mean deployment cost (USD)", fontsize=9)
    fig.legend(
        handles=_legend_handles(), loc="lower center", ncol=5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"SCOPE — {_version_title(version)} — seed={seed}", fontsize=18
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.98), h_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", type=Path, default=ROOT / "data/scope")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "experiments/combined_objective/results/scope_random",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    all_results = {version: {} for version in VERSIONS}
    for slug, title in BENCHMARKS:
        print(f"running {title}")
        models, datapoints, table = load_scope(str(args.scope_dir / slug))
        results = run_budget_sweep(
            models,
            datapoints,
            table,
            budget_fractions=PLOT_BUDGET_FRACTIONS,
            seeds=(args.seed,),
        )
        for version in VERSIONS:
            all_results[version][title] = [
                result for result in results if result.version == version
            ]
    for version in VERSIONS:
        write_contact_sheet(
            all_results[version],
            version=version,
            output_path=args.outdir / f"scope_{version}_seed-{args.seed}.png",
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
