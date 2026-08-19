#!/usr/bin/env python3
"""Plot radial-Gittins HV regret curves and Pareto snapshots.

Runs HotpotQA / MathQA offline replay at budget fraction 1.0, continues past
the endogenous Gittins stop so the trajectory covers later budgets, and writes:

* HV regret vs observed budget fraction (with a stop marker)
* 2-D Pareto snapshots at Gittins stop, 50%, and end
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from offline_radial_gittins import (  # noqa: E402
    RadialSimulationResult,
    RecommendationCheckpoint,
    _full_raw_objective_vectors,
    _require_data_path,
    load_pickle,
    nondominated_indices,
    simulate_radial_gittins,
)


SNAPSHOT_FRACTIONS = (0.5, 1.0)


def _short_model(name: str) -> str:
    name = name.replace("planner=", "P:").replace("solver=", "S:")
    if len(name) <= 42:
        return name
    return name[:39] + "..."


def _checkpoint_at_or_after(
    trajectory: Sequence[RecommendationCheckpoint],
    fraction: float,
) -> RecommendationCheckpoint:
    for point in trajectory:
        if point.budget_fraction + 1e-12 >= fraction:
            return point
    return trajectory[-1]


def _select_snapshots(
    result: RadialSimulationResult,
) -> Dict[str, RecommendationCheckpoint]:
    trajectory = result.recommendation_trajectory
    if not trajectory:
        raise ValueError("result has no recommendation trajectory")
    snapshots: Dict[str, RecommendationCheckpoint] = {}
    stop_points = [p for p in trajectory if p.event == "gittins_stop"]
    if stop_points:
        snapshots["Gittins stop"] = stop_points[0]
    elif result.gittins_stop_budget_fraction is not None:
        snapshots["Gittins stop"] = _checkpoint_at_or_after(
            trajectory,
            result.gittins_stop_budget_fraction,
        )
    for fraction in SNAPSHOT_FRACTIONS:
        label = f"{fraction:.0%}"
        snapshots[label] = _checkpoint_at_or_after(trajectory, fraction)
    return snapshots


def _pareto_min_cost_indices(points: np.ndarray) -> List[int]:
    """Nondominated under maximize accuracy / minimize cost."""
    flipped = np.column_stack([points[:, 0], -points[:, 1]])
    return nondominated_indices(flipped)


def run_benchmark(
    *,
    pickle_path: Path,
    name: str,
    batch_size: int,
    seed: int,
    grid_size: int,
    cache: RadialGittinsBoundaryCache,
) -> Tuple[RadialSimulationResult, np.ndarray, Tuple[int, ...]]:
    models, datapoints, table = load_pickle(pickle_path)
    grid = RadialGittinsGrid(
        z_size=grid_size,
        delta_size=grid_size,
        state_size=grid_size,
        boundary_margin_cells=max(2, min(4, grid_size // 32)),
    )
    print(f"\n=== {name}: {len(models)} models, seed={seed}, grid={grid_size} ===")
    result = simulate_radial_gittins(
        models,
        datapoints,
        table,
        batch_size=batch_size,
        observation_budget_fraction=1.0,
        seed=seed,
        boundary_grid=grid,
        boundary_cache=cache,
        halt_on_gittins_stop=False,
        record_recommendation_trajectory=True,
        question_universe="common",
    )
    eval_dps = tuple(
        sorted(
            set.intersection(
                *[
                    {q for q in datapoints if q in table.get(model, {})}
                    for model in models
                ]
            )
        )
    )
    raw_vectors = _full_raw_objective_vectors(models, eval_dps, table)
    print(
        f"stop_marker={result.gittins_stop_budget_fraction}, "
        f"final_frac={result.total_evaluations / result.params['available_cells_in_universe']:.3f}, "
        f"stop_reason={result.stop_reason}, "
        f"traj_points={len(result.recommendation_trajectory)}, "
        f"policy_s={result.policy_wall_time_seconds:.1f}"
    )
    return result, raw_vectors, eval_dps


def plot_hv_curves(
    results: Dict[str, RadialSimulationResult],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, (name, result) in zip(axes, results.items()):
        traj = result.recommendation_trajectory
        xs = [p.budget_fraction for p in traj]
        ys = [p.hypervolume_regret for p in traj]
        ax.plot(xs, ys, color="#1f4e79", linewidth=1.8, label="HV regret")
        if result.gittins_stop_budget_fraction is not None:
            ax.axvline(
                result.gittins_stop_budget_fraction,
                color="#c45c26",
                linestyle="--",
                linewidth=1.4,
                label=(
                    f"Gittins stop "
                    f"({result.gittins_stop_budget_fraction:.1%})"
                ),
            )
        ax.set_title(name)
        ax.set_xlabel("Observed budget fraction")
        ax.set_ylabel("Hypervolume regret")
        ax.set_xlim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Radial-Gittins recommendation quality vs search budget",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_pareto_snapshots(
    *,
    name: str,
    result: RadialSimulationResult,
    raw_vectors: np.ndarray,
    output_path: Path,
) -> None:
    snapshots = _select_snapshots(result)
    labels = list(snapshots.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.2), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    true_front_idx = _pareto_min_cost_indices(raw_vectors)
    true_front = raw_vectors[true_front_idx]
    true_front = true_front[np.argsort(true_front[:, 1])]

    for ax, label in zip(axes, labels):
        checkpoint = snapshots[label]
        ax.scatter(
            raw_vectors[:, 1],
            raw_vectors[:, 0],
            s=14,
            c="#b8c0cc",
            alpha=0.55,
            label="all configs",
            zorder=1,
        )
        ax.plot(
            true_front[:, 1],
            true_front[:, 0],
            color="#6b7280",
            linewidth=1.2,
            marker="o",
            markersize=3.5,
            label="true Pareto front",
            zorder=2,
        )
        if checkpoint.selected_arm_indices:
            selected = raw_vectors[list(checkpoint.selected_arm_indices)]
            order = np.argsort(selected[:, 1])
            selected = selected[order]
            ax.plot(
                selected[:, 1],
                selected[:, 0],
                color="#1f4e79",
                linewidth=1.8,
                marker="s",
                markersize=6,
                label="recommended set",
                zorder=3,
            )
            for arm_index in checkpoint.selected_arm_indices:
                ax.annotate(
                    _short_model(
                        result.model_results[arm_index].model_name
                        if arm_index < len(result.model_results)
                        else str(arm_index)
                    ),
                    (
                        raw_vectors[arm_index, 1],
                        raw_vectors[arm_index, 0],
                    ),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6,
                    color="#1f4e79",
                )
        ax.set_title(
            f"{label}\n"
            f"budget={checkpoint.budget_fraction:.1%}, "
            f"HV regret={checkpoint.hypervolume_regret:.4f}"
        )
        ax.set_xlabel("Mean deployment cost (USD)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"{name}: recommended Pareto snapshots", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/results/radial_gittins_plots"),
    )
    parser.add_argument("--batch-size", type=int, default=4, choices=(4, 8))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=129,
        help="Boundary DP grid resolution (129 is much faster than production 513)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["hotpotqa", "mathqa"],
        choices=["hotpotqa", "mathqa"],
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cache = RadialGittinsBoundaryCache()
    results: Dict[str, RadialSimulationResult] = {}
    raw_by_name: Dict[str, np.ndarray] = {}
    summary = {}

    for bench in args.benchmarks:
        pickle_path = _require_data_path(
            f"experiments/results/cache_db_results/{bench}_lookup.pkl"
        )
        display = "HotpotQA" if bench == "hotpotqa" else "MathQA"
        result, raw_vectors, _ = run_benchmark(
            pickle_path=Path(pickle_path),
            name=display,
            batch_size=args.batch_size,
            seed=args.seed,
            grid_size=args.grid_size,
            cache=cache,
        )
        results[display] = result
        raw_by_name[display] = raw_vectors
        summary[display] = {
            "stop_reason": result.stop_reason,
            "gittins_stop_budget_fraction": result.gittins_stop_budget_fraction,
            "gittins_stop_evaluations": result.gittins_stop_evaluations,
            "final_evaluations": result.total_evaluations,
            "available_cells": result.params["available_cells_in_universe"],
            "final_hv_regret": result.recommendation_trajectory[-1].hypervolume_regret
            if result.recommendation_trajectory
            else result.hypervolume_regret,
            "completed_arm_archive_hv_regret": result.hypervolume_regret,
            "trajectory_len": len(result.recommendation_trajectory),
            "policy_wall_time_seconds": result.policy_wall_time_seconds,
        }
        plot_pareto_snapshots(
            name=display,
            result=result,
            raw_vectors=raw_vectors,
            output_path=outdir / f"{bench}_pareto_snapshots.png",
        )

    plot_hv_curves(results, outdir / "hv_regret_curves.png")
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
