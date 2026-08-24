#!/usr/bin/env python3
"""Plot radial-Gittins recommendation and provisional-archive diagnostics.

Runs HotpotQA / MathQA offline replay at budget fraction 1.0, continues past
the endogenous Gittins stop so the trajectory covers later budgets, and writes:

* the completed-only deployable HV regret from the Gittins stop onward, marked
  at the stop itself, alongside the all-posterior provisional HV regret curve;
* separate deployable and provisional raw-archive comparisons at Gittins
  stop, 50%, and the actual end fraction.

The replay continues after the endogenous stop only to show counterfactual
fixed-budget diagnostics.  The deployable archive is the terminal
recommendation contract; unfinished provisional winners are never promoted to
that archive.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    RadialSimulationResult,
    RecommendationCheckpoint,
    _full_raw_objective_vectors,
    _require_data_path,
    load_pickle,
    raw_nondominated_indices,
    simulate_radial_gittins,
)


SNAPSHOT_FRACTIONS = (0.5,)


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


def _gittins_stop_index(result: RadialSimulationResult) -> Optional[int]:
    """Index of the first checkpoint at or after the endogenous Gittins stop."""
    trajectory = result.recommendation_trajectory
    for index, point in enumerate(trajectory):
        if point.event == "gittins_stop":
            return index
    fraction = result.gittins_stop_budget_fraction
    if fraction is None:
        return None
    for index, point in enumerate(trajectory):
        if point.budget_fraction + 1e-12 >= fraction:
            return index
    return None


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
    snapshots["End"] = trajectory[-1]
    return snapshots


def _pareto_min_cost_indices(points: np.ndarray) -> List[int]:
    """Nondominated under maximize accuracy / minimize cost."""
    return raw_nondominated_indices(points)


def _checkpoint_archive_view(
    checkpoint: RecommendationCheckpoint,
    archive_kind: str,
) -> Tuple[
    Tuple[int, ...],
    Tuple[Tuple[float, float], ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """Return aligned winners, estimates, online archive, and oracle archive."""
    if archive_kind == "deployable":
        return (
            checkpoint.deployable_direction_winner_arm_indices,
            checkpoint.deployable_estimated_raw_winner_vectors,
            checkpoint.deployable_online_raw_archive_arm_indices,
            checkpoint.deployable_oracle_raw_winner_archive_arm_indices,
        )
    if archive_kind == "provisional":
        return (
            checkpoint.direction_winner_arm_indices,
            checkpoint.estimated_raw_winner_vectors,
            checkpoint.online_raw_archive_arm_indices,
            checkpoint.oracle_raw_winner_archive_arm_indices,
        )
    raise ValueError("archive_kind must be 'deployable' or 'provisional'")


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
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(5.25 * len(results), 4.0),
        sharey=False,
        squeeze=False,
    )
    for ax, (name, result) in zip(axes[0], results.items()):
        traj = result.recommendation_trajectory
        xs = [p.budget_fraction for p in traj]
        provisional_ys = [p.hypervolume_regret for p in traj]
        ax.plot(
            xs,
            provisional_ys,
            color="#7a8ca5",
            linewidth=1.3,
            linestyle="--",
            label="provisional all-posterior",
        )
        # The deployable archive is only a recommendation once the policy has
        # stopped, so pre-stop values are not a regret anyone would incur.
        stop_index = _gittins_stop_index(result)
        if stop_index is None:
            ax.plot(
                xs,
                [p.deployable_hypervolume_regret for p in traj],
                color="#1f4e79",
                linewidth=1.9,
                label="deployable completed-only (no endogenous stop)",
            )
        else:
            post_stop = traj[stop_index:]
            stop_point = traj[stop_index]
            if len(post_stop) > 1:
                ax.plot(
                    [p.budget_fraction for p in post_stop],
                    [p.deployable_hypervolume_regret for p in post_stop],
                    color="#1f4e79",
                    linewidth=1.9,
                    label="deployable completed-only (post-stop)",
                )
            ax.scatter(
                [stop_point.budget_fraction],
                [stop_point.deployable_hypervolume_regret],
                s=95,
                color="#1f4e79",
                marker="o",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
                label=(
                    "deployable regret at stop "
                    f"({stop_point.deployable_hypervolume_regret:.4f})"
                ),
            )
        if result.gittins_stop_budget_fraction is not None:
            ax.axvspan(
                result.gittins_stop_budget_fraction,
                1.02,
                color="#687386",
                alpha=0.08,
                label="forced post-stop diagnostic",
            )
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
        ax.set_ylabel("Normalized-desirability hypervolume regret")
        ax.set_xlim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Completed-only recommendation from the Gittins stop onward "
        "vs all-posterior diagnostic",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_raw_archive_comparison(
    *,
    name: str,
    result: RadialSimulationResult,
    raw_vectors: np.ndarray,
    archive_kind: str,
    output_path: Path,
) -> None:
    """Compare one explicitly scoped online and full-data raw archive."""
    if archive_kind not in {"deployable", "provisional"}:
        raise ValueError("archive_kind must be 'deployable' or 'provisional'")
    is_deployable = archive_kind == "deployable"
    scope_label = (
        "completed-only deployable" if is_deployable else "all-posterior provisional"
    )
    snapshots = _select_snapshots(result)
    labels = list(snapshots.keys())
    fig, axes = plt.subplots(
        2,
        len(labels),
        figsize=(4.5 * len(labels), 7.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    global_front_idx = _pareto_min_cost_indices(raw_vectors)
    global_front = raw_vectors[global_front_idx]
    global_front = global_front[np.argsort(global_front[:, 1])]
    archive_views = [
        _checkpoint_archive_view(checkpoint, archive_kind)
        for checkpoint in snapshots.values()
    ]
    estimated_vectors = [
        np.asarray(view[1], dtype=np.float64).reshape((-1, 2))
        for view in archive_views
    ]
    all_costs = np.concatenate(
        [raw_vectors[:, 1]]
        + [points[:, 1] for points in estimated_vectors if len(points)]
    )
    positive_costs = all_costs[all_costs > 0.0]
    use_log_cost = (
        len(positive_costs) == len(all_costs)
        and float(np.max(positive_costs) / np.min(positive_costs)) >= 20.0
    )

    for panel_index, label in enumerate(labels):
        checkpoint = snapshots[label]
        online_ax = axes[0, panel_index]
        truth_ax = axes[1, panel_index]
        archive_view = archive_views[panel_index]
        candidate_arms = list(archive_view[0])
        estimated_points = estimated_vectors[panel_index]
        online = set(archive_view[2])
        oracle = set(archive_view[3])
        overlap = sorted(online & oracle)
        online_only = sorted(online - oracle)
        oracle_only = sorted(oracle - online)

        online_ax.scatter(
            estimated_points[:, 1],
            estimated_points[:, 0],
            s=26,
            c="#b8c8db",
            alpha=0.75,
            label=f"{scope_label} direction winners",
            zorder=1,
        )
        online_positions = [
            position
            for position, arm_index in enumerate(candidate_arms)
            if arm_index in online
        ]
        if online_positions:
            online_points = estimated_points[online_positions]
            order = np.argsort(online_points[:, 1])
            online_points = online_points[order]
            online_ax.plot(
                online_points[:, 1],
                online_points[:, 0],
                color="#1f4e79",
                linewidth=1.5,
                marker="s",
                markersize=6,
                label=f"{scope_label} online archive",
                zorder=3,
            )

        truth_ax.scatter(
            raw_vectors[:, 1],
            raw_vectors[:, 0],
            s=16,
            c="#cbd2dc",
            alpha=0.55,
            label="all configurations at full-data values",
            zorder=1,
        )
        truth_ax.plot(
            global_front[:, 1],
            global_front[:, 0],
            color="#687386",
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label="global full-data raw front",
            zorder=2,
        )

        groups = (
            (overlap, "#2f855a", "*", 105, "in both archives"),
            (online_only, "#1f4e79", "s", 62, "online estimated only"),
            (oracle_only, "#c45c26", "D", 58, "offline oracle only"),
        )
        for arm_indices, color, marker, size, legend_label in groups:
            if not arm_indices:
                continue
            points = raw_vectors[arm_indices]
            truth_ax.scatter(
                points[:, 1],
                points[:, 0],
                s=size,
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.7,
                label=legend_label,
                zorder=4,
            )

        # Label only disagreements; shared labels add clutter without helping
        # explain the difference between the two filters.
        for arm_index in online_only + oracle_only:
            truth_ax.annotate(
                _short_model(result.model_results[arm_index].model_name),
                (raw_vectors[arm_index, 1], raw_vectors[arm_index, 0]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color="#263445",
            )

        if use_log_cost:
            online_ax.set_xscale("log")
            truth_ax.set_xscale("log")
        online_ax.set_title(
            f"{label} ({checkpoint.budget_fraction:.1%})\n"
            f"{scope_label} archive={len(online)}"
        )
        truth_ax.set_title(
            f"Full-data evaluation: oracle={len(oracle)}, "
            f"overlap={len(overlap)}"
        )
        truth_ax.set_xlabel(
            "Mean deployment cost (USD, log scale)"
            if use_log_cost
            else "Mean deployment cost (USD)"
        )
        online_ax.set_ylabel("Accuracy (online estimate)")
        truth_ax.set_ylabel("Accuracy (full data)")
        online_ax.grid(True, alpha=0.25)
        truth_ax.grid(True, alpha=0.25)

    legend_entries: Dict[str, object] = {}
    for ax in axes.flat:
        handles, labels_for_axis = ax.get_legend_handles_labels()
        for handle, legend_label in zip(handles, labels_for_axis):
            legend_entries.setdefault(legend_label, handle)
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(4, max(1, len(legend_entries))),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"{name}: {scope_label} raw-space archives\n"
        "Top: values available online. Bottom: membership at full-data values.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/combined_objective/results/radial_gittins_plots"),
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
            str(ROOT / "experiments/data/lookup" / f"{bench}_lookup.pkl")
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
        stop_checkpoints = [
            checkpoint
            for checkpoint in result.recommendation_trajectory
            if checkpoint.event == "gittins_stop"
        ]
        stop_checkpoint = stop_checkpoints[0] if stop_checkpoints else None
        final_checkpoint = (
            result.recommendation_trajectory[-1]
            if result.recommendation_trajectory
            else None
        )
        summary[display] = {
            "stop_reason": result.stop_reason,
            "gittins_stop_budget_fraction": result.gittins_stop_budget_fraction,
            "gittins_stop_evaluations": result.gittins_stop_evaluations,
            "final_evaluations": result.total_evaluations,
            "available_cells": result.params["available_cells_in_universe"],
            "final_hv_regret": (
                final_checkpoint.deployable_hypervolume_regret
                if final_checkpoint is not None
                else result.hypervolume_regret
            ),
            "final_deployable_hv_regret": (
                final_checkpoint.deployable_hypervolume_regret
                if final_checkpoint is not None
                else result.hypervolume_regret
            ),
            "final_provisional_hv_regret": (
                final_checkpoint.hypervolume_regret
                if final_checkpoint is not None
                else None
            ),
            "gittins_stop_deployable_hv_regret": (
                stop_checkpoint.deployable_hypervolume_regret
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_provisional_hv_regret": (
                stop_checkpoint.hypervolume_regret
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_deployable_models": (
                list(stop_checkpoint.deployable_online_raw_archive_models)
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_provisional_models": (
                list(stop_checkpoint.online_raw_archive_models)
                if stop_checkpoint is not None
                else None
            ),
            "completed_arm_archive_hv_regret": result.hypervolume_regret,
            "trajectory_len": len(result.recommendation_trajectory),
            "policy_wall_time_seconds": result.policy_wall_time_seconds,
            "final_online_raw_models": result.online_raw_archive_models,
            "final_oracle_raw_winner_models": (
                result.oracle_raw_winner_archive_models
            ),
            "final_posterior_archive_models": result.posterior_archive_models,
        }
        plot_raw_archive_comparison(
            name=display,
            result=result,
            raw_vectors=raw_vectors,
            archive_kind="deployable",
            output_path=outdir / f"{bench}_raw_archive_comparison.png",
        )
        plot_raw_archive_comparison(
            name=display,
            result=result,
            raw_vectors=raw_vectors,
            archive_kind="provisional",
            output_path=(
                outdir / f"{bench}_provisional_raw_archive_comparison.png"
            ),
        )

    plot_hv_curves(results, outdir / "hv_regret_curves.png")
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
