#!/usr/bin/env python3
"""Plot the radial-Gittins recommendation trajectory.

Runs HotpotQA / MathQA offline replay at budget fraction 1.0, continues past
the endogenous Gittins stop so the trajectory covers later budgets, and writes:

* one HV-regret, GD, and IGD curve per benchmark against cumulative search
  cost as a fraction of brute-force spend, dashed while the trajectory is the
  all-posterior provisional diagnostic and solid once it becomes the
  completed-only deployable recommendation at the Gittins stop;
* a raw-archive comparison at the Gittins stop, at 50% of brute-force search
  cost, and at the actual end fraction.

Each checkpoint carries a single archive whose scope it reports.  Before the
policy stops nothing is deployable, so the curve tracks the provisional
archive; from the stop onward it is the recommendation contract and unfinished
winners are never promoted into it.  The replay continues after the stop only
to show counterfactual fixed-budget diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _save_rgb_png(figure, output_path: Path, *, dpi: int = 160) -> None:
    """Write an opaque RGB PNG. RGBA files fail Cursor's image preview."""
    from io import BytesIO

    output_path = Path(output_path)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(output_path, format="PNG")
    print(f"wrote {output_path}")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins import (  # noqa: E402
    DEFAULT_DIRECTIONS,
)
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


def _scope_label(checkpoint: RecommendationCheckpoint) -> str:
    return (
        "completed-only deployable"
        if checkpoint.is_deployable
        else "all-posterior provisional"
    )


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


def run_benchmark(
    *,
    pickle_path: Path,
    name: str,
    batch_size: int,
    seed: int,
    grid_size: int,
    directions: Sequence[Sequence[float]],
    eta: float,
    boundary_z_padding_extra: float,
    cache: RadialGittinsBoundaryCache,
    boundary_margin_cells: int | None = None,
) -> Tuple[RadialSimulationResult, np.ndarray, Tuple[int, ...]]:
    models, datapoints, table = load_pickle(pickle_path)
    grid = RadialGittinsGrid(
        z_size=grid_size,
        delta_size=grid_size,
        state_size=grid_size,
        boundary_margin_cells=(
            max(2, min(4, grid_size // 32))
            if boundary_margin_cells is None
            else boundary_margin_cells
        ),
    )
    print(f"\n=== {name}: {len(models)} models, seed={seed}, grid={grid_size} ===")
    result = simulate_radial_gittins(
        models,
        datapoints,
        table,
        batch_size=batch_size,
        directions=directions,
        search_cost_scale_eta=eta,
        boundary_z_padding_extra=boundary_z_padding_extra,
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
        f"final_cost_frac={result.total_cost / result.params['bruteforce_search_cost_usd']:.3f}, "
        f"final_eval_frac={result.total_evaluations / result.params['available_cells_in_universe']:.3f}, "
        f"stop_reason={result.stop_reason}, "
        f"traj_points={len(result.recommendation_trajectory)}, "
        f"policy_s={result.policy_wall_time_seconds:.1f}"
    )
    return result, raw_vectors, eval_dps


TRAJECTORY_METRICS = (
    {
        "field": "hypervolume_regret",
        "ylabel": "Normalized-desirability hypervolume regret",
        "handover": "deployable HV regret at handover",
        "filename": "hv_regret_curves.png",
    },
    {
        "field": "generational_distance",
        "ylabel": "Generational distance (GD)",
        "handover": "deployable GD at handover",
        "filename": "gd_curves.png",
    },
    {
        "field": "inverted_generational_distance",
        "ylabel": "Inverted generational distance (IGD)",
        "handover": "deployable IGD at handover",
        "filename": "igd_curves.png",
    },
)


def _plot_scope_series(
    ax,
    result: RadialSimulationResult,
    *,
    field: str,
    ylabel: str,
    handover_label: str,
    show_legend: bool = True,
) -> None:
    traj = result.recommendation_trajectory
    # The scope switches exactly once, so these are a prefix and a suffix.
    # They are drawn as separate segments because the archive itself
    # changes at the handover; joining them would imply a continuous curve.
    provisional = [p for p in traj if not p.is_deployable]
    deployable = [p for p in traj if p.is_deployable]
    if provisional:
        ax.plot(
            [p.budget_fraction for p in provisional],
            [getattr(p, field) for p in provisional],
            color="#7a8ca5",
            linewidth=1.3,
            linestyle="--",
            label="provisional all-posterior (pre-stop diagnostic)",
        )
    if len(deployable) > 1:
        ax.plot(
            [p.budget_fraction for p in deployable],
            [getattr(p, field) for p in deployable],
            color="#1f4e79",
            linewidth=1.9,
            label="deployable completed-only recommendation",
        )
    if deployable:
        handover = deployable[0]
        value = getattr(handover, field)
        ax.scatter(
            [handover.budget_fraction],
            [value],
            s=95,
            color="#1f4e79",
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label=f"{handover_label} ({value:.4f})",
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
    ax.set_xlabel("Fraction of brute-force search cost")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)


def plot_hv_curves(
    results: Dict[str, RadialSimulationResult],
    output_path: Path,
) -> None:
    plot_metric_curves(
        results,
        output_path,
        field="hypervolume_regret",
        ylabel="Normalized-desirability hypervolume regret",
        handover_label="deployable regret at handover",
        title=(
            "One recommendation trajectory per benchmark: all-posterior "
            "diagnostic before the Gittins stop, completed-only after"
        ),
    )


def plot_metric_curves(
    results: Dict[str, RadialSimulationResult],
    output_path: Path,
    *,
    field: str,
    ylabel: str,
    handover_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(5.25 * len(results), 4.0),
        sharey=False,
        squeeze=False,
    )
    for ax, (name, result) in zip(axes[0], results.items()):
        _plot_scope_series(
            ax,
            result,
            field=field,
            ylabel=ylabel,
            handover_label=handover_label,
        )
        ax.set_title(name)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_rgb_png(fig, output_path, dpi=160)
    plt.close(fig)


def plot_front_quality_curves(
    results: Dict[str, RadialSimulationResult],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(TRAJECTORY_METRICS),
        len(results),
        figsize=(5.25 * len(results), 3.55 * len(TRAJECTORY_METRICS)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    names = list(results)
    for row, spec in enumerate(TRAJECTORY_METRICS):
        for col, name in enumerate(names):
            ax = axes[row, col]
            _plot_scope_series(
                ax,
                results[name],
                field=spec["field"],
                ylabel=spec["ylabel"],
                handover_label=spec["handover"],
                show_legend=(row == 0 and col == len(names) - 1),
            )
            if row == 0:
                ax.set_title(name)
            if row < len(TRAJECTORY_METRICS) - 1:
                ax.set_xlabel("")
    fig.suptitle(
        "One recommendation trajectory per benchmark: HV regret, GD, and IGD",
        fontsize=12,
    )
    fig.tight_layout()
    _save_rgb_png(fig, output_path, dpi=160)
    plt.close(fig)


def plot_raw_archive_comparison(
    *,
    name: str,
    result: RadialSimulationResult,
    raw_vectors: np.ndarray,
    output_path: Path,
) -> None:
    """Compare each snapshot's online and full-data raw archive.

    Every panel reports the scope its checkpoint recorded, so the pre-stop
    provisional snapshots and the post-stop deployable ones stay legible as
    the different objects they are.
    """
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
    estimated_vectors = [
        np.asarray(
            checkpoint.estimated_raw_winner_vectors, dtype=np.float64
        ).reshape((-1, 2))
        for checkpoint in snapshots.values()
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
        scope_label = _scope_label(checkpoint)
        candidate_arms = list(checkpoint.direction_winner_arm_indices)
        estimated_points = estimated_vectors[panel_index]
        online = set(checkpoint.online_raw_archive_arm_indices)
        oracle = set(checkpoint.oracle_raw_winner_archive_arm_indices)
        overlap = sorted(online & oracle)
        online_only = sorted(online - oracle)
        oracle_only = sorted(oracle - online)

        online_ax.scatter(
            estimated_points[:, 1],
            estimated_points[:, 0],
            s=26,
            c="#b8c8db",
            alpha=0.75,
            label="direction winners",
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
                label="online archive",
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
        f"{name}: recorded raw-space archives\n"
        "Top: values available online. Bottom: membership at full-data values.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    _save_rgb_png(fig, output_path, dpi=180)
    plt.close(fig)


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
        "--eta",
        type=float,
        default=1.0,
        help="USD-to-normalized-utility search-cost scale (default: 1.0)",
    )
    parser.add_argument(
        "--extra-direction",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("ACCURACY", "COST_DESIRABILITY"),
        help=(
            "Append a positive simplex direction to the nine defaults; may be "
            "specified more than once"
        ),
    )
    parser.add_argument(
        "--boundary-z-padding-extra",
        type=float,
        default=0.0,
        help="Extra z-grid guard band for low-cost or extreme directions",
    )
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
    directions = tuple(DEFAULT_DIRECTIONS) + tuple(
        tuple(float(component) for component in direction)
        for direction in args.extra_direction
    )

    cache = RadialGittinsBoundaryCache()
    results: Dict[str, RadialSimulationResult] = {}
    raw_by_name: Dict[str, np.ndarray] = {}
    summary = {}
    cost_trajectory_rows = []

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
            directions=directions,
            eta=args.eta,
            boundary_z_padding_extra=args.boundary_z_padding_extra,
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
            "eta": args.eta,
            "directions": [list(direction) for direction in directions],
            "boundary_z_padding_extra": args.boundary_z_padding_extra,
            "stop_reason": result.stop_reason,
            "gittins_stop_budget_fraction": result.gittins_stop_budget_fraction,
            "gittins_stop_evaluations": result.gittins_stop_evaluations,
            "gittins_stop_cost_usd": result.gittins_stop_cost_usd,
            "final_evaluations": result.total_evaluations,
            "final_cost_usd": result.total_cost,
            "available_cells": result.params["available_cells_in_universe"],
            "bruteforce_search_cost_usd": result.params["bruteforce_search_cost_usd"],
            "final_hv_regret": (
                final_checkpoint.hypervolume_regret
                if final_checkpoint is not None
                else result.hypervolume_regret
            ),
            "final_generational_distance": (
                final_checkpoint.generational_distance
                if final_checkpoint is not None
                else result.generational_distance
            ),
            "final_inverted_generational_distance": (
                final_checkpoint.inverted_generational_distance
                if final_checkpoint is not None
                else result.inverted_generational_distance
            ),
            "final_archive_scope": (
                final_checkpoint.archive_scope
                if final_checkpoint is not None
                else None
            ),
            "gittins_stop_hv_regret": (
                stop_checkpoint.hypervolume_regret
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_generational_distance": (
                stop_checkpoint.generational_distance
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_inverted_generational_distance": (
                stop_checkpoint.inverted_generational_distance
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_archive_scope": (
                stop_checkpoint.archive_scope
                if stop_checkpoint is not None
                else None
            ),
            "gittins_stop_models": (
                list(stop_checkpoint.online_raw_archive_models)
                if stop_checkpoint is not None
                else None
            ),
            "completed_arm_archive_hv_regret": result.hypervolume_regret,
            "completed_arm_archive_gd": result.generational_distance,
            "completed_arm_archive_igd": result.inverted_generational_distance,
            "trajectory_len": len(result.recommendation_trajectory),
            "policy_wall_time_seconds": result.policy_wall_time_seconds,
            "final_online_raw_models": result.online_raw_archive_models,
            "final_oracle_raw_winner_models": (
                result.oracle_raw_winner_archive_models
            ),
            "final_posterior_archive_models": result.posterior_archive_models,
        }
        for checkpoint in result.recommendation_trajectory:
            cost_trajectory_rows.append(
                {
                    "benchmark": display,
                    "event": checkpoint.event,
                    "budget_fraction": checkpoint.budget_fraction,
                    "cumulative_evaluations": checkpoint.cumulative_evaluations,
                    "cumulative_search_cost_usd": checkpoint.cumulative_search_cost_usd,
                    "archive_scope": checkpoint.archive_scope,
                    "hv_regret": checkpoint.hypervolume_regret,
                    "generational_distance": checkpoint.generational_distance,
                    "inverted_generational_distance": (
                        checkpoint.inverted_generational_distance
                    ),
                    "archive_size": len(checkpoint.online_raw_archive_models),
                }
            )
        plot_raw_archive_comparison(
            name=display,
            result=result,
            raw_vectors=raw_vectors,
            output_path=outdir / f"{bench}_raw_archive_comparison.png",
        )

    for spec in TRAJECTORY_METRICS:
        plot_metric_curves(
            results,
            outdir / spec["filename"],
            field=spec["field"],
            ylabel=spec["ylabel"],
            handover_label=spec["handover"],
            title=(
                "One recommendation trajectory per benchmark: all-posterior "
                "diagnostic before the Gittins stop, completed-only after"
            ),
        )
    plot_front_quality_curves(results, outdir / "front_quality_curves.png")
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    trajectory_path = outdir / "cost_trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(cost_trajectory_rows)
    print(f"wrote {trajectory_path}")


if __name__ == "__main__":
    main()
