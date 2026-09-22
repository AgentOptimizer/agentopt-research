#!/usr/bin/env python3
"""Run one Pareto-baseline dataset/seed and retain 10%-spaced cost checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_pareto_baselines import (
    DEFAULT_COST_CHECKPOINT_FRACTIONS,
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.single_objective.offline_selector_sim import load_scope


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "analysis/final_run_baselines"
DATASETS = (
    "hotpotqa", "mathqa", "restaurant_test", "stackoverflow", "bird_dev",
    "restaurant_valid", "bing_querylogs", "bird_mini_dev",
)
METHODS = (EGE_SH, EGE_SR, APE_K, QNEHVI)
EXPECTED_EVENTS = {
    f"cost_checkpoint_{int(round(100 * value))}pct"
    for value in DEFAULT_COST_CHECKPOINT_FRACTIONS
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    os.replace(temporary, path)


def _atomic_csv(rows: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_vectors(result: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        estimated_raw_vectors=result.estimated_raw_vectors,
        truth_raw_vectors=result.truth_raw_vectors,
        truth_vectors=result.truth_vectors,
    )
    os.replace(temporary, path)


def _dataset_path(dataset: str) -> Path:
    return ROOT / "data" / dataset if dataset in {"hotpotqa", "mathqa"} else ROOT / "data/scope" / dataset


def _has_all_checkpoints(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = {row.get("event", "") for row in rows}
    if EXPECTED_EVENTS.issubset(events):
        return True
    # Older full trajectories recorded every recommendation change rather
    # than named cost-threshold events. They still define the exact current
    # recommendation at a target as the latest change at or before it.
    selected_rows = [row for row in rows if "selected_arm_indices" in row]
    return all(
        any(float(row["budget_fraction"]) <= target + 1e-12 for row in selected_rows)
        for target in DEFAULT_COST_CHECKPOINT_FRACTIONS
    )


def _source_metadata() -> dict[str, object]:
    paths = (
        Path(__file__),
        ROOT / "experiments/combined_objective/offline_pareto_baselines.py",
        ROOT / "src/agentopt/model_selection/pareto_identification.py",
        ROOT / "src/agentopt/model_selection/qnehvi.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_sha256": digest.hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=range(42, 62))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--qnehvi-candidate-batch-size", type=int,
        default=int(os.environ.get("QNEHVI_CANDIDATE_BATCH_SIZE", "64")),
    )
    args = parser.parse_args()

    output_dir = args.output_root / args.dataset / args.method / f"seed-{args.seed}"
    trajectory_path = output_dir / "cost_trajectory.csv"
    required = (output_dir / "summary.json", output_dir / "vectors.npz")
    if all(path.is_file() for path in required) and _has_all_checkpoints(trajectory_path):
        print(f"SKIP complete 10%-spaced checkpoints: {output_dir}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    models, questions, table = load_scope(str(_dataset_path(args.dataset)))
    result = simulate_pareto_baseline(
        models,
        questions,
        table,
        method=args.method,
        seed=args.seed,
        batch_size=4,
        observation_budget_fraction=1.0,
        recommendation_checkpoint_interval=25,
        recommendation_cost_checkpoint_fractions=DEFAULT_COST_CHECKPOINT_FRACTIONS,
        qnehvi_mc_samples=64,
        qnehvi_refit_every=32,
        qnehvi_candidate_batch_size=(
            None if args.qnehvi_candidate_batch_size == 0
            else args.qnehvi_candidate_batch_size
        ),
    )
    rows = [
        {
            "benchmark": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "event": point.event,
            "budget_fraction": point.budget_fraction,
            "cumulative_evaluations": point.cumulative_evaluations,
            "cumulative_search_cost_usd": point.cumulative_search_cost_usd,
            "hypervolume": point.hypervolume,
            "hv_regret": point.hypervolume_regret,
            "generational_distance": point.generational_distance,
            "inverted_generational_distance": point.inverted_generational_distance,
            "archive_size": len(point.selected_arm_indices),
            "selected_arm_indices": ";".join(map(str, point.selected_arm_indices)),
            "selected_models": ";".join(point.selected_models),
        }
        for point in result.recommendation_trajectory
    ]
    summary: dict[str, object] = {
        "benchmark": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "params": result.params,
        "stop_reason": result.stop_reason,
        "total_evaluations": result.total_evaluations,
        "total_search_cost_usd": result.total_search_cost_usd,
        "hypervolume": result.hypervolume,
        "ground_truth_hypervolume": result.ground_truth_hypervolume,
        "hypervolume_regret": result.hypervolume_regret,
        "generational_distance": result.generational_distance,
        "inverted_generational_distance": result.inverted_generational_distance,
        "true_front_recall": result.true_front_recall,
        "recommendation_precision": result.recommendation_precision,
        "false_positive_count": result.false_positive_count,
        "selected_arm_indices": list(result.selected_arm_indices),
        "selected_models": list(result.selected_models),
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "trajectory_points": len(rows),
        **_source_metadata(),
    }
    _atomic_csv(rows, trajectory_path)
    _atomic_vectors(result, output_dir / "vectors.npz")
    _atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
