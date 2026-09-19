#!/usr/bin/env python3
"""Run one method/dataset/seed cell for the unified 20-seed benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_multiobjective_random_search import (
    DEFAULT_BUDGET_FRACTIONS,
    VERSIONS as RANDOM_METHODS,
    normalized_truth_vectors,
    run_budget_sweep,
)
from experiments.combined_objective.offline_pareto_baselines import (
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.single_objective.offline_selector_sim import load_scope


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "analysis/final_run_baselines"
DATASETS = (
    "hotpotqa",
    "mathqa",
    "restaurant_test",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
)
PARETO_METHODS = (EGE_SH, EGE_SR, APE_K, QNEHVI)
METHODS = PARETO_METHODS + RANDOM_METHODS


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


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("trajectory must contain at least one row")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_vectors(
    estimated_raw: np.ndarray,
    truth_raw: np.ndarray,
    truth_normalized: np.ndarray,
    path: Path,
) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        estimated_raw_vectors=estimated_raw,
        truth_raw_vectors=truth_raw,
        truth_vectors=truth_normalized,
    )
    os.replace(temporary, path)


def _dataset_path(dataset: str) -> Path:
    if dataset in {"hotpotqa", "mathqa"}:
        return ROOT / "data" / dataset
    return ROOT / "data/scope" / dataset


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
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    return {
        "source_commit": commit,
        "source_sha256": digest.hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _run_pareto(
    dataset: str,
    method: str,
    seed: int,
    models: list[str],
    questions: list[int],
    table: Any,
    qnehvi_candidate_batch_size: Optional[int],
) -> tuple[dict[str, object], list[dict[str, object]], tuple[np.ndarray, ...]]:
    result = simulate_pareto_baseline(
        models,
        questions,
        table,
        method=method,
        seed=seed,
        batch_size=4,
        observation_budget_fraction=1.0,
        recommendation_checkpoint_interval=25,
        qnehvi_mc_samples=64,
        qnehvi_refit_every=32,
        qnehvi_candidate_batch_size=qnehvi_candidate_batch_size,
    )
    rows = [
        {
            "benchmark": dataset,
            "method": method,
            "seed": seed,
            "event": point.event,
            "budget_fraction": point.budget_fraction,
            "cumulative_evaluations": point.cumulative_evaluations,
            "cumulative_search_cost_usd": point.cumulative_search_cost_usd,
            "hypervolume": point.hypervolume,
            "hv_regret": point.hypervolume_regret,
            "generational_distance": point.generational_distance,
            "inverted_generational_distance": point.inverted_generational_distance,
            "archive_size": len(point.selected_arm_indices),
            "selected_arm_indices": ";".join(
                str(index) for index in point.selected_arm_indices
            ),
            "selected_models": ";".join(point.selected_models),
        }
        for point in result.recommendation_trajectory
    ]
    summary: dict[str, object] = {
        "benchmark": dataset,
        "method": method,
        "seed": seed,
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
        "completed_arm_indices": list(result.completed_arm_indices),
        "completed_models": list(result.completed_models),
        "completed_pareto_arm_indices": list(result.completed_pareto_arm_indices),
        "completed_pareto_models": list(result.completed_pareto_models),
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "trajectory_points": len(rows),
    }
    vectors = (
        result.estimated_raw_vectors,
        result.truth_raw_vectors,
        result.truth_vectors,
    )
    return summary, rows, vectors


def _run_random(
    dataset: str,
    method: str,
    seed: int,
    models: list[str],
    questions: list[int],
    table: Any,
) -> tuple[dict[str, object], list[dict[str, object]], tuple[np.ndarray, ...]]:
    started = time.perf_counter()
    results = run_budget_sweep(
        models,
        questions,
        table,
        versions=(method,),
        budget_fractions=DEFAULT_BUDGET_FRACTIONS,
        seeds=(seed,),
    )
    rows = [
        {
            "benchmark": dataset,
            "method": method,
            "seed": seed,
            "event": "random_budget",
            "budget_fraction": result.budget_fraction,
            "cumulative_evaluations": result.total_evaluations,
            "cumulative_search_cost_usd": result.total_search_cost_usd,
            "hypervolume": result.hypervolume,
            "hv_regret": result.hypervolume_regret,
            "generational_distance": result.generational_distance,
            "inverted_generational_distance": result.inverted_generational_distance,
            "archive_size": len(result.selected_arm_indices),
            "selected_arm_indices": ";".join(
                str(index) for index in result.selected_arm_indices
            ),
            "selected_models": ";".join(result.selected_models),
        }
        for result in results
    ]
    final = results[-1]
    summary: dict[str, object] = {
        "benchmark": dataset,
        "method": method,
        "seed": seed,
        "params": {"budget_fractions": list(DEFAULT_BUDGET_FRACTIONS)},
        "stop_reason": "budget_sweep_complete",
        "total_evaluations": final.total_evaluations,
        "total_search_cost_usd": final.total_search_cost_usd,
        "hypervolume": final.hypervolume,
        "ground_truth_hypervolume": final.ground_truth_hypervolume,
        "hypervolume_regret": final.hypervolume_regret,
        "generational_distance": final.generational_distance,
        "inverted_generational_distance": final.inverted_generational_distance,
        "true_front_recall": final.true_front_recall,
        "recommendation_precision": final.recommendation_precision,
        "false_positive_count": final.false_positive_count,
        "selected_arm_indices": list(final.selected_arm_indices),
        "selected_models": list(final.selected_models),
        "completed_arm_indices": list(final.completed_arm_indices),
        "completed_models": list(final.completed_models),
        "completed_pareto_arm_indices": list(final.completed_pareto_arm_indices),
        "completed_pareto_models": list(final.completed_pareto_models),
        "policy_wall_time_seconds": time.perf_counter() - started,
        "trajectory_points": len(rows),
    }
    truth_normalized = normalized_truth_vectors(
        final.truth_vectors, final.cost_reference_usd,
    )
    vectors = (final.estimated_vectors, final.truth_vectors, truth_normalized)
    return summary, rows, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--qnehvi-candidate-batch-size",
        type=int,
        default=int(os.environ.get("QNEHVI_CANDIDATE_BATCH_SIZE", "512")),
        help="Candidates per acquisition call; use 0 for one full batch",
    )
    args = parser.parse_args()
    if args.seed not in range(42, 62):
        parser.error("--seed must be in [42, 61]")

    output_dir = args.output_root / args.dataset / args.method / f"seed-{args.seed}"
    required = tuple(
        output_dir / name
        for name in ("summary.json", "vectors.npz", "cost_trajectory.csv")
    )
    if all(path.is_file() for path in required):
        print(f"SKIP complete output: {output_dir}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata = _source_metadata()
    models, questions, table = load_scope(str(_dataset_path(args.dataset)))
    if args.method in PARETO_METHODS:
        summary, rows, vectors = _run_pareto(
            args.dataset,
            args.method,
            args.seed,
            models,
            questions,
            table,
            (
                None
                if args.qnehvi_candidate_batch_size == 0
                else args.qnehvi_candidate_batch_size
            ),
        )
    else:
        summary, rows, vectors = _run_random(
            args.dataset, args.method, args.seed, models, questions, table,
        )
    summary.update(source_metadata)
    _atomic_csv(rows, output_dir / "cost_trajectory.csv")
    _atomic_vectors(*vectors, output_dir / "vectors.npz")
    _atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
