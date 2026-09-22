#!/usr/bin/env python3
"""Run one exact observed-cell-budget Pareto baseline task.

Unlike the cost-checkpoint runs, each task terminates at an exact fixed share
of the common model-by-question matrix.  The output records both the nominal
uniform-cell cost estimate and the realized lookup-table spend, together with
estimated and full-data objective vectors for every recommended arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_pareto_baselines import (
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.single_objective.offline_selector_sim import load_scope


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "analysis/eval_budget_results"
METHODS = (EGE_SH, EGE_SR, APE_K, QNEHVI)
DATASETS = (
    "hotpotqa", "mathqa", "restaurant_test", "stackoverflow", "bird_dev",
    "restaurant_valid", "bing_querylogs", "bird_mini_dev",
)
SEEDS = tuple(range(42, 62))
TARGETS = (0.10, 0.30)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_vectors(result: Any, selected: np.ndarray, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        selected_arm_indices=selected,
        selected_estimated_raw_vectors=result.estimated_raw_vectors[selected],
        selected_actual_raw_vectors=result.truth_raw_vectors[selected],
        estimated_raw_vectors=result.estimated_raw_vectors,
        truth_raw_vectors=result.truth_raw_vectors,
        truth_vectors=result.truth_vectors,
    )
    os.replace(temporary, path)


def _dataset_path(dataset: str) -> Path:
    return (
        ROOT / "data" / dataset
        if dataset in {"hotpotqa", "mathqa"}
        else ROOT / "data/scope" / dataset
    )


def _mapping(
    task_id: int,
    methods: Sequence[str],
    datasets: Sequence[str],
    targets: Sequence[float],
) -> tuple[str, str, int, float]:
    cells = [
        (method, dataset, seed, target)
        for method in methods
        for dataset in datasets
        for seed in SEEDS
        for target in targets
    ]
    if not 0 <= task_id < len(cells):
        raise ValueError(f"task id must be in [0, {len(cells) - 1}], got {task_id}")
    return cells[task_id]


def _complete(path: Path, target: float) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("budget_type") == "observed_cell_fraction"
        and math.isclose(float(payload.get("target_eval_fraction", -1.0)), target)
        and int(payload.get("total_evaluations", -1))
        == int(payload.get("target_evaluations", -2))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--targets", nargs="+", type=float, default=TARGETS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--qnehvi-candidate-batch-size",
        type=int,
        default=int(os.environ.get("QNEHVI_CANDIDATE_BATCH_SIZE", "64")),
    )
    args = parser.parse_args()
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    targets = tuple(float(value) for value in args.targets)
    if any(value not in TARGETS for value in targets):
        parser.error(f"--targets must be drawn from {TARGETS}")
    method, dataset, seed, target = _mapping(
        task_id, tuple(args.methods), tuple(args.datasets), targets
    )

    tag = f"{int(round(100 * target))}pct_eval"
    output_dir = args.output_root / tag / method / dataset / f"seed-{seed}"
    summary_path = output_dir / "summary.json"
    if _complete(summary_path, target):
        print(f"SKIP complete {summary_path}", flush=True)
        return

    models, datapoints, table = load_scope(str(_dataset_path(dataset)))
    result = simulate_pareto_baseline(
        models,
        datapoints,
        table,
        method=method,
        seed=seed,
        batch_size=4,
        observation_budget_fraction=target,
        record_recommendation_trajectory=False,
        recommendation_cost_checkpoint_fractions=(),
        qnehvi_mc_samples=64,
        qnehvi_refit_every=32,
        qnehvi_candidate_batch_size=(
            None
            if args.qnehvi_candidate_batch_size == 0
            else args.qnehvi_candidate_batch_size
        ),
    )
    full_cost = float(result.params["bruteforce_search_cost_usd"])
    cell_count = len(models) * int(result.params["common_question_count"])
    target_evaluations = max(1, int(math.floor(target * cell_count)))
    if result.total_evaluations != target_evaluations:
        raise RuntimeError(
            f"{method}/{dataset}/seed-{seed}: expected {target_evaluations} "
            f"evaluations, got {result.total_evaluations}"
        )
    actual_eval_fraction = result.total_evaluations / cell_count
    estimated_search_cost = actual_eval_fraction * full_cost
    actual_search_cost = float(result.total_search_cost_usd)
    selected = np.asarray(result.selected_arm_indices, dtype=np.int64)
    estimated_vectors = result.estimated_raw_vectors[selected]
    actual_vectors = result.truth_raw_vectors[selected]
    payload: dict[str, object] = {
        "schema_version": 1,
        "budget_type": "observed_cell_fraction",
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "target_eval_fraction": target,
        "cell_count": cell_count,
        "target_evaluations": target_evaluations,
        "total_evaluations": result.total_evaluations,
        "actual_eval_fraction": actual_eval_fraction,
        "full_matrix_search_cost_usd": full_cost,
        "estimated_search_cost_usd": estimated_search_cost,
        "estimated_cost_fraction": estimated_search_cost / full_cost,
        "estimated_search_cost_definition": (
            "actual_eval_fraction * full_matrix_search_cost_usd"
        ),
        "actual_search_cost_usd": actual_search_cost,
        "actual_cost_fraction": actual_search_cost / full_cost,
        "selected_arm_indices": list(result.selected_arm_indices),
        "selected_models": list(result.selected_models),
        "selected_estimated_raw_vectors": estimated_vectors,
        "selected_actual_raw_vectors": actual_vectors,
        "selected_estimated_deployment_cost_usd_per_query": (
            estimated_vectors[:, 1] if len(estimated_vectors) else []
        ),
        "selected_actual_deployment_cost_usd_per_query": (
            actual_vectors[:, 1] if len(actual_vectors) else []
        ),
        "stop_reason": result.stop_reason,
        "hypervolume": result.hypervolume,
        "ground_truth_hypervolume": result.ground_truth_hypervolume,
        "hypervolume_regret": result.hypervolume_regret,
        "generational_distance": result.generational_distance,
        "inverted_generational_distance": result.inverted_generational_distance,
        "true_front_recall": result.true_front_recall,
        "recommendation_precision": result.recommendation_precision,
        "false_positive_count": result.false_positive_count,
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "params": result.params,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_vectors(result, selected, output_dir / "vectors.npz")
    _atomic_json(payload, summary_path)
    print(json.dumps({
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "target_eval_fraction": target,
        "actual_eval_fraction": actual_eval_fraction,
        "estimated_search_cost_usd": estimated_search_cost,
        "actual_search_cost_usd": actual_search_cost,
        "actual_cost_fraction": actual_search_cost / full_cost,
        "output": str(summary_path),
    }), flush=True)


if __name__ == "__main__":
    main()
