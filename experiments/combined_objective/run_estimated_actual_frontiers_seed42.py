#!/usr/bin/env python3
"""Run one Seed-42 method/dataset cell for estimated-vs-actual appendices."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH
from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.compare_lcb_recommendations import (
    compact_lcb_run,
)
from experiments.combined_objective.estimated_actual_frontiers import (
    TARGET_FRACTIONS,
    build_frontier_artifact,
)
from experiments.combined_objective.offline_cost_budget_random_search import (
    RANDOM_CONFIGURATIONS,
    RANDOM_QUESTIONS,
    simulate_cost_budget_random_search,
)
from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_pareto_baselines import (
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.combined_objective.offline_radial_gittins import (
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR,
    simulate_radial_gittins,
)
from experiments.combined_objective.run_two_direction_ablation import (
    PAIRS,
    load_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "analysis/estimated_actual_frontiers_seed42"
SEED = 42
DATASETS = ("mathqa", "stackoverflow")
METHODS = (
    "radial_gittins",
    EGE_SH,
    APE_K,
    QNEHVI,
    RANDOM_CONFIGURATIONS,
    RANDOM_QUESTIONS,
)
PAIR = "gauss_radau_accuracy_endpoint"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def _source_metadata() -> dict[str, Any]:
    sources = (
        Path(__file__),
        ROOT / "experiments/combined_objective/estimated_actual_frontiers.py",
        ROOT / "experiments/combined_objective/offline_pareto_baselines.py",
        ROOT / "experiments/combined_objective/offline_cost_budget_random_search.py",
        ROOT / "experiments/combined_objective/offline_radial_gittins.py",
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    return {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_sha256": digest.hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _gittins_snapshots(run: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for point in run["points"]:
        yield {
            "event": point["event"],
            "role": point["snapshot_role"],
            "cumulative_evaluations": point["evaluations"],
            "selected_arm_indices": point["selected_arm_indices"],
        }


def _baseline_snapshots(result: Any) -> Iterable[dict[str, Any]]:
    for point in result.recommendation_trajectory:
        if point.event in {"recommendation_initial", "recommendation_changed"}:
            yield {
                "event": point.event,
                "role": point.event,
                "cumulative_evaluations": point.cumulative_evaluations,
                "selected_arm_indices": point.selected_arm_indices,
            }
    yield {
        "event": "terminal",
        "role": "terminal",
        "cumulative_evaluations": result.total_evaluations,
        "selected_arm_indices": result.selected_arm_indices,
    }


def _parse_indices(value: object) -> tuple[int, ...]:
    text = str(value)
    return tuple(int(item) for item in text.split(";") if item)


def _random_snapshots(result: Any) -> Iterable[dict[str, Any]]:
    for point in result.trajectory:
        if point["event"] in {"recommendation_initial", "recommendation_changed"}:
            yield {
                "event": point["event"],
                "role": point["event"],
                "cumulative_evaluations": point["cumulative_evaluations"],
                "selected_arm_indices": _parse_indices(
                    point["selected_arm_indices"]
                ),
            }
    yield {
        "event": "terminal",
        "role": "terminal",
        "cumulative_evaluations": result.summary["total_evaluations"],
        "selected_arm_indices": result.summary["selected_arm_indices"],
    }


def _run_gittins(
    dataset: str,
    models: list[str],
    questions: list[int],
    table: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = simulate_radial_gittins(
        models,
        questions,
        table,
        directions=PAIRS[PAIR],
        anytime=True,
        direction_scheduler="round_robin",
        eta_decay_schedule="direction_stop",
        recommendation_rule="finite_lcb",
        recommendation_beta=1.0,
        recommendation_min_samples=0,
        question_order="independent",
        warm_start_question_order="independent",
        cost_model="raw_mean",
        seed=SEED,
        batch_size=4,
        warm_start_batch_size=4,
        lambda_initial=1.0,
        lambda_decay=0.5,
        search_cost_scale_eta=1.0,
        observation_budget_fraction=1.0,
        boundary_z_padding_extra=2.0,
        effective_cost_bin_ratio=2.0,
        boundary_grid=RadialGittinsGrid(
            z_size=129,
            delta_size=129,
            state_size=129,
            boundary_margin_cells=4,
        ),
        boundary_cache=RadialGittinsBoundaryCache(
            cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR
        ),
        record_trace=True,
        record_recommendation_trajectory=True,
        recommendation_checkpoint_interval=None,
        recommendation_changes_only=True,
        defer_recommendation_diagnostics=True,
    )
    run = compact_lcb_run(result)
    artifact = build_frontier_artifact(
        dataset=dataset,
        method="radial_gittins",
        seed=SEED,
        models=models,
        truth_raw_vectors=result.raw_truth_vectors,
        full_data_pareto_arm_indices=run["full_data_pareto_arm_indices"],
        full_search_cost_usd=float(result.params["bruteforce_search_cost_usd"]),
        trace=result.trace,
        snapshots=_gittins_snapshots(run),
        cost_estimator="frozen_warm_arm_mean",
        expected_batch_costs_usd=result.params["expected_batch_costs_usd"],
        nominal_batch_size=int(result.params["batch_size"]),
    )
    details = {
        "selector": "two-direction Gauss-Radau radial Gittins",
        "pair": PAIR,
        "directions": [list(direction) for direction in PAIRS[PAIR]],
        "stop_reason": result.stop_reason,
        "total_evaluations": result.total_evaluations,
        "total_search_cost_usd": result.total_cost,
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "parameters": result.params,
    }
    return artifact, details


def _run_pareto(
    dataset: str,
    method: str,
    models: list[str],
    questions: list[int],
    table: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = simulate_pareto_baseline(
        models,
        questions,
        table,
        method=method,
        seed=SEED,
        batch_size=4,
        observation_budget_fraction=1.0,
        recommendation_checkpoint_interval=25,
        recommendation_cost_checkpoint_fractions=TARGET_FRACTIONS,
        qnehvi_mc_samples=64,
        qnehvi_refit_every=32,
        qnehvi_candidate_batch_size=512,
        stop_after_cost_checkpoint_fraction=(0.30 if method == QNEHVI else None),
    )
    truth = np.asarray(result.truth_raw_vectors, dtype=np.float64)
    artifact = build_frontier_artifact(
        dataset=dataset,
        method=method,
        seed=SEED,
        models=models,
        truth_raw_vectors=truth,
        full_data_pareto_arm_indices=pareto_min_cost_indices(truth),
        full_search_cost_usd=float(result.params["bruteforce_search_cost_usd"]),
        trace=result.physical_trace,
        snapshots=_baseline_snapshots(result),
        cost_estimator="causal_observed_arm_mean",
    )
    details = {
        "selector": method,
        "stop_reason": result.stop_reason,
        "total_evaluations": result.total_evaluations,
        "total_search_cost_usd": result.total_search_cost_usd,
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "parameters": result.params,
    }
    return artifact, details


def _run_random(
    dataset: str,
    method: str,
    models: list[str],
    questions: list[int],
    table: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = simulate_cost_budget_random_search(
        models,
        questions,
        table,
        method=method,
        seed=SEED,
        cost_checkpoint_fractions=TARGET_FRACTIONS,
    )
    truth = np.asarray(result.truth_raw_vectors, dtype=np.float64)
    artifact = build_frontier_artifact(
        dataset=dataset,
        method=method,
        seed=SEED,
        models=models,
        truth_raw_vectors=truth,
        full_data_pareto_arm_indices=pareto_min_cost_indices(truth),
        full_search_cost_usd=float(
            result.summary["params"]["bruteforce_search_cost_usd"]
        ),
        trace=result.physical_trace,
        snapshots=_random_snapshots(result),
        cost_estimator="causal_observed_arm_mean",
    )
    details = {
        "selector": method,
        "stop_reason": result.summary["stop_reason"],
        "total_evaluations": result.summary["total_evaluations"],
        "total_search_cost_usd": result.summary["total_search_cost_usd"],
        "parameters": result.summary["params"],
    }
    return artifact, details


def run_one(dataset: str, method: str, output_root: Path) -> Path:
    output_dir = output_root / "runs" / dataset / method / f"seed-{SEED}"
    event_path = output_dir / "frontier_events.json.gz"
    summary_path = output_dir / "summary.json"
    if event_path.is_file() and summary_path.is_file():
        print(f"SKIP complete output: {output_dir}", flush=True)
        return event_path

    models, datapoints, table, input_hashes = load_benchmark(dataset)
    questions = list(common_question_ids(models, datapoints, table))
    if method == "radial_gittins":
        artifact, details = _run_gittins(dataset, models, questions, table)
    elif method in {EGE_SH, APE_K, QNEHVI}:
        artifact, details = _run_pareto(
            dataset, method, models, questions, table
        )
    else:
        artifact, details = _run_random(
            dataset, method, models, questions, table
        )

    metadata = _source_metadata()
    artifact["input_sha256"] = input_hashes
    artifact["run_details"] = details
    artifact["source"] = metadata
    checkpoints = artifact["cost_checkpoints"]
    summary = {
        "dataset": dataset,
        "method": method,
        "seed": SEED,
        "frontier_change_frames": len(artifact["frontier_change_frames"]),
        "cost_estimator": artifact["cost_estimator"],
        "checkpoint_10pct": checkpoints["10pct"],
        "checkpoint_30pct": checkpoints["30pct"],
        "run_details": details,
        "source": metadata,
    }
    _atomic_gzip_json(event_path, artifact)
    _atomic_json(summary_path, summary)
    print(
        f"WROTE {event_path} "
        f"10%={checkpoints['10pct']['actual_search_cost_percent']:.3f}% "
        f"30%={checkpoints['30pct']['actual_search_cost_percent']:.3f}% "
        f"frames={len(artifact['frontier_change_frames'])}",
        flush=True,
    )
    return event_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=SEED, choices=(SEED,))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    run_one(args.dataset, args.method, args.output_root)


if __name__ == "__main__":
    main()
