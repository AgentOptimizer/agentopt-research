#!/usr/bin/env python3
"""Run one canonical three-objective ablation configuration/benchmark/seed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.offline_three_objective_gittins import (
    simulate_three_objective_gittins,
)
from experiments.combined_objective.run_three_objective_gittins import summarize
from experiments.combined_objective.run_three_objective_gittins_multiseed import (
    STORAGE_SCHEMA,
    compact_payload,
    dump_json,
)
from experiments.combined_objective.three_objective_ablation import (
    BENCHMARKS,
    CONFIGURATIONS,
    DEFAULT_OUTPUT_ROOT,
    FAMILIES,
    RUN_CONFIGURATIONS,
    SEEDS,
    result_path,
    task_count,
    task_mapping,
)
from experiments.combined_objective.three_objective_metrics import (
    enrich_run,
    load_three_objective_benchmark,
)


SOURCE_FILES = (
    "experiments/combined_objective/offline_three_objective_gittins.py",
    "experiments/combined_objective/run_three_objective_ablation_task.py",
    "experiments/combined_objective/three_objective_ablation.py",
    "experiments/combined_objective/three_objective_metrics.py",
    "experiments/combined_objective/offline_radial_gittins.py",
    "src/agentopt/model_selection/radial_gittins.py",
    "src/agentopt/model_selection/radial_gittins_dp.py",
    "src/agentopt/model_selection/axis_gittins_dp.py",
)


def source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def family_names(configuration: str) -> list[str]:
    return [family for family, members in FAMILIES.items() if configuration in members]


def expected_config(
    configuration: str,
    benchmark: str,
    seed: int,
    input_hashes: dict[str, str],
    observation_budget_fraction: float = 1.0,
    quality_prior_variance_mode: str = "warm",
) -> dict[str, Any]:
    settings = CONFIGURATIONS[configuration]
    return {
        "schema_version": 1,
        "ablation_configuration": configuration,
        "ablation_families": family_names(configuration),
        "label": settings["label"],
        "benchmark": benchmark,
        "seed": seed,
        "objective_order": ["Q", "L", "D"],
        "raw_objectives": [
            "accuracy_max",
            "mean_latency_seconds_min",
            "mean_cost_usd_min",
        ],
        "direction_set": settings["direction_set"],
        "directions": [list(direction) for direction in settings["directions"]],
        "direction_scheduler": "round_robin",
        "round_robin_order": [list(direction) for direction in settings["directions"]],
        "acquisition_cost_mode": settings["acquisition_cost_mode"],
        "actual_budget_accounting": "realized_API_cost_USD",
        "continuation_mode": settings["continuation_mode"],
        "eta_initial": 1.0,
        "eta_decay": 0.5,
        "eta_decay_schedule": (
            "direction_stop"
            if settings["continuation_mode"] == "eta_decay"
            else "disabled"
        ),
        "batch_size": 4,
        "warm_start_batch_size": 4,
        "question_order": "independent",
        "warm_start_question_order": "independent",
        "recommendation_rule": "finite_lcb",
        "recommendation_beta": 1.0,
        "latency_model": "raw_mean",
        "cost_model": "raw_mean",
        "prior_variance_modes": [quality_prior_variance_mode, "warm", "warm"],
        "grid_size": 129,
        "observation_budget_fraction": observation_budget_fraction,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes(),
    }


def validate_saved(payload: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    if payload.get("config") != expected:
        raise ValueError(f"Refusing incompatible saved result: {path}")
    if payload.get("storage", {}).get("schema") != STORAGE_SCHEMA:
        raise ValueError(f"Unexpected storage schema: {path}")
    run = payload.get("run", {})
    if not run.get("points") or run.get("total_evaluations") != run["points"][-1].get("total_evaluations"):
        raise ValueError(f"Incomplete saved result: {path}")


def slurm_metadata() -> dict[str, str]:
    names = (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_TIMELIMIT",
        "SLURM_JOB_NODELIST",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def run_one(configuration: str, benchmark: str, seed: int, output_root: Path,
            observation_budget_fraction: float = 1.0,
            quality_prior_variance_mode: str = "warm") -> dict[str, Any]:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {configuration}")
    if benchmark not in BENCHMARKS or seed not in SEEDS:
        raise ValueError("benchmark or seed is outside the canonical protocol")
    models, questions, table, input_hashes = load_three_objective_benchmark(benchmark)
    expected = expected_config(configuration, benchmark, seed, input_hashes,
                               observation_budget_fraction,
                               quality_prior_variance_mode)
    output = result_path(configuration, benchmark, seed, output_root)
    if output.exists():
        payload = json.loads(output.read_text())
        validate_saved(payload, expected, output)
        print(f"Reusing {output}", flush=True)
        return {
            "status": "reused",
            "path": str(output),
            "configuration": configuration,
            "benchmark": benchmark,
            "seed": seed,
        }

    settings = CONFIGURATIONS[configuration]
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    last_progress = started
    print(
        f"Running {configuration} / {benchmark} / seed {seed}; "
        f"{len(settings['directions'])} equal-round-robin directions",
        flush=True,
    )

    def progress(event: dict[str, Any]) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if now - last_progress >= 30.0:
            print(
                f"  {event['total_evaluations']} observations, "
                f"${event['total_cost']:.4f}, {now-started:.0f}s",
                flush=True,
            )
            last_progress = now

    run = simulate_three_objective_gittins(
        models,
        questions,
        table,
        seed=seed,
        directions=settings["directions"],
        batch_size=4,
        warm_start_batch_size=4,
        grid_size=129,
        question_order="independent",
        warm_start_question_order="independent",
        lambda_initial=1.0,
        lambda_decay=0.5,
        acquisition_cost_mode=settings["acquisition_cost_mode"],
        continuation_mode=settings["continuation_mode"],
        recommendation_beta=1.0,
        quality_prior_variance_mode=quality_prior_variance_mode,
        observation_budget_fraction=observation_budget_fraction,
        progress_callback=progress,
    )
    simulation_seconds = time.perf_counter() - started
    run = enrich_run(run)
    run["selector"] = f"three_objective_cc_gittins_{configuration}"
    dense_payload = {
        "config": expected,
        "timing": {
            "started_at_utc": started_at,
            "simulation_seconds": simulation_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "summary": summarize(run),
        "run": run,
        "provenance": {
            "source_commit": os.environ.get("AGENTOPT_SOURCE_COMMIT"),
            "slurm": slurm_metadata(),
        },
    }
    payload = compact_payload(dense_payload)
    dump_json(output, payload)
    validate_saved(payload, expected, output)
    print(
        f"Finished {configuration} / {benchmark} / seed {seed}: "
        f"{simulation_seconds:.1f}s; {output}",
        flush=True,
    )
    return {
        "status": "completed",
        "path": str(output),
        "configuration": configuration,
        "benchmark": benchmark,
        "seed": seed,
    }


def write_manifest(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id in range(task_count()):
        configuration, benchmark, seed = task_mapping(task_id)
        tasks.append({
            "task_id": task_id,
            "configuration": configuration,
            "benchmark": benchmark,
            "seed": seed,
            "result_path": str(result_path(configuration, benchmark, seed, output_root)),
        })
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_order": ["Q", "L", "D"],
        "benchmarks": list(BENCHMARKS),
        "seeds": list(SEEDS),
        "families": {key: list(value) for key, value in FAMILIES.items()},
        "configurations": CONFIGURATIONS,
        "scheduler_policy": "equal round-robin in the listed direction order",
        "scheduler_ablation_included": False,
        "task_count": task_count(),
        "one_array_task_per_run": True,
        "source_sha256": source_hashes(),
        "source_commit": os.environ.get("AGENTOPT_SOURCE_COMMIT"),
        "tasks": tasks,
    }
    path = output_root / "manifest.json"
    dump_json(path, manifest)
    print(path)
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--task-id", type=int)
    result.add_argument("--configuration", choices=RUN_CONFIGURATIONS)
    result.add_argument("--benchmark", choices=BENCHMARKS)
    result.add_argument("--seed", type=int, choices=SEEDS)
    result.add_argument("--quality-prior-variance-mode", choices=("fixed", "warm"),
                        default="warm")
    result.add_argument("--write-manifest", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    output_root = args.output_root.resolve()
    if args.write_manifest:
        write_manifest(output_root)
        return
    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is not None:
        configuration, benchmark, seed = task_mapping(task_id)
    elif args.configuration is not None and args.benchmark is not None and args.seed is not None:
        configuration, benchmark, seed = args.configuration, args.benchmark, args.seed
    else:
        raise SystemExit(
            "provide --task-id (or SLURM_ARRAY_TASK_ID), or configuration/benchmark/seed"
        )
    mapping = {
        "configuration": configuration,
        "benchmark": benchmark,
        "seed": seed,
        "output": str(result_path(configuration, benchmark, seed, output_root)),
    }
    if args.dry_run:
        print(json.dumps(mapping, indent=2))
        return
    run_one(configuration, benchmark, seed, output_root,
            quality_prior_variance_mode=args.quality_prior_variance_mode)


if __name__ == "__main__":
    main()
