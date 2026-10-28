#!/usr/bin/env python3
"""Run one three-objective direction task at a fixed 30% entry cap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any


for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.offline_three_objective_gittins import (
    simulate_three_objective_gittins,
)
from experiments.combined_objective.three_objective_ablation import (
    BENCHMARKS,
    CONFIGURATIONS,
    SEEDS,
)
from experiments.combined_objective.three_objective_metrics import (
    load_three_objective_benchmark,
)


RUNTIME_CONFIGURATIONS = (
    "q_only_real_cost",
    "l_only_real_cost",
    "d_only_real_cost",
    "pure_axes_real_cost",
    "four_direction_real_cost",
    "six_ql_dense_real_cost",
    "six_three_edges_real_cost",
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/yh6415/project/agentopt-research/analysis/"
    "usd_cost_checkpoints_latest_under_20seed/new_qa_figures/"
    "gittins_g2_main_figures/source_data/ablation/"
    "three_objective_direction_policy_runtime_fixed_30pct_entries_20seed"
)
OBSERVATION_BUDGET_FRACTION = 0.30
SOURCE_FILES = (
    "experiments/combined_objective/offline_three_objective_gittins.py",
    "experiments/combined_objective/run_three_objective_direction_runtime_task.py",
    "experiments/combined_objective/three_objective_ablation.py",
    "experiments/combined_objective/three_objective_metrics.py",
    "experiments/combined_objective/offline_radial_gittins.py",
    "src/agentopt/model_selection/radial_gittins.py",
    "src/agentopt/model_selection/radial_gittins_dp.py",
    "src/agentopt/model_selection/axis_gittins_dp.py",
)


def task_count() -> int:
    return len(RUNTIME_CONFIGURATIONS) * len(BENCHMARKS) * len(SEEDS)


def task_mapping(task_id: int) -> tuple[str, str, int]:
    if not 0 <= task_id < task_count():
        raise ValueError(f"task id must be in [0, {task_count()})")
    runs_per_configuration = len(BENCHMARKS) * len(SEEDS)
    configuration = RUNTIME_CONFIGURATIONS[task_id // runs_per_configuration]
    within = task_id % runs_per_configuration
    benchmark = BENCHMARKS[within // len(SEEDS)]
    seed = SEEDS[within % len(SEEDS)]
    return configuration, benchmark, seed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slurm_metadata() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in (
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_ID",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION",
            "SLURM_JOB_NODELIST",
        )
        if name in os.environ
    }


def output_path(root: Path, configuration: str, benchmark: str, seed: int) -> Path:
    return root / configuration / f"seed-{seed}" / benchmark / "runtime.json"


def run_one(
    configuration: str, benchmark: str, seed: int, output_root: Path,
    quality_prior_variance_mode: str = "warm",
) -> Path:
    output = output_path(output_root, configuration, benchmark, seed)
    if output.is_file():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("prior_variance_modes") != [quality_prior_variance_mode, "warm", "warm"]:
            raise ValueError(f"Incompatible saved prior modes: {output}")
        print(f"Reusing {output}", flush=True)
        return output

    models, questions, table, input_hashes = load_three_objective_benchmark(benchmark)
    settings = CONFIGURATIONS[configuration]
    print(
        f"Timing {configuration} / {benchmark} / seed {seed} at 30% entries",
        flush=True,
    )
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
        observation_budget_fraction=OBSERVATION_BUDGET_FRACTION,
        exact_observation_budget=True,
        force_to_observation_budget=True,
    )
    available_entries = int(run["n_arms"] * run["n_questions"])
    target_entry_cap = int(math.ceil(
        OBSERVATION_BUDGET_FRACTION * available_entries
    ))
    total_evaluations = int(run["total_evaluations"])
    if run["stop_reason"] != "observation_budget":
        raise RuntimeError(
            f"{configuration}/{benchmark}/seed-{seed} ended via "
            f"{run['stop_reason']!r}, not the fixed entry budget"
        )
    if total_evaluations != target_entry_cap:
        raise RuntimeError(
            f"terminal evaluations {total_evaluations} != cap {target_entry_cap}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "configuration": configuration,
        "benchmark": benchmark,
        "seed": seed,
        "directions": [list(direction) for direction in settings["directions"]],
        "direction_scheduler": "round_robin",
        "acquisition_cost_mode": settings["acquisition_cost_mode"],
        "continuation_mode": settings["continuation_mode"],
        "prior_variance_modes": [quality_prior_variance_mode, "warm", "warm"],
        "entry_budget_fraction": OBSERVATION_BUDGET_FRACTION,
        "available_entries": available_entries,
        "target_entry_cap": target_entry_cap,
        "realized_entry_fraction": total_evaluations / available_entries,
        "exact_observation_budget": True,
        "force_to_observation_budget": True,
        "forced_continuation_after_eta_floor": run[
            "forced_continuation_after_eta_floor"
        ],
        "native_gittins_stop_evaluations": run["gittins_stop_evaluations"],
        "native_gittins_stop_cost_usd": run["gittins_stop_cost_usd"],
        "policy_wall_time_seconds": run["policy_wall_time_seconds"],
        "sampling_wall_seconds": run["sampling_wall_seconds"],
        "total_evaluations": total_evaluations,
        "search_cost_usd": run["search_cost_usd"],
        "stop_reason": run["stop_reason"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "input_sha256": input_hashes,
        "source_sha256": {
            name: sha256(ROOT / name) for name in SOURCE_FILES
        },
        "slurm": slurm_metadata(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        f"Finished in {run['policy_wall_time_seconds']:.3f}s at "
        f"{total_evaluations}/{available_entries} entries: {output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quality-prior-variance-mode", choices=("fixed", "warm"), default="warm")
    args = parser.parse_args()
    task_id = args.task_id
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("provide --task-id or SLURM_ARRAY_TASK_ID")
        task_id = int(raw)
    configuration, benchmark, seed = task_mapping(task_id)
    run_one(configuration, benchmark, seed, args.output_root,
            quality_prior_variance_mode=args.quality_prior_variance_mode)


if __name__ == "__main__":
    main()
