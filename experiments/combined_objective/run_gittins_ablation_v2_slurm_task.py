#!/usr/bin/env python3
"""Run one dataset/seed task for the G0-centered Gittins ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

try:
    import resource
except ImportError:  # The standard-library resource module is Unix-only.
    resource = None


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.gittins_ablation_v2 import (  # noqa: E402
    BENCHMARKS,
    CONFIGURATIONS,
    DEFAULT_OUTPUT_ROOT,
    G0_CONFIGURATION,
    RUN_CONFIGURATIONS,
    SEEDS,
    result_path as protocol_result_path,
)
from experiments.combined_objective.anonymous_metadata import artifact_reference  # noqa: E402


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def truth_mismatch(
    result_payload: dict[str, Any], baseline_payload: dict[str, Any]
) -> dict[str, Any] | None:
    run = result_payload.get("run", {})
    baseline_run = baseline_payload.get("run", {})
    expected_truth = baseline_run.get("raw_truth_vectors", [])
    actual_truth = run.get("raw_truth_vectors", [])
    expected_names = baseline_run.get("model_names", [])
    actual_names = run.get("model_names", [])
    expected_cost = baseline_run.get("bruteforce_search_cost_usd")
    actual_cost = run.get("bruteforce_search_cost_usd")
    if (
        actual_truth != expected_truth
        or actual_names != expected_names
        or actual_cost != expected_cost
    ):
        return {
            "expected_arm_count": len(expected_truth),
            "actual_arm_count": len(actual_truth),
            "expected_model_count": len(expected_names),
            "actual_model_count": len(actual_names),
            "expected_bruteforce_search_cost_usd": expected_cost,
            "actual_bruteforce_search_cost_usd": actual_cost,
        }
    return None


def task_mapping(task_id: int) -> tuple[str, int]:
    task_count = len(BENCHMARKS) * len(SEEDS)
    if not 0 <= task_id < task_count:
        raise ValueError(f"task id must be in [0, {task_count - 1}], got {task_id}")
    benchmark_index, seed_index = divmod(task_id, len(SEEDS))
    return BENCHMARKS[benchmark_index], SEEDS[seed_index]


def selected_slurm_environment() -> dict[str, str]:
    """Record resource allocations, excluding cluster and job identifiers."""
    names = (
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_TIMELIMIT",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def child_resource_usage() -> Any | None:
    """Return Unix child-process counters when this platform supports them."""
    if resource is None:
        return None
    try:
        return resource.getrusage(resource.RUSAGE_CHILDREN)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=RUN_CONFIGURATIONS, required=True)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        raw_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_task_id is None:
            parser.error("--task-id or SLURM_ARRAY_TASK_ID is required")
        task_id = int(raw_task_id)

    benchmark, seed = task_mapping(task_id)
    settings = CONFIGURATIONS[args.configuration]
    pair_name = str(settings["pair_name"])
    output_root = args.output_root.resolve()
    seed_root = output_root / args.configuration / f"seed-{seed}"
    result_path = seed_root / pair_name / benchmark / "result.json"
    metadata_path = result_path.with_name("slurm_task_metadata.json")
    runner_path = ROOT / "experiments/combined_objective/run_two_direction_ablation.py"
    engine_path = ROOT / "experiments/combined_objective/offline_radial_gittins.py"
    command = [
        sys.executable,
        "-u",
        str(runner_path),
        "--benchmarks",
        benchmark,
        "--pairs",
        pair_name,
        "--seed",
        str(seed),
        "--observation-budget-fraction",
        "1.0",
        "--direction-scheduler",
        str(settings["direction_scheduler"]),
        "--acquisition-cost-mode",
        str(settings["acquisition_cost_mode"]),
        "--continuation-mode",
        str(settings["continuation_mode"]),
        "--ablation-name",
        args.configuration,
        "--outdir",
        str(seed_root),
        "--skip-combined-export",
    ]
    # The actual subprocess uses local absolute paths; the recorded command is
    # relocatable and never includes the submitter's home or output directory.
    public_command = command.copy()
    public_command[0] = "python"
    public_command[2] = runner_path.relative_to(ROOT).as_posix()
    public_command[public_command.index("--outdir") + 1] = (
        Path("results") / seed_root.relative_to(output_root)
    ).as_posix()
    mapping = {
        "task_id": task_id,
        "task_count": len(BENCHMARKS) * len(SEEDS),
        "configuration": args.configuration,
        "configuration_settings": settings,
        "benchmark": benchmark,
        "seed": seed,
        "pair_name": pair_name,
        "path_base": "results",
        "result_path": (Path("results") / result_path.relative_to(output_root)).as_posix(),
        "metadata_path": (Path("results") / metadata_path.relative_to(output_root)).as_posix(),
        "command": public_command,
    }
    if args.dry_run:
        print(json.dumps(mapping, indent=2))
        return

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    before_usage = child_resource_usage()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    after_usage = child_resource_usage()
    wall_time = time.perf_counter() - started
    ended_at = datetime.now(timezone.utc)

    result_payload = None
    if result_path.exists():
        try:
            result_payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result_payload = None
    validation_error = None
    effective_exit_code = completed.returncode
    if completed.returncode == 0 and result_payload is None:
        validation_error = {"error": "runner completed without a readable result"}
    elif completed.returncode == 0 and args.configuration != G0_CONFIGURATION:
        baseline_path = protocol_result_path(
            G0_CONFIGURATION, benchmark, SEEDS[0], output_root=output_root,
        )
        try:
            baseline_payload = json.loads(baseline_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            validation_error = {
                "baseline_path": artifact_reference(baseline_path, root=output_root),
                "error": f"failed to load G0 truth reference ({type(error).__name__})",
            }
        else:
            mismatch = truth_mismatch(result_payload, baseline_payload)
            if mismatch is not None:
                validation_error = {
                    "baseline_path": artifact_reference(baseline_path, root=output_root),
                    "error": "result dataset or arm ordering differs from G0",
                    **mismatch,
                }
    if validation_error is not None:
        effective_exit_code = 2
    metadata = {
        "schema_version": 2,
        **mapping,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "task_wall_time_seconds": wall_time,
        "simulation_wall_time_seconds": (
            None
            if result_payload is None
            else result_payload.get("config", {}).get("wall_time_seconds")
        ),
        "child_user_cpu_seconds": (
            after_usage.ru_utime - before_usage.ru_utime
            if after_usage is not None and before_usage is not None else None
        ),
        "child_system_cpu_seconds": (
            after_usage.ru_stime - before_usage.ru_stime
            if after_usage is not None and before_usage is not None else None
        ),
        "child_max_rss_kib": (
            after_usage.ru_maxrss / (1024 if sys.platform == "darwin" else 1)
            if after_usage is not None else None
        ),
        "exit_code": effective_exit_code,
        "simulation_exit_code": completed.returncode,
        "g0_truth_validation_error": validation_error,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "python": platform.python_version(),
        "slurm": selected_slurm_environment(),
        "source_sha256": {
            "task_wrapper": sha256_file(Path(__file__)),
            "runner": sha256_file(runner_path),
            "engine": sha256_file(engine_path),
        },
        "result_exists": result_path.exists(),
        "result_size_bytes": result_path.stat().st_size if result_path.exists() else None,
        "result_sha256": sha256_file(result_path),
        "boundary_cache_stats": (
            None
            if result_payload is None
            else result_payload.get("parameters", {})
            .get("boundary_cache", {})
            .get("stats")
        ),
    }
    atomic_json_dump(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    raise SystemExit(effective_exit_code)


if __name__ == "__main__":
    main()
