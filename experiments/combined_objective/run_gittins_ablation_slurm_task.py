#!/usr/bin/env python3
"""Run one configuration/benchmark/seed task from the Gittins ablation arrays."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.run_two_direction_ablation import (  # noqa: E402
    ABLATION_CONFIGS,
)


BENCHMARKS = (
    "restaurant_test",
    "hotpotqa",
    "mathqa",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
)
SEEDS = tuple(range(42, 62))
RUN_CONFIGURATIONS = tuple(
    name for name in ABLATION_CONFIGS if name != "g0_gauss_radau"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "experiments/combined_objective/results"
    / "gittins_ablation_8bench_20seed"
)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def task_mapping(task_id: int) -> tuple[str, int]:
    task_count = len(BENCHMARKS) * len(SEEDS)
    if not 0 <= task_id < task_count:
        raise ValueError(f"task id must be in [0, {task_count - 1}], got {task_id}")
    benchmark_index, seed_index = divmod(task_id, len(SEEDS))
    return BENCHMARKS[benchmark_index], SEEDS[seed_index]


def selected_slurm_environment() -> dict[str, str]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_TIMELIMIT",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_NODELIST",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=RUN_CONFIGURATIONS, required=True)
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Defaults to SLURM_ARRAY_TASK_ID.",
    )
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
    configuration = ABLATION_CONFIGS[args.configuration]
    pair_name = str(configuration["pair_name"])
    output_root = args.output_root.resolve()
    configuration_root = output_root / args.configuration
    seed_root = configuration_root / f"seed-{seed}"
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
        "--eta-decay-schedule",
        str(configuration["eta_decay_schedule"]),
        "--direction-scheduler",
        str(configuration["direction_scheduler"]),
        "--ablation-name",
        args.configuration,
        "--outdir",
        str(seed_root),
        "--skip-combined-export",
    ]
    mapping = {
        "task_id": task_id,
        "task_count": len(BENCHMARKS) * len(SEEDS),
        "configuration": args.configuration,
        "configuration_settings": configuration,
        "benchmark": benchmark,
        "seed": seed,
        "pair_name": pair_name,
        "result_path": str(result_path),
        "metadata_path": str(metadata_path),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(mapping, indent=2))
        return

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_time = time.perf_counter() - started
    ended_at = datetime.now(timezone.utc)

    result_payload = None
    if result_path.exists():
        try:
            result_payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result_payload = None
    cache_stats = None
    simulation_wall_time = None
    if result_payload is not None:
        simulation_wall_time = result_payload.get("config", {}).get("wall_time_seconds")
        cache_stats = result_payload.get("parameters", {}).get("boundary_cache", {}).get("stats")

    metadata = {
        "schema_version": 1,
        **mapping,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "task_wall_time_seconds": wall_time,
        "simulation_wall_time_seconds": simulation_wall_time,
        "child_user_cpu_seconds": after_usage.ru_utime - before_usage.ru_utime,
        "child_system_cpu_seconds": after_usage.ru_stime - before_usage.ru_stime,
        "child_max_rss_kib": after_usage.ru_maxrss,
        "exit_code": completed.returncode,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm": selected_slurm_environment(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "source_sha256": {
            "task_wrapper": sha256_file(Path(__file__)),
            "runner": sha256_file(runner_path),
            "engine": sha256_file(engine_path),
        },
        "result_exists": result_path.exists(),
        "result_size_bytes": result_path.stat().st_size if result_path.exists() else None,
        "result_sha256": sha256_file(result_path),
        "boundary_cache_stats": cache_stats,
    }
    atomic_json_dump(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
