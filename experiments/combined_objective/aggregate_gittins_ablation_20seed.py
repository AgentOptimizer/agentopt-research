#!/usr/bin/env python3
"""Validate and aggregate all six new 8-benchmark x 20-seed ablation groups."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.run_gittins_ablation_slurm_task import (
    BENCHMARKS,
    RUN_CONFIGURATIONS,
    SEEDS,
)
from experiments.combined_objective.run_two_direction_ablation import (
    ABLATION_CONFIGS,
    PAIRS,
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--slurm-array-job-ids", nargs="*", default=[])
    args = parser.parse_args()
    root = args.results_root.resolve()
    aggregate_dir = root / "aggregate"
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for configuration_name in RUN_CONFIGURATIONS:
        settings = ABLATION_CONFIGS[configuration_name]
        pair_name = str(settings["pair_name"])
        expected_directions = [list(direction) for direction in PAIRS[pair_name]]
        for benchmark in BENCHMARKS:
            for seed in SEEDS:
                run_dir = (
                    root
                    / configuration_name
                    / f"seed-{seed}"
                    / pair_name
                    / benchmark
                )
                result_path = run_dir / "result.json"
                metadata_path = run_dir / "slurm_task_metadata.json"
                result = read_json(result_path)
                metadata = read_json(metadata_path)
                if result is None:
                    missing.append(
                        {
                            "configuration": configuration_name,
                            "benchmark": benchmark,
                            "seed": seed,
                            "result_path": str(result_path),
                            "metadata_exists": metadata is not None,
                            "exit_code": None if metadata is None else metadata.get("exit_code"),
                        }
                    )
                    continue

                config = result.get("config", {})
                mismatches = {}
                expected = {
                    "ablation_name": configuration_name,
                    "benchmark": benchmark,
                    "seed": seed,
                    "pair_name": pair_name,
                    "directions": expected_directions,
                    "eta_decay_schedule": settings["eta_decay_schedule"],
                    "direction_scheduler": settings["direction_scheduler"],
                }
                for key, value in expected.items():
                    if config.get(key) != value:
                        mismatches[key] = {"expected": value, "actual": config.get(key)}
                if metadata is not None and metadata.get("exit_code") != 0:
                    mismatches["exit_code"] = {
                        "expected": 0,
                        "actual": metadata.get("exit_code"),
                    }
                if mismatches:
                    invalid.append(
                        {
                            "configuration": configuration_name,
                            "benchmark": benchmark,
                            "seed": seed,
                            "result_path": str(result_path),
                            "mismatches": mismatches,
                        }
                    )
                    continue

                rows.append(
                    {
                        "configuration": configuration_name,
                        "benchmark": benchmark,
                        "seed": seed,
                        "pair_name": pair_name,
                        "direction_count": len(expected_directions),
                        "eta_decay_schedule": settings["eta_decay_schedule"],
                        "direction_scheduler": settings["direction_scheduler"],
                        "result_path": str(result_path),
                        "result_sha256": None if metadata is None else metadata.get("result_sha256"),
                        "task_wall_time_seconds": None if metadata is None else metadata.get("task_wall_time_seconds"),
                        **result.get("summary", {}),
                    }
                )

    write_csv(aggregate_dir / "runs.csv", rows)
    expected_count = len(RUN_CONFIGURATIONS) * len(BENCHMARKS) * len(SEEDS)
    summary = {
        "expected_run_count": expected_count,
        "completed_valid_run_count": len(rows),
        "missing_run_count": len(missing),
        "invalid_run_count": len(invalid),
        "missing": missing,
        "invalid": invalid,
    }
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    if args.slurm_array_job_ids:
        completed = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(args.slurm_array_job_ids),
                "--parsable2",
                "--noheader",
                "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,AllocCPUS,ReqMem,Start,End",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        (aggregate_dir / "slurm_sacct.psv").write_text(completed.stdout)
        (aggregate_dir / "slurm_sacct.stderr.txt").write_text(completed.stderr)

    print(json.dumps(summary, indent=2))
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
