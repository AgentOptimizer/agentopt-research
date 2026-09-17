#!/usr/bin/env python3
"""Aggregate Gauss-Radau 8-benchmark x 20-seed results and runtime metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any


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
PAIR = "gauss_radau_accuracy_endpoint"


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
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--slurm-array-job-id")
    args = parser.parse_args()
    root = args.results_root.resolve()
    aggregate_dir = root / "aggregate"
    run_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for benchmark in BENCHMARKS:
        for seed in SEEDS:
            run_dir = root / f"seed-{seed}" / PAIR / benchmark
            result_path = run_dir / "result.json"
            metadata_path = run_dir / "slurm_task_metadata.json"
            result = read_json(result_path)
            metadata = read_json(metadata_path)
            if result is None:
                missing.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "result_path": str(result_path),
                        "metadata_exists": metadata is not None,
                        "exit_code": None if metadata is None else metadata.get("exit_code"),
                    }
                )
                continue

            config = result["config"]
            summary = result["summary"]
            cache = result.get("parameters", {}).get("boundary_cache", {})
            cache_stats = cache.get("stats", {})
            row: dict[str, Any] = {
                "benchmark": benchmark,
                "seed": seed,
                "pair_name": PAIR,
                "result_path": str(result_path),
                "stop_reason": summary.get("stop_reason"),
                "simulation_wall_time_seconds": config.get("wall_time_seconds"),
                "data_load_wall_time_seconds": config.get("data_load_wall_time_seconds"),
                "postprocess_wall_time_seconds": config.get("postprocess_wall_time_seconds"),
                "pre_write_wall_time_seconds": config.get("pre_write_wall_time_seconds"),
                "task_wall_time_seconds": None if metadata is None else metadata.get("task_wall_time_seconds"),
                "child_user_cpu_seconds": None if metadata is None else metadata.get("child_user_cpu_seconds"),
                "child_system_cpu_seconds": None if metadata is None else metadata.get("child_system_cpu_seconds"),
                "child_max_rss_kib": None if metadata is None else metadata.get("child_max_rss_kib"),
                "slurm_job_id": None if metadata is None else metadata.get("slurm", {}).get("SLURM_JOB_ID"),
                "slurm_array_job_id": None if metadata is None else metadata.get("slurm", {}).get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": None if metadata is None else metadata.get("slurm", {}).get("SLURM_ARRAY_TASK_ID"),
                "result_sha256": None if metadata is None else metadata.get("result_sha256"),
                "cache_builds": cache_stats.get("builds"),
                "cache_memory_hits": cache_stats.get("memory_hits"),
                "cache_disk_hits": cache_stats.get("disk_hits"),
                "cache_disk_misses": cache_stats.get("disk_misses"),
                "cache_corruptions": cache_stats.get("corruptions"),
                "cache_read_failures": cache_stats.get("read_failures"),
                "cache_write_failures": cache_stats.get("write_failures"),
            }
            row.update(summary)
            run_rows.append(row)

            for target in result.get("plotting", {}).get("targets", []):
                target_row = {
                    "benchmark": benchmark,
                    "seed": seed,
                    **target,
                }
                target_rows.append(target_row)
                if not target.get("available"):
                    continue
                selected = target.get("selected_arm_indices", [])
                names = target.get("selected_model_names", [])
                vectors = target.get("selected_full_data_vectors", [])
                intervals = target.get("selected_membership_intervals", {})
                for arm, name, vector in zip(selected, names, vectors):
                    arm_intervals = intervals.get(str(arm), [])
                    if not arm_intervals:
                        arm_intervals = [None]
                    for interval in arm_intervals:
                        interval_rows.append(
                            {
                                "benchmark": benchmark,
                                "seed": seed,
                                "target_cost_fraction": target["target_cost_fraction"],
                                "arm_index": arm,
                                "model_name": name,
                                "full_data_accuracy": vector[0],
                                "full_data_mean_cost_usd": vector[1],
                                "membership_start_cost_fraction": None if interval is None else interval[0],
                                "membership_end_cost_fraction": None if interval is None else interval[1],
                            }
                        )

    write_csv(aggregate_dir / "runs.csv", run_rows)
    write_csv(aggregate_dir / "plot_targets.csv", target_rows)
    write_csv(aggregate_dir / "recommendation_intervals.csv", interval_rows)
    summary_payload = {
        "expected_run_count": len(BENCHMARKS) * len(SEEDS),
        "completed_run_count": len(run_rows),
        "missing_run_count": len(missing),
        "missing": missing,
    }
    (aggregate_dir / "collection_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n"
    )

    if args.slurm_array_job_id:
        fields = (
            "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,TotalCPU,CPUTimeRAW,"
            "MaxRSS,AllocCPUS,ReqMem,Start,End"
        )
        completed = subprocess.run(
            [
                "sacct",
                "-j",
                args.slurm_array_job_id,
                "--parsable2",
                "--noheader",
                f"--format={fields}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        (aggregate_dir / "slurm_sacct.psv").write_text(completed.stdout)
        (aggregate_dir / "slurm_sacct.stderr.txt").write_text(completed.stderr)

    print(json.dumps(summary_payload, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
