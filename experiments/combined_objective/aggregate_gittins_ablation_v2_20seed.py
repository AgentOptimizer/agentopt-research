#!/usr/bin/env python3
"""Strictly validate and aggregate the G0-centered 8x20 ablation grid."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.gittins_ablation_v2 import (  # noqa: E402
    BENCHMARKS,
    CONFIGURATIONS,
    DEFAULT_OUTPUT_ROOT,
    G0_CONFIGURATION,
    RUN_CONFIGURATIONS,
    SEEDS,
    result_path,
)
from experiments.combined_objective.run_two_direction_ablation import PAIRS  # noqa: E402


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


def validate_cost_mode(
    settings: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any] | None:
    expected = result.get("parameters", {}).get("expected_batch_costs_usd", [])
    if settings["acquisition_cost_mode"] == "unit":
        if not expected or not all(
            math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in expected
        ):
            return {"expected": "all unit continuation costs", "actual": expected}
    elif expected and all(
        math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
        for value in expected
    ):
        return {"expected": "calibrated per-arm USD continuation costs", "actual": expected}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, nargs="?", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--slurm-array-job-ids", nargs="*", default=[])
    args = parser.parse_args()
    root = args.results_root.resolve()
    aggregate_dir = root / "aggregate"
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    truth_references: dict[str, dict[str, Any]] = {}
    reused_g0_run_count = 0

    for benchmark in BENCHMARKS:
        baseline_path = result_path(
            G0_CONFIGURATION, benchmark, SEEDS[0], output_root=root
        )
        baseline = read_json(baseline_path)
        if baseline is None:
            raise FileNotFoundError(f"missing G0 truth reference: {baseline_path}")
        baseline_run = baseline["run"]
        truth_references[benchmark] = {
            "raw_truth": np.asarray(
                baseline_run["raw_truth_vectors"], dtype=np.float64
            ),
            "model_names": list(baseline_run["model_names"]),
            "bruteforce_search_cost_usd": float(
                baseline_run["bruteforce_search_cost_usd"]
            ),
        }

    for configuration, settings in CONFIGURATIONS.items():
        pair_name = str(settings["pair_name"])
        expected_directions = [list(direction) for direction in PAIRS[pair_name]]
        for benchmark in BENCHMARKS:
            for seed in SEEDS:
                path = result_path(
                    configuration, benchmark, seed, output_root=root
                )
                metadata_path = path.with_name("slurm_task_metadata.json")
                result = read_json(path)
                metadata = read_json(metadata_path)
                if result is None:
                    missing.append(
                        {
                            "configuration": configuration,
                            "benchmark": benchmark,
                            "seed": seed,
                            "result_path": str(path),
                            "metadata_exists": metadata is not None,
                            "exit_code": None if metadata is None else metadata.get("exit_code"),
                        }
                    )
                    continue

                config = result.get("config", {})
                legacy_g0 = (
                    configuration == G0_CONFIGURATION
                    and config.get("ablation_name") == "g2_exact_axes"
                )
                expected = {
                    "ablation_name": (
                        "g2_exact_axes" if legacy_g0 else configuration
                    ),
                    "benchmark": benchmark,
                    "seed": seed,
                    "pair_name": pair_name,
                    "directions": expected_directions,
                    "eta_decay_schedule": settings["eta_decay_schedule"],
                    "direction_scheduler": settings["direction_scheduler"],
                }
                mismatches = {
                    key: {"expected": value, "actual": config.get(key)}
                    for key, value in expected.items()
                    if config.get(key) != value
                }
                cost_mismatch = validate_cost_mode(settings, result)
                if cost_mismatch is not None:
                    mismatches["acquisition_cost_mode"] = cost_mismatch
                run = result.get("run", {})
                truth_reference = truth_references[benchmark]
                raw_truth = np.asarray(
                    run.get("raw_truth_vectors", []), dtype=np.float64
                )
                reference_truth = truth_reference["raw_truth"]
                if raw_truth.shape != reference_truth.shape or not np.allclose(
                    raw_truth, reference_truth, rtol=0.0, atol=1e-15
                ):
                    mismatches["raw_truth_vectors"] = {
                        "expected_shape": list(reference_truth.shape),
                        "actual_shape": list(raw_truth.shape),
                        "expected": "identical to the G0 dataset and arm ordering",
                    }
                if list(run.get("model_names", [])) != truth_reference["model_names"]:
                    mismatches["model_names"] = {
                        "expected_count": len(truth_reference["model_names"]),
                        "actual_count": len(run.get("model_names", [])),
                        "expected": "identical to the G0 arm ordering",
                    }
                actual_bruteforce_cost = run.get("bruteforce_search_cost_usd")
                if actual_bruteforce_cost is None or not math.isclose(
                    float(actual_bruteforce_cost),
                    truth_reference["bruteforce_search_cost_usd"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    mismatches["bruteforce_search_cost_usd"] = {
                        "expected": truth_reference["bruteforce_search_cost_usd"],
                        "actual": actual_bruteforce_cost,
                    }
                if not legacy_g0:
                    for key in ("acquisition_cost_mode", "continuation_mode"):
                        if config.get(key) != settings[key]:
                            mismatches[key] = {
                                "expected": settings[key],
                                "actual": config.get(key),
                            }
                    if metadata is None or metadata.get("exit_code") != 0:
                        mismatches["exit_code"] = {
                            "expected": 0,
                            "actual": None if metadata is None else metadata.get("exit_code"),
                        }
                if mismatches:
                    invalid.append(
                        {
                            "configuration": configuration,
                            "benchmark": benchmark,
                            "seed": seed,
                            "result_path": str(path),
                            "mismatches": mismatches,
                        }
                    )
                    continue

                if legacy_g0:
                    reused_g0_run_count += 1
                rows.append(
                    {
                        "configuration": configuration,
                        "label": settings["label"],
                        "benchmark": benchmark,
                        "seed": seed,
                        "pair_name": pair_name,
                        "directions": expected_directions,
                        "direction_count": len(expected_directions),
                        "acquisition_cost_mode": settings["acquisition_cost_mode"],
                        "continuation_mode": settings["continuation_mode"],
                        "eta_decay_schedule": settings["eta_decay_schedule"],
                        "direction_scheduler": settings["direction_scheduler"],
                        "result_path": str(path),
                        "result_sha256": (
                            None if metadata is None else metadata.get("result_sha256")
                        ),
                        "task_wall_time_seconds": (
                            None
                            if metadata is None
                            else metadata.get("task_wall_time_seconds")
                        ),
                        **result.get("summary", {}),
                    }
                )

    write_csv(aggregate_dir / "runs.csv", rows)
    expected_count = len(CONFIGURATIONS) * len(BENCHMARKS) * len(SEEDS)
    summary = {
        "configuration_count": len(CONFIGURATIONS),
        "submitted_configuration_count": len(RUN_CONFIGURATIONS),
        "reused_g0_run_count": reused_g0_run_count,
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
