#!/usr/bin/env python3
"""Build wall-clock data for the paper frontier's exact hybrid run set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CURVE_ROOT = ROOT / "analysis/20seed_results/figures/eight_dataset_curves"
LATEST_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
LEGACY_BASELINE_ROOT = ROOT / "analysis/final_run_baselines"
RADIAL_ROOT = ROOT / "analysis/20seed_results/radial_gittins"
DEFAULT_OUTPUT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed"
    / "figures_preview/time"
)

DATASETS = (
    "hotpotqa",
    "restaurant_valid",
    "bird_mini_dev",
    "bing_querylogs",
    "mathqa",
    "restaurant_test",
    "bird_dev",
    "stackoverflow",
)
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
EXPECTED_SEEDS = tuple(range(42, 62))

# These are the exact completed qNEHVI seeds used when the 0--100% curve
# summary and plot metadata were written. BIRD Mini seed 43 completed later.
QNEHVI_CURVE_SEEDS = {
    "bing_querylogs": tuple(range(42, 60)),
    "bird_mini_dev": (42, *range(44, 62)),
}
LEGACY_QNEHVI_DATASETS = {
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
}

SCOPE_DATASETS = (
    "bing_querylogs",
    "bird_dev",
    "bird_mini_dev",
    "restaurant_test",
    "restaurant_valid",
    "stackoverflow",
)


def legacy_scope_slurm_ids(dataset: str, method: str, seed: int) -> tuple[str, str]:
    """Recover IDs for original SCOPE tasks written before IDs entered JSON."""
    if method == "ege_sh":
        parent = "17423586"
        task = 20 * SCOPE_DATASETS.index(dataset) + seed - 42
    elif method == "ape_k":
        parent = "17423678"
        task = 20 * SCOPE_DATASETS.index(dataset) + seed - 42
    elif method == "qnehvi" and dataset in {
        "bird_dev",
        "restaurant_test",
        "stackoverflow",
    }:
        parent = "17423681"
        task = 20 * ("bird_dev", "restaurant_test", "stackoverflow").index(dataset) + seed - 42
    elif method == "qnehvi" and dataset in {"bird_mini_dev", "restaurant_valid"}:
        parent = "17423682"
        task = 20 * ("bird_mini_dev", "restaurant_valid").index(dataset) + seed - 42
    elif method == "qnehvi" and dataset == "bing_querylogs":
        parent = "17423691"
        task = seed - 42
    else:
        raise KeyError(f"no legacy Slurm mapping for {dataset}/{method}/seed-{seed}")
    return parent, f"{parent}_{task}"


def slurm_ids(dataset: str, method: str, seed: int, payload: dict) -> tuple[str, str]:
    if "slurm_array_job_id" in payload and "slurm_job_id" in payload:
        return str(payload["slurm_array_job_id"]), str(payload["slurm_job_id"])
    return legacy_scope_slurm_ids(dataset, method, seed)


def selected_seeds(dataset: str, method: str) -> tuple[int, ...]:
    if method == "qnehvi" and dataset in QNEHVI_CURVE_SEEDS:
        return QNEHVI_CURVE_SEEDS[dataset]
    return EXPECTED_SEEDS


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slurm_elapsed_seconds(parent_ids: set[str]) -> dict[str, tuple[float, str]]:
    command = [
        "sacct",
        "-j",
        ",".join(sorted(parent_ids, key=int)),
        "--parsable2",
        "--noheader",
        "-X",
        "-o",
        "JobID,JobIDRaw,ElapsedRaw,State",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed: dict[str, tuple[float, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[3] == "COMPLETED":
            display_id, raw_id, seconds = fields[:3]
            record = (float(seconds), raw_id)
            elapsed[display_id] = record
            elapsed[raw_id] = record
    return elapsed


def collect_rows() -> list[dict[str, str | int | float]]:
    baseline_records: list[tuple[str, str, int, Path, str]] = []
    parent_ids: set[str] = set()
    for dataset in DATASETS:
        for method in METHODS[1:]:
            for seed in selected_seeds(dataset, method):
                source_root = (
                    LEGACY_BASELINE_ROOT
                    if method == "qnehvi" and dataset in LEGACY_QNEHVI_DATASETS
                    else LATEST_ROOT
                )
                path = source_root / dataset / method / f"seed-{seed}" / "summary.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing curve-source result: {path}")
                payload = load_json(path)
                parent_id, job_id = slurm_ids(dataset, method, seed, payload)
                parent_ids.add(parent_id)
                baseline_records.append((dataset, method, seed, path, job_id))

    elapsed = slurm_elapsed_seconds(parent_ids)
    rows: list[dict[str, str | int | float]] = []
    for dataset, method, seed, path, job_id in baseline_records:
        if job_id not in elapsed:
            raise RuntimeError(f"no completed Slurm accounting record for {job_id}: {path}")
        seconds, raw_job_id = elapsed[job_id]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "total_wall_clock_seconds": seconds,
                "source": str(path.relative_to(ROOT)),
                "slurm_job_id": raw_job_id,
            }
        )

    for dataset in DATASETS:
        for seed in EXPECTED_SEEDS:
            path = RADIAL_ROOT / dataset / f"seed-{seed}" / "slurm_task_metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"missing curve-source timing: {path}")
            payload = load_json(path)
            rows.append(
                {
                    "dataset": dataset,
                    "method": "radial_gittins",
                    "seed": seed,
                    "total_wall_clock_seconds": float(payload["task_wall_time_seconds"]),
                    "source": str(path.relative_to(ROOT)),
                    "slurm_job_id": str(payload.get("slurm", {}).get("job_id", "")),
                }
            )

    order = {
        (dataset, method): index
        for index, (dataset, method) in enumerate(
            (dataset, method) for dataset in DATASETS for method in METHODS
        )
    }
    rows.sort(key=lambda row: (order[(str(row["dataset"]), str(row["method"]))], int(row["seed"])))
    return rows


def write_outputs(rows: list[dict[str, str | int | float]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "runtime_by_seed.csv"
    summary_path = output / "runtime_summary.csv"

    raw_fields = (
        "dataset",
        "method",
        "seed",
        "total_wall_clock_seconds",
        "source",
        "slurm_job_id",
    )
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["total_wall_clock_seconds"]))

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("dataset", "method", "n", "mean_seconds", "error_2se_seconds")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                values = grouped[(dataset, method)]
                mean = statistics.fmean(values)
                two_se = 2.0 * statistics.stdev(values) / math.sqrt(len(values))
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": method,
                        "n": len(values),
                        "mean_seconds": f"{mean:.9f}",
                        "error_2se_seconds": f"{two_se:.9f}",
                    }
                )

    print(raw_path)
    print(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows()
    write_outputs(rows, args.output)


if __name__ == "__main__":
    main()
