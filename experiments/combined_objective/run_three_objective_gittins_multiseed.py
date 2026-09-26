#!/usr/bin/env python3
"""Run the four-direction or Q/L/D-axis CC-Gittins variant over seeds 42..61.

Compact outputs retain every physical-batch recommendation and metric, while
omitting repeated all-arm posterior arrays and the detailed physical trace.
The four-direction seed 42 is copied from the validated original pilot, which
remains untouched. The axes-only variant evaluates every seed independently.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

# A worker owns one numerical replay. Avoid multiplying BLAS thread pools.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.offline_three_objective_gittins import simulate_three_objective_gittins
from experiments.combined_objective.run_three_objective_gittins import (
    DEFAULT_OUTDIR as ORIGINAL_ROOT,
    DIRECTIONS,
    DIRECTION_LABELS,
    source_hashes,
    summarize,
)
from experiments.combined_objective.three_objective_metrics import enrich_run, load_three_objective_benchmark

DEFAULT_OUTDIR = ROOT / "experiments/combined_objective/results/three_objective_20seed"
DEFAULT_AXES_OUTDIR = ROOT / "experiments/combined_objective/results/three_objective_axes_20seed"
VARIANT_DIRECTIONS = {
    "four_directions": DIRECTIONS,
    "axes_only": ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
}
VARIANT_LABELS = {"four_directions": DIRECTION_LABELS, "axes_only": ("Q", "L", "D")}
STORAGE_SCHEMA = "compact_three_objective_v1"
HEAVY_POINT_FIELDS = frozenset((
    "finite_target_mean_vectors", "finite_target_std_vectors",
    "recommendation_raw_vectors", "observed_mean_vectors",
))
HEAVY_RUN_FIELDS = frozenset((
    "trace", "observed_cells", "raw_posterior_means", "raw_posterior_variances",
))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("w") as output:
        json.dump(value, output, allow_nan=False, separators=(",", ":"))
        output.write("\n")
    temporary.replace(path)


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the existing config/run schema and all checkpoint metric values.

    This function does not mutate the dense source or recompute its metrics.
    All lightweight top-level metadata, including calibration, survives.
    """
    result = {key: value for key, value in payload.items() if key not in ("run", "checkpoints")}
    source = payload["run"]
    run = {key: value for key, value in source.items() if key not in HEAVY_RUN_FIELDS and key != "points"}
    run["points"] = [
        {key: value for key, value in point.items() if key not in HEAVY_POINT_FIELDS}
        for point in source["points"]
    ]
    result["run"] = run
    if "checkpoints" in payload:
        result["checkpoints"] = [
            {key: value for key, value in point.items() if key not in HEAVY_POINT_FIELDS}
            for point in payload["checkpoints"]
        ]
    result["storage"] = {
        "schema": STORAGE_SCHEMA,
        "points": "all physical-batch recommendation sets and saved oracle metrics",
        "omitted": "all-arm per-checkpoint moment arrays, physical cell trace, final raw posterior arrays",
    }
    return result


def base_config(benchmark: str, seed: int, input_hashes: dict[str, str],
                variant: str = "four_directions") -> dict[str, Any]:
    """The exact original seed-42 numerical protocol and seven source hashes."""
    if variant not in VARIANT_DIRECTIONS:
        raise ValueError(f"unknown direction variant: {variant}")
    config = {
        "schema_version": 1, "benchmark": benchmark, "seed": seed,
        "objective_order": ["Q", "L", "D"],
        "raw_objectives": ["accuracy_max", "mean_latency_seconds_min", "mean_cost_usd_min"],
        "directions": [list(direction) for direction in VARIANT_DIRECTIONS[variant]],
        "direction_labels": list(VARIANT_LABELS[variant]),
        "direction_scheduler": "round_robin", "eta_decay_schedule": "direction_stop",
        "eta_initial": 1., "eta_decay": .5,
        "batch_size": 4, "warm_start_batch_size": 4,
        "question_order": "independent", "warm_start_question_order": "independent",
        "recommendation_rule": "finite_lcb", "recommendation_beta": 1.,
        "latency_model": "raw_mean", "cost_model": "raw_mean",
        "acquisition_cost_unit": "USD", "grid_size": 129,
        "observation_budget_fraction": 1.,
        "input_sha256": input_hashes, "source_sha256": source_hashes(),
    }
    # Keep four-direction configs byte-compatible with the preserved pilot.
    if variant != "four_directions":
        config["variant"] = variant
    return config


def validate_config(saved: dict[str, Any], expected: dict[str, Any], path: Path | str) -> None:
    """Fail closed when inputs, protocol or any frozen source have changed."""
    mismatch = [key for key, value in expected.items() if saved.get("config", {}).get(key) != value]
    if mismatch:
        raise ValueError(f"Refusing to reuse {path}; changed {mismatch}")
    run = saved.get("run", {})
    if not run.get("points") or "stop_reason" not in run:
        raise ValueError(f"Refusing incomplete saved result: {path}")
    if int(run.get("total_evaluations", -1)) != int(run["points"][-1]["total_evaluations"]):
        raise ValueError(f"Final checkpoint and run observation totals disagree: {path}")


def _receipt(payload: dict[str, Any], path: Path, status: str) -> dict[str, Any]:
    config, run = payload["config"], payload["run"]
    return {
        "benchmark": config["benchmark"], "seed": config["seed"], "status": status,
        "path": str(path), "bytes": path.stat().st_size,
        "total_evaluations": run["total_evaluations"], "cost_fraction": run["cost_fraction"],
        "stop_reason": run["stop_reason"], "point_count": len(run["points"]),
        "simulation_seconds": payload.get("timing", {}).get("simulation_seconds"),
        "summary": payload["summary"],
    }


def run_one(benchmark: str, seed: int, outdir: str | Path | None = None,
            original_root: str | Path = ORIGINAL_ROOT,
            variant: str = "four_directions") -> dict[str, Any]:
    """Replay or resume one pair, writing compact JSON and returning a receipt."""
    models, questions, table, input_hashes = load_three_objective_benchmark(benchmark)
    expected = base_config(benchmark, seed, input_hashes, variant=variant)
    if outdir is None:
        outdir = DEFAULT_AXES_OUTDIR if variant == "axes_only" else DEFAULT_OUTDIR
    path = Path(outdir) / f"seed_{seed}" / benchmark / "result.json"
    if path.exists():
        saved = json.loads(path.read_text())
        validate_config(saved, expected, path)
        if saved.get("storage", {}).get("schema") != STORAGE_SCHEMA:
            raise ValueError(f"Unexpected storage schema: {path}")
        if saved["run"]["model_names"] != models or saved["run"]["question_ids"] != questions:
            raise ValueError(f"Saved arm/question order differs: {path}")
        result = _receipt(saved, path, "resumed")
        print(f"Resumed {benchmark} seed {seed}: {result['point_count']} checkpoints", flush=True)
        return result

    original = Path(original_root) / f"seed_{seed}" / benchmark / "result.json"
    if variant == "four_directions" and seed == 42 and original.exists():
        saved = json.loads(original.read_text())
        validate_config(saved, expected, original)
        if saved["run"]["model_names"] != models or saved["run"]["question_ids"] != questions:
            raise ValueError(f"Original arm/question order differs: {original}")
        payload = compact_payload(saved)
        payload["provenance"] = {
            "source_result": str(original),
            "source_result_sha256": hashlib.file_digest(original.open("rb"), "sha256").hexdigest(),
            "compacted_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        dump_json(path, payload)
        result = _receipt(payload, path, "reused_original")
        del saved, payload
        gc.collect()
        print(f"Compacted original {benchmark} seed {seed}: {result['bytes']/1e6:.1f} MB", flush=True)
        return result

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    last_progress = started
    print(f"Running {variant} {benchmark} seed {seed} [{os.getpid()}]", flush=True)

    def progress(event: dict[str, Any]) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if now - last_progress >= 15.:
            print(f"  {benchmark} seed {seed}: {event['total_evaluations']} observations, "
                  f"${event['total_cost']:.4f}, {now-started:.0f}s", flush=True)
            last_progress = now

    run = simulate_three_objective_gittins(
        models, questions, table, seed=seed, directions=VARIANT_DIRECTIONS[variant],
        batch_size=4, warm_start_batch_size=4, grid_size=129,
        question_order="independent", warm_start_question_order="independent",
        lambda_initial=1., lambda_decay=.5, recommendation_beta=1.,
        observation_budget_fraction=1., progress_callback=progress,
    )
    simulation_seconds = time.perf_counter() - started
    run = enrich_run(run)
    if variant == "axes_only":
        run["selector"] = "cc_gittins_axes"
        if any(run["radial_boundary_cache_stats"].values()):
            raise RuntimeError("axes-only replay unexpectedly touched the radial DP cache")
    payload = compact_payload({
        "config": expected,
        "timing": {"started_at_utc": started_utc, "simulation_seconds": simulation_seconds,
                   "total_seconds": time.perf_counter() - started},
        "summary": summarize(run), "run": run,
        "provenance": {"launcher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    })
    del run
    gc.collect()
    dump_json(path, payload)
    result = _receipt(payload, path, "completed")
    print(f"Finished {benchmark} seed {seed}: {result['cost_fraction']:.2%} USD, "
          f"{simulation_seconds:.1f}s, {result['bytes']/1e6:.1f} MB", flush=True)
    return result


def _worker(arguments: tuple[str, int, str, str, str]) -> dict[str, Any]:
    return run_one(*arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=("mathqa", "hotpotqa"), default=["mathqa", "hotpotqa"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--workers", type=int, default=2,
                        help="Default 2 for the 8 GB host; compact saving does not shrink the live engine state")
    parser.add_argument("--variant", choices=tuple(VARIANT_DIRECTIONS), default="four_directions")
    parser.add_argument("--outdir", type=Path,
                        help="Defaults to separate four-direction or axes-only result root")
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    args = parser.parse_args()
    if args.workers < 1 or not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("workers must be positive and seeds unique nonnegative integers")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("benchmarks must be unique")
    if args.outdir is None:
        args.outdir = DEFAULT_AXES_OUTDIR if args.variant == "axes_only" else DEFAULT_OUTDIR
    jobs = [(benchmark, seed, str(args.outdir), str(args.original_root), args.variant)
            for seed in args.seeds for benchmark in args.benchmarks]
    receipts = []
    # Dense seed-42 sources are deliberately loaded serially before workers
    # start, limiting the peak memory required by JSON decoding.
    remaining = []
    for job in jobs:
        if job[1] == 42 and args.variant == "four_directions":
            receipts.append(_worker(job))
        else:
            remaining.append(job)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": args.seeds, "benchmarks": args.benchmarks, "workers": args.workers,
                "variant": args.variant, "storage_schema": STORAGE_SCHEMA,
                "requested_runs": len(jobs), "runs": receipts}
    dump_json(args.outdir / "manifest.json", manifest)
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_worker, job): job for job in remaining}
            for future in as_completed(futures):
                receipts.append(future.result())
                manifest["runs"] = sorted(receipts, key=lambda row: (row["seed"], row["benchmark"]))
                dump_json(args.outdir / "manifest.json", manifest)
                print(f"Completed {len(receipts)}/{len(jobs)} requested runs", flush=True)
    print(f"All {len(receipts)} runs complete: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
