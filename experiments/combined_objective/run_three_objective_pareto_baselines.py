#!/usr/bin/env python3
"""Replay EGE-SH/SR, APE-k, and qNEHVI on aligned Q/L/D QA matrices.

The selector uses observed cells only.  The runner recomputes the completed
recommendation trajectory with the same three-objective diagnostics used for
the CC-Gittins and random baselines. Oracle diagnostics never feed acquisition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.offline_pareto_baselines import (  # noqa: E402
    METHODS,
    simulate_pareto_baseline,
)
from experiments.combined_objective.three_objective_metrics import (  # noqa: E402
    enrich_run,
    load_three_objective_benchmark,
)

DEFAULT_OUTDIR = ROOT / "experiments/combined_objective/results/three_objective_pareto_baselines"
CHECKPOINTS = (.05, .10, .20, .30, .50)


def _jsonable(value: Any) -> Any:
    """Encode baseline dataclasses and unsampled NaN estimates as strict JSON."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), *[
        ROOT / "experiments/combined_objective" / name for name in (
            "offline_pareto_baselines.py", "three_objective_metrics.py",
        )], *[
        ROOT / "src/agentopt/model_selection" / name for name in (
            "pareto_identification.py", "qnehvi.py",
        )]]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _result_as_run(result: Any, models: list[str], questions: list[int]) -> dict[str, Any]:
    """Convert the completed replay into the shared three-objective schema."""
    exhaustive_cost = float(result.params["bruteforce_search_cost_usd"])
    points = [{
        "event": point.event,
        "total_evaluations": int(point.cumulative_evaluations),
        "cumulative_evaluations": int(point.cumulative_evaluations),
        "cumulative_search_cost_usd": float(point.cumulative_search_cost_usd),
        "selected_arm_indices": list(point.selected_arm_indices),
    } for point in result.recommendation_trajectory]
    if not points:
        raise ValueError("Pareto baseline returned no recommendation checkpoints")
    final_selection = list(result.selected_arm_indices)
    if points[-1]["selected_arm_indices"] != final_selection:
        points.append({
            "event": "terminal",
            "total_evaluations": int(result.total_evaluations),
            "cumulative_evaluations": int(result.total_evaluations),
            "cumulative_search_cost_usd": float(result.total_search_cost_usd),
            "selected_arm_indices": final_selection,
        })
    if points[-1]["total_evaluations"] != result.total_evaluations:
        raise ValueError("Pareto baseline trajectory does not reach its final evaluation")
    if not math.isclose(points[-1]["cumulative_search_cost_usd"], result.total_search_cost_usd,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Pareto baseline trajectory does not reach its final USD spend")
    if any(new["total_evaluations"] < old["total_evaluations"] or
           new["cumulative_search_cost_usd"] + 1e-12 < old["cumulative_search_cost_usd"]
           for old, new in zip(points, points[1:])):
        raise ValueError("Pareto baseline checkpoints are not cumulative")

    run = {
        "selector": result.selector,
        "seed": result.seed,
        "params": result.params,
        "model_names": models,
        "question_ids": questions,
        "raw_truth_vectors": result.truth_raw_vectors.tolist(),
        "points": points,
        "selected_arm_indices": final_selection,
        "total_evaluations": int(result.total_evaluations),
        "search_cost_usd": float(result.total_search_cost_usd),
        "cumulative_search_cost_usd": float(result.total_search_cost_usd),
        "bruteforce_search_cost_usd": exhaustive_cost,
        "stop_reason": result.stop_reason,
        "selection_wall_seconds": float(result.policy_wall_time_seconds),
    }
    return enrich_run(run)


def _latest_at_or_below(run: dict[str, Any], target: float) -> dict[str, Any]:
    """Take the last completed physical batch under an actual-USD target."""
    if run["cost_fraction"] + 1e-12 < target:
        return {"target_cost_fraction": target, "available": False,
                "terminal_cost_fraction": run["cost_fraction"],
                "reason": "target_not_reached"}
    eligible = [point for point in run["points"] if point["cost_fraction"] <= target + 1e-12]
    if not eligible:
        return {"target_cost_fraction": target, "available": False,
                "reason": "no_complete_batch_under_budget"}
    return {"target_cost_fraction": target, "available": True, **eligible[-1]}


def run_one(
    benchmark: str | Path, *, method: str, seed: int, outdir: Path,
    batch_size: int = 4, observation_budget_fraction: float = 1.0,
    qnehvi_mc_samples: int = 64, qnehvi_refit_every: int = 32,
) -> dict[str, Any]:
    """Run or validate one cached replay, with source and input fingerprints."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    benchmark_name = Path(str(benchmark)).name
    if not benchmark_name or benchmark_name in (".", ".."):
        raise ValueError("benchmark must name a dataset or directory")
    models, questions, table, input_hashes = load_three_objective_benchmark(benchmark)
    output = outdir / f"seed_{seed}" / benchmark_name / method / "result.json"
    config = {
        "schema_version": 1,
        "benchmark": benchmark_name,
        "seed": int(seed),
        "method": method,
        "objective_order": ["Q", "L", "D"],
        "batch_size": int(batch_size),
        "observation_budget_fraction": float(observation_budget_fraction),
        "qnehvi_mc_samples": int(qnehvi_mc_samples),
        "qnehvi_refit_every": int(qnehvi_refit_every),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes(),
    }
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("config") != config:
            raise ValueError(f"Saved Pareto baseline inputs or protocol differ: {output}")
        return saved

    started = time.perf_counter()
    result = simulate_pareto_baseline(
        models, questions, table, method=method, seed=seed, batch_size=batch_size,
        observation_budget_fraction=observation_budget_fraction,
        objectives=("Q", "L", "D"), record_recommendation_trajectory=True,
        qnehvi_mc_samples=qnehvi_mc_samples, qnehvi_refit_every=qnehvi_refit_every,
        evaluation_question_ids=questions,
    )
    run = _result_as_run(result, models, questions)
    final = run["points"][-1]
    payload = {
        "config": config,
        "timing": {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "total_seconds": time.perf_counter() - started},
        "summary": {
            "stop_reason": run["stop_reason"],
            "cost_fraction": run["cost_fraction"],
            "search_cost_usd": run["search_cost_usd"],
            "bruteforce_search_cost_usd": run["bruteforce_search_cost_usd"],
            "total_evaluations": run["total_evaluations"],
            "selected_count": len(final["selected_arm_indices"]),
            **{key: final[key] for key in (
                "pareto_precision", "pareto_recall", "relative_hv_regret",
                "pareto_false_positive_count", "pareto_false_negative_count", "exact_frontier",
            )},
        },
        "checkpoints": [_latest_at_or_below(run, target) for target in CHECKPOINTS],
        "baseline_result": result,
        "run": run,
    }
    _write_json(output, payload)
    print(f"Finished {benchmark_name} {method} seed {seed}: "
          f"${run['search_cost_usd']:.4f} search spend, "
          f"3D HV regret {final['relative_hv_regret']:.2%}; {output}", flush=True)
    return _jsonable(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["mathqa", "hotpotqa"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--observation-budget-fraction", type=float, default=1.0)
    parser.add_argument("--qnehvi-mc-samples", type=int, default=64)
    parser.add_argument("--qnehvi-refit-every", type=int, default=32)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    for benchmark in args.benchmarks:
        for method in args.methods:
            for seed in args.seeds:
                run_one(benchmark, method=method, seed=seed, outdir=args.outdir,
                        batch_size=args.batch_size,
                        observation_budget_fraction=args.observation_budget_fraction,
                        qnehvi_mc_samples=args.qnehvi_mc_samples,
                        qnehvi_refit_every=args.qnehvi_refit_every)


if __name__ == "__main__":
    main()
