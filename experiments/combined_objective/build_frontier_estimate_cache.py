#!/usr/bin/env python3
"""Build per-run normal and completed-only recommendation caches for plotting.

Every output file represents exactly one ``(benchmark, method, seed)`` run.
The default seed set is the continuous range 42--61; no seed is excluded or
substituted. Independent runs can be executed concurrently with ``--jobs``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    simulate_multiobjective_random_search,
)
from experiments.combined_objective.offline_pareto_baselines import (  # noqa: E402
    APE_K,
    EGE_SH,
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.combined_objective.plot_paper_20seed_comparison import PICKLES  # noqa: E402
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


SEEDS = tuple(range(42, 62))
BENCHMARKS = ("hotpotqa", "mathqa")
METHODS = (
    "gittins", EGE_SH, APE_K, QNEHVI,
    "random_questions", "random_configurations",
)


def _write_cache(
    path: Path,
    *,
    benchmark: str,
    method: str,
    seed: int,
    estimated_names: tuple[str, ...],
    completed_names: tuple[str, ...],
    metadata: dict[str, object],
) -> None:
    payload = {
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "estimated_selected_models": list(estimated_names),
        "completed_selected_models": list(completed_names),
        "selection_semantics": {
            "estimated": "algorithm-normal recommendation; completion not required",
            "completed": "empirical Pareto set restricted to fully evaluated arms",
        },
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gittins_cache(data_dir: Path, benchmark: str, seed: int):
    candidates = (
        data_dir / "gittins_run" / "raw_runs" / f"{benchmark}_seed-{seed}.pkl",
        data_dir / "raw_gittins" / f"{benchmark}_seed-{seed}.pkl",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    run = saved[0] if isinstance(saved, tuple) else saved
    stop = next(
        point for point in run.recommendation_trajectory
        if point.event == "gittins_stop"
    )
    names_by_arm = {item.arm_index: item.model_name for item in run.model_results}
    estimated_arms = tuple(stop.posterior_archive_arm_indices)
    completed_arms = tuple(stop.online_raw_archive_arm_indices)
    return (
        tuple(names_by_arm[arm] for arm in estimated_arms),
        tuple(names_by_arm[arm] for arm in completed_arms),
        {"event": "gittins_stop", "source": str(path)},
    )


def _build_one(task: tuple[str, str, int, str, bool]) -> str:
    benchmark, method, seed, data_dir_raw, force = task
    data_dir = Path(data_dir_raw)
    destination = data_dir / "frontier_estimates" / benchmark / method / f"seed-{seed}.json"
    if destination.exists() and not force:
        return f"cached {destination}"

    if method == "gittins":
        estimated_names, completed_names, metadata = _gittins_cache(data_dir, benchmark, seed)
    else:
        models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
        if method in (EGE_SH, APE_K, QNEHVI):
            kwargs = {"qnehvi_refit_every": 32} if method == QNEHVI else {}
            result = simulate_pareto_baseline(
                models, datapoints, table, method=method, seed=seed,
                observation_budget_fraction=0.1, **kwargs,
            )
            indices = list(result.selected_arm_indices)
            estimated_names = tuple(result.selected_models)
            completed_names = tuple(result.completed_pareto_models)
            metadata = {
                "observation_budget_fraction": 0.1,
                "total_evaluations": result.total_evaluations,
                "total_search_cost_usd": result.total_search_cost_usd,
            }
        else:
            result = simulate_multiobjective_random_search(
                models, datapoints, table, version=method,
                budget_fraction=0.1, seed=seed,
            )
            indices = list(result.selected_arm_indices)
            estimated_names = tuple(result.selected_models)
            completed_names = tuple(result.completed_pareto_models)
            metadata = {
                "budget_fraction": 0.1,
                "total_evaluations": result.total_evaluations,
                "total_search_cost_usd": result.total_search_cost_usd,
            }
    _write_cache(
        destination, benchmark=benchmark, method=method, seed=seed,
        estimated_names=estimated_names, completed_names=completed_names,
        metadata=metadata,
    )
    return f"wrote {destination}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "analysis/continuous_seeds_42_61/data",
    )
    args = parser.parse_args()
    invalid = sorted(set(args.seeds) - set(SEEDS))
    if invalid:
        raise SystemExit(f"seeds must be within the continuous range 42--61: {invalid}")
    tasks = [
        (benchmark, method, seed, str(args.data_dir.resolve()), args.force)
        for benchmark in args.benchmarks
        for method in args.methods
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_build_one, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
