#!/usr/bin/env python3
"""Run and persist the paper's matched 20-seed radial-UCB comparison."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    run_budget_sweep,
    write_results_csv,
)
from experiments.combined_objective.offline_radial_ucb import simulate_radial_ucb  # noqa: E402
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


PICKLES = {
    "gpqa": ROOT / "experiments/data/lookup/gpqa_lookup.pkl",
    "bfcl": ROOT / "experiments/data/lookup/bfcl_lookup.pkl",
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}
DEFAULT_BENCHMARKS = ("hotpotqa", "mathqa")
SEEDS = tuple(range(42, 58)) + (59, 60, 61, 62)


def _run_one(benchmark: str, seed: int, destination: str) -> str:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    result = simulate_radial_ucb(
        models,
        datapoints,
        table,
        batch_size=4,
        exploration_beta=2.0,
        search_cost_scale_eta=1.0,
        observation_budget_fraction=1.0,
        seed=seed,
        question_universe="common",
        # Record the adaptive stop, then continue the offline replay to the
        # full matrix so diagnostic trajectories reach cost fraction 1.0.
        halt_on_index_stop=False,
        record_recommendation_trajectory=True,
    )
    path = Path(destination)
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return f"{benchmark} seed={seed} stop={result.gittins_stop_budget_fraction}"


def _write_summaries(outdir: Path) -> None:
    seed_rows = []
    trajectory_rows = []
    for benchmark in PICKLES:
        if not all(
            (outdir / "raw_ucb" / f"{benchmark}_seed-{seed}.pkl").exists()
            for seed in SEEDS
        ):
            continue
        for seed in SEEDS:
            path = outdir / "raw_ucb" / f"{benchmark}_seed-{seed}.pkl"
            with path.open("rb") as handle:
                result = pickle.load(handle)
            trajectory = result.recommendation_trajectory
            stop_points = [point for point in trajectory if point.event == "gittins_stop"]
            stop = stop_points[0] if stop_points else trajectory[-1]
            seed_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "stop_budget_fraction": result.gittins_stop_budget_fraction,
                    "stop_search_cost_usd": result.gittins_stop_cost_usd,
                    "stop_hv_regret": stop.hypervolume_regret,
                    "n_recommended": len(stop.online_raw_archive_models),
                    "selected_arm_indices": json.dumps(list(stop.online_raw_archive_arm_indices)),
                    "selected_models": json.dumps(list(stop.online_raw_archive_models)),
                }
            )
            for point in trajectory:
                trajectory_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "budget_fraction": point.budget_fraction,
                        "cumulative_evaluations": point.cumulative_evaluations,
                        "cumulative_search_cost_usd": point.cumulative_search_cost_usd,
                        "hv_regret": point.hypervolume_regret,
                        "is_deployable": point.is_deployable,
                        "event": point.event,
                    }
                )
    for name, rows in (
        ("ucb_seed_results.csv", seed_rows),
        ("ucb_hv_regret_trajectories.csv", trajectory_rows),
    ):
        with (outdir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "analysis/paper_20seed_method_comparison",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute selected benchmark/seed caches even if they exist.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(PICKLES),
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to run; defaults to HotpotQA and MathQA.",
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    raw_dir = outdir / "raw_ucb"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for benchmark in args.benchmarks:
        for seed in SEEDS:
            destination = raw_dir / f"{benchmark}_seed-{seed}.pkl"
            if args.force or not destination.exists():
                pending.append((benchmark, seed, destination))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, benchmark, seed, str(destination)):
            (benchmark, seed)
            for benchmark, seed, destination in pending
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)

    _write_summaries(outdir)
    for benchmark in args.benchmarks:
        pickle_path = PICKLES[benchmark]
        models, datapoints, table = load_pickle(str(pickle_path))
        random_results = run_budget_sweep(
            models,
            datapoints,
            table,
            versions=("random_questions",),
            budget_fractions=(0.1, 0.4),
            seeds=SEEDS,
        )
        write_results_csv(random_results, outdir / f"{benchmark}_random_questions.csv")
    (outdir / "run_metadata.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "n_seeds": len(SEEDS),
                "excluded_seed": 58,
                "ucb": {
                    "exploration_beta": 2.0,
                    "bonus_mode": "posterior_sd",
                    "batch_size": 4,
                    "eta": 1.0,
                    "budget": "adaptive_stop_with_forced_full_diagnostic",
                    "halt_on_index_stop": False,
                },
                "random_questions_budgets": [0.1, 0.4],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote comparison data under {outdir}")


if __name__ == "__main__":
    main()
