"""Run the three-objective ablation grid on server-latency QA matrices.

Each result is independently fingerprinted and resumable by the underlying
runner. The search follows its natural stop over the full evaluation budget;
the plotter then selects checkpoints at 0--30% of exhaustive USD cost.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.combined_objective.run_three_objective_ablation_task import run_one
from experiments.combined_objective.three_objective_ablation import (
    BENCHMARKS, RUN_CONFIGURATIONS, SEEDS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--configurations", nargs="+", choices=RUN_CONFIGURATIONS,
                        default=RUN_CONFIGURATIONS)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        default=BENCHMARKS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    tasks = [(configuration, benchmark, seed, args.output_root)
             for configuration in args.configurations
             for benchmark in args.benchmarks for seed in args.seeds]
    print(f"Running {len(tasks)} ablation cases with {args.workers} workers", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {task[:3]}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
