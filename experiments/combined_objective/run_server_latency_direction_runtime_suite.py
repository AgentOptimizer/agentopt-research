"""Run seven matched direction configurations to the 30% entry cap."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.combined_objective.run_three_objective_direction_runtime_task import (
    RUNTIME_CONFIGURATIONS, run_one,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = [(configuration, dataset, seed, args.output_root)
             for configuration in RUNTIME_CONFIGURATIONS
             for dataset in ("hotpotqa", "mathqa") for seed in range(42, 62)]
    print(f"Running {len(tasks)} fixed-workload runtime cases", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{index}/{len(tasks)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
