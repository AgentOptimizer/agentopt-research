"""Run the three Pareto baselines used in the paper's three-objective figure."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.combined_objective.run_three_objective_pareto_baselines import run_one


METHODS = ("ege_sh", "ape_k", "qnehvi")
DATASETS = ("hotpotqa", "mathqa")
SEEDS = tuple(range(42, 62))


def run_task(dataset: str, method: str, seed: int, outdir: Path) -> tuple[str, str, int]:
    run_one(dataset, method=method, seed=seed, outdir=outdir,
            observation_budget_fraction=1.0)
    return dataset, method, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    tasks = [(dataset, method, seed, args.output_root) for dataset in DATASETS
             for method in METHODS for seed in SEEDS]
    print(f"Running {len(tasks)} Pareto baseline cases", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{index}/{len(tasks)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
