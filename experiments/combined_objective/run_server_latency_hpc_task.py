"""Dispatch one indexed paper-preview replay on a Slurm compute node."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("hotpotqa", "mathqa")
SEEDS = tuple(range(42, 62))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ablation", "pareto", "prior", "runtime"), required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quality-prior-variance-mode", choices=("fixed", "warm"), default="warm")
    args = parser.parse_args()
    from experiments.combined_objective.three_objective_ablation import task_count, task_mapping
    from experiments.combined_objective.run_server_latency_prior_factorial import VARIANTS
    from experiments.combined_objective.run_server_latency_pareto_suite import METHODS
    from experiments.combined_objective.run_three_objective_direction_runtime_task import RUNTIME_CONFIGURATIONS

    if args.kind == "ablation":
        if not 0 <= args.task_id < task_count():
            parser.error("ablation task ID out of range")
        configuration, dataset, seed = task_mapping(args.task_id)
        from experiments.combined_objective.run_three_objective_ablation_task import run_one
        result = run_one(configuration, dataset, seed, args.output_root / "three_ablation_warm",
                         quality_prior_variance_mode=args.quality_prior_variance_mode)
    elif args.kind == "pareto":
        if not 0 <= args.task_id < len(DATASETS) * len(METHODS) * len(SEEDS):
            parser.error("Pareto task ID out of range")
        dataset = DATASETS[args.task_id // (len(METHODS) * len(SEEDS))]
        method = METHODS[(args.task_id // len(SEEDS)) % len(METHODS)]
        seed = SEEDS[args.task_id % len(SEEDS)]
        from experiments.combined_objective.run_three_objective_pareto_baselines import run_one
        saved = run_one(dataset, method=method, seed=seed,
                        outdir=args.output_root / "three_objective_pareto_baselines",
                        observation_budget_fraction=1.0)
        result = {"dataset": dataset, "method": method, "seed": seed,
                  "evaluations": saved["run"]["total_evaluations"]}
    elif args.kind == "prior":
        if not 0 <= args.task_id < len(VARIANTS) * len(DATASETS) * len(SEEDS):
            parser.error("prior task ID out of range")
        modes = VARIANTS[args.task_id // (len(DATASETS) * len(SEEDS))]
        dataset = DATASETS[(args.task_id // len(SEEDS)) % len(DATASETS)]
        seed = SEEDS[args.task_id % len(SEEDS)]
        from experiments.combined_objective.run_server_latency_prior_factorial import run_one
        result = run_one(modes, dataset, seed, args.output_root / "three_prior_factorial")
    else:
        if not 0 <= args.task_id < len(RUNTIME_CONFIGURATIONS) * len(DATASETS) * len(SEEDS):
            parser.error("runtime task ID out of range")
        configuration = RUNTIME_CONFIGURATIONS[args.task_id // (len(DATASETS) * len(SEEDS))]
        dataset = DATASETS[(args.task_id // len(SEEDS)) % len(DATASETS)]
        seed = SEEDS[args.task_id % len(SEEDS)]
        from experiments.combined_objective.run_three_objective_direction_runtime_task import run_one
        result = {"path": str(run_one(configuration, dataset, seed,
                                      args.output_root / "three_direction_runtime",
                                      quality_prior_variance_mode=args.quality_prior_variance_mode))}
    print(json.dumps({"kind": args.kind, "task_id": args.task_id, "result": result}), flush=True)


if __name__ == "__main__":
    main()
