"""Compare recommendation rules on matched, full-query acquisition trajectories."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

from experiments.combined_objective.offline_multiobjective_random_search import mean_raw_vectors
from experiments.combined_objective.offline_radial_gittins import (
    front_quality_metrics,
    hypervolume_2d,
    mean_cost_metric_vectors,
    raw_pareto_front_indices,
)
from experiments.combined_objective.cost_aware_replay import (
    CC_GITTINS,
    DEFAULT_METHODS,
    GITTINS_ACQUISITION_TARGET,
    METHODS,
    RANDOM_QUESTIONS,
    READOUT,
    READOUTS,
    replay_search,
    write_json_once,
)
from experiments.combined_objective.run_two_direction_ablation import DATASETS, load_benchmark


TARGETS = tuple(i / 200 for i in range(61))
PROTOCOL = "paper_question_orders_and_warm_calibration"
DEFAULT_OUTDIR = Path("experiments/combined_objective/results/shared_readout")


def method_readouts(methods):
    """Pair posterior readouts and include the empirical Random Questions baseline."""
    return {
        method: READOUTS if method == RANDOM_QUESTIONS else ("finite_mean_beta0", READOUT)
        for method in methods
    }


def run_benchmark(benchmark, seeds, methods, outdir):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("methods must be distinct supported policies")
    readouts = method_readouts(methods)
    models, questions, table, hashes = load_benchmark(benchmark)
    full_cost = math.fsum(table[model][q].cost for model in models for q in questions)
    truth = mean_raw_vectors(models, questions, table)
    oracle = raw_pareto_front_indices(truth)
    oracle_ids = set(oracle)
    endpoint = max(oracle, key=lambda arm: (truth[arm, 0], -truth[arm, 1]))
    points, cost_reference = mean_cost_metric_vectors(truth)
    oracle_vectors = points[list(oracle)]
    origin = (0.0, 0.0)
    oracle_hv = hypervolume_2d(oracle_vectors, origin)

    @lru_cache(maxsize=None)
    def score(archive):
        metrics = front_quality_metrics(points[list(archive)], oracle_vectors, origin, oracle_hv)
        matched = len(oracle_ids.intersection(archive))
        return {
            "hv_regret": metrics.hypervolume_regret,
            "gd": metrics.generational_distance,
            "igd": metrics.inverted_generational_distance,
            "precision": matched / len(archive),
            "recall": matched / len(oracle),
            "endpoint_found": endpoint in archive,
        }

    outdir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        destination = outdir / f"{benchmark}-seed{seed}.json"
        metadata = {
            "benchmark": benchmark,
            "seed": seed,
            "methods": list(methods),
            "readouts_by_method": {method: list(rules) for method, rules in readouts.items()},
            "input_sha256": hashes,
            "protocol": PROTOCOL,
            "cc_gittins_acquisition_target": GITTINS_ACQUISITION_TARGET if CC_GITTINS in methods else None,
        }

        def check_existing():
            with destination.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            if any(existing.get(key) != value for key, value in metadata.items()):
                raise ValueError(f"{destination} uses a different protocol, method list, readout, or input matrix. "
                                 "Use a separate --outdir for this comparison.")
            print(f"Already complete: {destination}", flush=True)

        if destination.exists():
            check_existing()
            continue

        result = replay_search(
            models, questions, table, full_cost, seed,
            methods=methods, save_curve=False, targets=TARGETS, readouts=readouts,
        )
        for row in result["rows"]:
            row.update(score(tuple(row["recommendations"])))
        result.update(
            metadata, full_cost_usd=full_cost, cost_reference_usd=cost_reference,
            truth_frontier=oracle, accuracy_endpoint_arm=endpoint,
        )
        if write_json_once(destination, result):
            print(f"Saved {destination}", flush=True)
        else:
            check_existing()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(DATASETS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(42, 62)))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be unique")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique nonnegative integers")
    run_benchmark(args.benchmark, args.seeds, tuple(args.methods), args.outdir)


if __name__ == "__main__":
    main()
