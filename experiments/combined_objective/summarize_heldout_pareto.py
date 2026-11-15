"""Aggregate held-out Pareto replays at matched search-cost caps."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from itertools import product
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
from tempfile import NamedTemporaryFile


CAPS = tuple(i / 400 for i in range(121))
METRICS = ("hv", "hv_regret", "gd", "igd")
COST_TOLERANCE_USD = 1e-10
RUN_FILENAME = re.compile(r"(?P<benchmark>[a-z0-9_]+)-fold(?P<fold>[1-5])-seed(?P<seed>\d+)\.json")
FOLD_KEYS = (
    "models", "search_queries", "heldout_queries", "split_seed", "input_sha256",
    "full_search_cost_usd", "heldout_oracle_hv", "exhaustive_search_heldout_metrics",
)


def discover_runs(directory, benchmarks=None, folds=None, seeds=None):
    """Require one replay per benchmark, fold, and seed in the selected cohort."""
    paths = {}
    for path in directory.glob("*-fold*-seed*.json"):
        matched = RUN_FILENAME.fullmatch(path.name)
        if not matched:
            raise ValueError(f"Unrecognized replay filename: {path.name}")
        benchmark, fold, seed = matched.groups()
        key = (benchmark, int(fold), int(seed))
        if key in paths:
            raise ValueError(f"Duplicate replay: {path.name} and {paths[key].name}")
        paths[key] = path
    if not paths:
        raise ValueError(f"No held-out replay files found in {directory}")

    benchmarks = sorted(set(benchmarks or [key[0] for key in paths]))
    folds = sorted(set(folds or [key[1] for key in paths]))
    seeds = sorted(set(seeds or [key[2] for key in paths]))
    expected = set(product(benchmarks, folds, seeds))
    missing, extra = expected - paths.keys(), paths.keys() - expected
    if missing or extra:
        def examples(keys):
            return ", ".join(f"{benchmark}-fold{fold}-seed{seed}" for benchmark, fold, seed in sorted(keys)[:5])

        raise ValueError(
            f"Expected a complete grid of benchmarks, folds, and seeds. "
            f"Missing {len(missing)} ({examples(missing)}) and unexpected {len(extra)} ({examples(extra)})."
        )
    return paths, benchmarks, folds, seeds


def checked_metric(value, context):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: invalid metric value {value}")
    return value


def sample_curve(history, full_cost, warm_fraction, oracle_hv, context):
    """At each cap, use the last complete recommendation affordable at that cap."""
    max_cost = full_cost * CAPS[-1] + COST_TOLERANCE_USD
    if bool(history) != (warm_fraction * full_cost <= max_cost):
        raise ValueError(f"{context}: purchase history disagrees with warm-up cost")
    if history and not math.isclose(history[0]["actual_usd"], warm_fraction * full_cost,
                                    rel_tol=1e-10, abs_tol=COST_TOLERANCE_USD):
        raise ValueError(f"{context}: first purchase is not the complete warm-up")

    costs = []
    previous_cost, previous_evaluations = 0.0, 0
    for row in history:
        cost, evaluations = row["actual_usd"], row["evaluations"]
        if not previous_cost <= cost <= max_cost or evaluations <= previous_evaluations:
            raise ValueError(f"{context}: invalid purchase spend or evaluation count")
        costs.append(cost)
        previous_cost, previous_evaluations = cost, evaluations

    before_warm = {"hv": 0.0, "hv_regret": oracle_hv, "gd": None, "igd": None}
    rows = []
    for cap in CAPS:
        index = bisect_right(costs, full_cost * cap + COST_TOLERANCE_USD) - 1
        if index < 0:
            rows.append(before_warm)
        else:
            rows.append({metric: checked_metric(history[index][metric], context) for metric in METRICS})
    return rows


def summarize_fold(paths, benchmark, fold, seeds, methods, split_seed):
    fold_metadata = None
    samples = {}
    max_warm = 0.0
    snapshots = 0
    for seed in seeds:
        path = paths[benchmark, fold, seed]
        run = json.loads(path.read_text(encoding="utf-8"))
        if (run["benchmark"], run["fold"], run["seed"]) != (benchmark, fold, seed):
            raise ValueError(f"{path.name}: replay metadata disagrees with the filename")
        if run["readout"] != "finite_lcb_beta1" or run["split_seed"] != split_seed:
            raise ValueError(f"{path.name}: readout or split seed differs within the cohort")
        metadata = {key: run[key] for key in FOLD_KEYS}
        if fold_metadata is None:
            fold_metadata = metadata
        if metadata != fold_metadata:
            raise ValueError(f"{path.name}: fold inputs or reference metrics differ across seeds")
        full_cost = metadata["full_search_cost_usd"]
        if not math.isfinite(full_cost) or full_cost <= 0:
            raise ValueError(f"{path.name}: invalid exhaustive search-pool cost")
        oracle_hv = checked_metric(metadata["heldout_oracle_hv"], path.name)
        curves = run["curve_by_method"]
        warm_costs = run["warm_cost_fraction_by_method"]
        if set(curves) != set(methods) or curves.keys() != warm_costs.keys():
            raise ValueError(f"{path.name}: acquisition methods differ within the cohort")
        for method in methods:
            history = curves[method]
            warm = checked_metric(warm_costs[method], path.name)
            rows = sample_curve(history, full_cost, warm, oracle_hv, f"{path.name} ({method})")
            samples.setdefault(method, []).append(rows)
            max_warm = max(max_warm, warm)
            snapshots += len(history)

    means, warm_counts = {}, {}
    for method, seed_curves in samples.items():
        rows_by_cap = list(zip(*seed_curves))
        counts = [sum(row["gd"] is not None for row in rows) for rows in rows_by_cap]
        means[method] = {
            metric: [
                mean(row[metric] for row in rows)
                if metric in ("hv", "hv_regret") or count == len(seeds)
                else None
                for rows, count in zip(rows_by_cap, counts)
            ]
            for metric in METRICS
        }
        warm_counts[method] = counts
    return {
        "fold": fold,
        **fold_metadata,
        "max_warm_cost_fraction": max_warm,
        "warm_complete_seed_counts": warm_counts,
        "methods": means,
        "purchase_snapshots": snapshots,
    }


def summarize_benchmark(paths, benchmark, folds, seeds, methods, split_seed):
    summaries = [summarize_fold(paths, benchmark, fold, seeds, methods, split_seed) for fold in folds]
    all_queries, seen_heldout = None, set()
    first = summaries[0]
    for fold in summaries:
        search, heldout = set(fold["search_queries"]), set(fold["heldout_queries"])
        if (not search or not heldout or search & heldout or heldout & seen_heldout
                or len(search) != len(fold["search_queries"])
                or len(heldout) != len(fold["heldout_queries"])):
            raise ValueError(f"{benchmark}: invalid search and held-out query partition")
        if all_queries is None:
            all_queries = search | heldout
        if search | heldout != all_queries or any(
            fold[key] != first[key] for key in ("models", "input_sha256")
        ):
            raise ValueError(f"{benchmark}: inconsistent benchmark inputs across folds")
        seen_heldout.update(heldout)
    if len(folds) == 5 and seen_heldout != all_queries:
        raise ValueError(f"{benchmark}: five held-out folds do not cover all queries")

    counts = {
        method: [sum(seed_counts) for seed_counts in zip(
            *(fold["warm_complete_seed_counts"][method] for fold in summaries)
        )]
        for method in methods
    }
    total_runs = len(folds) * len(seeds)
    all_warm = [
        all(count == total_runs for count in counts_at_cap)
        for counts_at_cap in zip(*counts.values())
    ]
    all_warm_cap = next((cap for cap, eligible in zip(CAPS, all_warm) if eligible), None)
    means, deviations = {}, {}
    for method in methods:
        means[method], deviations[method] = {}, {}
        for metric in METRICS:
            eligible = all_warm if metric in ("gd", "igd") else [True] * len(CAPS)
            columns = list(zip(*(fold["methods"][method][metric] for fold in summaries)))
            means[method][metric] = [mean(values) if ready else None
                                     for ready, values in zip(eligible, columns)]
            deviations[method][metric] = [stdev(values) if ready and len(folds) > 1 else None
                                          for ready, values in zip(eligible, columns)]
    return {
        "methods": means,
        "fold_sd": deviations,
        "folds": summaries,
        "exhaustive_search_heldout_metrics": {
            metric: mean(checked_metric(fold["exhaustive_search_heldout_metrics"][metric], benchmark)
                         for fold in summaries)
            for metric in METRICS
        },
        "all_warm_cap": all_warm_cap,
        "max_warm_cost_fraction": max(fold["max_warm_cost_fraction"] for fold in summaries),
        "warm_complete_run_counts": counts,
        "purchase_snapshots": sum(fold["purchase_snapshots"] for fold in summaries),
        "model_count": len(first["models"]),
        "query_count": len(all_queries),
    }


def summarize(directory, benchmarks=None, folds=None, seeds=None):
    paths, benchmarks, folds, seeds = discover_runs(directory, benchmarks, folds, seeds)
    example = json.loads(paths[benchmarks[0], folds[0], seeds[0]].read_text(encoding="utf-8"))
    methods, split_seed = tuple(example["curve_by_method"]), example["split_seed"]
    if not methods:
        raise ValueError("Replay files contain no acquisition methods")
    return {
        "caps": CAPS,
        "search_seeds": seeds,
        "folds": folds,
        "split_seed": split_seed,
        "readout": "finite_lcb_beta1",
        "aggregation": "mean over seeds within each fold, then over folds",
        "before_warm_start": "empty archive for HV and regret. GD and IGD require warm-up for every method and run.",
        "benchmarks": {
            benchmark: summarize_benchmark(paths, benchmark, folds, seeds, methods, split_seed)
            for benchmark in benchmarks
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="Directory containing the run JSON files")
    parser.add_argument("--benchmarks", nargs="+", help="Expected benchmark IDs (default: discover)")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), help="Expected folds (default: discover)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Expected search seeds (default: discover)")
    args = parser.parse_args()
    try:
        report = summarize(args.outdir, args.benchmarks, args.folds, args.seeds)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Held-out summarization failed: {error}\n")

    destination = args.outdir / "curve_summary.json"
    with NamedTemporaryFile("w", dir=args.outdir, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except Exception:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
