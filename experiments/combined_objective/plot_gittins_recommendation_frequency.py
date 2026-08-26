#!/usr/bin/env python3
"""Plot Gittins stopping recommendations across saved multi-seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    mean_raw_vectors,
)
from experiments.combined_objective.plot_random_recommendation_frequency import (
    BENCHMARK_PICKLES,
    SavedRecommendation,
    plot_recommendation_frequency,
    plot_single_and_frequency_panel,
    plot_single_run,
)
from experiments.single_objective.offline_selector_sim import load_pickle


DISPLAY_NAMES = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}


def load_gittins_recommendations(
    path: Path, benchmark: str,
) -> list[SavedRecommendation]:
    """Load the deployable archive recorded at each run's Gittins stop."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["benchmark"] == DISPLAY_NAMES[benchmark]
        ]
    if not rows:
        raise ValueError(f"no {DISPLAY_NAMES[benchmark]} rows found in {path}")
    return [
        SavedRecommendation(
            seed=int(row["seed"]),
            budget_fraction=float(row["gittins_stop_budget_fraction"]),
            selected_models=tuple(json.loads(row["selected_models"])),
            # This field is not used by the plotting functions. Store a
            # monotone quality surrogate to keep the shared record type small.
            hypervolume_ratio=1.0 - float(row["stop_hv_regret"]),
        )
        for row in sorted(rows, key=lambda row: int(row["seed"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_PICKLES), default="hotpotqa")
    parser.add_argument("--single-seed", type=int, default=42)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "analysis/vs/method_comparison_20seeds/gittins_seed_results.csv",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "analysis/vs/method_comparison_20seeds/gittins_frequency_frontier",
    )
    args = parser.parse_args()

    models, datapoints, table = load_pickle(str(BENCHMARK_PICKLES[args.benchmark]))
    truth_vectors = mean_raw_vectors(
        models, common_question_ids(models, datapoints, table), table,
    )
    recommendations = load_gittins_recommendations(args.results, args.benchmark)
    try:
        single = next(row for row in recommendations if row.seed == args.single_seed)
    except StopIteration as error:
        raise ValueError(f"seed {args.single_seed} is not present in {args.results}") from error

    prefix = f"{args.benchmark}_gittins_stop"
    label = "Radial-Gittins"
    plot_single_run(
        truth_vectors, models, single, benchmark=args.benchmark,
        version_label=label, output_stem=args.outdir / f"{prefix}_single_run",
        budget_label="adaptive stopping",
    )
    plot_recommendation_frequency(
        truth_vectors, models, recommendations, benchmark=args.benchmark,
        version_label=label, output_stem=args.outdir / f"{prefix}_aggregate_frequency",
        budget_label="adaptive stopping",
    )
    plot_single_and_frequency_panel(
        truth_vectors, models, single, recommendations, benchmark=args.benchmark,
        version_label=label, output_stem=args.outdir / f"{prefix}_single_and_aggregate_panel",
        budget_label="adaptive stopping",
    )
    print(f"wrote Gittins frequency figures to {args.outdir}")


if __name__ == "__main__":
    main()
