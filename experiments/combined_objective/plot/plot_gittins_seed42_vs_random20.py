#!/usr/bin/env python3
"""Plot seed-42 Gittins against 20-seed random-search aggregates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.plot.plot_multiobjective_method_comparison import (  # noqa: E402
    RANDOM_STYLES,
    write_comparison_figure,
)


BENCHMARKS = {"HotpotQA": "hotpotqa", "MathQA": "mathqa"}


def _random_curves(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    full_cost = {
        (row["version"], int(row["seed"])): float(row["total_search_cost_usd"])
        for row in rows
        if abs(float(row["budget_fraction"]) - 1.0) <= 1e-12
    }
    curves = []
    for version in RANDOM_STYLES:
        grouped = defaultdict(list)
        grouped_x = defaultdict(list)
        for row in rows:
            if row["version"] != version:
                continue
            fraction = float(row["budget_fraction"])
            seed = int(row["seed"])
            grouped[fraction].append(float(row["hypervolume_regret"]))
            grouped_x[fraction].append(
                float(row["total_search_cost_usd"]) / full_cost[(version, seed)]
            )
        fractions = sorted(grouped)
        xs = np.asarray([np.mean(grouped_x[f]) for f in fractions])
        means = np.asarray([np.mean(grouped[f]) for f in fractions])
        ci95 = np.asarray(
            [
                1.96 * np.std(grouped[f], ddof=1) / np.sqrt(len(grouped[f]))
                for f in fractions
            ]
        )
        curves.append((version, xs, means, ci95))
    return curves, max(full_cost.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radial-dir", type=Path, default=ROOT / "analysis/vs/radial_gittins"
    )
    parser.add_argument(
        "--random-dir", type=Path,
        default=ROOT / "analysis/vs/random_search_20seeds",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            ROOT
            / "experiments/combined_objective/plots/qa_random_20seeds"
            / "gittins_seed42_vs_random20.png"
        ),
    )
    args = parser.parse_args()
    summary = json.loads(
        (args.radial_dir / "summary.json").read_text(encoding="utf-8")
    )
    trajectory_rows = list(
        csv.DictReader(
            (args.radial_dir / "cost_trajectory.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    panels = []
    for benchmark, slug in BENCHMARKS.items():
        random_curves, full_cost = _random_curves(
            args.random_dir / slug / "multi_seed_results.csv"
        )
        rows = [row for row in trajectory_rows if row["benchmark"] == benchmark]
        xs = np.asarray(
            [float(row["cumulative_search_cost_usd"]) / full_cost for row in rows]
        )
        deployable = np.asarray([float(row["deployable_hv_regret"]) for row in rows])
        provisional = np.asarray([float(row["provisional_hv_regret"]) for row in rows])
        zeros = np.zeros_like(xs)
        ones = np.ones_like(xs)
        stop = float(summary[benchmark]["gittins_stop_cost_usd"]) / full_cost
        panels.append(
            {
                "name": benchmark,
                "stop_mean": stop,
                "deployable": (xs, deployable, zeros, ones),
                "provisional": (xs, provisional, zeros, ones),
                "random": random_curves,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_figure(
        out_path=args.output,
        title="Completed-only Gittins from stopping vs random search",
        panels=panels,
        seeds=20,
        seed=42,
        x_axis="cost",
        gittins_seeds=1,
    )


if __name__ == "__main__":
    main()
