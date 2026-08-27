#!/usr/bin/env python3
"""Collect the 40%-budget recommendation sets used by the 2x6 plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "analysis/paper_20seed_method_comparison/pareto_baselines_40pct"
SEEDS = tuple(range(42, 58)) + (59, 60, 61, 62)
BENCHMARKS = ("hotpotqa", "mathqa")
METHODS = ("ege_sh", "ape_k")


def main() -> None:
    rows = []
    missing = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for seed in SEEDS:
                path = OUTDIR / "raw" / benchmark / method / f"seed-{seed}" / "summary.json"
                if not path.exists():
                    missing.append(path)
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))[method]
                params = result["params"]
                rows.append({
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "budget_fraction": params["observation_budget_fraction"],
                    "actual_cost_fraction": (
                        result["mean_cost_usd"] / params["bruteforce_search_cost_usd"]
                    ),
                    "selected_models": json.dumps(result["final_models"]),
                })
    if missing:
        raise SystemExit(f"missing {len(missing)} outputs; first: {missing[0]}")
    destination = OUTDIR / "recommendations.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {destination} with {len(rows)} runs")


if __name__ == "__main__":
    main()

