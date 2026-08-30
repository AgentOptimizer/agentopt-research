#!/usr/bin/env python3
"""Aggregate full-trajectory and 10%-budget qNEHVI runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "analysis/paper_20seed_method_comparison/qnehvi"
SEEDS = tuple(range(42, 58)) + (59, 60, 61, 62)
BENCHMARKS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}


def main() -> None:
    trajectory_rows = []
    recommendation_rows = []
    missing = []
    for benchmark, label in BENCHMARKS.items():
        for seed in SEEDS:
            full_path = OUTDIR / "full" / "raw" / benchmark / f"seed-{seed}" / "cost_trajectory.csv"
            summary_path = OUTDIR / "10pct" / "raw" / benchmark / f"seed-{seed}" / "summary.json"
            if not full_path.exists():
                missing.append(full_path)
            else:
                with full_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        trajectory_rows.append({"benchmark": label, **row})
            if not summary_path.exists():
                missing.append(summary_path)
            else:
                result = json.loads(summary_path.read_text(encoding="utf-8"))["qnehvi"]
                params = result["params"]
                recommendation_rows.append({
                    "benchmark": benchmark,
                    "method": "qnehvi",
                    "seed": seed,
                    "budget_fraction": params["observation_budget_fraction"],
                    "actual_cost_fraction": result["mean_cost_usd"] / params["bruteforce_search_cost_usd"],
                    "selected_models": json.dumps(result["final_models"]),
                })
    if missing:
        raise SystemExit(f"missing {len(missing)} outputs; first: {missing[0]}")
    for destination, rows in (
        (OUTDIR / "cost_trajectory.csv", trajectory_rows),
        (OUTDIR / "recommendations_10pct.csv", recommendation_rows),
    ):
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {destination} with {len(rows)} rows")


if __name__ == "__main__":
    main()

