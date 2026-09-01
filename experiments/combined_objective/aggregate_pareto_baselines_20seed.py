#!/usr/bin/env python3
"""Combine the 80 one-run baseline jobs into one plotting trajectory CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "analysis/paper_20seed_method_comparison/pareto_baselines"
SEEDS = tuple(range(42, 58)) + (59, 60, 61, 62)
BENCHMARKS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
METHODS = ("ege_sh", "ape_k")


def main() -> None:
    combined = []
    missing = []
    for benchmark, label in BENCHMARKS.items():
        for method in METHODS:
            for seed in SEEDS:
                path = OUTDIR / "raw" / benchmark / method / f"seed-{seed}" / "cost_trajectory.csv"
                if not path.exists():
                    missing.append(path)
                    continue
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        combined.append({"benchmark": label, **row})
    if missing:
        listing = "\n".join(str(path) for path in missing[:10])
        raise SystemExit(f"missing {len(missing)} run outputs; first paths:\n{listing}")
    destination = OUTDIR / "cost_trajectory.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"wrote {destination} from {len(SEEDS) * len(BENCHMARKS) * len(METHODS)} runs")


if __name__ == "__main__":
    main()

