#!/usr/bin/env python3
"""Write one-run-per-row manifests for USD 10%/30% checkpoint replays."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.combined_objective.run_cost_checkpoint_20seed_task import (
    DATASETS,
    METHODS,
    ROOT,
)


OUTPUT = ROOT / "analysis/usd_cost_checkpoints_20seed/manifests"
SEEDS = tuple(range(42, 62))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        path = OUTPUT / f"{method}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("method", "dataset", "seed"))
            for dataset in DATASETS:
                for seed in SEEDS:
                    writer.writerow((method, dataset, seed))
        print(f"{method}: {len(DATASETS) * len(SEEDS)} runs -> {path}")


if __name__ == "__main__":
    main()

