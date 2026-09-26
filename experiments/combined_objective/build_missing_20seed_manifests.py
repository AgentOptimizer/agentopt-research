#!/usr/bin/env python3
"""Build Slurm manifests containing only missing 20-seed result cells."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "analysis/20seed_results/inventory.json"
OUTPUT = ROOT / "analysis/complete_20seed_runs/manifests"
GROUPS = {
    "ege_sr": "ege_sr",
    "ape_k": "ape_k",
    "qnehvi": "qnehvi",
    "random_configurations": "random",
    "random_questions": "random",
}


def main() -> None:
    subprocess.run(
        ["python", str(ROOT / "experiments/combined_objective/organize_20seed_results.py")],
        cwd=ROOT,
        check=True,
    )
    records = json.loads(INDEX.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for record in records:
        method = str(record["method"])
        group = GROUPS.get(method)
        if group is None:
            continue
        for seed in record["missing_seeds"]:
            grouped[group].append((method, str(record["dataset"]), int(seed)))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for group in sorted(set(GROUPS.values())):
        rows = sorted(grouped[group], key=lambda row: (row[0], row[1], row[2]))
        path = OUTPUT / f"{group}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("method", "dataset", "seed"))
            writer.writerows(rows)
        counts[group] = len(rows)
        print(f"{group}: {len(rows)} tasks -> {path}")

    (OUTPUT / "counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
