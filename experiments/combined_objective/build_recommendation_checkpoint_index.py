#!/usr/bin/env python3
"""Build a unified 10%-spaced recommendation index and missing-run manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/final_run_baselines"
RESULTS_ROOT = ROOT / "analysis/20seed_results"
OUTPUT = RESULTS_ROOT / "recommendation_checkpoints.csv"
MANIFEST_DIR = ROOT / "analysis/job_manifests/recommendation_checkpoints"
DATASETS = (
    "hotpotqa", "mathqa", "restaurant_test", "stackoverflow", "bird_dev",
    "restaurant_valid", "bing_querylogs", "bird_mini_dev",
)
METHODS = ("ege_sh", "ege_sr", "ape_k", "qnehvi")
SEEDS = tuple(range(42, 62))
TARGETS = tuple(i / 10 for i in range(1, 11))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if "selected_arm_indices" in row]


def _checkpoint(rows: list[dict[str, str]], target: float) -> dict[str, str] | None:
    event = f"cost_checkpoint_{int(round(100 * target))}pct"
    explicit = [row for row in rows if row.get("event") == event]
    if explicit:
        return explicit[-1]
    eligible = [
        row for row in rows
        if row.get("selected_arm_indices") is not None
        and float(row["budget_fraction"]) <= target + 1e-12
    ]
    if eligible:
        return max(
            eligible,
            key=lambda row: (
                float(row["budget_fraction"]),
                int(float(row["cumulative_evaluations"])),
            ),
        )
    return None


def _trajectory_path(method: str, dataset: str, seed: int) -> Path:
    primary = RUN_ROOT / dataset / method / f"seed-{seed}" / "cost_trajectory.csv"
    if primary.is_file():
        return primary
    return RESULTS_ROOT / method / dataset / f"seed-{seed}" / "cost_trajectory.csv"


def build_index() -> tuple[list[dict[str, object]], list[tuple[str, str, int]]]:
    output: list[dict[str, object]] = []
    missing: list[tuple[str, str, int]] = []
    for method in METHODS:
        for dataset in DATASETS:
            for seed in SEEDS:
                path = _trajectory_path(method, dataset, seed)
                rows = _read_rows(path)
                cell_missing = False
                for target in TARGETS:
                    row = _checkpoint(rows, target)
                    if row is None:
                        cell_missing = True
                        continue
                    output.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "seed": seed,
                            "target_cost_fraction": target,
                            "actual_cost_fraction": float(row["budget_fraction"]),
                            "cumulative_search_cost_usd": float(
                                row["cumulative_search_cost_usd"]
                            ),
                            "cumulative_evaluations": int(
                                float(row["cumulative_evaluations"])
                            ),
                            "selected_arm_indices": row["selected_arm_indices"],
                            "selected_models": row.get("selected_models", ""),
                            "source_event": row.get("event", ""),
                            "source_path": str(path.relative_to(ROOT)),
                        }
                    )
                if cell_missing:
                    missing.append((method, dataset, seed))
    return output, missing


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "method", "dataset", "seed", "target_cost_fraction",
        "actual_cost_fraction", "cumulative_search_cost_usd",
        "cumulative_evaluations", "selected_arm_indices", "selected_models",
        "source_event", "source_path",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_manifest(rows: list[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("method", "dataset", "seed"))
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Write new missing-cell manifests; keep off while arrays are active.",
    )
    args = parser.parse_args()

    rows, missing = build_index()
    _write_csv(rows, args.output)
    if args.write_manifests:
        _write_manifest(
            [row for row in missing if row[0] != "qnehvi"],
            args.manifest_dir / "non_qnehvi.tsv",
        )
        qnehvi = [row for row in missing if row[0] == "qnehvi"]
        _write_manifest(
            [row for row in qnehvi if row[1] in {"hotpotqa", "mathqa"}],
            args.manifest_dir / "qnehvi_small.tsv",
        )
        _write_manifest(
            [row for row in qnehvi if row[1] not in {"hotpotqa", "mathqa"}],
            args.manifest_dir / "qnehvi_large.tsv",
        )
    complete_cells = len(METHODS) * len(DATASETS) * len(SEEDS) - len(missing)
    audit = {
        "expected_cells": len(METHODS) * len(DATASETS) * len(SEEDS),
        "complete_cells": complete_cells,
        "missing_cells": len(missing),
        "indexed_checkpoints": len(rows),
        "targets": list(TARGETS),
        "missing_by_method": {
            method: sum(1 for row in missing if row[0] == method)
            for method in METHODS
        },
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"wrote {args.output}")
    if args.write_manifests:
        print(f"wrote manifests under {args.manifest_dir}")


if __name__ == "__main__":
    main()
