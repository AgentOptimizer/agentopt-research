#!/usr/bin/env python3
"""Build a lightweight, method-first index of all known 20-seed results.

The index uses relative symbolic links, so it neither duplicates large result files
nor changes the paths consumed by existing plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = tuple(range(42, 62))
DATASETS = (
    "hotpotqa",
    "mathqa",
    "restaurant_test",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
)
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ege_sr",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
BENCHMARK_SLUGS = {
    "HotpotQA": "hotpotqa",
    "MathQA": "mathqa",
}


def ensure_link(link: Path, target: Path) -> None:
    """Create or refresh a relative symlink without replacing real files."""
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = Path(os.path.relpath(target, start=link.parent))
    if link.is_symlink():
        if Path(os.readlink(link)) == relative_target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")
    link.symlink_to(relative_target, target_is_directory=target.is_dir())


def csv_seed_index(
    path: Path,
    *,
    method_column: str = "method",
    dataset_column: str = "benchmark",
) -> dict[tuple[str, str], set[int]]:
    indexed: dict[tuple[str, str], set[int]] = defaultdict(set)
    if not path.is_file():
        return indexed
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = BENCHMARK_SLUGS.get(row[dataset_column], row[dataset_column].lower())
            indexed[(row[method_column], dataset)].add(int(row["seed"]))
    return indexed


def build_index(repo: Path, output: Path) -> list[dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    available: dict[tuple[str, str], set[int]] = defaultdict(set)
    source_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_roots: dict[tuple[str, str], set[str]] = defaultdict(set)

    # Current Gittins/Gauss-Radau run: one result directory per seed and dataset.
    gittins_root = (
        repo
        / "experiments/combined_objective/results"
        / "gauss_radau_8bench_20seed_independent"
    )
    for seed in EXPECTED_SEEDS:
        for dataset in DATASETS:
            source = (
                gittins_root
                / f"seed-{seed}"
                / "gauss_radau_accuracy_endpoint"
                / dataset
            )
            if (source / "result.json").is_file():
                ensure_link(output / "radial_gittins" / dataset / f"seed-{seed}", source)
                key = ("radial_gittins", dataset)
                available[key].add(seed)
                source_kinds[key].add("per_seed")
                source_roots[key].add(str(gittins_root.relative_to(repo)))

    # New and backfilled baselines store complete per-seed directories here.
    baseline_root = repo / "analysis/final_run_baselines"
    for dataset in DATASETS:
        for method in (
            "ege_sh",
            "ege_sr",
            "ape_k",
            "qnehvi",
            "random_configurations",
            "random_questions",
        ):
            for seed in EXPECTED_SEEDS:
                source = baseline_root / dataset / method / f"seed-{seed}"
                required = (
                    source / "summary.json",
                    source / "vectors.npz",
                    source / "cost_trajectory.csv",
                )
                if all(path.is_file() for path in required):
                    ensure_link(output / method / dataset / f"seed-{seed}", source)
                    key = (method, dataset)
                    available[key].add(seed)
                    source_kinds[key].add("per_seed")
                    source_roots[key].add(str(baseline_root.relative_to(repo)))

    # HotpotQA/MathQA EGE-SH, APE-k, and qNEHVI are available as plotting-ready
    # aggregate CSVs with a seed column rather than as uniform seed directories.
    continuous_root = repo / "analysis/continuous_seeds_42_61/data"
    aggregate_sources = (
        (
            continuous_root / "pareto_baselines/cost_trajectory.csv",
            continuous_root / "pareto_baselines/recommendations.csv",
            ("ege_sh", "ape_k"),
        ),
        (
            continuous_root / "qnehvi/cost_trajectory.csv",
            continuous_root / "qnehvi/recommendations_10pct.csv",
            ("qnehvi",),
        ),
    )
    for trajectory, recommendations, methods in aggregate_sources:
        indexed = csv_seed_index(trajectory)
        for method in methods:
            for dataset in ("hotpotqa", "mathqa"):
                seeds = indexed.get((method, dataset), set()) & set(EXPECTED_SEEDS)
                if not seeds:
                    continue
                destination = output / method / dataset
                ensure_link(destination / "aggregate_cost_trajectory.csv", trajectory)
                if recommendations.is_file():
                    ensure_link(destination / "aggregate_recommendations.csv", recommendations)
                key = (method, dataset)
                available[key].update(seeds)
                source_kinds[key].add("aggregate_csv")
                source_roots[key].add(str(trajectory.parent.relative_to(repo)))

    # The two random-search variants share one CSV per benchmark; the version
    # column distinguishes the algorithms.
    random_root = repo / "analysis/vs/random_search_20seeds"
    for dataset in ("hotpotqa", "mathqa"):
        source = random_root / dataset / "multi_seed_results.csv"
        # These files have no benchmark column, so read their version/seed
        # pairs directly; the dataset is encoded in the parent directory.
        versions: dict[str, set[int]] = defaultdict(set)
        if source.is_file():
            with source.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    versions[row["version"]].add(int(row["seed"]))
        for method in ("random_configurations", "random_questions"):
            seeds = versions.get(method, set()) & set(EXPECTED_SEEDS)
            if not seeds:
                continue
            ensure_link(output / method / dataset / "multi_seed_results.csv", source)
            key = (method, dataset)
            available[key].update(seeds)
            source_kinds[key].add("aggregate_csv")
            source_roots[key].add(str(source.parent.relative_to(repo)))

    records: list[dict[str, object]] = []
    expected = set(EXPECTED_SEEDS)
    for method in METHODS:
        (output / method).mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            seeds = available[(method, dataset)] & expected
            missing = expected - seeds
            status = "complete" if not missing else "partial" if seeds else "missing"
            records.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "status": status,
                    "available_seed_count": len(seeds),
                    "expected_seed_count": len(EXPECTED_SEEDS),
                    "available_seeds": sorted(seeds),
                    "missing_seeds": sorted(missing),
                    "source_kind": sorted(source_kinds[(method, dataset)]),
                    "source_roots": sorted(source_roots[(method, dataset)]),
                }
            )
    return records


def write_inventory(output: Path, records: list[dict[str, object]]) -> None:
    with (output / "inventory.json").open("w") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")

    columns = (
        "method",
        "dataset",
        "status",
        "available_seed_count",
        "expected_seed_count",
        "available_seeds",
        "missing_seeds",
        "source_kind",
        "source_roots",
    )
    with (output / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in ("available_seeds", "missing_seeds", "source_kind", "source_roots"):
                row[field] = ";".join(str(value) for value in row[field])
            writer.writerow(row)

    complete_by_method = {
        method: sum(
            record["status"] == "complete"
            for record in records
            if record["method"] == method
        )
        for method in METHODS
    }
    lines = [
        "# Unified 20-seed result index",
        "",
        "This directory is a lightweight index. Seed directories and aggregate files are",
        "relative symbolic links to the original results; no large result files were copied",
        "or moved, so existing plotting scripts continue to work.",
        "",
        "Expected seeds: `42` through `61`. Use `inventory.csv` or `inventory.json` to",
        "check completeness before plotting. `complete` means all 20 expected seeds exist.",
        "",
        "| Method | Complete datasets (out of 8) |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{method}` | {complete_by_method[method]} |" for method in METHODS
    )
    lines.extend(
        [
            "",
            "`radial_gittins` points to the current 8-benchmark Gauss-Radau run.",
            "Missing or partial cells remain visible in the inventory until their jobs finish.",
            "HotpotQA/MathQA aggregate CSVs retain their `seed` column and can be filtered",
            "directly by plotting code.",
            "",
            "Regenerate this index from the repository root with:",
            "",
            "```bash",
            "python experiments/combined_objective/organize_20seed_results.py",
            "```",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "analysis/20seed_results",
        help="Output index directory (default: analysis/20seed_results)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every method/dataset cell has all 20 seeds",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    records = build_index(repo, output)
    write_inventory(output, records)
    complete = sum(record["status"] == "complete" for record in records)
    partial = sum(record["status"] == "partial" for record in records)
    print(f"Wrote {output}")
    print(f"Method/dataset pairs: {complete} complete, {partial} partial")
    if args.require_complete:
        incomplete = [record for record in records if record["status"] != "complete"]
        if incomplete:
            for record in incomplete:
                print(
                    f"INCOMPLETE {record['method']}/{record['dataset']}: "
                    f"{record['available_seed_count']}/{record['expected_seed_count']}"
                )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
