#!/usr/bin/env python3
"""Run one method/dataset/seed job for the unified USD-checkpoint dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_cost_budget_random_search import (
    RANDOM_CONFIGURATIONS,
    RANDOM_QUESTIONS,
    simulate_cost_budget_random_search,
)
from experiments.combined_objective.offline_pareto_baselines import QNEHVI
from experiments.combined_objective.run_20seed_baseline_task import (
    DATASETS,
    _atomic_csv,
    _atomic_json,
    _atomic_vectors,
    _dataset_path,
    _json_default,
    _run_pareto,
    _source_metadata,
)
from experiments.single_objective.offline_selector_sim import load_scope


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "analysis/usd_cost_checkpoints_20seed/runs"
PARETO_METHODS = (EGE_SH, EGE_SR, APE_K, QNEHVI)
RANDOM_METHODS = (RANDOM_CONFIGURATIONS, RANDOM_QUESTIONS)
METHODS = PARETO_METHODS + RANDOM_METHODS
REQUIRED_EVENTS = {"cost_checkpoint_10pct", "cost_checkpoint_30pct"}


def _is_complete(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    trajectory_path = output_dir / "cost_trajectory.csv"
    vectors_path = output_dir / "vectors.npz"
    if not (summary_path.is_file() and trajectory_path.is_file() and vectors_path.is_file()):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        version = int(
            summary.get("params", {}).get(
                "recommendation_checkpoint_schema_version", 0
            )
        )
        with trajectory_path.open(newline="", encoding="utf-8") as handle:
            events = {row.get("event", "") for row in csv.DictReader(handle)}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return version >= 3 and REQUIRED_EVENTS <= events


def _run_random_cost(
    dataset: str,
    method: str,
    seed: int,
    models: list[str],
    questions: list[int],
    table: object,
) -> tuple[dict[str, object], list[dict[str, object]], tuple[object, ...]]:
    result = simulate_cost_budget_random_search(
        models,
        questions,
        table,
        method=method,
        seed=seed,
        cost_checkpoint_fractions=(0.10, 0.30),
    )
    rows = [{"benchmark": dataset, **row} for row in result.trajectory]
    summary = {"benchmark": dataset, **result.summary}
    vectors = (
        result.estimated_raw_vectors,
        result.truth_raw_vectors,
        result.truth_vectors,
    )
    return summary, rows, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--qnehvi-candidate-batch-size",
        type=int,
        default=int(os.environ.get("QNEHVI_CANDIDATE_BATCH_SIZE", "512")),
    )
    parser.add_argument(
        "--stop-after-recommendation-interval-fraction",
        type=float,
        default=float(
            os.environ.get("QNEHVI_STOP_AFTER_INTERVAL_FRACTION", "0")
        ),
        help=(
            "For qNEHVI, stop after the recommendation active at this realized "
            "USD fraction changes; zero disables early stopping."
        ),
    )
    args = parser.parse_args()
    if args.seed not in range(42, 62):
        parser.error("--seed must be in [42, 61]")

    output_dir = args.output_root / args.dataset / args.method / f"seed-{args.seed}"
    if _is_complete(output_dir):
        print(f"SKIP validated USD checkpoints: {output_dir}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _source_metadata()
    models, questions, table = load_scope(str(_dataset_path(args.dataset)))
    if args.method in PARETO_METHODS:
        summary, rows, vectors = _run_pareto(
            args.dataset,
            args.method,
            args.seed,
            models,
            questions,
            table,
            (
                None
                if args.qnehvi_candidate_batch_size == 0
                else args.qnehvi_candidate_batch_size
            ),
            (
                None
                if args.stop_after_recommendation_interval_fraction == 0.0
                else args.stop_after_recommendation_interval_fraction
            ),
        )
    else:
        summary, rows, vectors = _run_random_cost(
            args.dataset,
            args.method,
            args.seed,
            models,
            questions,
            table,
        )
    summary.update(metadata)
    _atomic_csv(rows, output_dir / "cost_trajectory.csv")
    _atomic_vectors(*vectors, output_dir / "vectors.npz")
    _atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
