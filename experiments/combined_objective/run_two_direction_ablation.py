#!/usr/bin/env python3
"""Run fixed two-direction Radial-Gittins ablations on the five main benchmarks.

This runner deliberately keeps acquisition directions fixed.  Adaptive direction
insertion is a separate algorithmic question and is not mixed into these runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.compare_lcb_recommendations import (  # noqa: E402
    compact_lcb_run,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR,
    load_pickle,
    load_scope,
    simulate_radial_gittins,
)


DATASETS = {
    "hotpotqa": ("pickle", "experiments/data/lookup/hotpotqa_lookup.pkl"),
    "mathqa": ("pickle", "experiments/data/lookup/mathqa_lookup.pkl"),
    "stackoverflow": ("scope", "data/scope/stackoverflow"),
    "bird_dev": ("scope", "data/scope/bird_dev"),
    "restaurant_valid": ("scope", "data/scope/restaurant_valid"),
}

GAUSS_RADAU_TWO_POINT_INTERIOR = 1.0 / 3.0

PAIRS = {
    "cost_near__accuracy_axis": ((0.1, 0.9), (1.0, 0.0)),
    "gauss_radau_accuracy_endpoint": (
        (GAUSS_RADAU_TWO_POINT_INTERIOR, 1.0 - GAUSS_RADAU_TWO_POINT_INTERIOR),
        (1.0, 0.0),
    ),
    "symmetric_interior": ((0.1, 0.9), (0.9, 0.1)),
    "exact_axes": ((0.0, 1.0), (1.0, 0.0)),
}
PRIMARY_PAIR = "cost_near__accuracy_axis"

DEFAULT_OUTDIR = (
    ROOT
    / "experiments/combined_objective/results"
    / "two_direction_finite_lcb_raw_mean_seed42_independent"
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _input_files(kind: str, path: Path) -> list[Path]:
    if kind == "pickle":
        return [path]
    return [path / name for name in ("accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json")]


def load_benchmark(benchmark: str):
    kind, relative = DATASETS[benchmark]
    path = ROOT / relative
    files = _input_files(kind, path)
    hashes = {
        str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in files
    }
    models, questions, table = (load_pickle if kind == "pickle" else load_scope)(str(path))
    return models, questions, table, hashes


def _first_point(points: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    return next((point for point in points if predicate(point)), None)


def _sustained_first_point(points: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    suffix = np.logical_and.accumulate(
        np.asarray([bool(predicate(point)) for point in points], dtype=bool)[::-1]
    )[::-1]
    return next((point for point, keep in zip(points, suffix) if keep), None)


def _fraction(point: dict[str, Any] | None) -> float | None:
    return None if point is None else float(point["cost_fraction"])


def enrich_frontier_metrics(run: dict[str, Any]) -> dict[str, Any]:
    truth = set(run["full_data_pareto_arm_indices"])
    endpoint = set(run["true_best_accuracy_arm_indices"])
    for point in run["points"]:
        selected = set(point["selected_arm_indices"])
        true_positive = selected & truth
        point.update(
            pareto_true_positive_count=len(true_positive),
            pareto_false_positive_count=len(selected - truth),
            pareto_false_negative_count=len(truth - selected),
            pareto_precision=(len(true_positive) / len(selected) if selected else 1.0),
            pareto_recall=(len(true_positive) / len(truth) if truth else 1.0),
            exact_frontier=(selected == truth),
            covers_frontier=truth.issubset(selected),
            contains_accuracy_endpoint=bool(selected & endpoint),
        )
    return run


def _eta_summary(events: Iterable[dict[str, Any]], direction_count: int) -> list[dict[str, Any]]:
    events = list(events)
    rows = []
    for index in range(direction_count):
        local = [event for event in events if int(event["direction_index"]) == index]
        decays = [event for event in local if event["event"] == "direction_eta_decay"]
        floors = [event for event in local if event["event"] == "direction_eta_floor_stop"]
        first = local[0] if local else None
        last = local[-1] if local else None
        first_floor = floors[0] if floors else None
        rows.append(
            {
                "direction_index": index,
                "direction": None if first is None else first["direction"],
                "eta_decay_count": len(decays),
                "eta_floor_visit_count": len(floors),
                "first_stop_eta": None if first is None else first["current_lambda"],
                "first_stop_bf_fraction": None if first is None else first["budget_fraction"],
                "first_floor_eta": None if first_floor is None else first_floor["current_lambda"],
                "first_floor_stage": None if first_floor is None else first_floor["lambda_stage"],
                "first_floor_bf_fraction": None if first_floor is None else first_floor["budget_fraction"],
                "last_event": None if last is None else last["event"],
                "last_eta": None if last is None else last["current_lambda"],
                "last_stage": None if last is None else last["lambda_stage"],
                "last_event_bf_fraction": None if last is None else last["budget_fraction"],
                "floor_reason": None if last is None else last.get("floor_reason"),
            }
        )
    return rows


def summarize_run(run: dict[str, Any], eta_events: list[dict[str, Any]]) -> dict[str, Any]:
    points = run["points"]
    first_endpoint = _first_point(points, lambda point: point["contains_accuracy_endpoint"])
    sustained_endpoint = _sustained_first_point(points, lambda point: point["contains_accuracy_endpoint"])
    first_zero_fp = _first_point(
        points,
        lambda point: bool(point["selected_arm_indices"])
        and point["pareto_false_positive_count"] == 0,
    )
    sustained_zero_fp = _sustained_first_point(
        points,
        lambda point: bool(point["selected_arm_indices"])
        and point["pareto_false_positive_count"] == 0,
    )
    first_exact = _first_point(points, lambda point: point["exact_frontier"])
    sustained_exact = _sustained_first_point(points, lambda point: point["exact_frontier"])
    first_cover = _first_point(points, lambda point: point["covers_frontier"])
    final = points[-1]
    observed_start = float(points[0]["cost_fraction"])
    observed_end = float(run["cost_fraction"])
    fp_width = 0.0
    for index, point in enumerate(points):
        left = max(observed_start, float(point["cost_fraction"]))
        right = min(
            observed_end,
            float(points[index + 1]["cost_fraction"]) if index + 1 < len(points) else observed_end,
        )
        if point["pareto_false_positive_count"]:
            fp_width += max(0.0, right - left)
    summary = {
        "stop_reason": run["stop_reason"],
        "terminal_bf_fraction": run["cost_fraction"],
        "terminal_evaluations": run["evaluations"],
        "truth_frontier_size": len(run["full_data_pareto_arm_indices"]),
        "first_accuracy_endpoint_bf_fraction": _fraction(first_endpoint),
        "sustained_accuracy_endpoint_bf_fraction": _fraction(sustained_endpoint),
        "first_nonempty_zero_fp_bf_fraction": _fraction(first_zero_fp),
        "sustained_nonempty_zero_fp_bf_fraction": _fraction(sustained_zero_fp),
        "fraction_observed_spend_with_any_false_positive": (
            fp_width / (observed_end - observed_start)
            if observed_end > observed_start
            else 0.0
        ),
        "first_full_recall_bf_fraction": _fraction(first_cover),
        "first_exact_frontier_bf_fraction": _fraction(first_exact),
        "sustained_exact_frontier_bf_fraction": _fraction(sustained_exact),
        "final_precision": final["pareto_precision"],
        "final_recall": final["pareto_recall"],
        "final_false_positive_count": final["pareto_false_positive_count"],
        "final_false_negative_count": final["pareto_false_negative_count"],
        "final_relative_hv_regret": final["relative_hv_regret"],
        "final_selected_arm_indices": final["selected_arm_indices"],
        "direction_eta": _eta_summary(eta_events, 2),
    }
    for target in (0.5, 0.8, 0.9, 1.0):
        tag = str(target).replace(".", "p")
        summary[f"first_recall_{tag}_bf_fraction"] = _fraction(
            _first_point(points, lambda point, threshold=target: point["pareto_recall"] >= threshold - 1e-12)
        )
        summary[f"first_recall_{tag}_with_zero_fp_bf_fraction"] = _fraction(
            _first_point(
                points,
                lambda point, threshold=target: point["pareto_recall"] >= threshold - 1e-12
                and point["pareto_false_positive_count"] == 0,
            )
        )
    return summary


def run_one(
    benchmark: str,
    pair_name: str,
    *,
    seed: int,
    outdir: Path,
    observation_budget_fraction: float,
) -> dict[str, Any]:
    directions = PAIRS[pair_name]
    output_path = outdir / pair_name / benchmark / "result.json"
    if output_path.exists():
        print(f"Reusing {output_path}", flush=True)
        return json.loads(output_path.read_text())

    models, questions, table, hashes = load_benchmark(benchmark)
    cache = RadialGittinsBoundaryCache(cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR)
    started = time.perf_counter()
    print(
        f"Running {benchmark} / {pair_name}: directions={directions}, seed={seed}",
        flush=True,
    )
    result = simulate_radial_gittins(
        models,
        questions,
        table,
        directions=directions,
        anytime=True,
        direction_scheduler="round_robin",
        eta_decay_schedule="direction_stop",
        recommendation_rule="finite_lcb",
        recommendation_beta=1.0,
        recommendation_min_samples=0,
        question_order="independent",
        warm_start_question_order="independent",
        cost_model="raw_mean",
        seed=seed,
        batch_size=4,
        warm_start_batch_size=4,
        lambda_initial=1.0,
        lambda_decay=0.5,
        search_cost_scale_eta=1.0,
        observation_budget_fraction=observation_budget_fraction,
        boundary_z_padding_extra=2.0,
        effective_cost_bin_ratio=2.0,
        boundary_grid=RadialGittinsGrid(
            z_size=129,
            delta_size=129,
            state_size=129,
            boundary_margin_cells=4,
        ),
        boundary_cache=cache,
        record_trace=True,
        record_recommendation_trajectory=True,
        recommendation_checkpoint_interval=None,
        recommendation_changes_only=True,
        defer_recommendation_diagnostics=True,
    )
    elapsed = time.perf_counter() - started
    run = enrich_frontier_metrics(compact_lcb_run(result))
    eta_events = list(result.direction_eta_events)
    payload = {
        "config": {
            "benchmark": benchmark,
            "seed": seed,
            "directions": [list(direction) for direction in directions],
            "pair_name": pair_name,
            "question_universe": "common",
            "question_order": "independent",
            "warm_start_question_order": "independent",
            "batch_size": 4,
            "recommendation_rule": "finite_lcb",
            "recommendation_beta": 1.0,
            "recommendation_min_samples": 0,
            "cost_model": "raw_mean",
            "eta_initial": 1.0,
            "eta_decay": 0.5,
            "eta_decay_schedule": "direction_stop",
            "direction_scheduler": "round_robin",
            "observation_budget_fraction": observation_budget_fraction,
            "input_sha256": hashes,
            "engine_source_sha256": hashlib.sha256(
                (ROOT / "experiments/combined_objective/offline_radial_gittins.py").read_bytes()
            ).hexdigest(),
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "wall_time_seconds": elapsed,
        },
        "summary": summarize_run(run, eta_events),
        "direction_eta_events": eta_events,
        "run": run,
        "parameters": result.params,
    }
    _json_dump(output_path, payload)
    print(
        f"Finished {benchmark} / {pair_name}: BF={run['cost_fraction']:.3%}, "
        f"stop={run['stop_reason']}, time={elapsed:.1f}s",
        flush=True,
    )
    return payload


def export_combined(outdir: Path, benchmarks: list[str], pairs: list[str]) -> None:
    rows = []
    for pair_name in pairs:
        for benchmark in benchmarks:
            path = outdir / pair_name / benchmark / "result.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            payload["summary"] = summarize_run(
                payload["run"], payload["direction_eta_events"]
            )
            _json_dump(path, payload)
            rows.append(
                {
                    "pair_name": pair_name,
                    "directions": payload["config"]["directions"],
                    "benchmark": benchmark,
                    **payload["summary"],
                }
            )
    _json_dump(outdir / "summary.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=tuple(PAIRS),
        default=[PRIMARY_PAIR],
        help=(
            "Direction pairs to run; defaults to the primary "
            "(0.1, 0.9) + (1, 0) configuration. Pass the other named pairs "
            "explicitly for ablations."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--observation-budget-fraction", type=float, default=1.0)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--skip-combined-export",
        action="store_true",
        help="Do not rewrite summary.json; useful when benchmark workers run concurrently.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.observation_budget_fraction) or not 0 < args.observation_budget_fraction <= 1:
        parser.error("--observation-budget-fraction must be in (0, 1]")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for pair_name in args.pairs:
            for benchmark in args.benchmarks:
                run_one(
                    benchmark,
                    pair_name,
                    seed=args.seed,
                    outdir=args.outdir,
                    observation_budget_fraction=args.observation_budget_fraction,
                )
    if not args.skip_combined_export:
        export_combined(args.outdir, args.benchmarks, args.pairs)


if __name__ == "__main__":
    main()
