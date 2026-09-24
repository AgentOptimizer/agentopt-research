#!/usr/bin/env python3
"""Build the latest-at-or-below USD 10%/30% recommendation dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
GITTINS_ROOT = ROOT / "analysis/20seed_results/radial_gittins"
OUTPUT_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed"
DATASETS = (
    "hotpotqa", "mathqa", "restaurant_test", "stackoverflow", "bird_dev",
    "restaurant_valid", "bing_querylogs", "bird_mini_dev",
)
TRAJECTORY_METHODS = (
    "ege_sh", "ege_sr", "ape_k", "qnehvi",
    "random_configurations", "random_questions",
)
METHODS = ("radial_gittins",) + TRAJECTORY_METHODS
SEEDS = tuple(range(42, 62))
TARGETS = (0.10, 0.30)
MEMBERSHIP_EVENTS = {
    "initial_empty",
    "recommendation_initial",
    "recommendation_changed",
}


def _indices(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        return tuple(int(value) for value in text.split(";") if value.strip())
    return tuple(int(value) for value in text)


def _trajectory_interval(
    rows: list[dict[str, str]], target: float,
) -> dict[str, Any]:
    checkpoint_event = f"cost_checkpoint_{int(round(100 * target))}pct"
    positions = [
        index for index, row in enumerate(rows)
        if row.get("event") == checkpoint_event
    ]
    if len(positions) != 1:
        raise ValueError(
            f"expected exactly one {checkpoint_event}, found {len(positions)}"
        )
    checkpoint_position = positions[0]
    checkpoint = rows[checkpoint_position]
    selected = _indices(checkpoint["selected_arm_indices"])
    checkpoint_fraction = float(checkpoint["budget_fraction"])
    if checkpoint_fraction > target + 1e-12:
        raise ValueError(
            f"{checkpoint_event} exceeds its target: {checkpoint_fraction}"
        )

    starts = [
        (index, row)
        for index, row in enumerate(rows[: checkpoint_position + 1])
        if row.get("event") in MEMBERSHIP_EVENTS
    ]
    if not starts and not selected:
        # A coarse atomic update may make the threshold's latest feasible
        # recommendation the zero-cost empty state.  The checkpoint is the
        # retained copy of that initial snapshot.
        start_position, start = checkpoint_position, checkpoint
    elif starts:
        start_position, start = starts[-1]
    else:
        raise ValueError(f"{checkpoint_event} has no preceding membership event")
    if _indices(start["selected_arm_indices"]) != selected:
        raise ValueError(f"membership state disagrees with {checkpoint_event}")

    end: dict[str, str] | None = None
    for row in rows[checkpoint_position + 1 :]:
        if (
            row.get("event") in MEMBERSHIP_EVENTS
            and _indices(row["selected_arm_indices"]) != selected
        ):
            end = row
            break
    end_censored = end is None
    if end is None:
        terminals = [row for row in rows if row.get("event") == "terminal"]
        if len(terminals) != 1:
            raise ValueError(f"expected one terminal row, found {len(terminals)}")
        end = terminals[0]

    start_fraction = float(start["budget_fraction"])
    end_fraction = float(end["budget_fraction"])
    if not start_fraction <= checkpoint_fraction + 1e-12:
        raise ValueError("recommendation interval starts after checkpoint")
    if end_fraction + 1e-12 < checkpoint_fraction:
        raise ValueError("recommendation interval ends before checkpoint")

    return {
        "target_cost_fraction": target,
        "checkpoint_cost_fraction": checkpoint_fraction,
        "actual_cost_fraction": checkpoint_fraction,
        "checkpoint_cost_usd": float(checkpoint["cumulative_search_cost_usd"]),
        "cumulative_search_cost_usd": float(checkpoint["cumulative_search_cost_usd"]),
        "checkpoint_evaluations": int(float(checkpoint["cumulative_evaluations"])),
        "cumulative_evaluations": int(float(checkpoint["cumulative_evaluations"])),
        "recommendation_start_cost_fraction": start_fraction,
        "recommendation_start_cost_usd": float(start["cumulative_search_cost_usd"]),
        "recommendation_start_evaluations": int(float(start["cumulative_evaluations"])),
        "recommendation_end_cost_fraction": end_fraction,
        "recommendation_end_cost_usd": float(end["cumulative_search_cost_usd"]),
        "recommendation_end_evaluations": int(float(end["cumulative_evaluations"])),
        "recommendation_end_censored_at_terminal": end_censored,
        "selected_arm_indices": ";".join(str(index) for index in selected),
        "selected_models": checkpoint.get("selected_models", ""),
        "checkpoint_source_event": checkpoint_event,
        "source_event": checkpoint_event,
        "interval_start_event": start.get("event", ""),
        "interval_end_event": end.get("event", ""),
    }


def _load_trajectory_cell(
    method: str, dataset: str, seed: int, target: float,
) -> dict[str, Any]:
    directory = RUN_ROOT / dataset / method / f"seed-{seed}"
    summary_path = directory / "summary.json"
    trajectory_path = directory / "cost_trajectory.csv"
    if not summary_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(directory)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    version = int(
        summary.get("params", {}).get("recommendation_checkpoint_schema_version", 0)
    )
    if version < 3:
        raise ValueError(
            f"schema version {version} does not use latest-at-or-below checkpoints"
        )
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    interval = _trajectory_interval(rows, target)
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        **interval,
        "source_path": str(trajectory_path.relative_to(ROOT)),
        "source_commit": summary.get("source_commit", ""),
        "source_sha256": summary.get("source_sha256", ""),
    }


def _load_gittins_cell(
    dataset: str, seed: int, target: float,
) -> dict[str, Any]:
    result_path = GITTINS_ROOT / dataset / f"seed-{seed}" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    points = list(result["run"]["points"])
    terminal_fraction = float(result["run"]["cost_fraction"])
    if terminal_fraction + 1e-12 < target:
        raise ValueError(
            f"Gittins run terminates below target {target}: {terminal_fraction}"
        )
    eligible = [point for point in points if float(point["cost_fraction"]) <= target + 1e-12]
    if not eligible:
        raise ValueError("no Gittins recommendation at target")
    start = eligible[-1]
    selected = _indices(start["selected_arm_indices"])
    end = next(
        (
            point for point in points[len(eligible) :]
            if _indices(point["selected_arm_indices"]) != selected
        ),
        None,
    )
    end_censored = end is None
    if end is None:
        end = points[-1]
    bruteforce_cost = float(result["run"]["bruteforce_search_cost_usd"])
    model_names = list(result["run"]["model_names"])
    return {
        "method": "radial_gittins",
        "dataset": dataset,
        "seed": seed,
        "target_cost_fraction": target,
        "checkpoint_cost_fraction": target,
        "actual_cost_fraction": target,
        "checkpoint_cost_usd": target * bruteforce_cost,
        "cumulative_search_cost_usd": target * bruteforce_cost,
        "checkpoint_evaluations": "",
        "cumulative_evaluations": "",
        "recommendation_start_cost_fraction": float(start["cost_fraction"]),
        "recommendation_start_cost_usd": float(start["cost_usd"]),
        "recommendation_start_evaluations": int(start["evaluations"]),
        "recommendation_end_cost_fraction": float(end["cost_fraction"]),
        "recommendation_end_cost_usd": float(end["cost_usd"]),
        "recommendation_end_evaluations": int(end["evaluations"]),
        "recommendation_end_censored_at_terminal": end_censored,
        "selected_arm_indices": ";".join(str(index) for index in selected),
        "selected_models": ";".join(model_names[index] for index in selected),
        "checkpoint_source_event": "membership_interval_at_target",
        "source_event": "membership_interval_at_target",
        "interval_start_event": start.get("event", ""),
        "interval_end_event": end.get("event", ""),
        "source_path": str(result_path.relative_to(ROOT)),
        "source_commit": result.get("config", {}).get("source_commit", ""),
        "source_sha256": result.get("config", {}).get("engine_source_sha256", ""),
    }


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for method in METHODS:
        for dataset in DATASETS:
            for seed in SEEDS:
                for target in TARGETS:
                    try:
                        row = (
                            _load_gittins_cell(dataset, seed, target)
                            if method == "radial_gittins"
                            else _load_trajectory_cell(method, dataset, seed, target)
                        )
                        rows.append(row)
                    except Exception as exc:
                        errors.append(
                            {
                                "method": method,
                                "dataset": dataset,
                                "seed": seed,
                                "target_cost_fraction": target,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    return rows, errors


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write an empty checkpoint dataset")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, errors = build()
    _write_csv(rows, args.output_root / "recommendation_checkpoints.csv")
    for target in TARGETS:
        _write_csv(
            [row for row in rows if row["target_cost_fraction"] == target],
            args.output_root / f"recommendation_intervals_{int(100 * target)}pct.csv",
        )
    audit = {
        "dataset_version": "usd-checkpoints-latest-under-v1",
        "checkpoint_semantics": (
            "latest completed atomic policy update at or below each realized "
            "USD fraction; Radial-Gittins uses the recommendation membership "
            "interval active at the target"
        ),
        "expected_rows": len(METHODS) * len(DATASETS) * len(SEEDS) * len(TARGETS),
        "available_rows": len(rows),
        "missing_or_invalid_rows": len(errors),
        "targets": list(TARGETS),
        "methods": list(METHODS),
        "errors": errors,
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "errors"}, indent=2))
    if args.require_complete and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
