#!/usr/bin/env python3
"""Redraw the paper's two-objective curves with the three-objective alignment rule.

The exporter keeps the corrected complete-returned-set GD/IGD definition and
uses the three-objective plot convention:

* evaluate a fixed 0.5%-spaced budget grid from 0.5% through 30%;
* leave a seed missing before its first published recommendation;
* use the latest recommendation at or below each budget target;
* carry a naturally stopped run's final recommendation forward; and
* average only the seeds available at each target.

It writes reproducible source assets to a separate analysis directory; reviewed
assets are copied into the paper repository as a distinct versioned step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective import plot_g2_main_figures as main_plot  # noqa: E402
from experiments.combined_objective import plot_gittins_ablation_v2_20seed as ablation_plot  # noqa: E402
from experiments.combined_objective.gittins_ablation_v2 import (  # noqa: E402
    CONFIGURATIONS,
    FAMILIES,
)


CORRECTED_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    / "gittins_g2_main_figures"
    / "corrected_gd_igd_complete_returned_set_20260929"
)
DEFAULT_OUTPUT = CORRECTED_ROOT / "two_objective_3d_style_aggregation"
GRID = np.arange(1, 61, dtype=np.float64) / 200.0


class AblationCacheUnpickler(pickle.Unpickler):
    """Map the cache's script-local MeanCurve class to the importable class."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "MeanCurve":
            return ablation_plot.MeanCurve
        return super().find_class(module, name)


def summarize_available(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the three-objective mean/2SE treatment of missing early seeds."""
    counts = np.sum(np.isfinite(aligned), axis=0)
    means = np.full(len(GRID), np.nan, dtype=np.float64)
    two_se = np.full(len(GRID), np.nan, dtype=np.float64)
    for column, count in enumerate(counts):
        values = aligned[:, column]
        values = values[np.isfinite(values)]
        if count:
            means[column] = float(np.mean(values))
            two_se[column] = (
                2.0 * float(np.std(values, ddof=1)) / math.sqrt(int(count))
                if count > 1
                else 0.0
            )
    return means, two_se, counts


def align_trajectory(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    carry_terminal: bool,
) -> np.ndarray:
    """Take the last value at/below each grid point without start backfilling."""
    positions = np.searchsorted(xs, GRID, side="right") - 1
    available = positions >= 0
    if not carry_terminal:
        available &= GRID <= xs[-1] + 1e-12
    aligned = np.full(len(GRID), np.nan, dtype=np.float64)
    aligned[available] = ys[np.clip(positions[available], 0, len(xs) - 1)]
    return aligned


def aggregate_trajectories(
    trajectories: list[tuple[np.ndarray, dict[str, np.ndarray]]],
    *,
    carry_terminal: bool,
) -> dict[str, main_plot.MeanCurve]:
    output: dict[str, main_plot.MeanCurve] = {}
    for metric in main_plot.METRICS:
        rows = []
        for xs, values in trajectories:
            finite = np.isfinite(values[metric])
            if not np.any(finite):
                raise ValueError(f"trajectory has no finite {metric} value")
            rows.append(
                align_trajectory(
                    np.asarray(xs[finite], dtype=np.float64),
                    np.asarray(values[metric][finite], dtype=np.float64),
                    carry_terminal=carry_terminal,
                )
            )
        means, two_se, counts = summarize_available(np.asarray(rows))
        output[metric] = main_plot.MeanCurve(
            GRID.copy(), means, two_se, counts
        )
    return output


def build_main_curves(
    cache: dict[str, Any],
) -> dict[tuple[str, str, str], main_plot.MeanCurve]:
    curves: dict[tuple[str, str, str], main_plot.MeanCurve] = {}
    for dataset in main_plot.DATASETS:
        for method in main_plot.METHODS:
            if method == "radial_gittins":
                trajectories = main_plot.g2_metric_trajectories(
                    cache["runs"][dataset]
                )
                # Every early two-objective Gittins termination in these data is
                # direction_eta_numerical_floor, a natural stop in the 3-D rule.
                carry_terminal = True
            else:
                trajectories = main_plot.saved_baseline_trajectories(dataset, method)
                # Saved baseline trajectories reach exhaustive completion.
                carry_terminal = False
            method_curves = aggregate_trajectories(
                trajectories, carry_terminal=carry_terminal
            )
            for metric, curve in method_curves.items():
                curves[(dataset, method, metric)] = curve
    return curves


def ablation_trajectories(
    runs: list[dict[str, Any]],
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    trajectories = []
    for run in runs:
        points = [
            point for point in run["points"]
            if not point.get("synthetic_terminal", False)
        ]
        trajectories.append(
            (
                np.asarray(
                    [float(point["cost_fraction"]) for point in points],
                    dtype=np.float64,
                ),
                {
                    metric: np.asarray(
                        [float(point[metric]) for point in points],
                        dtype=np.float64,
                    )
                    for metric in ablation_plot.METRICS
                },
            )
        )
    return trajectories


def build_ablation_curves(
    cache: dict[str, Any],
) -> dict[tuple[str, str, str], ablation_plot.MeanCurve]:
    curves: dict[tuple[str, str, str], ablation_plot.MeanCurve] = {}
    for configuration in CONFIGURATIONS:
        for dataset in ablation_plot.BENCHMARKS:
            runs = cache["runs"][(configuration, dataset)]
            unexpected = {
                str(run["stop_reason"])
                for run in runs
                if str(run["stop_reason"])
                not in {"direction_eta_numerical_floor", "all_arms_completed"}
            }
            if unexpected:
                raise ValueError(
                    f"{configuration}/{dataset}: non-natural stops {unexpected}"
                )
            aggregated = aggregate_trajectories(
                ablation_trajectories(runs), carry_terminal=True
            )
            for metric, curve in aggregated.items():
                curves[(configuration, dataset, metric)] = ablation_plot.MeanCurve(
                    curve.x, curve.mean, curve.two_se, curve.count
                )
    return curves


def write_curve_csv(
    path: Path,
    curves: dict[tuple[str, str, str], Any],
    first_keys: tuple[str, ...],
) -> None:
    rows = []
    for key in sorted(curves):
        curve = curves[key]
        for x, mean, two_se, count in zip(
            curve.x, curve.mean, curve.two_se, curve.count
        ):
            row = dict(zip(first_keys, key))
            row.update(
                cost_fraction=float(x),
                mean="" if not np.isfinite(mean) else float(mean),
                two_se="" if not np.isfinite(two_se) else float(two_se),
                n_runs=int(count),
            )
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_count_summary(
    curves: dict[tuple[str, str, str], Any],
    expected_counts: dict[tuple[str, ...], int],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(curves):
        if key[-1] != "hv_regret":
            continue
        curve = curves[key]
        positive = np.flatnonzero(curve.count > 0)
        expected = expected_counts[key[:-1]]
        full = np.flatnonzero(curve.count == expected)
        rows.append(
            {
                "series": "/".join(key[:-1]),
                "expected_seed_count": expected,
                "first_visible_cost_fraction": (
                    float(curve.x[positive[0]]) if len(positive) else ""
                ),
                "seed_count_at_first_visible": (
                    int(curve.count[positive[0]]) if len(positive) else 0
                ),
                "first_full_seed_cost_fraction": (
                    float(curve.x[full[0]]) if len(full) else ""
                ),
                "seed_count_at_30pct": int(curve.count[-1]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CORRECTED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_cache = main_plot.read_cache(source / "main_2d/plot_cache.pkl")
    with (source / "ablation_2d/ablation_plot_cache.pkl").open("rb") as handle:
        ablation_cache = AblationCacheUnpickler(handle).load()
    if ablation_cache.get("version") != ablation_plot.CACHE_VERSION:
        raise ValueError("unsupported ablation cache version")

    main_plot.configure_matplotlib()
    styles = {name: style.copy() for name, style in main_cache["styles"].items()}
    styles["radial_gittins"]["label"] = main_plot.SHORT_METHOD_NAME
    for method in main_plot.METHODS:
        styles[method]["linewidth"] = main_plot.CURVE_LINEWIDTH
    styles["radial_gittins"]["linewidth"] = main_plot.HIGHLIGHT_LINEWIDTH

    main_curves = build_main_curves(main_cache)
    main_root = output / "main_2d"
    write_curve_csv(
        main_root / "curve_summary_3d_style.csv",
        main_curves,
        ("dataset", "method", "metric"),
    )
    for metric in main_plot.METRICS:
        main_plot.plot_metric(
            metric,
            main_curves,
            styles,
            main_root / "curves" / f"eight_datasets_{metric}_3d_style",
            x_max=0.30,
        )

    ablation_plot.configure_matplotlib()
    ablation_curves = build_ablation_curves(ablation_cache)
    ablation_root = output / "ablation_2d"
    write_curve_csv(
        ablation_root / "ablation_curve_summary_3d_style.csv",
        ablation_curves,
        ("configuration", "dataset", "metric"),
    )
    for family, configurations in FAMILIES.items():
        for metric in ablation_plot.METRICS:
            ablation_plot.plot_family_metric(
                ablation_root,
                family,
                configurations,
                metric,
                ablation_curves,
            )

    main_expected = {}
    for dataset in main_plot.DATASETS:
        for method in main_plot.METHODS:
            main_expected[(dataset, method)] = len(
                main_plot.QNEHVI_SEEDS.get(dataset, main_plot.SEEDS)
                if method == "qnehvi"
                else main_plot.SEEDS
            )
    ablation_expected = {
        (configuration, dataset): len(ablation_plot.SEEDS)
        for configuration in CONFIGURATIONS
        for dataset in ablation_plot.BENCHMARKS
    }
    main_counts = seed_count_summary(main_curves, main_expected)
    ablation_counts = seed_count_summary(ablation_curves, ablation_expected)
    write_rows(main_root / "seed_count_summary.csv", main_counts)
    write_rows(ablation_root / "seed_count_summary.csv", ablation_counts)

    manifest = {
        "purpose": "paper two-objective redraw using three-objective alignment",
        "source": str(source),
        "displayed_cost_fraction_grid": [float(value) for value in GRID],
        "aggregation": (
            "fixed 0.5% grid; no mean-start alignment or first-value backfill; "
            "latest recommendation at/below each target; natural terminal output "
            "carried forward; mean +/- 2 SE over seeds available at each target"
        ),
        "distance_metric": (
            "corrected GD/IGD over every returned configuration, including "
            "true-space dominated returned configurations"
        ),
        "paper_figures_written_directly": False,
        "main_curve_files": 3,
        "ablation_curve_files": 12,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Two-objective curves with three-objective aggregation\n\n"
        "Paper-curve redraw generated from the corrected saved trajectories. "
        "No optimizer or API experiment was rerun. The exporter writes source "
        "assets here rather than mutating the paper repository directly. Curves "
        "use a fixed 0.5%--30% budget grid, do not align at "
        "the mean seed start, do not backfill late starts, and average only seeds "
        "with an available recommendation at each target. Natural terminal "
        "recommendations remain valid at later targets. GD/IGD use the complete "
        "returned configuration set.\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
