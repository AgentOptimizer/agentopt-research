#!/usr/bin/env python3
"""Plot HV-regret, GD, and IGD curves for all eight 20-seed datasets.

The script reads the lightweight ``analysis/20seed_results`` index.  It keeps
the evaluation metric used by the baseline runs (accuracy together with the
reciprocal-cost desirability) and re-evaluates the newer Gauss--Radau Gittins
recommendations in that same space before comparing methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "analysis/20seed_results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures/eight_dataset_curves"
FINAL_RUN_ROOT = ROOT / "analysis/final_run_baselines"

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
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "restaurant_test": "Restaurant Test",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD Dev",
    "restaurant_valid": "Restaurant Valid",
    "bing_querylogs": "Bing Query Logs",
    "bird_mini_dev": "BIRD Mini Dev",
}
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ege_sr",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METHOD_STYLES: dict[str, dict[str, str | float]] = {
    "radial_gittins": {
        "color": "tab:orange", "label": "Radial Gittins",
        "linewidth": 3.0, "linestyle": "-",
    },
    "ege_sh": {
        "color": "tab:green", "label": "EGE-SH",
        "linewidth": 2.4, "linestyle": "-",
    },
    "ege_sr": {
        "color": "tab:blue", "label": "EGE-SR",
        "linewidth": 2.4, "linestyle": "-",
    },
    "ape_k": {
        "color": "tab:purple", "label": "APE-k",
        "linewidth": 2.4, "linestyle": "-",
    },
    "qnehvi": {
        "color": "tab:pink", "label": "qNEHVI",
        "linewidth": 2.4, "linestyle": "-",
    },
    "random_configurations": {
        "color": "tab:brown", "label": "Random configurations",
        "linewidth": 2.25, "linestyle": "-",
    },
    "random_questions": {
        "color": "tab:cyan", "label": "Random questions",
        "linewidth": 2.25, "linestyle": "-",
    },
}
METRICS = {
    "hv_regret": ("Hypervolume Regret", "hv_regret"),
    "generational_distance": ("Generational Distance (GD)", "generational_distance"),
    "inverted_generational_distance": (
        "Inverted Generational Distance (IGD)",
        "inverted_generational_distance",
    ),
}
EXPECTED_SEEDS = tuple(range(42, 62))


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


@dataclass
class RunCurve:
    x: np.ndarray
    values: dict[str, np.ndarray]


@dataclass
class MeanCurve:
    x: np.ndarray
    mean: np.ndarray
    two_se: np.ndarray
    count: np.ndarray
    total_runs: int


def _nondominated(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        dominates[index] = False
        keep[index] = not bool(np.any(dominates))
    return points[keep]


def _hypervolume(points: np.ndarray) -> float:
    return _hypervolume_front(_nondominated(points))


def _hypervolume_front(front: np.ndarray) -> float:
    """Hypervolume of an already nondominated 2-D maximization front."""
    front = np.asarray(front, dtype=np.float64).reshape((-1, 2))
    front = front[np.all(front >= 0.0, axis=1)]
    if not len(front):
        return 0.0
    front = front[np.argsort(front[:, 0])]
    previous_x = 0.0
    area = 0.0
    for x_value, y_value in front:
        area += max(0.0, float(x_value) - previous_x) * max(0.0, float(y_value))
        previous_x = max(previous_x, float(x_value))
    return area


def _distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(source) or not len(target):
        return np.full(len(source), np.inf)
    return np.min(np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2), axis=1)


def _front_metrics(
    selected: np.ndarray, truth_front: np.ndarray, truth_hv: float,
) -> dict[str, float]:
    selected_front = _nondominated(selected) if len(selected) else selected.reshape((0, 2))
    selected_hv = _hypervolume_front(selected_front)
    gd = float(np.mean(_distances(selected_front, truth_front))) if len(selected_front) else math.inf
    igd = float(np.mean(_distances(truth_front, selected_front))) if len(selected_front) else math.inf
    return {
        "hv_regret": max(0.0, truth_hv - selected_hv),
        "generational_distance": gd,
        "inverted_generational_distance": igd,
    }


def _reciprocal_metric(raw: np.ndarray, reference: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).reshape((-1, 2))
    return np.column_stack((raw[:, 0], reference / (reference + raw[:, 1])))


def _deduplicate_curve(
    xs: Iterable[float], values: dict[str, Iterable[float]],
) -> RunCurve:
    latest: dict[float, dict[str, float]] = {}
    value_lists = {name: list(metric_values) for name, metric_values in values.items()}
    x_list = list(xs)
    for index, x_value in enumerate(x_list):
        latest[float(x_value)] = {
            name: float(metric_values[index]) for name, metric_values in value_lists.items()
        }
    ordered_x = np.asarray(sorted(latest), dtype=np.float64)
    ordered_values = {
        name: np.asarray([latest[x][name] for x in ordered_x], dtype=np.float64)
        for name in values
    }
    return RunCurve(ordered_x, ordered_values)


def _load_gittins(results_root: Path, dataset: str) -> tuple[dict[int, RunCurve], list[float]]:
    runs: dict[int, RunCurve] = {}
    stops: list[float] = []
    for seed_dir in sorted((results_root / "radial_gittins" / dataset).glob("seed-*")):
        result_path = seed_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run = payload["run"]
        raw_truth = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
        positive_costs = raw_truth[:, 1][raw_truth[:, 1] > 0.0]
        reference = float(np.median(positive_costs))
        truth = _reciprocal_metric(raw_truth, reference)
        # Reciprocal cost is strictly monotone in raw cost, so the Pareto arm
        # indices saved by the run remain valid in the baseline metric space.
        truth_front = truth[np.asarray(run["full_data_pareto_arm_indices"], dtype=int)]
        truth_hv = _hypervolume_front(truth_front)
        xs: list[float] = []
        metric_values = {metric: [] for metric in METRICS}
        for point in run["points"]:
            raw_selected = np.asarray(point["offline_raw_selected_vectors"], dtype=np.float64)
            raw_selected = raw_selected.reshape((-1, 2)) if raw_selected.size else np.empty((0, 2))
            selected = _reciprocal_metric(raw_selected, reference)
            scored = _front_metrics(selected, truth_front, truth_hv)
            xs.append(float(point["cost_fraction"]))
            for metric in METRICS:
                metric_values[metric].append(scored[metric])
        if xs:
            seed = int(run["seed"])
            runs[seed] = _deduplicate_curve(xs, metric_values)
            stop = payload.get("plotting", {}).get("terminal_cost_fraction")
            if stop is not None:
                stops.append(float(stop))
    return runs, stops


def _rows_to_curves(
    rows: Iterable[dict[str, str]], grid: np.ndarray,
) -> dict[int, RunCurve]:
    return _numeric_rows_to_curves(
        (
            (
                int(row["seed"]),
                float(row["budget_fraction"]),
                float(row.get("hv_regret", "nan")),
                float(row.get("generational_distance", "nan")),
                float(row.get("inverted_generational_distance", "nan")),
            )
            for row in rows
        ),
        grid,
    )


def _numeric_rows_to_curves(
    rows: Iterable[tuple[int, float, float, float, float]], grid: np.ndarray,
) -> dict[int, RunCurve]:
    """Downsample sorted trajectory rows while streaming them from disk."""
    sampled: dict[int, dict[str, np.ndarray]] = {}
    cursors: dict[int, int] = defaultdict(int)
    last_x: dict[int, float] = {}
    last_values: dict[int, dict[str, float]] = {}

    metric_order = tuple(METRICS)
    for seed, x_value, hv_regret, gd, igd in rows:
        if seed not in sampled:
            sampled[seed] = {
                metric: np.full(len(grid), np.nan, dtype=np.float64)
                for metric in METRICS
            }
        cursor = cursors[seed]
        if seed in last_x:
            while cursor < len(grid) and grid[cursor] < x_value - 1e-12:
                for metric in METRICS:
                    sampled[seed][metric][cursor] = last_values[seed][metric]
                cursor += 1
        else:
            while cursor < len(grid) and grid[cursor] < x_value - 1e-12:
                cursor += 1
        cursors[seed] = cursor
        last_x[seed] = x_value
        last_values[seed] = dict(zip(metric_order, (hv_regret, gd, igd)))

    curves: dict[int, RunCurve] = {}
    for seed, arrays in sampled.items():
        cursor = cursors[seed]
        while cursor < len(grid) and grid[cursor] <= last_x[seed] + 1e-12:
            for metric in METRICS:
                arrays[metric][cursor] = last_values[seed][metric]
            cursor += 1
        present = np.any(
            np.column_stack([np.isfinite(values) for values in arrays.values()]), axis=1
        )
        curves[seed] = RunCurve(
            grid[present],
            {metric: values[present] for metric, values in arrays.items()},
        )
    return curves


def _fast_numeric_rows(
    path: Path, dataset: str, method: str,
) -> Iterable[tuple[int, float, float, float, float]]:
    """Read only the five numeric columns needed from a large trajectory CSV."""
    accepted_benchmarks = {dataset, DATASET_LABELS[dataset].lower()}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split(",")
        positions = {name: index for index, name in enumerate(header)}
        required = (
            "seed",
            "budget_fraction",
            "hv_regret",
            "generational_distance",
            "inverted_generational_distance",
        )
        last_needed = max(positions[name] for name in required)
        for line in handle:
            fields = line.rstrip("\n").split(",", last_needed + 1)
            if "method" in positions and fields[positions["method"]].lower() != method:
                continue
            if (
                "benchmark" in positions
                and fields[positions["benchmark"]].lower() not in accepted_benchmarks
            ):
                continue
            yield (
                int(fields[positions["seed"]]),
                float(fields[positions["budget_fraction"]]),
                float(fields[positions["hv_regret"]]),
                float(fields[positions["generational_distance"]]),
                float(fields[positions["inverted_generational_distance"]]),
            )


def _load_csv_method(
    results_root: Path, method: str, dataset: str, grid: np.ndarray,
) -> dict[int, RunCurve]:
    method_dir = results_root / method / dataset
    seed_paths_by_name = {
        path.parent.name: path
        for path in method_dir.glob("seed-*/cost_trajectory.csv")
    }
    # Newly completed array tasks land in the canonical run directory before
    # the lightweight 20seed symlink index is refreshed. Prefer those files.
    seed_paths_by_name.update(
        {
            path.parent.name: path
            for path in (FINAL_RUN_ROOT / dataset / method).glob(
                "seed-*/cost_trajectory.csv"
            )
        }
    )
    seed_paths = [seed_paths_by_name[name] for name in sorted(seed_paths_by_name)]
    if seed_paths:
        curves: dict[int, RunCurve] = {}
        for path in seed_paths:
            curves.update(
                _numeric_rows_to_curves(
                    _fast_numeric_rows(path, dataset, method), grid,
                )
            )
        return curves

    aggregate = method_dir / "aggregate_cost_trajectory.csv"
    if not aggregate.exists():
        return {}
    return _numeric_rows_to_curves(
        _fast_numeric_rows(aggregate, dataset, method), grid,
    )


def _load_random_csv(
    results_root: Path, method: str, dataset: str, grid: np.ndarray,
) -> dict[int, RunCurve]:
    method_dir = results_root / method / dataset
    seed_paths = sorted(method_dir.glob("seed-*/cost_trajectory.csv"))
    if seed_paths:
        return _load_csv_method(results_root, method, dataset, grid)

    aggregate = method_dir / "multi_seed_results.csv"
    if not aggregate.exists():
        return {}
    selected_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    with aggregate.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] == method:
                selected_rows[int(row["seed"])].append(row)
    curves: dict[int, RunCurve] = {}
    for seed, rows in selected_rows.items():
        full_cost = max(float(row["total_search_cost_usd"]) for row in rows)
        xs = [float(row["total_search_cost_usd"]) / full_cost for row in rows]
        values = {
            "hv_regret": [float(row["hypervolume_regret"]) for row in rows],
            "generational_distance": [math.nan] * len(rows),
            "inverted_generational_distance": [math.nan] * len(rows),
        }
        curves[seed] = _deduplicate_curve(xs, values)
    return curves


def _aggregate(runs: dict[int, RunCurve], metric: str, grid: np.ndarray) -> MeanCurve | None:
    if not runs:
        return None
    aligned = np.full((len(runs), len(grid)), np.nan, dtype=np.float64)
    for row_index, curve in enumerate(runs.values()):
        metric_values = curve.values[metric]
        positions = np.searchsorted(curve.x, grid, side="right") - 1
        available = (positions >= 0) & (grid <= curve.x[-1] + 1e-12)
        chosen = positions[available]
        selected_values = metric_values[chosen]
        finite = np.isfinite(selected_values)
        target_columns = np.flatnonzero(available)[finite]
        aligned[row_index, target_columns] = selected_values[finite]
    count = np.sum(np.isfinite(aligned), axis=0)
    valid = count > 0
    if not np.any(valid):
        return None
    mean = np.full(len(grid), np.nan)
    two_se = np.full(len(grid), np.nan)
    mean[valid] = np.nanmean(aligned[:, valid], axis=0)
    for column in np.flatnonzero(valid):
        if count[column] > 1:
            two_se[column] = 2.0 * np.nanstd(aligned[:, column], ddof=1) / math.sqrt(count[column])
        else:
            two_se[column] = 0.0
    return MeanCurve(grid, mean, two_se, count, len(runs))


def _load_preaggregated_random_distances(path: Path) -> dict[tuple[str, str, str], MeanCurve]:
    output: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    if not path.exists():
        return {}
    labels_to_dataset = {label.lower(): dataset for dataset, label in DATASET_LABELS.items()}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split(",")
        positions = {name: index for index, name in enumerate(header)}
        for line in handle:
            if ",random_" not in line:
                continue
            fields = line.rstrip("\n").split(",")
            method = fields[positions["method"]]
            metric = fields[positions["metric"]]
            dataset = labels_to_dataset.get(fields[positions["benchmark"]].lower())
            if (
                dataset not in {"hotpotqa", "mathqa"}
                or method not in {"random_configurations", "random_questions"}
                or metric not in {"generational_distance", "inverted_generational_distance"}
            ):
                continue
            output[(dataset, method, metric)].append(
                (
                    float(fields[positions["cost_fraction"]]),
                    float(fields[positions["mean"]]),
                    float(fields[positions["two_se"]]),
                    int(fields[positions["n_runs"]]),
                )
            )
    resolved: dict[tuple[str, str, str], MeanCurve] = {}
    for key, rows in output.items():
        rows.sort()
        resolved[key] = MeanCurve(
            x=np.asarray([row[0] for row in rows]),
            mean=np.asarray([row[1] for row in rows]),
            two_se=np.asarray([row[2] for row in rows]),
            count=np.asarray([row[3] for row in rows]),
            total_runs=max(row[3] for row in rows),
        )
    return resolved


def _load_all_curves(results_root: Path, grid: np.ndarray):
    runs: dict[tuple[str, str], dict[int, RunCurve]] = {}
    stops: dict[str, list[float]] = {}
    for dataset in DATASETS:
        print(f"loading {dataset} ...", flush=True)
        gittins_runs, dataset_stops = _load_gittins(results_root, dataset)
        runs[(dataset, "radial_gittins")] = gittins_runs
        stops[dataset] = dataset_stops
        for method in ("ege_sh", "ege_sr", "ape_k", "qnehvi"):
            runs[(dataset, method)] = _load_csv_method(results_root, method, dataset, grid)
        for method in ("random_configurations", "random_questions"):
            runs[(dataset, method)] = _load_random_csv(results_root, method, dataset, grid)
    return runs, stops


def _plot_metric(
    metric: str,
    curves: dict[tuple[str, str], MeanCurve | None],
    stops: dict[str, list[float]],
    output_stem: Path,
    xmax: float,
) -> None:
    ylabel = METRICS[metric][0]
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    for axis, dataset in zip(axes.flat, DATASETS):
        for method in METHODS:
            curve = curves.get((dataset, method))
            if curve is None:
                continue
            shown = (
                np.isfinite(curve.mean)
                & np.isfinite(curve.two_se)
                & (curve.x <= xmax + 1e-12)
            )
            if not np.any(shown):
                continue
            style = METHOD_STYLES[method]
            axis.plot(
                curve.x[shown],
                curve.mean[shown],
                color=str(style["color"]),
                linewidth=float(style["linewidth"]),
                linestyle=str(style["linestyle"]),
                label=str(style["label"]),
                zorder=4 if method == "radial_gittins" else 3,
            )
            axis.fill_between(
                curve.x[shown],
                np.maximum(0.0, curve.mean[shown] - curve.two_se[shown]),
                curve.mean[shown] + curve.two_se[shown],
                color=str(style["color"]),
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=23, pad=10)
        axis.set_xlim(0.0, xmax)
        axis.set_ylim(bottom=0.0)
        axis.xaxis.set_major_formatter(percent)
        axis.set_xticks(np.linspace(0.0, xmax, 3))
        x_tick_labels = axis.get_xticklabels()
        x_tick_labels[0].set_ha("left")
        x_tick_labels[-1].set_ha("right")
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.tick_params(axis="both", labelsize=23, width=1.1, length=6)
        axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
        axis.set_axisbelow(True)
    # Place shared axis labels relative to the subplot block rather than the
    # complete canvas, whose bottom also contains a two-row legend.
    subplot_bottom = 0.265
    subplot_top = 0.945
    figure.text(
        0.5, 0.168, "Percentage of Exhaustive Evaluation Cost",
        ha="center", va="center", fontsize=28,
    )
    figure.text(
        0.023, (subplot_bottom + subplot_top) / 2.0, ylabel,
        ha="center", va="center", rotation=90, fontsize=28,
    )
    handles = [
        Line2D(
            [], [], color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            linestyle=str(style["linestyle"]), label=str(style["label"]),
        )
        for style in METHOD_STYLES.values()
    ]
    handles.append(
        Patch(facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE")
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.001),
        ncol=4,
        frameon=False,
        fontsize=22,
        handlelength=3.15,
        columnspacing=1.55,
        handletextpad=0.75,
        labelspacing=0.70,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, top=subplot_top, bottom=subplot_bottom,
        wspace=0.34, hspace=0.34,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def _write_summary(
    path: Path,
    curves: dict[tuple[str, str, str], MeanCurve | None],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "method", "metric", "cost_fraction", "mean", "two_se", "n_runs"),
        )
        writer.writeheader()
        for (dataset, method, metric), curve in curves.items():
            if curve is None:
                continue
            for x_value, mean, two_se, count in zip(curve.x, curve.mean, curve.two_se, curve.count):
                if not (np.isfinite(mean) and np.isfinite(two_se) and count > 0):
                    continue
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "cost_fraction": x_value,
                        "mean": mean,
                        "two_se": two_se,
                        "n_runs": int(count),
                    }
                )


def _read_summary(path: Path) -> dict[tuple[str, str, str], MeanCurve]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"], row["metric"])
            grouped[key].append(
                (
                    float(row["cost_fraction"]),
                    float(row["mean"]),
                    float(row["two_se"]),
                    int(row["n_runs"]),
                )
            )
    output: dict[tuple[str, str, str], MeanCurve] = {}
    for key, rows in grouped.items():
        rows.sort()
        output[key] = MeanCurve(
            x=np.asarray([row[0] for row in rows]),
            mean=np.asarray([row[1] for row in rows]),
            two_se=np.asarray([row[2] for row in rows]),
            count=np.asarray([row[3] for row in rows], dtype=int),
            total_runs=max(row[3] for row in rows),
        )
    return output


def _write_styles(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(METHOD_STYLES, indent=2) + "\n", encoding="utf-8")


def _load_styles(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = set(METHODS) - set(payload)
    if missing:
        raise ValueError(f"style file is missing methods: {sorted(missing)}")
    for method in METHODS:
        style = payload[method]
        for field in ("color", "label", "linewidth", "linestyle"):
            if field not in style:
                raise ValueError(f"style for {method!r} is missing {field!r}")
    METHOD_STYLES.clear()
    METHOD_STYLES.update({method: payload[method] for method in METHODS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="Redraw from curve_summary.csv without rescanning raw trajectories.",
    )
    parser.add_argument(
        "--style-file",
        type=Path,
        help="Editable JSON style file; defaults to OUTPUT_DIR/method_styles.json.",
    )
    args = parser.parse_args()
    if not 0.0 < args.xmax <= 1.0:
        parser.error("--xmax must lie in (0, 1]")
    if args.grid_points < 2:
        parser.error("--grid-points must be at least 2")

    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    style_path = (
        args.style_file.resolve()
        if args.style_file is not None
        else output_dir / "method_styles.json"
    )
    if style_path.exists():
        _load_styles(style_path)
    else:
        _write_styles(style_path)

    summary_path = output_dir / "curve_summary.csv"
    metadata_path = output_dir / "plot_metadata.json"
    if args.reuse_summary:
        if not summary_path.exists() or not metadata_path.exists():
            parser.error("--reuse-summary requires curve_summary.csv and plot_metadata.json")
        aggregated = _read_summary(summary_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stops = {
            dataset: (
                [float(metadata["gittins_mean_stop_cost_fraction"][dataset])]
                if metadata["gittins_mean_stop_cost_fraction"].get(dataset) is not None
                else []
            )
            for dataset in DATASETS
        }
        print(f"reusing plot-ready curves from {summary_path}", flush=True)
    else:
        grid = np.linspace(0.0, 1.0, args.grid_points)
        raw_runs, stops = _load_all_curves(results_root, grid)
        fallback = _load_preaggregated_random_distances(
            ROOT / "analysis/continuous_seeds_42_61/data/all_method_gd_igd_summary.csv"
        )
        aggregated: dict[tuple[str, str, str], MeanCurve | None] = {}
        for dataset in DATASETS:
            for method in METHODS:
                for metric in METRICS:
                    curve = _aggregate(raw_runs[(dataset, method)], metric, grid)
                    if curve is None:
                        curve = fallback.get((dataset, method, metric))
                    aggregated[(dataset, method, metric)] = curve

        _write_summary(summary_path, aggregated)
        metadata = {
            "results_root": str(results_root),
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "available_seed_counts": {
                dataset: {
                    method: len(raw_runs[(dataset, method)]) for method in METHODS
                }
                for dataset in DATASETS
            },
            "gittins_mean_stop_cost_fraction": {
                dataset: float(np.mean(values)) if values else None
                for dataset, values in stops.items()
            },
            "metric_space": "reciprocal-cost baseline evaluation space",
            "uncertainty": "mean +/- 2 standard errors",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    for metric in METRICS:
        metric_curves = {
            (dataset, method): aggregated.get((dataset, method, metric))
            for dataset in DATASETS
            for method in METHODS
        }
        _plot_metric(
            metric,
            metric_curves,
            stops,
            output_dir / f"eight_datasets_{metric}",
            args.xmax,
        )
    print(f"wrote PNG/PDF/SVG figures under {output_dir}")


if __name__ == "__main__":
    main()
