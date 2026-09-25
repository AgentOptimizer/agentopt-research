#!/usr/bin/env python3
"""Build the G2 main-text curves and paper frontier figures.

The default mode replaces only the ``radial_gittins`` series.  Cache-only
modes can rebuild the 0--30% curves or all six methods' frontier figures from
saved trajectories without rerunning an optimizer or reading result JSONs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIGURE_ROOT = ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
SOURCE_FIGURES = FIGURE_ROOT / "hv_frontier_source"
SOURCE_TIMING = FIGURE_ROOT / "time"
SAVED_BASELINE_ROOT = (
    ROOT / "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
)
NEW_QA_BASELINE_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_runs/baselines"
)
SAVED_QNEHVI_ROOT = ROOT / "analysis/final_run_baselines"
G2_RUN_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_runs/combined"
    / "gittins_ablation_8bench_20seed/g2_exact_axes"
)
DEFAULT_OUTPUT = FIGURE_ROOT / "gittins_g2_main_figures"

DATASETS = (
    "hotpotqa",
    "restaurant_valid",
    "bird_mini_dev",
    "bing_querylogs",
    "mathqa",
    "restaurant_test",
    "bird_dev",
    "stackoverflow",
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
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METRICS = {
    "hv_regret": "Hypervolume Regret",
    "generational_distance": "Generational Distance (GD)",
    "inverted_generational_distance": "Inverted Generational Distance (IGD)",
}
SEEDS = tuple(range(42, 62))
QNEHVI_SEEDS = {
    "bird_mini_dev": tuple(seed for seed in SEEDS if seed != 43),
    "bing_querylogs": tuple(range(42, 60)),
}
GRID = np.linspace(0.0, 1.0, 201)
CHECKPOINTS = (0.10, 0.30)
CACHE_VERSION = 2
FULL_METHOD_NAME = "Cost-Coupled Gittins"
SHORT_METHOD_NAME = "CC-Gittins"


@dataclass
class MeanCurve:
    x: np.ndarray
    mean: np.ndarray
    two_se: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class FrontierSummary:
    counts: Counter[int]
    available: int
    mean_start_usd: float
    mean_end_usd: float
    mean_start_fraction: float
    mean_end_fraction: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nondominated_indices(points: np.ndarray) -> np.ndarray:
    """Indices not dominated when both coordinates are maximized."""
    array = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)


def hypervolume_2d(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=np.float64)
    eligible = array[np.all(array >= 0.0, axis=1)]
    if not len(eligible):
        return 0.0
    front = eligible[nondominated_indices(eligible)]
    front = front[np.argsort(front[:, 0])]
    suffix_y = np.maximum.accumulate(front[::-1, 1])[::-1]
    area = 0.0
    previous_x = 0.0
    for point, max_y in zip(front, suffix_y):
        x = max(float(point[0]), previous_x)
        area += (x - previous_x) * max(0.0, float(max_y))
        previous_x = x
    return float(area)


def front_distance(obtained: np.ndarray, reference: np.ndarray) -> float:
    obtained_front = obtained[nondominated_indices(obtained)]
    reference_front = reference[nondominated_indices(reference)]
    if not len(obtained_front):
        return math.inf if len(reference_front) else 0.0
    if not len(reference_front):
        return math.inf
    delta = obtained_front[:, None, :] - reference_front[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1).mean())


def evaluation_space(raw_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    positive_costs = raw_truth[:, 1][raw_truth[:, 1] > 0.0]
    if not len(positive_costs):
        raise ValueError("truth vectors contain no positive deployment costs")
    cost_reference = float(np.median(positive_costs))
    metric_truth = np.column_stack(
        (raw_truth[:, 0], cost_reference / (cost_reference + raw_truth[:, 1]))
    )
    truth_front = metric_truth[nondominated_indices(metric_truth)]
    return metric_truth, truth_front, cost_reference, hypervolume_2d(truth_front)


def score_selection(
    selected: tuple[int, ...], metric_truth: np.ndarray, truth_front: np.ndarray,
    truth_hv: float,
) -> dict[str, float]:
    obtained = metric_truth[np.asarray(selected, dtype=int)]
    return {
        "hv_regret": max(0.0, truth_hv - hypervolume_2d(obtained)),
        "generational_distance": front_distance(obtained, truth_front),
        "inverted_generational_distance": front_distance(truth_front, obtained),
    }


def result_path(dataset: str, seed: int) -> Path:
    return G2_RUN_ROOT / f"seed-{seed}" / "exact_axes" / dataset / "result.json"


def load_g2_runs() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, np.ndarray],
    dict[str, Any],
    list[dict[str, str]],
]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_truth: dict[str, np.ndarray] = {}
    spaces: dict[str, Any] = {}
    runtime_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        metric_cache: dict[tuple[int, ...], dict[str, float]] = {}
        metric_truth = truth_front = None
        truth_hv = math.nan
        for seed in SEEDS:
            path = result_path(dataset, seed)
            if not path.is_file():
                raise FileNotFoundError(f"missing G2 result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            expected = {
                "ablation_name": "g2_exact_axes",
                "pair_name": "exact_axes",
                "directions": [[0.0, 1.0], [1.0, 0.0]],
                "direction_scheduler": "round_robin",
                "eta_decay_schedule": "direction_stop",
            }
            for field, value in expected.items():
                if config.get(field) != value:
                    raise ValueError(f"{path}: unexpected {field}={config.get(field)!r}")
            run = payload["run"]
            if int(run["seed"]) != seed:
                raise ValueError(f"{path}: seed mismatch")
            raw = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
            full_search_cost = float(run["bruteforce_search_cost_usd"])
            if dataset not in raw_truth:
                raw_truth[dataset] = raw
                metric_truth, truth_front, cost_reference, truth_hv = evaluation_space(raw)
                spaces[dataset] = {
                    "cost_reference_usd": cost_reference,
                    "ground_truth_hypervolume": truth_hv,
                    "n_configurations": len(raw),
                    "bruteforce_search_cost_usd": full_search_cost,
                }
            elif raw.shape != raw_truth[dataset].shape or not np.allclose(
                raw, raw_truth[dataset], rtol=0.0, atol=1e-15
            ):
                raise ValueError(f"{path}: truth vectors changed across seeds")
            elif not math.isclose(
                full_search_cost,
                float(spaces[dataset]["bruteforce_search_cost_usd"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{path}: brute-force search cost changed across seeds")
            assert metric_truth is not None and truth_front is not None
            points = []
            for point in run["points"]:
                selected = tuple(sorted(map(int, point["selected_arm_indices"])))
                metrics = metric_cache.get(selected)
                if metrics is None:
                    metrics = score_selection(selected, metric_truth, truth_front, truth_hv)
                    metric_cache[selected] = metrics
                points.append(
                    {
                        "cost_fraction": float(point["cost_fraction"]),
                        "cost_usd": float(point["cost_usd"]),
                        "selected_arm_indices": selected,
                        **metrics,
                    }
                )
            points.sort(key=lambda item: item["cost_fraction"])
            if not points:
                raise ValueError(f"{path}: no recommendation points")
            runs[dataset].append({"seed": seed, "points": points})
            metadata_path = path.with_name("slurm_task_metadata.json")
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing G2 timing metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "method": "radial_gittins",
                    "seed": str(seed),
                    "total_wall_clock_seconds": str(
                        float(metadata["task_wall_time_seconds"])
                    ),
                    "source": str(metadata_path),
                    "slurm_job_id": str(
                        metadata.get("slurm", {}).get(
                            "SLURM_JOB_ID",
                            metadata.get("slurm", {}).get("job_id", ""),
                        )
                    ),
                }
            )
        if [item["seed"] for item in runs[dataset]] != list(SEEDS):
            raise AssertionError(f"{dataset}: incomplete or unordered seed grid")
    for dataset in ("hotpotqa", "mathqa"):
        if int(spaces[dataset]["n_configurations"]) != 100:
            raise ValueError(
                f"{dataset}: expected the 10x10 configuration grid (100 arms), "
                f"got {spaces[dataset]['n_configurations']}"
            )
    return dict(runs), raw_truth, spaces, runtime_rows


def aggregate_g2(runs: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], MeanCurve]:
    output = {}
    for dataset in DATASETS:
        aligned = {
            metric: np.full((len(SEEDS), len(GRID)), np.nan, dtype=np.float64)
            for metric in METRICS
        }
        for row_index, run in enumerate(runs[dataset]):
            xs = np.asarray([point["cost_fraction"] for point in run["points"]])
            positions = np.searchsorted(xs, GRID, side="right") - 1
            available = (positions >= 0) & (GRID <= xs[-1] + 1e-12)
            for metric in METRICS:
                ys = np.asarray([point[metric] for point in run["points"]])
                aligned[metric][row_index, available] = ys[positions[available]]
        for metric, values in aligned.items():
            count = np.sum(np.isfinite(values), axis=0)
            mean = np.full(len(GRID), np.nan)
            two_se = np.full(len(GRID), np.nan)
            for column in np.flatnonzero(count):
                finite = values[:, column][np.isfinite(values[:, column])]
                mean[column] = float(np.mean(finite))
                two_se[column] = (
                    2.0 * float(np.std(finite, ddof=1)) / math.sqrt(len(finite))
                    if len(finite) > 1 else 0.0
                )
            output[(dataset, metric)] = MeanCurve(GRID.copy(), mean, two_se, count)
    return output


def read_source_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    summary = SOURCE_FIGURES / "curve_summary.csv"
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    styles = json.loads((SOURCE_FIGURES / "method_styles.json").read_text(encoding="utf-8"))
    return rows, styles


def read_timing_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with (SOURCE_TIMING / "runtime_by_seed.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        by_seed = list(csv.DictReader(handle))
    with (SOURCE_TIMING / "runtime_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))
    return by_seed, summary


def combined_timing_rows(
    g2_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_by_seed, source_summary = read_timing_rows()
    baseline_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_by_seed:
        if row["method"] != "radial_gittins":
            baseline_by_key[(row["dataset"], row["method"])].append(row.copy())
    g2_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in g2_rows:
        g2_by_dataset[row["dataset"]].append(row.copy())
    source_summary_by_key = {
        (row["dataset"], row["method"]): row.copy()
        for row in source_summary
        if row["method"] != "radial_gittins"
    }

    by_seed: list[dict[str, str]] = []
    summary: list[dict[str, str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            if method == "radial_gittins":
                rows = sorted(g2_by_dataset[dataset], key=lambda row: int(row["seed"]))
                if [int(row["seed"]) for row in rows] != list(SEEDS):
                    raise ValueError(f"{dataset}: incomplete G2 runtime grid")
                by_seed.extend(rows)
                values = [float(row["total_wall_clock_seconds"]) for row in rows]
                summary.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "n": str(len(values)),
                        "mean_seconds": f"{statistics.fmean(values):.9f}",
                        "error_2se_seconds": (
                            f"{2.0 * statistics.stdev(values) / math.sqrt(len(values)):.9f}"
                        ),
                    }
                )
            else:
                by_seed.extend(baseline_by_key[(dataset, method)])
                summary.append(source_summary_by_key[(dataset, method)])
    source_baseline_by_seed = [
        row for row in source_by_seed if row["method"] != "radial_gittins"
    ]
    output_baseline_by_seed = [
        row for row in by_seed if row["method"] != "radial_gittins"
    ]
    if Counter(tuple(sorted(row.items())) for row in output_baseline_by_seed) != Counter(
        tuple(sorted(row.items())) for row in source_baseline_by_seed
    ):
        raise AssertionError("baseline per-seed timing rows changed")
    if [row for row in summary if row["method"] != "radial_gittins"] != [
        source_summary_by_key[(dataset, method)]
        for dataset in DATASETS
        for method in METHODS
        if method != "radial_gittins"
    ]:
        raise AssertionError("baseline timing summary rows changed")
    return by_seed, summary


def combined_rows(
    source_rows: list[dict[str, str]], g2_curves: dict[tuple[str, str], MeanCurve]
) -> list[dict[str, str]]:
    baselines = [row.copy() for row in source_rows if row["method"] != "radial_gittins"]
    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in baselines:
        by_key[(row["dataset"], row["method"], row["metric"])].append(row)
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for metric in METRICS:
                if method != "radial_gittins":
                    rows.extend(by_key[(dataset, method, metric)])
                    continue
                curve = g2_curves[(dataset, metric)]
                for x, mean, two_se, count in zip(
                    curve.x, curve.mean, curve.two_se, curve.count
                ):
                    if np.isfinite(mean) and np.isfinite(two_se) and count > 0:
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "metric": metric,
                                "cost_fraction": str(float(x)),
                                "mean": str(float(mean)),
                                "two_se": str(float(two_se)),
                                "n_runs": str(int(count)),
                            }
                        )
    if [row for row in rows if row["method"] != "radial_gittins"] != baselines:
        raise AssertionError("baseline row order or values changed")
    return rows


def curves_from_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], MeanCurve]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["metric"])].append(
            (float(row["cost_fraction"]), float(row["mean"]), float(row["two_se"]), int(row["n_runs"]))
        )
    return {
        key: MeanCurve(
            np.asarray([row[0] for row in values]),
            np.asarray([row[1] for row in values]),
            np.asarray([row[2] for row in values]),
            np.asarray([row[3] for row in values], dtype=int),
        )
        for key, values in grouped.items()
    }


def latest_point(points: list[dict[str, Any]], checkpoint: float) -> dict[str, Any] | None:
    eligible = [point for point in points if point["cost_fraction"] <= checkpoint + 1e-12]
    return eligible[-1] if eligible else None


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_cache(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"unsupported cache version in {path}")
    return payload


def write_curve_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "dataset", "method", "metric", "cost_fraction", "mean", "two_se", "n_runs"
        ))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(metric: str, curves: dict[tuple[str, str, str], MeanCurve],
                styles: dict[str, Any], output: Path, *, x_max: float = 1.0) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    for axis, dataset in zip(axes.flat, DATASETS):
        for method in METHODS:
            curve = curves[(dataset, method, metric)]
            shown = (
                np.isfinite(curve.mean)
                & np.isfinite(curve.two_se)
                & (curve.x <= x_max + 1e-12)
            )
            style = styles[method]
            is_random = method in {"random_configurations", "random_questions"}
            axis.plot(
                curve.x[shown], curve.mean[shown], color=style["color"],
                linewidth=float(style["linewidth"]), linestyle=style["linestyle"],
                label=style["label"], zorder=4 if method == "radial_gittins" else 3,
                drawstyle="steps-post" if is_random else "default",
            )
            axis.fill_between(
                curve.x[shown], np.maximum(0.0, curve.mean[shown] - curve.two_se[shown]),
                curve.mean[shown] + curve.two_se[shown], color=style["color"], alpha=0.10,
                linewidth=0, zorder=1, step="post" if is_random else None,
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=25, pad=10)
        margin = 0.015 * x_max
        if math.isclose(x_max, 0.3):
            axis.set_xlim(-margin, x_max + 0.006)
        else:
            axis.set_xlim(-margin, x_max + margin)
        y_top = axis.get_ylim()[1]
        axis.set_ylim(-0.02 * y_top, y_top)
        axis.set_xticks(
            (0.0, 0.1, 0.2, 0.3) if math.isclose(x_max, 0.3) else (0.0, 0.5, 1.0)
        )
        axis.xaxis.set_major_formatter(percent)
        if not math.isclose(x_max, 0.3):
            axis.get_xticklabels()[0].set_ha("left")
            axis.get_xticklabels()[-1].set_ha("right")
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.tick_params(axis="both", labelsize=25, width=1.1, length=6)
        axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
        axis.set_axisbelow(True)
    figure.text(0.5, 0.168, "Percentage of Exhaustive Evaluation Cost",
                ha="center", va="center", fontsize=30)
    figure.text(0.023, (0.265 + 0.945) / 2.0, METRICS[metric], ha="center",
                va="center", rotation=90, fontsize=30)
    handles = [Line2D([], [], color=styles[method]["color"],
                      linewidth=float(styles[method]["linewidth"]),
                      linestyle=styles[method]["linestyle"], label=styles[method]["label"])
               for method in METHODS]
    handles.append(Patch(facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.014),
                  ncol=4, frameon=False, fontsize=25, handlelength=3.15,
                  columnspacing=1.55, handletextpad=0.75, labelspacing=0.70)
    figure.subplots_adjust(
        left=0.085, right=0.985, top=0.945, bottom=0.265,
        wspace=0.34, hspace=0.44 if math.isclose(x_max, 0.3) else 0.34,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def raw_front_indices(truth: np.ndarray) -> np.ndarray:
    metric = np.column_stack((truth[:, 0], -truth[:, 1]))
    return nondominated_indices(metric)


def carry_terminal_recommendations(
    runs: dict[str, list[dict[str, Any]]],
    spaces: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Keep each stopped run's last published recommendation valid to 100%.

    The synthetic endpoint changes only the budget-threshold readout.  It does
    not represent another evaluation, and the cache payload is never mutated.
    """
    extended: dict[str, list[dict[str, Any]]] = {}
    for dataset, dataset_runs in runs.items():
        full_search_cost = float(spaces[dataset]["bruteforce_search_cost_usd"])
        extended[dataset] = []
        for run in dataset_runs:
            points = [point.copy() for point in run["points"]]
            if not points:
                raise ValueError(f"{dataset} seed {run['seed']}: empty trajectory")
            if float(points[-1]["cost_fraction"]) < 1.0 - 1e-12:
                terminal = points[-1].copy()
                terminal.update(
                    cost_fraction=1.0,
                    cost_usd=full_search_cost,
                    synthetic_terminal=True,
                )
                points.append(terminal)
            extended[dataset].append({**run, "points": points})
    return extended


def _mix(
    color: tuple[float, float, float],
    other: tuple[float, float, float],
    amount: float,
) -> tuple[float, float, float]:
    return tuple(
        (1.0 - amount) * value + amount * target
        for value, target in zip(color, other)
    )


def _frequency_colormap(
    color: str = "tab:orange",
    name: str = "recommendation_frequency",
) -> LinearSegmentedColormap:
    base = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        name,
        (
            _mix(base, (1, 1, 1), 0.82),
            _mix(base, (1, 1, 1), 0.45),
            base,
            _mix(base, (0, 0, 0), 0.38),
        ),
        N=256,
    )


def _usd(value: float) -> str:
    if value < 1.0:
        return f"${value:.2f}"
    if value < 100.0:
        return f"${value:.1f}"
    return f"${value:,.0f}"


def frontier_summary(
    dataset: str,
    checkpoint: float,
    runs: dict[str, list[dict[str, Any]]],
    spaces: dict[str, Any],
) -> tuple[FrontierSummary, list[dict[str, Any]]]:
    counts: Counter[int] = Counter()
    starts: list[float] = []
    ends: list[float] = []
    rows: list[dict[str, Any]] = []
    for run in runs[dataset]:
        points = run["points"]
        eligible = [
            point
            for point in points
            if point["cost_fraction"] <= checkpoint + 1e-12
        ]
        if not eligible:
            continue
        start = eligible[-1]
        selected = start["selected_arm_indices"]
        end = next(
            (
                point
                for point in points[len(eligible) :]
                if point["selected_arm_indices"] != selected
            ),
            points[-1],
        )
        counts.update(selected)
        starts.append(start["cost_usd"])
        ends.append(end["cost_usd"])
        rows.append(
            {
                "dataset": dataset,
                "seed": run["seed"],
                "target_cost_fraction": checkpoint,
                "snapshot_cost_fraction": start["cost_fraction"],
                "snapshot_cost_usd": start["cost_usd"],
                "membership_end_cost_fraction": end["cost_fraction"],
                "membership_end_cost_usd": end["cost_usd"],
                "selected_arm_indices": json.dumps(selected, separators=(",", ":")),
                **{metric: start[metric] for metric in METRICS},
            }
        )
    available = len(starts)
    if not available:
        return FrontierSummary(counts, 0, math.nan, math.nan, math.nan, math.nan), rows
    full_search_cost = float(spaces[dataset]["bruteforce_search_cost_usd"])
    mean_start = float(np.mean(starts))
    mean_end = float(np.mean(ends))
    return (
        FrontierSummary(
            counts,
            available,
            mean_start,
            mean_end,
            mean_start / full_search_cost,
            mean_end / full_search_cost,
        ),
        rows,
    )


def draw_frontier_panel(
    axis: plt.Axes,
    truth: np.ndarray,
    summary: FrontierSummary,
    title: str,
    cmap: mpl.colors.Colormap,
    *,
    title_fontsize: float = 27,
    subtitle_fontsize: float = 26,
    tick_labelsize: float = 23,
) -> None:
    frontier = raw_front_indices(truth)
    frontier = frontier[np.argsort(truth[frontier, 1])]
    axis.scatter(
        truth[:, 1], truth[:, 0], s=48, color="#c0c5cc", alpha=0.64,
        edgecolors="none", zorder=1,
    )
    axis.plot(
        truth[frontier, 1], truth[frontier, 0], color="#25282c",
        linewidth=1.7, zorder=2,
    )
    axis.scatter(
        truth[frontier, 1], truth[frontier, 0], s=72, facecolors="white",
        edgecolors="#25282c", linewidths=1.25, zorder=3,
    )
    shown = np.asarray(sorted(summary.counts), dtype=int)
    if len(shown):
        frequencies = np.asarray([summary.counts[int(index)] for index in shown])
        axis.scatter(
            truth[shown, 1], truth[shown, 0], c=frequencies, cmap=cmap,
            norm=mpl.colors.Normalize(vmin=1, vmax=20), s=175,
            edgecolors="#6f4a22", linewidths=1.1, zorder=4,
        )
    shown_title = (
        title if summary.available == 20 else f"{title}  [n={summary.available}/20]"
    )
    axis.text(
        0.5, 1.22, shown_title, transform=axis.transAxes, ha="center",
        va="bottom", fontsize=title_fontsize,
    )
    if summary.available:
        start_usd = _usd(summary.mean_start_usd).replace("$", r"\$")
        end_usd = _usd(summary.mean_end_usd).replace("$", r"\$")
        subtitle = (
            f"{start_usd}–{end_usd} "
            f"({summary.mean_start_fraction:.1%}–{summary.mean_end_fraction:.1%})"
        )
    else:
        subtitle = "pending"
    axis.text(
        0.5, 1.17, subtitle, transform=axis.transAxes, ha="center",
        va="top", fontsize=subtitle_fontsize,
    )
    if np.all(truth[:, 1] > 0.0):
        axis.set_xscale("log")
    axis.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
    axis.tick_params(axis="both", labelsize=tick_labelsize)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())


def finish_frontier_figure(
    figure: plt.Figure,
    *,
    checkpoint: float,
    cmap: mpl.colors.Colormap,
    method_label: str,
    stem: Path,
) -> None:
    figure.supxlabel(
        "Mean deployment cost (USD per query, log scale)", fontsize=30, y=0.115
    )
    figure.supylabel("Mean accuracy", fontsize=30, x=0.022)
    figure.suptitle(
        f"{method_label} recommendations at {checkpoint:.0%} "
        "of brute-force search cost",
        fontsize=34,
        y=0.915,
    )
    figure.legend(
        handles=(
            Line2D(
                [], [], linestyle="none", marker="o", markersize=13,
                markerfacecolor="#c0c5cc", markeredgecolor="none",
                label="All configurations",
            ),
            Line2D(
                [], [], color="#25282c", linewidth=2, marker="o",
                markersize=10, markerfacecolor="white", markeredgewidth=1.25,
                label="Full-data Pareto frontier",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor=cmap(0.6), markeredgecolor="#6f4a22",
                label="Recommendations (color = frequency)",
            ),
        ),
        loc="lower center",
        bbox_to_anchor=(0.48, 0.035),
        ncol=3,
        frameon=False,
        fontsize=27,
        columnspacing=2.2,
        handletextpad=0.65,
    )
    figure.subplots_adjust(
        left=0.080, right=0.865, bottom=0.205, top=0.775,
        wspace=0.26, hspace=0.58,
    )
    colorbar_axis = figure.add_axes([0.900, 0.245, 0.021, 0.52])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, 20), cmap=cmap),
        cax=colorbar_axis,
    )
    colorbar.set_label(
        "Recommendation frequency (out of 20 seeds)", fontsize=28, labelpad=19
    )
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=24)
    colorbar.outline.set_linewidth(0.9)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def plot_frontiers(
    output: Path,
    runs: dict[str, list[dict[str, Any]]],
    raw_truth: dict[str, np.ndarray],
    spaces: dict[str, Any],
) -> list[dict[str, Any]]:
    cmap = _frequency_colormap()
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        figure, axes = plt.subplots(2, 4, figsize=(24, 14.5))
        for axis, dataset in zip(axes.flat, DATASETS):
            summary, rows = frontier_summary(dataset, checkpoint, runs, spaces)
            checkpoint_rows.extend(rows)
            draw_frontier_panel(
                axis, raw_truth[dataset], summary, DATASET_LABELS[dataset], cmap
            )
        finish_frontier_figure(
            figure,
            checkpoint=checkpoint,
            cmap=cmap,
            method_label=FULL_METHOD_NAME,
            stem=(
                output
                / f"radial_gittins_g2_20seed_frequency_frontiers_"
                f"{round(checkpoint * 100)}pct"
            ),
        )
    return checkpoint_rows


def plot_stackoverflow_method_frontier(
    output: Path,
    *,
    checkpoint: float,
    runs_by_method: dict[str, dict[str, list[dict[str, Any]]]],
    payload: dict[str, Any],
    styles: dict[str, Any],
) -> None:
    """Draw the native 2x3 Stack Overflow comparison without a suptitle."""
    method_order = (
        "radial_gittins",
        "ege_sh",
        "ape_k",
        "qnehvi",
        "random_configurations",
        "random_questions",
    )
    cmap = _frequency_colormap(
        "tab:orange", "stackoverflow_recommendation_frequency"
    )
    figure, axes = plt.subplots(2, 3, figsize=(24, 14.5))
    for axis, method in zip(axes.flat, method_order):
        summary, _ = frontier_summary(
            "stackoverflow", checkpoint, runs_by_method[method], payload["spaces"]
        )
        method_label = (
            FULL_METHOD_NAME
            if method == "radial_gittins"
            else styles[method]["label"]
        )
        draw_frontier_panel(
            axis,
            payload["raw_truth"]["stackoverflow"],
            summary,
            method_label,
            cmap,
            title_fontsize=29,
            subtitle_fontsize=28,
            tick_labelsize=26,
        )
    figure.supxlabel(
        "Mean deployment cost (USD per query, log scale)", fontsize=32, y=0.115
    )
    figure.supylabel("Mean accuracy", fontsize=32, x=0.022)
    figure.legend(
        handles=(
            Line2D(
                [], [], linestyle="none", marker="o", markersize=13,
                markerfacecolor="#c0c5cc", markeredgecolor="none",
                label="All configurations",
            ),
            Line2D(
                [], [], color="#25282c", linewidth=2, marker="o",
                markersize=10, markerfacecolor="white", markeredgewidth=1.25,
                label="Full-data Pareto frontier",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor=cmap(0.6), markeredgecolor="#6f4a22",
                label="Recommendations (color = frequency)",
            ),
        ),
        loc="lower center",
        bbox_to_anchor=(0.48, 0.035),
        ncol=3,
        frameon=False,
        fontsize=30,
        columnspacing=2.2,
        handletextpad=0.65,
    )
    figure.subplots_adjust(
        left=0.080, right=0.865, bottom=0.205, top=0.865,
        wspace=0.26, hspace=0.58,
    )
    colorbar_axis = figure.add_axes([0.900, 0.245, 0.021, 0.60])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, 20), cmap=cmap),
        cax=colorbar_axis,
    )
    colorbar.set_label(
        "Recommendation frequency (out of 20 seeds)", fontsize=30, labelpad=19
    )
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=26)
    colorbar.outline.set_linewidth(0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def plot_all_method_frontiers(
    output: Path,
    payload: dict[str, Any],
    styles: dict[str, Any],
) -> list[dict[str, Any]]:
    """Redraw every method frontier with one shared paper layout.

    This is a format-only rebuild from saved trajectories.  The last published
    recommendation remains current through the full budget, matching the
    recommendation-output semantics used for the G2 figures.
    """
    checkpoint_rows: list[dict[str, Any]] = []
    runs_by_method: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method in METHODS:
        if method == "radial_gittins":
            runs = payload["runs"]
            method_label = FULL_METHOD_NAME
            filename_method = "radial_gittins"
        else:
            runs = saved_frontier_runs(method)
            method_label = styles[method]["label"]
            filename_method = method
        runs = carry_terminal_recommendations(runs, payload["spaces"])
        runs_by_method[method] = runs
        cmap = _frequency_colormap(
            styles[method]["color"], f"{method}_recommendation_frequency"
        )
        method_output = output / filename_method
        for checkpoint in CHECKPOINTS:
            figure, axes = plt.subplots(2, 4, figsize=(24, 14.5))
            for axis, dataset in zip(axes.flat, DATASETS):
                summary, rows = frontier_summary(
                    dataset, checkpoint, runs, payload["spaces"]
                )
                checkpoint_rows.extend(
                    {"method": method, **row} for row in rows
                )
                draw_frontier_panel(
                    axis,
                    payload["raw_truth"][dataset],
                    summary,
                    DATASET_LABELS[dataset],
                    cmap,
                )
            finish_frontier_figure(
                figure,
                checkpoint=checkpoint,
                cmap=cmap,
                method_label=method_label,
                stem=(
                    method_output
                    / f"{filename_method}_20seed_frequency_frontiers_"
                    f"{round(checkpoint * 100)}pct"
                ),
            )
    plot_stackoverflow_method_frontier(
        output.parent
        / "by_dataset"
        / "stackoverflow_2x3_method_frontiers_10pct",
        checkpoint=0.10,
        runs_by_method=runs_by_method,
        payload=payload,
        styles=styles,
    )
    return checkpoint_rows


def write_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_timing_data(
    output: Path,
    by_seed: list[dict[str, str]],
    summary: list[dict[str, str]],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    by_seed_path = output / "runtime_by_seed.csv"
    with by_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "method",
                "seed",
                "total_wall_clock_seconds",
                "source",
                "slurm_job_id",
            ),
        )
        writer.writeheader()
        writer.writerows(by_seed)
    summary_path = output / "runtime_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "method",
                "n",
                "mean_seconds",
                "error_2se_seconds",
            ),
        )
        writer.writeheader()
        writer.writerows(summary)
    return summary_path


def plot_timing(summary_path: Path, output: Path) -> None:
    from experiments.combined_objective import plot_runtime_comparison

    color, _ = plot_runtime_comparison.METHOD_STYLES["radial_gittins"]
    plot_runtime_comparison.METHOD_STYLES["radial_gittins"] = (
        color,
        SHORT_METHOD_NAME,
    )
    plot_runtime_comparison.plot_runtime(summary_path, output)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def read_saved_metric_trajectory(
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty saved trajectory: {path}")
    rows.sort(key=lambda row: float(row["budget_fraction"]))
    return (
        np.asarray([float(row["budget_fraction"]) for row in rows]),
        {
            metric: np.asarray([float(row[metric]) for row in rows])
            for metric in METRICS
        },
    )


def saved_baseline_path(dataset: str, method: str, seed: int) -> Path:
    if dataset in {"hotpotqa", "mathqa"}:
        root = NEW_QA_BASELINE_ROOT
    elif method == "qnehvi":
        root = SAVED_QNEHVI_ROOT
    else:
        root = SAVED_BASELINE_ROOT
    return root / dataset / method / f"seed-{seed}" / "cost_trajectory.csv"


def saved_baseline_trajectories(
    dataset: str,
    method: str,
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    seeds = QNEHVI_SEEDS.get(dataset, SEEDS) if method == "qnehvi" else SEEDS
    trajectories = []
    for seed in seeds:
        path = saved_baseline_path(dataset, method, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing saved baseline trajectory: {path}")
        trajectories.append(read_saved_metric_trajectory(path))
    return trajectories


def saved_frontier_runs(
    method: str,
) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        seeds = QNEHVI_SEEDS.get(dataset, SEEDS) if method == "qnehvi" else SEEDS
        runs[dataset] = []
        for seed in seeds:
            path = saved_baseline_path(dataset, method, seed)
            if not path.is_file():
                raise FileNotFoundError(f"missing saved baseline trajectory: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError(f"empty saved baseline trajectory: {path}")
            points = []
            for row in rows:
                selected = tuple(
                    int(value)
                    for value in row["selected_arm_indices"].split(";")
                    if value
                )
                points.append(
                    {
                        "cost_fraction": float(row["budget_fraction"]),
                        "cost_usd": float(row["cumulative_search_cost_usd"]),
                        "selected_arm_indices": selected,
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
            runs[dataset].append({"seed": seed, "points": points})
    return runs


def g2_metric_trajectories(
    dataset_runs: list[dict[str, Any]],
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    trajectories = []
    for run in dataset_runs:
        points = run["points"]
        trajectories.append(
            (
                np.asarray([float(point["cost_fraction"]) for point in points]),
                {
                    metric: np.asarray([float(point[metric]) for point in points])
                    for metric in METRICS
                },
            )
        )
    return trajectories


def aggregate_from_mean_start(
    trajectories: list[tuple[np.ndarray, dict[str, np.ndarray]]],
    *,
    exact_event_grid: bool,
    cutoff: float,
) -> tuple[dict[str, MeanCurve], dict[str, dict[str, float]]]:
    """Align runs at the mean first-event cost with all seeds contributing.

    Runs beginning before the mean are read at the mean using their latest
    output. Runs beginning after it carry their first output backward to the
    mean. Terminal outputs are likewise carried forward as recommendation
    outputs, so every displayed point has the full seed count.
    """
    if not trajectories:
        raise ValueError("expected at least one trajectory")
    output: dict[str, MeanCurve] = {}
    alignment: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        finite_trajectories = []
        for xs, metric_values in trajectories:
            finite = np.isfinite(metric_values[metric])
            if not np.any(finite):
                raise ValueError(f"trajectory has no finite {metric} output")
            finite_trajectories.append((xs[finite], metric_values[metric][finite]))
        starts = np.asarray([float(xs[0]) for xs, _ in finite_trajectories])
        mean_start = float(np.mean(starts))
        if exact_event_grid:
            later_events = np.concatenate(
                [
                    xs[(xs > mean_start) & (xs <= cutoff + 1e-12)]
                    for xs, _ in finite_trajectories
                ]
            )
            grid = np.unique(np.concatenate(([mean_start, cutoff], later_events)))
        else:
            grid = np.unique(
                np.concatenate(
                    (
                        [mean_start],
                        GRID[(GRID > mean_start) & (GRID <= cutoff + 1e-12)],
                    )
                )
            )
        aligned = np.empty((len(trajectories), len(grid)), dtype=np.float64)
        for row_index, (xs, values) in enumerate(finite_trajectories):
            positions = np.searchsorted(xs, grid, side="right") - 1
            positions = np.clip(positions, 0, len(xs) - 1)
            aligned[row_index] = values[positions]
        mean = np.mean(aligned, axis=0)
        two_se = 2.0 * np.std(aligned, axis=0, ddof=1) / math.sqrt(len(trajectories))
        output[metric] = MeanCurve(
            grid.copy(),
            mean,
            two_se,
            np.full(len(grid), len(trajectories), dtype=int),
        )
        alignment[metric] = {
            "min_start_cost_fraction": float(np.min(starts)),
            "mean_start_cost_fraction": mean_start,
            "max_start_cost_fraction": float(np.max(starts)),
        }
    return output, alignment


def curve_rows_from_curves(
    curves: dict[tuple[str, str, str], MeanCurve],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for metric in METRICS:
                curve = curves[(dataset, method, metric)]
                for x, mean, two_se, count in zip(
                    curve.x, curve.mean, curve.two_se, curve.count
                ):
                    if not np.isfinite(mean) or not np.isfinite(two_se):
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "metric": metric,
                            "cost_fraction": str(float(x)),
                            "mean": str(float(mean)),
                            "two_se": str(float(two_se)),
                            "n_runs": str(int(count)),
                        }
                    )
    return rows


def write_carried_recommendation_outputs(
    output: Path,
    payload: dict[str, Any],
    styles: dict[str, Any],
    cache_path: Path,
) -> None:
    cutoff = 0.30
    extended_runs = carry_terminal_recommendations(
        payload["runs"], payload["spaces"]
    )
    aligned_curves: dict[tuple[str, str, str], MeanCurve] = {}
    start_alignment: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        dataset: {} for dataset in DATASETS
    }
    for dataset in DATASETS:
        for method in METHODS:
            if method == "radial_gittins":
                trajectories = g2_metric_trajectories(extended_runs[dataset])
            else:
                trajectories = saved_baseline_trajectories(dataset, method)
            method_curves, alignment = aggregate_from_mean_start(
                trajectories,
                exact_event_grid=method in {
                    "random_configurations", "random_questions"
                },
                cutoff=cutoff,
            )
            start_alignment[dataset][method] = alignment
            for metric, curve in method_curves.items():
                expected_count = len(
                    QNEHVI_SEEDS.get(dataset, SEEDS)
                    if method == "qnehvi"
                    else SEEDS
                )
                if not np.all(curve.count == expected_count):
                    raise AssertionError(
                        f"{dataset} {method} {metric}: incomplete seed count"
                    )
                aligned_curves[(dataset, method, metric)] = curve

    cutoff_rows = curve_rows_from_curves(aligned_curves)

    curve_output = output / "curves_0_30pct"
    curve_output.mkdir(parents=True, exist_ok=True)
    write_curve_csv(curve_output / "curve_summary_0_30pct.csv", cutoff_rows)
    (curve_output / "method_styles.json").write_text(
        json.dumps(styles, indent=2) + "\n", encoding="utf-8"
    )
    for metric in METRICS:
        plot_metric(
            metric,
            aligned_curves,
            styles,
            curve_output / f"eight_datasets_{metric}",
            x_max=cutoff,
        )

    for checkpoint in CHECKPOINTS:
        for dataset in DATASETS:
            original, original_rows = frontier_summary(
                dataset, checkpoint, payload["runs"], payload["spaces"]
            )
            extended, extended_rows = frontier_summary(
                dataset, checkpoint, extended_runs, payload["spaces"]
            )
            if original.counts != extended.counts:
                raise AssertionError(
                    f"{dataset} {checkpoint:.0%}: recommendation frequencies changed"
                )
            original_membership = {
                (row["seed"], row["selected_arm_indices"])
                for row in original_rows
            }
            extended_membership = {
                (row["seed"], row["selected_arm_indices"])
                for row in extended_rows
            }
            if original_membership != extended_membership:
                raise AssertionError(
                    f"{dataset} {checkpoint:.0%}: checkpoint recommendations changed"
                )

    checkpoint_rows = plot_frontiers(
        output / "frontier",
        extended_runs,
        payload["raw_truth"],
        payload["spaces"],
    )
    write_checkpoint_csv(output / "frontier_checkpoint_data.csv", checkpoint_rows)

    extended_seed_counts = {
        dataset: sum(
            len(extended["points"]) > len(original["points"])
            for original, extended in zip(payload["runs"][dataset], extended_runs[dataset])
        )
        for dataset in DATASETS
    }
    manifest = {
        "configuration": "g2_exact_axes",
        "public_method_label": FULL_METHOD_NAME,
        "legend_method_label": SHORT_METHOD_NAME,
        "recommendation_output_semantics": (
            "last published recommendation carried forward to 100% budget"
        ),
        "synthetic_terminal_points_are_evaluations": False,
        "displayed_cost_fraction_range": [0.0, cutoff],
        "aggregation": (
            "align each method-dataset at the mean per-seed first-event cost; "
            "truncate earlier prefixes, backfill later starts with their first "
            "output, then use stepwise last observation carried forward; "
            "mean +/- 2 SE over the retained seed set"
        ),
        "seeds": list(SEEDS),
        "qnehvi_seeds_by_dataset": {
            dataset: list(QNEHVI_SEEDS.get(dataset, SEEDS))
            for dataset in DATASETS
        },
        "qnehvi_seed_counts_by_dataset": {
            dataset: len(QNEHVI_SEEDS.get(dataset, SEEDS))
            for dataset in DATASETS
        },
        "datasets": list(DATASETS),
        "start_alignment": start_alignment,
        "terminal_extensions_by_dataset": extended_seed_counts,
        "all_curve_points_have_full_retained_seed_set": True,
        "baseline_source": (
            "saved per-seed cost_trajectory.csv files; no optimizer reruns"
        ),
        "qa_baseline_source": (
            "analysis/usd_cost_checkpoints_latest_under_20seed/"
            "new_qa_runs/baselines"
        ),
        "qnehvi_non_qa_source": "analysis/final_run_baselines",
        "other_non_qa_baseline_source": (
            "analysis/usd_cost_checkpoints_latest_under_20seed/runs"
        ),
        "frontier_checkpoint_recommendations_unchanged": True,
        "frontier_recommendation_frequencies_unchanged": True,
        "source_plot_cache": "../plot_cache.pkl",
        "source_plot_cache_sha256": sha256(cache_path),
        "plot_ready_curve_data": "curve_summary_0_30pct.csv",
        "frontier_checkpoint_data": "../frontier_checkpoint_data.csv",
    }
    (curve_output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (curve_output / "README.md").write_text(
        "# G2 curves at 0--30% budget\n\n"
        "These curves use the existing G2 plot cache. A stopped seed's last "
        "published recommendation is carried forward as an output through 100% "
        "budget, without adding evaluations. Only 0--30% is displayed. For every "
        "method and dataset, the common start is the mean first-event cost across "
        "the retained seeds: earlier prefixes are truncated and later starts carry their "
        "first output backward. Every displayed aggregate point therefore contains "
        "the full retained set (20 normally; qNEHVI uses 19 for BIRD Mini Dev "
        "and 18 for Bing Query Logs). Baselines are rebuilt only from saved per-seed trajectory "
        "CSVs; no optimizer is rerun.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Draw using plot_cache.pkl without reading result JSONs.")
    parser.add_argument(
        "--carry-terminal-to-full-budget",
        action="store_true",
        help=(
            "Carry stopped runs' last recommendations to 100%, draw separate "
            "0-30% curves, and update frontier stability windows. Requires "
            "--reuse-cache."
        ),
    )
    parser.add_argument(
        "--refresh-all-frontiers",
        action="store_true",
        help=(
            "Redraw all six methods' 10% and 30% frontier figures from saved "
            "trajectories using the shared paper layout. Requires --reuse-cache."
        ),
    )
    args = parser.parse_args()
    if args.carry_terminal_to_full_budget and not args.reuse_cache:
        parser.error("--carry-terminal-to-full-budget requires --reuse-cache")
    if args.refresh_all_frontiers and not args.reuse_cache:
        parser.error("--refresh-all-frontiers requires --reuse-cache")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "plot_cache.pkl"
    configure_matplotlib()

    if args.reuse_cache:
        payload = read_cache(cache_path)
    else:
        source_rows, styles = read_source_rows()
        runs, raw_truth, spaces, g2_runtime_rows = load_g2_runs()
        g2_curves = aggregate_g2(runs)
        rows = combined_rows(source_rows, g2_curves)
        runtime_by_seed, runtime_summary = combined_timing_rows(g2_runtime_rows)
        payload = {
            "version": CACHE_VERSION,
            "method": "radial_gittins",
            "configuration": "g2_exact_axes",
            "directions": ((0.0, 1.0), (1.0, 0.0)),
            "seeds": SEEDS,
            "datasets": DATASETS,
            "curve_rows": rows,
            "styles": styles,
            "runs": runs,
            "raw_truth": raw_truth,
            "spaces": spaces,
            "runtime_by_seed": runtime_by_seed,
            "runtime_summary": runtime_summary,
            "source_curve_summary_sha256": sha256(SOURCE_FIGURES / "curve_summary.csv"),
            "source_runtime_by_seed_sha256": sha256(
                SOURCE_TIMING / "runtime_by_seed.csv"
            ),
            "source_runtime_summary_sha256": sha256(
                SOURCE_TIMING / "runtime_summary.csv"
            ),
        }
        write_cache(cache_path, payload)

    rows = payload["curve_rows"]
    styles = {
        method: style.copy()
        for method, style in payload["styles"].items()
    }
    styles["radial_gittins"]["label"] = SHORT_METHOD_NAME
    if args.refresh_all_frontiers:
        frontier_output = FIGURE_ROOT / "frontier" / "by_method"
        checkpoint_rows = plot_all_method_frontiers(
            frontier_output, payload, styles
        )
        write_checkpoint_csv(
            frontier_output / "all_method_frontier_checkpoint_data.csv",
            checkpoint_rows,
        )
        print(frontier_output)
        return
    if args.carry_terminal_to_full_budget:
        write_carried_recommendation_outputs(
            output, payload, styles, cache_path
        )
        print(output)
        return
    curves = curves_from_rows(rows)
    write_curve_csv(output / "curve_summary.csv", rows)
    (output / "method_styles.json").write_text(
        json.dumps(styles, indent=2) + "\n", encoding="utf-8"
    )
    curve_dir = output / "curves"
    for metric in METRICS:
        plot_metric(metric, curves, styles, curve_dir / f"eight_datasets_{metric}")
    checkpoint_rows = plot_frontiers(
        output / "frontier",
        payload["runs"],
        payload["raw_truth"],
        payload["spaces"],
    )
    write_checkpoint_csv(output / "frontier_checkpoint_data.csv", checkpoint_rows)
    timing_summary_path = write_timing_data(
        output / "time", payload["runtime_by_seed"], payload["runtime_summary"]
    )
    plot_timing(timing_summary_path, output / "time")

    source_rows, _ = read_source_rows()
    baseline_source = [row for row in source_rows if row["method"] != "radial_gittins"]
    baseline_output = [row for row in rows if row["method"] != "radial_gittins"]
    if baseline_output != baseline_source:
        raise AssertionError("baseline curves differ from the source main figures")
    manifest = {
        "configuration": "g2_exact_axes",
        "public_method_label": FULL_METHOD_NAME,
        "legend_method_label": SHORT_METHOD_NAME,
        "directions": [[0.0, 1.0], [1.0, 0.0]],
        "seeds": list(SEEDS),
        "datasets": list(DATASETS),
        "updated_scope": [
            "radial_gittins main curves",
            "radial_gittins frequency frontiers",
            "radial_gittins wall-clock timing",
        ],
        "untouched_scope": ["all five baselines", "ablation figures"],
        "baseline_rows_copied_verbatim": True,
        "baseline_row_count": len(baseline_output),
        "baseline_timing_rows_copied_verbatim": True,
        "qa_configuration_grid": {
            "hotpotqa": "10x10 (100 configurations)",
            "mathqa": "10x10 (100 configurations)",
        },
        "frontier_style_source": (
            "historical plot_latest_under_checkpoint_frontiers.py layout; "
            "24x14.5 inches at 200 dpi"
        ),
        "metric_space": "shared reciprocal-cost desirability space",
        "cost_coordinate": "c_ref / (c_ref + mean_cost_usd)",
        "cost_reference": "median positive exhaustive mean USD cost per dataset",
        "hv_regret": "absolute exhaustive-front HV minus recommendation HV",
        "uncertainty": "mean +/- 2 standard errors",
        "plot_cache": "plot_cache.pkl",
        "plot_ready_curve_data": "curve_summary.csv",
        "plot_ready_frontier_data": "frontier_checkpoint_data.csv",
        "plot_ready_timing_data": "time/runtime_summary.csv",
        "source_curve_summary_sha256": payload["source_curve_summary_sha256"],
        "source_runtime_by_seed_sha256": payload["source_runtime_by_seed_sha256"],
        "source_runtime_summary_sha256": payload["source_runtime_summary_sha256"],
        "spaces": payload["spaces"],
    }
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Gittins G2 main figures\n\n"
        f"Only {FULL_METHOD_NAME} is replaced by `g2_exact_axes`; all five baseline curve "
        "rows are copied verbatim from `../hv_frontier_source/curve_summary.csv`. "
        f"The timing plot likewise replaces only {FULL_METHOD_NAME}. Frontier plots "
        "reuse the exact historical 4800x2900 layout. HotpotQA and MathQA are "
        "validated as 10x10 (100-configuration) grids. Ablation figures are not "
        "modified.\n\n"
        "For format-only changes without reading raw result JSONs:\n\n"
        "```bash\n"
        ".venv-plot/bin/python experiments/combined_objective/plot_g2_main_figures.py --reuse-cache\n"
        "```\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
