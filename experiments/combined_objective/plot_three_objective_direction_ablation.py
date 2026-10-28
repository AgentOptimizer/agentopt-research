#!/usr/bin/env python3
"""Plot the three-objective direction-set ablation in the paper style.

All compared runs use real acquisition costs, independent direction-local
eta decay, and equal round-robin scheduling. GD and IGD are recomputed from
the complete returned set, including truth-dominated returned assignments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RELATIVE = Path(
    "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    "/gittins_g2_main_figures"
)
DATA_REPOSITORY = ROOT
LINKED_REPOSITORY = ROOT.parent / "agentopt-research"
if not (DATA_REPOSITORY / ANALYSIS_RELATIVE / "source_data").is_dir() and (
    LINKED_REPOSITORY / ANALYSIS_RELATIVE / "source_data"
).is_dir():
    DATA_REPOSITORY = LINKED_REPOSITORY
SOURCE = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "source_data/ablation/three_objective_20seed"
)
OUTPUT_DIR = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "corrected_gd_igd_complete_returned_set_20260929"
    / "three_objective/direction_ablation"
)

SEEDS = tuple(range(42, 62))
DATASETS = ("hotpotqa", "mathqa")
DATASET_LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
GRID = np.arange(1, 61, dtype=np.float64) / 200.0
CHECKPOINTS = (0.05, 0.10, 0.20, 0.30)
CURVE_LINEWIDTH = 2.4
HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

CONFIGURATIONS = (
    "q_only_real_cost",
    "l_only_real_cost",
    "d_only_real_cost",
    "pure_axes_real_cost",
    "four_direction_real_cost",
    "six_ql_dense_real_cost",
    "six_three_edges_real_cost",
)
DIRECTIONS = {
    "q_only_real_cost": ((1.0, 0.0, 0.0),),
    "l_only_real_cost": ((0.0, 1.0, 0.0),),
    "d_only_real_cost": ((0.0, 0.0, 1.0),),
    "pure_axes_real_cost": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "four_direction_real_cost": (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "six_ql_dense_real_cost": (
        (1.0, 0.0, 0.0),
        (0.75, 0.25, 0.0),
        (0.5, 0.5, 0.0),
        (0.25, 0.75, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "six_three_edges_real_cost": (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5),
    ),
}
STYLES: dict[str, dict[str, str | float]] = {
    "q_only_real_cost": {
        "color": "tab:gray", "label": "Q-only",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "l_only_real_cost": {
        "color": "tab:brown", "label": "L-only",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "d_only_real_cost": {
        "color": "tab:olive", "label": "D-only",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "pure_axes_real_cost": {
        "color": "tab:blue", "label": "Pure axes",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "four_direction_real_cost": {
        "color": "tab:orange", "label": "Four directions (+ Q-L midpoint)",
        "linewidth": HIGHLIGHT_LINEWIDTH, "linestyle": "-",
    },
    "six_ql_dense_real_cost": {
        "color": "tab:green", "label": "Six directions (Q-L dense)",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
    "six_three_edges_real_cost": {
        "color": "tab:purple", "label": "Six directions (three edges)",
        "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
    },
}
METRICS = (
    ("hv_regret", "HV Regret"),
    ("generational_distance", "GD"),
    ("inverted_generational_distance", "IGD"),
)
NATURAL_STOPS = {
    "direction_eta_numerical_floor",
    "all_arms_completed",
    "all_cells_observed",
    "question_budget",
    "all_arms_exhausted",
}


def nondominated_indices(points: np.ndarray) -> np.ndarray:
    """Return indices not dominated when all scoring coordinates are maximized."""
    array = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)


def front_distance(obtained: np.ndarray, reference: np.ndarray) -> float:
    """Mean nearest-point distance without filtering either supplied set."""
    if not len(obtained):
        return math.inf if len(reference) else 0.0
    if not len(reference):
        return math.inf
    delta = obtained[:, None, :] - reference[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1).mean())


def evaluation_space(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw_truth_vectors must have shape (n, 3)")
    positive_latency = raw[:, 1][raw[:, 1] > 0.0]
    positive_cost = raw[:, 2][raw[:, 2] > 0.0]
    latency_reference = float(np.median(positive_latency)) if len(positive_latency) else 1.0
    cost_reference = float(np.median(positive_cost)) if len(positive_cost) else 1.0
    metric_truth = np.column_stack((
        raw[:, 0],
        latency_reference / (latency_reference + raw[:, 1]),
        cost_reference / (cost_reference + raw[:, 2]),
    ))
    return metric_truth, metric_truth[nondominated_indices(metric_truth)]


def validate_run(
    payload: dict[str, Any], configuration: str, dataset: str, seed: int,
) -> None:
    config = payload["config"]
    run = payload["run"]
    checks = {
        "ablation_configuration": configuration,
        "benchmark": dataset,
        "seed": seed,
        "directions": [list(direction) for direction in DIRECTIONS[configuration]],
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "direction_scheduler": "round_robin",
    }
    for field, expected in checks.items():
        if config.get(field) != expected:
            raise ValueError(
                f"{field} mismatch for {configuration}/{dataset}/seed-{seed}: "
                f"{config.get(field)!r} != {expected!r}"
            )
    expected_selector = f"three_objective_cc_gittins_{configuration}"
    if run.get("selector") != expected_selector:
        raise ValueError(f"selector mismatch for {configuration}/{dataset}/seed-{seed}")
    if not run.get("points"):
        raise ValueError(f"empty trajectory for {configuration}/{dataset}/seed-{seed}")


def aligned_run(run: dict[str, Any]) -> dict[str, np.ndarray]:
    points = run["points"]
    costs = np.asarray([float(point["cost_fraction"]) for point in points])
    if np.any(np.diff(costs) < -1e-12):
        raise ValueError("cost trajectory is not nondecreasing")
    positions = np.searchsorted(costs, GRID + 1e-12, side="right") - 1
    available = positions >= 0
    terminal = float(run["cost_fraction"])
    if run["stop_reason"] not in NATURAL_STOPS:
        available &= GRID <= terminal + 1e-12

    metric_truth, truth_front = evaluation_space(run)
    output = {
        field: np.full(len(GRID), np.nan, dtype=np.float64)
        for field, _ in METRICS
    }
    for column in np.flatnonzero(available):
        point = points[int(positions[column])]
        selected = np.asarray(point["selected_arm_indices"], dtype=int)
        obtained = metric_truth[selected]
        values = {
            "hv_regret": float(point["hv_regret"]),
            "generational_distance": front_distance(obtained, truth_front),
            "inverted_generational_distance": front_distance(truth_front, obtained),
        }
        for field, value in values.items():
            if math.isfinite(value):
                output[field][column] = value
    return output


def summarize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    two_se = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column, count in enumerate(counts):
        finite = values[:, column][np.isfinite(values[:, column])]
        if count:
            means[column] = float(np.mean(finite))
            two_se[column] = (
                2.0 * float(np.std(finite, ddof=1)) / math.sqrt(int(count))
                if count > 1 else 0.0
            )
    return means, two_se, counts


def result_path(source: Path, configuration: str, dataset: str, seed: int) -> Path:
    return source / configuration / f"seed-{seed}" / dataset / "result.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_tables(source: Path, output_dir: Path) -> None:
    curve_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    input_hashes: dict[tuple[str, int], dict[str, str]] = {}
    checkpoint_columns = {
        checkpoint: int(np.flatnonzero(np.isclose(GRID, checkpoint))[0])
        for checkpoint in CHECKPOINTS
    }

    for dataset in DATASETS:
        for configuration in CONFIGURATIONS:
            print(f"Aggregating {dataset} / {configuration}", flush=True)
            metric_rows = {field: [] for field, _ in METRICS}
            for seed in SEEDS:
                path = result_path(source, configuration, dataset, seed)
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                validate_run(payload, configuration, dataset, seed)
                hash_key = (dataset, seed)
                hashes = payload["config"]["input_sha256"]
                if hash_key in input_hashes and input_hashes[hash_key] != hashes:
                    raise ValueError(f"input hashes differ across configurations: {path}")
                input_hashes[hash_key] = hashes
                aligned = aligned_run(payload["run"])
                for field, _ in METRICS:
                    metric_rows[field].append(aligned[field])
                    for checkpoint, column in checkpoint_columns.items():
                        value = aligned[field][column]
                        seed_rows.append({
                            "dataset": dataset,
                            "configuration": configuration,
                            "label": STYLES[configuration]["label"],
                            "seed": seed,
                            "cost_fraction": checkpoint,
                            "metric": field,
                            "value": None if not np.isfinite(value) else float(value),
                        })
                del payload

            for field, label in METRICS:
                values = np.asarray(metric_rows[field], dtype=np.float64)
                means, two_se, counts = summarize(values)
                for x, mean, spread, count in zip(GRID, means, two_se, counts):
                    curve_rows.append({
                        "dataset": dataset,
                        "configuration": configuration,
                        "label": STYLES[configuration]["label"],
                        "metric": field,
                        "metric_label": label,
                        "cost_fraction": float(x),
                        "mean": None if not np.isfinite(mean) else float(mean),
                        "two_se": None if not np.isfinite(spread) else float(spread),
                        "n_runs": int(count),
                    })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "direction_curve_summary.csv", curve_rows)
    write_csv(output_dir / "direction_seed_checkpoints.csv", seed_rows)

    checkpoint_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for configuration in CONFIGURATIONS:
            for checkpoint in CHECKPOINTS:
                for field, label in METRICS:
                    values = [
                        float(row["value"])
                        for row in seed_rows
                        if row["dataset"] == dataset
                        and row["configuration"] == configuration
                        and row["cost_fraction"] == checkpoint
                        and row["metric"] == field
                        and row["value"] is not None
                    ]
                    count = len(values)
                    checkpoint_rows.append({
                        "dataset": dataset,
                        "configuration": configuration,
                        "label": STYLES[configuration]["label"],
                        "cost_fraction": checkpoint,
                        "metric": field,
                        "metric_label": label,
                        "mean": float(np.mean(values)) if values else None,
                        "two_se": (
                            2.0 * float(np.std(values, ddof=1)) / math.sqrt(count)
                            if count > 1 else 0.0 if count else None
                        ),
                        "n_runs": count,
                    })
    write_csv(output_dir / "direction_checkpoint_summary.csv", checkpoint_rows)

    auc_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for configuration in CONFIGURATIONS:
            for field, label in METRICS:
                selected = [
                    row for row in curve_rows
                    if row["dataset"] == dataset
                    and row["configuration"] == configuration
                    and row["metric"] == field
                    and 0.05 <= float(row["cost_fraction"]) <= 0.30
                    and row["mean"] is not None
                ]
                xs = np.asarray([row["cost_fraction"] for row in selected], dtype=float)
                ys = np.asarray([row["mean"] for row in selected], dtype=float)
                auc_rows.append({
                    "dataset": dataset,
                    "configuration": configuration,
                    "label": STYLES[configuration]["label"],
                    "metric": field,
                    "metric_label": label,
                    "range_start": 0.05,
                    "range_end": 0.30,
                    "normalized_auc": (
                        float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
                        if len(xs) > 1 else None
                    ),
                })
    write_csv(output_dir / "direction_auc_5_30_summary.csv", auc_rows)

    pair_rows: list[dict[str, Any]] = []
    seed_lookup = {
        (
            row["dataset"], row["configuration"], row["cost_fraction"],
            row["metric"], row["seed"],
        ): row["value"]
        for row in seed_rows
    }
    pairs = (
        ("four_direction_real_cost", "pure_axes_real_cost"),
        ("four_direction_real_cost", "six_ql_dense_real_cost"),
        ("four_direction_real_cost", "six_three_edges_real_cost"),
        ("six_ql_dense_real_cost", "six_three_edges_real_cost"),
    )
    for dataset in DATASETS:
        for checkpoint in CHECKPOINTS:
            for field, label in METRICS:
                for reference, comparison in pairs:
                    differences = []
                    for seed in SEEDS:
                        ref = seed_lookup[(dataset, reference, checkpoint, field, seed)]
                        comp = seed_lookup[(dataset, comparison, checkpoint, field, seed)]
                        if ref is not None and comp is not None:
                            differences.append(float(comp) - float(ref))
                    values = np.asarray(differences, dtype=float)
                    tolerance = 1e-12
                    pair_rows.append({
                        "dataset": dataset,
                        "cost_fraction": checkpoint,
                        "metric": field,
                        "metric_label": label,
                        "reference_configuration": reference,
                        "comparison_configuration": comparison,
                        "mean_comparison_minus_reference": float(np.mean(values)),
                        "two_se_difference": (
                            2.0 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
                            if len(values) > 1 else 0.0
                        ),
                        "reference_wins": int(np.sum(values > tolerance)),
                        "ties": int(np.sum(np.abs(values) <= tolerance)),
                        "comparison_wins": int(np.sum(values < -tolerance)),
                        "n_pairs": len(values),
                    })
    write_csv(output_dir / "direction_pairwise_checkpoints.csv", pair_rows)


def load_curve_summary(
    path: Path,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["mean"] or not row["two_se"]:
                continue
            key = (row["dataset"], row["configuration"], row["metric"])
            grouped.setdefault(key, []).append((
                float(row["cost_fraction"]),
                float(row["mean"]),
                float(row["two_se"]),
                int(row["n_runs"]),
            ))
    curves = {}
    for key, rows in grouped.items():
        array = np.asarray(rows, dtype=np.float64)
        curves[key] = (array[:, 0], array[:, 1], array[:, 2], array[:, 3])
    return curves


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def plot(summary_path: Path, output_stem: Path) -> None:
    curves = load_curve_summary(summary_path)
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    compact = FuncFormatter(lambda value, _: f"{value:g}")

    for row, dataset in enumerate(DATASETS):
        for column, (field, title) in enumerate(METRICS):
            axis = axes[row, column]
            for configuration in CONFIGURATIONS:
                xs, means, two_se, _ = curves[(dataset, configuration, field)]
                style = STYLES[configuration]
                axis.plot(
                    xs, means, color=str(style["color"]),
                    linewidth=float(style["linewidth"]),
                    linestyle=str(style["linestyle"]),
                    zorder=5 if configuration == "four_direction_real_cost" else 3,
                )
                axis.fill_between(
                    xs, np.maximum(0.0, means - two_se), means + two_se,
                    color=str(style["color"]), alpha=0.10, linewidth=0, zorder=1,
                )
            if row == 0:
                axis.set_title(title, fontsize=28, pad=13)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis="both", labelsize=26, width=1.0, length=5)
            axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)

    subplot_bottom = 0.37
    subplot_top = 0.925
    figure.text(
        0.535, 0.265, "Percentage of Exhaustive Evaluation Cost",
        ha="center", va="center", fontsize=30,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, top=subplot_top, bottom=subplot_bottom,
        wspace=0.30, hspace=0.34,
    )
    row_centers = [
        (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0
        for row in range(2)
    ]
    for row, dataset in enumerate(DATASETS):
        figure.text(
            0.020, row_centers[row], DATASET_LABELS[dataset],
            ha="center", va="center", rotation=90, fontsize=31,
        )
    handles = [
        Line2D(
            [], [], color=str(STYLES[configuration]["color"]),
            linewidth=float(STYLES[configuration]["linewidth"]),
            linestyle=str(STYLES[configuration]["linestyle"]),
            label=str(STYLES[configuration]["label"]),
        )
        for configuration in CONFIGURATIONS
    ]
    handles.append(Patch(
        facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE",
    ))
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, frameon=False, fontsize=20, handlelength=2.8,
        columnspacing=1.0, handletextpad=0.55, labelspacing=0.45,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.12}
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, **save_options)
    figure.savefig(output_stem.with_suffix(".pdf"), **save_options)
    figure.savefig(output_stem.with_suffix(".svg"), **save_options)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reuse-summary", action="store_true")
    args = parser.parse_args()
    configure_matplotlib()
    summary = args.output_dir / "direction_curve_summary.csv"
    if not args.reuse_summary or not summary.is_file():
        build_tables(args.source, args.output_dir)
    output_stem = args.output_dir / "three_objective_direction_ablation_hv_gd_igd"
    plot(summary, output_stem)
    metadata = {
        "source": str(args.source),
        "configurations": list(CONFIGURATIONS),
        "directions": {key: DIRECTIONS[key] for key in CONFIGURATIONS},
        "fixed_protocol": {
            "acquisition_cost_mode": "real",
            "continuation_mode": "eta_decay",
            "direction_scheduler": "round_robin",
        },
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "cost_range": [float(GRID[0]), float(GRID[-1])],
        "grid_step": 0.005,
        "checkpoints": list(CHECKPOINTS),
        "budget_alignment": (
            "latest completed recommendation at or below each actual-USD target; "
            "no value before a seed's first recommendation; retain the final "
            "recommendation after a natural policy stop"
        ),
        "uncertainty": "mean +/- 2 sample standard errors over available matched seeds",
        "hv_metric": "absolute hypervolume regret",
        "distance_metric": (
            "GD/IGD recomputed from every returned configuration, including "
            "configurations dominated in true evaluation space"
        ),
        "styles": STYLES,
        "paper_style_template": "plot_three_objective_corrected_gd_igd.py",
    }
    (args.output_dir / "direction_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
