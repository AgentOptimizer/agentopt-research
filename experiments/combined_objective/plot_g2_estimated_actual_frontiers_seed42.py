#!/usr/bin/env python3
"""Plot seed-42 estimated/actual frontiers with the main Gittins method.

Baseline checkpoint payloads are reused unchanged.  Only the Radial Gittins
row is rebuilt from the exact-axes main run.  Plot-ready inputs are cached so
``--reuse-cache`` does not read the result JSON or baseline gzip files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_runs/combined"
    / "estimated_actual_frontiers_seed42/runs"
)
G2_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_runs/combined"
    / "gittins_ablation_8bench_20seed/g2_exact_axes/seed-42/exact_axes"
)
DEFAULT_OUTPUT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    / "gittins_g2_main_figures/estimated_actual"
)
DATASETS = (("hotpotqa", "HotpotQA"), ("stackoverflow", "Stack Overflow"))
FULL_METHOD_NAME = "Cost-Coupled Gittins"
METHODS = (
    ("radial_gittins", FULL_METHOD_NAME),
    ("ege_sh", "EGE-SH"),
    ("ape_k", "APE-k"),
    ("qnehvi", "qNEHVI"),
    ("random_questions", "Random questions"),
    ("random_configurations", "Random configurations"),
)
TARGETS = (0.10, 0.30)
SEED = 42
CACHE_VERSION = 1
RECOMMENDATION_COLOR = "#d55e00"
RECOMMENDATION_EDGE = "#725b46"
BACKGROUND_COLOR = "#d8dce2"
FRONTIER_COLOR = "#3f4854"


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _source_payload(dataset: str, method: str) -> dict[str, Any]:
    path = SOURCE_ROOT / dataset / method / f"seed-{SEED}" / "frontier_events.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported source schema in {path}")
    return payload


def _compact_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_arm_indices": tuple(map(int, checkpoint["selected_arm_indices"])),
        "estimated_vectors": np.asarray(
            checkpoint["estimated_raw_archive_vectors"], dtype=np.float64
        ).reshape((-1, 2)),
        "actual_vectors": np.asarray(
            checkpoint["actual_raw_archive_vectors"], dtype=np.float64
        ).reshape((-1, 2)),
        "snapshot_cost_fraction": float(checkpoint["actual_search_cost_percent"]) / 100.0,
        "cumulative_evaluations": int(checkpoint["cumulative_evaluations"]),
    }


def _g2_checkpoints(dataset: str, truth: np.ndarray) -> dict[float, dict[str, Any]]:
    path = G2_ROOT / dataset / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    if config.get("pair_name") != "exact_axes" or config.get("directions") != [
        [0.0, 1.0], [1.0, 0.0]
    ]:
        raise ValueError(f"not an exact-axes main run: {path}")
    run = payload["run"]
    local_truth = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
    if local_truth.shape != truth.shape or not np.allclose(
        local_truth, truth, rtol=0.0, atol=1e-15
    ):
        raise ValueError(f"truth mismatch between G2 and plotting source for {dataset}")
    output = {}
    for target in TARGETS:
        eligible = [
            point
            for point in run["points"]
            if float(point["cost_fraction"]) <= target + 1e-12
        ]
        if not eligible:
            raise ValueError(f"no Gittins recommendation at {target:.0%}: {path}")
        point = eligible[-1]
        selected = tuple(map(int, point["selected_arm_indices"]))
        estimated = np.asarray(
            point["estimated_raw_archive_vectors"], dtype=np.float64
        ).reshape((-1, 2))
        actual = np.asarray(
            point["offline_raw_selected_vectors"], dtype=np.float64
        ).reshape((-1, 2))
        if len(selected) != len(estimated) or len(selected) != len(actual):
            raise ValueError(f"archive length mismatch in {path} at {target:.0%}")
        if not np.allclose(actual, truth[np.asarray(selected)], rtol=0.0, atol=1e-15):
            raise ValueError(f"actual archive mismatch in {path} at {target:.0%}")
        output[target] = {
            "selected_arm_indices": selected,
            "estimated_vectors": estimated,
            "actual_vectors": actual,
            "snapshot_cost_fraction": float(point["cost_fraction"]),
            "cumulative_evaluations": int(point["evaluations"]),
        }
    return output


def build_cache() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, _ in DATASETS:
        radial_source = _source_payload(dataset, "radial_gittins")
        truth = np.asarray(radial_source["truth_raw_vectors"], dtype=np.float64)
        if dataset in {"hotpotqa", "mathqa"} and truth.shape != (100, 2):
            raise ValueError(
                f"{dataset} must use the 10x10 grid (100 configurations), "
                f"got {truth.shape}"
            )
        frontier = np.asarray(
            radial_source["full_data_pareto_arm_indices"], dtype=int
        )
        method_data: dict[str, Any] = {
            "radial_gittins": _g2_checkpoints(dataset, truth)
        }
        for method, _ in METHODS[1:]:
            payload = _source_payload(dataset, method)
            local_truth = np.asarray(payload["truth_raw_vectors"], dtype=np.float64)
            if local_truth.shape != truth.shape or not np.allclose(
                local_truth, truth, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"baseline truth mismatch for {dataset}/{method}")
            method_data[method] = {
                target: _compact_checkpoint(
                    payload["cost_checkpoints"][f"{round(target * 100)}pct"]
                )
                for target in TARGETS
            }
        datasets[dataset] = {
            "truth": truth,
            "frontier_indices": frontier,
            "methods": method_data,
        }
    return {
        "version": CACHE_VERSION,
        "seed": SEED,
        "targets": TARGETS,
        "datasets": datasets,
        "configuration_grids": {
            "hotpotqa": "10x10 (100 configurations)",
        },
    }


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


def _draw_panel(
    axis: Any,
    *,
    truth: np.ndarray,
    frontier: np.ndarray,
    vectors: np.ndarray,
) -> None:
    axis.scatter(
        truth[:, 1], truth[:, 0], s=48, color=BACKGROUND_COLOR,
        alpha=0.78, edgecolors="none", zorder=1,
    )
    axis.plot(
        frontier[:, 1], frontier[:, 0], color=FRONTIER_COLOR,
        linewidth=2.1, marker="o", markersize=8,
        markerfacecolor="white", markeredgewidth=1.6, zorder=3,
    )
    if len(vectors):
        axis.scatter(
            vectors[:, 1], vectors[:, 0], s=180,
            color=RECOMMENDATION_COLOR, edgecolors=RECOMMENDATION_EDGE,
            linewidths=1.2, zorder=5,
        )
    plotted_costs = np.concatenate((truth[:, 1], vectors[:, 1]))
    if np.all(plotted_costs > 0.0):
        axis.set_xscale("log")
    axis.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(color="#d9dde3", linewidth=0.55, alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=23)


def render_target(cache: dict[str, Any], target: float, output: Path) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(len(METHODS), 4, figsize=(18.0, 20.0))
    rows: list[dict[str, Any]] = []
    for dataset_index, (dataset, _) in enumerate(DATASETS):
        actual_column = 2 * dataset_index
        estimated_column = actual_column + 1
        data = cache["datasets"][dataset]
        truth = np.asarray(data["truth"])
        frontier = truth[np.asarray(data["frontier_indices"], dtype=int)]
        frontier = frontier[np.argsort(frontier[:, 1])]
        scale_vectors = [truth]
        for row, (method, method_label) in enumerate(METHODS):
            checkpoint = data["methods"][method][target]
            estimated = np.asarray(checkpoint["estimated_vectors"])
            actual = np.asarray(checkpoint["actual_vectors"])
            scale_vectors.extend((estimated, actual))
            _draw_panel(
                axes[row, actual_column], truth=truth, frontier=frontier,
                vectors=actual,
            )
            _draw_panel(
                axes[row, estimated_column], truth=truth, frontier=frontier,
                vectors=estimated,
            )
            if dataset_index == 0:
                axes[row, 0].set_ylabel(method_label, fontsize=22, labelpad=14)
            rows.append({
                "target_cost_fraction": target,
                "dataset": dataset,
                "method": method,
                "seed": SEED,
                "snapshot_cost_fraction": checkpoint["snapshot_cost_fraction"],
                "cumulative_evaluations": checkpoint["cumulative_evaluations"],
                "selected_arm_indices": json.dumps(
                    checkpoint["selected_arm_indices"], separators=(",", ":")
                ),
                "estimated_vectors": json.dumps(estimated.tolist(), separators=(",", ":")),
                "actual_vectors": json.dumps(actual.tolist(), separators=(",", ":")),
            })
        stacked = np.vstack([values for values in scale_vectors if len(values)])
        xmin, xmax = float(stacked[:, 1].min()), float(stacked[:, 1].max())
        ymin, ymax = float(stacked[:, 0].min()), float(stacked[:, 0].max())
        ypad = max(0.08 * (ymax - ymin), 0.02)
        if xmin > 0.0:
            log_min, log_max = np.log10((xmin, xmax))
            log_pad = max(0.05 * (log_max - log_min), 0.03)
            xlimits = (10 ** (log_min - log_pad), 10 ** (log_max + log_pad))
        else:
            xpad = max(0.05 * (xmax - xmin), 1e-9)
            xlimits = (xmin - xpad, xmax + xpad)
        for row in range(len(METHODS)):
            for column in (actual_column, estimated_column):
                axis = axes[row, column]
                axis.set_xlim(*xlimits)
                axis.set_ylim(ymin - ypad, ymax + ypad)
                if column != 0:
                    axis.tick_params(labelleft=False)
                if row < len(METHODS) - 1:
                    axis.tick_params(labelbottom=False)

    for dataset_index, (_, dataset_label) in enumerate(DATASETS):
        for column, estimate_label in (
            (2 * dataset_index, "Actual"),
            (2 * dataset_index + 1, "Estimated"),
        ):
            axis = axes[0, column]
            axis.set_title(dataset_label, fontsize=26, pad=41)
            axis.text(
                0.5, 1.005, estimate_label,
                transform=axis.transAxes,
                ha="center", va="bottom", fontsize=25,
            )
    figure.subplots_adjust(
        left=0.125, right=0.965, bottom=0.095, top=0.90,
        wspace=0.09, hspace=0.16,
    )
    figure.supxlabel("Mean deployment cost (USD, log scale)", fontsize=27, y=0.045)
    figure.supylabel("Mean accuracy", fontsize=27, x=0.018)
    handles = (
        Line2D([], [], linestyle="none", marker="o", markersize=11,
               markerfacecolor=BACKGROUND_COLOR, markeredgecolor="none",
               label="All configurations"),
        Line2D([], [], color=FRONTIER_COLOR, linewidth=1.7, marker="o",
               markersize=10, markerfacecolor="white", markeredgewidth=1.5,
               label="Full-data Pareto frontier"),
        Line2D([], [], linestyle="none", marker="o", markersize=13,
               markerfacecolor=RECOMMENDATION_COLOR,
               markeredgecolor=RECOMMENDATION_EDGE,
               label="Recommended configurations"),
    )
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
        ncol=3, frameon=False, fontsize=24,
        columnspacing=1.5, handletextpad=0.7,
    )
    figure.suptitle(
        f"Seed {SEED}: estimations and actual values of Pareto recommendations "
        f"at the {target:.0%} search-cost checkpoint",
        fontsize=30, y=0.985,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "plot_cache.pkl"
    if args.reuse_cache:
        cache = read_cache(cache_path)
    else:
        cache = build_cache()
        write_cache(cache_path, cache)
    plot_rows = []
    for target in TARGETS:
        plot_rows.extend(render_target(
            cache,
            target,
            output / f"estimated_actual_pareto_{round(target * 100)}pct_seed{SEED}.png",
        ))
    with (output / "estimated_actual_plot_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    (output / "plot_manifest.json").write_text(
        json.dumps({
            "method": FULL_METHOD_NAME,
            "seed": SEED,
            "targets": list(TARGETS),
            "datasets": [dataset for dataset, _ in DATASETS],
            "configuration_grids": {
                "hotpotqa": "10x10 (100 configurations)",
            },
            "baseline_checkpoint_payloads_unchanged": True,
            "output_format": "PNG only",
            "cost_axis": "log scale",
            "plot_cache": "plot_cache.pkl",
            "plot_ready_data": "estimated_actual_plot_data.csv",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
