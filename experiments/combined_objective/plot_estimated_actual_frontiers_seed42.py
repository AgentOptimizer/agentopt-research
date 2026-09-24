#!/usr/bin/env python3
"""Plot Seed-42 estimated/actual Pareto frontiers at 10% and 30% spend."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "analysis/estimated_actual_frontiers_seed42"
DATASETS = (("mathqa", "MathQA"), ("stackoverflow", "Stack Overflow"))
METHODS = (
    ("radial_gittins", "Radial Gittins"),
    ("ege_sh", "EGE-SH"),
    ("ape_k", "APE-k"),
    ("qnehvi", "qNEHVI"),
    ("random_questions", "Random questions"),
    ("random_configurations", "Random configurations"),
)
SEED = 42
RECOMMENDATION_COLOR = "#d55e00"
RECOMMENDATION_EDGE = "#725b46"
BACKGROUND_COLOR = "#d8dce2"
FRONTIER_COLOR = "#3f4854"


def _load(results: Path, dataset: str, method: str) -> dict[str, Any]:
    path = (
        results / "runs" / dataset / method / f"seed-{SEED}"
        / "frontier_events.json.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported or missing schema in {path}")
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
            color=RECOMMENDATION_COLOR,
            edgecolors=RECOMMENDATION_EDGE, linewidths=1.2, zorder=5,
        )
    else:
        axis.text(
            0.5, 0.5, "No recommendation", transform=axis.transAxes,
            ha="center", va="center", fontsize=14, color="#56636c",
        )
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.grid(color="#d9dde3", linewidth=0.55, alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=19)


def render_target(results: Path, target_percent: int) -> tuple[Path, Path]:
    tag = f"{target_percent}pct"
    payloads = {
        (dataset, method): _load(results, dataset, method)
        for dataset, _ in DATASETS
        for method, _ in METHODS
    }
    # Four paper-style columns per row: actual then estimated for each dataset.
    figure, axes = plt.subplots(len(METHODS), 4, figsize=(18.0, 20.0))
    summary_rows: list[dict[str, Any]] = []

    for dataset_index, (dataset, dataset_label) in enumerate(DATASETS):
        actual_column = 2 * dataset_index
        estimated_column = actual_column + 1
        truth = np.asarray(
            payloads[(dataset, METHODS[0][0])]["truth_raw_vectors"], dtype=float
        )
        scale_vectors = [truth]

        for row, (method, method_label) in enumerate(METHODS):
            payload = payloads[(dataset, method)]
            checkpoint = payload["cost_checkpoints"][tag]
            local_truth = np.asarray(payload["truth_raw_vectors"], dtype=float)
            if not np.allclose(local_truth, truth, rtol=0, atol=1e-12):
                raise ValueError(f"truth mismatch for {dataset}/{method}")
            frontier = truth[
                np.asarray(payload["full_data_pareto_arm_indices"], dtype=int)
            ]
            frontier = frontier[np.argsort(frontier[:, 1])]
            selected = [int(arm) for arm in checkpoint["selected_arm_indices"]]
            estimated = np.asarray(
                checkpoint["estimated_raw_archive_vectors"], dtype=float
            ).reshape((-1, 2))
            actual = np.asarray(
                checkpoint["actual_raw_archive_vectors"], dtype=float
            ).reshape((-1, 2))
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
                axes[row, 0].set_ylabel(method_label, fontsize=18, labelpad=14)
            estimated_percent = float(
                checkpoint["estimated_search_cost_percent"]
            )
            actual_percent = float(checkpoint["actual_search_cost_percent"])
            summary_rows.append(
                {
                    "target_actual_search_cost_percent": target_percent,
                    "dataset": dataset,
                    "method": method,
                    "seed": SEED,
                    "cost_estimator": payload["cost_estimator"],
                    "selected_count": len(selected),
                    "cumulative_evaluations": checkpoint[
                        "cumulative_evaluations"
                    ],
                    "estimated_search_cost_percent": estimated_percent,
                    "actual_search_cost_percent": actual_percent,
                    "estimated_minus_actual_percentage_points": (
                        estimated_percent - actual_percent
                    ),
                    "frontier_change_frames": len(
                        payload["frontier_change_frames"]
                    ),
                }
            )

        stacked = np.vstack([values for values in scale_vectors if len(values)])
        xmin, xmax = float(stacked[:, 1].min()), float(stacked[:, 1].max())
        ymin, ymax = float(stacked[:, 0].min()), float(stacked[:, 0].max())
        xpad = max(0.05 * (xmax - xmin), 1e-9)
        ypad = max(0.08 * (ymax - ymin), 0.02)
        for row in range(len(METHODS)):
            for column in (actual_column, estimated_column):
                axis = axes[row, column]
                axis.set_xlim(xmin - xpad, xmax + xpad)
                axis.set_ylim(ymin - ypad, ymax + ypad)
                if column != 0:
                    axis.tick_params(labelleft=False)
                if row < len(METHODS) - 1:
                    axis.tick_params(labelbottom=False)

    axes[0, 0].set_title("MathQA — Actual", fontsize=22, pad=12)
    axes[0, 1].set_title("MathQA — Estimated", fontsize=22, pad=12)
    axes[0, 2].set_title("Stack Overflow — Actual", fontsize=22, pad=12)
    axes[0, 3].set_title("Stack Overflow — Estimated", fontsize=22, pad=12)
    figure.subplots_adjust(
        left=0.125, right=0.965, bottom=0.095, top=0.90,
        wspace=0.09, hspace=0.16,
    )
    figure.supxlabel("Mean deployment cost (USD)", fontsize=23, y=0.045)
    figure.supylabel("Mean accuracy", fontsize=23, x=0.018)

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
        ncol=3, frameon=False, fontsize=20,
        columnspacing=1.5, handletextpad=0.7,
    )
    figure.suptitle(
        f"Seed {SEED}: estimations and actual values of Pareto recommendations "
        f"at the {target_percent}% search-cost checkpoint",
        fontsize=26, y=0.975,
    )

    output_dir = results / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"estimated_actual_pareto_{tag}_seed{SEED}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    # Keep a fixed paper canvas, matching plot_paper_20seed_comparison.py.
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"WROTE {png}\nWROTE {pdf}\nWROTE {csv_path}", flush=True)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--targets", type=int, nargs="+", choices=(10, 30), default=(10, 30),
    )
    args = parser.parse_args()
    for target in args.targets:
        render_target(args.results, target)
    artifacts = []
    for dataset, _ in DATASETS:
        for method, _ in METHODS:
            payload = _load(args.results, dataset, method)
            artifacts.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": SEED,
                    "path": str(
                        Path("runs") / dataset / method / f"seed-{SEED}"
                        / "frontier_events.json.gz"
                    ),
                    "frontier_change_frames": len(
                        payload["frontier_change_frames"]
                    ),
                    "cost_estimator": payload["cost_estimator"],
                }
            )
    manifest = {
        "schema_version": 1,
        "purpose": "self-contained frontier membership frames for later animation",
        "seed": SEED,
        "artifacts": artifacts,
    }
    manifest_path = args.results / "animation_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
