#!/usr/bin/env python3
"""Plot estimated and actual Pareto recommendations at 10% actual search spend.

The two panels use the same selected configurations. Estimated coordinates are
their observed sample means at the checkpoint; actual coordinates are their
means over the complete common-question matrix. The gray landscape and dotted
frontier are full-data references in both panels, not selector information.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.plot.plot_lcb_recommendations import (  # noqa: E402
    _annotate_selected,
    config_id,
)


from experiments.combined_objective.run_two_direction_ablation import (  # noqa: E402
    DEFAULT_OUTDIR as DEFAULT_RESULTS,
    PRIMARY_PAIR as DEFAULT_PAIR,
)


DEFAULT_BENCHMARKS = (
    "hotpotqa",
    "mathqa",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
)
BENCHMARK_LABELS = {
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD dev",
    "restaurant_valid": "Restaurant valid",
}
ESTIMATED_COLOR = "#286f9b"
ACTUAL_COLOR = "#b35c32"
REFERENCE_COLOR = "#b8c1c9"


def _vectors(value: object, *, name: str, count: int | None = None) -> np.ndarray:
    vectors = np.asarray(value, dtype=float)
    if vectors.size == 0:
        vectors = vectors.reshape((0, 2))
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"{name} must be an N-by-2 array")
    if count is not None and len(vectors) != count:
        raise ValueError(f"{name} has {len(vectors)} rows; expected {count}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} contains a non-finite value")
    return vectors


def _checkpoint_data(payload: dict) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        checkpoint = payload["search_cost_checkpoint_10pct"]
        run = payload["run"]
    except KeyError as error:
        raise ValueError(f"missing saved 10% checkpoint or run: {error}") from error
    if checkpoint is None:
        raise ValueError("the run did not reach a 10% actual search-cost checkpoint")

    truth = _vectors(run["raw_truth_vectors"], name="raw_truth_vectors")
    selected = np.asarray(checkpoint["selected_arm_indices"], dtype=int)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(truth)):
        raise ValueError("selected_arm_indices must index raw_truth_vectors")
    if len(set(selected.tolist())) != len(selected):
        raise ValueError("selected_arm_indices contains duplicates")
    estimated = _vectors(
        checkpoint["estimated_raw_archive_vectors"],
        name="estimated_raw_archive_vectors",
        count=len(selected),
    )
    actual = _vectors(
        checkpoint["offline_raw_selected_vectors"],
        name="offline_raw_selected_vectors",
        count=len(selected),
    )
    if not np.allclose(actual, truth[selected], rtol=0, atol=1e-10):
        raise ValueError("offline_raw_selected_vectors are not aligned with selected_arm_indices")
    counts = np.asarray(checkpoint["selected_sample_counts"], dtype=int)
    if counts.shape != selected.shape or np.any(counts < 0):
        raise ValueError("selected_sample_counts must align with selected_arm_indices")
    frontier = np.asarray(run["full_data_pareto_arm_indices"], dtype=int)
    if frontier.ndim != 1 or np.any(frontier < 0) or np.any(frontier >= len(truth)):
        raise ValueError("full_data_pareto_arm_indices must index raw_truth_vectors")
    for field in ("estimated_search_cost_percent", "actual_search_cost_percent"):
        value = float(checkpoint[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    return run, checkpoint, truth, estimated, actual, frontier


def render_one(result_path: Path) -> tuple[Path, Path]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    run, checkpoint, truth, estimated, actual, frontier = _checkpoint_data(payload)
    selected = [int(arm) for arm in checkpoint["selected_arm_indices"]]
    counts = [int(count) for count in checkpoint["selected_sample_counts"]]
    names = run.get("model_names", [])
    labels = [config_id(names[arm], arm) if arm < len(names) else str(arm) for arm in selected]
    frontier_points = truth[frontier]
    frontier_points = frontier_points[np.argsort(frontier_points[:, 1])]
    all_points = np.vstack((truth, estimated, actual))
    use_log_cost = bool(np.all(all_points[:, 1] > 0))

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), sharex=True, sharey=True)
    benchmark = payload.get("config", {}).get("benchmark", result_path.parent.name)
    benchmark_label = BENCHMARK_LABELS.get(benchmark, str(benchmark))
    seed = payload.get("config", {}).get("seed", run.get("seed", "?"))
    panels = (
        (axes[0], estimated, ESTIMATED_COLOR, "Estimated", "estimated_search_cost_percent"),
        (axes[1], actual, ACTUAL_COLOR, "Actual", "actual_search_cost_percent"),
    )
    for axis, vectors, color, label, percent_field in panels:
        axis.scatter(truth[:, 1], truth[:, 0], s=25, color=REFERENCE_COLOR,
                     alpha=0.78, zorder=1)
        axis.plot(frontier_points[:, 1], frontier_points[:, 0], ":",
                  color="#66727b", linewidth=1.4, zorder=2)
        if selected:
            axis.scatter(vectors[:, 1], vectors[:, 0], s=78, marker="o",
                         facecolors=color, edgecolors="white", linewidths=1.1,
                         zorder=4)
        else:
            axis.text(0.5, 0.5, "No recommendation", transform=axis.transAxes,
                      ha="center", va="center", fontsize=11)
        if use_log_cost:
            axis.set_xscale("log")
        axis.set_title(
            f"{label} recommendation coordinates\n"
            f"{label} search cost: {float(checkpoint[percent_field]):.2f}% of full search",
            fontsize=13, linespacing=1.4,
        )
        axis.set_xlabel("Mean deployment cost (USD / question)", fontsize=11)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.17)
        axis.spines[["top", "right"]].set_visible(False)

    if use_log_cost:
        xmin, xmax = float(all_points[:, 1].min()), float(all_points[:, 1].max())
        axes[0].set_xlim(xmin / 1.28, xmax * 1.28)
    else:
        xmin, xmax = float(all_points[:, 1].min()), float(all_points[:, 1].max())
        padding = max((xmax - xmin) * 0.09, 1e-8)
        axes[0].set_xlim(xmin - padding, xmax + padding)
    ymin, ymax = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    ypadding = max((ymax - ymin) * 0.11, 0.025)
    axes[0].set_ylim(ymin - ypadding, ymax + ypadding)
    axes[0].set_ylabel("Mean accuracy", fontsize=11)

    # The same labels in both panels make the paired membership explicit.
    figure.canvas.draw()
    for axis, vectors, *_ in panels:
        if selected:
            _annotate_selected(axis, vectors, labels, counts)

    handles = (
        Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor=ESTIMATED_COLOR, markeredgecolor="white",
               label="Recommended · estimated observed mean"),
        Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor=ACTUAL_COLOR, markeredgecolor="white",
               label="Same recommendations · actual full-data mean"),
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor=REFERENCE_COLOR, markeredgecolor="none",
               label="All configurations · full-data reference"),
        Line2D([], [], linestyle=":", color="#66727b", linewidth=1.4,
               label="Full-data Pareto frontier"),
    )
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085),
                  ncol=2, fontsize=9, frameon=False)
    figure.suptitle(
        f"{benchmark_label} · seed {seed} · first available state at or above 10% actual search cost",
        fontsize=15, y=0.975,
    )
    figure.text(
        0.5, 0.025,
        "Both percentages use the same full-matrix actual search cost as denominator. "
        "Gray points and dotted frontier are offline references; recommended points are not connected.",
        ha="center", va="bottom", fontsize=8.7, color="#47545e",
    )
    figure.tight_layout(rect=(0.03, 0.18, 0.99, 0.91), w_pad=2.5)

    output = result_path.parent / "pareto_estimated_vs_actual_10pct"
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=220, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    print(f"wrote {png} and {pdf}", flush=True)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--benchmarks", nargs="+", choices=DEFAULT_BENCHMARKS,
                        default=list(DEFAULT_BENCHMARKS))
    args = parser.parse_args()
    for benchmark in args.benchmarks:
        render_one(args.results / args.pair / benchmark / "result.json")


if __name__ == "__main__":
    main()
