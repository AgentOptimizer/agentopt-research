#!/usr/bin/env python3
"""Plot two BF-target frontiers and recommendation membership intervals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from experiments.combined_objective.plot.plot_lcb_recommendations import config_id


COLORS = {
    "both": "#7952A8",
    "10% only": "#3973AC",
    "30% only": "#D9822B",
}
GRAY = "#B5C0CA"
FRONTIER_GRAY = "#687783"


def state_at(points: list[dict], target: float) -> tuple[dict, float]:
    eligible = [point for point in points if float(point["cost_fraction"]) <= target]
    if not eligible:
        raise ValueError(
            f"Target {target:.1%} precedes the first saved state "
            f"at {float(points[0]['cost_fraction']):.1%}."
        )
    point = eligible[-1]
    return point, float(point["cost_fraction"])


def membership_intervals(points: list[dict], terminal: float) -> dict[int, list[tuple[float, float]]]:
    intervals: dict[int, list[tuple[float, float]]] = {}
    active: dict[int, float] = {}
    previous: set[int] = set()
    for point in points:
        cost = float(point["cost_fraction"])
        current = set(map(int, point["selected_arm_indices"]))
        for arm in previous - current:
            intervals.setdefault(arm, []).append((active.pop(arm), cost))
        for arm in current - previous:
            active[arm] = cost
        previous = current
    for arm, start in active.items():
        intervals.setdefault(arm, []).append((start, terminal))
    return intervals


def _annotate_ids(axis, vectors: np.ndarray, labels: list[str]) -> None:
    anchors = axis.transAxes.inverted().transform(axis.transData.transform(vectors[:, [1, 0]]))
    placed: list[tuple[float, float, float, float]] = []
    offsets = (0.04, -0.04, 0.10, -0.10, 0.16, -0.16, 0.22, -0.22, 0.28, -0.28)
    for index in np.lexsort((anchors[:, 1], anchors[:, 0])):
        vector, anchor = vectors[index], anchors[index]
        label = labels[index]
        width, height = min(0.24, max(0.065, len(label) * 0.014)), 0.045
        candidates = []
        for preferred_x in (anchor[0] + 0.018, anchor[0] - width - 0.018):
            x = float(np.clip(preferred_x, 0.012, 0.988 - width))
            for offset in offsets:
                y = float(np.clip(anchor[1] + offset, 0.04, 0.91))
                rectangle = (x, y - height / 2, x + width, y + height / 2)
                overlap = sum(
                    max(0, min(rectangle[2], other[2]) - max(rectangle[0], other[0]))
                    * max(0, min(rectangle[3], other[3]) - max(rectangle[1], other[1]))
                    for other in placed
                )
                distance = (y - anchor[1]) ** 2 + 0.35 * (x + width / 2 - anchor[0]) ** 2
                candidates.append((1000 * overlap + distance, rectangle))
        _, rectangle = min(candidates, key=lambda item: item[0])
        placed.append(rectangle)
        axis.annotate(
            label,
            (vector[1], vector[0]),
            xytext=(rectangle[0], (rectangle[1] + rectangle[3]) / 2),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#27323B",
            zorder=6,
            arrowprops={
                "arrowstyle": "-",
                "color": "#77838B",
                "linewidth": 0.6,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.82},
        )


def draw_frontier(
    axis,
    *,
    run: dict,
    selected: list[int],
    labels: dict[int, str],
    target: float,
    source_cost: float,
) -> None:
    truth = np.asarray(run["raw_truth_vectors"], dtype=float)
    frontier = truth[np.asarray(run["full_data_pareto_arm_indices"], dtype=int)]
    frontier = frontier[np.argsort(frontier[:, 1])]
    axis.scatter(truth[:, 1], truth[:, 0], s=22, color=GRAY, alpha=0.58, zorder=1)
    axis.plot(frontier[:, 1], frontier[:, 0], ":", color=FRONTIER_GRAY, linewidth=1.5, zorder=2)
    vectors = truth[np.asarray(selected, dtype=int)]
    axis.scatter(
        vectors[:, 1],
        vectors[:, 0],
        s=72,
        marker="o",
        facecolor=COLORS["both"],
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    _annotate_ids(axis, vectors, [labels[arm] for arm in selected])
    if np.all(truth[:, 1] > 0):
        axis.set_xscale("log")
    axis.margins(x=0.14)
    axis.set_ylim(
        max(-0.025, float(truth[:, 0].min()) - 0.025),
        min(1.05, float(truth[:, 0].max()) + 0.105),
    )
    persistence = (
        f"membership unchanged since {source_cost:.2%} BF"
        if source_cost < target - 1e-10
        else "membership changed at this BF"
    )
    axis.set_title(
        f"Recommendation at {target:.0%} BF\n"
        f"{len(selected)} configurations; {persistence}",
        fontsize=11,
    )
    axis.set_xlabel("Full-data mean cost (USD / question)")
    axis.set_ylabel("Full-data accuracy")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def render(result_path: Path, output_path: Path, targets: tuple[float, float]) -> None:
    payload = json.loads(result_path.read_text())
    run = payload["run"]
    points = run["points"]
    terminal = float(run["cost_fraction"])
    if any(target > terminal + 1e-12 for target in targets):
        raise ValueError(f"Target exceeds terminal BF {terminal:.3%}.")

    states = [state_at(points, target) for target in targets]
    selected_sets = [set(map(int, point["selected_arm_indices"])) for point, _ in states]
    selected_union = selected_sets[0] | selected_sets[1]
    intervals = membership_intervals(points, terminal)
    labels = {
        arm: config_id(run["model_names"][arm], arm)
        for arm in selected_union
    }

    benchmark = str(payload["config"]["benchmark"])
    benchmark_label = {
        "bird_dev": "BIRD Dev",
        "hotpotqa": "HotpotQA",
        "mathqa": "MathQA",
        "stackoverflow": "Stack Overflow",
    }.get(benchmark, benchmark.replace("_", " ").title())

    def category(arm: int) -> str:
        in_10 = arm in selected_sets[0]
        in_30 = arm in selected_sets[1]
        return "both" if in_10 and in_30 else "10% only" if in_10 else "30% only"

    relevant_intervals = {
        arm: [
            (start, end)
            for start, end in intervals[arm]
            if any(
                arm in selected_set and start <= target < end
                for target, selected_set in zip(targets, selected_sets)
            )
        ]
        for arm in selected_union
    }
    truth = np.asarray(run["raw_truth_vectors"], dtype=float)
    ordered = sorted(
        selected_union,
        key=lambda arm: (
            truth[arm, 1],
            truth[arm, 0],
            labels[arm],
        ),
    )

    figure, (frontier_axis, timeline_axis) = plt.subplots(
        1,
        2,
        figsize=(16.2, 7.8),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": 0.24},
    )

    frontier = truth[np.asarray(run["full_data_pareto_arm_indices"], dtype=int)]
    frontier = frontier[np.argsort(frontier[:, 1])]
    frontier_axis.scatter(truth[:, 1], truth[:, 0], s=16, color=GRAY, alpha=0.38, zorder=1)
    frontier_axis.plot(frontier[:, 1], frontier[:, 0], ":", color=FRONTIER_GRAY, linewidth=1.6, zorder=2)
    marker_by_category = {"both": "o", "10% only": "s", "30% only": "D"}
    for name in ("both", "10% only", "30% only"):
        arms = [arm for arm in selected_union if category(arm) == name]
        if not arms:
            continue
        vectors = truth[np.asarray(arms, dtype=int)]
        frontier_axis.scatter(
            vectors[:, 1],
            vectors[:, 0],
            s=82,
            marker=marker_by_category[name],
            facecolor=COLORS[name],
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
    union_sorted = sorted(selected_union, key=lambda arm: (truth[arm, 1], truth[arm, 0]))
    _annotate_ids(
        frontier_axis,
        truth[np.asarray(union_sorted, dtype=int)],
        [labels[arm] for arm in union_sorted],
    )
    if np.all(truth[:, 1] > 0):
        frontier_axis.set_xscale("log")
    frontier_axis.margins(x=0.14)
    frontier_axis.set_ylim(
        max(-0.025, float(truth[:, 0].min()) - 0.025),
        min(1.05, float(truth[:, 0].max()) + 0.105),
    )
    frontier_axis.set_xlabel("Full-data mean cost (USD / question)")
    frontier_axis.set_ylabel("Full-data accuracy")
    frontier_axis.yaxis.set_major_formatter(PercentFormatter(1))
    frontier_axis.grid(alpha=0.18)
    frontier_axis.spines[["top", "right"]].set_visible(False)
    frontier_axis.set_title("Which configurations are recommended?", fontsize=12)
    shared_count = len(selected_sets[0] & selected_sets[1])
    frontier_axis.text(
        0.98,
        0.035,
        f"10% BF: {len(selected_sets[0])}   |   30% BF: {len(selected_sets[1])}   |   shared: {shared_count}",
        transform=frontier_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#52606B",
    )

    for lane, arm in enumerate(ordered):
        name = category(arm)
        for start, end in relevant_intervals[arm]:
            start_pct, end_pct = 100 * start, 100 * end
            timeline_axis.plot(
                [start_pct, end_pct],
                [lane, lane],
                color=COLORS[name],
                linewidth=9,
                solid_capstyle="butt",
                zorder=3,
            )
            interval_label = (
                f"{start_pct:.1f}% → terminal {end_pct:.1f}%"
                if math.isclose(end, terminal, rel_tol=0, abs_tol=1e-10)
                else f"{start_pct:.1f}–{end_pct:.1f}%"
            )
            width = end_pct - start_pct
            if width >= 24:
                timeline_axis.text(
                    (start_pct + end_pct) / 2,
                    lane,
                    interval_label,
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="white",
                    zorder=4,
                )
            else:
                timeline_axis.text(
                    end_pct + 0.8,
                    lane,
                    interval_label,
                    ha="left",
                    va="center",
                    fontsize=7.6,
                    color="#27323B",
                    zorder=4,
                )

    for target, color in zip(targets, (COLORS["10% only"], COLORS["30% only"])):
        target_pct = 100 * target
        timeline_axis.axvline(
            target_pct,
            color=color,
            linestyle="--",
            linewidth=1.7,
            alpha=0.95,
            zorder=2,
        )
        timeline_axis.text(
            target_pct,
            0.985,
            f"{target:.0%} BF",
            transform=timeline_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=color,
        )
    timeline_axis.axvline(100 * terminal, color="#7A8791", linestyle=":", linewidth=1.2, zorder=1)
    timeline_axis.text(
        100 * terminal,
        0.985,
        f"terminal {terminal:.1%}",
        transform=timeline_axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8.5,
        color="#52606B",
    )
    timeline_axis.set_yticks(range(len(ordered)), [labels[arm] for arm in ordered])
    timeline_axis.invert_yaxis()
    timeline_axis.set_xlim(0, max(100.0, 100 * terminal + 4.0))
    timeline_axis.set_xlabel("Search cost / brute-force cost (BF %)")
    timeline_axis.set_ylabel("Configuration ID")
    timeline_axis.set_title(
        "How long does each recommendation persist?\n"
        "Top to bottom follows the frontier from left to right",
        fontsize=12,
    )
    timeline_axis.grid(axis="x", alpha=0.20)
    timeline_axis.spines[["top", "right"]].set_visible(False)

    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["both"],
               markeredgecolor="white", markersize=8, label="Recommended at both 10% and 30%"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor=COLORS["10% only"],
               markeredgecolor="white", markersize=8, label="Recommended at 10% only"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=COLORS["30% only"],
               markeredgecolor="white", markersize=8, label="Recommended at 30% only"),
        Line2D([], [], marker="o", linestyle="none", color=GRAY, markersize=6,
               label="All configurations"),
        Line2D([], [], linestyle=":", color=FRONTIER_GRAY, label="Full-data Pareto frontier"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    directions = " + ".join(
        f"({direction[0]:g}, {direction[1]:g})"
        for direction in payload["config"]["directions"]
    )
    figure.suptitle(
        f"Gauss-Radau Gittins — {benchmark_label}: recommendations at 10% vs 30% BF\n"
        f"directions {directions}; finite-LCB beta=1; seed {payload['config']['seed']}",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.083,
        "Only membership intervals containing the 10% or 30% target are shown. "
        "Frontier coordinates are offline full-data diagnostics.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#52606B",
    )
    figure.subplots_adjust(top=0.84, bottom=0.18, left=0.07, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"Rendered {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--targets", nargs=2, type=float, default=(0.10, 0.30))
    args = parser.parse_args()
    render(args.result, args.output, tuple(args.targets))


if __name__ == "__main__":
    main()
