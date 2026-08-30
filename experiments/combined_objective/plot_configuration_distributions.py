#!/usr/bin/env python3
"""Plot per-configuration accuracy and deployment-cost distributions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
)
from experiments.combined_objective.plot_paper_20seed_comparison import (  # noqa: E402
    PICKLES,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


BENCHMARKS = (("hotpotqa", "HotpotQA"), ("mathqa", "MathQA"))
HISTOGRAM_COLOR = "#6699c2"
MEAN_COLOR = "#ef332d"
EXTREME_COLOR = "#0aae78"


def _load_values(benchmark: str) -> np.ndarray:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    question_ids = common_question_ids(models, datapoints, table)
    return mean_raw_vectors(models, question_ids, table)


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _base_figure(title: str) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    figure.suptitle(title, fontsize=25, fontweight="bold", y=.965)
    figure.subplots_adjust(left=.065, right=.970, top=.735, bottom=.245, wspace=.18)
    return figure, np.asarray(axes)


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=22, pad=5)
    axis.set_ylabel("Number of configurations", fontsize=17)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", color="#d9dde3", linewidth=.6, alpha=.5)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(output)
    print(output.with_suffix(".pdf"))


def plot_accuracy(output: Path) -> None:
    figure, axes = _base_figure(
        "Distribution of Per-Configuration Mean Accuracy",
    )
    for axis, (benchmark, title) in zip(axes, BENCHMARKS):
        accuracy = _load_values(benchmark)[:, 0]
        best = int(np.argmax(accuracy))
        axis.hist(
            accuracy, bins=20, range=(0, 1), color=HISTOGRAM_COLOR,
            alpha=.9, edgecolor="white", linewidth=.8,
        )
        axis.axvline(float(np.mean(accuracy)), color=MEAN_COLOR, linewidth=2.5)
        axis.axvline(float(accuracy[best]), color=EXTREME_COLOR,
                     linewidth=2.5, linestyle=":")
        axis.text(
            .02, .96, f"{len(accuracy)} arms: best = c{best}",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=14, color="#4b4b4b",
        )
        axis.set_xlim(0, 1)
        axis.set_xticks(np.linspace(0, 1, 5))
        axis.yaxis.set_major_locator(MultipleLocator(5))
        axis.set_xlabel("Per-arm mean accuracy", fontsize=17)
        _style_axis(axis, title)
    handles = (
        Line2D([], [], color=MEAN_COLOR, linewidth=2.5, label="Empirical mean"),
        Line2D([], [], color=EXTREME_COLOR, linewidth=2.5, linestyle=":",
               label="Best arm"),
    )
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                  fontsize=16, bbox_to_anchor=(.5, .01), columnspacing=2.6)
    _save(figure, output)


def plot_cost(output: Path) -> None:
    figure, axes = _base_figure(
        "Distribution of Per-Configuration Mean Deployment Cost",
    )
    limits = {"hotpotqa": .014, "mathqa": .040}
    for axis, (benchmark, title) in zip(axes, BENCHMARKS):
        cost = _load_values(benchmark)[:, 1]
        lowest = int(np.argmin(cost))
        limit = limits[benchmark]
        axis.hist(
            cost, bins=20, range=(0, limit), color=HISTOGRAM_COLOR,
            alpha=.9, edgecolor="white", linewidth=.8,
        )
        axis.axvline(float(np.mean(cost)), color=MEAN_COLOR, linewidth=2.5)
        axis.axvline(float(cost[lowest]), color=EXTREME_COLOR,
                     linewidth=2.5, linestyle=":")
        axis.text(
            .02, .96, f"{len(cost)} arms: lowest-cost = c{lowest}",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=14, color="#4b4b4b",
        )
        axis.set_xlim(0, limit)
        axis.set_xticks(np.linspace(0, limit, 5))
        axis.yaxis.set_major_locator(MultipleLocator(10))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.3f}"))
        axis.set_xlabel("Per-arm mean deployment cost (USD)", fontsize=17)
        _style_axis(axis, title)
    handles = (
        Line2D([], [], color=MEAN_COLOR, linewidth=2.5, label="Empirical mean"),
        Line2D([], [], color=EXTREME_COLOR, linewidth=2.5, linestyle=":",
               label="Lowest-cost arm"),
    )
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                  fontsize=16, bbox_to_anchor=(.5, .01), columnspacing=2.6)
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis/paper_20seed_method_comparison/figures",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    _style()
    plot_accuracy(output_dir / "configuration_mean_accuracy_distribution.png")
    plot_cost(output_dir / "configuration_mean_cost_distribution.png")


if __name__ == "__main__":
    main()
