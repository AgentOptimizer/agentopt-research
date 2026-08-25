#!/usr/bin/env python3
"""Reproduce per-configuration accuracy and deployment-cost histograms."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


DATASETS = (
    ("HotpotQA", ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl"),
    ("MathQA", ROOT / "experiments/data/lookup/mathqa_lookup.pkl"),
)

BLUE = "#4C86B6"
RED = "#E6322B"
GREEN = "#17A878"


def configuration_means(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    models, _, table = load_pickle(path)
    # Match the original diagnostic: each arm is averaged over every result
    # available for that arm, rather than restricting all arms to the common
    # question intersection.
    accuracy = np.asarray([
        np.mean([result.score for result in table[model].values()])
        for model in models
    ])
    cost = np.asarray([
        np.mean([result.cost for result in table[model].values()])
        for model in models
    ])
    return list(models), accuracy, cost


def common_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 24,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
    })


def finish(fig: plt.Figure, output: Path, handles: list[Line2D]) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        handlelength=2.2,
        columnspacing=2.5,
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


def plot_accuracy(records: list[tuple[str, list[str], np.ndarray, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.25, 5.47))
    for ax, (title, models, accuracy, _) in zip(axes, records):
        best = int(np.argmax(accuracy))
        ax.hist(
            accuracy,
            bins=np.linspace(0.0, 1.0, 21),
            color=BLUE,
            alpha=0.78,
            rwidth=0.94,
        )
        ax.axvline(float(np.mean(accuracy)), color=RED, linewidth=3)
        ax.axvline(float(accuracy[best]), color=GREEN, linewidth=3, linestyle=":")
        ax.set(xlim=(0, 1), xlabel="Per-arm mean accuracy",
               ylabel="Number of configurations", title=title)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.18)
        ax.text(0.02, 0.96, f"{len(models)} arms: best = c{best}",
                transform=ax.transAxes, va="top", fontsize=13, color="#444444")

    fig.suptitle("Distribution of Per-Configuration Mean Accuracy",
                 fontsize=27, fontweight="bold", y=1.01)
    finish(fig, output, [
        Line2D([], [], color=RED, linewidth=3, label="Empirical mean"),
        Line2D([], [], color=GREEN, linewidth=3, linestyle=":", label="Best arm"),
    ])


def plot_cost(records: list[tuple[str, list[str], np.ndarray, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.25, 5.47))
    for ax, (title, models, _, cost) in zip(axes, records):
        best = int(np.argmin(cost))
        upper = math.ceil(float(np.max(cost)) * 1000.0) / 1000.0
        ax.hist(
            cost,
            bins=np.linspace(0.0, upper, 21),
            color=BLUE,
            alpha=0.78,
            rwidth=0.94,
        )
        ax.axvline(float(np.mean(cost)), color=RED, linewidth=3)
        ax.axvline(float(cost[best]), color=GREEN, linewidth=3, linestyle=":")
        ax.set(xlim=(0, upper), xlabel="Per-arm mean deployment cost (USD)",
               ylabel="Number of configurations", title=title)
        ax.set_xticks(np.linspace(0, upper, 5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("$%.3f"))
        ax.grid(axis="y", alpha=0.18)
        ax.text(0.02, 0.96, f"{len(models)} arms: lowest-cost = c{best}",
                transform=ax.transAxes, va="top", fontsize=13, color="#444444")

    fig.suptitle("Distribution of Per-Configuration Mean Deployment Cost",
                 fontsize=27, fontweight="bold", y=1.01)
    finish(fig, output, [
        Line2D([], [], color=RED, linewidth=3, label="Empirical mean"),
        Line2D([], [], color=GREEN, linewidth=3, linestyle=":", label="Lowest-cost arm"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "analysis/vs")
    args = parser.parse_args()
    common_style()
    records = []
    for title, path in DATASETS:
        models, accuracy, cost = configuration_means(path)
        records.append((title, models, accuracy, cost))
    plot_accuracy(records, args.outdir / "configuration_mean_accuracy_distribution.png")
    plot_cost(records, args.outdir / "configuration_mean_cost_distribution.png")


if __name__ == "__main__":
    main()
