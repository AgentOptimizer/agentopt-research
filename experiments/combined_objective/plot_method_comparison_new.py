#!/usr/bin/env python3
"""Redraw the method comparison without the pre-stop completed-arm artefact.

This is deliberately a post-processing script: it consumes the frozen method
summary and radial-Gittins stop summary, so producing the paper figures does
not rerun either selector.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("HotpotQA", "MathQA")
METHODS = {
    "radial_gittins_deployable": ("#1f4e79", "Radial-Gittins completed-only"),
    "radial_gittins_provisional": ("#1f4e79", "Radial-Gittins provisional"),
    "random_configurations": ("#7b61a8", "Random configurations"),
    "random_questions": ("#2a9d8f", "Random shared questions"),
}


def load_series(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["benchmark"], row["method"])].append(
                (
                    float(row["budget_fraction"]),
                    float(row["mean_hv_regret"]),
                    float(row["ci95_half_width"]),
                )
            )
    result = {}
    for key, values in grouped.items():
        values.sort()
        result[key] = tuple(np.asarray(column) for column in zip(*values))
    return result


def first_at_or_after(
    series: tuple[np.ndarray, np.ndarray, np.ndarray], threshold: float
) -> tuple[float, float, float]:
    xs, ys, cis = series
    index = min(int(np.searchsorted(xs, threshold, side="left")), len(xs) - 1)
    return float(xs[index]), float(ys[index]), float(cis[index])


def post_stop_series(
    series: tuple[np.ndarray, np.ndarray, np.ndarray], stop: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, cis = series
    keep = xs >= stop - 1e-12
    return xs[keep], ys[keep], cis[keep]


def plot_post_stop_curves(series, stops, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    for ax, benchmark in zip(axes, BENCHMARKS):
        stop = stops[benchmark]
        for method in (
            "radial_gittins_deployable",
            "random_configurations",
            "random_questions",
        ):
            if method == "radial_gittins_deployable":
                xs, ys, cis = post_stop_series(series[(benchmark, method)], stop)
            else:
                xs, ys, cis = series[(benchmark, method)]
            color, label = METHODS[method]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0 if method == "radial_gittins_deployable" else 1.6,
                marker=None if method == "radial_gittins_deployable" else "o",
                markersize=4,
                label=label,
            )
            ax.fill_between(xs, np.maximum(0.0, ys - cis), ys + cis, color=color, alpha=0.12)
        ax.axvline(stop, color="#c45c26", linestyle="--", linewidth=1.3,
                   label=f"Gittins stop ({stop:.1%})")
        ax.set(title=benchmark, xlabel="Observed budget fraction",
               ylabel="Normalized-desirability HV regret")
        ax.set_xlim(0.0, 1.01)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    figure.suptitle(
        "Completed-only Gittins from stopping; random-search baselines over the full budget",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_stop_points(series, stops, stop_summary, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    compared = (
        "radial_gittins_deployable",
        "random_configurations",
        "random_questions",
    )
    for ax, benchmark in zip(axes, BENCHMARKS):
        stop = stops[benchmark]
        values = []
        labels = []
        colors = []
        errors = []
        budgets = []
        for method in compared:
            if method == "radial_gittins_deployable":
                budget = stop
                regret = float(stop_summary[benchmark]["gittins_stop_deployable_hv_regret"])
                ci = 0.0
            else:
                budget, regret, ci = first_at_or_after(series[(benchmark, method)], stop)
            values.append(regret)
            errors.append(ci)
            budgets.append(budget)
            colors.append(METHODS[method][0])
            labels.append(
                {
                    "radial_gittins_deployable": "Gittins\ncompleted-only",
                    "random_configurations": "Random\nconfigurations",
                    "random_questions": "Random\nshared questions",
                }[method]
            )
        positions = np.arange(len(compared))
        for position, value, error, color in zip(positions, values, errors, colors):
            ax.errorbar(position, value, yerr=error, fmt="o", color=color,
                        markersize=7, elinewidth=1.5, capsize=4, zorder=3)
        for x, value, budget in zip(positions, values, budgets):
            ax.annotate(f"{value:.4f}\n@ {budget:.1%}", (x, value), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(positions, labels)
        ax.set(title=benchmark, ylabel="Normalized-desirability HV regret")
        ax.set_ylim(0.0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.3)
    figure.suptitle("HV regret at the Gittins stopping budget", fontsize=13)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_provisional_diagnostic(series, stops, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    for ax, benchmark in zip(axes, BENCHMARKS):
        for method in ("radial_gittins_provisional", "random_configurations", "random_questions"):
            xs, ys, cis = series[(benchmark, method)]
            color, label = METHODS[method]
            ax.plot(xs, ys, color=color, linewidth=2.0 if method.endswith("provisional") else 1.8, linestyle="-",
                    marker=None if method.endswith("provisional") else "o", markersize=4, label=label)
            ax.fill_between(xs, np.maximum(0.0, ys - cis), ys + cis, color=color, alpha=0.1)
        ax.axvline(stops[benchmark], color="#c45c26", linestyle="--", linewidth=1.3,
                   label=f"Gittins stop ({stops[benchmark]:.1%})")
        ax.set(title=benchmark, xlabel="Observed budget fraction",
               ylabel="Normalized-desirability HV regret", xlim=(0.0, 1.01))
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    figure.suptitle("Provisional archive diagnostic (not the deployable recommendation)", fontsize=13)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "analysis/vs/method_comparison/method_hv_regret_summary.csv")
    parser.add_argument("--stop-summary", type=Path, default=ROOT / "analysis/vs/radial_gittins/summary.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "analysis/vs/method_comparison_new")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    series = load_series(args.csv)
    stop_summary = json.loads(args.stop_summary.read_text(encoding="utf-8"))
    stops = {name: float(stop_summary[name]["gittins_stop_budget_fraction"]) for name in BENCHMARKS}
    plot_post_stop_curves(series, stops, args.outdir / "completed_only_hv_regret_from_stopping.png")
    plot_stop_points(series, stops, stop_summary, args.outdir / "hv_regret_at_stopping.png")
    plot_provisional_diagnostic(series, stops, args.outdir / "provisional_hv_regret_diagnostic.png")
    print(f"wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
