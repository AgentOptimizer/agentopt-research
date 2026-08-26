#!/usr/bin/env python3
"""Plot existing Gittins/random checkpoints against actual cumulative USD cost."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RADIAL_DIR = ROOT / "analysis/vs/radial_gittins"
RANDOM_DIR = ROOT / "analysis/vs/random_search"
OUTDIR = ROOT / "analysis/vs/method_comparison_cost_axis"
BENCHMARKS = {"HotpotQA": "hotpotqa", "MathQA": "mathqa"}
STYLES = {
    "gittins": ("#1f4e79", "Radial-Gittins completed-only"),
    "provisional": ("#1f4e79", "Radial-Gittins provisional"),
    "random_configurations": ("#7b61a8", "Random configurations"),
    "random_questions": ("#2a9d8f", "Random shared questions"),
}


def load_radial() -> dict[str, list[dict[str, float | str]]]:
    grouped = defaultdict(list)
    with (RADIAL_DIR / "cost_trajectory.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["benchmark"]].append(
                {
                    "event": row["event"],
                    "cost": float(row["cumulative_search_cost_usd"]),
                    "deployable": float(row["deployable_hv_regret"]),
                    "provisional": float(row["provisional_hv_regret"]),
                }
            )
    return grouped


def load_random(slug: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # The comparison figures use the same seed=42 checkpoints as the snapshots.
    grouped = defaultdict(list)
    path = RANDOM_DIR / slug / "multi_seed_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) == 42:
                grouped[row["version"]].append(
                    (float(row["total_search_cost_usd"]), float(row["hypervolume_regret"]))
                )
    result = {}
    for method, values in grouped.items():
        values.sort()
        result[method] = tuple(np.asarray(column) for column in zip(*values))
    return result


def draw(completed_only: bool, output: Path) -> None:
    radial = load_radial()
    summary = json.loads((RADIAL_DIR / "summary.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharey=False)
    for ax, (benchmark, slug) in zip(axes, BENCHMARKS.items()):
        stop_cost = float(summary[benchmark]["gittins_stop_cost_usd"])
        random_series = load_random(slug)
        full_matrix_cost = max(float(xs[-1]) for xs, _ in random_series.values())
        stop_cost_fraction = stop_cost / full_matrix_cost
        points = radial[benchmark]
        if completed_only:
            points = [point for point in points if float(point["cost"]) >= stop_cost - 1e-12]
            field, method = "deployable", "gittins"
        else:
            field, method = "provisional", "provisional"
        color, label = STYLES[method]
        ax.plot([float(p["cost"]) / full_matrix_cost for p in points],
                [p[field] for p in points],
                color=color, linewidth=2.0, label=label)

        for random_method, (xs, ys) in random_series.items():
            color, label = STYLES[random_method]
            ax.plot(xs / full_matrix_cost, ys, color=color, marker="o", markersize=4,
                    linewidth=1.6, label=label)
        ax.axvline(stop_cost_fraction, color="#c45c26", linestyle="--", linewidth=1.4,
                   label=f"Gittins stop ({stop_cost_fraction:.1%}, ${stop_cost:.2f})")
        ax.set_title(benchmark)
        ax.set_xlabel("Cumulative search cost fraction")
        ax.set_ylabel("Normalized-desirability HV regret")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    title = (
        "Completed-only Gittins from stopping vs random search by cumulative cost fraction"
        if completed_only
        else "Provisional Gittins diagnostic vs random search by cumulative cost fraction"
    )
    figure.suptitle(title, fontsize=12.5)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    draw(True, OUTDIR / "completed_only_hv_regret_vs_cumulative_cost.png")
    draw(False, OUTDIR / "provisional_hv_regret_vs_cumulative_cost.png")
    print(f"wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
