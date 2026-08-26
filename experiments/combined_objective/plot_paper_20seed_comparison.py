#!/usr/bin/env python3
"""Create the consolidated 2x4 frontier and all-method HV-regret figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
    pareto_min_cost_indices,
)
from experiments.combined_objective.plot_multiobjective_method_comparison import (  # noqa: E402
    _radial_regret_series,
    panels_from_csv,
    write_comparison_figure,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


PICKLES = {
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}
LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}


def _load_json_sets(path: Path, benchmark: str, *, budget: float | None = None) -> dict[int, set[str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    output = {}
    for row in rows:
        row_benchmark = row.get("benchmark", benchmark).lower()
        if row_benchmark not in (benchmark, LABELS[benchmark].lower()):
            continue
        if budget is not None and not np.isclose(float(row["budget_fraction"]), budget):
            continue
        output[int(row["seed"])] = set(json.loads(row["selected_models"]))
    return output


def _mean_cost_fraction(
    path: Path,
    benchmark: str,
    seeds: set[int],
    *,
    fraction_field: str | None = None,
    budget: float | None = None,
) -> float:
    """Return mean actual search cost / full-search cost over matched seeds."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    matching = [
        row for row in rows
        if row.get("benchmark", benchmark).lower() in (benchmark, LABELS[benchmark].lower())
        and int(row["seed"]) in seeds
    ]
    if fraction_field is not None:
        return float(np.mean([float(row[fraction_field]) for row in matching]))

    # The consolidated random CSV keeps only the displayed 10% and 40%
    # checkpoints. Recover each run's full-matrix USD cost from the matched
    # Gittins summary: stop_cost / stop_cost_fraction uses that same denominator.
    gittins_rows = list(csv.DictReader(
        (path.parent / "gittins_seed_results.csv").open(encoding="utf-8")
    ))
    full_cost = {
        int(row["seed"]): (
            float(row["gittins_stop_cost_usd"])
            / float(row["gittins_stop_cost_fraction"])
        )
        for row in gittins_rows
        if row["benchmark"].lower() in (benchmark, LABELS[benchmark].lower())
        and int(row["seed"]) in seeds
    }
    selected = [
        row for row in matching
        if budget is not None and np.isclose(float(row["budget_fraction"]), budget)
    ]
    if len(selected) != len(seeds) or set(full_cost) != seeds:
        raise ValueError(f"missing cost checkpoints in {path}")
    return float(np.mean([
        float(row["total_search_cost_usd"]) / full_cost[int(row["seed"])]
        for row in selected
    ]))


def _draw_landscape(ax, truth: np.ndarray) -> None:
    indices = np.asarray(pareto_min_cost_indices(truth), dtype=int)
    indices = indices[np.argsort(truth[indices, 1])]
    front = truth[indices]
    ax.scatter(truth[:, 1], truth[:, 0], s=15, color="#d8dce2", alpha=0.7,
               edgecolors="none", zorder=1)
    ax.plot(front[:, 1], front[:, 0], color="#3f4854", linewidth=1.4,
            marker="o", markersize=3, markerfacecolor="white", zorder=3)
    ax.grid(color="#d9dde3", linewidth=0.55, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def _frequency(models: list[str], recommendations: dict[int, set[str]]) -> np.ndarray:
    counts = Counter()
    for selected in recommendations.values():
        counts.update(selected)
    return np.asarray([counts[model] for model in models], dtype=int)


def plot_frontier_grid(benchmark: str, data_dir: Path, output_stem: Path) -> None:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    truth = mean_raw_vectors(models, common_question_ids(models, datapoints, table), table)
    gittins = _load_json_sets(data_dir / "gittins_seed_results.csv", benchmark)
    ucb = _load_json_sets(data_dir / "ucb_seed_results.csv", benchmark)
    random_path = data_dir / f"{benchmark}_random_questions.csv"
    random10 = _load_json_sets(random_path, benchmark, budget=0.1)
    random40 = _load_json_sets(random_path, benchmark, budget=0.4)
    matched_seeds = set(gittins)
    mean_costs = (
        _mean_cost_fraction(data_dir / "gittins_seed_results.csv", benchmark,
                            matched_seeds, fraction_field="gittins_stop_cost_fraction"),
        _mean_cost_fraction(data_dir / "ucb_seed_results.csv", benchmark,
                            matched_seeds, fraction_field="stop_budget_fraction"),
        _mean_cost_fraction(random_path, benchmark, matched_seeds, budget=0.1),
        _mean_cost_fraction(random_path, benchmark, matched_seeds, budget=0.4),
    )
    conditions = (
        (f"Gittins adaptive stop\nmean actual cost: {mean_costs[0]:.1%}", gittins),
        (f"Radial UCB adaptive stop\nmean actual cost: {mean_costs[1]:.1%}", ucb),
        (f"Random questions, 10% budget\nmean actual cost: {mean_costs[2]:.1%}", random10),
        (f"Random questions, 40% budget\nmean actual cost: {mean_costs[3]:.1%}", random40),
    )
    seed_sets = [set(values) for _, values in conditions]
    if any(len(values) != 20 for values in seed_sets) or len(set(map(frozenset, seed_sets))) != 1:
        raise ValueError("every panel must contain the same 20 seeds")

    base = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "YlOrRd_paper_20", base(np.linspace(0.18, 1.0, 256)),
    )
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.7), sharex=True, sharey=True)
    for col, (title, recommendations) in enumerate(conditions):
        for row in range(2):
            _draw_landscape(axes[row, col], truth)
        axes[0, col].set_title(title, fontsize=12)
        selected = np.asarray(
            [models.index(name) for name in recommendations[42]], dtype=int,
        )
        points = truth[selected]
        axes[0, col].scatter(points[:, 1], points[:, 0], s=48, color="#d55e00",
                             edgecolors="white", linewidths=0.75, zorder=5)
        axes[0, col].text(0.82, 0.04, f"{len(selected)} recommended", transform=axes[0, col].transAxes,
                          ha="right", va="bottom", fontsize=8.5)
        counts = _frequency(models, recommendations)
        shown = counts > 0
        axes[1, col].scatter(
            truth[shown, 1], truth[shown, 0], c=counts[shown], cmap=cmap, norm=norm,
            s=46, edgecolors="#725b46", linewidths=0.5, zorder=5,
        )
    axes[0, 0].set_ylabel("Seed 42\n\nMean accuracy")
    axes[1, 0].set_ylabel("20-seed frequency\n\nMean accuracy")
    for ax in axes[1]:
        ax.set_xlabel("Mean deployment cost (USD)")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Reserve a dedicated axis outside the four frontier columns.  Passing
    # ``ax=axes`` lets Matplotlib steal space unevenly and can overlap the last
    # panel once the manual paper layout is applied.
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.82,
                        wspace=0.08, hspace=0.12)
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.014, 0.57])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)")
    colorbar.set_ticks([1, 5, 10, 15, 20])
    fig.suptitle(
        f"{LABELS[benchmark]}: single-run and 20-seed Pareto recommendations",
        fontsize=15,
        y=0.975,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_hv_comparison(data_dir: Path, output_path: Path) -> None:
    source_csv = data_dir / "gittins_random_hv_regret_summary.csv"
    panels = panels_from_csv(source_csv, {}, "cost")
    for panel in panels:
        panel["random"] = [
            series for series in panel["random"]
            if series[0] == "random_questions"
        ]
    ucb_rows = list(csv.DictReader((data_dir / "ucb_hv_regret_trajectories.csv").open()))
    appended_rows = []
    for row in csv.DictReader(source_csv.open()):
        if row["method"] == "random_configurations":
            continue
        row = dict(row)
        row["ci95_half_width"] = float(row["ci95_half_width"]) * 2.0 / 1.96
        appended_rows.append(row)
    for panel in panels:
        benchmark = panel["name"].lower()
        trajectories = []
        for seed in sorted({int(row["seed"]) for row in ucb_rows if row["benchmark"] == benchmark}):
            rows = [row for row in ucb_rows if row["benchmark"] == benchmark and int(row["seed"]) == seed]
            trajectories.append((
                np.asarray([float(row["budget_fraction"]) for row in rows]),
                np.asarray([float(row["hv_regret"]) for row in rows]),
            ))
        series = _radial_regret_series(trajectories)
        panel["ucb"] = series
        for x, mean, ci, count in zip(*series):
            appended_rows.append({
                "benchmark": panel["name"], "method": "radial_ucb",
                "budget_fraction": x, "mean_hv_regret": mean,
                "ci95_half_width": ci, "n_runs": int(count),
                "cost_reference_usd": "", "x_axis": "cost",
                "stop_axis_fraction": panel["stop_mean"],
            })
    write_comparison_figure(
        out_path=output_path,
        title="Gittins, radial UCB, and random search (20 matched seeds)",
        panels=panels, seeds=20, seed=42, x_axis="cost",
    )
    with (data_dir / "all_method_hv_regret_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(appended_rows[0]))
        writer.writeheader(); writer.writerows(appended_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "analysis/paper_20seed_method_comparison")
    args = parser.parse_args()
    outdir = args.data_dir / "figures"
    for benchmark in PICKLES:
        plot_frontier_grid(benchmark, args.data_dir,
                           outdir / f"{benchmark}_2x4_frontier_comparison")
    write_hv_comparison(args.data_dir, outdir / "all_methods_20seed_hv_regret.png")
    print(f"wrote consolidated figures under {outdir}")


if __name__ == "__main__":
    main()
