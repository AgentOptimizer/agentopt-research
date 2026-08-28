#!/usr/bin/env python3
"""Draw normal, completed-only, and 20-seed completed-only frontiers."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.combined_objective.offline_multiobjective_random_search import common_question_ids, mean_raw_vectors  # noqa: E402
from experiments.combined_objective.offline_pareto_baselines import APE_K, EGE_SH, QNEHVI  # noqa: E402
from experiments.combined_objective.plot_paper_20seed_comparison import LABELS, PICKLES, _draw_landscape, _frequency  # noqa: E402
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402

SEEDS = tuple(range(42, 62))
BENCHMARKS = ("hotpotqa", "mathqa")
METHODS = (("gittins", "Radial Gittins\nAdaptive stopping"),
           (EGE_SH, "EGE-SH\n10% total evaluations"),
           (APE_K, "APE-k\n10% total evaluations"),
           (QNEHVI, "qNEHVI\n10% total evaluations"),
           ("random_questions", "Random questions\n10% total evaluations"),
           ("random_configurations", "Random configurations\n10% total evaluations"))

def _load_cache(data_dir: Path, benchmark: str, method: str, seed: int) -> dict:
    path = data_dir / "frontier_estimates" / benchmark / method / f"seed-{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"estimated_selected_models", "completed_selected_models"}
    if not required <= payload.keys():
        raise ValueError(f"obsolete cache {path}; rebuild it")
    return payload

def _scatter(ax, truth, models, names, **kwargs) -> int:
    indices = np.asarray([models.index(name) for name in names], dtype=int)
    if indices.size:
        ax.scatter(truth[indices, 1], truth[indices, 0], **kwargs)
    return int(indices.size)

def _mean_run_fractions(method: str, method_caches: dict[int, dict],
                        full_search_cost: float,
                        full_total_evaluations: int) -> tuple[float, float]:
    costs = []
    evaluations = []
    for cache in method_caches.values():
        metadata = cache["metadata"]
        if method == "gittins":
            with Path(metadata["source"]).open("rb") as handle:
                saved = pickle.load(handle)
            run = saved[0] if isinstance(saved, tuple) else saved
            search_cost = run.gittins_stop_cost_usd
            total_evaluations = run.gittins_stop_evaluations
        else:
            search_cost = metadata["total_search_cost_usd"]
            total_evaluations = metadata["total_evaluations"]
        if search_cost is None or total_evaluations is None:
            raise ValueError(f"missing run totals for {method}")
        costs.append(float(search_cost) / full_search_cost)
        evaluations.append(int(total_evaluations) / full_total_evaluations)
    return float(np.mean(costs)), float(np.mean(evaluations))

def plot_benchmark(benchmark: str, data_dir: Path, output: Path, seed: int,
                   methods=METHODS) -> None:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    question_ids = common_question_ids(models, datapoints, table)
    truth = mean_raw_vectors(models, question_ids, table)
    full_search_cost = float(sum(
        table[model][question].cost for model in models for question in question_ids
    ))
    full_total_evaluations = len(models) * len(question_ids)
    caches = {method: {s: _load_cache(data_dir, benchmark, method, s) for s in SEEDS}
              for method, _ in methods}
    base = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("YlOrRd_paper_20", base(np.linspace(.18, 1, 256)))
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    n_methods = len(methods)
    figure, axes = plt.subplots(3, n_methods, figsize=(4 * n_methods, 13),
                                sharex=True, sharey=True, squeeze=False)
    style = dict(s=110, color="#d55e00", edgecolors="#725b46", linewidths=.85, zorder=5)
    for col, (method, title) in enumerate(methods):
        for row in range(3):
            _draw_landscape(axes[row, col], truth)
            # Use enough contrast to keep unrecommended configurations visible
            # after the figure is reduced to paper-column width.
            axes[row, col].collections[0].set_facecolor("#c4cad2")
            axes[row, col].collections[0].set_alpha(.85)
            axes[row, col].tick_params(axis="both", labelsize=20)
        mean_cost, _ = _mean_run_fractions(
            method, caches[method], full_search_cost, full_total_evaluations,
        )
        run_details = title.split("\n", 1)[1]
        method_name = title.split("\n", 1)[0]
        axes[0, col].set_title(
            f"{method_name}\n{run_details}\nMean search cost: {mean_cost:.1%}",
            fontsize=21, pad=10,
        )
        n_normal = _scatter(axes[0, col], truth, models, caches[method][seed]["estimated_selected_models"], **style)
        n_complete = _scatter(axes[1, col], truth, models, caches[method][seed]["completed_selected_models"], **style)
        recommendations = {s: set(caches[method][s]["completed_selected_models"]) for s in SEEDS}
        counts = _frequency(models, recommendations)
        shown = counts > 0
        if np.any(shown):
            axes[2, col].scatter(truth[shown, 1], truth[shown, 0], c=counts[shown], cmap=cmap,
                                 norm=norm, s=110, edgecolors="#725b46", linewidths=.85, zorder=5)
        for row, text_value in enumerate((f"{n_normal} recommended", f"{n_complete} recommended",
                                          f"{np.count_nonzero(shown)} unique")):
            axes[row, col].text(.96, .05, text_value, transform=axes[row, col].transAxes,
                                ha="right", fontsize=21)
    legend_handles = (
        Line2D([], [], linestyle="none", marker="o", markersize=13,
               markerfacecolor="#d55e00", markeredgecolor="#725b46",
               label=f"Seed {seed} recommendations (rows 1–2)"),
        Line2D([], [], linestyle="none", marker="o", markersize=13,
               markerfacecolor=cmap(norm(4)), markeredgecolor="#725b46",
               label="20-seed recommendations (row 3; color = frequency)"),
        Line2D([], [], linestyle="none", marker="o", markersize=13,
               markerfacecolor="#c4cad2", markeredgecolor="none",
               label="All configurations"),
        Line2D([], [], color="#3f4854", linewidth=1.7, marker="o",
               markersize=10, markerfacecolor="white", markeredgewidth=1.2,
               label="Empirical Pareto frontier"),
    )
    figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(.5, .002),
        ncol=4, frameon=False, fontsize=24, columnspacing=1.0,
        handlelength=2.0, handletextpad=.6,
    )
    figure.supxlabel("Mean deployment cost (USD)", fontsize=32, x=.515, y=.067)
    figure.supylabel("Mean accuracy", fontsize=32, x=.030, y=.475)
    row_labels = (
        (.700, "Estimated", f"Seed {seed}"),
        (.475, "Actual", f"Seed {seed}"),
        (.250, "Actual", "Seeds 42–61 frequency"),
    )
    for y, archive_label, seed_label in row_labels:
        figure.text(.073, y, archive_label, rotation=90,
                    ha="center", va="center", fontsize=22)
        figure.text(.096, y, seed_label, rotation=90,
                    ha="center", va="center", fontsize=22)
    cax = figure.add_axes([.938, .150, .012, .65])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)", fontsize=26,
                       labelpad=16)
    colorbar.set_ticks([1, 5, 10, 15, 20]); colorbar.ax.tick_params(labelsize=19)
    figure.subplots_adjust(left=.125, right=.91, bottom=.15, top=.80, wspace=.08, hspace=.14)
    figure.suptitle(
        f"{LABELS[benchmark]}: estimations and actual values of Pareto recommendations",
                    fontsize=34, y=.975)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220); figure.savefig(output.with_suffix(".pdf")); plt.close(figure)
    print(f"wrote {output} and {output.with_suffix('.pdf')}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=SEEDS)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=[item[0] for item in METHODS],
                        default=[item[0] for item in METHODS])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "analysis/continuous_seeds_42_61/data")
    args = parser.parse_args()
    methods = tuple(item for item in METHODS if item[0] in args.methods)
    suffix = "frontier_comparison" if len(methods) == len(METHODS) else "frontier_preview"
    for benchmark in args.benchmarks:
        plot_benchmark(benchmark, args.data_dir,
                       args.data_dir.parent / "figures" / f"{benchmark}_3x{len(methods)}_{suffix}.png",
                       args.seed, methods)

if __name__ == "__main__":
    main()
