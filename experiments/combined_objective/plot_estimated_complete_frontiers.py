#!/usr/bin/env python3
"""Draw normal, completed-only, and 20-seed completed-only frontiers."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.combined_objective.offline_multiobjective_random_search import common_question_ids, mean_raw_vectors  # noqa: E402
from experiments.combined_objective.offline_pareto_baselines import APE_K, EGE_SH, QNEHVI  # noqa: E402
from experiments.combined_objective.plot_paper_20seed_comparison import LABELS, PICKLES, _draw_landscape, _frequency  # noqa: E402
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402

SEEDS = tuple(range(42, 62))
BENCHMARKS = ("hotpotqa", "mathqa")
METHODS = (("gittins", "Gittins\nAdaptive stop"), (EGE_SH, "EGE-SH\n10% budget"),
           (APE_K, "APE-k\n10% budget"), (QNEHVI, "qNEHVI\n10% budget"),
           ("random_questions", "Random questions\n10% budget"),
           ("random_configurations", "Random configurations\n10% budget"))

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

def plot_benchmark(benchmark: str, data_dir: Path, output: Path, seed: int,
                   methods=METHODS) -> None:
    models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
    truth = mean_raw_vectors(models, common_question_ids(models, datapoints, table), table)
    caches = {method: {s: _load_cache(data_dir, benchmark, method, s) for s in SEEDS}
              for method, _ in methods}
    base = mpl.colormaps["YlOrRd"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("YlOrRd_paper_20", base(np.linspace(.18, 1, 256)))
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    n_methods = len(methods)
    figure, axes = plt.subplots(3, n_methods, figsize=(4 * n_methods, 12),
                                sharex=True, sharey=True, squeeze=False)
    style = dict(s=110, color="#d55e00", edgecolors="#725b46", linewidths=.85, zorder=5)
    for col, (method, title) in enumerate(methods):
        for row in range(3):
            _draw_landscape(axes[row, col], truth)
            axes[row, col].tick_params(axis="both", labelsize=16)
        axes[0, col].set_title(title, fontsize=19, pad=10)
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
                                ha="right", fontsize=14)
    figure.supxlabel("Mean deployment cost (USD)", fontsize=22, y=.025)
    figure.supylabel("Mean accuracy", fontsize=22, x=.010)
    row_labels = (
        (.735, "Estimated", f"Seed {seed}"),
        (.475, "Completed-only", f"Seed {seed}"),
        (.215, "Completed-only", "Seeds 42–61 frequency"),
    )
    for y, archive_label, seed_label in row_labels:
        figure.text(.073, y, archive_label, rotation=90,
                    ha="center", va="center", fontsize=18)
        figure.text(.102, y, seed_label, rotation=90,
                    ha="center", va="center", fontsize=18)
    cax = figure.add_axes([.94, .105, .010, .72])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)", fontsize=18)
    colorbar.set_ticks([1, 5, 10, 15, 20]); colorbar.ax.tick_params(labelsize=15)
    figure.subplots_adjust(left=.125, right=.91, bottom=.10, top=.84, wspace=.08, hspace=.14)
    figure.suptitle(f"{LABELS[benchmark]}: estimated and completed-only Pareto recommendations",
                    fontsize=25, y=.955)
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
