"""Plot the three-objective Q/L/D prior-variance factorial from saved runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from experiments.combined_objective import plot_three_objective_direction_ablation as common
from experiments.combined_objective.run_server_latency_prior_factorial import VARIANTS, variant_name


COLORS = ("tab:orange", "#5254a3", "#637939", "#8c6d31", "tab:blue",
          "tab:cyan", "#888888", "tab:green")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-quality-prior-mode", choices=("fixed", "warm"),
                        default="warm")
    args = parser.parse_args()
    main_modes = (args.main_quality_prior_mode, "warm", "warm")
    alternate_modes = ("warm" if args.main_quality_prior_mode == "fixed"
                       else "fixed", "warm", "warm")
    all_modes = (main_modes, alternate_modes) + tuple(
        modes for modes in VARIANTS if modes not in (main_modes, alternate_modes)
    )
    labels = {modes: "G{} Q/L/D: {}".format(
        index if index == 0 else index + 10,
        "/".join("warm" if value == "warm" else "0.04" for value in modes),
    ) for index, modes in enumerate(all_modes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.configure_matplotlib()
    rows = []
    curves = {}
    for dataset in common.DATASETS:
        for modes in all_modes:
            variant = variant_name(modes)
            source = args.ablation_root if modes == main_modes else args.factorial_root
            configuration = "four_direction_real_cost" if modes == main_modes else variant
            runs = []
            for seed in common.SEEDS:
                path = source / configuration / f"seed-{seed}" / dataset / "result.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["config"]["benchmark"] != dataset or payload["config"]["seed"] != seed:
                    raise ValueError(f"Run identity mismatch: {path}")
                if tuple(payload["config"]["prior_variance_modes"]) != modes:
                    raise ValueError(f"Prior modes mismatch: {path}")
                runs.append(common.aligned_run(payload["run"]))
            for metric, _ in common.METRICS:
                mean, two_se, count = common.summarize(
                    np.stack([run[metric] for run in runs]))
                curves[(dataset, modes, metric)] = (mean, two_se)
                for x, y, band, n in zip(common.GRID, mean, two_se, count):
                    rows.append({"dataset": dataset, "configuration": variant,
                                 "metric": metric, "cost_fraction": float(x),
                                 "mean": "" if not math.isfinite(y) else float(y),
                                 "two_se": "" if not math.isfinite(band) else float(band),
                                 "n_runs": int(n)})
    summary = args.output_dir / "prior_variance_curve_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 3, figsize=(20, 11.2), sharex=True)
    figure.subplots_adjust(left=.092, right=.987, top=.93, bottom=.295,
                           wspace=.34, hspace=.32)
    for row, dataset in enumerate(common.DATASETS):
        for column, (metric, title) in enumerate(common.METRICS):
            axis = axes[row, column]
            upper = 0.
            for modes, color in zip(all_modes[1:] + all_modes[:1], COLORS[1:] + COLORS[:1]):
                mean, band = curves[(dataset, modes, metric)]
                main = modes == main_modes
                axis.fill_between(common.GRID, np.maximum(0, mean-band), mean+band,
                                  color=color, alpha=.14 if main else .05,
                                  linewidth=0, zorder=2 if main else 1)
                axis.plot(common.GRID, mean, color=color, linewidth=3.24 if main else 2.4,
                          zorder=10 if main else 3)
                upper = max(upper, float(np.nanmax(mean+band)))
            axis.set_xlim(-.005, .305)
            axis.set_ylim(-.025*upper, 1.06*upper)
            axis.set_xticks([0, .1, .2, .3])
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            axis.tick_params(axis="both", labelsize=28, width=1, length=5)
            axis.grid(True, alpha=.22, color="#9da9bb", linewidth=.7)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(title, fontsize=32, pad=13)
            if column == 0:
                axis.set_ylabel({"hotpotqa": "HotpotQA", "mathqa": "MathQA"}[dataset],
                                fontsize=32, labelpad=11)
    xlabel = figure.supxlabel("Percentage of Exhaustive Evaluation Cost", fontsize=30, y=.194)
    figure.canvas.draw()
    top = (xlabel.get_window_extent(figure.canvas.get_renderer()).y0 - figure.dpi*.15) / figure.bbox.height
    handles = [Line2D([], [], color=color, linewidth=3.24 if modes == main_modes else 2.4,
                      label=labels[modes]) for modes, color in zip(all_modes, COLORS)]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, top),
                  ncol=2, frameon=False, fontsize=17, handlelength=2.3,
                  columnspacing=.8, handletextpad=.5, labelspacing=.55)
    stem = args.output_dir / "three_objective_prior_variance"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix("." + extension), dpi=300,
                       bbox_inches="tight", pad_inches=.12, facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
