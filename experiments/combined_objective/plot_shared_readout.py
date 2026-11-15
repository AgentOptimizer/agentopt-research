"""Plot the paired CC-Gittins and Random Questions readouts from saved replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from experiments.combined_objective.run_shared_readout import CC_GITTINS, RANDOM_QUESTIONS, READOUT, TARGETS


METRICS = {
    "precision": "Precision ↑",
    "recall": "Recall ↑",
    "gd": "GD ↓",
    "igd": "IGD ↓",
    "hv_regret": "HV Regret ↓",
}
RANDOM_QUESTIONS_COLOR = "#0086a8"
STYLE = {
    (CC_GITTINS, READOUT): ("CC-Gittins (conservative)", "#ff7f0e", "-"),
    (CC_GITTINS, "finite_mean_beta0"): ("CC-Gittins (posterior mean)", "#1f77b4", "-"),
    (RANDOM_QUESTIONS, READOUT): ("Random Questions (conservative)", RANDOM_QUESTIONS_COLOR, "-"),
    (RANDOM_QUESTIONS, "finite_mean_beta0"): ("Random Questions (posterior mean)", RANDOM_QUESTIONS_COLOR, "-."),
    (RANDOM_QUESTIONS, "empirical_means"): ("Random Questions (empirical mean)", RANDOM_QUESTIONS_COLOR, "--"),
}


def load_curves(results: Path, benchmark: str, seeds: list[int]) -> np.ndarray:
    """Load metric values in seed, readout, budget, metric order."""
    curves = np.full((len(seeds), len(STYLE), len(TARGETS), len(METRICS)), np.nan)
    series_index = {key: i for i, key in enumerate(STYLE)}
    budget_index = {target: i for i, target in enumerate(TARGETS)}
    provenance = None
    for i, seed in enumerate(seeds):
        path = results / f"{benchmark}-seed{seed}.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        if (run["benchmark"], run["seed"]) != (benchmark, seed):
            raise ValueError(f"{path}: benchmark or seed does not match the filename")
        inputs = (run["input_sha256"], run["protocol"], run["cc_gittins_acquisition_target"])
        if provenance is None:
            provenance = inputs
        elif inputs != provenance:
            raise ValueError(f"{path}: incompatible inputs or acquisition protocol across seeds")
        for row in run["rows"]:
            series = (row["method"], row["readout"])
            if series not in series_index:
                continue
            values = curves[i, series_index[series], budget_index[row["target_cost_fraction"]]]
            if not np.isnan(values).all():
                raise ValueError(f"{path}: duplicate checkpoint for {series}")
            values[:] = [row[metric] for metric in METRICS]
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: nonfinite metrics for {series}")
        for j, series in enumerate(STYLE):
            if np.isnan(curves[i, j]).all():
                raise ValueError(f"{path}: no affordable checkpoints for {series}")
    return curves


def plot(results: Path, output: Path, benchmarks: list[str], seeds: list[int]) -> None:
    datasets = [load_curves(results, benchmark, seeds) for benchmark in benchmarks]
    benchmark_labels = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
    x = np.asarray(TARGETS)
    height = 1.95 * len(benchmarks) + 1.50
    with plt.rc_context({"font.family": "serif", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(benchmarks), len(METRICS), squeeze=False, sharex=True,
                                 figsize=(14, height))
        for row, (benchmark, curves) in enumerate(zip(benchmarks, datasets)):
            # Show only checkpoints with all selected seeds available for all readouts.
            available = np.isfinite(curves).all(axis=(0, 1, 3))
            if not available.any():
                raise ValueError(f"{benchmark}: no checkpoints shared by all methods and seeds")
            for column, (metric, label) in enumerate(METRICS.items()):
                ax = axes[row, column]
                for series, (_, color, linestyle) in enumerate(STYLE.values()):
                    values = curves[:, series, :, column][:, available]
                    mean = values.mean(axis=0)
                    if len(seeds) > 1:
                        two_se = 2 * values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
                        upper = mean + two_se
                        if metric in {"precision", "recall"}:
                            upper = np.minimum(upper, 1)
                        ax.fill_between(x[available], np.maximum(0, mean - two_se), upper,
                                        color=color, alpha=0.10, linewidth=0)
                    ax.plot(x[available], mean, color=color, linestyle=linestyle, linewidth=1.45)
                if row == 0:
                    ax.set_title(label, fontsize=13)
                if column == 0:
                    ax.set_ylabel(benchmark_labels[benchmark], fontsize=13)
                ax.set_xlim(0, TARGETS[-1])
                ax.set_ylim(bottom=0)
                if metric in {"precision", "recall"}:
                    ax.set_ylim(top=1.02)
                ax.set_xticks([0, .1, .2, .3])
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(color="#e3e8ec", linewidth=0.5)
                ax.set_axisbelow(True)
        handles = [Line2D([], [], color=color, linestyle=linestyle, label=label, linewidth=1.45)
                   for label, color, linestyle in STYLE.values()]
        if len(seeds) > 1:
            handles.append(Patch(facecolor="#e4e4e4", label="±2 SE"))
        fig.subplots_adjust(left=0.065, right=0.99, top=1 - 0.30 / height,
                            bottom=1.12 / height, hspace=0.37, wspace=0.42)
        fig.supxlabel("Percentage of Exhaustive Evaluation Cost", y=0.57 / height, fontsize=13)
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=("hotpotqa", "mathqa"),
                        default=["hotpotqa", "mathqa"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be unique nonnegative integers")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("--benchmarks must be unique")
    plot(args.results, args.output, args.benchmarks, args.seeds)


if __name__ == "__main__":
    main()
