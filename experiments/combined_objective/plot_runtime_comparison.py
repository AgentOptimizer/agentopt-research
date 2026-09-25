#!/usr/bin/env python3
"""Plot mean total wall-clock runtime for the paper's six methods."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed"
    / "figures_preview/time"
)

DATASETS = (
    "hotpotqa",
    "restaurant_valid",
    "bird_mini_dev",
    "bing_querylogs",
    "mathqa",
    "restaurant_test",
    "bird_dev",
    "stackoverflow",
)
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "restaurant_test": "Restaurant Test",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD Dev",
    "restaurant_valid": "Restaurant Valid",
    "bing_querylogs": "Bing Query Logs",
    "bird_mini_dev": "BIRD Mini Dev",
}

# Keep this mapping identical to the paper's hypervolume-regret curves.
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METHOD_STYLES = {
    "radial_gittins": ("tab:orange", "Radial Gittins"),
    "ege_sh": ("tab:green", "EGE-SH"),
    "ape_k": ("tab:purple", "APE-k"),
    "qnehvi": ("tab:pink", "qNEHVI"),
    "random_configurations": ("tab:brown", "Random Configurations"),
    "random_questions": ("tab:cyan", "Random Questions"),
}


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    values: dict[tuple[str, str], dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"])
            mean = float(row["mean_seconds"]) if row["mean_seconds"] else math.nan
            error = (
                float(row["error_2se_seconds"])
                if row["error_2se_seconds"]
                else math.nan
            )
            values[key] = {"n": int(row["n"]), "mean": mean, "error": error}
    return values


def plot_runtime(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    values = load_summary(summary_path)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16.6, 7.0),
        sharey=True,
        constrained_layout=False,
    )
    axes = axes.ravel()
    x = np.arange(len(METHODS), dtype=float)

    for ax, dataset in zip(axes, DATASETS):
        means = np.array(
            [values[(dataset, method)]["mean"] for method in METHODS],
            dtype=float,
        )
        errors = np.array(
            [values[(dataset, method)]["error"] for method in METHODS],
            dtype=float,
        )
        available = np.isfinite(means)

        for index, method in enumerate(METHODS):
            color, _ = METHOD_STYLES[method]
            if available[index]:
                ax.bar(
                    x[index],
                    means[index],
                    width=0.78,
                    color=color,
                    edgecolor="black",
                    linewidth=0.55,
                    yerr=errors[index],
                    error_kw={
                        "ecolor": "black",
                        "elinewidth": 0.8,
                        "capsize": 2.2,
                        "capthick": 0.8,
                    },
                    zorder=3,
                )
            else:
                ax.text(
                    x[index],
                    6.2,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=12,
                    color="#555555",
                )

        ax.set_title(DATASET_LABELS[dataset], fontsize=20, pad=7)
        ax.set_yscale("log")
        ax.set_ylim(3.5, 3.0e5)
        ax.set_xlim(-0.65, len(METHODS) - 0.35)
        ax.set_xticks([])
        ax.grid(axis="y", which="major", color="#d7d7d7", linewidth=0.65, alpha=0.72)
        ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.45, alpha=0.55)
        ax.tick_params(axis="y", which="both", labelsize=18)
        ax.set_axisbelow(True)

    fig.supylabel("Mean total wall-clock runtime (s)", fontsize=22, x=0.010, y=0.56)
    legend_handles = [
        Patch(
            facecolor=METHOD_STYLES[method][0],
            edgecolor="black",
            linewidth=0.55,
            label=METHOD_STYLES[method][1],
        )
        for method in METHODS
    ]
    legend_handles.append(
        axes[0].errorbar(
            [np.nan],
            [np.nan],
            yerr=[1.0],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.2,
            label="Error bars: $\\pm 2$ SE",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.025),
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.94, bottom=0.18, wspace=0.10, hspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "total_wall_clock_runtime_mean.png"
    pdf_path = output_dir / "total_wall_clock_runtime_mean.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "runtime_summary.csv",
        help="Plotting-ready mean and 2-SE CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and PDF outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, pdf_path = plot_runtime(args.summary, args.output_dir)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
