#!/usr/bin/env python3
"""Plot policy wall-clock runtime for the direction-count ablations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiments.combined_objective.gittins_ablation_v2 import (
    DEFAULT_FIGURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    G0_CONFIGURATION,
    SEEDS,
    result_path,
)
from experiments.combined_objective.plot_gittins_ablation_v2_20seed import (
    CONFIG_COLORS,
    DATASET_LABELS,
    PANEL_ORDER,
    configure_matplotlib,
)


CONFIGURATIONS = (
    G0_CONFIGURATION,
    "g5_axes_midpoint_real_cost",
    "g6_five_directions_real_cost",
)
CONFIGURATION_LABELS = {
    G0_CONFIGURATION: "G0 Two directions",
    "g5_axes_midpoint_real_cost": "G5 Three directions",
    "g6_five_directions_real_cost": "G6 Five directions",
}
POLICY_TIME_PATTERN = re.compile(
    rb'"policy_wall_time_seconds"\s*:\s*([0-9.eE+-]+)'
)


def policy_wall_time(result_path_: Path) -> float:
    with result_path_.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            matches = POLICY_TIME_PATTERN.findall(mapped)
    if len(matches) != 1:
        raise ValueError(
            f"{result_path_}: expected one policy_wall_time_seconds value, "
            f"found {len(matches)}"
        )
    seconds = float(matches[0])
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{result_path_}: invalid policy wall time {seconds}")
    return seconds


def load_runtime_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        for dataset in PANEL_ORDER:
            for seed in SEEDS:
                path = result_path(
                    configuration,
                    dataset,
                    seed,
                    output_root=results_root,
                )
                if not path.is_file():
                    raise FileNotFoundError(f"missing ablation result: {path}")
                rows.append(
                    {
                        "configuration": configuration,
                        "dataset": dataset,
                        "seed": seed,
                        "policy_wall_time_seconds": policy_wall_time(path),
                        "source": str(path),
                    }
                )
    return rows


def summarize_runtime(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["configuration"]), str(row["dataset"]))
        grouped.setdefault(key, []).append(float(row["policy_wall_time_seconds"]))

    summary: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        for dataset in PANEL_ORDER:
            values = grouped[(configuration, dataset)]
            if len(values) != len(SEEDS):
                raise ValueError(
                    f"{configuration}/{dataset}: expected {len(SEEDS)} runs, "
                    f"found {len(values)}"
                )
            summary.append(
                {
                    "configuration": configuration,
                    "dataset": dataset,
                    "n_runs": len(values),
                    "mean_seconds": statistics.fmean(values),
                    "error_2se_seconds": (
                        2.0 * statistics.stdev(values) / math.sqrt(len(values))
                    ),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime(
    summary: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    values = {
        (str(row["configuration"]), str(row["dataset"])): row
        for row in summary
    }
    configure_matplotlib()
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    x = np.arange(len(CONFIGURATIONS), dtype=float)

    for axis, dataset in zip(axes.flat, PANEL_ORDER):
        means = np.asarray(
            [values[(name, dataset)]["mean_seconds"] for name in CONFIGURATIONS],
            dtype=float,
        )
        errors = np.asarray(
            [
                values[(name, dataset)]["error_2se_seconds"]
                for name in CONFIGURATIONS
            ],
            dtype=float,
        )
        axis.bar(
            x,
            means,
            yerr=errors,
            width=0.72,
            color=[CONFIG_COLORS[name] for name in CONFIGURATIONS],
            edgecolor="black",
            linewidth=0.7,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.1,
                "capsize": 4.0,
                "capthick": 1.1,
            },
            zorder=3,
        )
        axis.set_title(DATASET_LABELS[dataset], fontsize=25, pad=9)
        axis.set_xlim(-0.65, len(CONFIGURATIONS) - 0.35)
        axis.set_xticks([])
        axis.set_ylim(0.0, float(np.max(means + errors)) * 1.16)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.tick_params(axis="y", labelsize=23, width=1.0, length=5)
        axis.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.55)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        axis.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=CONFIG_COLORS[name],
            edgecolor="black",
            linewidth=0.7,
            label=CONFIGURATION_LABELS[name],
        )
        for name in CONFIGURATIONS
    ]
    legend_handles.append(
        axes[0, 0].errorbar(
            [np.nan],
            [np.nan],
            yerr=[1.0],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4.0,
            capthick=1.2,
            label=r"$\pm 2$ SE",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=23,
        columnspacing=0.8,
        handlelength=1.7,
        handletextpad=0.5,
    )
    figure.supylabel(
        "Mean policy wall-clock runtime (s)", fontsize=27, x=0.010, y=0.585
    )
    figure.suptitle("Direction Runtime Comparison", fontsize=29, y=1.055)
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.93,
        bottom=0.25,
        wspace=0.34,
        hspace=0.42,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "direction_runtime_comparison_20seed"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(
        png_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_ROOT / "runtime",
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()

    rows = load_runtime_rows(results_root)
    summary = summarize_runtime(rows)
    write_csv(output_dir / "direction_runtime_by_seed.csv", rows)
    write_csv(output_dir / "direction_runtime_summary.csv", summary)
    (output_dir / "direction_runtime_manifest.json").write_text(
        json.dumps(
            {
                "configurations": {
                    name: CONFIGURATION_LABELS[name] for name in CONFIGURATIONS
                },
                "datasets": list(PANEL_ORDER),
                "seeds": list(SEEDS),
                "runtime_field": "result.run.policy_wall_time_seconds",
                "uncertainty": "mean +/- 2 standard errors over 20 matched seeds",
                "caveat": (
                    "Policy-only wall-clock comparison for isolating the "
                    "computational effect of direction count. It excludes "
                    "data loading, result serialization, and Slurm queue time. "
                    "Runs were independently scheduled, so compute-node "
                    "conditions may differ."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    png_path, pdf_path = plot_runtime(summary, output_dir)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
