#!/usr/bin/env python3
"""Plot policy runtime for the three-objective direction-set ablation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RELATIVE = Path(
    "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    "/gittins_g2_main_figures"
)
DATA_REPOSITORY = ROOT
LINKED_REPOSITORY = ROOT.parents[1] / "agentopt-research"
if not (DATA_REPOSITORY / ANALYSIS_RELATIVE / "source_data").is_dir() and (
    LINKED_REPOSITORY / ANALYSIS_RELATIVE / "source_data"
).is_dir():
    DATA_REPOSITORY = LINKED_REPOSITORY
SOURCE = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "source_data/ablation/"
    "three_objective_direction_policy_runtime_fixed_30pct_entries_20seed"
)
OUTPUT_DIR = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "corrected_gd_igd_complete_returned_set_20260929"
    / "three_objective/direction_ablation"
)

SEEDS = tuple(range(42, 62))
DATASETS = ("hotpotqa", "mathqa")
DATASET_LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
CONFIGURATIONS = (
    "q_only_real_cost",
    "l_only_real_cost",
    "d_only_real_cost",
    "pure_axes_real_cost",
    "four_direction_real_cost",
    "six_ql_dense_real_cost",
    "six_three_edges_real_cost",
)
CONFIGURATION_LABELS = {
    "q_only_real_cost": "Q-only",
    "l_only_real_cost": "L-only",
    "d_only_real_cost": "D-only",
    "pure_axes_real_cost": "Pure axes",
    "four_direction_real_cost": "Four directions",
    "six_ql_dense_real_cost": "Six directions (Q-L dense)",
    "six_three_edges_real_cost": "Six directions (three edges)",
}
CONFIGURATION_COLORS = {
    "q_only_real_cost": "tab:gray",
    "l_only_real_cost": "tab:brown",
    "d_only_real_cost": "tab:olive",
    "pure_axes_real_cost": "tab:blue",
    "four_direction_real_cost": "tab:orange",
    "six_ql_dense_real_cost": "tab:green",
    "six_three_edges_real_cost": "tab:purple",
}
def result_path(source: Path, configuration: str, dataset: str, seed: int) -> Path:
    return source / configuration / f"seed-{seed}" / dataset / "runtime.json"


def load_rows(source: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        for dataset in DATASETS:
            for seed in SEEDS:
                path = result_path(source, configuration, dataset, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"missing ablation result: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["stop_reason"] != "observation_budget":
                    raise ValueError(
                        f"{path}: expected observation_budget, got "
                        f"{payload['stop_reason']!r}"
                    )
                if int(payload["total_evaluations"]) != int(
                    payload["target_entry_cap"]
                ):
                    raise ValueError(f"{path}: did not reach exact entry cap")
                seconds = float(payload["policy_wall_time_seconds"])
                if not math.isfinite(seconds) or seconds <= 0.0:
                    raise ValueError(f"{path}: invalid policy wall time {seconds}")
                rows.append({
                    "configuration": configuration,
                    "dataset": dataset,
                    "seed": seed,
                    "policy_wall_time_seconds": seconds,
                    "available_entries": int(payload["available_entries"]),
                    "total_evaluations": int(payload["total_evaluations"]),
                    "realized_entry_fraction": float(
                        payload["realized_entry_fraction"]
                    ),
                    "forced_continuation_after_eta_floor": bool(
                        payload["forced_continuation_after_eta_floor"]
                    ),
                    "source": str(path),
                })
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["configuration"]), str(row["dataset"]))
        grouped.setdefault(key, []).append(float(row["policy_wall_time_seconds"]))

    summary: list[dict[str, object]] = []
    for dataset in DATASETS:
        baseline = statistics.fmean(grouped[(CONFIGURATIONS[0], dataset)])
        for configuration in CONFIGURATIONS:
            values = grouped[(configuration, dataset)]
            if len(values) != len(SEEDS):
                raise ValueError(
                    f"{configuration}/{dataset}: expected {len(SEEDS)} runs, "
                    f"found {len(values)}"
                )
            mean = statistics.fmean(values)
            summary.append({
                "configuration": configuration,
                "label": CONFIGURATION_LABELS[configuration],
                "dataset": dataset,
                "n_runs": len(values),
                "mean_seconds": mean,
                "error_2se_seconds": (
                    2.0 * statistics.stdev(values) / math.sqrt(len(values))
                ),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "ratio_to_pure_axes": mean / baseline,
                "mean_total_evaluations": statistics.fmean(
                    float(row["total_evaluations"])
                    for row in rows
                    if row["configuration"] == configuration
                    and row["dataset"] == dataset
                ),
                "mean_realized_entry_fraction": statistics.fmean(
                    float(row["realized_entry_fraction"])
                    for row in rows
                    if row["configuration"] == configuration
                    and row["dataset"] == dataset
                ),
                "forced_continuation_runs": sum(
                    bool(row["forced_continuation_after_eta_floor"])
                    for row in rows
                    if row["configuration"] == configuration
                    and row["dataset"] == dataset
                ),
            })
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def plot_runtime(
    summary: list[dict[str, object]], output_dir: Path,
) -> tuple[Path, Path, Path]:
    values = {
        (str(row["configuration"]), str(row["dataset"])): row
        for row in summary
    }
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    x = np.arange(len(CONFIGURATIONS), dtype=float)
    global_top = max(
        float(row["mean_seconds"]) + float(row["error_2se_seconds"])
        for row in summary
    ) * 1.15

    for axis, dataset in zip(axes, DATASETS):
        means = np.asarray(
            [values[(name, dataset)]["mean_seconds"] for name in CONFIGURATIONS],
            dtype=float,
        )
        errors = np.asarray(
            [values[(name, dataset)]["error_2se_seconds"] for name in CONFIGURATIONS],
            dtype=float,
        )
        axis.bar(
            x,
            means,
            yerr=errors,
            width=0.72,
            color=[CONFIGURATION_COLORS[name] for name in CONFIGURATIONS],
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
        axis.set_title(DATASET_LABELS[dataset], fontsize=27, pad=9)
        axis.set_xlim(-0.65, len(CONFIGURATIONS) - 0.35)
        axis.set_xticks([])
        axis.set_ylim(0.0, global_top)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(axis="y", labelsize=23, width=1.0, length=5)
        axis.grid(axis="y", color="#d9dde3", linewidth=0.65, alpha=0.65)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    legend_handles = [
        Patch(
            facecolor=CONFIGURATION_COLORS[name],
            edgecolor="black",
            linewidth=0.7,
            label=CONFIGURATION_LABELS[name],
        )
        for name in CONFIGURATIONS
    ]
    legend_handles.append(
        axes[0].errorbar(
            [np.nan], [np.nan], yerr=[1.0], fmt="none",
            ecolor="black", elinewidth=1.2, capsize=4.0, capthick=1.2,
            label=r"$\pm2$ SE",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=20,
        columnspacing=1.0,
        handlelength=1.7,
        handletextpad=0.55,
    )
    figure.supylabel(
        "Mean policy wall-clock runtime (s)", fontsize=27, x=0.012, y=0.59
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, top=0.93, bottom=0.31, wspace=0.16
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        "three_objective_direction_policy_runtime_"
        "fixed_30pct_entries_20seed"
    )
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.10}
    figure.savefig(png_path, dpi=300, **options)
    figure.savefig(pdf_path, **options)
    figure.savefig(svg_path, **options)
    plt.close(figure)
    return png_path, pdf_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.source)
    summary = summarize(rows)
    write_csv(args.output_dir / "direction_runtime_by_seed.csv", rows)
    write_csv(args.output_dir / "direction_runtime_summary.csv", summary)
    (args.output_dir / "direction_runtime_manifest.json").write_text(
        json.dumps({
            "configurations": {
                name: CONFIGURATION_LABELS[name] for name in CONFIGURATIONS
            },
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "entry_budget_fraction": 0.30,
            "entry_budget_rule": (
                "exactly ceil(0.30 * available entries), allowing a shorter "
                "final batch"
            ),
            "runtime_field": "runtime.policy_wall_time_seconds",
            "uncertainty": "mean +/- 2 sample standard errors over 20 matched seeds",
            "caveat": (
                "Policy-only wall-clock comparison at a common 30% entry cap. "
                "The timer begins inside simulate_three_objective_gittins and "
                "excludes data loading, JSON serialization, and Slurm queue "
                "time. Native eta-floor stops are recorded before forced "
                "continuation to the common cap. Runs were independently "
                "scheduled, so compute-node conditions may differ."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in plot_runtime(summary, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
