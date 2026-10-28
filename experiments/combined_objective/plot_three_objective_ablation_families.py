#!/usr/bin/env python3
"""Plot the cost-mechanism and continuation three-objective ablations.

The direction-set panel is produced by
``plot_three_objective_direction_ablation.py``.  This companion script uses
the same alignment, corrected complete-returned-set GD/IGD, typography, and
mean +/- 2 SE convention for the other two planned ablation families.  It
only reads completed result JSON files; it never reruns an experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

import plot_three_objective_direction_ablation as common


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RELATIVE = Path(
    "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    "/gittins_g2_main_figures"
)
DATA_REPOSITORY = ROOT
LINKED_REPOSITORY = ROOT.parent / "agentopt-research"
if not (DATA_REPOSITORY / ANALYSIS_RELATIVE / "source_data").is_dir() and (
    LINKED_REPOSITORY / ANALYSIS_RELATIVE / "source_data"
).is_dir():
    DATA_REPOSITORY = LINKED_REPOSITORY
SOURCE = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "source_data/ablation/three_objective_20seed"
)
OUTPUT_ROOT = (
    DATA_REPOSITORY
    / ANALYSIS_RELATIVE
    / "corrected_gd_igd_complete_returned_set_20260929"
    / "three_objective"
)

CHECKPOINTS = (0.05, 0.10, 0.20, 0.30)
CURVE_LINEWIDTH = 2.4
HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

FAMILIES: dict[str, dict[str, Any]] = {
    "cost_mechanism": {
        "output_dir": "cost_mechanism_ablation",
        "stem": "three_objective_cost_mechanism_ablation_hv_gd_igd",
        "configurations": (
            "q_only_real_cost",
            "q_only_unit_cost",
            "pure_axes_real_cost",
            "pure_axes_unit_cost",
            "four_direction_real_cost",
            "four_direction_unit_cost",
        ),
        "styles": {
            "q_only_real_cost": {
                "color": "tab:blue", "label": "Q-only · real cost",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
            },
            "q_only_unit_cost": {
                "color": "tab:blue", "label": "Q-only · unit cost",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "--",
            },
            "pure_axes_real_cost": {
                "color": "tab:orange", "label": "Pure axes · real cost",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "-",
            },
            "pure_axes_unit_cost": {
                "color": "tab:orange", "label": "Pure axes · unit cost",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "--",
            },
            "four_direction_real_cost": {
                "color": "tab:green", "label": "Four directions · real cost",
                "linewidth": HIGHLIGHT_LINEWIDTH, "linestyle": "-",
            },
            "four_direction_unit_cost": {
                "color": "tab:green", "label": "Four directions · unit cost",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "--",
            },
        },
    },
    "continuation": {
        "output_dir": "continuation_ablation",
        "stem": "three_objective_continuation_ablation_hv_gd_igd",
        "configurations": (
            "four_direction_real_cost",
            "four_direction_fixed_eta",
        ),
        "styles": {
            "four_direction_real_cost": {
                "color": "tab:orange", "label": r"Direction-local $\eta$ decay",
                "linewidth": HIGHLIGHT_LINEWIDTH, "linestyle": "-",
            },
            "four_direction_fixed_eta": {
                "color": "tab:blue", "label": r"Fixed $\eta=1$ (no stop)",
                "linewidth": CURVE_LINEWIDTH, "linestyle": "--",
            },
        },
    },
}

EXPECTED_PROTOCOL = {
    "q_only_real_cost": ("real", "eta_decay"),
    "q_only_unit_cost": ("unit", "eta_decay"),
    "pure_axes_real_cost": ("real", "eta_decay"),
    "pure_axes_unit_cost": ("unit", "eta_decay"),
    "four_direction_real_cost": ("real", "eta_decay"),
    "four_direction_unit_cost": ("unit", "eta_decay"),
    "four_direction_fixed_eta": ("real", "fixed_eta_no_stop"),
}


def validate_run(
    payload: dict[str, Any], configuration: str, dataset: str, seed: int,
) -> None:
    config = payload["config"]
    run = payload["run"]
    cost_mode, continuation_mode = EXPECTED_PROTOCOL[configuration]
    expected = {
        "ablation_configuration": configuration,
        "benchmark": dataset,
        "seed": seed,
        "acquisition_cost_mode": cost_mode,
        "continuation_mode": continuation_mode,
        "direction_scheduler": "round_robin",
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"{field} mismatch for {configuration}/{dataset}/seed-{seed}: "
                f"{config.get(field)!r} != {value!r}"
            )
    expected_selector = f"three_objective_cc_gittins_{configuration}"
    if run.get("selector") != expected_selector:
        raise ValueError(f"selector mismatch for {configuration}/{dataset}/seed-{seed}")
    if not run.get("points"):
        raise ValueError(f"empty trajectory for {configuration}/{dataset}/seed-{seed}")


def result_path(source: Path, configuration: str, dataset: str, seed: int) -> Path:
    return source / configuration / f"seed-{seed}" / dataset / "result.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_family(
    source: Path, output_dir: Path, configurations: tuple[str, ...],
    styles: dict[str, dict[str, str | float]],
) -> None:
    curve_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_columns = {
        checkpoint: int(np.flatnonzero(np.isclose(common.GRID, checkpoint))[0])
        for checkpoint in CHECKPOINTS
    }
    input_hashes: dict[tuple[str, int], dict[str, str]] = {}

    for dataset in common.DATASETS:
        for configuration in configurations:
            print(f"Aggregating {dataset} / {configuration}", flush=True)
            metric_rows = {field: [] for field, _ in common.METRICS}
            for seed in common.SEEDS:
                path = result_path(source, configuration, dataset, seed)
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                validate_run(payload, configuration, dataset, seed)
                hash_key = (dataset, seed)
                hashes = payload["config"]["input_sha256"]
                if hash_key in input_hashes and input_hashes[hash_key] != hashes:
                    raise ValueError(f"input hashes differ across configurations: {path}")
                input_hashes[hash_key] = hashes
                aligned = common.aligned_run(payload["run"])
                for field, _ in common.METRICS:
                    metric_rows[field].append(aligned[field])
                del payload

            for field, label in common.METRICS:
                values = np.asarray(metric_rows[field], dtype=np.float64)
                means, two_se, counts = common.summarize(values)
                for x, mean, spread, count in zip(common.GRID, means, two_se, counts):
                    curve_rows.append({
                        "dataset": dataset,
                        "configuration": configuration,
                        "label": styles[configuration]["label"],
                        "metric": field,
                        "metric_label": label,
                        "cost_fraction": float(x),
                        "mean": None if not np.isfinite(mean) else float(mean),
                        "two_se": None if not np.isfinite(spread) else float(spread),
                        "n_runs": int(count),
                    })
                for checkpoint, column in checkpoint_columns.items():
                    finite = values[:, column][np.isfinite(values[:, column])]
                    count = len(finite)
                    checkpoint_rows.append({
                        "dataset": dataset,
                        "configuration": configuration,
                        "label": styles[configuration]["label"],
                        "cost_fraction": checkpoint,
                        "metric": field,
                        "metric_label": label,
                        "mean": float(np.mean(finite)) if count else None,
                        "two_se": (
                            2.0 * float(np.std(finite, ddof=1)) / np.sqrt(count)
                            if count > 1 else 0.0 if count else None
                        ),
                        "n_runs": count,
                    })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "curve_summary.csv", curve_rows)
    write_csv(output_dir / "checkpoint_summary.csv", checkpoint_rows)


def load_curves(
    path: Path,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["mean"] or not row["two_se"]:
                continue
            key = (row["dataset"], row["configuration"], row["metric"])
            grouped.setdefault(key, []).append((
                float(row["cost_fraction"]),
                float(row["mean"]),
                float(row["two_se"]),
            ))
    output = {}
    for key, rows in grouped.items():
        array = np.asarray(rows, dtype=np.float64)
        output[key] = (array[:, 0], array[:, 1], array[:, 2])
    return output


def plot_family(
    summary: Path, output_stem: Path, configurations: tuple[str, ...],
    styles: dict[str, dict[str, str | float]],
) -> None:
    curves = load_curves(summary)
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f"{value:.0%}")
    compact = FuncFormatter(lambda value, _: f"{value:g}")

    for row, dataset in enumerate(common.DATASETS):
        for column, (field, title) in enumerate(common.METRICS):
            axis = axes[row, column]
            for configuration in configurations:
                xs, means, two_se = curves[(dataset, configuration, field)]
                style = styles[configuration]
                axis.plot(
                    xs, means, color=str(style["color"]),
                    linewidth=float(style["linewidth"]),
                    linestyle=str(style["linestyle"]), zorder=3,
                )
                axis.fill_between(
                    xs, np.maximum(0.0, means - two_se), means + two_se,
                    color=str(style["color"]), alpha=0.08, linewidth=0, zorder=1,
                )
            if row == 0:
                axis.set_title(title, fontsize=28, pad=13)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis="both", labelsize=26, width=1.0, length=5)
            axis.grid(color="#D5D9DE", linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)

    figure.subplots_adjust(
        left=0.085, right=0.985, top=0.925, bottom=0.32,
        wspace=0.30, hspace=0.34,
    )
    figure.text(
        0.535, 0.225, "Percentage of Exhaustive Evaluation Cost",
        ha="center", va="center", fontsize=30,
    )
    row_centers = [
        (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0
        for row in range(2)
    ]
    for row, dataset in enumerate(common.DATASETS):
        figure.text(
            0.020, row_centers[row], common.DATASET_LABELS[dataset],
            ha="center", va="center", rotation=90, fontsize=31,
        )
    handles = [
        Line2D(
            [], [], color=str(styles[configuration]["color"]),
            linewidth=float(styles[configuration]["linewidth"]),
            linestyle=str(styles[configuration]["linestyle"]),
            label=str(styles[configuration]["label"]),
        )
        for configuration in configurations
    ]
    handles.append(Patch(
        facecolor="#777777", alpha=0.14, edgecolor="none", label=r"$\pm2$ SE",
    ))
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
        ncol=3 if len(configurations) > 2 else 3, frameon=False,
        fontsize=22 if len(configurations) > 2 else 24,
        handlelength=2.8, columnspacing=1.1, handletextpad=0.6, labelspacing=0.7,
    )
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.12}
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, **options)
    figure.savefig(output_stem.with_suffix(".pdf"), **options)
    figure.savefig(output_stem.with_suffix(".svg"), **options)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=tuple(FAMILIES) + ("all",), default="all",
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    common.configure_matplotlib()

    names = tuple(FAMILIES) if args.family == "all" else (args.family,)
    for name in names:
        spec = FAMILIES[name]
        configurations = tuple(spec["configurations"])
        styles = spec["styles"]
        output_dir = args.output_root / spec["output_dir"]
        aggregate_family(args.source, output_dir, configurations, styles)
        output_stem = output_dir / spec["stem"]
        plot_family(output_dir / "curve_summary.csv", output_stem, configurations, styles)
        metadata = {
            "family": name,
            "source": str(args.source),
            "configurations": list(configurations),
            "datasets": list(common.DATASETS),
            "seeds": list(common.SEEDS),
            "cost_range": [float(common.GRID[0]), float(common.GRID[-1])],
            "grid_step": 0.005,
            "checkpoints": list(CHECKPOINTS),
            "budget_alignment": (
                "latest completed recommendation at or below each actual-USD target; "
                "retain the final recommendation after a natural policy stop"
            ),
            "uncertainty": "mean +/- 2 sample standard errors over 20 matched seeds",
            "hv_metric": "absolute hypervolume regret",
            "distance_metric": (
                "GD/IGD recomputed from every returned configuration, including "
                "configurations dominated in true evaluation space"
            ),
            "styles": styles,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        print(output_stem.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
