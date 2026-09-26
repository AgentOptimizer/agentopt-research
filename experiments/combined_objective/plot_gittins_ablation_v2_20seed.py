#!/usr/bin/env python3
"""Plot grouped G0-centered Gittins ablations beside the G2 main figures."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.gittins_ablation_v2 import (  # noqa: E402
    BENCHMARKS,
    CONFIGURATIONS,
    DEFAULT_FIGURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FAMILIES,
    G0_CONFIGURATION,
    SEEDS,
    result_path,
)
from experiments.combined_objective.plot_g2_main_figures import (  # noqa: E402
    evaluation_space,
    score_selection,
)


PANEL_ORDER = (
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
    "restaurant_test": "Restaurant Test",
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD Dev",
    "restaurant_valid": "Restaurant Valid",
    "bing_querylogs": "Bing Query Logs",
    "bird_mini_dev": "BIRD Mini Dev",
}
FAMILY_LABELS = {
    "cost_mechanism": "Cost mechanism",
    "directions": "Directions",
    "continuation": "Continuation mechanism",
    "scheduler": "Scheduler",
}
METRICS = {
    "hv_regret": "Hypervolume Regret",
    "generational_distance": "Generational Distance (GD)",
    "inverted_generational_distance": "Inverted Generational Distance (IGD)",
}
CONFIG_COLORS = {
    G0_CONFIGURATION: "#e67e22",
    "g1_q_only_real_cost": "#1f77b4",
    "g2_q_only_unit_cost": "#2ca02c",
    "g3_two_axis_unit_cost": "#e377c2",
    "g4_d_only_real_cost": "#9467bd",
    "g5_axes_midpoint_real_cost": "#8c564b",
    "g6_five_directions_real_cost": "#17becf",
    "g7_fixed_eta_no_stop": "#7f7f7f",
    "g8_q_then_d": "#bcbd22",
    "g9_d_then_q": "#003f5c",
}
CUTOFF = 0.30
GRID = np.linspace(0.0, CUTOFF, 61)
CACHE_VERSION = 1


@dataclass
class MeanCurve:
    x: np.ndarray
    mean: np.ndarray
    two_se: np.ndarray
    count: np.ndarray


def load_runs(
    results_root: Path, configurations: tuple[str, ...]
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]
]:
    runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    spaces: dict[str, Any] = {}
    for configuration in configurations:
        for benchmark in BENCHMARKS:
            dataset_runs = []
            metric_cache: dict[tuple[int, ...], dict[str, float]] = {}
            for seed in SEEDS:
                path = result_path(
                    configuration, benchmark, seed, output_root=results_root
                )
                if not path.is_file():
                    raise FileNotFoundError(f"missing ablation result: {path}")
                payload = json.loads(path.read_text())
                run = payload["run"]
                raw = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
                if benchmark not in spaces:
                    metric_truth, truth_front, cost_reference, truth_hv = evaluation_space(raw)
                    spaces[benchmark] = {
                        "raw_truth": raw,
                        "metric_truth": metric_truth,
                        "truth_front": truth_front,
                        "truth_hv": truth_hv,
                        "cost_reference_usd": cost_reference,
                        "bruteforce_search_cost_usd": float(
                            run["bruteforce_search_cost_usd"]
                        ),
                    }
                else:
                    stored = spaces[benchmark]["raw_truth"]
                    if stored.shape != raw.shape or not np.allclose(
                        stored, raw, rtol=0.0, atol=1e-15
                    ):
                        raise ValueError(
                            f"{configuration}/{benchmark}/seed-{seed}: truth changed"
                        )
                space = spaces[benchmark]
                points = []
                for point in run["points"]:
                    selected = tuple(sorted(map(int, point["selected_arm_indices"])))
                    metrics = metric_cache.get(selected)
                    if metrics is None:
                        metrics = score_selection(
                            selected,
                            space["metric_truth"],
                            space["truth_front"],
                            space["truth_hv"],
                        )
                        metric_cache[selected] = metrics
                    points.append(
                        {
                            "cost_fraction": float(point["cost_fraction"]),
                            "cost_usd": float(point["cost_usd"]),
                            "selected_arm_indices": selected,
                            **metrics,
                        }
                    )
                points.sort(key=lambda item: item["cost_fraction"])
                if not points:
                    raise ValueError(f"{path}: no recommendation points")
                observed_terminal = float(points[-1]["cost_fraction"])
                if observed_terminal < 1.0 - 1e-12:
                    synthetic = points[-1].copy()
                    synthetic.update(
                        cost_fraction=1.0,
                        cost_usd=float(space["bruteforce_search_cost_usd"]),
                        synthetic_terminal=True,
                    )
                    points.append(synthetic)
                dataset_runs.append(
                    {
                        "seed": seed,
                        "points": points,
                        "observed_terminal_cost_fraction": observed_terminal,
                        "stop_reason": run["stop_reason"],
                    }
                )
            runs[(configuration, benchmark)] = dataset_runs
    return runs, spaces


def aggregate(
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    configurations: tuple[str, ...],
) -> dict[tuple[str, str, str], MeanCurve]:
    curves = {}
    for configuration in configurations:
        for benchmark in BENCHMARKS:
            dataset_runs = runs[(configuration, benchmark)]
            starts = np.asarray(
                [float(run["points"][0]["cost_fraction"]) for run in dataset_runs]
            )
            mean_start = float(np.mean(starts))
            grid = np.unique(
                np.concatenate(
                    ([mean_start], GRID[(GRID > mean_start) & (GRID <= CUTOFF + 1e-12)])
                )
            )
            for metric in METRICS:
                aligned = np.empty((len(dataset_runs), len(grid)), dtype=np.float64)
                for row_index, run in enumerate(dataset_runs):
                    xs = np.asarray(
                        [float(point["cost_fraction"]) for point in run["points"]]
                    )
                    ys = np.asarray([float(point[metric]) for point in run["points"]])
                    positions = np.searchsorted(xs, grid, side="right") - 1
                    positions = np.clip(positions, 0, len(xs) - 1)
                    aligned[row_index] = ys[positions]
                curves[(configuration, benchmark, metric)] = MeanCurve(
                    x=grid,
                    mean=np.mean(aligned, axis=0),
                    two_se=(
                        2.0
                        * np.std(aligned, axis=0, ddof=1)
                        / math.sqrt(len(dataset_runs))
                    ),
                    count=np.full(len(grid), len(dataset_runs), dtype=int),
                )
    return curves


def write_curve_csv(
    path: Path,
    curves: dict[tuple[str, str, str], MeanCurve],
    configurations: tuple[str, ...],
) -> None:
    rows = []
    for configuration in configurations:
        for benchmark in PANEL_ORDER:
            for metric in METRICS:
                curve = curves[(configuration, benchmark, metric)]
                for x, mean, two_se, count in zip(
                    curve.x, curve.mean, curve.two_se, curve.count
                ):
                    rows.append(
                        {
                            "configuration": configuration,
                            "dataset": benchmark,
                            "metric": metric,
                            "cost_fraction": float(x),
                            "mean": float(mean),
                            "two_se": float(two_se),
                            "n_runs": int(count),
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run_summary(
    path: Path,
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    configurations: tuple[str, ...],
) -> None:
    rows = []
    for configuration in configurations:
        for benchmark in PANEL_ORDER:
            dataset_runs = runs[(configuration, benchmark)]
            terminals = [
                float(run["observed_terminal_cost_fraction"])
                for run in dataset_runs
            ]
            reasons = sorted({str(run["stop_reason"]) for run in dataset_runs})
            rows.append(
                {
                    "configuration": configuration,
                    "dataset": benchmark,
                    "n_runs": len(dataset_runs),
                    "terminal_min": min(terminals),
                    "terminal_median": statistics.median(terminals),
                    "terminal_mean": statistics.fmean(terminals),
                    "terminal_max": max(terminals),
                    "stopped_before_30pct": sum(value < CUTOFF for value in terminals),
                    "stop_reasons": json.dumps(reasons, separators=(",", ":")),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint_csv(
    path: Path,
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    configurations: tuple[str, ...],
) -> None:
    rows = []
    for configuration in configurations:
        for benchmark in PANEL_ORDER:
            for run in runs[(configuration, benchmark)]:
                for checkpoint in (0.10, 0.30):
                    eligible = [
                        point
                        for point in run["points"]
                        if float(point["cost_fraction"]) <= checkpoint + 1e-12
                    ]
                    if not eligible:
                        continue
                    point = eligible[-1]
                    rows.append(
                        {
                            "configuration": configuration,
                            "dataset": benchmark,
                            "seed": run["seed"],
                            "target_cost_fraction": checkpoint,
                            "snapshot_cost_fraction": point["cost_fraction"],
                            "snapshot_cost_usd": point["cost_usd"],
                            "observed_terminal_cost_fraction": run[
                                "observed_terminal_cost_fraction"
                            ],
                            "selected_arm_indices": json.dumps(
                                point["selected_arm_indices"], separators=(",", ":")
                            ),
                            **{metric: point[metric] for metric in METRICS},
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_family_metric(
    output: Path,
    family: str,
    configurations: tuple[str, ...],
    metric: str,
    curves: dict[tuple[str, str, str], MeanCurve],
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    for axis, benchmark in zip(axes.flat, PANEL_ORDER):
        for configuration in configurations:
            curve = curves[(configuration, benchmark, metric)]
            color = CONFIG_COLORS[configuration]
            is_g0 = configuration == G0_CONFIGURATION
            axis.plot(
                curve.x,
                curve.mean,
                color=color,
                linewidth=2.8 if is_g0 else 2.0,
                zorder=10 if is_g0 else 3,
            )
            axis.fill_between(
                curve.x,
                np.maximum(0.0, curve.mean - curve.two_se),
                curve.mean + curve.two_se,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(DATASET_LABELS[benchmark], fontsize=25, pad=9)
        axis.set_xlim(-0.0045, 0.306)
        y_top = axis.get_ylim()[1]
        axis.set_ylim(-0.02 * y_top, y_top)
        axis.set_xticks((0.0, 0.1, 0.2, 0.3))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.tick_params(axis="both", labelsize=23, width=1.0, length=5)
        axis.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            color=CONFIG_COLORS[name],
            linewidth=2.8 if name == G0_CONFIGURATION else 2.0,
            label=str(CONFIGURATIONS[name]["label"]),
        )
        for name in configurations
    ]
    handles.append(
        mpl.patches.Patch(
            facecolor="#777777", alpha=0.18, edgecolor="none", label=r"$\pm 2$ SE"
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=len(handles),
        frameon=False,
        fontsize=23,
        columnspacing=0.6,
        handlelength=1.6,
        handletextpad=0.5,
    )
    figure.supxlabel(
        "Percentage of Exhaustive Evaluation Cost", fontsize=27, x=0.54, y=0.175
    )
    figure.supylabel(METRICS[metric], fontsize=27, x=0.010, y=0.585)
    figure.suptitle(FAMILY_LABELS[family], fontsize=29, y=1.055)
    figure.subplots_adjust(
        left=0.08, right=0.985, top=0.93, bottom=0.30, wspace=0.34, hspace=0.42
    )
    stem = output / "curves" / family / f"{family}_{metric}_0_30pct"
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILIES),
        default=None,
        help="Plot only these ablation families as soon as their runs finish.",
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    families = tuple(FAMILIES) if args.families is None else tuple(args.families)
    requested = {
        configuration
        for family in families
        for configuration in FAMILIES[family]
    }
    configurations = tuple(
        configuration
        for configuration in CONFIGURATIONS
        if configuration in requested
    )
    is_full_plot = families == tuple(FAMILIES)
    suffix = "" if is_full_plot else "_" + "_".join(families)
    cache_path = output / f"ablation_plot_cache{suffix}.pkl"
    if args.reuse_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            payload.get("version") != CACHE_VERSION
            or tuple(payload.get("configurations", ())) != configurations
        ):
            raise ValueError(f"unsupported cache version in {cache_path}")
        runs = payload["runs"]
        spaces = payload["spaces"]
        curves = payload["curves"]
    else:
        runs, spaces = load_runs(results_root, configurations)
        curves = aggregate(runs, configurations)
        write_cache(
            cache_path,
            {
                "version": CACHE_VERSION,
                "configurations": configurations,
                "runs": runs,
                "spaces": spaces,
                "curves": curves,
            },
        )

    configure_matplotlib()
    curve_name = f"ablation_curve_summary{suffix}.csv"
    terminal_name = f"ablation_terminal_summary{suffix}.csv"
    checkpoint_name = f"ablation_checkpoint_metrics{suffix}.csv"
    write_curve_csv(output / curve_name, curves, configurations)
    write_run_summary(output / terminal_name, runs, configurations)
    write_checkpoint_csv(output / checkpoint_name, runs, configurations)
    for family in families:
        family_configurations = FAMILIES[family]
        for metric in METRICS:
            plot_family_metric(
                output, family, family_configurations, metric, curves
            )

    manifest = {
        "baseline": G0_CONFIGURATION,
        "baseline_source_configuration": "g2_exact_axes",
        "baseline_reused_without_rerun": True,
        "configurations": {
            name: CONFIGURATIONS[name] for name in configurations
        },
        "families": {name: FAMILIES[name] for name in families},
        "datasets": PANEL_ORDER,
        "seeds": SEEDS,
        "displayed_cost_fraction_range": [0.0, CUTOFF],
        "recommendation_output_semantics": (
            "last published recommendation carried forward to 100% without evaluations"
        ),
        "metric_space": "same shared reciprocal-cost desirability space as G2 main figures",
        "uncertainty": "mean +/- 2 standard errors over 20 matched seeds",
        "all_displayed_curve_points_have_n_runs": len(SEEDS),
        "plot_ready_curve_data": curve_name,
        "checkpoint_data": checkpoint_name,
        "terminal_data": terminal_name,
    }
    (output / f"plot_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if is_full_plot:
        (output / "README.md").write_text(
            "# Gittins ablations around G0\n\n"
            "G0 is the current paper method previously stored as `g2_exact_axes`: "
            "two exact axes, real per-arm continuation cost, asynchronous eta decay, "
            "and 1:1 interleaving. The plots are grouped into cost mechanism, "
            "directions, continuation, and scheduler comparisons. Overlapping controls "
            "are run once and reused across panels. Curves show 0--30% of exhaustive "
            "USD evaluation cost and carry a stopped run's last recommendation forward "
            "as an output only; no synthetic point is treated as an evaluation.\n",
            encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()
