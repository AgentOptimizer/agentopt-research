#!/usr/bin/env python3
"""Plot the matched-seed Gittins ablations at realized USD checkpoints.

The baseline G0 run lives in its original result tree; G1 and later groups
live in the ablation tree.  Only configurations with all 8 x 20 runs are
included by default, so a partially completed SLURM array cannot silently
bias a comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    generational_distance,
    inverted_generational_distance,
    mean_cost_metric_vectors,
    nondominated_indices,
    raw_nondominated_indices,
)
from experiments.combined_objective.run_gittins_ablation_slurm_task import (  # noqa: E402
    BENCHMARKS,
    SEEDS,
)
from experiments.combined_objective.run_two_direction_ablation import (  # noqa: E402
    ABLATION_CONFIGS,
)


BASELINE_ROOT = (
    ROOT
    / "experiments/combined_objective/results"
    / "gauss_radau_8bench_20seed_independent"
)
ABLATION_ROOT = (
    ROOT
    / "experiments/combined_objective/results"
    / "gittins_ablation_8bench_20seed"
)
DEFAULT_OUTPUT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed"
    / "figures_preview/frontier/ablation"
)
CHECKPOINTS = (0.10, 0.30)
CURVE_GRID = np.linspace(0.0, 1.0, 101)
PLOT_CACHE_VERSION = 1
# Columns deliberately group the two benchmark instances with the same role
# count: 2, 3, 4, then 5 roles from left to right.
METRIC_PANEL_ORDER = (
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
CONFIG_LABELS = {
    "g0_gauss_radau": "G0 Radau + accuracy axis",
    "g7_gauss_radau_cost_endpoint": "G1 Radau + cost axis",
    "g2_exact_axes": "G2 exact axes",
    "g3_cost_near_endpoint": "G3 near-cost + accuracy axis",
    "g1_gauss_legendre": "G4 Gauss–Legendre",
    "g4_dense_grid_10": "G5 dense grid (10)",
    "g6_weighted_3_to_1": "G6 3:1 scheduling",
    "g5_global_eta": "G7 synchronized eta",
}
DISPLAY_CONFIG_ORDER = (
    "g0_gauss_radau",
    "g7_gauss_radau_cost_endpoint",
    "g2_exact_axes",
    "g3_cost_near_endpoint",
    "g1_gauss_legendre",
    "g4_dense_grid_10",
    "g6_weighted_3_to_1",
    "g5_global_eta",
)
CONFIG_COLORS = {
    # Keep the proposed method visually dominant and reserve orange for it.
    "g0_gauss_radau": "tab:orange",
    "g1_gauss_legendre": "tab:blue",
    "g2_exact_axes": "tab:purple",
    "g3_cost_near_endpoint": "tab:cyan",
    "g4_dense_grid_10": "tab:brown",
    "g5_global_eta": "tab:pink",
    "g6_weighted_3_to_1": "tab:gray",
    "g7_gauss_radau_cost_endpoint": "#003f5c",
}
METRICS = {
    "relative_hv_regret": "Hypervolume Regret",
    "generational_distance": "Generational Distance",
    "inverted_generational_distance": (
        "Inverted Generational Distance"
    ),
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def result_path(configuration: str, benchmark: str, seed: int) -> Path:
    settings = ABLATION_CONFIGS[configuration]
    pair = str(settings["pair_name"])
    if configuration == "g0_gauss_radau":
        return BASELINE_ROOT / f"seed-{seed}" / pair / benchmark / "result.json"
    return (
        ABLATION_ROOT
        / configuration
        / f"seed-{seed}"
        / pair
        / benchmark
        / "result.json"
    )


def availability() -> dict[str, dict[str, Any]]:
    expected = len(BENCHMARKS) * len(SEEDS)
    inventory = {}
    for configuration in ABLATION_CONFIGS:
        missing = [
            str(result_path(configuration, benchmark, seed).relative_to(ROOT))
            for benchmark in BENCHMARKS
            for seed in SEEDS
            if not result_path(configuration, benchmark, seed).is_file()
        ]
        inventory[configuration] = {
            "expected_run_count": expected,
            "available_run_count": expected - len(missing),
            "complete": not missing,
            "missing": missing,
        }
    return inventory


def _point_metrics(
    point: dict[str, Any], truth_vectors: np.ndarray, truth_front: np.ndarray
) -> dict[str, Any]:
    selected = tuple(map(int, point["selected_arm_indices"]))
    obtained = truth_vectors[np.asarray(selected, dtype=int)] if selected else np.empty((0, 2))
    return {
        "cost_fraction": float(point["cost_fraction"]),
        "cost_usd": float(point["cost_usd"]),
        "selected_arm_indices": selected,
        "relative_hv_regret": float(point["relative_hv_regret"]),
        "generational_distance": generational_distance(obtained, truth_front),
        "inverted_generational_distance": inverted_generational_distance(
            obtained, truth_front
        ),
        "pareto_precision": float(point["pareto_precision"]),
        "pareto_recall": float(point["pareto_recall"]),
    }


def load_runs(configurations: list[str]):
    runs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_truth: dict[str, np.ndarray] = {}
    for configuration in configurations:
        for benchmark in BENCHMARKS:
            for seed in SEEDS:
                path = result_path(configuration, benchmark, seed)
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                run = payload["run"]
                raw = np.asarray(run["raw_truth_vectors"], dtype=np.float64)
                if benchmark not in raw_truth:
                    raw_truth[benchmark] = raw
                elif raw_truth[benchmark].shape != raw.shape or not np.allclose(
                    raw_truth[benchmark], raw, rtol=0.0, atol=1e-15
                ):
                    raise ValueError(f"truth vectors changed across runs for {benchmark}")
                metric_truth, _ = mean_cost_metric_vectors(raw)
                truth_front = metric_truth[nondominated_indices(metric_truth)]
                points = [
                    _point_metrics(point, metric_truth, truth_front)
                    for point in run["points"]
                ]
                points.sort(key=lambda point: point["cost_fraction"])
                runs[(configuration, benchmark)].append(
                    {"seed": seed, "points": points}
                )
    return runs, raw_truth


def write_plot_cache(
    path: Path,
    configurations: list[str],
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    raw_truth: dict[str, np.ndarray],
) -> None:
    payload = {
        "version": PLOT_CACHE_VERSION,
        "configurations": configurations,
        "runs": dict(runs),
        "raw_truth": raw_truth,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_plot_cache(
    path: Path, configurations: list[str]
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, np.ndarray],
]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != PLOT_CACHE_VERSION:
        raise ValueError(f"unsupported plot-cache version in {path}")
    if payload.get("configurations") != configurations:
        raise ValueError(
            "plot-cache configurations do not match the requested complete grid"
        )
    return payload["runs"], payload["raw_truth"]


def latest_point(points: list[dict[str, Any]], checkpoint: float):
    eligible = [point for point in points if point["cost_fraction"] <= checkpoint + 1e-12]
    return eligible[-1] if eligible else None


def write_checkpoint_csv(
    output: Path,
    configurations: list[str],
    runs: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    rows = []
    for configuration in configurations:
        for benchmark in BENCHMARKS:
            for run in runs[(configuration, benchmark)]:
                for checkpoint in CHECKPOINTS:
                    point = latest_point(run["points"], checkpoint)
                    if point is None:
                        continue
                    rows.append(
                        {
                            "configuration": configuration,
                            "benchmark": benchmark,
                            "seed": run["seed"],
                            "target_cost_fraction": checkpoint,
                            "snapshot_cost_fraction": point["cost_fraction"],
                            "snapshot_cost_usd": point["cost_usd"],
                            "relative_hv_regret": point["relative_hv_regret"],
                            "generational_distance": point["generational_distance"],
                            "inverted_generational_distance": point[
                                "inverted_generational_distance"
                            ],
                            "pareto_precision": point["pareto_precision"],
                            "pareto_recall": point["pareto_recall"],
                            "selected_arm_indices": json.dumps(
                                point["selected_arm_indices"], separators=(",", ":")
                            ),
                        }
                    )
    path = output / "ablation_checkpoint_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: plt.Figure, stem: Path, *, dpi: int = 240) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def plot_metric_curves(
    output: Path,
    configurations: list[str],
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    paper_output: Path | None = None,
) -> None:
    colors = {name: CONFIG_COLORS[name] for name in configurations}
    curve_rows: list[dict[str, Any]] = []
    for field, label in METRICS.items():
        # Match the main-text HV figure canvas and typography exactly.
        figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), sharex=True)
        for axis, benchmark in zip(axes.flat, METRIC_PANEL_ORDER):
            for configuration in configurations:
                series = []
                for run in runs[(configuration, benchmark)]:
                    values = []
                    for checkpoint in CURVE_GRID:
                        point = latest_point(run["points"], float(checkpoint))
                        values.append(np.nan if point is None else point[field])
                    series.append(values)
                values = np.asarray(series, dtype=np.float64)
                finite_counts = np.sum(np.isfinite(values), axis=0)
                means = np.divide(
                    np.nansum(values, axis=0),
                    finite_counts,
                    out=np.full(values.shape[1], np.nan),
                    where=finite_counts > 0,
                )
                stds = np.asarray(
                    [
                        np.std(column[np.isfinite(column)], ddof=1)
                        if np.sum(np.isfinite(column)) > 1
                        else 0.0
                        for column in values.T
                    ]
                )
                ci = np.divide(
                    2.0 * stds,
                    np.sqrt(finite_counts),
                    out=np.zeros_like(stds),
                    where=finite_counts > 0,
                )
                for checkpoint, mean, error, count in zip(
                    CURVE_GRID, means, ci, finite_counts
                ):
                    curve_rows.append(
                        {
                            "metric": field,
                            "dataset": benchmark,
                            "configuration": configuration,
                            "cost_fraction": float(checkpoint),
                            "mean": float(mean),
                            "error_2se": float(error),
                            "n": int(count),
                        }
                    )
                axis.plot(
                    CURVE_GRID,
                    means,
                    color=colors[configuration],
                    linewidth=2.4 if configuration == "g0_gauss_radau" else 2.0,
                    label=CONFIG_LABELS[configuration],
                    zorder=10 if configuration == "g0_gauss_radau" else 3,
                )
                axis.fill_between(
                    CURVE_GRID,
                    np.maximum(0.0, means - ci),
                    means + ci,
                    color=colors[configuration],
                    alpha=0.13,
                    linewidth=0,
                )
            axis.set_title(DATASET_LABELS[benchmark], fontsize=23)
            axis.grid(True, alpha=0.3)
            # The main-text 2 x 4 figures show percentage ticks on both rows.
            axis.tick_params(axis="both", labelsize=22, labelbottom=True)
            axis.set_xlim(-0.015, 1.02)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda value, _: f"{value:.0%}")
            )
            axis.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}")
            )
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)
        handles = [
            Line2D(
                [],
                [],
                color=colors[name],
                linewidth=2.4 if name == "g0_gauss_radau" else 2.0,
                label=CONFIG_LABELS[name],
            )
            for name in configurations
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.50, -0.01),
            ncol=5,
            frameon=False,
            fontsize=22,
            columnspacing=0.75,
            handletextpad=0.55,
        )
        handles.append(
            mpl.patches.Patch(
                facecolor="#7a7a7a",
                edgecolor="none",
                alpha=0.20,
                label=r"$\pm 2$ SE",
            )
        )
        # Recreate the legend after adding the uncertainty-band key.
        figure.legends[-1].remove()
        shared_legend = figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.02),
            ncol=5,
            fontsize=22,
            columnspacing=0.75,
            handletextpad=0.55,
            frameon=False,
        )
        shared_legend.set_in_layout(False)
        figure.tight_layout(rect=(0.065, 0.22, 0.985, 0.98))
        shared_xlabel = figure.supxlabel(
            "Percentage of Exhaustive Evaluation Cost",
            fontsize=25,
            x=0.54,
            y=0.16,
        )
        shared_ylabel = figure.supylabel(
            label,
            fontsize=25,
            x=0.025,
            y=0.55,
        )
        if paper_output is not None:
            paper_output.mkdir(parents=True, exist_ok=True)
            paper_stem = paper_output / f"ablation_20seed_{field}_0_100pct"
            figure.savefig(
                paper_stem.with_suffix(".png"), dpi=300, facecolor="white"
            )
            figure.savefig(paper_stem.with_suffix(".pdf"), facecolor="white")
        _save_figure(
            figure,
            output / "metrics" / f"ablation_20seed_{field}_0_100pct",
            dpi=300,
        )

    curve_path = output / "ablation_curve_summary.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)


def _draw_landscape(axis: plt.Axes, truth: np.ndarray) -> None:
    frontier = np.asarray(raw_nondominated_indices(truth), dtype=int)
    frontier = frontier[np.argsort(truth[frontier, 1])]
    axis.scatter(
        truth[:, 1], truth[:, 0], s=18, color="#cbd1d8", alpha=0.68,
        edgecolors="none", zorder=1,
    )
    axis.plot(
        truth[frontier, 1], truth[frontier, 0], color="#343b43", linewidth=1.45,
        marker="o", markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
        zorder=3,
    )
    if np.all(truth[:, 1] > 0.0):
        axis.set_xscale("log")
    axis.grid(color="#d9dde3", linewidth=0.55, alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)


def plot_frequency_frontiers(
    output: Path,
    configurations: list[str],
    runs: dict[tuple[str, str], list[dict[str, Any]]],
    raw_truth: dict[str, np.ndarray],
) -> None:
    base = mpl.colormaps["YlOrBr"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ablation_frequency", base(np.linspace(0.18, 0.95, 256))
    )
    norm = mpl.colors.Normalize(vmin=1, vmax=len(SEEDS))
    for configuration in configurations:
        settings = ABLATION_CONFIGS[configuration]
        for checkpoint in CHECKPOINTS:
            figure, axes = plt.subplots(2, 4, figsize=(24, 14))
            for axis, benchmark in zip(axes.flat, BENCHMARKS):
                truth = raw_truth[benchmark]
                _draw_landscape(axis, truth)
                counts: Counter[int] = Counter()
                snapshot_fractions = []
                available = 0
                for run in runs[(configuration, benchmark)]:
                    point = latest_point(run["points"], checkpoint)
                    if point is None:
                        continue
                    available += 1
                    snapshot_fractions.append(point["cost_fraction"])
                    counts.update(point["selected_arm_indices"])
                if counts:
                    selected = np.asarray(sorted(counts), dtype=int)
                    frequencies = np.asarray([counts[index] for index in selected])
                    axis.scatter(
                        truth[selected, 1], truth[selected, 0], c=frequencies,
                        cmap=cmap, norm=norm, s=86, edgecolors="#725b46",
                        linewidths=0.8, zorder=5,
                    )
                mean_snapshot = float(np.mean(snapshot_fractions)) if snapshot_fractions else math.nan
                axis.set_title(
                    f"{DATASET_LABELS[benchmark]}\n"
                    f"n={available}; mean latest snapshot={mean_snapshot:.1%}",
                    fontsize=16,
                    pad=10,
                )
            figure.supxlabel(
                "Mean deployment cost (USD per query, log scale)", fontsize=19, y=0.055
            )
            figure.supylabel("Mean accuracy", fontsize=19, x=0.018)
            directions = " + ".join(
                f"({a:g}, {b:g})" for a, b in payload_directions(configuration)
            )
            figure.suptitle(
                f"{CONFIG_LABELS[configuration]} recommendations at {checkpoint:.0%} search cost\n"
                f"directions: {directions}; scheduler={settings['direction_scheduler']}; "
                f"eta={settings['eta_decay_schedule']}",
                fontsize=23,
                y=0.985,
            )
            handles = [
                Line2D([], [], marker="o", linestyle="none", color="#cbd1d8",
                       markersize=7, label="All configurations"),
                Line2D([], [], marker="o", linestyle="-", color="#343b43",
                       markerfacecolor="white", markersize=6,
                       label="Full-data Pareto frontier"),
                Line2D([], [], marker="o", linestyle="none", color="#d97706",
                       markeredgecolor="#725b46", markersize=7,
                       label="Recommendations (color = frequency)"),
            ]
            figure.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.40, 0.012),
                ncol=3, frameon=False, fontsize=14,
            )
            figure.subplots_adjust(
                left=0.065, right=0.91, bottom=0.13, top=0.86,
                wspace=0.23, hspace=0.30,
            )
            colorbar_axis = figure.add_axes([0.935, 0.23, 0.012, 0.52])
            colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
            colorbar.set_label(
                f"Recommendation frequency (out of {len(SEEDS)} seeds)", fontsize=15
            )
            colorbar.set_ticks([1, 5, 10, 15, 20])
            colorbar.ax.tick_params(labelsize=12)
            stem = (
                output
                / "by_configuration"
                / configuration
                / f"{configuration}_20seed_frequency_frontiers_{round(checkpoint * 100)}pct"
            )
            _save_figure(figure, stem)


def payload_directions(configuration: str) -> tuple[tuple[float, float], ...]:
    from experiments.combined_objective.run_two_direction_ablation import PAIRS

    return PAIRS[str(ABLATION_CONFIGS[configuration]["pair_name"])]


def write_readme(output: Path, configurations: list[str], inventory: dict[str, Any]) -> None:
    lines = [
        "# Gittins ablation figures",
        "",
        "All plots use matched seeds 42–61 and the latest completed recommendation",
        "snapshot at or below each realized USD search-cost checkpoint.",
        "",
        "Included complete configurations:",
        "",
    ]
    lines.extend(f"- `{name}`: {CONFIG_LABELS[name]}" for name in configurations)
    incomplete = [name for name, item in inventory.items() if not item["complete"]]
    if incomplete:
        lines.extend(["", "Not yet included because the run grid is incomplete:", ""])
        lines.extend(
            f"- `{name}`: {inventory[name]['available_run_count']}/"
            f"{inventory[name]['expected_run_count']} runs"
            for name in incomplete
        )
    lines.extend(
        [
            "",
            "Preprocessed trajectories are stored in `ablation_plot_cache.pkl`.",
            "The exact plot-ready 0--100% means and +/- 2 SE values are stored in",
            "`ablation_curve_summary.csv`; checkpoint-level values are stored in",
            "`ablation_checkpoint_metrics.csv`.",
            "",
            "The 2 x 4 panels group datasets by role count (2, 3, 4, then 5)",
            "from left to right. G0 is drawn at exactly 1.2 times the linewidth",
            "of the other configurations, and both axes retain a small margin",
            "below zero so curves at the origin are not clipped.",
            "For style-only changes, reuse that cache instead of reading all run JSON files:",
            "",
            "```bash",
            ".venv-plot/bin/python experiments/combined_objective/plot_gittins_ablation_20seed.py --strict --metrics-only --reuse-cache",
            "```",
            "",
            "Regenerate from the repository root:",
            "",
            "```bash",
            ".venv-plot/bin/python experiments/combined_objective/plot_gittins_ablation_20seed.py",
            "```",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include configurations with incomplete run grids (diagnostics only).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless every configured ablation has all 8 x 20 results.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Regenerate only the three 2 x 4 metric curves.",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        help="Also save the three metric figures in this manuscript directory.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Load preprocessed trajectories from the output plot cache.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = availability()
    incomplete = [name for name, item in inventory.items() if not item["complete"]]
    if args.strict and incomplete:
        raise SystemExit("incomplete configurations: " + ", ".join(incomplete))
    available_configurations = {
        name
        for name, item in inventory.items()
        if item["complete"] or (args.include_partial and item["available_run_count"])
    }
    configurations = [
        name for name in DISPLAY_CONFIG_ORDER if name in available_configurations
    ]
    if not configurations:
        raise SystemExit("no ablation configuration has usable results")

    cache_path = output / "ablation_plot_cache.pkl"
    if args.reuse_cache:
        try:
            runs, raw_truth = read_plot_cache(cache_path, configurations)
        except (OSError, ValueError, pickle.UnpicklingError) as error:
            raise SystemExit(f"cannot reuse plot cache: {error}") from error
    else:
        runs, raw_truth = load_runs(configurations)
        write_plot_cache(cache_path, configurations, runs, raw_truth)
    write_checkpoint_csv(output, configurations, runs)
    paper_output = (
        None if args.paper_output is None else args.paper_output.resolve()
    )
    plot_metric_curves(output, configurations, runs, paper_output)
    if not args.metrics_only:
        plot_frequency_frontiers(output, configurations, runs, raw_truth)
    (output / "plot_manifest.json").write_text(
        json.dumps(
            {
                "checkpoints": CHECKPOINTS,
                "curve_range": [float(CURVE_GRID[0]), float(CURVE_GRID[-1])],
                "metric_role_columns": [2, 3, 4, 5],
                "metric_panel_order": METRIC_PANEL_ORDER,
                "line_widths": {
                    "other_configurations": 2.0,
                    "g0_gauss_radau": 2.4,
                    "g0_multiplier": 1.2,
                },
                "axis_origin_padding": {"x_min": -0.015, "y_fraction": -0.02},
                "plot_ready_curve_data": "ablation_curve_summary.csv",
                "included_configurations": configurations,
                "inventory": inventory,
            },
            indent=2,
        )
        + "\n"
    )
    write_readme(output, configurations, inventory)
    print(f"Wrote ablation figures to {output}")
    if incomplete:
        print("Excluded incomplete configurations: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
