#!/usr/bin/env python3
"""Draw the existing two Radial-Gittins figures for all six SCOPE datasets.

The visual encodings, labels, metrics, and plotting primitives are reused from
the original HotpotQA/MathQA plotting code. Only the input adapter and the
2-by-3 panel layout differ.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    run_budget_sweep,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    _full_raw_objective_vectors,
    raw_nondominated_indices,
    simulate_radial_gittins,
)
from experiments.combined_objective.plot.plot_multiobjective_method_comparison import (  # noqa: E402
    RANDOM_STYLES,
    _radial_regret_series,
    _random_regret_series,
    write_comparison_figure,
)
from experiments.combined_objective.plot.plot_radial_gittins_eta_stop_archives import (  # noqa: E402
    legend_handles,
)
from experiments.single_objective.offline_selector_sim import load_scope  # noqa: E402


BENCHMARKS = (
    ("bing_querylogs", "Bing Query Logs"),
    ("bird_dev", "BIRD Dev"),
    ("bird_mini_dev", "BIRD Mini Dev"),
    ("restaurant_test", "Restaurant Test"),
    ("restaurant_valid", "Restaurant Valid"),
    ("stackoverflow", "Stack Overflow"),
)


def draw_stop_panel(ax, raw_vectors, recommended, title, stop_cost_fraction):
    """Exact SCOPE-data counterpart of the existing ``draw_panel``."""
    points = np.asarray(raw_vectors, dtype=float)
    truth = set(raw_nondominated_indices(points))
    recommended = set(map(int, recommended))
    true_positive = truth & recommended
    false_positive = recommended - truth
    missed = truth - recommended
    recall = len(true_positive) / len(truth) if truth else 1.0
    precision = len(true_positive) / len(recommended) if recommended else 1.0

    ax.scatter(
        points[:, 1], points[:, 0], s=28, color="#CBD4DF",
        alpha=0.62, edgecolors="none", zorder=1,
    )
    ordered_front = sorted(truth, key=lambda index: points[index, 1])
    ax.plot(
        points[ordered_front, 1], points[ordered_front, 0], color="#65748A",
        marker="o", markersize=4, linewidth=1.8, zorder=2,
    )
    if missed:
        idx = sorted(missed)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="D", s=70,
            color="#D76B27", edgecolor="white", linewidth=0.6, zorder=4,
        )
    if false_positive:
        idx = sorted(false_positive)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="s", s=72,
            color="#22577A", edgecolor="white", linewidth=0.6, zorder=5,
        )
    if true_positive:
        idx = sorted(true_positive)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="*", s=247.5,
            color="#279060", edgecolor="white", linewidth=0.6, zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean deployment cost (USD, log scale)")
    ax.set_ylabel("Accuracy (full data)")
    ax.grid(alpha=0.22)
    ax.set_title(
        f"{title}: stop, cost={stop_cost_fraction:.1%}\n"
        f"Pareto={len(truth)}, recommended={len(recommended)}, "
        f"recall={recall:.2f}, precision={precision:.2f}"
    )


def write_stop_figure(records, output):
    ncols = 2 if len(records) <= 2 else 3
    nrows = int(np.ceil(len(records) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(7.25 * ncols, 6.8 * nrows), squeeze=False
    )
    for ax, record in zip(axes.ravel(), records):
        draw_stop_panel(
            ax,
            record["raw_vectors"],
            record["recommended"],
            record["name"],
            record["stop_cost_fraction"],
        )
    for ax in axes.ravel()[len(records):]:
        ax.remove()
    figure.legend(
        handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.suptitle("Radial-Gittins recommendations at stopping", fontsize=18)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}")


def _trajectory_series(runs, field, x_axis):
    trajectories = []
    for run in runs:
        trajectory = run.recommendation_trajectory
        if x_axis == "cost":
            full_cost = trajectory[-1].cumulative_search_cost_usd
            xs = np.asarray(
                [point.cumulative_search_cost_usd / full_cost for point in trajectory]
            )
        else:
            xs = np.asarray([point.budget_fraction for point in trajectory])
        ys = np.asarray([getattr(point, field) for point in trajectory])
        trajectories.append((xs, ys))
    return _radial_regret_series(trajectories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", type=Path, default=ROOT / "data/scope")
    parser.add_argument("--outdir", type=Path, default=ROOT / "analysis/vs/scope")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4, choices=(4, 8))
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--z-padding-extra", type=float, default=0.0)
    parser.add_argument("--x-axis", choices=("cost", "cells"), default="cost")
    parser.add_argument(
        "--benchmark", choices=tuple(slug for slug, _ in BENCHMARKS), default=None,
        help="Run and cache only one SCOPE dataset.",
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="Read all per-dataset caches and render the two combined figures.",
    )
    parser.add_argument(
        "--allow-missing-cache", action="store_true",
        help="With --render-only, render available datasets and skip missing caches.",
    )
    args = parser.parse_args()
    scope_dir = args.scope_dir if args.scope_dir.is_absolute() else ROOT / args.scope_dir
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    def cache_path(slug):
        return outdir / "cache" / f"{slug}_seed-{args.seed}.npz"

    def load_cached(slug, title):
        path = cache_path(slug)
        with np.load(path, allow_pickle=False) as data:
            record = {
                "name": title,
                "raw_vectors": data["raw_vectors"],
                "recommended": data["recommended"],
                "stop_cost_fraction": float(data["stop_cost_fraction"]),
            }
            deployable = tuple(data[f"deployable_{key}"] for key in ("x", "y", "ci", "n"))
            provisional = tuple(data[f"provisional_{key}"] for key in ("x", "y", "ci", "n"))
            random_curves = [
                (
                    version,
                    data[f"{version}_x"],
                    data[f"{version}_y"],
                    data[f"{version}_ci"],
                )
                for version in RANDOM_STYLES
            ]
            panel = {
                "name": title,
                "stop_mean": float(data["stop_mean"]),
                "deployable": deployable,
                "provisional": provisional,
                "random": random_curves,
            }
        return record, panel

    if args.render_only:
        render_benchmarks = BENCHMARKS
        if args.allow_missing_cache:
            render_benchmarks = tuple(
                (slug, title) for slug, title in BENCHMARKS
                if cache_path(slug).is_file()
            )
            missing = [
                slug for slug, _ in BENCHMARKS if not cache_path(slug).is_file()
            ]
            if missing:
                print(f"skipping missing caches: {', '.join(missing)}")
        if not render_benchmarks:
            parser.error("no per-dataset caches are available to render")
        stop_records, panels = zip(
            *(load_cached(slug, title) for slug, title in render_benchmarks)
        )
        write_stop_figure(
            stop_records, outdir / "radial_gittins_stop_recommendations.png"
        )
        write_comparison_figure(
            out_path=outdir / "radial_gittins_vs_random_search_hv_regret.png",
            title="Completed-only Gittins from stopping vs random search",
            panels=panels,
            seeds=args.seeds,
            seed=args.seed,
            x_axis=args.x_axis,
        )
        return

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13})
    cache = RadialGittinsBoundaryCache()
    panels = []
    stop_records = []
    rows = []
    selected_benchmarks = tuple(
        item for item in BENCHMARKS
        if args.benchmark is None or item[0] == args.benchmark
    )
    for slug, title in selected_benchmarks:
        models, datapoints, table = load_scope(str(scope_dir / slug))
        raw_vectors = _full_raw_objective_vectors(models, datapoints, table)
        full_matrix_cost = float(sum(sample.cost for row in table.values() for sample in row.values()))
        radial_runs = []
        for seed in range(args.seed, args.seed + args.seeds):
            print(f"=== {title}: {len(models)} configurations, seed={seed} ===")
            radial_runs.append(
                simulate_radial_gittins(
                    models,
                    datapoints,
                    table,
                    batch_size=args.batch_size,
                    observation_budget_fraction=1.0,
                    search_cost_scale_eta=args.eta,
                    seed=seed,
                    boundary_grid=RadialGittinsGrid(
                        z_size=args.grid_size,
                        delta_size=args.grid_size,
                        state_size=args.grid_size,
                        boundary_margin_cells=1,
                    ),
                    boundary_cache=cache,
                    boundary_z_padding_extra=args.z_padding_extra,
                    halt_on_gittins_stop=False,
                    record_recommendation_trajectory=True,
                    question_universe="common",
                )
            )
        radial_by_seed = {run.seed: run for run in radial_runs}
        stop_fractions = []
        for run in radial_runs:
            if run.gittins_stop_budget_fraction is None:
                continue
            stop_fractions.append(
                run.gittins_stop_cost_usd / full_matrix_cost
                if args.x_axis == "cost"
                else run.gittins_stop_budget_fraction
            )
        stop_mean = float(np.mean(stop_fractions)) if stop_fractions else None
        first = radial_runs[0]
        stop_point = next(
            (point for point in first.recommendation_trajectory if point.event == "gittins_stop"),
            first.recommendation_trajectory[-1],
        )
        stop_records.append(
            {
                "name": title,
                "raw_vectors": raw_vectors,
                "recommended": stop_point.deployable_online_raw_archive_arm_indices,
                "stop_cost_fraction": (
                    stop_point.cumulative_search_cost_usd / full_matrix_cost
                ),
            }
        )

        deployable = _trajectory_series(radial_runs, "deployable_hypervolume_regret", args.x_axis)
        provisional = _trajectory_series(radial_runs, "hypervolume_regret", args.x_axis)
        random_results = run_budget_sweep(
            models,
            datapoints,
            table,
            seeds=range(args.seed, args.seed + args.seeds),
        )
        random_curves = []
        for version in RANDOM_STYLES:
            xs, means, ci95 = _random_regret_series(
                random_results,
                version=version,
                radial_by_seed=radial_by_seed,
                x_axis=args.x_axis,
            )
            random_curves.append((version, xs, means, ci95))
            for x, mean, ci in zip(xs, means, ci95):
                rows.append((title, version, x, mean, ci))
        panels.append(
            {
                "name": title,
                "stop_mean": stop_mean,
                "deployable": deployable,
                "provisional": provisional,
                "random": random_curves,
            }
        )
        for method, series in (
            ("radial_gittins_deployable", deployable),
            ("radial_gittins_provisional", provisional),
        ):
            xs, means, cis, _ = series
            rows.extend((title, method, x, mean, ci) for x, mean, ci in zip(xs, means, cis))

        cache_path(slug).parent.mkdir(parents=True, exist_ok=True)
        cache_values = {
            "raw_vectors": raw_vectors,
            "recommended": np.asarray(
                stop_point.deployable_online_raw_archive_arm_indices, dtype=np.int64
            ),
            "stop_cost_fraction": stop_point.cumulative_search_cost_usd / full_matrix_cost,
            "stop_mean": stop_mean if stop_mean is not None else np.nan,
            "policy_wall_time_seconds": np.asarray(
                [run.policy_wall_time_seconds for run in radial_runs]
            ),
        }
        for prefix, series in (("deployable", deployable), ("provisional", provisional)):
            for key, values in zip(("x", "y", "ci", "n"), series):
                cache_values[f"{prefix}_{key}"] = values
        for version, xs, means, ci95 in random_curves:
            cache_values[f"{version}_x"] = xs
            cache_values[f"{version}_y"] = means
            cache_values[f"{version}_ci"] = ci95
        np.savez_compressed(cache_path(slug), **cache_values)
        print(f"cached {cache_path(slug)}")

    if args.benchmark is not None:
        return

    write_stop_figure(stop_records, outdir / "radial_gittins_stop_recommendations.png")
    write_comparison_figure(
        out_path=outdir / "radial_gittins_vs_random_search_hv_regret.png",
        title="Completed-only Gittins from stopping vs random search",
        panels=panels,
        seeds=args.seeds,
        seed=args.seed,
        x_axis=args.x_axis,
    )
    with (outdir / "method_hv_regret_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "method", "budget_fraction", "mean_hv_regret", "ci95_half_width"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
