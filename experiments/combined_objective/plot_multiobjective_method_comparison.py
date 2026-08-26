#!/usr/bin/env python3
"""Compare Radial-Gittins and both random-search Pareto baselines.

Both figures use the Gittins normalized-desirability hypervolume: accuracy
together with ``C_ref / (C_ref + cost)``, scored against the Gittins reference
point. Random search is rescored in that same space.

``radial_gittins_vs_random_search_hv_regret.png`` shows the completed-only
deployable recommendation (solid) against random search, with provisional
all-posterior regret retained as a dashed diagnostic overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
)
from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    MultiObjectiveRandomSearchResult,
    run_budget_sweep,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    RadialSimulationResult,
    hypervolume_2d,
)
from experiments.combined_objective.plot_radial_gittins_trajectories import (  # noqa: E402
    run_benchmark,
)


PICKLES = {
    "HotpotQA": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "MathQA": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}

RANDOM_STYLES = {
    "random_configurations": ("#7b61a8", "Random configurations"),
    "random_questions": ("#2a9d8f", "Random shared questions"),
}


def _gittins_hv_space(
    result: RadialSimulationResult,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Return the desirability vectors, reference point, and ground-truth HV."""
    if result.truth_vectors is None:
        raise ValueError("RadialSimulationResult.truth_vectors is missing")
    reference = tuple(float(x) for x in result.params["reference_point"])
    if len(reference) != 2:
        raise ValueError("params['reference_point'] must be a length-2 vector")
    return (
        np.asarray(result.truth_vectors, dtype=np.float64),
        (reference[0], reference[1]),
        float(result.ground_truth_hypervolume),
    )


def _selected_regret(
    truth_vectors: np.ndarray,
    selected_indices: Sequence[int],
    ground_truth_hv: float,
    reference: Sequence[float],
) -> float:
    selected_hv = (
        hypervolume_2d(truth_vectors[list(selected_indices)], reference)
        if selected_indices
        else 0.0
    )
    return float(max(0.0, ground_truth_hv - selected_hv))


def _random_regret_series(
    results: Sequence[MultiObjectiveRandomSearchResult],
    *,
    version: str,
    radial_by_seed: Dict[int, RadialSimulationResult],
    x_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[float, list[float]] = defaultdict(list)
    grouped_x: Dict[float, list[float]] = defaultdict(list)
    full_cost_by_seed = {
        result.seed: result.total_search_cost_usd
        for result in results
        if result.version == version and abs(result.budget_fraction - 1.0) <= 1e-12
    }
    for result in results:
        if result.version != version:
            continue
        radial = radial_by_seed[result.seed]
        truth_vectors, reference, ground_truth_hv = _gittins_hv_space(radial)
        grouped[result.budget_fraction].append(
            _selected_regret(
                truth_vectors,
                result.selected_arm_indices,
                ground_truth_hv,
                reference,
            )
        )
        grouped_x[result.budget_fraction].append(
            result.total_search_cost_usd / full_cost_by_seed[result.seed]
            if x_axis == "cost"
            else result.budget_fraction
        )
    checkpoints = sorted(grouped)
    xs = np.asarray([np.mean(grouped_x[x]) for x in checkpoints], dtype=np.float64)
    means = np.asarray([np.mean(grouped[x]) for x in checkpoints], dtype=np.float64)
    ci95 = np.asarray(
        [
            1.96 * np.std(grouped[x], ddof=1) / np.sqrt(len(grouped[x]))
            if len(grouped[x]) > 1
            else 0.0
            for x in checkpoints
        ]
    )
    return xs, means, ci95


def _radial_regret_series(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate irregular trajectories without reducing them to 10% checkpoints.

    At each checkpoint in the union of all runs, a run contributes its most
    recent recommendation. Runs that have not reached their first checkpoint do
    not contribute yet.
    """
    xs = np.unique(np.concatenate([x for x, _ in trajectories]))
    aligned = np.full((len(trajectories), len(xs)), np.nan, dtype=np.float64)
    for row, (run_x, run_y) in enumerate(trajectories):
        positions = np.searchsorted(run_x, xs, side="right") - 1
        available = positions >= 0
        aligned[row, available] = run_y[positions[available]]

    counts = np.sum(np.isfinite(aligned), axis=0)
    means = np.nanmean(aligned, axis=0)
    ci95 = np.zeros_like(means)
    for column, count in enumerate(counts):
        if count > 1:
            ci95[column] = (
                1.96
                * np.nanstd(aligned[:, column], ddof=1)
                / np.sqrt(count)
            )
    return xs, means, ci95, counts


def _draw_stop_markers(ax, stop_mean: float | None, seeds: int) -> None:
    if stop_mean is None:
        return
    ax.axvspan(
        stop_mean,
        1.02,
        color="#687386",
        alpha=0.08,
        label="forced post-stop diagnostic",
    )
    ax.axvline(
        stop_mean,
        color="#c45c26",
        linestyle="--",
        linewidth=1.4,
        label=(
            f"Gittins stop ({stop_mean:.1%})"
            if seeds == 1
            else f"Mean Gittins stop ({stop_mean:.1%})"
        ),
    )


def _plot_regret_line(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    ci95: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str | None = None,
    fill: bool = True,
    zorder: int = 2,
    linewidth: float = 1.9,
) -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        label=label,
        zorder=zorder,
    )
    if fill:
        ax.fill_between(
            xs,
            np.maximum(0.0, ys - ci95),
            ys + ci95,
            color=color,
            alpha=0.13,
            linewidth=0,
        )


def _post_stop_series(
    xs: np.ndarray,
    ys: np.ndarray,
    ci95: np.ndarray,
    stop_fraction: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict a deployable series to its meaningful post-stop portion."""
    if stop_fraction is None:
        return xs, ys, ci95
    keep = xs >= stop_fraction - 1e-12
    return xs[keep], ys[keep], ci95[keep]


def write_comparison_figure(
    *,
    out_path: Path,
    title: str,
    panels: Sequence[dict],
    seeds: int,
    seed: int,
    x_axis: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    for ax, panel in zip(axes, panels):
        _draw_stop_markers(ax, panel["stop_mean"], seeds)
        deployable_x, deployable_y, deployable_ci95, _ = panel["deployable"]
        provisional_x, provisional_y, provisional_ci95, _ = panel["provisional"]
        deployable_x, deployable_y, deployable_ci95 = _post_stop_series(
            deployable_x,
            deployable_y,
            deployable_ci95,
            panel["stop_mean"],
        )
        _plot_regret_line(
            ax,
            deployable_x,
            deployable_y,
            deployable_ci95,
            color="#1f4e79",
            label=(
                "deployable completed-only (post-stop)"
                if seeds == 1
                else f"deployable completed-only post-stop (mean, n={seeds})"
            ),
            zorder=3,
        )
        if len(deployable_x):
            ax.scatter(
                [deployable_x[0]],
                [deployable_y[0]],
                s=90,
                color="#1f4e79",
                marker="o",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
                label=(
                    f"deployable regret at stop ({deployable_y[0]:.4f})"
                    if seeds == 1
                    else f"mean deployable regret at stop ({deployable_y[0]:.4f})"
                ),
            )
        _plot_regret_line(
            ax,
            provisional_x,
            provisional_y,
            provisional_ci95,
            color="#7a8ca5",
            label="provisional all-posterior diagnostic",
            linestyle="--",
            fill=False,
            zorder=2,
            linewidth=1.3,
        )
        for version, xs, means, ci95 in panel["random"]:
            color, base_label = RANDOM_STYLES[version]
            plot_label = (
                f"{base_label} (seed={seed})" if seeds == 1 else base_label
            )
            _plot_regret_line(
                ax,
                xs,
                means,
                ci95,
                color=color,
                label=plot_label,
                marker="o",
                linewidth=1.5,
            )
        ax.set_title(panel["name"])
        ax.set_xlabel(
            "Cumulative search cost fraction"
            if x_axis == "cost"
            else "Observed cell-budget fraction"
        )
        ax.set_ylabel("Normalized-desirability hypervolume regret")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out_path}")


def panels_from_csv(
    csv_path: Path,
    stop_by_benchmark: Dict[str, float | None],
    x_axis: str,
) -> list[dict]:
    grouped: Dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"x": [], "y": [], "ci": [], "n": []}
    )
    stored_stops: Dict[str, float] = {}
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        stored_axis = row.get("x_axis", "cells")
        if stored_axis != x_axis:
            raise ValueError(
                f"{csv_path} stores x_axis={stored_axis!r}, requested {x_axis!r}; "
                "rerun the comparison to generate the requested axis"
            )
        key = (row["benchmark"], row["method"])
        grouped[key]["x"].append(float(row["budget_fraction"]))
        grouped[key]["y"].append(float(row["mean_hv_regret"]))
        grouped[key]["ci"].append(float(row["ci95_half_width"]))
        grouped[key]["n"].append(float(row["n_runs"]))
        if row.get("stop_axis_fraction"):
            stored_stops[row["benchmark"]] = float(row["stop_axis_fraction"])

    def _series(benchmark: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = grouped[(benchmark, method)]
        order = np.argsort(data["x"])
        xs = np.asarray(data["x"], dtype=np.float64)[order]
        ys = np.asarray(data["y"], dtype=np.float64)[order]
        cis = np.asarray(data["ci"], dtype=np.float64)[order]
        counts = np.asarray(data["n"], dtype=np.float64)[order]
        return xs, ys, cis, counts

    panels = []
    for benchmark in PICKLES:
        random_curves = []
        for version in RANDOM_STYLES:
            xs, means, ci95, _ = _series(benchmark, version)
            random_curves.append((version, xs, means, ci95))
        panels.append(
            {
                "name": benchmark,
                "stop_mean": stored_stops.get(
                    benchmark, stop_by_benchmark.get(benchmark)
                ),
                "deployable": _series(benchmark, "radial_gittins_deployable"),
                "provisional": _series(benchmark, "radial_gittins_provisional"),
                "random": random_curves,
            }
        )
    return panels


def write_comparison_output(
    *,
    outdir: Path,
    panels: Sequence[dict],
    seeds: int,
    seed: int,
    x_axis: str,
) -> None:
    write_comparison_figure(
        out_path=outdir / "radial_gittins_vs_random_search_hv_regret.png",
        title="Completed-only Gittins from stopping vs random search",
        panels=panels,
        seeds=seeds,
        seed=seed,
        x_axis=x_axis,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/method_comparison"))
    parser.add_argument("--seed", type=int, default=42, help="First run seed.")
    parser.add_argument(
        "--seeds",
        "--random-seeds",
        dest="seeds",
        type=int,
        default=1,
        help="Number of identical run seeds used by every method.",
    )
    parser.add_argument("--batch-size", type=int, default=4, choices=(4, 8))
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument(
        "--x-axis",
        choices=("cost", "cells"),
        default="cost",
        help=(
            "Horizontal budget axis. 'cost' (default) uses cumulative USD cost "
            "divided by full-matrix cost; 'cells' uses observed matrix cells."
        ),
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Redraw the comparison figure from method_hv_regret_summary.csv.",
    )
    parser.add_argument(
        "--stop-summary",
        type=Path,
        default=None,
        help="JSON with per-benchmark gittins_stop_budget_fraction; used with --from-csv.",
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.from_csv is not None:
        csv_path = args.from_csv if args.from_csv.is_absolute() else ROOT / args.from_csv
        if args.stop_summary is None:
            raise SystemExit("--from-csv requires --stop-summary")
        stop_path = (
            args.stop_summary
            if args.stop_summary.is_absolute()
            else ROOT / args.stop_summary
        )
        summary = json.loads(stop_path.read_text(encoding="utf-8"))
        stop_by_benchmark = {
            name: (
                float(payload["gittins_stop_budget_fraction"])
                if payload.get("gittins_stop_budget_fraction") is not None
                else None
            )
            for name, payload in summary.items()
        }
        write_comparison_output(
            outdir=outdir,
            panels=panels_from_csv(csv_path, stop_by_benchmark, args.x_axis),
            seeds=args.seeds,
            seed=args.seed,
            x_axis=args.x_axis,
        )
        return

    cache = RadialGittinsBoundaryCache()
    rows = []
    panels = []

    for benchmark, pickle_path in PICKLES.items():
        run_seeds = range(args.seed, args.seed + args.seeds)
        radial_runs = []
        raw_vectors = None
        evaluation_questions = None
        for run_seed in run_seeds:
            radial, run_vectors, run_questions = run_benchmark(
                pickle_path=pickle_path,
                name=benchmark,
                batch_size=args.batch_size,
                seed=run_seed,
                grid_size=args.grid_size,
                cache=cache,
            )
            radial_runs.append(radial)
            if raw_vectors is None:
                raw_vectors = run_vectors
                evaluation_questions = run_questions
            elif not np.allclose(raw_vectors, run_vectors):
                raise ValueError("Ground-truth vectors changed across run seeds")
        assert raw_vectors is not None and evaluation_questions is not None
        radial_by_seed = {run.seed: run for run in radial_runs}
        cost_reference = float(radial_runs[0].cost_reference_usd)

        def _series_from_field(field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            trajectories = []
            for radial in radial_runs:
                if args.x_axis == "cost":
                    full_cost = radial.recommendation_trajectory[-1].cumulative_search_cost_usd
                    run_x = np.asarray(
                        [
                            point.cumulative_search_cost_usd / full_cost
                            for point in radial.recommendation_trajectory
                        ]
                    )
                else:
                    run_x = np.asarray(
                        [point.budget_fraction for point in radial.recommendation_trajectory]
                    )
                run_y = np.asarray(
                    [getattr(point, field) for point in radial.recommendation_trajectory]
                )
                trajectories.append((run_x, run_y))
            return _radial_regret_series(trajectories)

        deployable_x, deployable_y, deployable_ci95, deployable_counts = (
            _series_from_field("deployable_hypervolume_regret")
        )
        provisional_x, provisional_y, provisional_ci95, provisional_counts = (
            _series_from_field("hypervolume_regret")
        )

        stop_fractions = np.asarray([
            (
                run.gittins_stop_cost_usd
                / run.recommendation_trajectory[-1].cumulative_search_cost_usd
                if args.x_axis == "cost"
                else run.gittins_stop_budget_fraction
            )
            for run in radial_runs
            if run.gittins_stop_budget_fraction is not None
            and (args.x_axis != "cost" or run.gittins_stop_cost_usd is not None)
        ], dtype=np.float64)
        stop_mean = float(np.mean(stop_fractions)) if len(stop_fractions) else None

        for method, xs, ys, cis, counts in (
            (
                "radial_gittins_deployable",
                deployable_x,
                deployable_y,
                deployable_ci95,
                deployable_counts,
            ),
            (
                "radial_gittins_provisional",
                provisional_x,
                provisional_y,
                provisional_ci95,
                provisional_counts,
            ),
        ):
            for x, mean, ci, count in zip(xs, ys, cis, counts):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "budget_fraction": x,
                        "mean_hv_regret": mean,
                        "ci95_half_width": ci,
                        "n_runs": int(count),
                        "cost_reference_usd": cost_reference,
                        "x_axis": args.x_axis,
                        "stop_axis_fraction": stop_mean,
                    }
                )

        random_results = run_budget_sweep(
            list(radial_runs[0].model_results[i].model_name for i in range(len(raw_vectors))),
            list(evaluation_questions),
            load_table_from_pickle(pickle_path),
            seeds=run_seeds,
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
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": version,
                        "budget_fraction": x,
                        "mean_hv_regret": mean,
                        "ci95_half_width": ci,
                        "n_runs": args.seeds,
                        "cost_reference_usd": cost_reference,
                        "x_axis": args.x_axis,
                        "stop_axis_fraction": stop_mean,
                    }
                )

        ax.set_title(benchmark)
        ax.set_xlabel("Fraction of brute-force search cost")
        ax.set_ylabel("Hypervolume regret")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)

    write_comparison_output(
        outdir=outdir,
        panels=panels,
        seeds=args.seeds,
        seed=args.seed,
        x_axis=args.x_axis,
    )

    csv_path = outdir / "method_hv_regret_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


def load_table_from_pickle(pickle_path: Path):
    """Load only the lookup table while keeping the plotting loop readable."""
    from experiments.single_objective.offline_selector_sim import load_pickle

    _, _, table = load_pickle(str(pickle_path))
    return table


if __name__ == "__main__":
    main()
