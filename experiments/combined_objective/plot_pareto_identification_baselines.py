"""Plot HV regret, GD, and IGD for Pareto identification baselines.

New methods run to the full cell budget. Radial-Gittins is overlaid from a
saved ``cost_trajectory.csv`` so the three-metric panel matches the existing
Gittins trajectory figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.pareto_identification import (  # noqa: E402
    APE_K,
    EGE_SH,
    EGE_SR,
)
from experiments.combined_objective.offline_pareto_baselines import (  # noqa: E402
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    _require_data_path,
    load_pickle,
)


PICKLES = {
    "HotpotQA": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "MathQA": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}

STYLES = {
    "radial_gittins": ("#1f4e79", "Radial-Gittins"),
    EGE_SH: ("#c45c26", "EGE-SH"),
    EGE_SR: ("#e09f3e", "EGE-SR"),
    APE_K: ("#2a9d8f", "APE-k"),
    QNEHVI: ("#7b61a8", "qNEHVI"),
}

METRICS = (
    ("hv_regret", "Normalized-desirability hypervolume regret"),
    ("generational_distance", "Generational distance (GD)"),
    ("inverted_generational_distance", "Inverted generational distance (IGD)"),
)

GITTINS_STOP = {
    "HotpotQA": 0.061873479670447845,
    "MathQA": 0.043713619812500715,
}


def _align_series(
    trajectories: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(np.concatenate([x for x, _ in trajectories]))
    aligned = np.full((len(trajectories), len(xs)), np.nan, dtype=np.float64)
    for row, (run_x, run_y) in enumerate(trajectories):
        positions = np.searchsorted(run_x, xs, side="right") - 1
        available = positions >= 0
        aligned[row, available] = run_y[positions[available]]
    means = np.nanmean(aligned, axis=0)
    counts = np.sum(np.isfinite(aligned), axis=0)
    ci95 = np.zeros_like(means)
    for column, count in enumerate(counts):
        if count > 1:
            ci95[column] = (
                1.96 * np.nanstd(aligned[:, column], ddof=1) / np.sqrt(count)
            )
    return xs, means, ci95


def _downsample(
    xs: np.ndarray, ys: np.ndarray, *, max_points: int = 800,
) -> Tuple[np.ndarray, np.ndarray]:
    if xs.size <= max_points:
        return xs, ys
    keep = np.unique(np.linspace(0, xs.size - 1, max_points).astype(int))
    return xs[keep], ys[keep]


def load_gittins_metric(
    path: Path,
    benchmark: str,
    metric: str,
    *,
    archive_scope: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("benchmark") != benchmark:
                continue
            if "method" in row and row["method"] not in ("", "radial_gittins"):
                continue
            if archive_scope is not None and row.get("archive_scope") != archive_scope:
                continue
            if metric not in row or row[metric] == "":
                continue
            xs.append(float(row["budget_fraction"]))
            ys.append(float(row[metric]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _plot_line(ax, xs, ys, *, color, label, linestyle="-", linewidth=1.9) -> None:
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def plot_metric_grid(
    *,
    out_path: Path,
    panels: Sequence[dict],
    gittins_path: Path,
    seeds: int,
) -> None:
    figure, axes = plt.subplots(
        len(METRICS),
        len(panels),
        figsize=(5.6 * len(panels), 3.55 * len(METRICS)),
        sharex=True,
        squeeze=False,
    )
    for col, panel in enumerate(panels):
        stop = GITTINS_STOP.get(panel["name"])
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            gx, gy = load_gittins_metric(
                gittins_path, panel["name"], metric, archive_scope="provisional",
            )
            if len(gx):
                _plot_line(
                    ax, gx, gy,
                    color="#7a8ca5",
                    linestyle="--",
                    linewidth=1.3,
                    label="Radial-Gittins provisional",
                )
            gx, gy = load_gittins_metric(
                gittins_path, panel["name"], metric, archive_scope="deployable",
            )
            if len(gx):
                _plot_line(
                    ax, gx, gy,
                    color="#1f4e79",
                    label="Radial-Gittins deployable",
                )
                ax.scatter(
                    [gx[0]], [gy[0]],
                    s=70, color="#1f4e79", marker="o",
                    edgecolors="white", linewidths=0.8, zorder=5,
                )
            if stop is not None:
                ax.axvline(
                    stop, color="#c45c26", linestyle="--", linewidth=1.2,
                    label=f"Gittins stop ({stop:.1%})",
                )
                ax.axvspan(stop, 1.02, color="#687386", alpha=0.06)
            for method, series_by_metric in panel["methods"]:
                xs, ys, ci95 = series_by_metric[metric]
                if not len(xs):
                    continue
                color, label = STYLES[method]
                _plot_line(ax, xs, ys, color=color, label=label)
                if seeds > 1 and len(ci95):
                    xs_d, lo = _downsample(xs, np.maximum(0.0, ys - ci95))
                    _, hi = _downsample(xs, ys + ci95)
                    ax.fill_between(xs_d, lo, hi, color=color, alpha=0.12, linewidth=0)
            if row == 0:
                ax.set_title(panel["name"])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Fraction of brute-force search cost")
            ax.set_xlim(0.0, 1.02)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == len(panels) - 1:
                ax.legend(loc="upper right", fontsize=7.2)
    figure.suptitle(
        "HV regret, GD, and IGD for full-budget Pareto identification baselines",
        fontsize=13,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_single_metric(
    *,
    out_path: Path,
    panels: Sequence[dict],
    gittins_path: Path,
    metric: str,
    ylabel: str,
    title: str,
    seeds: int,
) -> None:
    figure, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.15), squeeze=False,
    )
    for ax, panel in zip(axes[0], panels):
        stop = GITTINS_STOP.get(panel["name"])
        gx, gy = load_gittins_metric(
            gittins_path, panel["name"], metric, archive_scope="provisional",
        )
        if len(gx):
            _plot_line(
                ax, gx, gy, color="#7a8ca5", linestyle="--",
                linewidth=1.3, label="Radial-Gittins provisional",
            )
        gx, gy = load_gittins_metric(
            gittins_path, panel["name"], metric, archive_scope="deployable",
        )
        if len(gx):
            _plot_line(ax, gx, gy, color="#1f4e79", label="Radial-Gittins deployable")
        if stop is not None:
            ax.axvline(
                stop, color="#c45c26", linestyle="--", linewidth=1.2,
                label=f"Gittins stop ({stop:.1%})",
            )
        for method, series_by_metric in panel["methods"]:
            xs, ys, ci95 = series_by_metric[metric]
            if not len(xs):
                continue
            color, label = STYLES[method]
            _plot_line(ax, xs, ys, color=color, label=label)
            if seeds > 1 and len(ci95):
                xs_d, lo = _downsample(xs, np.maximum(0.0, ys - ci95))
                _, hi = _downsample(xs, ys + ci95)
                ax.fill_between(xs_d, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_title(panel["name"])
        ax.set_xlabel("Fraction of brute-force search cost")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out_path}")


def _series_from_rows(
    rows: Iterable[dict], *, method: str, benchmark: str, metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["method"] != method or row["benchmark"] != benchmark:
            continue
        grouped[int(row["seed"])].append(
            (float(row["budget_fraction"]), float(row[metric]))
        )
    trajectories = []
    for seed_rows in grouped.values():
        seed_rows.sort()
        xs = np.asarray([x for x, _ in seed_rows], dtype=np.float64)
        ys = np.asarray([y for _, y in seed_rows], dtype=np.float64)
        trajectories.append((xs, ys))
    if not trajectories:
        return np.asarray([]), np.asarray([]), np.asarray([])
    return _align_series(trajectories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[EGE_SH, APE_K],
        choices=[EGE_SH, EGE_SR, APE_K, QNEHVI],
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--qnehvi-refit-every", type=int, default=32)
    parser.add_argument(
        "--trajectory-checkpoint-interval",
        type=int,
        default=10,
        help=(
            "Retain one baseline trajectory point every N policy batches "
            "(default: %(default)s; use 1 for the legacy full trace)"
        ),
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Redraw from an existing cost_trajectory.csv instead of replaying.",
    )
    parser.add_argument(
        "--gittins-trajectory",
        type=Path,
        default=ROOT / "experiments/combined_objective/results/radial_gittins_plots/cost_trajectory.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "experiments/combined_objective/results/pareto_baselines",
    )
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    gittins_path = (
        args.gittins_trajectory
        if args.gittins_trajectory.is_absolute()
        else ROOT / args.gittins_trajectory
    )

    all_rows: List[dict] = []
    if args.from_csv is not None:
        csv_path = args.from_csv if args.from_csv.is_absolute() else ROOT / args.from_csv
        with csv_path.open(encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        methods = list(dict.fromkeys(row["method"] for row in all_rows))
    else:
        methods = list(args.methods)
        for benchmark, pickle_path in PICKLES.items():
            path = Path(_require_data_path(str(pickle_path)))
            models, datapoints, table = load_pickle(str(path))
            for method in methods:
                for offset in range(args.seeds):
                    seed = args.base_seed + offset
                    print(f"\n=== {benchmark} {method} seed={seed} ===")
                    result = simulate_pareto_baseline(
                        models,
                        datapoints,
                        table,
                        method=method,
                        seed=seed,
                        batch_size=args.batch_size,
                        qnehvi_refit_every=args.qnehvi_refit_every,
                        recommendation_checkpoint_interval=(
                            args.trajectory_checkpoint_interval
                        ),
                    )
                    print(
                        f"regret={result.hypervolume_regret:.5f} "
                        f"gd={result.generational_distance:.5f} "
                        f"igd={result.inverted_generational_distance:.5f} "
                        f"evals={result.total_evaluations} "
                        f"wall={result.policy_wall_time_seconds:.1f}s"
                    )
                    full_cost = float(result.params["bruteforce_search_cost_usd"])
                    for point in result.recommendation_trajectory:
                        all_rows.append(
                            {
                                "benchmark": benchmark,
                                "method": method,
                                "seed": seed,
                                "event": point.event,
                                "budget_fraction": point.cumulative_search_cost_usd / full_cost,
                                "cumulative_evaluations": point.cumulative_evaluations,
                                "cumulative_search_cost_usd": point.cumulative_search_cost_usd,
                                "hv_regret": point.hypervolume_regret,
                                "generational_distance": point.generational_distance,
                                "inverted_generational_distance": (
                                    point.inverted_generational_distance
                                ),
                                "archive_size": len(point.selected_arm_indices),
                            }
                        )
        csv_path = outdir / "cost_trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {csv_path}")

    panels = []
    for benchmark in PICKLES:
        methods_series = []
        for method in methods:
            by_metric = {
                metric: _series_from_rows(
                    all_rows, method=method, benchmark=benchmark, metric=metric,
                )
                for metric, _ in METRICS
            }
            methods_series.append((method, by_metric))
        panels.append({"name": benchmark, "methods": methods_series})

    plot_metric_grid(
        out_path=outdir / "front_quality_baselines.png",
        panels=panels,
        gittins_path=gittins_path,
        seeds=args.seeds,
    )
    plot_single_metric(
        out_path=outdir / "hv_regret_baselines.png",
        panels=panels,
        gittins_path=gittins_path,
        metric="hv_regret",
        ylabel="Normalized-desirability hypervolume regret",
        title="Hypervolume regret: Gittins vs full-budget baselines",
        seeds=args.seeds,
    )
    plot_single_metric(
        out_path=outdir / "gd_baselines.png",
        panels=panels,
        gittins_path=gittins_path,
        metric="generational_distance",
        ylabel="Generational distance (GD)",
        title="Generational distance: Gittins vs full-budget baselines",
        seeds=args.seeds,
    )
    plot_single_metric(
        out_path=outdir / "igd_baselines.png",
        panels=panels,
        gittins_path=gittins_path,
        metric="inverted_generational_distance",
        ylabel="Inverted generational distance (IGD)",
        title="Inverted generational distance: Gittins vs full-budget baselines",
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
