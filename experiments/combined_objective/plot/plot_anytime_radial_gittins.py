#!/usr/bin/env python3
"""Run decaying-lambda radial Gittins and plot every recommendation addition.

Run from the repository root. Defaults replay HotpotQA and MathQA, with one
shared warm start per benchmark, nine interior directions plus (1, 0),
lambda=1 and halving at global stopping.
Use --plot-only to redraw the saved JSON without rerunning the selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.offline_radial_gittins import (
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR,
    _cli_directions,
    _jsonable_result,
    _require_data_path,
    load_pickle,
    raw_nondominated_indices,
    simulate_radial_gittins,
)

BLUE = "#175d8d"
ORANGE = "#d26b35"
GRAY = "#b9c2cc"
METRICS = (
    ("relative_hv_regret_percent", "Relative HV regret (%) ↓", "hv_regret_curves"),
    ("generational_distance", "GD ↓", "gd_curves"),
    ("inverted_generational_distance", "IGD ↓", "igd_curves"),
)
DISPLAY_NAMES = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}


def key_checkpoints(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain post-warmup additions to the deployable set, even replacements.

    Comparing membership instead of cardinality preserves an improvement that
    replaces a previous recommendation without increasing the archive size.
    Calibration snapshots establish the initial set but never receive a C label,
    including when a small benchmark completes some arms during warmup itself.
    """
    warmup_evaluations = max(
        (p["cumulative_evaluations"] for p in run["recommendation_trajectory"]
         if p["event"] in {"warm_start", "after_warm_start"}),
        default=0,
    )
    previous: set[int] = set()
    checkpoints = []
    for point in run["recommendation_trajectory"]:
        if point["archive_scope"] != "deployable":
            continue
        selected = set(point["selected_arm_indices"])
        if point["cumulative_evaluations"] <= warmup_evaluations:
            previous = selected
            continue
        added = selected - previous
        removed = previous - selected
        if added:
            checkpoints.append({
                **point,
                "checkpoint": f"C{len(checkpoints) + 1}",
                "added_arm_indices": sorted(added),
                "removed_arm_indices": sorted(removed),
            })
        previous = selected
    return checkpoints


def _save_figure(fig: Any, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png')}", flush=True)


def _metric_value(point: Mapping[str, Any], run: Mapping[str, Any], field: str) -> float | None:
    if field == "relative_hv_regret_percent":
        reference = run["ground_truth_hypervolume"]
        return 100 * point["hypervolume_regret"] / reference if reference > 0 else None
    return point[field]


def _metric_axis(
    ax: Any, run: Mapping[str, Any], field: str, label: str,
    *, cost_limit: float | None = None, label_checkpoints: bool = False,
) -> None:
    trajectory = run["recommendation_trajectory"]
    xs = [p["budget_fraction"] for p in trajectory]
    values = [_metric_value(p, run, field) for p in trajectory]
    ys = [np.nan if value is None else value for value in values]
    ax.step(xs, ys, where="post", color=BLUE, linewidth=1.8,
            label="Current recommendation")
    keys = key_checkpoints(run)
    finite_keys = [p for p in keys if _metric_value(p, run, field) is not None]
    ax.scatter([p["budget_fraction"] for p in finite_keys],
               [_metric_value(p, run, field) for p in finite_keys], color=BLUE, s=24, zorder=4,
               label="Recommendation addition")
    # Several lambda halvings can happen at exactly the same spend.
    stop_fractions = sorted({p["budget_fraction"] for p in trajectory
                             if p["event"] == "lambda_stop"})
    for i, fraction in enumerate(stop_fractions):
        ax.axvline(fraction, color=ORANGE, linestyle=":", alpha=0.55,
                   linewidth=0.9, label="Stopping / λ update" if i == 0 else None)
    if field == "relative_hv_regret_percent":
        ax.axhline(0, color="#65717e", linestyle="--",
                   linewidth=1.1, label="Zero regret (full-data HV)")
        if label_checkpoints:
            for i, point in enumerate(finite_keys):
                ax.annotate(point["checkpoint"],
                            (point["budget_fraction"], _metric_value(point, run, field)),
                            xytext=(3, 7 + 14 * (i % 2)), textcoords="offset points",
                            fontsize=7, color=BLUE)
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        message = ("Relative regret is undefined\n(reference HV is zero)"
                   if field == "relative_hv_regret_percent" else
                   "No completed recommendation\nGD / IGD are undefined (∞)")
        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, ha="center", fontsize=9)
    end = trajectory[-1]["budget_fraction"]
    ax.set_xlim(0, cost_limit if cost_limit is not None else max(0.02, end * 1.045))
    ax.set_ylim(bottom=0)
    if field == "relative_hv_regret_percent":
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(top=100)
        ax.set_yticks([0, .01, .1, 1, 10, 100], ["0", "0.01", "0.1", "1", "10", "100"])
        label += "\n(log scale above 0.01%)"
    elif finite_values:
        upper = max(finite_values)
        ax.set_ylim(top=max(0.01, upper * 1.18))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Cumulative search cost (% of exhaustive cost)", fontsize=9)
    ax.set_ylabel(label)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def plot_metrics(runs: Mapping[str, Any], outdir: Path) -> None:
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4 * len(runs)), squeeze=False)
    for row, (name, run) in enumerate(runs.items()):
        for col, (field, label, _) in enumerate(METRICS):
            _metric_axis(axes[row, col], run, field, label)
            axes[row, col].set_title(f"{DISPLAY_NAMES[name]} · {label}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Anytime radial Gittins · completed combinations · λ halves at stopping", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save_figure(fig, outdir / "metrics")
    # Preserve the full cost axis above, and give early changes enough room to
    # read their checkpoint IDs even when the rest of the run is a long plateau.
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4 * len(runs)), squeeze=False)
    for row, (name, run) in enumerate(runs.items()):
        keys = key_checkpoints(run)
        stops = run["lambda_stop_events"]
        final_fraction = run["recommendation_trajectory"][-1]["budget_fraction"]
        last_change = keys[-1]["budget_fraction"] if keys else final_fraction
        first_stop = stops[0]["budget_fraction"] if stops else last_change
        cost_limit = max(0.02, min(final_fraction, max(last_change, first_stop)) * 1.15)
        for col, (field, label, _) in enumerate(METRICS):
            _metric_axis(axes[row, col], run, field, label,
                         cost_limit=cost_limit, label_checkpoints=True)
            axes[row, col].set_title(f"{DISPLAY_NAMES[name]} · {label}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Anytime radial Gittins · recommendation changes in detail", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save_figure(fig, outdir / "metrics_checkpoint_zoom")
    for field, label, filename in METRICS:
        fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.3), squeeze=False)
        for ax, (name, run) in zip(axes[0], runs.items()):
            _metric_axis(ax, run, field, label)
            ax.set_title(DISPLAY_NAMES[name])
        fig.suptitle(f"Anytime radial Gittins · {label}")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save_figure(fig, outdir / filename)


def plot_lambda_schedule(runs: Mapping[str, Any], outdir: Path) -> None:
    """Show actual lambda updates, including repeated stops at equal spend."""
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.4), squeeze=False)
    for ax, (name, run) in zip(axes[0], runs.items()):
        trajectory = run["recommendation_trajectory"]
        updates = [stop for stop in run["lambda_stop_events"] if stop["continued"]]
        fractions = [trajectory[0]["budget_fraction"]]
        lambdas = [run["params"]["lambda_initial"]]
        for stop in updates:
            fractions.append(stop["budget_fraction"])
            lambdas.append(stop["next_lambda"])
        fractions.append(trajectory[-1]["budget_fraction"])
        lambdas.append(run["current_lambda"])
        ax.step(fractions, lambdas, where="post", color=BLUE, linewidth=1.8)
        ax.scatter(fractions[1:-1], lambdas[1:-1], color=ORANGE, s=22, zorder=3,
                   label="λ halved after global stopping")
        ax.set_yscale("log", base=2)
        ax.set_xlim(0, max(0.02, fractions[-1] * 1.045))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel("Cumulative search cost (% of exhaustive cost)", fontsize=9)
        ax.set_ylabel("λ (log₂ scale)")
        ax.set_title(f"{DISPLAY_NAMES[name]} · {len(updates)} halvings")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Recorded lambda schedule · prior warmup is not a key checkpoint", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, outdir / "lambda_schedule")


def plot_frontiers(name: str, run: Mapping[str, Any], outdir: Path) -> None:
    keys = key_checkpoints(run)
    final = run["recommendation_trajectory"][-1]
    panels = list(keys)
    if not panels or final["cumulative_evaluations"] != panels[-1]["cumulative_evaluations"]:
        panels.append({**final, "checkpoint": "Final", "added_arm_indices": []})
    raw = np.asarray(run["raw_truth_vectors"], dtype=float)
    reference = raw[raw_nondominated_indices(raw)]
    reference = reference[np.argsort(reference[:, 1])]
    log_cost = bool(np.all(raw[:, 1] > 0) and np.ptp(np.log10(raw[:, 1])) >= 1.3)
    for page_start in range(0, len(panels), 9):
        page = panels[page_start:page_start + 9]
        ncols = min(3, len(page))
        nrows = math.ceil(len(page) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)
        for ax, point in zip(axes.flat, page):
            ax.scatter(raw[:, 1], raw[:, 0], s=17, color=GRAY, alpha=0.65,
                       label="All combinations (full data)")
            ax.plot(reference[:, 1], reference[:, 0], "--", color="#7d8995", linewidth=1,
                    label="Full-data Pareto frontier")
            selected = point["selected_arm_indices"]
            if selected:
                # Completed configurations have observed every common question;
                # use recorded online values, not oracle membership, for the line.
                vectors = dict(zip(point["direction_winner_arm_indices"],
                                   point["estimated_raw_winner_vectors"]))
                front = np.asarray([vectors[i] for i in selected])
                front = front[np.argsort(front[:, 1])]
                ax.plot(front[:, 1], front[:, 0], "o-", color=BLUE, markersize=5,
                        linewidth=1.8, label="Recommended frontier", zorder=3)
                for arm in selected:
                    accuracy, cost = vectors[arm]
                    ax.annotate(f"A{arm + 1}", (cost, accuracy), xytext=(4, 5),
                                textcoords="offset points", fontsize=7, color=BLUE)
                added = point["added_arm_indices"]
                if added:
                    points = np.asarray([vectors[i] for i in added])
                    ax.scatter(points[:, 1], points[:, 0], marker="*", s=130,
                               color=ORANGE, edgecolor="white", linewidth=0.5,
                               zorder=4, label="Newly recommended")
            else:
                ax.text(0.5, 0.5, "No completed recommendation", ha="center", transform=ax.transAxes)
            if log_cost:
                ax.set_xscale("log")
            ax.set_ylim(max(0, raw[:, 0].min() - 0.05), min(1.02, raw[:, 0].max() + 0.08))
            ax.set_xlabel("Mean deployment cost (USD" + (", log scale)" if log_cost else ")"))
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{point['checkpoint']} · {point['budget_fraction']:.3%} of exhaustive cost\n"
                         f"λ={point['current_lambda']:.4g} · {len(selected)} recommended", fontsize=10)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(page):]:
            ax.set_visible(False)
        legend: dict[str, Any] = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        fig.legend(list(legend.values()), list(legend), loc="lower center", ncol=3,
                   frameon=False, fontsize=9)
        fig.suptitle(f"{DISPLAY_NAMES[name]} · recommendation checkpoints\n"
                     "A labels identify combinations in combinations.csv; orange stars mark additions",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.095 if nrows == 1 else 0.065, 1, 0.90 if nrows == 1 else 0.93))
        page_number = page_start // 9 + 1
        _save_figure(fig, outdir / f"{name}_frontiers_{page_number:02d}")


def export_tables(runs: Mapping[str, Any], outdir: Path) -> None:
    trajectory_rows, key_rows, combo_rows, stop_rows, timing_rows = [], [], [], [], []
    for name, run in runs.items():
        keys = key_checkpoints(run)
        for p in run["recommendation_trajectory"]:
            trajectory_rows.append({
                "benchmark": name, "event": p["event"],
                "cost_percent": 100 * p["budget_fraction"],
                "cumulative_cost_usd": p["cumulative_search_cost_usd"],
                "evaluations": p["cumulative_evaluations"],
                "lambda": p["current_lambda"], "lambda_stage": p["lambda_stage"],
                "recommendations": len(p["selected_arm_indices"]),
                **{label: p[field] for field, label in (("hypervolume", "HV"),
                    ("generational_distance", "GD"), ("inverted_generational_distance", "IGD"))},
                "HV_regret": p["hypervolume_regret"],
                "relative_HV_regret_percent": _metric_value(p, run, "relative_hv_regret_percent"),
            })
        for point in keys:
            key_rows.append({
                "benchmark": name, "checkpoint": point["checkpoint"],
                "cost_percent": 100 * point["budget_fraction"],
                "cumulative_cost_usd": point["cumulative_search_cost_usd"],
                "evaluations": point["cumulative_evaluations"],
                "lambda": point["current_lambda"], "lambda_stage": point["lambda_stage"],
                "recommendations": len(point["selected_arm_indices"]),
                "added_ids": "; ".join(f"A{i + 1}" for i in point["added_arm_indices"]),
                "removed_ids": "; ".join(f"A{i + 1}" for i in point["removed_arm_indices"]),
                "selected_ids": "; ".join(f"A{i + 1}" for i in point["selected_arm_indices"]),
                "added_models": "; ".join(run["model_results"][i]["model_name"]
                                            for i in point["added_arm_indices"]),
                "HV": point["hypervolume"], "GD": point["generational_distance"],
                "IGD": point["inverted_generational_distance"],
                "HV_regret": point["hypervolume_regret"],
                "relative_HV_regret_percent": _metric_value(point, run, "relative_hv_regret_percent"),
            })
        for model, raw in zip(run["model_results"], run["raw_truth_vectors"]):
            combo_rows.append({"benchmark": name, "id": f"A{model['arm_index'] + 1}",
                               "model": model["model_name"], "accuracy": raw[0],
                               "mean_deployment_cost_usd": raw[1]})
        for stop in run["lambda_stop_events"]:
            stop_rows.append({"benchmark": name, **stop})
        for timing in run.get("stage_timing_events", []):
            timing_rows.append({"benchmark": name, **timing})
    for filename, rows in (("trajectory.csv", trajectory_rows), ("checkpoints.csv", key_rows),
                           ("combinations.csv", combo_rows)):
        if not rows:
            continue
        with (outdir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (outdir / "lambda_stops.json").write_text(json.dumps(stop_rows, indent=2, allow_nan=False) + "\n")
    if timing_rows:
        fieldnames = list(dict.fromkeys(
            key for row in timing_rows for key in row
        ))
        with (outdir / "stage_timings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timing_rows)
    summary = {
        name: {"lambda_initial": run["params"]["lambda_initial"],
               "lambda_final": run["current_lambda"], "lambda_stage": run["lambda_stage"],
               "stop_reason": run["stop_reason"],
               "cost_percent": 100 * run["total_cost"] / run["params"]["bruteforce_search_cost_usd"],
               "total_cost_usd": run["total_cost"], "evaluations": run["total_evaluations"],
               "key_checkpoints": len(key_checkpoints(run)),
               "selected_models": run["selected_models"],
               "directions": run["params"]["directions"],
               "HV": run["hypervolume"], "reference_HV": run["ground_truth_hypervolume"],
               "HV_regret": run["hypervolume_regret"],
               "relative_HV_regret_percent": _metric_value(run, run, "relative_hv_regret_percent"),
               "GD": run["generational_distance"], "IGD": run["inverted_generational_distance"]}
        for name, run in runs.items()
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=list(DISPLAY_NAMES), choices=list(DISPLAY_NAMES))
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "experiments/combined_objective/results/anytime_radial_gittins")
    parser.add_argument("--lambda-initial", type=float, default=1.0)
    parser.add_argument("--lambda-decay", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=129,
                        help="DP resolution; 129 is the quick plotting grid, 513 is production")
    parser.add_argument("--budget-fraction", type=float, default=1.0,
                        help="Maximum fraction of available question evaluations")
    parser.add_argument("--max-search-cost-usd", type=float)
    parser.add_argument("--extra-direction", type=float, nargs=2, action="append", default=[],
                        help="Append a simplex direction; defaults already include (1, 0)")
    parser.add_argument("--boundary-z-padding-extra", type=float, default=0.0)
    parser.add_argument("--boundary-cache-dir", type=Path, default=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR)
    parser.add_argument("--boundary-build-backend", choices=("auto", "scipy", "jax"), default="auto")
    parser.add_argument("--trajectory-checkpoint-interval", type=int, default=10)
    parser.add_argument("--trajectory-target-checkpoints", type=int)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    runs = {}
    cache = RadialGittinsBoundaryCache(cache_dir=args.boundary_cache_dir)
    for name in args.benchmarks:
        run_path = outdir / f"{name}_run.json"
        if args.plot_only:
            runs[name] = json.loads(run_path.read_text())
            continue
        path = _require_data_path(str(ROOT / "experiments/data/lookup" / f"{name}_lookup.pkl"))
        models, datapoints, table = load_pickle(path)
        print(f"Running {DISPLAY_NAMES[name]}: {len(models)} combinations; "
              f"λ₀={args.lambda_initial}, decay={args.lambda_decay}, grid={args.grid_size}", flush=True)
        result = simulate_radial_gittins(
            models, datapoints, table,
            anytime=True, lambda_initial=args.lambda_initial, lambda_decay=args.lambda_decay,
            search_cost_scale_eta=args.eta, seed=args.seed, batch_size=args.batch_size,
            directions=_cli_directions(anytime=True, extra_directions=args.extra_direction),
            observation_budget_fraction=args.budget_fraction,
            max_search_cost_usd=args.max_search_cost_usd,
            boundary_grid=RadialGittinsGrid(z_size=args.grid_size, delta_size=args.grid_size,
                                           state_size=args.grid_size,
                                           boundary_margin_cells=max(2, min(4, args.grid_size // 32))),
            boundary_z_padding_extra=args.boundary_z_padding_extra,
            boundary_cache=cache, boundary_build_backend=args.boundary_build_backend,
            record_recommendation_trajectory=True,
            recommendation_checkpoint_interval=args.trajectory_checkpoint_interval,
            recommendation_checkpoint_target=args.trajectory_target_checkpoints,
        )
        runs[name] = _jsonable_result(result)
        run_path.write_text(json.dumps(runs[name], indent=2, allow_nan=False) + "\n")
        print(f"{DISPLAY_NAMES[name]}: {result.stop_reason}, "
              f"cost={result.total_cost / result.params['bruteforce_search_cost_usd']:.2%}, "
              f"{len(key_checkpoints(runs[name]))} recommendation checkpoints", flush=True)
    export_tables(runs, outdir)
    plot_metrics(runs, outdir)
    plot_lambda_schedule(runs, outdir)
    for name, run in runs.items():
        plot_frontiers(name, run, outdir)


if __name__ == "__main__":
    main()
