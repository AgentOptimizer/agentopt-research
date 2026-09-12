#!/usr/bin/env python3
"""Compare global and per-direction eta decay on complete BIRD dev, seed 42."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

from agentopt.model_selection.radial_gittins_dp import RadialGittinsBoundaryCache, RadialGittinsGrid
from experiments.combined_objective.compare_direction_schedulers import compact_run, at_budget
from experiments.combined_objective.offline_radial_gittins import (
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR, load_scope,
    materialize_recommendation_diagnostics, simulate_radial_gittins,
)

BASELINE = ROOT / "experiments/combined_objective/results/direction_schedulers_bird_dev_completed_only_seed42/comparison.json"
OUTDIR = ROOT / "experiments/combined_objective/results/direction_eta_decay_bird_dev_completed_only_seed42"
LABELS = {"global_stop": "Global eta decay", "direction_stop": "Independent eta per direction"}
COLORS = {"global_stop": "#175d8d", "direction_stop": "#d26b35"}
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.675, 0.75, 1.0)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_accuracy_best(run):
    return next((point for point in run["points"] if point["contains_true_accuracy_best"]), None)


def export(saved, outdir, *, plots=False):
    runs = saved["runs"]
    summaries = []
    matched = []
    for method, run in runs.items():
        first = first_accuracy_best(run)
        summaries.append({
            "method": method, "seed": 42,
            "first_best_accuracy_cost_fraction": first["cost_fraction"] if first else None,
            "first_best_accuracy_cost_usd": first["cost_usd"] if first else None,
            "first_best_accuracy_evaluations": first["evaluations"] if first else None,
            "final_cost_fraction": run["cost_fraction"], "final_evaluations": run["evaluations"],
            "final_relative_hv_regret": run["points"][-1]["relative_hv_regret"],
            "final_selected_models": json.dumps([run["model_names"][i] for i in run["points"][-1]["selected_arm_indices"]]),
            "stop_reason": run["stop_reason"],
        })
        for budget in BUDGETS:
            point = at_budget(run, budget)
            matched.append({
                "method": method, "budget_fraction": budget,
                "actual_checkpoint_fraction": point["cost_fraction"] if point else None,
                "best_recommended_accuracy": point["best_recommended_accuracy"] if point else None,
                "relative_hv_regret": point["relative_hv_regret"] if point else None,
                "selected_arm_indices": json.dumps(point["selected_arm_indices"]) if point else "[]",
            })
    write_csv(outdir / "summary.csv", summaries)
    write_csv(outdir / "matched_budgets.csv", matched)

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        from experiments.combined_objective.plot.plot_eta_key_checkpoints import export_key_checkpoint_frontiers

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for method, run in runs.items():
            points = run["points"]
            x = np.array([point["cost_fraction"] for point in points])
            acc = np.array([np.nan if point["best_recommended_accuracy"] is None else point["best_recommended_accuracy"] for point in points])
            hv = np.array([point["relative_hv_regret"] for point in points])
            if x[-1] < 1.0:
                x = np.r_[x, 1.0]
                acc = np.r_[acc, acc[-1]]
                hv = np.r_[hv, hv[-1]]
            style = dict(color=COLORS[method], label=LABELS[method], linewidth=2.1,
                         linestyle="-" if method == "global_stop" else "--", where="post")
            axes[0].step(x, acc, **style)
            axes[1].step(x, hv, **style)
            first = first_accuracy_best(run)
            if first:
                axes[0].scatter(first["cost_fraction"], first["best_recommended_accuracy"], color=COLORS[method], zorder=5)
        axes[0].axhline(runs["global_stop"]["oracle_best_accuracy"], color="#777777", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Highest accuracy in completed recommendation")
        axes[0].set_title("Highest-accuracy recovery")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set_ylabel("Relative hypervolume regret")
        axes[1].set_title("Whole Pareto recommendation")
        axes[1].set_yscale("symlog", linthresh=0.001)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_xlabel("Cumulative search cost / full matrix cost")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        fig.suptitle("BIRD dev · independent η decay · seed 42 · completed-only")
        fig.text(0.5, 0.01, "Same ten directions, warm start, posterior model and cost bins; only the decay trigger differs.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"comparison.{ext}", dpi=180)
        plt.close(fig)

        export_key_checkpoint_frontiers(runs, outdir)

        events = saved["direction_eta_events"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        stages = [event for event in runs["global_stop"]["stage_timing_events"] if event["event"] == "lambda_stage_stop"]
        gx = [0.0] + [event["budget_fraction"] for event in stages]
        gy = [1.0] + [event.get("next_lambda") or event["current_lambda"] for event in stages]
        axes[0].step(gx, gy, where="post", color=COLORS["global_stop"])
        axes[0].set_title("Global decay: wait for all directions")
        palette = plt.get_cmap("tab10")
        for i, direction in enumerate(saved["parameters"]["directions"]):
            local = [event for event in events if event["direction_index"] == i and event["event"] == "direction_eta_decay"]
            x = [0.0] + [event["cumulative_search_cost_usd"] / runs["direction_stop"]["bruteforce_search_cost_usd"] for event in local]
            y = [1.0] + [event["next_lambda"] for event in local]
            if x[-1] < runs["direction_stop"]["cost_fraction"]:
                x.append(runs["direction_stop"]["cost_fraction"])
                y.append(y[-1])
            axes[1].step(x, y, where="post", label=str(tuple(direction)), color=palette(i), linewidth=1.1)
        axes[1].set_title("Independent decay after each direction stops")
        axes[1].legend(fontsize=7, ncol=2)
        axes[0].set_ylabel("Effective η multiplier (initial η = 1)")
        for ax in axes:
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("Cumulative search cost / full matrix cost")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"eta_schedule.{ext}", dpi=180)
        plt.close(fig)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument("--plot-only", action="store_true", help="Render saved results without replaying")
    rendering.add_argument("--plots", action="store_true", help="Also render figures after replay")
    rendering.add_argument("--no-plots", action="store_true", help="Alias for the default: save data without figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "comparison.json"
    if args.plot_only:
        export(json.loads(output.read_text()), args.outdir, plots=True)
        return
    saved_baseline = json.loads(args.baseline.read_text())
    baseline = saved_baseline["runs"]["bird_dev"]["round_robin"]
    lookup = ROOT / "data/scope/bird_dev"
    hashes = {name: hashlib.sha256((lookup / name).read_bytes()).hexdigest()
              for name in ("accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json")}
    assert hashes == saved_baseline["config"]["lookup_sha256"]["bird_dev"]
    source_hash = hashlib.sha256((ROOT / "experiments/combined_objective/offline_radial_gittins.py").read_bytes()).hexdigest()
    models, questions, table = load_scope(str(lookup))
    finished = threading.Event()
    started = time.perf_counter()

    def progress():
        while not finished.wait(45):
            print(f"BIRD direction_stop, seed42: {time.perf_counter() - started:.0f}s elapsed", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    print("Running BIRD dev, independent eta decay, seed42, completed-only", flush=True)
    try:
        result = simulate_radial_gittins(
            models, questions, table, anytime=True, direction_scheduler="round_robin",
            eta_decay_schedule="direction_stop", seed=42, batch_size=4,
            lambda_initial=1.0, lambda_decay=0.5, search_cost_scale_eta=1.0,
            observation_budget_fraction=1.0, boundary_z_padding_extra=2.0,
            effective_cost_bin_ratio=2.0,
            boundary_grid=RadialGittinsGrid(z_size=129, delta_size=129, state_size=129, boundary_margin_cells=4),
            boundary_cache=RadialGittinsBoundaryCache(cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR),
            record_trace=True, record_recommendation_trajectory=True,
            recommendation_checkpoint_interval=None, recommendation_changes_only=True,
            defer_recommendation_diagnostics=True,
        )
    finally:
        finished.set()
    materialize_recommendation_diagnostics(result)
    run = compact_run(result)
    run["recommendation_recording"] = dict(result.params["recommendation_recording"])
    for key in ("warm_start_sha256", "calibration_sha256", "model_names", "raw_truth_vectors",
                "bruteforce_search_cost_usd", "ground_truth_hypervolume"):
        assert baseline[key] == run[key], key
    visits = [event for event in result.trace if event["event"] == "direction_visit"]
    assert all(event["direction_index"] == i % 10 for i, event in enumerate(visits))
    payload = {
        "config": {"benchmark": "bird_dev", "seed": 42, "recommendation_eligibility": "completed_only",
                   "eta_initial": 1.0, "eta_decay": 0.5, "direction_scheduler": "round_robin",
                   "recommendation_changes_only": True, "defer_recommendation_diagnostics": True,
                   "baseline_file": str(args.baseline), "lookup_sha256": hashes,
                   "replay_source_sha256": source_hash,
                   "baseline_replay_source_sha256": saved_baseline["config"]["replay_source_sha256"],
                   "matched_warm_calibration_inputs_and_hv_verified": True,
                   "budget_alignment": "Latest recommendation at or below budget; final recommendation carried after termination",
                   "runtime_note": "Baseline reused; wall times are not a controlled runtime comparison"},
        "runs": {"global_stop": baseline, "direction_stop": run},
        "parameters": result.params,
        "direction_eta_events": result.direction_eta_events,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with gzip.open(args.outdir / "direction_stop_trace.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(result.trace, handle)
    export(payload, args.outdir, plots=args.plots)


if __name__ == "__main__":
    main()
