#!/usr/bin/env python3
"""Compare the two anytime direction schedulers on frozen QA/SCOPE lookups.

Only completed arms are eligible for recommendations. Same-seed runs share
warm-start cells, calibration, question schedules, and budget limits. Report
cost-aligned trajectories rather than comparing only near-exhaustive endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.offline_radial_gittins import (
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR,
    load_pickle,
    load_scope,
    raw_nondominated_indices,
    simulate_radial_gittins,
)

SCHEDULERS = ("round_robin", "accuracy_last")
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
THRESHOLDS = (0.10, 0.05, 0.02, 0.01, 0.0)
DISPLAY_NAMES = {"hotpotqa": "HotpotQA", "mathqa": "MathQA", "bird_dev": "BIRD dev"}
COLORS = {"round_robin": "#175d8d", "accuracy_last": "#d26b35"}
LABELS = {"round_robin": "Round robin", "accuracy_last": "Accuracy endpoint last"}
PARETO_BUDGETS = (0.05, 0.10, 0.20, 0.40, 0.60, 1.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact_run(result):
    truth_accuracy = result.raw_truth_vectors[:, 0]
    best = float(np.max(truth_accuracy))
    snapshots = [result.recommendation_initial_snapshot]
    snapshots += result.recommendation_trajectory
    snapshots += [result.recommendation_final_snapshot]
    points = []
    for checkpoint in snapshots:
        if checkpoint is None:
            continue
        selected = checkpoint.selected_arm_indices
        assert set(selected) <= set(checkpoint.completed_arm_indices)
        raw_by_arm = dict(zip(checkpoint.direction_winner_arm_indices,
                              checkpoint.estimated_raw_winner_vectors))
        estimated_raw = [list(raw_by_arm[arm]) for arm in selected]
        if selected:
            # Completed-only archives use every common question, so their
            # observed raw estimates must match full-data diagnostic vectors.
            np.testing.assert_allclose(estimated_raw, result.raw_truth_vectors[list(selected)],
                                       rtol=0.0, atol=1e-12)
        accuracy = float(np.max(truth_accuracy[list(selected)])) if selected else None
        point = {
            "cost_usd": checkpoint.cumulative_search_cost_usd,
            "cost_fraction": checkpoint.budget_fraction,
            "evaluations": checkpoint.cumulative_evaluations,
            "relative_hv_regret": checkpoint.hypervolume_regret / result.ground_truth_hypervolume,
            "best_recommended_accuracy": accuracy,
            "accuracy_gap": best - accuracy if accuracy is not None else None,
            "contains_true_accuracy_best": accuracy is not None and abs(accuracy - best) <= 1e-12,
            "selected_arm_indices": list(selected),
            "estimated_raw_archive_vectors": estimated_raw,
            "lambda_stage": checkpoint.lambda_stage,
            "current_lambda": checkpoint.current_lambda,
        }
        if points and point == points[-1]:
            continue
        points.append(point)
    points.sort(key=lambda point: point["evaluations"])
    warm = [event for event in result.trace if event["event"] == "warm_start"]
    assert len(result.observed_cells) == result.total_evaluations
    assert len(set(result.observed_cells)) == result.total_evaluations
    return {
        "scheduler": result.params["direction_scheduler"],
        "seed": result.seed,
        "warm_start_sha256": digest(warm),
        "calibration_sha256": digest({key: result.params[key] for key in (
            "prior_variance", "obs_noise_variance", "cost_reference_usd", "expected_batch_costs_usd")}),
        "ground_truth_hypervolume": result.ground_truth_hypervolume,
        "oracle_best_accuracy": best,
        "bruteforce_search_cost_usd": result.params["bruteforce_search_cost_usd"],
        "common_question_count": result.params["common_question_count"],
        "model_count": len(result.model_results),
        "model_names": [arm.model_name for arm in result.model_results],
        "raw_truth_vectors": result.raw_truth_vectors.tolist(),
        "cost_usd": result.total_cost,
        "cost_fraction": result.total_cost / result.params["bruteforce_search_cost_usd"],
        "evaluations": result.total_evaluations,
        "stop_reason": result.stop_reason,
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "boundary_cache_stats": result.params["boundary_cache"]["stats"],
        "lambda_stage_count": result.params["lambda_stage_count"],
        "stage_timing_events": getattr(result, "stage_timing_events", []),
        "duplicate_observations": 0,
        "points": points,
    }


def at_budget(run, fraction):
    eligible = [point for point in run["points"] if point["cost_fraction"] <= fraction + 1e-12]
    return eligible[-1] if eligible else None


def threshold_point(run, threshold, sustained=False):
    points = run["points"]
    eligible = [point["relative_hv_regret"] <= threshold + 1e-12 for point in points]
    if sustained:
        eligible = list(np.logical_and.accumulate(eligible[::-1]))[::-1]
    return next((point for point, meets in zip(points, eligible) if meets), None)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pareto_snapshot_selection(benchmark, pair):
    if benchmark != "bird_dev":
        return {"rule": "fixed shared budgets", "budgets": list(PARETO_BUDGETS)}
    boundaries = sorted({0.0, 1.0, *(point["cost_fraction"] for run in pair.values()
                                   for point in run["points"] if 0 <= point["cost_fraction"] <= 1)})
    differing_intervals = []
    for left, right in zip(boundaries, boundaries[1:]):
        middle = (left + right) / 2
        points = [at_budget(run, middle) for run in pair.values()]
        memberships = [set(point["selected_arm_indices"]) if point else set() for point in points]
        if len(memberships) == 2 and memberships[0] != memberships[1]:
            differing_intervals.append((left, right))
    widest = max(differing_intervals, key=lambda interval: interval[1] - interval[0], default=None)
    comparison_budget = sum(widest) / 2 if widest else 0.60
    return {
        "rule": "0.5%, 1%, 2%, 10%, midpoint of longest interval with differing recommendation membership (60% fallback), final",
        "budgets": sorted([0.005, 0.01, 0.02, 0.10, comparison_budget, 1.0]),
        "differing_membership_interval": list(widest) if widest else None,
        "differing_membership_midpoint": comparison_budget if widest else None,
    }


def export_pareto_snapshots(runs, outdir):
    """Plot observed recommendation frontiers at identical realized budgets."""
    rows, configurations, selections = [], [], {}
    for benchmark, pair in runs.items():
        first_run = next(iter(pair.values()))
        if "raw_truth_vectors" not in first_run:
            print(f"Skipping {benchmark} Pareto snapshots: older compact results have no raw vectors.", flush=True)
            continue  # Older compact QA results can still be redrawn.
        truth = np.asarray(first_run["raw_truth_vectors"], dtype=float)
        selection = selections[benchmark] = pareto_snapshot_selection(benchmark, pair)
        configurations.extend({
            "benchmark": benchmark, "arm_index": arm, "model_name": name,
            "config_id": name.split("config_id=")[-1] if "config_id=" in name else "",
            "full_data_accuracy": float(truth[arm, 0]),
            "full_data_mean_cost_usd": float(truth[arm, 1]),
        } for arm, name in enumerate(first_run["model_names"]))
        front = truth[raw_nondominated_indices(truth)]
        front = front[np.argsort(front[:, 1])]
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for ax, budget in zip(axes.flat, selection["budgets"]):
            labelled_arms = set()
            ax.scatter(truth[:, 1], truth[:, 0] * 100, s=16, color="#aeb7c0", alpha=0.6,
                       label="All configurations (full data)")
            ax.plot(front[:, 1], front[:, 0] * 100, "--", color="#444444", lw=1.2,
                    label="Full-data Pareto frontier")
            for scheduler, run in pair.items():
                point = at_budget(run, budget)
                if point is None or not point["selected_arm_indices"]:
                    continue
                vectors = np.asarray(point["estimated_raw_archive_vectors"], dtype=float)
                order = np.argsort(vectors[:, 1])
                ax.plot(vectors[order, 1], vectors[order, 0] * 100,
                        color=COLORS[scheduler], lw=1.4, alpha=0.85)
                marker_style = ({"marker": "o", "facecolors": "none", "edgecolors": COLORS[scheduler]}
                                if scheduler == "round_robin" else {"marker": "x", "color": COLORS[scheduler]})
                ax.scatter(vectors[:, 1], vectors[:, 0] * 100, **marker_style, linewidths=1.6, s=55,
                           label=LABELS[scheduler], zorder=4)
                for arm, vector in zip(point["selected_arm_indices"], vectors):
                    if budget == 1.0 and arm not in labelled_arms:
                        name = run["model_names"][arm]
                        label = name.split("config_id=")[-1] if "config_id=" in name else f"A{arm}"
                        ax.annotate(label, (vector[1], vector[0] * 100),
                                    xytext=(4, 5 + 11 * (len(labelled_arms) % 2)),
                                    textcoords="offset points", fontsize=7)
                        labelled_arms.add(arm)
                    rows.append({
                        "benchmark": benchmark, "budget_fraction": budget,
                        "snapshot_role": ("final" if budget == 1.0 else "largest_membership_difference_midpoint"
                                          if budget == selection.get("differing_membership_midpoint") else "fixed_budget"),
                        "scheduler": scheduler, "checkpoint_cost_fraction": point["cost_fraction"],
                        "arm_index": arm, "model_name": run["model_names"][arm],
                        "observed_accuracy": float(vector[0]),
                        "observed_mean_cost_usd": float(vector[1]),
                        "full_data_accuracy": float(truth[arm, 0]),
                        "full_data_mean_cost_usd": float(truth[arm, 1]),
                    })
            title = "Final recommendations" if budget == 1.0 else f"At {budget:.2%} search spend"
            if budget == selection.get("differing_membership_midpoint"):
                title = f"At {budget:.6%} spend · differing archives"
            ax.set_title(title, fontsize=10)
            if np.all(truth[:, 1] > 0):
                ax.set_xscale("log")
            ax.margins(x=0.12)
            ax.grid(alpha=0.18)
        for ax in axes[-1]:
            ax.set_xlabel("Mean deployment cost per question (USD)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy (%)")
        handles, labels = axes.flat[-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.047),
                      ncol=2, frameon=False, fontsize=9)
        figure.suptitle(f"Radial-Gittins-decay · {DISPLAY_NAMES.get(benchmark, benchmark)} · completed-only Pareto recommendations · seed 42")
        note = "Raw accuracy–mean cost shown; HV uses normalized objectives. Final labels are SCOPE config IDs (or A indices for QA)."
        interval = selection.get("differing_membership_interval")
        if interval:
            note += (f"\nThe selected differing-archive interval is only {interval[0]:.6%}–{interval[1]:.6%} "
                     "of exhaustive spend; this panel does not represent a broad gap.")
        figure.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=8.5)
        figure.tight_layout(rect=(0, 0.14, 1, 0.95))
        figure.savefig(outdir / f"{benchmark}_pareto_snapshots.png", dpi=180)
        figure.savefig(outdir / f"{benchmark}_pareto_snapshots.pdf")
        plt.close(figure)
    if rows:
        write_csv(outdir / "pareto_snapshots.csv", rows)
        write_csv(outdir / "configurations.csv", configurations)
    return selections


def export(runs, outdir):
    budget_rows, threshold_rows, summary_rows = [], [], []
    for benchmark, pair in runs.items():
        for scheduler, run in pair.items():
            recovery = next((point for point in run["points"] if point["contains_true_accuracy_best"]), None)
            summary_rows.append({
                "benchmark": benchmark, "scheduler": scheduler, "seed": run["seed"],
                "cost_fraction": run["cost_fraction"], "evaluations": run["evaluations"],
                "stop_reason": run["stop_reason"], "runtime_seconds": run["policy_wall_time_seconds"],
                "first_accuracy_best_cost_fraction": recovery["cost_fraction"] if recovery else None,
                "final_relative_hv_regret": run["points"][-1]["relative_hv_regret"],
            })
        for budget in BUDGETS:
            row = {"benchmark": benchmark, "budget_fraction": budget}
            for scheduler, run in pair.items():
                point = at_budget(run, budget)
                for field in ("relative_hv_regret", "accuracy_gap", "contains_true_accuracy_best"):
                    row[f"{scheduler}_{field}"] = point[field] if point else None
            rr, al = (row[f"{scheduler}_relative_hv_regret"] for scheduler in SCHEDULERS)
            row["accuracy_last_minus_round_robin_regret_pp"] = 100 * (al - rr) if rr is not None and al is not None else None
            budget_rows.append(row)
        for threshold in THRESHOLDS:
            for sustained in (False, True):
                row = {"benchmark": benchmark, "relative_hv_regret_threshold": threshold,
                       "threshold_policy": "sustained_to_end" if sustained else "first_hit"}
                for scheduler, run in pair.items():
                    point = threshold_point(run, threshold, sustained)
                    row[f"{scheduler}_cost_fraction"] = point["cost_fraction"] if point else None
                    row[f"{scheduler}_cost_usd"] = point["cost_usd"] if point else None
                rr, al = (row[f"{scheduler}_cost_fraction"] for scheduler in SCHEDULERS)
                row["accuracy_last_minus_round_robin_cost_pp"] = 100 * (al - rr) if rr is not None and al is not None else None
                threshold_rows.append(row)
    write_csv(outdir / "summary.csv", summary_rows)
    write_csv(outdir / "matched_budgets.csv", budget_rows)
    write_csv(outdir / "hv_threshold_costs.csv", threshold_rows)
    figure, axes = plt.subplots(2, len(runs), figsize=(12, 7.5), squeeze=False, sharex=True)
    for column, (benchmark, pair) in enumerate(runs.items()):
        for scheduler, run in pair.items():
            xs = [point["cost_fraction"] * 100 for point in run["points"]]
            regret = [point["relative_hv_regret"] * 100 for point in run["points"]]
            gap = [point["accuracy_gap"] * 100 if point["accuracy_gap"] is not None else np.nan for point in run["points"]]
            # Carry the last recommendation after the policy has terminated.
            xs.append(100.0)
            regret.append(regret[-1])
            gap.append(gap[-1])
            style = "-" if scheduler == "round_robin" else "--"
            axes[0, column].step(xs, regret, where="post", color=COLORS[scheduler],
                                 linestyle=style, label=LABELS[scheduler])
            axes[1, column].step(xs, gap, where="post", color=COLORS[scheduler],
                                 linestyle=style, label=LABELS[scheduler])
        axes[0, column].set_title(DISPLAY_NAMES.get(benchmark, benchmark))
        axes[0, column].set_yscale("symlog", linthresh=0.1)
        axes[0, column].set_ylim(bottom=0)
        axes[0, column].set_yticks([0, 0.1, 1, 10, 100], labels=["0", "0.1", "1", "10", "100"])
        axes[1, column].set_ylim(bottom=0)
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 100)
        axes[1, column].set_xlabel("Search spend / exhaustive search cost (%)")
    axes[0, 0].set_ylabel("Relative HV regret (%) ↓\n(symlog scale)")
    axes[1, 0].set_ylabel("Best recommended accuracy gap (pp) ↓")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Radial-Gittins-decay · direction scheduler comparison · seed 42 · completed arms only · grid 129")
    figure.text(0.5, 0.015, "HV uses normalized objectives; Pareto snapshots show raw accuracy–cost. Accuracy gap starts at the first completed recommendation.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    figure.savefig(outdir / "comparison.png", dpi=180)
    figure.savefig(outdir / "comparison.pdf")
    plt.close(figure)
    return export_pareto_snapshots(runs, outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(DISPLAY_NAMES),
                        default=["hotpotqa", "mathqa"])
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--plot-only", action="store_true", help="Rebuild CSVs and figure from the saved compact results")
    args = parser.parse_args()
    if args.outdir is None:
        suffix = "_bird_dev" if args.benchmarks == ["bird_dev"] else ""
        if args.benchmarks != ["hotpotqa", "mathqa"] and not suffix:
            suffix = "_" + "_".join(args.benchmarks)
        args.outdir = ROOT / f"experiments/combined_objective/results/direction_schedulers{suffix}_completed_only_seed42"
    args.outdir.mkdir(parents=True, exist_ok=True)
    result_path = args.outdir / "comparison.json"
    if args.plot_only:
        saved = json.loads(result_path.read_text())
        saved["config"]["pareto_snapshots"] = export(saved["runs"], args.outdir)
        result_path.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
        return
    config = {
        "algorithm": "Radial-Gittins-decay", "anytime": True,
        "seed": 42, "benchmarks": args.benchmarks, "schedulers": SCHEDULERS,
        "recommendation_eligibility": "completed_only", "grid_size": 129,
        "batch_size": 4, "lambda_initial": 1.0, "lambda_decay": 0.5,
        "search_cost_scale_eta": 1.0, "observation_budget_fraction": 1.0,
        "boundary_z_padding_extra": 2.0, "effective_cost_bin_ratio": 2.0,
        "max_search_cost_usd": None, "question_universe": "common",
        "matched_budget_fractions": BUDGETS, "hv_regret_thresholds": THRESHOLDS,
        "default_pareto_budget_fractions": PARETO_BUDGETS,
        "checkpoint_recording": "Completion and lambda-stop events; no ordinary pull checkpoints",
        "budget_alignment": "Last recommendation at or before realized spend; carry final recommendation after termination",
        "threshold_sustained_semantics": "Below threshold at every later checkpoint through this replay's end",
        "accuracy_metric": "Maximum full-data accuracy among currently recommended completed arms",
        "runtime_caveat": "Shared disk/memory cache and fixed run order; timings are descriptive, not a fair speed comparison",
        "boundary_cache_dir": str(DEFAULT_RADIAL_BOUNDARY_CACHE_DIR),
        "replay_source_sha256": hashlib.sha256((ROOT / "experiments/combined_objective/offline_radial_gittins.py").read_bytes()).hexdigest(),
        "lookup_sha256": {},
    }
    runs = {}
    cache = RadialGittinsBoundaryCache(cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR)
    grid = RadialGittinsGrid(z_size=129, delta_size=129, state_size=129, boundary_margin_cells=4)
    for benchmark in config["benchmarks"]:
        if benchmark == "bird_dev":
            lookup = ROOT / "data/scope/bird_dev"
            config["lookup_sha256"][benchmark] = {
                filename: hashlib.sha256((lookup / filename).read_bytes()).hexdigest()
                for filename in ("accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json")
            }
            models, questions, table = load_scope(str(lookup))
        else:
            lookup = ROOT / f"experiments/data/lookup/{benchmark}_lookup.pkl"
            config["lookup_sha256"][benchmark] = hashlib.sha256(lookup.read_bytes()).hexdigest()
            models, questions, table = load_pickle(str(lookup))
        pair = runs[benchmark] = {}
        for scheduler in SCHEDULERS:
            print(f"Running {benchmark}, {scheduler}, seed=42, completed-only", flush=True)
            finished = threading.Event()
            started = time.perf_counter()

            def progress():
                while not finished.wait(45):
                    stats = cache.stats_snapshot()
                    print(f"  {benchmark}, {scheduler}: {time.perf_counter()-started:.0f}s elapsed; "
                          f"shared cache builds={stats.builds}, disk_hits={stats.disk_hits}", flush=True)

            reporter = threading.Thread(target=progress, daemon=True)
            reporter.start()
            try:
                result = simulate_radial_gittins(
                    models, questions, table, anytime=True, direction_scheduler=scheduler,
                    seed=42, batch_size=4, lambda_initial=1.0, lambda_decay=0.5,
                    search_cost_scale_eta=1.0, observation_budget_fraction=1.0,
                    boundary_z_padding_extra=2.0, effective_cost_bin_ratio=2.0,
                    boundary_grid=grid, boundary_cache=cache,
                    record_recommendation_trajectory=True,
                    recommendation_checkpoint_interval=None, recommendation_changes_only=False,
                )
            finally:
                finished.set()
                reporter.join()
            pair[scheduler] = compact_run(result)
            print(f"Finished {benchmark}, {scheduler}: {result.policy_wall_time_seconds:.1f}s, "
                  f"stop={result.stop_reason}, cost={pair[scheduler]['cost_fraction']:.2%}, "
                  f"checkpoints={len(pair[scheduler]['points'])}", flush=True)
            result_path.write_text(json.dumps({"config": config, "runs": runs}, indent=2, allow_nan=False) + "\n")
            del result
        for field in ("warm_start_sha256", "calibration_sha256", "bruteforce_search_cost_usd", "ground_truth_hypervolume"):
            assert pair["round_robin"][field] == pair["accuracy_last"][field], field
        config["pareto_snapshots"] = export(runs, args.outdir)
        result_path.write_text(json.dumps({"config": config, "runs": runs}, indent=2, allow_nan=False) + "\n")
        print(f"Paired outputs updated: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
