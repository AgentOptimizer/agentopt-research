#!/usr/bin/env python3
"""Compare completed-only and a finite-test output rule on one BIRD replay.

Acquisition uses asynchronous per-direction eta decay and required completion
for both rules. Full-data quality and dominance checks below are offline
diagnostics; neither the finite-test recommendation nor acquisition can access them.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

RESULTS = ROOT / "experiments/combined_objective/results"
RAW_BASELINE = RESULTS / "completed_raw_pareto_seed42/bird_dev/comparison.json"
BUDGETS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.0)
CALIBRATION_KEYS = ("prior_variance", "obs_noise_variance", "cost_reference_usd", "expected_batch_costs_usd")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def physical_pulls(trace):
    """Warm starts and actual adaptive batches, excluding skipped visits."""
    for event in trace:
        if event["event"] == "warm_start":
            yield event["arm_index"], event
        elif event["event"] == "direction_visit" and event.get("selected_arm") is not None:
            yield event["selected_arm"], event


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if "wall_time" not in key and not key.endswith("elapsed_seconds")}
    if isinstance(value, (list, tuple)):
        return [without_timing(item) for item in value]
    return value


def verify_identical_acquisition(trace, baseline_trace, eta_events, baseline_eta_events):
    """Check every decision, posterior update, physical cell, cost and eta event."""
    actual, expected = without_timing(trace), without_timing(baseline_trace)
    if len(actual) != len(expected):
        raise AssertionError(f"Acquisition trace lengths differ: {len(actual)} != {len(expected)}")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            differing = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
            raise AssertionError(f"Acquisition event {index} differs in {differing}")
    if without_timing(eta_events) != without_timing(baseline_eta_events):
        raise AssertionError("Per-direction eta event sequences differ")
    samples = [(arm, question) for arm, event in physical_pulls(trace) for question in event["question_ids"]]
    expected_samples = [(arm, question) for arm, event in physical_pulls(baseline_trace)
                        for question in event["question_ids"]]
    if samples != expected_samples or len(set(samples)) != len(samples):
        raise AssertionError("Physical sample order differs or a cell was repeated")
    return {
        "exact_acquisition_trace_ignoring_timing": True,
        "exact_eta_event_sequence": True,
        "exact_physical_sample_sequence": True,
        "exact_batch_and_cumulative_costs": True,
        "trace_event_count": len(trace), "physical_evaluations": len(samples),
        "duplicate_observations": 0, "physical_sample_sequence_sha256": digest(samples),
        "acquisition_trace_without_timing_sha256": digest(actual),
        "direction_eta_events_without_timing_sha256": digest(without_timing(eta_events)),
    }


def enrich_counts(run, trace):
    """Recover exact sample counts at saved checkpoints without replaying policy."""
    counts = [0] * run["model_count"]
    pulls = iter(physical_pulls(trace))
    pending = next(pulls, None)
    target_arms = run["true_best_accuracy_arm_indices"]
    for point in run["points"]:
        while pending is not None and pending[1]["cumulative_evaluations"] <= point["evaluations"]:
            arm, event = pending
            counts[arm] += len(event["question_ids"])
            pending = next(pulls, None)
        selected_counts = [counts[arm] for arm in point["selected_arm_indices"]]
        if "selected_sample_counts" in point and point["selected_sample_counts"] != selected_counts:
            raise AssertionError("Checkpoint winner sample counts disagree with physical trace")
        point["selected_sample_counts"] = selected_counts
        point["completed_arm_indices"] = [arm for arm, count in enumerate(counts)
                                           if count == run["common_question_count"]]
        point["selected_completed_flags"] = [arm in point["completed_arm_indices"]
                                              for arm in point["selected_arm_indices"]]
        point["partial_recommended_count"] = sum(not flag for flag in point["selected_completed_flags"])
        point["true_best_accuracy_sample_counts"] = [counts[arm] for arm in target_arms]


def add_truth_diagnostics(run, normalized_truth, reference):
    from experiments.combined_objective.offline_radial_gittins import hypervolume_2d, raw_nondominated_indices

    raw = np.asarray(run["raw_truth_vectors"], dtype=float)
    best = float(raw[:, 0].max())
    run["true_best_accuracy_arm_indices"] = np.flatnonzero(np.abs(raw[:, 0] - best) <= 1e-12).tolist()
    frontier = set(raw_nondominated_indices(raw))
    run["full_data_pareto_arm_indices"] = sorted(frontier)
    run["oracle_best_accuracy"] = best
    for point in run["points"]:
        selected = point["selected_arm_indices"]
        chosen = normalized_truth[selected] if selected else np.empty((0, 2))
        regret = max(0.0, 1 - hypervolume_2d(chosen, reference) / run["ground_truth_hypervolume"])
        if not math.isclose(point["relative_hv_regret"], regret, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("Checkpoint HV does not evaluate the actual selected arms")
        accuracy = float(raw[selected, 0].max()) if selected else None
        point.update({
            "best_recommended_accuracy": accuracy,
            "accuracy_gap": best - accuracy if accuracy is not None else None,
            "contains_true_accuracy_best": bool(set(selected) & set(run["true_best_accuracy_arm_indices"])),
            "offline_raw_selected_vectors": raw[selected].tolist(),
            "offline_dominated_selected_arm_indices": [arm for arm in selected if arm not in frontier],
            "offline_dominated_selected_count": sum(arm not in frontier for arm in selected),
            "offline_dominated_selected_fraction": sum(arm not in frontier for arm in selected) / len(selected)
            if selected else None,
        })


def compact_lcb_run(result):
    # Frozen recommendation events were captured while sampling. Build their
    # offline archives and metrics only now, when the report requires them.
    from experiments.combined_objective.offline_radial_gittins import materialize_recommendation_diagnostics

    materialize_recommendation_diagnostics(result)
    initial, final = result.recommendation_initial_snapshot, result.recommendation_final_snapshot
    if initial is None or final is None:
        raise AssertionError("Warm-start and final snapshots are required")
    snapshots = [("warm_start", initial)]
    previous = set(initial.selected_arm_indices)
    for checkpoint in result.recommendation_trajectory:
        current = set(checkpoint.selected_arm_indices)
        if current != previous:
            snapshots.append(("membership_change", checkpoint))
            previous = current
    if set(final.selected_arm_indices) != previous:
        snapshots.append(("membership_change", final))
    snapshots.append(("final", final))
    points = []
    for role, checkpoint in snapshots:
        selected = list(checkpoint.selected_arm_indices)
        winners = list(checkpoint.direction_winner_arm_indices)
        observed = dict(zip(winners, checkpoint.estimated_raw_winner_vectors))
        counts = dict(zip(winners, checkpoint.direction_winner_sample_counts))
        lcb = dict(zip(winners, checkpoint.recommendation_desirability_vectors))
        points.append({
            "snapshot_role": role, "event": checkpoint.event,
            "cost_usd": checkpoint.cumulative_search_cost_usd,
            "cost_fraction": checkpoint.budget_fraction,
            "evaluations": checkpoint.cumulative_evaluations,
            "relative_hv_regret": checkpoint.hypervolume_regret / result.ground_truth_hypervolume,
            "selected_arm_indices": selected,
            "selected_sample_counts": [counts[arm] for arm in selected],
            "completed_arm_indices": list(checkpoint.completed_arm_indices),
            "estimated_raw_archive_vectors": [list(observed[arm]) for arm in selected],
            "selected_recommendation_desirability_vectors": [list(lcb[arm]) for arm in selected] if lcb else [],
            "direction_winner_arm_indices": winners,
            "direction_winner_sample_counts": list(checkpoint.direction_winner_sample_counts),
            "recommendation_desirability_vectors": [list(vector) for vector in checkpoint.recommendation_desirability_vectors],
            "estimated_raw_winner_vectors": [list(vector) for vector in checkpoint.estimated_raw_winner_vectors],
            "recommendation_rule": checkpoint.recommendation_rule,
            "recommendation_beta": checkpoint.recommendation_beta,
            "recommendation_min_samples": result.params.get("recommendation_min_samples", 0),
            "archive_scope": checkpoint.archive_scope,
            "direction_eta_multipliers": list(checkpoint.direction_eta_multipliers),
            "direction_eta_stages": list(checkpoint.direction_eta_stages),
        })
        # Optional finite-test evidence is frozen at this membership event,
        # aligned with the legacy direction_winner_arm_indices evidence slots.
        if checkpoint.recommendation_rule in ("finite_lcb", "finite_mean"):
            for name in ("finite_target_mean_vectors", "finite_target_std_vectors", "recommendation_raw_vectors"):
                points[-1][name] = [list(vector) for vector in getattr(checkpoint, name, ())]
    run = {
        "recommendation_rule": result.params["recommendation_rule"],
        "recommendation_filter": result.params.get("recommendation_filter"),
        "recommendation_evidence_scope": result.params.get("recommendation_evidence_scope", "direction_winners"),
        "recommendation_space": result.params.get("recommendation_space"),
        "recommendation_min_samples": result.params.get("recommendation_min_samples", 0),
        "recommendation_eligibility": result.params.get("recommendation_eligibility", "no_minimum_sample_gate"),
        "cost_model": result.params.get("cost_model", "reciprocal"),
        "cost_reference_usd": result.cost_reference_usd,
        "metric_space": result.params.get("metric_space", "reciprocal_cost_desirability"),
        "metric_cost_reference_usd": result.params.get("metric_cost_reference_usd", result.cost_reference_usd),
        "recommendation_beta": result.params["recommendation_beta"] if result.params["recommendation_rule"] != "completed_only" else None,
        "scheduler": result.params["direction_scheduler"], "eta_decay_schedule": result.params["eta_decay_schedule"],
        "seed": result.seed,
        "question_order": result.params.get("question_order", "independent"),
        "warm_start_sha256": digest([event for event in result.trace if event["event"] == "warm_start"]),
        "calibration_sha256": digest({key: result.params[key] for key in (
            (*CALIBRATION_KEYS, "cost_model", "reward_prior_variance", "reward_obs_noise_variance")
            if result.params.get("cost_model") == "raw_mean" else CALIBRATION_KEYS
        )}),
        "ground_truth_hypervolume": result.ground_truth_hypervolume,
        "bruteforce_search_cost_usd": result.params["bruteforce_search_cost_usd"],
        "common_question_count": result.params["common_question_count"],
        "model_count": len(result.model_results),
        "model_names": [arm.model_name for arm in result.model_results],
        "raw_truth_vectors": result.raw_truth_vectors.tolist(),
        "cost_usd": result.total_cost,
        "cost_fraction": result.total_cost / result.params["bruteforce_search_cost_usd"],
        "evaluations": result.total_evaluations, "stop_reason": result.stop_reason,
        "policy_wall_time_seconds": result.policy_wall_time_seconds,
        "boundary_cache_stats": result.params["boundary_cache"]["stats"],
        "recommendation_recording": copy.deepcopy(result.params["recommendation_recording"]),
        "points": points,
    }
    run.update({key: copy.deepcopy(value) for key, value in result.params.items() if key.startswith("finite_target_")})
    for key in ("question_order_semantics", "question_order_rng_scheme"):
        if key in result.params:
            run[key] = copy.deepcopy(result.params[key])
    add_truth_diagnostics(run, result.truth_vectors, result.params.get("metric_reference_point", result.params["reference_point"]))
    enrich_counts(run, result.trace)
    return run


def at_budget(run, fraction):
    return next((point for point in reversed(run["points"]) if point["cost_fraction"] <= fraction + 1e-12), None)


def recovery_point(run, sustained=False):
    points = run["points"]
    eligible = [point["contains_true_accuracy_best"] for point in points]
    if sustained:
        eligible = np.logical_and.accumulate(eligible[::-1])[::-1].tolist()
    return next((point for point, meets in zip(points, eligible) if meets), None)


def summarize(method, run):
    points = run["points"]
    summary = {
        "method": method, "seed": 42, "beta": run.get("recommendation_beta"),
        "question_order": run.get("question_order", "independent"),
        "membership_change_count": sum(point.get("snapshot_role") == "membership_change" for point in points),
        "highest_accuracy_retraction_count": sum(left["contains_true_accuracy_best"] and not right["contains_true_accuracy_best"]
                                                  for left, right in zip(points, points[1:])),
    }
    for prefix, sustained in (("first", False), ("sustained", True)):
        point = recovery_point(run, sustained=sustained)
        for name in ("cost_fraction", "cost_usd", "evaluations"):
            summary[f"{prefix}_best_accuracy_{name}"] = point[name] if point else None
        summary[f"{prefix}_best_accuracy_target_samples"] = json.dumps(point["true_best_accuracy_sample_counts"]) if point else None
    # Integrals use cost units, start after the shared warm start, and include
    # every membership interval. An empty archive has HV regret 1 and no
    # recommendation to classify as dominated.
    for end in (0.01, 0.05, 0.10, 1.0):
        area, duration, dominated_duration, empty_duration = 0.0, 0.0, 0.0, 0.0
        for index, point in enumerate(points):
            left = point["cost_fraction"]
            right = min(points[index + 1]["cost_fraction"] if index + 1 < len(points) else 1.0, end)
            width = max(0.0, right - left)
            duration += width
            area += width * point["relative_hv_regret"]
            dominated_duration += width * (point["offline_dominated_selected_count"] > 0)
            empty_duration += width * (not point["selected_arm_indices"])
        tag = f"through_{end:g}_budget"
        summary[f"mean_hv_regret_{tag}"] = area / duration if duration else None
        summary[f"fraction_spend_with_dominated_selection_{tag}"] = dominated_duration / duration if duration else None
        summary[f"fraction_spend_without_recommendation_{tag}"] = empty_duration / duration if duration else None
    summary.update({
        "final_cost_usd": run["cost_usd"], "final_cost_fraction": run["cost_fraction"],
        "final_evaluations": run["evaluations"], "final_relative_hv_regret": points[-1]["relative_hv_regret"],
        "final_selected_models": json.dumps([run["model_names"][arm] for arm in points[-1]["selected_arm_indices"]]),
        "stop_reason": run["stop_reason"],
    })
    return summary


def export(saved, outdir, *, plots=False):
    started = time.perf_counter()
    runs = saved["runs"]
    summaries = [summarize(method, run) for method, run in runs.items()]
    write_csv(outdir / "summary.csv", summaries)
    # Include every change cost in addition to familiar reference budgets, so
    # comparison rows cover every interval where the recommendation differs.
    budgets = sorted({*BUDGETS, *(point["cost_fraction"] for run in runs.values() for point in run["points"])})
    rows = []
    for budget in budgets:
        for method, run in runs.items():
            point = at_budget(run, budget)
            rows.append({
                "method": method, "budget_fraction": budget,
                "budget_usd": budget * run["bruteforce_search_cost_usd"],
                "recommendation_checkpoint_cost_fraction": point["cost_fraction"] if point else None,
                "recommendation_checkpoint_evaluations": point["evaluations"] if point else None,
                "best_recommended_accuracy": point["best_recommended_accuracy"] if point else None,
                "relative_hv_regret": point["relative_hv_regret"] if point else None,
                "contains_true_accuracy_best": point["contains_true_accuracy_best"] if point else False,
                "offline_dominated_selected_count": point["offline_dominated_selected_count"] if point else None,
                "selected_arm_indices": json.dumps(point["selected_arm_indices"] if point else []),
            })
    write_csv(outdir / "matched_budgets.csv", rows)
    table_seconds = time.perf_counter() - started
    plot_seconds = 0.0
    if plots:
        # A results-only replay need not initialize Matplotlib or render pages.
        from experiments.combined_objective.plot.plot_lcb_recommendations import export_plots

        plot_started = time.perf_counter()
        saved["key_checkpoint_manifest"] = export_plots(saved, outdir)
        plot_seconds = time.perf_counter() - plot_started
    saved["config"]["last_export"] = {
        "plots_requested": plots,
        "table_wall_time_seconds": table_seconds,
        "plot_wall_time_seconds": plot_seconds,
    }
    (outdir / "comparison.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        help="Saved completed reference; defaults to the raw empirical run")
    parser.add_argument("--recommendation-rule", choices=("finite_lcb", "finite_mean"), default="finite_lcb",
                        help="finite_lcb/finite_mean use full-test means with/without a std penalty and require raw_mean")
    parser.add_argument("--recommendation-min-samples", type=int, default=0,
                        help="Minimum observed questions per recommended arm (finite_lcb/finite_mean only)")
    parser.add_argument("--cost-model", choices=("reciprocal", "raw_mean"), default="raw_mean")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--question-order", choices=("shared", "independent"), default="independent",
                        help="Question order must match the completed baseline for this exact-acquisition comparison; default reproduces historical independent runs")
    parser.add_argument("--outdir", type=Path)
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument("--plot-only", action="store_true",
                           help="Redraw all figures and tables from the saved comparison")
    rendering.add_argument("--plots", action="store_true",
                           help="Also render all figures after replay (default: save results without plots)")
    rendering.add_argument("--no-plots", action="store_true",
                           help="Compatibility alias for the default: save results without rendering figures")
    args = parser.parse_args()
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    rule = args.recommendation_rule
    if rule in ("finite_lcb", "finite_mean") and args.cost_model != "raw_mean":
        parser.error(f"{rule} requires --cost-model raw_mean")
    if args.recommendation_min_samples < 0 or (args.recommendation_min_samples and rule not in ("finite_lcb", "finite_mean")):
        parser.error("--recommendation-min-samples must be nonnegative and only applies to finite_lcb/finite_mean")
    if rule == "finite_mean":
        args.beta = 0.0
    args.baseline = args.baseline or RAW_BASELINE
    suffix = "_raw_mean" if args.cost_model == "raw_mean" else ""
    if args.question_order == "shared":
        suffix += "_shared_questions"
    method_name = f"{rule}_recommendations_bird_dev_seed42"
    if rule != "finite_mean":
        method_name += f"_beta{args.beta:g}"
    if args.recommendation_min_samples:
        method_name += f"_min{args.recommendation_min_samples}"
    outdir = args.outdir or RESULTS / f"{method_name}{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "comparison.json"
    if args.plot_only:
        export(json.loads(output.read_text()), outdir, plots=True)
        return

    # Import the evolving selector only when an actual replay is requested.
    replay_source_sha256 = hashlib.sha256(
        (ROOT / "experiments/combined_objective/offline_radial_gittins.py").read_bytes()
    ).hexdigest()
    from agentopt.model_selection.radial_gittins_dp import RadialGittinsBoundaryCache, RadialGittinsGrid
    from experiments.combined_objective.offline_radial_gittins import DEFAULT_RADIAL_BOUNDARY_CACHE_DIR, load_scope, simulate_radial_gittins

    saved_baseline = json.loads(args.baseline.read_text())
    baseline_method = "completed_only" if "completed_only" in saved_baseline["runs"] else "direction_stop"
    baseline = copy.deepcopy(saved_baseline["runs"][baseline_method])
    if baseline.get("question_order", "independent") != args.question_order:
        raise ValueError("Exact-acquisition comparison requires the same question_order as its baseline; use the benchmark runner and plot references for comparisons across question orders")
    if baseline.get("cost_model", "reciprocal") != args.cost_model:
        raise ValueError("The saved completed reference must use the requested acquisition cost model")
    if rule in ("finite_lcb", "finite_mean") and baseline.get("recommendation_filter") != "all_completed_empirical_raw_pareto":
        raise ValueError(f"{rule} requires a completed empirical Pareto reference, not historical direction-filtered recommendations")
    baseline_trace_path = args.baseline.parent / f"{baseline_method}_trace.json.gz"
    with gzip.open(baseline_trace_path, "rt", encoding="utf-8") as handle:
        baseline_trace = json.load(handle)
    lookup = ROOT / "data/scope/bird_dev"
    hashes = {name: hashlib.sha256((lookup / name).read_bytes()).hexdigest()
              for name in ("accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json")}
    baseline_hashes = {Path(name).name: value for name, value in saved_baseline["config"]["lookup_sha256"].items()}
    if hashes != baseline_hashes:
        raise AssertionError("Frozen BIRD inputs differ from the saved baseline")
    models, questions, table = load_scope(str(lookup))
    finished, started = threading.Event(), time.perf_counter()

    def progress():
        while not finished.wait(45):
            print(f"BIRD {rule} beta={args.beta:g}, seed42: {time.perf_counter() - started:.0f}s elapsed", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    print(f"Running BIRD dev: {rule} beta={args.beta:g}, asynchronous eta, seed42, all membership changes", flush=True)
    try:
        result = simulate_radial_gittins(
            models, questions, table, anytime=True, direction_scheduler="round_robin",
            eta_decay_schedule="direction_stop", recommendation_rule=rule, recommendation_beta=args.beta,
            recommendation_min_samples=args.recommendation_min_samples,
            question_order=args.question_order,
            cost_model=args.cost_model,
            seed=42, batch_size=4, lambda_initial=1.0, lambda_decay=0.5, search_cost_scale_eta=1.0,
            observation_budget_fraction=1.0, boundary_z_padding_extra=2.0, effective_cost_bin_ratio=2.0,
            boundary_grid=RadialGittinsGrid(z_size=129, delta_size=129, state_size=129, boundary_margin_cells=4),
            boundary_cache=RadialGittinsBoundaryCache(cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR),
            record_trace=True, record_recommendation_trajectory=True,
            recommendation_checkpoint_interval=None, recommendation_changes_only=True,
            defer_recommendation_diagnostics=True,
        )
    finally:
        finished.set()
    replay_seconds = time.perf_counter() - started
    trace_path = outdir / f"{rule}_trace.json.gz"
    trace_started = time.perf_counter()
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(result.trace, handle)
    trace_seconds = time.perf_counter() - trace_started
    report_started = time.perf_counter()
    run = compact_lcb_run(result)
    report_seconds = time.perf_counter() - report_started
    for key in ("warm_start_sha256", "calibration_sha256", "model_names", "raw_truth_vectors",
                "bruteforce_search_cost_usd", "ground_truth_hypervolume", "cost_usd", "evaluations", "stop_reason"):
        if baseline[key] != run[key]:
            raise AssertionError(f"Baseline mismatch: {key}")
    validation = verify_identical_acquisition(result.trace, baseline_trace, result.direction_eta_events,
                                             saved_baseline["direction_eta_events"])
    baseline.update(recommendation_rule="completed_only", recommendation_beta=None, eta_decay_schedule="direction_stop")
    # Normalize the old compact format: preserve every change and both ends.
    normalized_points = []
    previous = set(baseline["points"][0]["selected_arm_indices"])
    for index, point in enumerate(baseline["points"]):
        current = set(point["selected_arm_indices"])
        if index == 0 or current != previous:
            normalized_points.append({**point, "snapshot_role": "warm_start" if index == 0 else "membership_change"})
        previous = current
    normalized_points.append({**baseline["points"][-1], "snapshot_role": "final"})
    baseline["points"] = normalized_points
    add_truth_diagnostics(baseline, result.truth_vectors, result.params.get("metric_reference_point", result.params["reference_point"]))
    enrich_counts(baseline, baseline_trace)
    for point in baseline["points"]:
        if point["partial_recommended_count"]:
            raise AssertionError("Saved completed-only baseline recommends an unfinished arm")
    payload = {
        "config": {
            "benchmark": "bird_dev", "seed": 42, "algorithm": "Radial-Gittins-decay", "anytime": True,
            "question_order": args.question_order,
            "eta_decay_schedule": "direction_stop", "direction_scheduler": "round_robin",
            "recommendation_rule": rule, "recommendation_beta": args.beta,
            "recommendation_min_samples": args.recommendation_min_samples,
            "recommendation_eligibility": result.params.get("recommendation_eligibility", "no_minimum_sample_gate"),
            "cost_model": args.cost_model,
            "recommendation_changes_only": True, "batch_size": 4, "grid_size": 129,
            "defer_recommendation_diagnostics": True,
            "recommendation_recording": copy.deepcopy(result.params["recommendation_recording"]),
            "timing": {
                "replay_wall_time_seconds": replay_seconds,
                "trace_write_wall_time_seconds": trace_seconds,
                "report_wall_time_seconds": report_seconds,
            },
            "boundary_z_padding_extra": 2.0, "eta_initial": 1.0, "eta_decay": 0.5,
            "baseline_file": str(args.baseline), "baseline_trace_file": str(baseline_trace_path),
            "trace_file": str(trace_path),
            "baseline_file_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "baseline_trace_file_sha256": hashlib.sha256(baseline_trace_path.read_bytes()).hexdigest(),
            "lookup_sha256": hashes,
            "replay_source_sha256": replay_source_sha256,
            "matched_warm_calibration_inputs_and_hv_verified": True,
            "acquisition_validation": validation,
            "recommendation_formula": (
                "Pareto frontier of eligible arms at full-fixed-test predictive mean accuracy and mean USD cost, without a standard-deviation penalty; eligibility uses actual observed question count >= recommendation_min_samples"
                if rule == "finite_mean" else
                "Pareto frontier of eligible arms at full-fixed-test predictive mean accuracy minus beta standard deviations and mean USD cost plus beta standard deviations; eligibility uses actual observed question count >= recommendation_min_samples; completed rows have their empirical means and zero variance"
            ),
            "metric_space": run["metric_space"],
            "budget_alignment": "Latest recommendation at or below each shared budget; held until next membership change",
            "early_mistake_definition": "Selected configuration strictly dominated in raw accuracy and mean cost by some full-data configuration; offline diagnostic only",
            "sustained_recovery_definition": "First checkpoint containing a highest-accuracy configuration with no later withdrawal through final",
            "comparison_scope": "Output recommendations only; exact same acquisitions and total search spend, no acquisition savings",
            "runtime_note": "Baseline reused and shared cache; elapsed times are not a controlled runtime comparison",
        },
        "runs": {"completed_only": baseline, rule: run},
        "parameters": result.params,
        "direction_eta_events": result.direction_eta_events,
    }
    print(f"Exact acquisition parity verified: {validation['physical_evaluations']:,} cells; {len(run['points'])} {rule} snapshots", flush=True)
    export(payload, outdir, plots=args.plots)


if __name__ == "__main__":
    main()
