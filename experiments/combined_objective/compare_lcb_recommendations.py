"""Shared recommendation reporting helpers for the maintained ablation runners.

Full-data quality and dominance checks are offline diagnostics. This module has
no command-line entry point; experiment configuration belongs to the ablations.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time


import numpy as np

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
