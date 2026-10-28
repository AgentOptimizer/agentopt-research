"""Nested observed-data random baselines for max Q / min latency / min USD.

Random questions evaluates one shared random question at a time across every
configuration. Random configurations fully evaluates one random configuration
at a time. Recommendations use the empirical raw three-objective Pareto set
over the observations available at that complete unit. Oracle scoring starts
only after the entire recommendation trajectory has been frozen.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.combined_objective.three_objective_metrics import (
    enrich_run,
    raw_pareto_indices,
)


VERSIONS = ("random_questions", "random_configurations")


def _sample_values(sample: Any) -> np.ndarray:
    values = np.asarray((sample.score, sample.latency_seconds, sample.cost), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("observed Q, latency, and cost must be finite")
    if not 0.0 <= values[0] <= 1.0 or np.any(values[1:] < 0.0):
        raise ValueError("observed Q must lie in [0, 1]; latency and cost must be nonnegative")
    return values


def simulate_three_objective_random_search(
    models: Sequence[str],
    questions: Sequence[int],
    table: Mapping[str, Mapping[int, Any]],
    *,
    version: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a complete nested trajectory, scored afterward in shared 3D space.

    A checkpoint represents a whole shared question or a whole configuration;
    no partial-unit result or prorated USD cost is recorded. Every matrix cell
    is read exactly once during the complete trajectory. Per-checkpoint counts
    and the run-level permutations identify every observed prefix compactly.
    No full-data quantity influences the permutations or recommendations.
    """
    started = time.perf_counter()
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    model_names = list(models)
    question_ids = list(questions)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("models must be nonempty and unique")
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("questions must be nonempty and unique")
    if any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) for q in question_ids):
        raise ValueError("question IDs must be integers")
    question_ids = [int(q) for q in question_ids]
    n_arms, n_questions = len(model_names), len(question_ids)
    rng = np.random.default_rng(int(seed))
    arm_order = list(range(n_arms))
    question_positions = list(range(n_questions))
    if version == "random_questions":
        question_positions = rng.permutation(n_questions).tolist()
    else:
        arm_order = rng.permutation(n_arms).tolist()

    # Unfilled cells are never used to form a recommendation. Retaining only
    # observed values also provides canonical-order summation at completion.
    observed = np.full((n_arms, n_questions, 3), np.nan, dtype=np.float64)
    completed_means = np.full((n_arms, 3), np.nan, dtype=np.float64)
    points: list[dict[str, Any]] = []
    unit_costs: list[float] = []
    total_evaluations = 0
    steps = n_questions if version == "random_questions" else n_arms

    def observe(arm: int, question_position: int) -> np.ndarray:
        try:
            sample = table[model_names[arm]][question_ids[question_position]]
        except KeyError as error:
            raise ValueError("three-objective random search requires a complete aligned matrix") from error
        value = _sample_values(sample)
        observed[arm, question_position] = value
        return value

    for step in range(1, steps + 1):
        if version == "random_questions":
            position = question_positions[step - 1]
            batch = np.asarray([observe(arm, position) for arm in range(n_arms)])
            seen_positions = sorted(question_positions[:step])
            means = observed[:, seen_positions, :].mean(axis=1)
            selected = raw_pareto_indices(means)
            completed = list(range(n_arms)) if step == n_questions else []
            sampled_arm_count, sampled_question_count = n_arms, step
            unit_evaluations = n_arms
            unit_name = "shared_question"
            unit_id = question_ids[position]
        else:
            arm = arm_order[step - 1]
            batch = np.asarray([observe(arm, position) for position in range(n_questions)])
            completed_means[arm] = batch.mean(axis=0)
            completed = sorted(arm_order[:step])
            local_front = raw_pareto_indices(completed_means[completed])
            selected = [completed[local_index] for local_index in local_front]
            sampled_arm_count, sampled_question_count = step, n_questions
            unit_evaluations = n_questions
            unit_name = "complete_configuration"
            unit_id = arm
        unit_costs.append(float(batch[:, 2].sum()))
        total_evaluations += unit_evaluations
        cumulative_cost = math.fsum(unit_costs)
        points.append({
            "event": unit_name,
            "step": step,
            "unit_id": int(unit_id),
            "sampled_arm_count": sampled_arm_count,
            "sampled_question_count": sampled_question_count,
            "completed_arm_indices": completed,
            "selected_arm_indices": selected,
            "total_evaluations": total_evaluations,
            "cumulative_evaluations": total_evaluations,
            "cumulative_search_cost_usd": cumulative_cost,
            "total_cost": cumulative_cost,
            "actual_unit_search_cost_usd": unit_costs[-1],
        })

    selection_seconds = time.perf_counter() - started
    # Every cell is now observed and every recommendation above is already
    # fixed. Full-data means and normalization are diagnostic from here onward.
    truth = observed.mean(axis=1)
    brute_force_cost = float(observed[:, :, 2].sum())
    final_cost = math.fsum(unit_costs)
    parameters = {
        "version": version,
        "seed": int(seed),
        "objective_order": ["Q", "L", "D"],
        "objective_units": ["accuracy", "seconds", "USD"],
        "raw_objectives": ["accuracy_max", "mean_latency_seconds_min", "mean_cost_usd_min"],
        "recommendation_rule": "empirical_raw_pareto",
        "question_universe": "complete_aligned",
        "unit": "shared_question_across_all_configurations" if version == "random_questions" else "all_questions_for_one_configuration",
        "selection_uses_full_data_truth": False,
        "scoring_timing": "after_complete_recommendation_trajectory",
    }
    run = {
        "selector": version,
        "version": version,
        "seed": int(seed),
        "params": parameters,
        "parameters": parameters,
        "model_names": model_names,
        "question_ids": question_ids,
        "n_arms": n_arms,
        "n_questions": n_questions,
        "sampled_arm_order": arm_order,
        "sampled_question_order": [question_ids[position] for position in question_positions],
        "raw_truth_vectors": truth.tolist(),
        "points": points,
        "selected_arm_indices": points[-1]["selected_arm_indices"],
        "total_evaluations": total_evaluations,
        "search_cost_usd": final_cost,
        "cumulative_search_cost_usd": final_cost,
        "total_cost": final_cost,
        "bruteforce_search_cost_usd": brute_force_cost,
        "stop_reason": "all_cells_observed",
        "selection_wall_seconds": selection_seconds,
    }
    enrich_run(run)
    run["total_wall_seconds"] = time.perf_counter() - started
    return run


def latest_under_checkpoint(run: Mapping[str, Any], target_cost_fraction: float) -> dict[str, Any]:
    """Return the latest complete unit at/below a reached actual-USD target.

    Targets below the first whole unit are unavailable. The helper performs
    neither interpolation nor partial accounting and does not silently use a
    future unit or an unreached target.
    """
    target = float(target_cost_fraction)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_cost_fraction must be finite and nonnegative")
    terminal = float(run["cost_fraction"])
    eligible = [point for point in run["points"] if float(point["cost_fraction"]) <= target + 1e-12]
    if terminal + 1e-12 < target or not eligible:
        return {
            "target_cost_fraction": target,
            "available": False,
            "terminal_cost_fraction": terminal,
            "reason": "target_not_reached" if terminal + 1e-12 < target else "no_complete_unit_under_budget",
        }
    return {**eligible[-1], "target_cost_fraction": target, "available": True}
