#!/usr/bin/env python3
"""Random-search baselines with checkpoints on the realized USD-cost axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    mean_raw_vectors,
    normalized_truth_vectors,
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_radial_gittins import (
    _sample_values,
    generational_distance,
    hypervolume_2d,
    inverted_generational_distance,
)


RANDOM_CONFIGURATIONS = "random_configurations"
RANDOM_QUESTIONS = "random_questions"


@dataclass(frozen=True)
class CostBudgetRandomResult:
    summary: dict[str, object]
    trajectory: list[dict[str, object]]
    estimated_raw_vectors: np.ndarray
    truth_raw_vectors: np.ndarray
    truth_vectors: np.ndarray


def _checkpoint_row(
    *,
    event: str,
    method: str,
    seed: int,
    models: Sequence[str],
    selected: tuple[int, ...],
    evaluations: int,
    cost_usd: float,
    bruteforce_cost_usd: float,
    truth_vectors: np.ndarray,
    true_front_vectors: np.ndarray,
    ground_truth_hv: float,
) -> dict[str, object]:
    obtained = (
        truth_vectors[list(selected)]
        if selected
        else np.empty((0, 2), dtype=np.float64)
    )
    hv = hypervolume_2d(obtained) if selected else 0.0
    return {
        "method": method,
        "seed": int(seed),
        "event": event,
        "budget_fraction": float(cost_usd) / float(bruteforce_cost_usd),
        "cumulative_evaluations": int(evaluations),
        "cumulative_search_cost_usd": float(cost_usd),
        "hypervolume": float(hv),
        "hv_regret": float(max(0.0, ground_truth_hv - hv)),
        "generational_distance": float(
            generational_distance(obtained, true_front_vectors)
        ),
        "inverted_generational_distance": float(
            inverted_generational_distance(obtained, true_front_vectors)
        ),
        "archive_size": len(selected),
        "selected_arm_indices": ";".join(str(index) for index in selected),
        "selected_models": ";".join(models[index] for index in selected),
    }


def simulate_cost_budget_random_search(
    models: Sequence[str],
    questions: Sequence[int],
    table: Any,
    *,
    method: str,
    seed: int,
    cost_checkpoint_fractions: Sequence[float] = (0.10, 0.30),
) -> CostBudgetRandomResult:
    """Replay one random policy to full data and retain USD checkpoints.

    ``random_configurations`` evaluates complete configurations in a random
    order. ``random_questions`` evaluates one shared random question across all
    configurations per policy step.  Each policy step is atomic, so a USD
    checkpoint is the latest completed step whose cumulative cost does not
    exceed the requested threshold.
    """
    if method not in {RANDOM_CONFIGURATIONS, RANDOM_QUESTIONS}:
        raise ValueError(f"unsupported random method: {method!r}")
    targets = tuple(sorted({float(value) for value in cost_checkpoint_fractions}))
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("cost checkpoint fractions must lie in (0, 1]")

    models = tuple(models)
    questions = common_question_ids(models, questions, table)
    truth_raw = mean_raw_vectors(models, questions, table)
    positive_costs = truth_raw[:, 1][truth_raw[:, 1] > 0.0]
    cost_reference = float(np.median(positive_costs))
    truth_vectors = normalized_truth_vectors(truth_raw, cost_reference)
    true_front = tuple(pareto_min_cost_indices(truth_raw))
    true_front_vectors = truth_vectors[list(true_front)]
    ground_truth_hv = hypervolume_2d(true_front_vectors)
    bruteforce_cost = float(
        sum(
            float(_sample_values(table[model][question])[1])
            for model in models
            for question in questions
        )
    )

    rng = np.random.default_rng(seed)
    sum_score = np.zeros(len(models), dtype=np.float64)
    sum_cost = np.zeros(len(models), dtype=np.float64)
    counts = np.zeros(len(models), dtype=np.int64)
    trajectory: list[dict[str, object]] = []
    last_selected: tuple[int, ...] | None = None
    next_target = 0
    evaluations = 0
    cumulative_cost = 0.0
    membership_checks = 0
    membership_changes = 0
    previous_update_snapshot: dict[str, object] | None = None

    def estimated_vectors() -> np.ndarray:
        result = np.full((len(models), 2), np.nan, dtype=np.float64)
        sampled = counts > 0
        result[sampled, 0] = sum_score[sampled] / counts[sampled]
        result[sampled, 1] = sum_cost[sampled] / counts[sampled]
        return result

    def current_recommendation() -> tuple[int, ...]:
        estimated = estimated_vectors()
        sampled = np.flatnonzero(counts > 0)
        if sampled.size == 0:
            return ()
        local = pareto_min_cost_indices(estimated[sampled])
        return tuple(int(sampled[index]) for index in local)

    def append(event: str, selected: tuple[int, ...]) -> None:
        trajectory.append(
            _checkpoint_row(
                event=event,
                method=method,
                seed=seed,
                models=models,
                selected=selected,
                evaluations=evaluations,
                cost_usd=cumulative_cost,
                bruteforce_cost_usd=bruteforce_cost,
                truth_vectors=truth_vectors,
                true_front_vectors=true_front_vectors,
                ground_truth_hv=ground_truth_hv,
            )
        )

    def record_step() -> None:
        nonlocal last_selected, next_target, membership_checks, membership_changes
        nonlocal previous_update_snapshot
        selected = current_recommendation()
        fraction = cumulative_cost / bruteforce_cost
        current_snapshot = _checkpoint_row(
            event="policy_update",
            method=method,
            seed=seed,
            models=models,
            selected=selected,
            evaluations=evaluations,
            cost_usd=cumulative_cost,
            bruteforce_cost_usd=bruteforce_cost,
            truth_vectors=truth_vectors,
            true_front_vectors=true_front_vectors,
            ground_truth_hv=ground_truth_hv,
        )
        while next_target < len(targets) and fraction > targets[next_target] + 1e-12:
            assert previous_update_snapshot is not None
            target = targets[next_target]
            checkpoint = dict(previous_update_snapshot)
            checkpoint["event"] = f"cost_checkpoint_{int(round(100 * target))}pct"
            trajectory.append(checkpoint)
            next_target += 1
        membership_checks += 1
        if last_selected is None:
            append("recommendation_initial", selected)
            last_selected = selected
        elif selected != last_selected:
            append("recommendation_changed", selected)
            last_selected = selected
            membership_changes += 1
        while (
            next_target < len(targets)
            and abs(fraction - targets[next_target]) <= 1e-12
        ):
            target = targets[next_target]
            append(f"cost_checkpoint_{int(round(100 * target))}pct", selected)
            next_target += 1
        previous_update_snapshot = current_snapshot

    previous_update_snapshot = _checkpoint_row(
        event="initial_empty",
        method=method,
        seed=seed,
        models=models,
        selected=(),
        evaluations=0,
        cost_usd=0.0,
        bruteforce_cost_usd=bruteforce_cost,
        truth_vectors=truth_vectors,
        true_front_vectors=true_front_vectors,
        ground_truth_hv=ground_truth_hv,
    )

    if method == RANDOM_CONFIGURATIONS:
        for arm in rng.permutation(len(models)):
            arm = int(arm)
            for question in questions:
                score, cost, _ = _sample_values(table[models[arm]][question])
                sum_score[arm] += float(score)
                sum_cost[arm] += float(cost)
                counts[arm] += 1
                evaluations += 1
                cumulative_cost += float(cost)
            record_step()
    else:
        for question in rng.permutation(questions):
            for arm, model in enumerate(models):
                score, cost, _ = _sample_values(table[model][int(question)])
                sum_score[arm] += float(score)
                sum_cost[arm] += float(cost)
                counts[arm] += 1
                evaluations += 1
                cumulative_cost += float(cost)
            record_step()

    final_selected = current_recommendation()
    append("terminal", final_selected)
    final = trajectory[-1]
    summary: dict[str, object] = {
        "method": method,
        "seed": int(seed),
        "params": {
            "method": method,
            "cost_budget_axis": "realized_usd_over_bruteforce_usd",
            "recommendation_cost_checkpoint_fractions": list(targets),
            "recommendation_checkpoint_schema_version": 3,
            "recommendation_cost_checkpoint_semantics": (
                "latest_completed_policy_update_at_or_below_realized_usd_fraction"
            ),
            "bruteforce_search_cost_usd": bruteforce_cost,
            "n_models": len(models),
            "n_questions": len(questions),
            "cost_reference_usd": cost_reference,
            "recommendation_membership_checks": membership_checks,
            "recommendation_membership_changes": membership_changes,
        },
        "stop_reason": "full_replay_complete",
        "total_evaluations": evaluations,
        "total_search_cost_usd": cumulative_cost,
        "hypervolume": final["hypervolume"],
        "ground_truth_hypervolume": ground_truth_hv,
        "hypervolume_regret": final["hv_regret"],
        "generational_distance": final["generational_distance"],
        "inverted_generational_distance": final[
            "inverted_generational_distance"
        ],
        "selected_arm_indices": list(final_selected),
        "selected_models": [models[index] for index in final_selected],
        "trajectory_points": len(trajectory),
    }
    return CostBudgetRandomResult(
        summary=summary,
        trajectory=trajectory,
        estimated_raw_vectors=estimated_vectors(),
        truth_raw_vectors=truth_raw,
        truth_vectors=truth_vectors,
    )
