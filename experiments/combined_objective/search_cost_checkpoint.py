"""Post-hoc cost and Pareto snapshot at an actual-dollar search checkpoint.

The estimator uses the realized warm-start spend and the frozen per-arm
expected batch costs for later pulls. It is conditional on the realized pull
sequence; it is not a prediction of how many pulls the policy would take.
"""

from __future__ import annotations

import math
from typing import Any


class CheckpointNotReachedError(ValueError):
    """The replay ended before reaching the requested actual-dollar spend."""


def _physical_pulls(trace: list[dict[str, Any]]):
    for event in trace:
        if event.get("event") == "warm_start":
            yield True, int(event["arm_index"]), event
        elif event.get("event") == "direction_visit" and event.get("selected_arm") is not None:
            yield False, int(event["selected_arm"]), event


def actual_cost_checkpoint(
    result: Any,
    target_fraction: float = 0.1,
    *,
    acquisition_cost_mode: str = "real",
) -> dict[str, Any]:
    """Return the first post-warm state at or above the actual-dollar threshold.

    ``result`` must retain its physical trace and changes-only recommendation
    history. The full-matrix denominator is used only here, after the run, for
    evaluation. Warm-start charges are known exactly; subsequent batch charges
    are predicted from the frozen warm-start per-arm mean.
    """
    if not math.isfinite(target_fraction) or not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    if acquisition_cost_mode not in {"real", "unit"}:
        raise ValueError("acquisition_cost_mode must be 'real' or 'unit'")
    params = result.params
    if not params.get("record_trace") or not params.get("record_recommendation_trajectory"):
        raise ValueError("the checkpoint requires a physical trace and recommendation history")
    if not params.get("recommendation_changes_only"):
        raise ValueError("the checkpoint requires recommendation_changes_only=True")
    if not result.trace or result.recommendation_initial_snapshot is None:
        raise ValueError("missing physical pulls or warm-start recommendation")

    full_cost = float(params["bruteforce_search_cost_usd"])
    batch_size = int(params["batch_size"])
    acquisition_batch_costs = tuple(float(x) for x in params["expected_batch_costs_usd"])
    n_arms = len(result.model_results)
    if (not math.isfinite(full_cost) or full_cost <= 0.0 or batch_size <= 0
            or len(acquisition_batch_costs) != n_arms
            or any(not math.isfinite(cost) or cost < 0.0 for cost in acquisition_batch_costs)):
        raise ValueError("invalid full cost, batch size, or expected batch costs")
    if acquisition_cost_mode == "unit" and any(
        not math.isclose(cost, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for cost in acquisition_batch_costs
    ):
        raise ValueError("unit acquisition requires one unit per planned batch")

    counts = [0] * n_arms
    score_sums = [0.0] * n_arms
    cost_sums = [0.0] * n_arms
    actual_cost = estimated_cost = warm_cost = 0.0
    evaluations = 0
    pulls = list(_physical_pulls(result.trace))
    warm_pulls = [pull for pull in pulls if pull[0]]
    adaptive_pulls = [pull for pull in pulls if not pull[0]]
    if len(warm_pulls) != n_arms:
        raise ValueError("the estimator requires one recorded warm-start batch per arm")
    expected_batch_costs = list(acquisition_batch_costs)
    for _, arm, event in warm_pulls:
        warm_count = len(event["question_ids"])
        warm_batch_cost = float(event["actual_batch_search_cost_usd"])
        if warm_count <= 0 or not math.isfinite(warm_batch_cost) or warm_batch_cost < 0:
            raise ValueError("invalid warm-start batch count or realized cost")
        warm_expected = warm_batch_cost * batch_size / warm_count
        if acquisition_cost_mode == "unit":
            # Unit acquisition penalties are not dollar estimates. Keep the
            # same warm-start USD estimator used by the real-cost protocol.
            expected_batch_costs[arm] = warm_expected
        elif not math.isclose(
            expected_batch_costs[arm], warm_expected, rel_tol=1e-9, abs_tol=1e-10,
        ):
            raise ValueError("frozen expected batch costs are not the per-arm warm-start means")

    def consume(pull: tuple[bool, int, dict[str, Any]]) -> None:
        nonlocal actual_cost, estimated_cost, warm_cost, evaluations
        is_warm, arm, event = pull
        count = len(event["question_ids"])
        realized = float(event["actual_batch_search_cost_usd"])
        if count <= 0 or realized < 0.0:
            raise ValueError("physical pull has no questions or a negative cost")
        counts[arm] += count
        score_sums[arm] += count * float(event["batch_score_mean"])
        cost_sums[arm] += realized
        actual_cost += realized
        if is_warm:
            warm_cost += realized
            estimated_cost += realized
        else:
            predicted = expected_batch_costs[arm] * count / batch_size
            acquisition_predicted = acquisition_batch_costs[arm] * count / batch_size
            recorded = float(event["expected_batch_search_cost_usd"])
            if not math.isclose(acquisition_predicted, recorded, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("the recorded expected batch cost differs from the frozen estimate")
            estimated_cost += predicted
        evaluations += count
        if not math.isclose(actual_cost, float(event["cumulative_search_cost_usd"]),
                            rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("physical pull costs do not match recorded cumulative spend")
        if evaluations != int(event["cumulative_evaluations"]):
            raise ValueError("physical pull counts do not match recorded evaluations")

    for pull in warm_pulls:
        consume(pull)
    crossing_event = warm_pulls[-1][2] if actual_cost >= target_fraction * full_cost else None
    if crossing_event is None:
        for pull in adaptive_pulls:
            consume(pull)
            if actual_cost >= target_fraction * full_cost:
                crossing_event = pull[2]
                break
    if crossing_event is None:
        raise CheckpointNotReachedError(
            f"the run spent only {100 * result.total_cost / full_cost:.3f}% of full cost; "
            f"it never reached the requested {100 * target_fraction:g}% checkpoint"
        )

    snapshots = [result.recommendation_initial_snapshot, *result.recommendation_trajectory]
    eligible = [snapshot for snapshot in snapshots if snapshot.cumulative_evaluations <= evaluations]
    if not eligible:
        raise ValueError("no recommendation was recorded by the cost checkpoint")
    selected = tuple(int(arm) for arm in eligible[-1].selected_arm_indices)
    if any(counts[arm] == 0 for arm in selected):
        raise ValueError("a recommended arm has no observations at the checkpoint")

    return {
        "target_actual_search_cost_percent": 100.0 * target_fraction,
        "checkpoint_rule": "first post-warm completed physical batch reaching the actual-dollar threshold",
        "cost_estimator": "actual warm-start spend plus frozen warm-start per-arm predicted adaptive batch spend",
        "coordinate_estimator": "observed per-arm sample means at the checkpoint",
        "actual_search_cost_usd": actual_cost,
        "estimated_search_cost_usd": estimated_cost,
        "bruteforce_search_cost_usd": full_cost,
        "warm_start_search_cost_usd": warm_cost,
        "actual_search_cost_percent": 100.0 * actual_cost / full_cost,
        "estimated_search_cost_percent": 100.0 * estimated_cost / full_cost,
        "cumulative_evaluations": evaluations,
        "crossing_batch_arm_index": (
            int(crossing_event["arm_index"])
            if crossing_event["event"] == "warm_start"
            else int(crossing_event["selected_arm"])
        ),
        "selected_arm_indices": list(selected),
        "selected_model_names": [result.model_results[arm].model_name for arm in selected],
        "selected_sample_counts": [counts[arm] for arm in selected],
        "estimated_raw_archive_vectors": [
            [score_sums[arm] / counts[arm], cost_sums[arm] / counts[arm]]
            for arm in selected
        ],
        "offline_raw_selected_vectors": [
            [float(value) for value in result.raw_truth_vectors[arm]]
            for arm in selected
        ],
    }
