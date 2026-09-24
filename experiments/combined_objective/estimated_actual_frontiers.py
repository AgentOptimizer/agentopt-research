"""Freeze reproducible estimated/actual Pareto-frontier events.

The saved events are intentionally self-contained: every recommendation
membership change includes the selected arm IDs, sample counts, online sample
means, full-data means, and both estimated and realized cumulative search
spend.  A later animation therefore needs no policy replay.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_FRACTIONS = (0.10, 0.30)


def physical_pulls(trace: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Gittins and baseline physical traces to one schema."""
    pulls: list[dict[str, Any]] = []
    for index, event in enumerate(trace):
        kind = event.get("event")
        if kind == "direction_visit":
            if event.get("selected_arm") is None:
                continue
            arm = int(event["selected_arm"])
            warm = False
        elif kind == "warm_start":
            arm = int(event["arm_index"])
            warm = True
        elif "arm_index" in event and "question_ids" in event:
            arm = int(event["arm_index"])
            warm = False
        else:
            continue
        questions = [int(question) for question in event["question_ids"]]
        if not questions:
            raise ValueError("a physical pull must contain at least one question")
        pulls.append(
            {
                "trace_index": index,
                "arm_index": arm,
                "question_ids": questions,
                "batch_score_mean": float(event["batch_score_mean"]),
                "actual_batch_search_cost_usd": float(
                    event["actual_batch_search_cost_usd"]
                ),
                "cumulative_evaluations": int(event["cumulative_evaluations"]),
                "cumulative_search_cost_usd": float(
                    event["cumulative_search_cost_usd"]
                ),
                "is_warm_start": warm,
                "atomic_update_index": int(event.get("atomic_update_index", index)),
            }
        )
    if not pulls:
        raise ValueError("the result contains no physical pulls")
    return pulls


def _validate_snapshots(
    snapshots: Iterable[Mapping[str, Any]], n_arms: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: tuple[int, ...] | None = None
    for snapshot in snapshots:
        selected = tuple(int(arm) for arm in snapshot["selected_arm_indices"])
        if len(set(selected)) != len(selected) or any(
            arm < 0 or arm >= n_arms for arm in selected
        ):
            raise ValueError("snapshot selected_arm_indices are invalid")
        # Only membership changes are animation frames. A final duplicate is
        # metadata, not another visible frontier transition.
        if selected == previous:
            continue
        result.append(
            {
                "source_event": str(snapshot.get("event", "frontier_changed")),
                "source_role": str(snapshot.get("role", "frontier_changed")),
                "cumulative_evaluations": int(snapshot["cumulative_evaluations"]),
                "selected_arm_indices": selected,
            }
        )
        previous = selected
    if not result:
        raise ValueError("at least one recommendation snapshot is required")
    return result


def _replay_states(
    pulls: Sequence[Mapping[str, Any]],
    *,
    n_arms: int,
    full_search_cost_usd: float,
    estimator: str,
    expected_batch_costs_usd: Sequence[float] | None,
    nominal_batch_size: int | None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if not math.isfinite(full_search_cost_usd) or full_search_cost_usd <= 0:
        raise ValueError("full_search_cost_usd must be finite and positive")
    if estimator not in {"frozen_warm_arm_mean", "causal_observed_arm_mean"}:
        raise ValueError(f"unsupported cost estimator: {estimator}")
    if estimator == "frozen_warm_arm_mean":
        if nominal_batch_size is None or nominal_batch_size <= 0:
            raise ValueError("the frozen warm estimator requires nominal_batch_size")
        if expected_batch_costs_usd is None or len(expected_batch_costs_usd) != n_arms:
            raise ValueError("the frozen warm estimator requires one arm cost estimate")

    counts = np.zeros(n_arms, dtype=np.int64)
    score_sums = np.zeros(n_arms, dtype=np.float64)
    cost_sums = np.zeros(n_arms, dtype=np.float64)
    actual_cost = 0.0
    estimated_cost = 0.0
    evaluations = 0
    states: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for pull in pulls:
        arm = int(pull["arm_index"])
        questions = pull["question_ids"]
        count = len(questions)
        realized = float(pull["actual_batch_search_cost_usd"])
        if not 0 <= arm < n_arms or realized < 0 or not math.isfinite(realized):
            raise ValueError("physical trace contains an invalid arm or cost")

        if estimator == "frozen_warm_arm_mean":
            if pull["is_warm_start"]:
                predicted = realized
            else:
                assert expected_batch_costs_usd is not None
                assert nominal_batch_size is not None
                predicted = (
                    float(expected_batch_costs_usd[arm])
                    * count
                    / nominal_batch_size
                )
        elif counts[arm] > 0:
            predicted = count * cost_sums[arm] / counts[arm]
        elif evaluations > 0:
            predicted = count * actual_cost / evaluations
        else:
            # The first observation initializes the causal price estimator.
            predicted = realized

        counts[arm] += count
        score_sums[arm] += count * float(pull["batch_score_mean"])
        cost_sums[arm] += realized
        actual_cost += realized
        estimated_cost += predicted
        evaluations += count
        recorded_evaluations = int(pull["cumulative_evaluations"])
        recorded_cost = float(pull["cumulative_search_cost_usd"])
        if evaluations != recorded_evaluations:
            raise ValueError("physical trace cumulative evaluation counts disagree")
        if not math.isclose(actual_cost, recorded_cost, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("physical trace cumulative costs disagree")
        state = {
            "cumulative_evaluations": evaluations,
            "actual_search_cost_usd": actual_cost,
            "estimated_search_cost_usd": estimated_cost,
            "actual_search_cost_percent": 100.0 * actual_cost / full_search_cost_usd,
            "estimated_search_cost_percent": (
                100.0 * estimated_cost / full_search_cost_usd
            ),
            "counts": counts.copy(),
            "score_sums": score_sums.copy(),
            "cost_sums": cost_sums.copy(),
            "atomic_update_index": int(pull["atomic_update_index"]),
        }
        states[evaluations] = state
        ordered.append(state)
    return states, ordered


def _state_at_or_before(
    states: Mapping[int, Mapping[str, Any]], evaluations: int,
) -> Mapping[str, Any]:
    eligible = [count for count in states if count <= evaluations]
    if not eligible:
        raise ValueError("snapshot predates the first physical observation")
    return states[max(eligible)]


def _freeze_event(
    snapshot: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    event: str,
    models: Sequence[str],
    truth_raw_vectors: np.ndarray,
    target_fraction: float | None = None,
) -> dict[str, Any]:
    selected = [int(arm) for arm in snapshot["selected_arm_indices"]]
    counts = np.asarray(state["counts"], dtype=np.int64)
    score_sums = np.asarray(state["score_sums"], dtype=np.float64)
    cost_sums = np.asarray(state["cost_sums"], dtype=np.float64)
    if any(counts[arm] <= 0 for arm in selected):
        raise ValueError("a recommended arm has no observations at its snapshot")
    result = {
        "event": event,
        "source_event": snapshot.get("source_event"),
        "source_role": snapshot.get("source_role"),
        "cumulative_evaluations": int(state["cumulative_evaluations"]),
        "actual_search_cost_usd": float(state["actual_search_cost_usd"]),
        "estimated_search_cost_usd": float(state["estimated_search_cost_usd"]),
        "actual_search_cost_percent": float(state["actual_search_cost_percent"]),
        "estimated_search_cost_percent": float(
            state["estimated_search_cost_percent"]
        ),
        "selected_arm_indices": selected,
        "selected_model_names": [models[arm] for arm in selected],
        "selected_sample_counts": [int(counts[arm]) for arm in selected],
        "estimated_raw_archive_vectors": [
            [float(score_sums[arm] / counts[arm]), float(cost_sums[arm] / counts[arm])]
            for arm in selected
        ],
        "actual_raw_archive_vectors": truth_raw_vectors[selected].tolist(),
    }
    if target_fraction is not None:
        result["target_actual_search_cost_percent"] = 100.0 * target_fraction
    return result


def build_frontier_artifact(
    *,
    dataset: str,
    method: str,
    seed: int,
    models: Sequence[str],
    truth_raw_vectors: Sequence[Sequence[float]],
    full_data_pareto_arm_indices: Sequence[int],
    full_search_cost_usd: float,
    trace: Iterable[Mapping[str, Any]],
    snapshots: Iterable[Mapping[str, Any]],
    cost_estimator: str,
    expected_batch_costs_usd: Sequence[float] | None = None,
    nominal_batch_size: int | None = None,
    target_fractions: Sequence[float] = TARGET_FRACTIONS,
) -> dict[str, Any]:
    """Create membership-change frames and first-crossing cost checkpoints."""
    truth = np.asarray(truth_raw_vectors, dtype=np.float64)
    if truth.shape != (len(models), 2) or not np.all(np.isfinite(truth)):
        raise ValueError("truth_raw_vectors must be a finite n_arms-by-2 matrix")
    normalized_pulls = physical_pulls(trace)
    changes = _validate_snapshots(snapshots, len(models))
    states, ordered_states = _replay_states(
        normalized_pulls,
        n_arms=len(models),
        full_search_cost_usd=full_search_cost_usd,
        estimator=cost_estimator,
        expected_batch_costs_usd=expected_batch_costs_usd,
        nominal_batch_size=nominal_batch_size,
    )

    frames: list[dict[str, Any]] = []
    for index, snapshot in enumerate(changes):
        state = _state_at_or_before(states, int(snapshot["cumulative_evaluations"]))
        frame = _freeze_event(
            snapshot,
            state,
            event="frontier_initial" if index == 0 else "frontier_changed",
            models=models,
            truth_raw_vectors=truth,
        )
        frame["frame_index"] = index
        frames.append(frame)

    checkpoints: dict[str, dict[str, Any]] = {}
    for target in sorted({float(value) for value in target_fractions}):
        if not math.isfinite(target) or not 0 < target <= 1:
            raise ValueError("target fractions must lie in (0, 1]")
        crossing = next(
            (
                state
                for state in ordered_states
                if state["actual_search_cost_percent"] + 1e-10 >= 100.0 * target
            ),
            None,
        )
        if crossing is None:
            raise ValueError(
                f"{dataset}/{method} never reached {100 * target:g}% actual cost"
            )
        # A random-question policy step is atomic. If its threshold is crossed
        # mid-step, expose the state only after the complete step is available.
        atomic_index = int(crossing["atomic_update_index"])
        same_update = [
            state for state in ordered_states
            if int(state["atomic_update_index"]) == atomic_index
        ]
        if same_update:
            crossing = same_update[-1]
        available = [
            snapshot
            for snapshot in changes
            if int(snapshot["cumulative_evaluations"])
            <= int(crossing["cumulative_evaluations"])
        ]
        if not available:
            # Warm-up can cross an unusually small target before any selector
            # recommendation exists. Use the first post-warm state instead.
            snapshot = changes[0]
            crossing = _state_at_or_before(
                states, int(snapshot["cumulative_evaluations"])
            )
        else:
            snapshot = available[-1]
        tag = f"{int(round(100 * target))}pct"
        checkpoints[tag] = _freeze_event(
            snapshot,
            crossing,
            event=f"cost_checkpoint_{tag}",
            models=models,
            truth_raw_vectors=truth,
            target_fraction=target,
        )

    return {
        "schema_version": 1,
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "model_names": list(models),
        "truth_raw_vectors": truth.tolist(),
        "full_data_pareto_arm_indices": [
            int(arm) for arm in full_data_pareto_arm_indices
        ],
        "bruteforce_search_cost_usd": float(full_search_cost_usd),
        "cost_estimator": cost_estimator,
        "cost_estimator_semantics": (
            "warm-start spend is realized; later Gittins batches use the frozen "
            "per-arm warm-start mean"
            if cost_estimator == "frozen_warm_arm_mean"
            else "before each pull use that arm's observed mean cost; use the "
            "pooled observed mean for an unseen arm; initialize from the first pull"
        ),
        "frontier_change_frames": frames,
        "cost_checkpoints": checkpoints,
    }


__all__ = ["TARGET_FRACTIONS", "build_frontier_artifact", "physical_pulls"]
