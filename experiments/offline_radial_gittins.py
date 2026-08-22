#!/usr/bin/env python3
"""Offline replay for the cost-aware two-objective radial-Gittins selector.

The selector observes only cells it explicitly pulls from a frozen lookup
table.  Full-matrix objective vectors are computed after selection only for
evaluation metrics; they never enter calibration, indices, or stopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from . import offline_selector_sim_v2 as _offline_v2
except ImportError:  # Direct ``python experiments/offline_radial_gittins.py``.
    import offline_selector_sim_v2 as _offline_v2  # type: ignore[no-redef]

# Older local pickles record ``offline_selector_sim_v2.SampleResult`` because
# the loader was historically executed as a script. Preserve that import name
# when this module is imported through the ``experiments`` namespace package.
sys.modules.setdefault("offline_selector_sim_v2", _offline_v2)
LookupTable = _offline_v2.LookupTable
SampleResult = _offline_v2.SampleResult
_require_data_path = _offline_v2._require_data_path
load_jsonl = _offline_v2.load_jsonl
load_pickle = _offline_v2.load_pickle

from agentopt.model_selection.radial_gittins import (
    DEFAULT_DIRECTIONS,
    GaussianVectorPosterior,
    PerArmQuestionSchedule,
    fit_empirical_bayes_warm_start,
)
from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
    direction_aware_grid,
    radial_posterior_coordinates,
    terminal_expected_radial_utility,
)


OFFLINE_PRODUCTION_BASE_GRID = RadialGittinsGrid(
    z_size=513,
    delta_size=513,
    state_size=513,
    boundary_margin_cells=4,
)


@dataclass(frozen=True)
class DirectionWinner:
    direction: Tuple[float, float]
    model_name: str
    arm_index: int
    terminal_utility: float


@dataclass
class RadialArmSummary:
    model_name: str
    arm_index: int
    posterior_mean: Tuple[float, float]
    posterior_var: Tuple[float, float]
    observed_accuracy: float
    observed_mean_cost_usd: float
    observed_total_cost_usd: float
    equivalent_posterior_cost_usd: float
    n_samples_evaluated: int
    n_batches: int
    completed: bool
    is_direction_winner: bool = False
    is_nondominated: bool = False
    is_posterior_nondominated: bool = False
    is_oracle_raw_nondominated: bool = False


@dataclass(frozen=True)
class RecommendationCheckpoint:
    """Recommendation snapshot for budget-curve evaluation.

    Unlike the final completed-arm archive, this uses every warm-started arm's
    current posterior to identify direction winners (paper-style fixed-budget
    recommendation). ``selected_*`` is the leakage-free online raw archive.
    The posterior-desirability and full-data oracle raw archives are retained
    separately for comparison.
    """

    cumulative_evaluations: int
    cumulative_search_cost_usd: float
    budget_fraction: float
    selected_arm_indices: Tuple[int, ...]
    selected_models: Tuple[str, ...]
    direction_winner_arm_indices: Tuple[int, ...]
    estimated_raw_winner_vectors: Tuple[Tuple[float, float], ...]
    posterior_archive_arm_indices: Tuple[int, ...]
    posterior_archive_models: Tuple[str, ...]
    online_raw_archive_arm_indices: Tuple[int, ...]
    online_raw_archive_models: Tuple[str, ...]
    oracle_raw_winner_archive_arm_indices: Tuple[int, ...]
    oracle_raw_winner_archive_models: Tuple[str, ...]
    hypervolume: float
    hypervolume_regret: float
    event: str


@dataclass
class RadialSimulationResult:
    selector: str
    seed: int
    params: Dict[str, Any]
    selected_models: List[str]
    direction_winners: List[DirectionWinner]
    nondominated_models: List[str]
    total_evaluations: int
    total_cost: float
    cost_budget_guard: str
    cost_budget_overshoot_usd: float
    n_models_evaluated: int
    policy_wall_time_seconds: float
    cost_reference_usd: float
    prior_mean: Tuple[float, float]
    prior_variance: Tuple[float, float]
    stop_reason: str
    stopped_by_gittins: bool
    hypervolume: float
    ground_truth_hypervolume: float
    hypervolume_regret: float
    contains_true_accuracy_best: bool
    model_results: List[RadialArmSummary] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    observed_cells: Tuple[Tuple[int, int], ...] = ()
    recommendation_trajectory: List[RecommendationCheckpoint] = field(
        default_factory=list
    )
    gittins_stop_evaluations: Optional[int] = None
    gittins_stop_cost_usd: Optional[float] = None
    gittins_stop_budget_fraction: Optional[float] = None
    truth_vectors: Optional[np.ndarray] = None
    raw_truth_vectors: Optional[np.ndarray] = None
    posterior_archive_arm_indices: Tuple[int, ...] = ()
    posterior_archive_models: List[str] = field(default_factory=list)
    online_raw_archive_arm_indices: Tuple[int, ...] = ()
    online_raw_archive_models: List[str] = field(default_factory=list)
    oracle_raw_winner_archive_arm_indices: Tuple[int, ...] = ()
    oracle_raw_winner_archive_models: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DirectionVisitContext:
    global_step: int
    direction_index: int
    direction: Tuple[float, float]
    visit_count: int
    completed_arms: Tuple[int, ...]
    unfinished_arms: Tuple[int, ...]
    adaptive_pulls: Tuple[int, ...]
    model_names: Tuple[str, ...]


IndexProvider = Callable[[DirectionVisitContext, int], float]


@dataclass(frozen=True)
class DirectionStatus:
    should_stop: bool
    best_completed_arm: Optional[int]
    best_completed_index: float
    best_unfinished_arm: Optional[int]
    best_unfinished_index: float


def _validate_directions(
    directions: Iterable[Sequence[float]],
) -> Tuple[Tuple[float, float], ...]:
    resolved: List[Tuple[float, float]] = []
    for value in directions:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError("each direction must be a finite length-2 vector")
        if np.any(array <= 0.0) or not math.isclose(
            float(array.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("direction components must be positive and sum to one")
        resolved.append((float(array[0]), float(array[1])))
    if not resolved:
        raise ValueError("at least one direction is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("directions must be unique")
    return tuple(resolved)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _optional_finite(value: float) -> Optional[float]:
    """Represent an absent numerical index with JSON-safe ``None``."""
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _resolve_per_arm_costs(
    value: Optional[float | Sequence[float] | Mapping[Any, float]],
    *,
    models: Sequence[str],
    default: Optional[np.ndarray],
    name: str,
) -> Optional[np.ndarray]:
    """Resolve a scalar, arm vector, or model/index mapping to a cost vector."""
    if value is None:
        return None if default is None else np.asarray(default, dtype=np.float64).copy()

    if isinstance(value, Mapping):
        resolved_values: List[float] = []
        for arm_index, model_name in enumerate(models):
            has_model_key = model_name in value
            has_index_key = arm_index in value
            if not has_model_key and not has_index_key:
                raise ValueError(
                    f"{name} mapping is missing model {model_name!r} "
                    f"(arm index {arm_index})"
                )
            if has_model_key and has_index_key:
                model_value = float(value[model_name])
                index_value = float(value[arm_index])
                if model_value != index_value:
                    raise ValueError(
                        f"{name} gives conflicting values for model "
                        f"{model_name!r} and arm index {arm_index}"
                    )
            resolved_values.append(
                float(value[model_name] if has_model_key else value[arm_index])
            )
        array = np.asarray(resolved_values, dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.repeat(array, len(models))
        if array.shape != (len(models),):
            raise ValueError(
                f"{name} must be a scalar, length-{len(models)} vector, "
                "or complete model/index mapping"
            )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite positive costs")
    return array.astype(np.float64, copy=True)


def _quantize_effective_costs(
    raw_costs: np.ndarray,
    *,
    anchor: float,
    bin_ratio: Optional[float],
) -> np.ndarray:
    """Quantize positive costs to nearest multiplicative bins around *anchor*."""
    raw = np.asarray(raw_costs, dtype=np.float64)
    if bin_ratio is None:
        return raw.copy()
    ratio = float(bin_ratio)
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("effective_cost_bin_ratio must be None or greater than 1")
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise ValueError("effective cost bin anchor must be finite and positive")
    exponents = np.rint(np.log(raw / anchor) / math.log(ratio))
    return anchor * np.power(ratio, exponents)


def _sample_values(sample: SampleResult) -> Tuple[float, float, float]:
    score = float(sample.score)
    cost = float(sample.cost)
    latency = float(sample.latency_seconds)
    if not math.isfinite(score):
        raise ValueError("lookup-table scores must be finite")
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("lookup-table costs must be finite and nonnegative")
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("lookup-table latencies must be finite and nonnegative")
    return score, cost, latency


def _best_index(indices: Mapping[int, float], tolerance: float) -> Tuple[Optional[int], float]:
    best_arm: Optional[int] = None
    best_value = float("-inf")
    for arm_index in sorted(indices):
        value = float(indices[arm_index])
        if not math.isfinite(value):
            raise ValueError("direction indices must be finite")
        if value > best_value + tolerance:
            best_arm = arm_index
            best_value = value
    return best_arm, best_value


def evaluate_direction_status(
    *,
    direction: Tuple[float, float],
    posteriors: Mapping[int, GaussianVectorPosterior],
    completed_arms: Sequence[int],
    unfinished_indices: Mapping[int, float],
    reference_point: Sequence[float],
    stop_tolerance: float,
) -> DirectionStatus:
    """Combine completed terminal values and unfinished Gittins indices."""
    completed_indices = {
        arm_index: terminal_expected_radial_utility(
            posteriors[arm_index].mean,
            posteriors[arm_index].var,
            direction,
            reference_point,
        )
        for arm_index in completed_arms
    }
    best_completed_arm, best_completed = _best_index(
        completed_indices,
        stop_tolerance,
    )
    best_unfinished_arm, best_unfinished = _best_index(
        unfinished_indices,
        stop_tolerance,
    )
    should_stop = best_completed_arm is not None and (
        best_unfinished_arm is None
        or best_completed >= best_unfinished - stop_tolerance
    )
    return DirectionStatus(
        should_stop=should_stop,
        best_completed_arm=best_completed_arm,
        best_completed_index=best_completed,
        best_unfinished_arm=best_unfinished_arm,
        best_unfinished_index=best_unfinished,
    )


def nondominated_indices(points: np.ndarray) -> List[int]:
    """Return indices not strictly dominated under maximization of both columns."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    keep: List[int] = []
    for i in range(points.shape[0]):
        dominated = False
        for j in range(points.shape[0]):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def raw_nondominated_indices(points: np.ndarray) -> List[int]:
    """Return nondominated indices for maximize-accuracy/minimize-cost points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    maximize_points = np.column_stack((points[:, 0], -points[:, 1]))
    return nondominated_indices(maximize_points)


def raw_archive_arm_indices(
    candidate_arms: Sequence[int],
    raw_points: np.ndarray,
) -> List[int]:
    """Filter candidate arms in raw ``(accuracy, mean USD cost)`` space.

    ``raw_points`` is aligned with ``candidate_arms`` rather than indexed by
    global arm id. Candidate order is preserved in the returned archive.
    """
    arms = list(candidate_arms)
    points = np.asarray(raw_points, dtype=np.float64)
    if points.shape != (len(arms), 2):
        raise ValueError(
            "raw_points must have shape (len(candidate_arms), 2)"
        )
    return [arms[position] for position in raw_nondominated_indices(points)]


def hypervolume_2d(
    points: np.ndarray,
    reference: Sequence[float] = (0.0, 0.0),
) -> float:
    """Exact 2-D dominated hypervolume for maximization objectives."""
    points = np.asarray(points, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if points.size == 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 2 or reference_array.shape != (2,):
        raise ValueError("points/reference must be (n,2)/(2,)")
    eligible = points[np.all(points >= reference_array, axis=1)]
    if eligible.size == 0:
        return 0.0
    front = eligible[nondominated_indices(eligible)]
    front = front[np.argsort(front[:, 0])]
    suffix_y = np.maximum.accumulate(front[::-1, 1])[::-1]
    area = 0.0
    previous_x = float(reference_array[0])
    for point, max_y in zip(front, suffix_y):
        x = max(float(point[0]), previous_x)
        area += (x - previous_x) * max(0.0, float(max_y - reference_array[1]))
        previous_x = x
    return float(area)


def provisional_direction_winner_arms(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    reference_point: Sequence[float],
    stop_tolerance: float,
    candidate_arms: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return deduplicated direction winners over current posteriors.

    Includes unfinished arms. Each direction picks the arm with the largest
    terminal expected radial utility under the current posterior.
    """
    arm_indices = (
        list(candidate_arms)
        if candidate_arms is not None
        else sorted(posteriors.keys())
    )
    if not arm_indices:
        return []
    winners: List[int] = []
    for direction in directions:
        utilities = {
            arm_index: terminal_expected_radial_utility(
                posteriors[arm_index].mean,
                posteriors[arm_index].var,
                direction,
                reference_point,
            )
            for arm_index in arm_indices
        }
        winner_arm, _ = _best_index(utilities, stop_tolerance)
        if winner_arm is not None and winner_arm not in winners:
            winners.append(winner_arm)
    if not winners:
        return []
    return winners


def provisional_archive_from_posteriors(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    reference_point: Sequence[float],
    stop_tolerance: float,
    candidate_arms: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return direction winners nondominated in posterior-mean space."""
    winners = provisional_direction_winner_arms(
        posteriors=posteriors,
        directions=directions,
        reference_point=reference_point,
        stop_tolerance=stop_tolerance,
        candidate_arms=candidate_arms,
    )
    if not winners:
        return []
    winner_points = np.asarray(
        [posteriors[i].mean for i in winners],
        dtype=np.float64,
    )
    return [winners[position] for position in nondominated_indices(winner_points)]


def _recommendation_checkpoint(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    models: Sequence[str],
    reference_point: Sequence[float],
    stop_tolerance: float,
    truth_vectors: np.ndarray,
    raw_truth_vectors: np.ndarray,
    observed_scores: Mapping[int, Sequence[float]],
    observed_costs: Mapping[int, Sequence[float]],
    ground_truth_hv: float,
    total_evaluations: int,
    total_cost: float,
    n_available: int,
    event: str,
) -> RecommendationCheckpoint:
    winner_arms = provisional_direction_winner_arms(
        posteriors=posteriors,
        directions=directions,
        reference_point=reference_point,
        stop_tolerance=stop_tolerance,
    )
    posterior_points = np.asarray(
        [posteriors[i].mean for i in winner_arms],
        dtype=np.float64,
    ).reshape((-1, 2))
    posterior_archive_arms = [
        winner_arms[position]
        for position in nondominated_indices(posterior_points)
    ]
    estimated_raw_points = np.asarray(
        [
            (
                float(np.mean(observed_scores[i])),
                float(np.mean(observed_costs[i])),
            )
            for i in winner_arms
        ],
        dtype=np.float64,
    ).reshape((-1, 2))
    online_raw_archive_arms = raw_archive_arm_indices(
        winner_arms,
        estimated_raw_points,
    )
    oracle_raw_archive_arms = raw_archive_arm_indices(
        winner_arms,
        raw_truth_vectors[winner_arms],
    )
    selected_truth = (
        truth_vectors[online_raw_archive_arms]
        if online_raw_archive_arms
        else np.empty((0, 2), dtype=np.float64)
    )
    selected_hv = hypervolume_2d(selected_truth, reference_point)
    return RecommendationCheckpoint(
        cumulative_evaluations=int(total_evaluations),
        cumulative_search_cost_usd=float(total_cost),
        budget_fraction=float(total_evaluations) / float(n_available),
        selected_arm_indices=tuple(online_raw_archive_arms),
        selected_models=tuple(models[i] for i in online_raw_archive_arms),
        direction_winner_arm_indices=tuple(winner_arms),
        estimated_raw_winner_vectors=tuple(
            (float(point[0]), float(point[1])) for point in estimated_raw_points
        ),
        posterior_archive_arm_indices=tuple(posterior_archive_arms),
        posterior_archive_models=tuple(models[i] for i in posterior_archive_arms),
        online_raw_archive_arm_indices=tuple(online_raw_archive_arms),
        online_raw_archive_models=tuple(
            models[i] for i in online_raw_archive_arms
        ),
        oracle_raw_winner_archive_arm_indices=tuple(oracle_raw_archive_arms),
        oracle_raw_winner_archive_models=tuple(
            models[i] for i in oracle_raw_archive_arms
        ),
        hypervolume=float(selected_hv),
        hypervolume_regret=float(max(0.0, ground_truth_hv - selected_hv)),
        event=event,
    )


def _planning_horizon(actual_horizon: int, bin_width: int) -> int:
    if actual_horizon <= 0:
        return 0
    return int(math.ceil(actual_horizon / bin_width) * bin_width)


def _equivalent_cost(cost_reference_usd: float, desirability: float) -> float:
    if desirability <= 0.0:
        return float("inf")
    return float(cost_reference_usd * (1.0 - desirability) / desirability)


def _full_truth_vectors(
    models: Sequence[str],
    datapoints: Sequence[int],
    table: LookupTable,
    normalizer: Any,
) -> np.ndarray:
    vectors = np.empty((len(models), 2), dtype=np.float64)
    for arm_index, model_name in enumerate(models):
        normalized: List[np.ndarray] = []
        model_data = table.get(model_name, {})
        for question_id in datapoints:
            sample = model_data.get(question_id)
            if sample is None:
                continue
            score, cost, _ = _sample_values(sample)
            normalized.append(normalizer.normalize_batch([score], [cost])[0])
        if not normalized:
            raise ValueError(f"model {model_name!r} has no available observations")
        vectors[arm_index] = np.mean(np.asarray(normalized), axis=0)
    return vectors


def _full_raw_objective_vectors(
    models: Sequence[str],
    datapoints: Sequence[int],
    table: LookupTable,
) -> np.ndarray:
    """Return per-arm (accuracy, mean deployment cost USD) on ``datapoints``."""
    vectors = np.empty((len(models), 2), dtype=np.float64)
    for arm_index, model_name in enumerate(models):
        scores: List[float] = []
        costs: List[float] = []
        model_data = table.get(model_name, {})
        for question_id in datapoints:
            sample = model_data.get(question_id)
            if sample is None:
                continue
            score, cost, _ = _sample_values(sample)
            scores.append(score)
            costs.append(cost)
        if not scores:
            raise ValueError(f"model {model_name!r} has no available observations")
        vectors[arm_index, 0] = float(np.mean(scores))
        vectors[arm_index, 1] = float(np.mean(costs))
    return vectors


def simulate_radial_gittins(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    batch_size: int = 4,
    directions: Iterable[Sequence[float]] = DEFAULT_DIRECTIONS,
    prior_variance: Sequence[float] | float = 0.04,
    obs_noise_variance: Optional[Sequence[float] | float] = None,
    cost_reference_usd: Optional[float] = None,
    reference_point: Sequence[float] = (0.0, 0.0),
    search_cost_scale_eta: float = 1.0,
    expected_batch_cost_usd: Optional[
        float | Sequence[float] | Mapping[Any, float]
    ] = None,
    guaranteed_batch_cost_usd: Optional[
        float | Sequence[float] | Mapping[Any, float]
    ] = None,
    effective_cost_bin_ratio: Optional[float] = 2.0,
    effective_cost_bin_anchor: float = 1e-4,
    observation_budget_fraction: float = 1.0,
    max_total_question_evaluations: Optional[int] = None,
    max_search_cost_usd: Optional[float] = None,
    horizon_bin_width: int = 1,
    stop_tolerance: float = 1e-9,
    seed: int = 42,
    history: Optional[List[Dict[str, Any]]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    boundary_grid: Optional[RadialGittinsGrid] = None,
    boundary_cache: Optional[RadialGittinsBoundaryCache] = None,
    index_provider: Optional[IndexProvider] = None,
    question_universe: str = "common",
    halt_on_gittins_stop: bool = True,
    record_recommendation_trajectory: bool = False,
) -> RadialSimulationResult:
    """Replay the complete warm-start + round-robin radial-Gittins policy.

    The default benchmark universe is the complete question intersection, so
    every arm is judged on the same questions. ``question_universe='per_arm'``
    retains ragged arm-specific tails as an explicitly diagnostic mode.

    Adaptive replay uses full batches only. At most ``batch_size - 1`` tail
    cells per arm remain unused so every DP transition and posterior update has
    the same declared observation noise.

    The dollar-budget reservation is soft when it uses frozen warm-start
    expected costs: a realized batch can overshoot the cap. Supplying
    ``guaranteed_batch_cost_usd`` makes the reservation a hard bound, and a
    replayed batch that violates the claimed bound raises an error.
    """
    wall_start = time.perf_counter()
    batch_size = _positive_integer(batch_size, "batch_size")
    horizon_bin_width = _positive_integer(horizon_bin_width, "horizon_bin_width")
    resolved_directions = _validate_directions(directions)
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be nonempty and unique")
    if not datapoints or len(set(datapoints)) != len(datapoints):
        raise ValueError("datapoints must be nonempty and unique")
    if not math.isfinite(search_cost_scale_eta) or search_cost_scale_eta <= 0.0:
        raise ValueError("search_cost_scale_eta must be finite and positive")
    if not 0.0 < observation_budget_fraction <= 1.0:
        raise ValueError("observation_budget_fraction must lie in (0, 1]")
    if not math.isfinite(stop_tolerance) or stop_tolerance < 0.0:
        raise ValueError("stop_tolerance must be finite and nonnegative")
    if (
        not math.isfinite(effective_cost_bin_anchor)
        or effective_cost_bin_anchor <= 0.0
    ):
        raise ValueError("effective_cost_bin_anchor must be finite and positive")
    if question_universe not in {"common", "per_arm"}:
        raise ValueError("question_universe must be 'common' or 'per_arm'")
    reference_array = np.asarray(reference_point, dtype=np.float64)
    if reference_array.shape != (2,) or not np.all(np.isfinite(reference_array)):
        raise ValueError("reference_point must be a finite length-2 vector")
    resolved_reference = (float(reference_array[0]), float(reference_array[1]))
    # ``boundary_grid`` is a numerical safety envelope/resolution override.
    # The actual table grid is always tightened to the reachable normalized
    # objective box for each direction; otherwise even a large point count is
    # wasted over unreachable delta values and can under-resolve learning.
    base_boundary_grid = (
        OFFLINE_PRODUCTION_BASE_GRID
        if boundary_grid is None
        else boundary_grid
    )
    boundary_grid_mode = (
        "direction_aware_default"
        if boundary_grid is None
        else "direction_aware_custom_base"
    )

    n_arms = len(models)
    raw_available_by_arm: Dict[int, Tuple[int, ...]] = {}
    for arm_index, model_name in enumerate(models):
        available = tuple(
            int(question_id)
            for question_id in datapoints
            if question_id in table.get(model_name, {})
        )
        raw_available_by_arm[arm_index] = available

    common_questions = tuple(
        int(question_id)
        for question_id in datapoints
        if all(
            question_id in table.get(model_name, {})
            for model_name in models
        )
    )
    if len(common_questions) < batch_size:
        raise ValueError(
            "the complete question intersection is smaller than batch_size: "
            f"need {batch_size}, found {len(common_questions)}"
        )

    if question_universe == "common":
        available_by_arm = {
            arm_index: common_questions for arm_index in range(n_arms)
        }
        evaluation_datapoints = common_questions
    else:
        available_by_arm = raw_available_by_arm
        for arm_index, model_name in enumerate(models):
            if len(available_by_arm[arm_index]) < batch_size:
                raise ValueError(
                    f"model {model_name!r} has fewer than batch_size observations"
                )
        evaluation_datapoints = tuple(int(x) for x in datapoints)
    n_available = sum(len(ids) for ids in available_by_arm.values())

    schedule = PerArmQuestionSchedule.create_from_available(
        available_by_arm,
        warm_start_batch_size=batch_size,
        seed=seed,
    )
    warm_batches = schedule.take_uniform_warm_start()
    warm_required = n_arms * batch_size
    fraction_cap = int(math.ceil(observation_budget_fraction * n_available))
    question_cap = fraction_cap
    if max_total_question_evaluations is not None:
        explicit_cap = _positive_integer(
            max_total_question_evaluations,
            "max_total_question_evaluations",
        )
        question_cap = min(question_cap, explicit_cap)
    if question_cap < warm_required:
        raise ValueError(
            "question budget cannot fund the mandatory uniform warm start: "
            f"need {warm_required}, cap is {question_cap}"
        )
    if max_search_cost_usd is not None and (
        not math.isfinite(max_search_cost_usd) or max_search_cost_usd <= 0.0
    ):
        raise ValueError("max_search_cost_usd must be finite and positive")

    warm_scores = np.empty((n_arms, batch_size), dtype=np.float64)
    warm_costs = np.empty((n_arms, batch_size), dtype=np.float64)
    observed_scores: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
    observed_costs: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
    observed_latencies: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
    observed_cells: set[Tuple[int, int]] = set()
    total_cost = 0.0

    for arm_index in range(n_arms):
        model_data = table[models[arm_index]]
        for column, question_id in enumerate(warm_batches[arm_index]):
            cell = (arm_index, question_id)
            if cell in observed_cells:
                raise RuntimeError(f"duplicate replay cell {cell}")
            score, cost, latency = _sample_values(model_data[question_id])
            warm_scores[arm_index, column] = score
            warm_costs[arm_index, column] = cost
            observed_scores[arm_index].append(score)
            observed_costs[arm_index].append(cost)
            observed_latencies[arm_index].append(latency)
            observed_cells.add(cell)
            total_cost += cost

    if max_search_cost_usd is not None and total_cost > max_search_cost_usd:
        raise ValueError(
            "dollar budget cannot fund the mandatory warm start: "
            f"warm cost ${total_cost:.6f}, cap ${max_search_cost_usd:.6f}"
        )

    calibration = fit_empirical_bayes_warm_start(
        warm_scores,
        warm_costs,
        arm_ids=range(n_arms),
        question_ids=schedule.warm_start_question_ids,
        prior_variance=prior_variance,
        obs_noise_variance=obs_noise_variance,
        cost_reference_usd=cost_reference_usd,
    )
    posteriors = calibration.initialize_posteriors()
    total_evaluations = warm_required
    expected_batch_costs = _resolve_per_arm_costs(
        expected_batch_cost_usd,
        models=models,
        default=batch_size * calibration.raw_cost_means_usd,
        name="expected_batch_cost_usd",
    )
    assert expected_batch_costs is not None
    guaranteed_batch_costs = _resolve_per_arm_costs(
        guaranteed_batch_cost_usd,
        models=models,
        default=None,
        name="guaranteed_batch_cost_usd",
    )
    if guaranteed_batch_costs is not None:
        warm_batch_totals = warm_costs.sum(axis=1)
        violations = np.flatnonzero(
            warm_batch_totals > guaranteed_batch_costs
        )
        if violations.size:
            arm_index = int(violations[0])
            raise ValueError(
                "guaranteed_batch_cost_usd is violated by the warm batch for "
                f"model {models[arm_index]!r}: observed "
                f"${warm_batch_totals[arm_index]:.6f}, bound "
                f"${guaranteed_batch_costs[arm_index]:.6f}"
            )

    raw_effective_pull_costs = search_cost_scale_eta * expected_batch_costs
    effective_pull_costs = _quantize_effective_costs(
        raw_effective_pull_costs,
        anchor=effective_cost_bin_anchor,
        bin_ratio=effective_cost_bin_ratio,
    )

    # With the required-completion convention, an arm can need to pay its
    # pull cost at every remaining stage before becoming selectable.  The
    # stopping root can therefore be displaced by roughly ``H * c`` for an
    # expensive arm.  Resolve grids per (direction, horizon, cost-bin) and
    # include that displacement in the root-search band; a direction-only
    # grid is too narrow for the highest real MathQA cost bins.
    resolved_boundary_grids: Dict[
        Tuple[Tuple[float, float], int, float], RadialGittinsGrid
    ] = {}

    def grid_for_arm(
        direction: Tuple[float, float],
        arm_index: int,
    ) -> RadialGittinsGrid:
        horizon = int(planning_horizons[arm_index])
        effective_cost = float(effective_pull_costs[arm_index])
        key = (direction, horizon, effective_cost)
        existing = resolved_boundary_grids.get(key)
        if existing is not None:
            return existing
        z_padding = max(1.0, horizon * effective_cost + 1.0)
        # The base grid supplies resolution and a normal safety envelope.  It
        # is expanded only when the mathematically required cumulative-cost
        # band would exceed that envelope.
        expanded_base = replace(
            base_boundary_grid,
            z_min=min(base_boundary_grid.z_min, -6.0 - z_padding),
            z_max=max(base_boundary_grid.z_max, 6.0 + z_padding),
        )
        resolved = direction_aware_grid(
            direction,
            base_grid=expanded_base,
            reference=resolved_reference,
            z_padding=z_padding,
        )
        resolved_boundary_grids[key] = resolved
        return resolved
    reservation_costs = (
        guaranteed_batch_costs
        if guaranteed_batch_costs is not None
        else expected_batch_costs
    )
    cost_budget_guard = (
        "hard_guaranteed_bound"
        if guaranteed_batch_costs is not None
        else "soft_expected_cost"
    )

    actual_horizons = np.asarray(
        [schedule.remaining(i) // batch_size for i in range(n_arms)],
        dtype=np.int64,
    )
    ragged_tail_cells_excluded = int(
        sum(schedule.remaining(i) % batch_size for i in range(n_arms))
    )
    planning_horizons = np.asarray(
        [
            _planning_horizon(int(horizon), horizon_bin_width)
            for horizon in actual_horizons
        ],
        dtype=np.int64,
    )
    adaptive_pulls = np.zeros(n_arms, dtype=np.int64)
    initial_var = next(iter(posteriors.values())).var.copy()
    noise_var = calibration.warm_obs_noise_var.copy()
    cache = (
        boundary_cache
        if boundary_cache is not None
        else RadialGittinsBoundaryCache()
    )
    cache_size_before = len(cache)
    trace: List[Dict[str, Any]] = []

    for arm_index in range(n_arms):
        trace.append(
            {
                "event": "warm_start",
                "arm_index": arm_index,
                "model_name": models[arm_index],
                "question_ids": list(warm_batches[arm_index]),
                "batch_score_mean": float(np.mean(warm_scores[arm_index])),
                "batch_deployment_cost_mean_usd": float(
                    np.mean(warm_costs[arm_index])
                ),
                "actual_batch_search_cost_usd": float(
                    np.sum(warm_costs[arm_index])
                ),
                "expected_batch_search_cost_usd": float(
                    expected_batch_costs[arm_index]
                ),
                "raw_effective_pull_cost": float(
                    raw_effective_pull_costs[arm_index]
                ),
                "quantized_effective_pull_cost": float(
                    effective_pull_costs[arm_index]
                ),
                "posterior_mean_after": posteriors[arm_index].mean.tolist(),
                "posterior_var_after": posteriors[arm_index].var.tolist(),
                "cumulative_evaluations": (arm_index + 1) * batch_size,
                "cumulative_search_cost_usd": float(
                    np.sum(warm_costs[: arm_index + 1])
                ),
            }
        )

    direction_index = 0
    skipped_since_last_evaluation = 0
    visit_counts = np.zeros(len(resolved_directions), dtype=np.int64)
    global_step = 0
    stop_reason = "all_directions_gittins_stop"
    gittins_stop_evaluations: Optional[int] = None
    gittins_stop_cost_usd: Optional[float] = None
    recommendation_trajectory: List[RecommendationCheckpoint] = []
    past_gittins_stop = False

    # Truth metrics are diagnostics for budget curves / final HV. Compute them
    # before the adaptive loop so trajectory checkpoints can reuse the vectors
    # without re-scanning the lookup table after every pull.
    truth_metric_start = time.perf_counter()
    truth_vectors = _full_truth_vectors(
        models,
        evaluation_datapoints,
        table,
        calibration.normalizer,
    )
    raw_truth_vectors = _full_raw_objective_vectors(
        models,
        evaluation_datapoints,
        table,
    )
    truth_front = truth_vectors[nondominated_indices(truth_vectors)]
    ground_truth_hv = hypervolume_2d(truth_front, resolved_reference)
    wall_start += time.perf_counter() - truth_metric_start

    def _append_recommendation_checkpoint(event: str) -> None:
        if not record_recommendation_trajectory:
            return
        recommendation_trajectory.append(
            _recommendation_checkpoint(
                posteriors=posteriors,
                directions=resolved_directions,
                models=models,
                reference_point=resolved_reference,
                stop_tolerance=stop_tolerance,
                truth_vectors=truth_vectors,
                raw_truth_vectors=raw_truth_vectors,
                observed_scores=observed_scores,
                observed_costs=observed_costs,
                ground_truth_hv=ground_truth_hv,
                total_evaluations=total_evaluations,
                total_cost=total_cost,
                n_available=n_available,
                event=event,
            )
        )

    _append_recommendation_checkpoint("after_warm_start")

    while True:
        completed = tuple(
            i for i in range(n_arms) if adaptive_pulls[i] >= actual_horizons[i]
        )
        unfinished = tuple(i for i in range(n_arms) if i not in completed)
        if not unfinished:
            stop_reason = "all_arms_completed"
            break

        # These checks happen before constructing any DP tables or indices.
        # An arm-specific dollar reservation still happens after an arm is
        # proposed, because different arms have different frozen pull costs.
        if total_evaluations + batch_size > question_cap:
            trace.append(
                {
                    "event": "budget_stop",
                    "reason": "question_budget",
                    "cumulative_evaluations": total_evaluations,
                    "cumulative_search_cost_usd": total_cost,
                }
            )
            stop_reason = "question_budget"
            break
        if (
            max_search_cost_usd is not None
            and total_cost >= max_search_cost_usd
        ):
            trace.append(
                {
                    "event": "budget_stop",
                    "reason": "search_cost_budget",
                    "cost_budget_guard": cost_budget_guard,
                    "cost_budget_overshoot_usd": max(
                        0.0, total_cost - max_search_cost_usd
                    ),
                    "cumulative_evaluations": total_evaluations,
                    "cumulative_search_cost_usd": total_cost,
                }
            )
            stop_reason = "search_cost_budget"
            break

        direction = resolved_directions[direction_index]
        context = DirectionVisitContext(
            global_step=global_step,
            direction_index=direction_index,
            direction=direction,
            visit_count=int(visit_counts[direction_index]),
            completed_arms=completed,
            unfinished_arms=unfinished,
            adaptive_pulls=tuple(int(x) for x in adaptive_pulls),
            model_names=tuple(models),
        )
        visit_counts[direction_index] += 1

        unfinished_indices: Dict[int, float] = {}
        for arm_index in unfinished:
            if index_provider is not None:
                index = float(index_provider(context, arm_index))
            else:
                table_for_arm = cache.get(
                    direction=direction,
                    effective_pull_cost=float(effective_pull_costs[arm_index]),
                    initial_var=initial_var,
                    obs_noise_var=noise_var,
                    horizon=int(planning_horizons[arm_index]),
                    grid=grid_for_arm(direction, arm_index),
                )
                u, delta, _ = radial_posterior_coordinates(
                    posteriors[arm_index].mean,
                    posteriors[arm_index].var,
                    direction,
                    resolved_reference,
                )
                boundary = table_for_arm.boundary(
                    int(adaptive_pulls[arm_index]),
                    delta,
                )
                index = u - boundary
            unfinished_indices[arm_index] = index

        status = evaluate_direction_status(
            direction=direction,
            posteriors=posteriors,
            completed_arms=completed,
            unfinished_indices=unfinished_indices,
            reference_point=resolved_reference,
            stop_tolerance=stop_tolerance,
        )
        visit_event: Dict[str, Any] = {
            "event": "direction_visit",
            "global_step": global_step,
            "direction_index": direction_index,
            "direction": list(direction),
            "direction_should_stop": status.should_stop,
            "best_completed_arm": status.best_completed_arm,
            "best_completed_model": (
                models[status.best_completed_arm]
                if status.best_completed_arm is not None
                else None
            ),
            "best_completed_terminal_index": _optional_finite(
                status.best_completed_index
            ),
            "best_unfinished_arm": status.best_unfinished_arm,
            "best_unfinished_model": (
                models[status.best_unfinished_arm]
                if status.best_unfinished_arm is not None
                else None
            ),
            "best_unfinished_gittins_index": _optional_finite(
                status.best_unfinished_index
            ),
            "selected_arm": None,
            "selected_model": None,
            "question_ids": [],
            "actual_batch_search_cost_usd": 0.0,
            "cumulative_evaluations": total_evaluations,
            "cumulative_search_cost_usd": total_cost,
        }

        if status.should_stop and not past_gittins_stop:
            skipped_since_last_evaluation += 1
            if skipped_since_last_evaluation == len(resolved_directions):
                if gittins_stop_evaluations is None:
                    gittins_stop_evaluations = int(total_evaluations)
                    gittins_stop_cost_usd = float(total_cost)
                    _append_recommendation_checkpoint("gittins_stop")
                if halt_on_gittins_stop:
                    trace.append(visit_event)
                    global_step += 1
                    stop_reason = "all_directions_gittins_stop"
                    break
                past_gittins_stop = True
                skipped_since_last_evaluation = 0
                # Fall through and force-pull under the remaining budget.
            else:
                trace.append(visit_event)
                global_step += 1
                direction_index = (direction_index + 1) % len(resolved_directions)
                continue

        selected_arm = status.best_unfinished_arm
        if selected_arm is None:
            trace.append(visit_event)
            stop_reason = "all_arms_completed"
            break
        reservation_cost = float(reservation_costs[selected_arm])
        visit_event.update(
            {
                "candidate_arm": selected_arm,
                "candidate_model": models[selected_arm],
                "expected_batch_search_cost_usd": float(
                    expected_batch_costs[selected_arm]
                ),
                "budget_reservation_cost_usd": reservation_cost,
                "cost_budget_guard": cost_budget_guard,
                "raw_effective_pull_cost": float(
                    raw_effective_pull_costs[selected_arm]
                ),
                "quantized_effective_pull_cost": float(
                    effective_pull_costs[selected_arm]
                ),
                "forced_after_gittins_stop": past_gittins_stop,
            }
        )
        if max_search_cost_usd is not None and (
            total_cost + reservation_cost > max_search_cost_usd
        ):
            trace.append(visit_event)
            stop_reason = "search_cost_budget"
            break
        if schedule.remaining(selected_arm) < batch_size:
            raise RuntimeError("unfinished arm cannot supply one full planned batch")

        question_ids = schedule.next_batch(selected_arm, batch_size)
        if len(question_ids) != batch_size:
            raise RuntimeError("adaptive replay produced a partial batch")
        batch_scores: List[float] = []
        batch_costs: List[float] = []
        batch_latencies: List[float] = []
        model_data = table[models[selected_arm]]
        for question_id in question_ids:
            cell = (selected_arm, question_id)
            if cell in observed_cells:
                raise RuntimeError(f"duplicate replay cell {cell}")
            score, cost, latency = _sample_values(model_data[question_id])
            batch_scores.append(score)
            batch_costs.append(cost)
            batch_latencies.append(latency)

        realized_batch_cost = float(sum(batch_costs))
        if (
            guaranteed_batch_costs is not None
            and realized_batch_cost
            > guaranteed_batch_costs[selected_arm]
        ):
            raise ValueError(
                "guaranteed_batch_cost_usd is violated by an adaptive batch for "
                f"model {models[selected_arm]!r}: observed "
                f"${realized_batch_cost:.6f}, bound "
                f"${guaranteed_batch_costs[selected_arm]:.6f}"
            )
        for question_id in question_ids:
            observed_cells.add((selected_arm, question_id))

        normalized_observation = calibration.normalizer.normalize_batch(
            batch_scores,
            batch_costs,
        ).mean(axis=0)
        posterior = posteriors[selected_arm]
        mean_before = posterior.mean.copy()
        var_before = posterior.var.copy()
        posterior.update(
            normalized_observation,
            noise_var,
            batch_size=batch_size,
        )
        adaptive_pulls[selected_arm] += 1
        observed_scores[selected_arm].extend(batch_scores)
        observed_costs[selected_arm].extend(batch_costs)
        observed_latencies[selected_arm].extend(batch_latencies)
        total_cost += realized_batch_cost
        total_evaluations += batch_size

        visit_event.update(
            {
                "selected_arm": selected_arm,
                "selected_model": models[selected_arm],
                "question_ids": list(question_ids),
                "batch_score_mean": float(np.mean(batch_scores)),
                "batch_deployment_cost_mean_usd": float(np.mean(batch_costs)),
                "actual_batch_search_cost_usd": realized_batch_cost,
                "posterior_mean_before": mean_before.tolist(),
                "posterior_mean_after": posterior.mean.tolist(),
                "posterior_var_before": var_before.tolist(),
                "posterior_var_after": posterior.var.tolist(),
                "cumulative_evaluations": total_evaluations,
                "cumulative_search_cost_usd": total_cost,
                "cost_budget_overshoot_usd_after": (
                    max(0.0, total_cost - max_search_cost_usd)
                    if max_search_cost_usd is not None
                    else 0.0
                ),
            }
        )
        trace.append(visit_event)
        if history is not None:
            history.append(dict(visit_event))
        _append_recommendation_checkpoint("adaptive_pull")
        skipped_since_last_evaluation = 0
        global_step += 1
        direction_index = (direction_index + 1) % len(resolved_directions)

    completed_final = [
        i for i in range(n_arms) if adaptive_pulls[i] >= actual_horizons[i]
    ]
    direction_winners: List[DirectionWinner] = []
    for direction in resolved_directions:
        if not completed_final:
            break
        utilities = {
            arm_index: terminal_expected_radial_utility(
                posteriors[arm_index].mean,
                posteriors[arm_index].var,
                direction,
                resolved_reference,
            )
            for arm_index in completed_final
        }
        winner_arm, utility = _best_index(utilities, stop_tolerance)
        assert winner_arm is not None
        direction_winners.append(
            DirectionWinner(
                direction=direction,
                model_name=models[winner_arm],
                arm_index=winner_arm,
                terminal_utility=utility,
            )
        )

    unique_winner_arms: List[int] = []
    for winner in direction_winners:
        if winner.arm_index not in unique_winner_arms:
            unique_winner_arms.append(winner.arm_index)
    if unique_winner_arms:
        winner_points = np.asarray(
            [posteriors[i].mean for i in unique_winner_arms],
            dtype=np.float64,
        )
        posterior_archive_positions = nondominated_indices(winner_points)
        posterior_archive_arms = [
            unique_winner_arms[position]
            for position in posterior_archive_positions
        ]
        estimated_raw_points = np.asarray(
            [
                (
                    float(np.mean(observed_scores[i])),
                    float(np.mean(observed_costs[i])),
                )
                for i in unique_winner_arms
            ],
            dtype=np.float64,
        )
        online_raw_archive_arms = raw_archive_arm_indices(
            unique_winner_arms,
            estimated_raw_points,
        )
        oracle_raw_archive_arms = raw_archive_arm_indices(
            unique_winner_arms,
            raw_truth_vectors[unique_winner_arms],
        )
    else:
        posterior_archive_arms = []
        online_raw_archive_arms = []
        oracle_raw_archive_arms = []
    # The deployable recommendation is filtered only with observations that
    # the selector actually acquired. The oracle archive is diagnostic-only.
    archive_arms = online_raw_archive_arms
    selected_models = [models[i] for i in online_raw_archive_arms]

    model_results: List[RadialArmSummary] = []
    winner_arm_set = set(unique_winner_arms)
    archive_arm_set = set(online_raw_archive_arms)
    posterior_archive_arm_set = set(posterior_archive_arms)
    oracle_raw_archive_arm_set = set(oracle_raw_archive_arms)
    for arm_index, model_name in enumerate(models):
        posterior = posteriors[arm_index]
        costs = observed_costs[arm_index]
        scores = observed_scores[arm_index]
        model_results.append(
            RadialArmSummary(
                model_name=model_name,
                arm_index=arm_index,
                posterior_mean=tuple(float(x) for x in posterior.mean),
                posterior_var=tuple(float(x) for x in posterior.var),
                observed_accuracy=float(np.mean(scores)),
                observed_mean_cost_usd=float(np.mean(costs)),
                observed_total_cost_usd=float(sum(costs)),
                equivalent_posterior_cost_usd=_equivalent_cost(
                    calibration.cost_reference_usd,
                    float(posterior.mean[1]),
                ),
                n_samples_evaluated=len(scores),
                n_batches=posterior.n_batches,
                completed=arm_index in completed_final,
                is_direction_winner=arm_index in winner_arm_set,
                is_nondominated=arm_index in archive_arm_set,
                is_posterior_nondominated=arm_index in posterior_archive_arm_set,
                is_oracle_raw_nondominated=arm_index in oracle_raw_archive_arm_set,
            )
        )

    # Stop the policy timer before any leftover final-archive bookkeeping.
    # Truth vectors / ground-truth HV were already computed above for trajectory
    # diagnostics and excluded from the policy wall clock.
    policy_wall_time = float(time.perf_counter() - wall_start)

    best_truth_accuracy = float(np.max(truth_vectors[:, 0]))
    truth_accuracy_best = set(
        int(i)
        for i in np.flatnonzero(
            np.isclose(
                truth_vectors[:, 0],
                best_truth_accuracy,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    )
    stopped_by_gittins = gittins_stop_evaluations is not None or stop_reason in {
        "all_directions_gittins_stop",
        "all_arms_completed",
    }
    gittins_stop_budget_fraction = (
        float(gittins_stop_evaluations) / float(n_available)
        if gittins_stop_evaluations is not None
        else None
    )
    if record_recommendation_trajectory:
        _append_recommendation_checkpoint("final")
    selected_truth = (
        truth_vectors[archive_arms]
        if archive_arms
        else np.empty((0, 2), dtype=np.float64)
    )
    selected_hv = hypervolume_2d(selected_truth, resolved_reference)
    if selected_hv > ground_truth_hv + max(stop_tolerance, 1e-12):
        raise RuntimeError(
            "selected-set hypervolume exceeds the full ground-truth front; "
            "check the evaluation universe and reference point"
        )
    cost_budget_overshoot = (
        max(0.0, total_cost - max_search_cost_usd)
        if max_search_cost_usd is not None
        else 0.0
    )

    params: Dict[str, Any] = {
        "batch_size": batch_size,
        "recommendation_space": "observed_raw_accuracy_mean_cost_usd",
        "posterior_archive_space": "normalized_posterior_mean_desirability",
        "oracle_raw_winner_archive_is_diagnostic": True,
        "directions": [list(x) for x in resolved_directions],
        "prior_variance": calibration.prior_var.tolist(),
        "obs_noise_variance": calibration.warm_obs_noise_var.tolist(),
        "cost_reference_usd": calibration.cost_reference_usd,
        "reference_point": list(resolved_reference),
        "search_cost_scale_eta": search_cost_scale_eta,
        "expected_batch_costs_usd": expected_batch_costs.tolist(),
        "guaranteed_batch_costs_usd": (
            guaranteed_batch_costs.tolist()
            if guaranteed_batch_costs is not None
            else None
        ),
        "cost_budget_guard": cost_budget_guard,
        "cost_budget_is_hard": guaranteed_batch_costs is not None,
        "cost_budget_overshoot_usd": cost_budget_overshoot,
        "effective_cost_bin_ratio": effective_cost_bin_ratio,
        "effective_cost_bin_anchor": effective_cost_bin_anchor,
        "raw_effective_pull_costs": raw_effective_pull_costs.tolist(),
        "quantized_effective_pull_costs": effective_pull_costs.tolist(),
        "observation_budget_fraction": observation_budget_fraction,
        "max_total_question_evaluations": max_total_question_evaluations,
        "max_search_cost_usd": max_search_cost_usd,
        "halt_on_gittins_stop": halt_on_gittins_stop,
        "record_recommendation_trajectory": record_recommendation_trajectory,
        "horizon_bin_width": horizon_bin_width,
        "actual_horizons": actual_horizons.tolist(),
        "planning_horizons": planning_horizons.tolist(),
        "question_universe": question_universe,
        "common_question_count": len(common_questions),
        "benchmark_question_count": (
            len(evaluation_datapoints)
            if question_universe == "common"
            else None
        ),
        "available_cells_in_universe": n_available,
        "ragged_tail_cells_excluded": ragged_tail_cells_excluded,
        "unobserved_cells_at_stop": int(
            sum(schedule.remaining(i) for i in range(n_arms))
        ),
        "boundary_grid_mode": boundary_grid_mode,
        "boundary_grids": [
            {
                "direction": list(direction),
                "horizon": horizon,
                "effective_pull_cost": effective_cost,
                "grid": asdict(grid),
            }
            for (direction, horizon, effective_cost), grid in sorted(
                resolved_boundary_grids.items()
            )
        ],
    }
    result = RadialSimulationResult(
        selector="radial_gittins",
        seed=seed,
        params=params,
        selected_models=selected_models,
        direction_winners=direction_winners,
        nondominated_models=list(selected_models),
        total_evaluations=total_evaluations,
        total_cost=total_cost,
        cost_budget_guard=cost_budget_guard,
        cost_budget_overshoot_usd=cost_budget_overshoot,
        n_models_evaluated=sum(bool(observed_scores[i]) for i in range(n_arms)),
        policy_wall_time_seconds=policy_wall_time,
        cost_reference_usd=calibration.cost_reference_usd,
        prior_mean=tuple(float(x) for x in calibration.prior_mean),
        prior_variance=tuple(float(x) for x in calibration.prior_var),
        stop_reason=stop_reason,
        stopped_by_gittins=stopped_by_gittins,
        hypervolume=selected_hv,
        ground_truth_hypervolume=ground_truth_hv,
        hypervolume_regret=max(0.0, ground_truth_hv - selected_hv),
        contains_true_accuracy_best=bool(
            truth_accuracy_best.intersection(archive_arm_set)
        ),
        model_results=model_results,
        trace=trace,
        observed_cells=tuple(sorted(observed_cells)),
        recommendation_trajectory=recommendation_trajectory,
        gittins_stop_evaluations=gittins_stop_evaluations,
        gittins_stop_cost_usd=gittins_stop_cost_usd,
        gittins_stop_budget_fraction=gittins_stop_budget_fraction,
        truth_vectors=truth_vectors,
        raw_truth_vectors=raw_truth_vectors,
        posterior_archive_arm_indices=tuple(posterior_archive_arms),
        posterior_archive_models=[models[i] for i in posterior_archive_arms],
        online_raw_archive_arm_indices=tuple(online_raw_archive_arms),
        online_raw_archive_models=list(selected_models),
        oracle_raw_winner_archive_arm_indices=tuple(oracle_raw_archive_arms),
        oracle_raw_winner_archive_models=[
            models[i] for i in oracle_raw_archive_arms
        ],
    )
    if run_metadata is not None:
        run_metadata.update(
            {
                "stop_reason": stop_reason,
                "stopped_by_gittins": stopped_by_gittins,
                "selected_models": list(selected_models),
                "posterior_archive_models": [
                    models[i] for i in posterior_archive_arms
                ],
                "oracle_raw_winner_archive_models": [
                    models[i] for i in oracle_raw_archive_arms
                ],
                "total_evaluations": total_evaluations,
                "total_cost": total_cost,
                "cost_budget_guard": cost_budget_guard,
                "cost_budget_overshoot_usd": cost_budget_overshoot,
                "cost_reference_usd": calibration.cost_reference_usd,
                "prior_mean": calibration.prior_mean.tolist(),
                "boundary_tables_built": len(cache) - cache_size_before,
                "boundary_tables_cached_total": len(cache),
                "policy_wall_time_seconds": policy_wall_time,
            }
        )
    return result


def summarize_radial_multi_seed(
    results: Sequence[RadialSimulationResult],
) -> Dict[str, Any]:
    if not results:
        raise ValueError("at least one result is required")
    return {
        "selector": "radial_gittins",
        "n_seeds": len(results),
        "mean_hypervolume_regret": float(
            np.mean([result.hypervolume_regret for result in results])
        ),
        "mean_returned_cardinality": float(
            np.mean([len(result.selected_models) for result in results])
        ),
        "mean_evaluations": float(
            np.mean([result.total_evaluations for result in results])
        ),
        "mean_cost": float(np.mean([result.total_cost for result in results])),
        "mean_cost_budget_overshoot_usd": float(
            np.mean([result.cost_budget_overshoot_usd for result in results])
        ),
        "gittins_stop_pct": 100.0
        * sum(result.stopped_by_gittins for result in results)
        / len(results),
        "contains_accuracy_best_pct": 100.0
        * sum(result.contains_true_accuracy_best for result in results)
        / len(results),
        # Warm-start calibration is seed dependent. Keeping every run's fitted
        # parameters avoids presenting the first seed's C_ref/prior/cost bins
        # as if they applied to the whole experiment.
        "params_by_seed": [
            {"seed": result.seed, "params": result.params} for result in results
        ],
    }


def print_radial_result(result: RadialSimulationResult) -> None:
    print(f"\n{'=' * 72}")
    print(f"radial_gittins (seed={result.seed})")
    print(f"stop={result.stop_reason}, C_ref=${result.cost_reference_usd:.6g}")
    print(
        f"evaluations={result.total_evaluations}, cost=${result.total_cost:.6f}, "
        f"HV regret={result.hypervolume_regret:.6f}"
    )
    if result.params["max_search_cost_usd"] is not None:
        print(
            f"cost guard={result.cost_budget_guard}, "
            f"overshoot=${result.cost_budget_overshoot_usd:.6f}"
        )
    print("direction winners:")
    for winner in result.direction_winners:
        print(
            f"  {winner.direction}: {winner.model_name} "
            f"(terminal={winner.terminal_utility:.6f})"
        )
    print(f"online raw-space recommendation: {result.selected_models}")
    print(f"posterior-desirability archive: {result.posterior_archive_models}")
    print(
        "offline oracle raw winner archive: "
        f"{result.oracle_raw_winner_archive_models}"
    )


def _jsonable_result(result: RadialSimulationResult) -> Dict[str, Any]:
    def json_safe(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, np.floating):
            resolved = float(value)
            return resolved if math.isfinite(resolved) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    return json_safe(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pickle", help="Path to a cached lookup pickle")
    source.add_argument("--jsonl", help="Path to a brute-force JSONL file")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=1.0,
        help="Fraction of common-universe question cells that may be replayed",
    )
    parser.add_argument(
        "--max-search-cost",
        type=float,
        default=None,
        help=(
            "Dollar guard; soft with warm-start expected costs, hard when "
            "--guaranteed-batch-cost is supplied"
        ),
    )
    parser.add_argument(
        "--guaranteed-batch-cost",
        type=float,
        default=None,
        help="Optional per-batch upper bound in USD, shared by every arm",
    )
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--effective-cost-bin-ratio", type=float, default=2.0)
    parser.add_argument("--effective-cost-bin-anchor", type=float, default=1e-4)
    parser.add_argument(
        "--ragged-diagnostic",
        action="store_true",
        help="Use arm-specific available tails instead of the common universe",
    )
    parser.add_argument("--horizon-bin-width", type=int, default=1)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help=(
            "Override all grid sizes; otherwise use the production 513-point "
            "direction-aware grids"
        ),
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.pickle:
        path = _require_data_path(args.pickle)
        models, datapoints, table = load_pickle(path)
    else:
        path = _require_data_path(args.jsonl)
        models, datapoints, table = load_jsonl(path)
    print(
        f"Loaded {len(models)} models, {len(datapoints)} questions, "
        f"{sum(len(row) for row in table.values())} cells from {path}"
    )

    grid = (
        RadialGittinsGrid(
            z_size=args.grid_size,
            delta_size=args.grid_size,
            state_size=args.grid_size,
        )
        if args.grid_size is not None
        else None
    )
    results: List[RadialSimulationResult] = []
    cache = RadialGittinsBoundaryCache()
    for offset in range(args.seeds):
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=args.batch_size,
            observation_budget_fraction=args.budget_fraction,
            max_search_cost_usd=args.max_search_cost,
            guaranteed_batch_cost_usd=args.guaranteed_batch_cost,
            search_cost_scale_eta=args.eta,
            effective_cost_bin_ratio=args.effective_cost_bin_ratio,
            effective_cost_bin_anchor=args.effective_cost_bin_anchor,
            horizon_bin_width=args.horizon_bin_width,
            seed=args.base_seed + offset,
            boundary_grid=grid,
            boundary_cache=cache,
            question_universe=("per_arm" if args.ragged_diagnostic else "common"),
        )
        print_radial_result(result)
        results.append(result)

    summary = summarize_radial_multi_seed(results)
    if len(results) > 1:
        print(f"\nSummary: {json.dumps(summary, indent=2, allow_nan=False)}")
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".csv":
            row = dict(summary)
            for key, value in tuple(row.items()):
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(value, allow_nan=False)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        else:
            payload = {
                "summary": summary,
                "results": [_jsonable_result(result) for result in results],
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
