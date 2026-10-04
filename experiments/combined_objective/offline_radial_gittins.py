#!/usr/bin/env python3
"""Offline replay for the cost-aware two-objective radial-Gittins selector.

The selector observes only cells it explicitly pulls from a frozen lookup
table.  Full-matrix objective vectors are computed after selection only for
evaluation metrics; they never enter calibration, indices, or stopping.

Everything except the unfinished-arm index is shared infrastructure: the
uniform warm start, the direction scheduler, the required-completion
stopping convention, the budget guards, and the archive/hypervolume/GD/IGD metrics.
An alternative acquisition rule therefore only has to supply an
``index_provider`` without changing the shared replay engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from experiments.single_objective import offline_selector_sim as _offline_sim
except ImportError:  # Direct ``python combined_objective/offline_radial_gittins.py``.
    from single_objective import offline_selector_sim as _offline_sim  # type: ignore[no-redef]

# Older local pickles record ``offline_selector_sim_v2.SampleResult``.
# The single-objective loader registers that alias; keep it here too when
# this module is imported through the ``experiments`` namespace package.
sys.modules.setdefault("offline_selector_sim_v2", _offline_sim)
LookupTable = _offline_sim.LookupTable
SampleResult = _offline_sim.SampleResult
_require_data_path = _offline_sim._require_data_path
load_jsonl = _offline_sim.load_jsonl
load_pickle = _offline_sim.load_pickle
load_scope = _offline_sim.load_scope

from agentopt.model_selection.radial_gittins import (
    DEFAULT_ANYTIME_DIRECTIONS,
    DEFAULT_DIRECTIONS,
    GaussianVectorPosterior,
    ObjectiveNormalizer,
    PerArmQuestionSchedule,
    QUESTION_ORDERS,
    build_fixed_prior_calibration,
    fit_empirical_bayes_warm_start,
)
from agentopt.model_selection.axis_gittins_dp import AxisGittinsBoundaryCache
from agentopt.model_selection.radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryCache,
    RadialGittinsBoundaryTable,
    RadialGittinsGrid,
    direction_aware_grid,
    radial_posterior_coordinates,
    terminal_expected_radial_utility,
)
from agentopt.model_selection.radial_gittins_prewarm import (
    RadialGittinsPrewarmRequest,
    RadialGittinsPrewarmStats,
    prewarm_radial_gittins_boundaries,
)


OFFLINE_PRODUCTION_BASE_GRID = RadialGittinsGrid(
    z_size=513,
    delta_size=513,
    state_size=513,
    boundary_margin_cells=4,
)


DEFAULT_RADIAL_BOUNDARY_CACHE_DIR = Path(
    os.environ.get(
        "AGENTOPT_RADIAL_GITTINS_CACHE_DIR",
        str(
            Path(__file__).resolve().parent
            / "results"
            / "cache_radial_gittins_boundaries"
        ),
    )
).expanduser()


_BOUNDARY_CACHE_STAT_FIELDS = (
    "builds",
    "memory_hits",
    "disk_hits",
    "disk_misses",
    "corruptions",
    "read_failures",
    "write_failures",
)
_BOUNDARY_PREWARM_STAT_FIELDS = tuple(
    RadialGittinsPrewarmStats.__dataclass_fields__
)
# On CPU, XLA compilation dominates one-off small grids. This conservative
# proxy routes the production 513-point, seven-cost/H=47 families to JAX while
# keeping the common 129/257-point plotting grids on the faster SciPy cold path.
# Explicit `boundary_build_backend="jax"` always overrides the heuristic.
_AUTO_JAX_MIN_CELL_STAGES = 50_000_000
_ANYTIME_BOUNDARY_MAX_WIDENING_RETRIES = 4


def _boundary_cache_stats_snapshot(cache: Any) -> Optional[Dict[str, int]]:
    """Read cache counters while remaining compatible with test doubles."""
    snapshot_method = getattr(cache, "stats_snapshot", None)
    if not callable(snapshot_method):
        return None
    snapshot = snapshot_method()
    return {
        name: int(getattr(snapshot, name))
        for name in _BOUNDARY_CACHE_STAT_FIELDS
    }


def _boundary_cache_stats_delta(
    before: Optional[Mapping[str, int]],
    after: Optional[Mapping[str, int]],
    *,
    fallback_builds: int,
) -> Dict[str, int]:
    if before is None or after is None:
        return {
            name: (max(0, int(fallback_builds)) if name == "builds" else 0)
            for name in _BOUNDARY_CACHE_STAT_FIELDS
        }
    return {
        name: max(0, int(after[name]) - int(before[name]))
        for name in _BOUNDARY_CACHE_STAT_FIELDS
    }


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
    observed_accuracy: Optional[float]
    observed_mean_cost_usd: Optional[float]
    observed_total_cost_usd: float
    equivalent_posterior_cost_usd: float
    n_samples_evaluated: int
    n_batches: int
    completed: bool
    is_direction_winner: bool = False
    is_nondominated: bool = False
    is_posterior_nondominated: bool = False
    is_oracle_raw_nondominated: bool = False
    raw_posterior_mean: Optional[Tuple[float, float]] = None
    raw_posterior_var: Optional[Tuple[float, float]] = None


PROVISIONAL_ARCHIVE_SCOPE = "provisional"
DEPLOYABLE_ARCHIVE_SCOPE = "deployable"
FINITE_ARCHIVE_SCOPE = "finite"
ARCHIVE_SCOPES = (PROVISIONAL_ARCHIVE_SCOPE, DEPLOYABLE_ARCHIVE_SCOPE, FINITE_ARCHIVE_SCOPE)
FINITE_RECOMMENDATION_RULES = ("finite_lcb", "finite_mean")
UNCERTAIN_RECOMMENDATION_RULES = FINITE_RECOMMENDATION_RULES
RECOMMENDATION_RULES = ("completed_only", *UNCERTAIN_RECOMMENDATION_RULES)


@dataclass(frozen=True)
class RecommendationCheckpoint:
    """One recorded recommendation and the arm scope that produced it.

    A checkpoint carries exactly one archive.  ``archive_scope`` states which
    arms were eligible for it:

    ``"provisional"``
        Historical snapshots over warm-started arms, including unfinished
        ones. Retained for loading older results; new completed-only runs
        always use the deployable scope.

    ``"deployable"``
        The empirical raw Pareto frontier over all ``completed_arm_indices``.
        Accuracy and cost are measured full-evaluation means. Neither
        posterior uncertainty nor exploration directions filter this set.

    ``"finite"``
        Scope for finite-test recommendation rules, which may include
        unfinished arms. ``"finite_lcb"`` uses a posterior prediction of each arm's
        complete finite-test mean, taking the raw accuracy LCB / mean-USD UCB
        Pareto frontier over every arm. No directional filter is applied.
        ``"finite_mean"`` uses the same finite-test means without a standard
        deviation penalty. The finite rules may additionally apply the actual
        observation-count gate recorded in run parameter ``recommendation_min_samples``.
        Observed raw vectors and full-data quality metrics are diagnostics.

    Full-data oracle fields are diagnostic only and never affect acquisition,
    stopping, or the archive. Completed-only runs use deployable scope throughout,
    including an empty
    warm-start archive when no arm is completed. ``added_arm_indices`` records
    new archive members since the previous recommendation, including a new
    member that replaces an existing one without increasing cardinality.

    For completed-only and finite-test snapshots, the legacy ``direction_winner_*``
    evidence fields contain the selected frontier, not direction winners.
    They retain their names for saved-result compatibility. Actual terminal
    direction winners remain separate acquisition diagnostics on the result.

    ``budget_fraction`` is the cost-aware share of brute-force search spend
    (``cumulative_search_cost_usd / bruteforce_search_cost_usd``), not the
    evaluation-count fraction.  Use ``cumulative_evaluations`` when the
    question-budget axis is needed.
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
    generational_distance: float
    inverted_generational_distance: float
    event: str
    completed_arm_indices: Tuple[int, ...] = ()
    archive_scope: str = PROVISIONAL_ARCHIVE_SCOPE
    current_lambda: float = 1.0
    lambda_stage: int = 0
    added_arm_indices: Tuple[int, ...] = ()
    added_models: Tuple[str, ...] = ()
    removed_arm_indices: Tuple[int, ...] = ()
    removed_models: Tuple[str, ...] = ()
    direction_eta_multipliers: Tuple[float, ...] = ()
    direction_eta_stages: Tuple[int, ...] = ()
    recommendation_rule: str = "completed_only"
    recommendation_beta: float = 1.0
    recommendation_desirability_vectors: Tuple[Tuple[float, float], ...] = ()
    direction_winner_sample_counts: Tuple[int, ...] = ()
    finite_target_mean_vectors: Tuple[Tuple[float, float], ...] = ()
    finite_target_std_vectors: Tuple[Tuple[float, float], ...] = ()
    recommendation_raw_vectors: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.archive_scope not in ARCHIVE_SCOPES:
            raise ValueError(f"archive_scope must be one of {ARCHIVE_SCOPES}")

    @property
    def is_deployable(self) -> bool:
        return self.archive_scope in {DEPLOYABLE_ARCHIVE_SCOPE, FINITE_ARCHIVE_SCOPE}


@dataclass(frozen=True)
class RecommendationEvent:
    """Immutable online evidence for a retained recommendation, without metrics.

    All vectors are copied at the event's time. No live posterior, observation
    history, full-data metric, or reference to an acquisition cache is retained.
    """

    cumulative_evaluations: int
    cumulative_search_cost_usd: float
    budget_fraction: float
    selected_arm_indices: Tuple[int, ...]
    direction_winner_arm_indices: Tuple[int, ...]
    winner_posterior_means: Tuple[Tuple[float, float], ...]
    estimated_raw_winner_vectors: Tuple[Tuple[float, float], ...]
    direction_winner_sample_counts: Tuple[int, ...]
    event: str
    completed_arm_indices: Tuple[int, ...] = ()
    archive_scope: str = PROVISIONAL_ARCHIVE_SCOPE
    current_lambda: float = 1.0
    lambda_stage: int = 0
    added_arm_indices: Tuple[int, ...] = ()
    removed_arm_indices: Tuple[int, ...] = ()
    direction_eta_multipliers: Tuple[float, ...] = ()
    direction_eta_stages: Tuple[int, ...] = ()
    recommendation_rule: str = "completed_only"
    recommendation_beta: float = 1.0
    recommendation_desirability_vectors: Tuple[Tuple[float, float], ...] = ()

    # Finite-test prediction evidence in raw accuracy / mean USD units, aligned
    # with direction_winner_arm_indices (selected frontier for finite rules).
    finite_target_mean_vectors: Tuple[Tuple[float, float], ...] = ()
    finite_target_std_vectors: Tuple[Tuple[float, float], ...] = ()
    recommendation_raw_vectors: Tuple[Tuple[float, float], ...] = ()


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
    generational_distance: float
    inverted_generational_distance: float
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
    current_lambda: float = 1.0
    lambda_stage: int = 0
    lambda_stop_events: List[Dict[str, Any]] = field(default_factory=list)
    stage_timing_events: List[Dict[str, Any]] = field(default_factory=list)
    gittins_stop_triggered: bool = False
    halted_by_gittins: bool = False
    recommendation_initial_snapshot: Optional[RecommendationCheckpoint] = None
    recommendation_final_snapshot: Optional[RecommendationCheckpoint] = None
    direction_eta_multipliers: Tuple[float, ...] = ()
    direction_eta_stages: Tuple[int, ...] = ()
    direction_eta_events: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_events: List[RecommendationEvent] = field(default_factory=list)
    recommendation_initial_event: Optional[RecommendationEvent] = None
    recommendation_final_event: Optional[RecommendationEvent] = None


@dataclass(frozen=True)
class DirectionVisitContext:
    """State an acquisition rule may read when scoring one direction visit.

    ``posteriors`` is a read-only view of the live shared posteriors, so it is
    only valid during the visit it was created for.  ``effective_pull_costs``
    holds the frozen per-arm continuation cost in normalized utility units,
    which is the same quantity the boundary dynamic program consumes.
    With independent eta decay, ``current_lambda`` and ``lambda_stage`` are
    the visited direction's multiplier/stage; the eta vectors describe all
    directions. The legacy scalar name is not a scalarization direction.
    """

    global_step: int
    direction_index: int
    direction: Tuple[float, float]
    visit_count: int
    completed_arms: Tuple[int, ...]
    unfinished_arms: Tuple[int, ...]
    adaptive_pulls: Tuple[int, ...]
    model_names: Tuple[str, ...]
    posteriors: Mapping[int, GaussianVectorPosterior] = field(
        default_factory=lambda: MappingProxyType({})
    )
    reference_point: Tuple[float, float] = (0.0, 0.0)
    effective_pull_costs: Tuple[float, ...] = ()
    current_lambda: float = 1.0
    lambda_stage: int = 0
    direction_eta_multipliers: Tuple[float, ...] = ()
    direction_eta_stages: Tuple[int, ...] = ()


IndexProvider = Callable[[DirectionVisitContext, int], float]
RadialTerminalUtilityProvider = Callable[[int, int], float]


class _DirectionScheduler:
    """Visit direction groups while tracking stops under unchanged observations.

    Accuracy-last drains the other directions first, then the exact accuracy
    endpoint.  The two sequential axis policies similarly drain one exact
    endpoint before visiting the other.  An observation invalidates earlier
    stops, so a previously drained group must be checked again before the
    policy may terminate.
    """

    def __init__(self, directions: Sequence[Tuple[float, float]], policy: str):
        policies = {
            "round_robin",
            "accuracy_last",
            "quality_then_deployment",
            "deployment_then_quality",
        }
        if policy not in policies:
            raise ValueError(
                "direction_scheduler must be 'round_robin', 'accuracy_last', "
                "'quality_then_deployment', "
                "or 'deployment_then_quality'"
            )
        self.policy = policy
        indices = tuple(range(len(directions)))
        if policy == "accuracy_last":
            primary = tuple(i for i in indices if _direction_axis(directions[i]) != 0)
            accuracy = tuple(i for i in indices if _direction_axis(directions[i]) == 0)
            self.groups = tuple(group for group in (primary, accuracy) if group)
        elif policy in {"quality_then_deployment", "deployment_then_quality"}:
            quality = tuple(i for i in indices if _direction_axis(directions[i]) == 0)
            deployment = tuple(i for i in indices if _direction_axis(directions[i]) == 1)
            if len(directions) != 2 or len(quality) != 1 or len(deployment) != 1:
                raise ValueError(
                    f"direction_scheduler={policy!r} requires exactly the quality "
                    "and deployment axes"
                )
            ordered = (
                (quality, deployment)
                if policy == "quality_then_deployment"
                else (deployment, quality)
            )
            self.groups = ordered
        else:
            self.groups = (indices,)
        self._direction_count = len(directions)
        self._group = 0
        self._position = 0
        self._stopped: set[int] = set()

    @property
    def direction_index(self) -> int:
        return self.groups[self._group][self._position]

    @property
    def all_stopped(self) -> bool:
        return len(self._stopped) == self._direction_count

    def record_stop(self) -> None:
        self._stopped.add(self.direction_index)

    def record_observation(self) -> None:
        self._stopped.clear()

    def advance(self, *, force_round_robin: bool = False) -> int:
        if force_round_robin:
            # Fixed-lambda diagnostics can ignore stops after the first global
            # stop. Keep visiting every direction in that forced continuation.
            next_index = (self.direction_index + 1) % self._direction_count
            for group_index, indices in enumerate(self.groups):
                if next_index in indices:
                    self._group = group_index
                    self._position = indices.index(next_index)
                    break
            return self.direction_index
        group = self.groups[self._group]
        if len(self.groups) > 1 and all(i in self._stopped for i in group):
            self._group = (self._group + 1) % len(self.groups)
            self._position = 0
        else:
            self._position = (self._position + 1) % len(group)
        return self.direction_index

    def start_next_stage(self) -> int:
        self._stopped.clear()
        if len(self.groups) > 1:
            self._group = self._position = 0
            return self.direction_index
        return self.advance()


class _VersionedArmValueCache:
    """Cache scalar arm values until that arm's posterior changes.

    Every online radial index and terminal utility is a pure function of
    immutable run settings plus one arm's posterior and adaptive-pull stage.
    A physical pull changes exactly one arm, so a per-arm generation counter
    gives precise lazy invalidation without clearing values for other arms or
    eagerly recomputing every search task.
    """

    def __init__(self, n_arms: int) -> None:
        self._versions = np.zeros(int(n_arms), dtype=np.int64)
        self._values: Dict[
            Hashable,
            Dict[int, Tuple[int, float]],
        ] = {}
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def invalidate(self, arm_index: int) -> None:
        """Mark every cached value for one arm stale in constant time."""
        self._versions[int(arm_index)] += 1
        self._invalidations += 1

    def clear_radial_indices(self) -> None:
        """Drop lambda-dependent indices while keeping terminal utilities."""
        for namespace in tuple(self._values):
            if isinstance(namespace, tuple) and namespace[0] == "radial_index":
                del self._values[namespace]

    def clear_direction_radial_indices(self, direction_index: int) -> None:
        """Invalidate one direction's cost-dependent indices after local decay."""
        self._values.pop(("radial_index", int(direction_index)), None)

    def get_or_compute(
        self,
        namespace: Hashable,
        arm_index: int,
        compute: Callable[[int], float],
    ) -> float:
        """Return the current value, evaluating ``compute`` only on a miss."""
        arm_index = int(arm_index)
        version = int(self._versions[arm_index])
        namespace_values = self._values.setdefault(namespace, {})
        cached = namespace_values.get(arm_index)
        if cached is not None and cached[0] == version:
            self._hits += 1
            return cached[1]
        self._misses += 1
        value = float(compute(arm_index))
        namespace_values[arm_index] = (version, value)
        return value

    def stats_snapshot(self) -> Dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": self._invalidations,
            "namespaces": len(self._values),
            "resident_values": sum(
                len(values) for values in self._values.values()
            ),
        }


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
        if np.any(array < 0.0) or not math.isclose(
            float(array.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("direction components must be nonnegative and sum to one")
        resolved.append((float(array[0]), float(array[1])))
    if not resolved:
        raise ValueError("at least one direction is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("directions must be unique")
    return tuple(resolved)


def _direction_axis(direction: Sequence[float]) -> Optional[int]:
    """Identify an exact endpoint without approximating it by a narrow ray."""
    if direction[1] == 0.0:
        return 0
    if direction[0] == 0.0:
        return 1
    return None


def _cli_directions(
    *,
    anytime: bool,
    extra_directions: Iterable[Sequence[float]] = (),
) -> Tuple[Tuple[float, float], ...]:
    """Append CLI directions once, including endpoints already in defaults."""
    directions = list(DEFAULT_ANYTIME_DIRECTIONS if anytime else DEFAULT_DIRECTIONS)
    for extra in extra_directions:
        direction = _validate_directions((extra,))[0]
        if direction not in directions:
            directions.append(direction)
    return tuple(directions)


def terminal_expected_direction_utility(
    mean: Sequence[float],
    var: Sequence[float],
    direction: Sequence[float],
    reference: Sequence[float] = (0.0, 0.0),
) -> float:
    """Use one objective at an endpoint and radial utility in the interior.

    The scalar endpoint intentionally ignores the inactive objective. Sending
    zero direction components through radial scaling would divide by zero.
    """
    resolved = _validate_directions((direction,))[0]
    axis = _direction_axis(resolved)
    if axis is None:
        return terminal_expected_radial_utility(mean, var, resolved, reference)
    mean_array = np.asarray(mean, dtype=np.float64)
    var_array = np.asarray(var, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if any(array.shape != (2,) for array in (mean_array, var_array, reference_array)):
        raise ValueError("mean, var, and reference must be length-2 vectors")
    if not all(np.all(np.isfinite(array)) for array in (mean_array, var_array, reference_array)):
        raise ValueError("mean, var, and reference must be finite")
    if np.any(var_array <= 0.0):
        raise ValueError("var components must be positive")
    return float(mean_array[axis] - reference_array[axis])


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


def _bruteforce_search_cost_usd(
    models: Sequence[str],
    available_by_arm: Mapping[int, Sequence[int]],
    table: LookupTable,
) -> float:
    """Sum lookup-table deployment costs over every cell in the universe."""
    total = 0.0
    for arm_index, question_ids in available_by_arm.items():
        model_data = table[models[arm_index]]
        for question_id in question_ids:
            sample = model_data.get(question_id)
            if sample is None:
                raise ValueError(
                    f"missing lookup cell for model {models[arm_index]!r}, "
                    f"question {question_id}"
                )
            _, cost, _ = _sample_values(sample)
            total += cost
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("brute-force search cost must be finite and positive")
    return float(total)


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


def _best_direction_index(
    indices: Mapping[int, float],
    tolerance: float,
    *,
    direction: Sequence[float],
    posteriors: Mapping[int, GaussianVectorPosterior],
    endpoint_secondary_values: Optional[Mapping[int, float]] = None,
) -> Tuple[Optional[int], float]:
    """Break endpoint utility ties by the other observed objective.

    The interior direction rule is unchanged. Standalone posterior-only
    callers use the other posterior mean as their secondary objective;
    replay archives supply observed accuracy or negative observed mean cost.
    """
    best_arm, best_value = _best_index(indices, tolerance)
    axis = _direction_axis(direction)
    if best_arm is None or axis is None:
        return best_arm, best_value
    # The tolerance-aware ordered fold above is intentionally retained for
    # interior rays. An endpoint's secondary objective may select another
    # arm, so anchor its tie set to the true maximum to avoid drifting more
    # than one tolerance below the best active-objective value.
    best_value = max(float(value) for value in indices.values())
    tied = [
        arm_index for arm_index, value in indices.items()
        if abs(float(value) - best_value) <= tolerance
    ]
    best_arm = max(
        tied,
        key=lambda arm_index: (
            float(endpoint_secondary_values[arm_index])
            if endpoint_secondary_values is not None
            else float(posteriors[arm_index].mean[1 - axis]),
            -arm_index,
        ),
    )
    return best_arm, float(indices[best_arm])


def evaluate_direction_status(
    *,
    direction: Tuple[float, float],
    posteriors: Mapping[int, GaussianVectorPosterior],
    completed_arms: Sequence[int],
    unfinished_indices: Mapping[int, float],
    reference_point: Sequence[float],
    stop_tolerance: float,
    completed_terminal_indices: Optional[Mapping[int, float]] = None,
    completed_tiebreak_values: Optional[Mapping[int, float]] = None,
) -> DirectionStatus:
    """Combine completed terminal values and unfinished Gittins indices."""
    if completed_terminal_indices is None:
        completed_indices = {
            arm_index: terminal_expected_direction_utility(
                posteriors[arm_index].mean,
                posteriors[arm_index].var,
                direction,
                reference_point,
            )
            for arm_index in completed_arms
        }
    else:
        completed_indices = {
            arm_index: float(completed_terminal_indices[arm_index])
            for arm_index in completed_arms
        }
    best_completed_arm, best_completed = _best_direction_index(
        completed_indices,
        stop_tolerance,
        direction=direction,
        posteriors=posteriors,
        endpoint_secondary_values=completed_tiebreak_values,
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
    """Return indices not strictly dominated under maximization of both columns.

    The two-objective sweep is exact and preserves the input order (including
    duplicate nondominated points).  Sorting replaces the former all-pairs
    scan, which made diagnostic checkpoints quadratic in the number of
    configurations on the large SCOPE matrices.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    n_points = int(points.shape[0])
    if n_points == 0:
        return []

    # Visit equal-x groups from largest x to smallest.  Within one group only
    # points attaining its largest y can survive.  Such a point is dominated
    # by an earlier (strictly larger x) group exactly when that prefix already
    # contains y >= its own.  Equal duplicate maxima survive together because
    # neither strictly dominates the other.
    order = np.argsort(-points[:, 0], kind="stable")
    keep_mask = np.zeros(n_points, dtype=bool)
    prefix_max_y = -math.inf
    start = 0
    while start < n_points:
        stop = start + 1
        x_value = points[order[start], 0]
        while stop < n_points and points[order[stop], 0] == x_value:
            stop += 1
        group = order[start:stop]
        group_max_y = float(np.max(points[group, 1]))
        if prefix_max_y < group_max_y:
            maxima = group[points[group, 1] == group_max_y]
            keep_mask[maxima] = True
        prefix_max_y = max(prefix_max_y, group_max_y)
        start = stop
    return np.flatnonzero(keep_mask).tolist()


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


def _front_points(points: np.ndarray) -> np.ndarray:
    """Return *points* as a validated ``(n, 2)`` array, possibly empty."""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return array


def _nondominated_front(points: np.ndarray) -> np.ndarray:
    array = _front_points(points)
    if array.shape[0] == 0:
        return array
    return array[nondominated_indices(array)]


def _nearest_front_distances(points: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of *points* to the nearest row of *front*."""
    delta = points[:, np.newaxis, :] - front[np.newaxis, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)


def generational_distance(
    obtained: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Mean distance from every returned point to the ground-truth front.

    Both arguments are points in the same maximization space used by
    :func:`hypervolume_2d` (normalized desirability).  Every returned point
    contributes, including points dominated in the true evaluation space.
    The reported value is

    ``GD(A, P*) = (1/|A|) sum_{a in A} min_{z in P*} ||a - z||_2``.

    This is zero when every obtained point lies on the reference front, even
    if the archive is only a subset of that front.  An empty obtained set is
    ``+inf`` unless the reference front is also empty.
    """
    obtained_points = _front_points(obtained)
    truth_front = _nondominated_front(reference_front)
    if obtained_points.shape[0] == 0:
        return 0.0 if truth_front.shape[0] == 0 else math.inf
    if truth_front.shape[0] == 0:
        return math.inf
    return float(np.mean(_nearest_front_distances(obtained_points, truth_front)))


def inverted_generational_distance(
    obtained: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Mean distance from the ground-truth front to the full returned set.

    ``IGD(A, P*) = (1/|P*|) sum_{z in P*} min_{a in A} ||z - a||_2``.

    All returned points remain eligible nearest neighbors, including points
    dominated in the true evaluation space.

    Unlike GD, a missing region of the true front increases IGD even when
    every returned point is itself Pareto optimal.  An empty obtained set is
    ``+inf`` unless the reference front is also empty.
    """
    obtained_points = _front_points(obtained)
    truth_front = _nondominated_front(reference_front)
    if truth_front.shape[0] == 0:
        return 0.0
    if obtained_points.shape[0] == 0:
        return math.inf
    return float(np.mean(_nearest_front_distances(truth_front, obtained_points)))


@dataclass(frozen=True)
class FrontQualityMetrics:
    hypervolume: float
    hypervolume_regret: float
    generational_distance: float
    inverted_generational_distance: float


def front_quality_metrics(
    selected_points: np.ndarray,
    reference_front: np.ndarray,
    reference_point: Sequence[float],
    ground_truth_hv: float,
) -> FrontQualityMetrics:
    """Score one archive against the ground-truth desirability front."""
    selected = _front_points(selected_points)
    selected_hv = hypervolume_2d(selected, reference_point)
    return FrontQualityMetrics(
        hypervolume=float(selected_hv),
        hypervolume_regret=float(max(0.0, ground_truth_hv - selected_hv)),
        generational_distance=generational_distance(selected, reference_front),
        inverted_generational_distance=inverted_generational_distance(
            selected, reference_front
        ),
    )


def provisional_direction_winner_arms(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    reference_point: Sequence[float],
    stop_tolerance: float,
    candidate_arms: Optional[Sequence[int]] = None,
    terminal_utility_provider: Optional[
        RadialTerminalUtilityProvider
    ] = None,
    endpoint_tiebreak_provider: Optional[
        RadialTerminalUtilityProvider
    ] = None,
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
    for direction_index, direction in enumerate(directions):
        if terminal_utility_provider is None:
            utilities = {
                arm_index: terminal_expected_direction_utility(
                    posteriors[arm_index].mean,
                    posteriors[arm_index].var,
                    direction,
                    reference_point,
                )
                for arm_index in arm_indices
            }
        else:
            utilities = {
                arm_index: float(
                    terminal_utility_provider(direction_index, arm_index)
                )
                for arm_index in arm_indices
            }
        secondary_values = (
            {
                arm_index: float(endpoint_tiebreak_provider(direction_index, arm_index))
                for arm_index in arm_indices
            }
            if endpoint_tiebreak_provider is not None and _direction_axis(direction) is not None
            else None
        )
        winner_arm, _ = _best_direction_index(
            utilities,
            stop_tolerance,
            direction=direction,
            posteriors=posteriors,
            endpoint_secondary_values=secondary_values,
        )
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


class _CompletedRawParetoArchive:
    """Incremental empirical Pareto frontier of completed observed rows.

    Completed histories are immutable. Each newly completed arm's means are
    read once; adding rows can only remove old frontier members. Incomplete
    rows never enter this cache, including when their sample means look worse.
    """

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.frontier: Dict[int, Tuple[float, float]] = {}

    @staticmethod
    def _dominates(left: Tuple[float, float], right: Tuple[float, float]) -> bool:
        return (left[0] >= right[0] and left[1] <= right[1]
                and (left[0] > right[0] or left[1] < right[1]))

    def update(
        self,
        completed_arms: Sequence[int],
        observed_scores: Mapping[int, Sequence[float]],
        observed_costs: Mapping[int, Sequence[float]],
    ) -> None:
        completed = set(completed_arms)
        if not self.seen.issubset(completed):
            raise ValueError("completed raw archive requires a monotone completed-arm set")
        for arm in sorted(completed - self.seen):
            scores, costs = observed_scores[arm], observed_costs[arm]
            if not len(scores) or len(scores) != len(costs):
                raise ValueError("completed raw archive requires matching nonempty observed scores and costs")
            point = (float(np.mean(scores)), float(np.mean(costs)))
            if not all(math.isfinite(value) for value in point):
                raise ValueError("completed raw archive requires finite observed means")
            if not any(self._dominates(other, point) for other in self.frontier.values()):
                removed = [other_arm for other_arm, other in self.frontier.items()
                           if self._dominates(point, other)]
                for other_arm in removed:
                    del self.frontier[other_arm]
                self.frontier[arm] = point
            self.seen.add(arm)


def finite_test_mean_moments(
    posterior: GaussianVectorPosterior,
    observed_sum: Sequence[float],
    n_observed: int,
    n_total: int,
    question_noise_var: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean/variance of a fixed test set's realized mean.

    Conditional on a latent Gaussian mean, unseen per-question outcomes are
    independent with the supplied per-question variance. Observed outcomes
    are fixed. Units must agree with the authoritative posterior (normalized
    accuracy and mean USD for raw_mean replay). This never mutates that state.
    The returned uncertainty concerns this finite test set, not generalization.
    """
    n_total = _positive_integer(n_total, "n_total")
    if (isinstance(n_observed, (bool, np.bool_))
            or not isinstance(n_observed, (int, np.integer))
            or not 0 <= n_observed <= n_total):
        raise ValueError("n_observed must be an integer between zero and n_total")
    observed_sum = np.asarray(observed_sum, dtype=np.float64)
    noise = np.asarray(question_noise_var, dtype=np.float64)
    if observed_sum.shape != (2,) or not np.all(np.isfinite(observed_sum)):
        raise ValueError("observed_sum must be a finite length-2 vector")
    if noise.shape != (2,) or not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise ValueError("question_noise_var must be a finite nonnegative length-2 vector")
    remaining = n_total - n_observed
    mean = (observed_sum + remaining * posterior.mean) / n_total
    variance = (remaining ** 2 * posterior.var + remaining * noise) / n_total ** 2
    return mean, variance


def raw_pareto_front_indices(points: np.ndarray) -> Tuple[int, ...]:
    """Strict 2-D Pareto front: maximize raw accuracy, minimize mean USD.

    Sorting plus a vectorized cost-group sweep is O(K log K). Equal objective
    pairs are all retained, with deterministic original arm order.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (K, 2) array")
    if not len(points):
        return ()
    order = np.lexsort((-points[:, 0], points[:, 1]))
    accuracy, cost = points[order, 0], points[order, 1]
    starts = np.r_[True, cost[1:] != cost[:-1]]
    groups = np.cumsum(starts) - 1
    group_best = accuracy[starts]
    previous_best = np.r_[-np.inf, np.maximum.accumulate(group_best)[:-1]]
    keep = ((accuracy == group_best[groups])
            & (group_best[groups] > previous_best[groups]))
    return tuple(sorted(int(i) for i in order[keep]))


def _validate_recommendation_min_samples(value: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError("recommendation_min_samples must be a nonnegative integer")
    return int(value)


class _FiniteRecommendationCache:
    """Per-arm finite-test predictions; only appended observations are read."""

    def __init__(self, n_total, question_noise_var, normalizer, beta, min_samples=0):
        self.n_total = tuple(_positive_integer(n, "n_total") for n in n_total)
        self.min_samples = _validate_recommendation_min_samples(min_samples)
        self.question_noise_var = np.asarray(question_noise_var, dtype=np.float64)
        self.normalizer = normalizer
        self.beta = float(beta)
        self.counts = np.zeros(len(self.n_total), dtype=np.int64)
        self.sums = np.zeros((len(self.n_total), 2), dtype=np.float64)
        self.observed_means = np.zeros_like(self.sums)
        self.means = np.zeros_like(self.sums)
        self.stds = np.zeros_like(self.sums)
        self.conservative = np.zeros_like(self.sums)

    def update(self, arm, posterior, observed_scores, observed_costs):
        n, before = len(observed_scores), int(self.counts[arm])
        if len(observed_costs) != n or n < before or n > self.n_total[arm]:
            raise ValueError("finite recommendation cache needs monotone matching observations")
        if n == before:
            return
        self.sums[arm] += (
            float(np.sum(observed_scores[before:])),
            float(np.sum(observed_costs[before:])),
        )
        self.counts[arm] = n
        self.observed_means[arm] = self.sums[arm] / n
        low, high = self.normalizer.score_bounds
        span = high - low
        normalized_sum = self.sums[arm].copy()
        normalized_sum[0] = (normalized_sum[0] - n * low) / span
        mean, variance = finite_test_mean_moments(
            posterior, normalized_sum, n, self.n_total[arm], self.question_noise_var,
        )
        self.means[arm] = (low + span * mean[0], mean[1])
        self.stds[arm] = np.sqrt(variance) * (span, 1.0)
        if n == self.n_total[arm]:
            # Use the same summation as completed_only, once, so rounding of
            # batch sums cannot break exact empirical ties at completion.
            self.observed_means[arm] = (float(np.mean(observed_scores)), float(np.mean(observed_costs)))
            self.means[arm] = self.observed_means[arm]
        self.conservative[arm] = self.means[arm] + self.beta * self.stds[arm] * (-1.0, 1.0)

    def select(self):
        if np.any(self.counts == 0):
            raise ValueError("finite recommendation requires every arm's warm observations")
        eligible = np.flatnonzero(self.counts >= self.min_samples)
        members = tuple(int(eligible[i]) for i in raw_pareto_front_indices(self.conservative[eligible]))
        positions = list(members)
        raw = self.conservative[positions]
        low, high = self.normalizer.score_bounds
        reward = raw.copy()
        reward[:, 0] = (raw[:, 0] - low) / (high - low)
        reward[:, 1] = 1.0 - raw[:, 1] / self.normalizer.cost_reference_usd
        freeze = lambda array: tuple(tuple(map(float, row)) for row in array)
        return _RecommendationSelection(
            selected_arm_indices=members,
            direction_winner_arm_indices=members,
            recommendation_desirability_vectors=freeze(reward),
            estimated_raw_winner_vectors=freeze(self.observed_means[positions]),
            finite_target_mean_vectors=freeze(self.means[positions]),
            finite_target_std_vectors=freeze(self.stds[positions]),
            recommendation_raw_vectors=freeze(raw),
        )


@dataclass(frozen=True)
class _RecommendationSelection:
    selected_arm_indices: Tuple[int, ...]
    direction_winner_arm_indices: Tuple[int, ...]
    recommendation_desirability_vectors: Tuple[Tuple[float, float], ...]
    # Needed online by the empirical raw and finite-test rules.
    estimated_raw_winner_vectors: Tuple[Tuple[float, float], ...] = ()
    finite_target_mean_vectors: Tuple[Tuple[float, float], ...] = ()
    finite_target_std_vectors: Tuple[Tuple[float, float], ...] = ()
    recommendation_raw_vectors: Tuple[Tuple[float, float], ...] = ()


def _recommendation_selection(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    reference_point: Sequence[float],
    stop_tolerance: float,
    observed_scores: Mapping[int, Sequence[float]],
    observed_costs: Mapping[int, Sequence[float]],
    completed_arms: Sequence[int],
    archive_scope: str,
    recommendation_rule: str,
    recommendation_beta: float,
    completed_raw_archive: Optional[_CompletedRawParetoArchive] = None,
    finite_recommendation_cache: Optional[_FiniteRecommendationCache] = None,
) -> _RecommendationSelection:
    """Select online members without computing any diagnostic archive or metric."""
    if recommendation_rule not in RECOMMENDATION_RULES:
        raise ValueError(f"recommendation_rule must be one of {RECOMMENDATION_RULES}")
    if archive_scope not in ARCHIVE_SCOPES:
        raise ValueError(f"archive_scope must be one of {ARCHIVE_SCOPES}")
    if recommendation_rule in FINITE_RECOMMENDATION_RULES:
        if finite_recommendation_cache is None:
            raise ValueError(f"{recommendation_rule} requires a finite recommendation cache")
        return finite_recommendation_cache.select()
    archive = completed_raw_archive if completed_raw_archive is not None else _CompletedRawParetoArchive()
    archive.update(completed_arms, observed_scores, observed_costs)
    members = tuple(sorted(archive.frontier))
    return _RecommendationSelection(
        selected_arm_indices=members,
        # Legacy evidence slots also support empirical recommendations with
        # more members than exploration directions. No directional ranking.
        direction_winner_arm_indices=members,
        recommendation_desirability_vectors=(),
        estimated_raw_winner_vectors=tuple(archive.frontier[i] for i in members),
    )


def _capture_recommendation_event(
    selection: _RecommendationSelection,
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    observed_scores: Mapping[int, Sequence[float]],
    observed_costs: Mapping[int, Sequence[float]],
    total_evaluations: int,
    total_cost: float,
    bruteforce_search_cost_usd: float,
    event: str,
    completed_arms: Sequence[int],
    archive_scope: str,
    current_lambda: float,
    lambda_stage: int,
    direction_eta_multipliers: Tuple[float, ...],
    direction_eta_stages: Tuple[int, ...],
    recommendation_rule: str,
    recommendation_beta: float,
) -> RecommendationEvent:
    """Freeze only a retained event's online evidence; never access full truth."""
    if not math.isfinite(bruteforce_search_cost_usd) or bruteforce_search_cost_usd <= 0.0:
        raise ValueError("bruteforce_search_cost_usd must be finite and positive")
    winners = selection.direction_winner_arm_indices
    raw = selection.estimated_raw_winner_vectors
    return RecommendationEvent(
        cumulative_evaluations=int(total_evaluations),
        cumulative_search_cost_usd=float(total_cost),
        budget_fraction=float(total_cost) / float(bruteforce_search_cost_usd),
        selected_arm_indices=selection.selected_arm_indices,
        direction_winner_arm_indices=winners,
        winner_posterior_means=tuple(tuple(map(float, posteriors[i].mean)) for i in winners),
        estimated_raw_winner_vectors=raw,
        direction_winner_sample_counts=tuple(len(observed_scores[i]) for i in winners),
        event=event,
        completed_arm_indices=tuple(int(i) for i in completed_arms),
        archive_scope=DEPLOYABLE_ARCHIVE_SCOPE if recommendation_rule == "completed_only" else archive_scope,
        current_lambda=current_lambda,
        lambda_stage=lambda_stage,
        direction_eta_multipliers=direction_eta_multipliers,
        direction_eta_stages=direction_eta_stages,
        recommendation_rule=recommendation_rule,
        recommendation_beta=recommendation_beta,
        recommendation_desirability_vectors=selection.recommendation_desirability_vectors,
        finite_target_mean_vectors=selection.finite_target_mean_vectors,
        finite_target_std_vectors=selection.finite_target_std_vectors,
        recommendation_raw_vectors=selection.recommendation_raw_vectors,
    )


def _materialize_recommendation_event(
    event: RecommendationEvent,
    *,
    models: Sequence[str],
    truth_vectors: np.ndarray,
    truth_front: np.ndarray,
    raw_truth_vectors: np.ndarray,
    reference_point: Sequence[float],
    ground_truth_hv: float,
) -> RecommendationCheckpoint:
    """Compute diagnostic archives/metrics from frozen evidence, after sampling."""
    winners = event.direction_winner_arm_indices
    posterior = np.asarray(event.winner_posterior_means).reshape((-1, 2))
    posterior_arms = tuple(winners[i] for i in nondominated_indices(posterior))
    raw_arms = tuple(raw_archive_arm_indices(
        winners, np.asarray(event.estimated_raw_winner_vectors).reshape((-1, 2)),
    ))
    oracle_arms = tuple(raw_archive_arm_indices(
        winners, raw_truth_vectors[list(winners)] if winners else np.empty((0, 2)),
    ))
    selected = list(event.selected_arm_indices)
    quality = front_quality_metrics(
        truth_vectors[selected] if selected else np.empty((0, 2)),
        truth_front, reference_point, ground_truth_hv,
    )
    fields = {key: value for key, value in vars(event).items() if key != "winner_posterior_means"}
    return RecommendationCheckpoint(
        **fields,
        selected_models=tuple(models[i] for i in selected),
        posterior_archive_arm_indices=posterior_arms,
        posterior_archive_models=tuple(models[i] for i in posterior_arms),
        online_raw_archive_arm_indices=raw_arms,
        online_raw_archive_models=tuple(models[i] for i in raw_arms),
        oracle_raw_winner_archive_arm_indices=oracle_arms,
        oracle_raw_winner_archive_models=tuple(models[i] for i in oracle_arms),
        added_models=tuple(models[i] for i in event.added_arm_indices),
        removed_models=tuple(models[i] for i in event.removed_arm_indices),
        hypervolume=quality.hypervolume,
        hypervolume_regret=quality.hypervolume_regret,
        generational_distance=quality.generational_distance,
        inverted_generational_distance=quality.inverted_generational_distance,
    )


def _materialize_recommendation_events(
    events: Sequence[RecommendationEvent],
    initial: Optional[RecommendationEvent],
    final: Optional[RecommendationEvent],
    *,
    models: Sequence[str],
    truth_vectors: np.ndarray,
    raw_truth_vectors: np.ndarray,
    reference_point: Sequence[float],
    ground_truth_hv: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    truth_front = truth_vectors[nondominated_indices(truth_vectors)]
    materialized: Dict[RecommendationEvent, RecommendationCheckpoint] = {}

    def materialize(event: Optional[RecommendationEvent]) -> Optional[RecommendationCheckpoint]:
        if event is None:
            return None
        if event not in materialized:
            materialized[event] = _materialize_recommendation_event(
                event, models=models, truth_vectors=truth_vectors,
                truth_front=truth_front, raw_truth_vectors=raw_truth_vectors,
                reference_point=reference_point, ground_truth_hv=ground_truth_hv,
            )
        return materialized[event]

    checkpoints = {
        "recommendation_trajectory": [materialize(event) for event in events],
        "recommendation_initial_snapshot": materialize(initial),
        "recommendation_final_snapshot": materialize(final),
    }
    return checkpoints, {
        "diagnostic_materializations": len(materialized),
        "diagnostic_wall_time_seconds": time.perf_counter() - started,
    }


def materialize_recommendation_diagnostics(result: RadialSimulationResult) -> None:
    """Populate legacy checkpoint fields on demand, without replaying acquisition.

    Existing callers get these same diagnostics after sampling by default.
    Repeated calls are free; event-only JSON can instead use the saved variant.
    """
    if result.recommendation_initial_event is None or result.recommendation_initial_snapshot is not None:
        return
    checkpoints, stats = _materialize_recommendation_events(
        result.recommendation_events, result.recommendation_initial_event,
        result.recommendation_final_event,
        models=[arm.model_name for arm in result.model_results],
        truth_vectors=result.truth_vectors, raw_truth_vectors=result.raw_truth_vectors,
        reference_point=result.params.get("metric_reference_point", result.params["reference_point"]),
        ground_truth_hv=result.ground_truth_hypervolume,
    )
    for name, value in checkpoints.items():
        setattr(result, name, value)
    result.params["recommendation_recording"].update(stats)


def materialize_saved_recommendation_diagnostics(payload: Dict[str, Any]) -> None:
    """Expand one event-only JSON result in place, even in a fresh process.

    Accepts one `_jsonable_result` dictionary, or one item of a CLI output's
    `results` list. No lookup files, live posteriors, or acquisition replay needed.
    """
    if payload.get("recommendation_initial_event") is None or payload.get("recommendation_initial_snapshot") is not None:
        return

    def restore(value: Optional[Mapping[str, Any]]) -> Optional[RecommendationEvent]:
        if value is None:
            return None
        def freeze(item: Any) -> Any:
            return tuple(freeze(v) for v in item) if isinstance(item, (list, tuple)) else item
        allowed = {item.name for item in fields(RecommendationEvent)}
        return RecommendationEvent(**{
            key: freeze(item) for key, item in value.items() if key in allowed
        })

    checkpoints, stats = _materialize_recommendation_events(
        [restore(event) for event in payload["recommendation_events"]],
        restore(payload["recommendation_initial_event"]),
        restore(payload["recommendation_final_event"]),
        models=[arm["model_name"] for arm in payload["model_results"]],
        truth_vectors=np.asarray(payload["truth_vectors"], dtype=np.float64),
        raw_truth_vectors=np.asarray(payload["raw_truth_vectors"], dtype=np.float64),
        reference_point=payload["params"].get("metric_reference_point", payload["params"]["reference_point"]),
        ground_truth_hv=payload["ground_truth_hypervolume"],
    )
    # Match CLI handling of tuple arrays and undefined empty-front distances.
    payload.update(_jsonable_value({
        name: [asdict(point) for point in value] if isinstance(value, list)
        else asdict(value) if value is not None else None
        for name, value in checkpoints.items()
    }))
    payload["params"]["recommendation_recording"].update(stats)


def _recommendation_checkpoint(
    *,
    posteriors: Mapping[int, GaussianVectorPosterior],
    directions: Sequence[Tuple[float, float]],
    models: Sequence[str],
    reference_point: Sequence[float],
    stop_tolerance: float,
    truth_vectors: np.ndarray,
    truth_front: np.ndarray,
    raw_truth_vectors: np.ndarray,
    observed_scores: Mapping[int, Sequence[float]],
    observed_costs: Mapping[int, Sequence[float]],
    ground_truth_hv: float,
    total_evaluations: int,
    total_cost: float,
    bruteforce_search_cost_usd: float,
    event: str,
    completed_arms: Sequence[int] = (),
    archive_scope: str = PROVISIONAL_ARCHIVE_SCOPE,
    current_lambda: float = 1.0,
    lambda_stage: int = 0,
    direction_eta_multipliers: Tuple[float, ...] = (),
    direction_eta_stages: Tuple[int, ...] = (),
    recommendation_rule: str = "completed_only",
    recommendation_beta: float = 1.0,
    completed_raw_archive: Optional[_CompletedRawParetoArchive] = None,
    finite_recommendation_cache: Optional[_FiniteRecommendationCache] = None,
) -> RecommendationCheckpoint:
    # Eager compatibility helper for callers constructing one checkpoint.
    if recommendation_rule == "finite_mean":
        recommendation_beta = 0.0
    if recommendation_rule in UNCERTAIN_RECOMMENDATION_RULES:
        archive_scope = FINITE_ARCHIVE_SCOPE
    else:
        archive_scope = DEPLOYABLE_ARCHIVE_SCOPE
    selection = _recommendation_selection(
        posteriors=posteriors, directions=directions, reference_point=reference_point,
        stop_tolerance=stop_tolerance, observed_scores=observed_scores,
        observed_costs=observed_costs, completed_arms=completed_arms,
        archive_scope=archive_scope, recommendation_rule=recommendation_rule,
        recommendation_beta=recommendation_beta,
        completed_raw_archive=completed_raw_archive,
        finite_recommendation_cache=finite_recommendation_cache,
    )
    snapshot = _capture_recommendation_event(
        selection, posteriors=posteriors, observed_scores=observed_scores,
        observed_costs=observed_costs, total_evaluations=total_evaluations,
        total_cost=total_cost, bruteforce_search_cost_usd=bruteforce_search_cost_usd,
        event=event, completed_arms=completed_arms, archive_scope=archive_scope,
        current_lambda=current_lambda, lambda_stage=lambda_stage,
        direction_eta_multipliers=direction_eta_multipliers,
        direction_eta_stages=direction_eta_stages,
        recommendation_rule=recommendation_rule, recommendation_beta=recommendation_beta,
    )
    return _materialize_recommendation_event(
        snapshot, models=models, truth_vectors=truth_vectors, truth_front=truth_front,
        raw_truth_vectors=raw_truth_vectors, reference_point=reference_point,
        ground_truth_hv=ground_truth_hv,
    )


def _planning_horizon(actual_horizon: int, bin_width: int) -> int:
    if actual_horizon <= 0:
        return 0
    return int(math.ceil(actual_horizon / bin_width) * bin_width)


def _adaptive_horizon(remaining: int, batch_size: int) -> int:
    """Return planned adaptive pulls, including a smaller final batch."""
    if remaining <= 0:
        return 0
    return (int(remaining) + int(batch_size) - 1) // int(batch_size)


def _next_batch_size(remaining: int, batch_size: int) -> int:
    if remaining <= 0:
        return 0
    return min(int(batch_size), int(remaining))


def _scaled_batch_noise(
    full_batch_noise: np.ndarray,
    *,
    full_batch_size: int,
    actual_batch_size: int,
) -> np.ndarray:
    """Scale batch-mean noise when the realized batch is smaller than planned."""
    if actual_batch_size <= 0 or actual_batch_size > full_batch_size:
        raise ValueError("actual_batch_size must lie in [1, full_batch_size]")
    if actual_batch_size == full_batch_size:
        return np.asarray(full_batch_noise, dtype=np.float64)
    return np.asarray(full_batch_noise, dtype=np.float64) * (
        float(full_batch_size) / float(actual_batch_size)
    )


def _equivalent_cost(cost_reference_usd: float, desirability: float) -> float:
    if desirability <= 0.0:
        return float("inf")
    return float(cost_reference_usd * (1.0 - desirability) / desirability)


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


def mean_cost_metric_vectors(raw_vectors: np.ndarray) -> Tuple[np.ndarray, float]:
    """Offline-only common affine metric space covering the full benchmark.

    The reference is 5% above the largest full-data mean USD cost. It is never
    used for calibration, acquisition, recommendation or numerical DP bounds.
    Use this same transform to reevaluate saved recommendations from baselines.
    """
    raw = np.asarray(raw_vectors, dtype=np.float64).reshape((-1, 2))
    largest = float(np.max(raw[:, 1]))
    reference = 1.05 * largest if largest > 0.0 else 1.0
    return np.column_stack((raw[:, 0], 1.0 - raw[:, 1] / reference)), reference


def simulate_radial_gittins(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    batch_size: int = 4,
    warm_start_batch_size: Optional[int] = None,
    warm_start_question_order: str = "shared",
    directions: Optional[Iterable[Sequence[float]]] = None,
    direction_scheduler: str = "round_robin",
    eta_decay_schedule: str = "global_stop",
    recommendation_rule: str = "completed_only",
    recommendation_beta: float = 1.0,
    recommendation_min_samples: int = 0,
    fixed_prior_mean: Optional[Sequence[float] | float] = None,
    prior_variance: Sequence[float] | float = 0.04,
    obs_noise_variance: Optional[Sequence[float] | float] = None,
    cost_reference_usd: Optional[float] = None,
    cost_model: str = "reciprocal",
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
    boundary_z_padding_extra: float = 0.0,
    observation_budget_fraction: float = 1.0,
    max_total_question_evaluations: Optional[int] = None,
    max_search_cost_usd: Optional[float] = None,
    horizon_bin_width: int = 1,
    stop_tolerance: float = 1e-9,
    seed: int = 42,
    question_order: str = "shared",
    history: Optional[List[Dict[str, Any]]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    boundary_grid: Optional[RadialGittinsGrid] = None,
    boundary_cache: Optional[RadialGittinsBoundaryCache] = None,
    boundary_build_backend: str = "auto",
    boundary_jax_min_batch_size: int = 4,
    index_provider: Optional[IndexProvider] = None,
    question_universe: str = "common",
    halt_on_gittins_stop: bool = True,
    anytime: bool = False,
    lambda_initial: float = 1.0,
    lambda_decay: float = 0.5,
    record_trace: bool = True,
    record_recommendation_trajectory: bool = False,
    recommendation_checkpoint_interval: Optional[int] = None,
    recommendation_checkpoint_target: Optional[int] = None,
    recommendation_changes_only: bool = False,
    defer_recommendation_diagnostics: bool = False,
    selector_name: str = "radial_gittins",
    extra_params: Optional[Mapping[str, Any]] = None,
) -> RadialSimulationResult:
    """Replay the complete warm-start and directional radial-Gittins policy.

    ``warm_start_batch_size=None`` preserves the historical size of one batch.
    ``warm_start_question_order='shared'`` preserves the historical paired
    warm questions, while ``'independent'`` draws a separate seeded warm batch
    for each arm. Setting the warm-start size to zero enables an explicit cold
    start: ``fixed_prior_mean`` and
    ``cost_reference_usd`` are then required, reciprocal cost normalization is
    used, and no response-matrix cell is consumed before adaptive selection.
    The fixed cost-prior mean also supplies the generic expected pull cost when
    ``expected_batch_cost_usd`` is omitted.

    The default benchmark universe is the complete question intersection, so
    every arm is judged on the same questions. ``question_universe='per_arm'``
    retains ragged arm-specific tails as an explicitly diagnostic mode.

    By default, every arm follows the same seeded random question order with
    its own cursor. The paired warm batch and its calibration match the
    historical ``question_order='independent'`` mode exactly. Shared mode
    changes which questions follow the warm batch; posterior updates remain
    per arm and do not infer cross-arm covariance. In ragged mode each arm
    follows the common order restricted to its available questions.

    Adaptive replay uses full batches, then one smaller final batch when the
    remaining questions do not fill ``batch_size``. Gittins tables still plan
    every stage as a full batch; only the realized last update uses the actual
    batch size and scaled observation noise.

    The dollar-budget reservation is soft when it uses frozen warm-start
    expected costs: a realized batch can overshoot the cap. Supplying
    ``guaranteed_batch_cost_usd`` makes the reservation a hard bound, and a
    replayed batch that violates the claimed bound raises an error.

    Supplying ``index_provider`` replaces the boundary-table index for
    unfinished arms, which is how alternative acquisition rules reuse this
    replay. ``selector_name`` and ``extra_params`` then label the result so a
    baseline is not reported as radial-Gittins. Boundary tables are never built
    in that mode, so the DP grid and cache arguments are ignored.

    Trajectory recording retains event checkpoints by default. Supplying
    ``recommendation_checkpoint_interval`` additionally retains one ordinary
    checkpoint after every N adaptive pulls. Warm-start, first Gittins-stop,
    recommendation changes, and final checkpoints are always retained; this
    changes diagnostic curve resolution only, never acquisition or stopping.
    ``recommendation_checkpoint_target`` instead derives an effective interval
    from the planned full-completion adaptive pulls so that each run retains
    approximately the requested number of ordinary checkpoints. Mandatory
    warm-start, recommendation-change, lambda-stop, and final points remain.

    During sampling, membership is checked online and retained events freeze
    only their online evidence. Diagnostic archives and checkpoint HV/GD/IGD
    are materialized after sampling. With ``defer_recommendation_diagnostics``,
    only ``recommendation_events`` and the independent warm/final events are
    returned; call ``materialize_recommendation_diagnostics(result)`` later to
    populate the legacy checkpoint fields without replaying the policy.

    ``anytime=True`` starts at ``lambda_initial`` and multiplies lambda by
    ``lambda_decay`` whenever every direction stops without an intervening
    observation. Lambda multiplies ``search_cost_scale_eta`` in the frozen
    expected pull penalty. Every stage keeps the same posteriors, calibration,
    question schedules, and direction scheduler. The stop restarts acquisition
    with fresh lambda-dependent boundaries instead of forcing a pull. The
    ``halt_on_gittins_stop`` flag applies only to fixed-lambda runs.

    If ``directions`` is omitted, anytime runs use the nine interior directions
    plus the exact accuracy endpoint ``(1, 0)``; fixed-lambda runs use the nine
    interior directions. An explicit direction sequence replaces the default.

    ``direction_scheduler='accuracy_last'`` runs the other directions round-robin
    until all stop, then runs the exact accuracy endpoint until it stops. Any
    new observation invalidates every earlier stop, so the other directions
    are revisited after endpoint acquisition before global stopping. Each new
    lambda stage starts with the other directions. Completed arms are reused;
    partially observed arms retain their posteriors and remaining questions.
    The default ``'round_robin'`` keeps all directions in one cycle.

    ``recommendation_rule='completed_only'`` directly recommends the empirical
    raw Pareto frontier of all completed arms, in both fixed-lambda and anytime
    runs. No completed arm means no recommendation. Raw means are cached once
    at completion; posterior means, uncertainty and directions never rank or
    filter this recommendation.

    ``recommendation_rule='finite_lcb'`` requires raw_mean cost. For each
    actual N-question evaluation set, n observed outcomes with sum S and a
    latent Gaussian posterior (m, v) imply finite-test prediction mean
    (S + (N-n)*m)/N and variance ((N-n)**2*v + (N-n)*question_noise)/N**2.
    Question noise is the frozen warm batch-mean noise times the warm batch
    size. Prediction moments are converted to raw accuracy / mean USD, then
    all arms' (accuracy mean - beta*std, cost mean + beta*std) points are
    filtered by strict Pareto dominance. Completed predictions naturally
    equal empirical means with zero uncertainty. Means/sums are updated only
    for the evaluated arm; membership uses a two-dimensional sorting sweep.
    Acquisition and its latent posterior stay unchanged. These are Gaussian
    model-based conservative coordinates, not a joint confidence guarantee
    or an optimality result for the finite-test target.

    ``recommendation_rule='finite_mean'`` uses those same finite-test means
    without any uncertainty penalty (effective beta is always zero). Both
    finite rules support ``recommendation_min_samples``: only arms with at
    least that many actual observed questions enter the Pareto comparison.
    The default zero preserves ungated selection. If no arm qualifies, the
    recommendation is empty; even completion does not waive this strict gate.
    The threshold changes recommendation eligibility only, not acquisition.

    ``cost_model='raw_mean'`` changes the statistical cost objective to mean
    USD. The frozen warm prior mean is the mean of arm means, its cost variance
    is their sample variance, and per-question cost noise is the average
    within-arm sample variance. Both variances have a ``1e-12 * R**2`` floor.
    R defaults to the largest warm arm mean (or ``cost_reference_usd``).
    Gaussian states are updated in raw USD; directional helpers receive the
    equivalent affine view ``(accuracy, 1 - cost/R)`` and scaled variances.
    Accuracy calibration is unchanged. Legacy ``posterior_*`` summary/trace
    fields describe reward coordinates; added ``raw_posterior_*`` fields
    expose the authoritative state. Generic prior/noise arguments affect only
    accuracy in raw mode; the cost entries are estimated from warm data.

    Raw-mode DP bounds grow from observed posterior evidence without changing
    R or clipping costs. Values beyond R are allowed; a fixed reference and
    finite direction set still need not cover every raw Pareto arm. Offline
    metrics for raw-mode or empirical completed-only recommendations use a
    separate affine scale 5% above the largest full mean cost so expensive
    frontier points remain visible to HV. That scale never enters acquisition,
    recommendation, or the numerical DP bounds.

    ``eta_decay_schedule='direction_stop'`` requires anytime round-robin.
    A direction that stops halves only its own cost multiplier, then yields
    to the next direction. It is reconsidered on its next visit using shared
    observations. Legacy ``current_lambda`` / ``lambda_stage`` fields refer
    to the visited direction in this mode; per-direction eta vectors are the
    complete state. A numerical-floor stop is rechecked after observations,
    and the run ends only when every direction stops at its own floor without
    intervening observations. The default ``'global_stop'`` retains one shared
    multiplier and the existing global stage trigger.

    ``recommendation_changes_only=True`` checks membership after every pull
    and records only changes, including removals. Initial/final snapshots and
    lambda-stop events are stored separately, so unchanged stops and endpoints
    do not create duplicate recommendation checkpoints. Otherwise each anytime
    membership change and lambda stop survives checkpoint downsampling.
    Runs finish at a budget, full completion, or a numerical lambda floor:
    the largest remaining cumulative penalty is below stopping precision.
    This floor is a numerical safeguard, not an exact zero-cost optimality
    certificate, because index roots can be more sensitive than raw penalties.
    Small-penalty boundary roots receive extra tail room; if a build still
    raises ``BoundaryGridError``, anytime mode doubles that direction's padding
    and retries up to four times. The resolved grids and expansions are recorded.

    With a real :class:`RadialGittinsBoundaryCache`, table construction is
    direction-lazy: the first visit to a direction deduplicates that direction's
    unfinished-arm table requests and batches sufficiently large cold groups
    through JAX. ``boundary_build_backend='auto'`` falls back to SciPy if JAX
    is absent or fails; ``'jax'`` is strict, and ``'scipy'`` disables JAX.
    Online indices remain exact per-arm lazy values after this cold prewarm.
    """
    wall_start = time.perf_counter()
    timing_origin = wall_start
    if cost_model not in {"reciprocal", "raw_mean"}:
        raise ValueError("cost_model must be 'reciprocal' or 'raw_mean'")
    if recommendation_rule not in RECOMMENDATION_RULES:
        raise ValueError(f"recommendation_rule must be one of {RECOMMENDATION_RULES}")
    if not math.isfinite(recommendation_beta) or recommendation_beta < 0.0:
        raise ValueError("recommendation_beta must be finite and nonnegative")
    recommendation_min_samples = _validate_recommendation_min_samples(recommendation_min_samples)
    if recommendation_min_samples and recommendation_rule not in FINITE_RECOMMENDATION_RULES:
        raise ValueError("recommendation_min_samples requires a finite recommendation rule")
    if recommendation_rule == "finite_mean":
        recommendation_beta = 0.0
    finite_recommendation = recommendation_rule in FINITE_RECOMMENDATION_RULES
    uncertain_recommendation = recommendation_rule in UNCERTAIN_RECOMMENDATION_RULES
    if finite_recommendation and cost_model != "raw_mean":
        raise ValueError(f"{recommendation_rule} requires cost_model='raw_mean'")
    completed_raw_archive_cache = _CompletedRawParetoArchive() if recommendation_rule == "completed_only" else None
    batch_size = _positive_integer(batch_size, "batch_size")
    if warm_start_batch_size is None:
        resolved_warm_start_batch_size = batch_size
    else:
        if (
            isinstance(warm_start_batch_size, bool)
            or int(warm_start_batch_size) != warm_start_batch_size
            or warm_start_batch_size < 0
        ):
            raise ValueError("warm_start_batch_size must be a nonnegative integer")
        resolved_warm_start_batch_size = int(warm_start_batch_size)
    if resolved_warm_start_batch_size not in {0, batch_size}:
        raise ValueError(
            "warm_start_batch_size must be zero or equal to batch_size"
        )
    cold_start = resolved_warm_start_batch_size == 0
    if cold_start:
        if fixed_prior_mean is None:
            raise ValueError(
                "fixed_prior_mean is required when warm_start_batch_size=0"
            )
        if cost_model != "reciprocal":
            raise ValueError(
                "warm_start_batch_size=0 currently requires cost_model='reciprocal'"
            )
        if cost_reference_usd is None:
            raise ValueError(
                "cost_reference_usd is required when warm_start_batch_size=0"
            )
    elif fixed_prior_mean is not None:
        raise ValueError(
            "fixed_prior_mean is only valid when warm_start_batch_size=0"
        )
    horizon_bin_width = _positive_integer(horizon_bin_width, "horizon_bin_width")
    resolved_directions = _validate_directions(
        (DEFAULT_ANYTIME_DIRECTIONS if anytime else DEFAULT_DIRECTIONS)
        if directions is None else directions
    )
    scheduler = _DirectionScheduler(resolved_directions, direction_scheduler)
    if eta_decay_schedule not in {"global_stop", "direction_stop"}:
        raise ValueError("eta_decay_schedule must be 'global_stop' or 'direction_stop'")
    independent_eta = eta_decay_schedule == "direction_stop"
    if independent_eta and not anytime:
        raise ValueError("eta_decay_schedule='direction_stop' requires anytime=True")
    if independent_eta and direction_scheduler not in {
        "round_robin",
        "quality_then_deployment",
        "deployment_then_quality",
    }:
        raise ValueError(
            "eta_decay_schedule='direction_stop' requires direction_scheduler "
            "to use a supported asynchronous policy"
        )
    requested_checkpoint_interval = (
        None
        if recommendation_checkpoint_interval is None
        else _positive_integer(
            recommendation_checkpoint_interval,
            "recommendation_checkpoint_interval",
        )
    )
    recommendation_checkpoint_interval = requested_checkpoint_interval
    if recommendation_checkpoint_target is not None:
        recommendation_checkpoint_target = _positive_integer(
            recommendation_checkpoint_target,
            "recommendation_checkpoint_target",
        )
    boundary_build_backend = str(boundary_build_backend).lower()
    if boundary_build_backend not in {"auto", "jax", "scipy"}:
        raise ValueError(
            "boundary_build_backend must be 'auto', 'jax', or 'scipy'"
        )
    boundary_jax_min_batch_size = _positive_integer(
        boundary_jax_min_batch_size,
        "boundary_jax_min_batch_size",
    )
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be nonempty and unique")
    if not datapoints or len(set(datapoints)) != len(datapoints):
        raise ValueError("datapoints must be nonempty and unique")
    if not math.isfinite(search_cost_scale_eta) or search_cost_scale_eta <= 0.0:
        raise ValueError("search_cost_scale_eta must be finite and positive")
    if not math.isfinite(lambda_initial) or lambda_initial <= 0.0:
        raise ValueError("lambda_initial must be finite and positive")
    if not math.isfinite(lambda_decay) or not 0.0 < lambda_decay < 1.0:
        raise ValueError("lambda_decay must be finite and lie in (0, 1)")
    if (
        not math.isfinite(boundary_z_padding_extra)
        or boundary_z_padding_extra < 0.0
    ):
        raise ValueError("boundary_z_padding_extra must be finite and nonnegative")
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
    if question_order not in QUESTION_ORDERS:
        raise ValueError(f"question_order must be one of {QUESTION_ORDERS}")
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
    bruteforce_search_cost_usd = _bruteforce_search_cost_usd(
        models,
        available_by_arm,
        table,
    )

    schedule = PerArmQuestionSchedule.create_from_available(
        available_by_arm,
        warm_start_batch_size=resolved_warm_start_batch_size,
        seed=seed,
        question_order=question_order,
        warm_start_question_order=warm_start_question_order,
    )
    warm_batches = schedule.take_uniform_warm_start()
    warm_required = n_arms * resolved_warm_start_batch_size
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

    warm_scores = np.empty(
        (n_arms, resolved_warm_start_batch_size), dtype=np.float64
    )
    warm_costs = np.empty(
        (n_arms, resolved_warm_start_batch_size), dtype=np.float64
    )
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

    calibration = (
        build_fixed_prior_calibration(
            range(n_arms),
            observation_batch_size=batch_size,
            cost_reference_usd=float(cost_reference_usd),
            prior_mean=fixed_prior_mean,
            prior_variance=prior_variance,
            obs_noise_variance=obs_noise_variance,
        )
        if cold_start
        else fit_empirical_bayes_warm_start(
            warm_scores,
            warm_costs,
            arm_ids=range(n_arms),
            # Independent warm batches have no shared dataset question IDs;
            # calibration only needs stable column labels for those B draws.
            question_ids=(
                schedule.warm_start_question_ids
                if warm_start_question_order == "shared"
                else None
            ),
            prior_variance=prior_variance,
            obs_noise_variance=obs_noise_variance,
            cost_reference_usd=cost_reference_usd,
            cost_model=cost_model,
        )
    )
    model_posteriors = calibration.initialize_posteriors()
    # Raw USD is the authoritative statistical state. Existing directional
    # helpers consume an affine reward view, refreshed after each observation.
    posteriors = (
        {arm: calibration.reward_posterior(p) for arm, p in model_posteriors.items()}
        if cost_model == "raw_mean" else model_posteriors
    )
    finite_recommendation_cache = None
    if finite_recommendation:
        finite_recommendation_cache = _FiniteRecommendationCache(
            [len(available_by_arm[i]) for i in range(n_arms)],
            calibration.warm_obs_noise_var * batch_size,
            calibration.normalizer,
            recommendation_beta,
            min_samples=recommendation_min_samples,
        )
        for arm in range(n_arms):
            finite_recommendation_cache.update(
                arm, model_posteriors[arm], observed_scores[arm], observed_costs[arm],
            )
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

    current_lambda = float(lambda_initial) if anytime else 1.0
    lambda_stage = 0
    direction_eta_multipliers = [current_lambda] * len(resolved_directions)
    direction_eta_stages = [0] * len(resolved_directions)
    direction_eta_events: List[Dict[str, Any]] = []
    # Only certifications at the unchanged current multiplier belong here.
    # A pre-decay stop does not certify the newly lowered multiplier.
    direction_floor_stops: set[int] = set()
    active_eta_direction_index = 0
    base_raw_effective_pull_costs = search_cost_scale_eta * expected_batch_costs
    base_effective_pull_costs = _quantize_effective_costs(
        base_raw_effective_pull_costs,
        anchor=effective_cost_bin_anchor,
        bin_ratio=effective_cost_bin_ratio,
    )
    # Quantize once, then scale all bins with lambda. In particular, halving
    # lambda halves every continuation penalty even with non-binary bins.
    raw_effective_pull_costs = current_lambda * base_raw_effective_pull_costs
    effective_pull_costs = current_lambda * base_effective_pull_costs
    if (
        not np.all(np.isfinite(raw_effective_pull_costs))
        or not np.all(np.isfinite(effective_pull_costs))
        or np.any(raw_effective_pull_costs <= 0.0)
        or np.any(effective_pull_costs <= 0.0)
    ):
        raise ValueError(
            "lambda-scaled effective pull costs must be finite and positive"
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
    boundary_padding_overrides: Dict[
        Tuple[Tuple[float, float], int, float], float
    ] = {}
    boundary_grid_expansions: List[Dict[str, Any]] = []
    objective_grid_lower = np.zeros(2)
    objective_grid_upper = np.ones(2)
    objective_grid_expansions: List[Dict[str, Any]] = []
    if cost_model == "raw_mean":
        warm_reward_means = np.asarray([p.mean for p in posteriors.values()])
        warm_reward_std = np.sqrt(next(iter(posteriors.values())).var)
        objective_grid_lower[1] = min(0.0, float(np.min(warm_reward_means[:, 1]) - 6 * warm_reward_std[1]))
        objective_grid_upper[1] = max(1.0, float(np.max(warm_reward_means[:, 1]) + 6 * warm_reward_std[1]))

    def padding_for_arm(arm_index: int) -> float:
        return max(
            2.0 if anytime else 1.0,
            max(
                1.0,
                int(planning_horizons[arm_index])
                * float(effective_pull_costs[arm_index]) + 1.0,
            ) + boundary_z_padding_extra,
        )

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
        # Small penalties move roots into Gaussian tails. Anytime decay needs
        # additional tail room even after H*c becomes negligible. Apply the
        # minimum after the explicit padding so existing +1 low-cost runs
        # retain their grid rather than receiving the same padding twice.
        z_padding = max(
            padding_for_arm(arm_index),
            boundary_padding_overrides.get(key, 0.0),
        )
        direction_array = np.asarray(direction, dtype=np.float64)
        factors = float(np.max(direction_array)) / direction_array
        scaled_lower = factors * (objective_grid_lower - reference_array)
        scaled_upper = factors * (objective_grid_upper - reference_array)
        reachable_delta_min = float(scaled_lower[0] - scaled_upper[1])
        reachable_delta_max = float(scaled_upper[0] - scaled_lower[1])
        reachable_u_min = float(np.sum(scaled_lower) / 2.0)
        reachable_u_max = float(np.sum(scaled_upper) / 2.0)
        # The base grid supplies resolution and a normal safety envelope.  It
        # is expanded when either the cumulative-cost band or an explicitly
        # configured near-endpoint direction exceeds that envelope.
        expanded_base = replace(
            base_boundary_grid,
            z_min=min(
                base_boundary_grid.z_min,
                -6.0 - z_padding,
                reachable_u_min - z_padding,
            ),
            z_max=max(
                base_boundary_grid.z_max,
                6.0 + z_padding,
                reachable_u_max + z_padding,
                (max(abs(reachable_delta_min), abs(reachable_delta_max)) + 0.2) / 2.0 + z_padding,
            ),
            delta_min=min(
                base_boundary_grid.delta_min,
                reachable_delta_min - 0.2,
            ),
            delta_max=max(
                base_boundary_grid.delta_max,
                reachable_delta_max + 0.2,
            ),
        )
        resolved = direction_aware_grid(
            direction,
            base_grid=expanded_base,
            reference=resolved_reference,
            z_padding=z_padding,
            objective_lower=objective_grid_lower,
            objective_upper=objective_grid_upper,
        )
        resolved_boundary_grids[key] = resolved
        return resolved

    def _build_with_boundary_retries(
        direction: Tuple[float, float],
        arm_indices: Sequence[int],
        build: Callable[[], Any],
    ) -> Any:
        """Widen failed anytime root-search bands, with an explicit retry bound."""
        for attempt in range(_ANYTIME_BOUNDARY_MAX_WIDENING_RETRIES + 1):
            try:
                return build()
            except BoundaryGridError as error:
                if (not anytime and cost_model != "raw_mean") or attempt == _ANYTIME_BOUNDARY_MAX_WIDENING_RETRIES:
                    raise
                keys_seen = set()
                for arm_index in arm_indices:
                    key = (
                        direction,
                        int(planning_horizons[arm_index]),
                        float(effective_pull_costs[arm_index]),
                    )
                    if key in keys_seen:
                        continue
                    keys_seen.add(key)
                    previous_padding = max(
                        padding_for_arm(arm_index),
                        boundary_padding_overrides.get(key, 0.0),
                    )
                    new_padding = 2.0 * previous_padding
                    boundary_padding_overrides[key] = new_padding
                    resolved_boundary_grids.pop(key, None)
                    boundary_grid_expansions.append({
                        "current_lambda": current_lambda,
                        "lambda_stage": lambda_stage,
                        "direction": list(direction),
                        "horizon": key[1],
                        "effective_pull_cost": key[2],
                        "previous_z_padding": previous_padding,
                        "new_z_padding": new_padding,
                        "build_retry": attempt + 1,
                        "reason": str(error),
                    })
        raise AssertionError("unreachable boundary retry state")

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

    remaining_after_warm = [schedule.remaining(i) for i in range(n_arms)]
    actual_horizons = np.asarray(
        [_adaptive_horizon(remaining, batch_size) for remaining in remaining_after_warm],
        dtype=np.int64,
    )
    planned_adaptive_pulls = int(np.sum(actual_horizons))
    if recommendation_checkpoint_target is not None:
        recommendation_checkpoint_interval = max(
            1,
            int(math.ceil(
                planned_adaptive_pulls / recommendation_checkpoint_target
            )),
        )
    planned_partial_tail_cells = int(
        sum(remaining % batch_size for remaining in remaining_after_warm)
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
    noise_var = calibration.reward_obs_noise_var.copy()
    cache = (
        boundary_cache
        if boundary_cache is not None
        else RadialGittinsBoundaryCache()
    )
    optimized_boundary_cache = type(cache) is RadialGittinsBoundaryCache
    if (
        index_provider is None
        and boundary_build_backend == "jax"
        and not optimized_boundary_cache
    ):
        raise TypeError(
            "boundary_build_backend='jax' requires a concrete "
            "RadialGittinsBoundaryCache"
        )
    cache_size_before = len(cache)
    cache_stats_before = _boundary_cache_stats_snapshot(cache)
    axis_cache = AxisGittinsBoundaryCache()
    online_value_cache = _VersionedArmValueCache(n_arms)
    direction_boundary_tables: Dict[
        Tuple[float, float],
        Dict[int, RadialGittinsBoundaryTable],
    ] = {}
    boundary_prewarm_stats = {
        name: 0 for name in _BOUNDARY_PREWARM_STAT_FIELDS
    }
    auto_jax_disabled = False
    auto_jax_workload_skips = 0

    def _expand_observed_objective_grid(arm_index: int, evaluations: int) -> None:
        """Expand from observed posterior evidence without clipping or oracle bounds."""
        if cost_model != "raw_mean":
            return
        posterior = posteriors[arm_index]
        radius = 6.0 * math.sqrt(float(posterior.var[1]))
        lower = float(posterior.mean[1]) - radius
        upper = float(posterior.mean[1]) + radius
        if lower >= objective_grid_lower[1] and upper <= objective_grid_upper[1]:
            return
        old_lower, old_upper = objective_grid_lower.copy(), objective_grid_upper.copy()
        # Geometric growth prevents tiny successive excursions from rebuilding
        # every table. The statistical model and frozen reference do not move.
        span = float(objective_grid_upper[1] - objective_grid_lower[1])
        if lower < objective_grid_lower[1]:
            objective_grid_lower[1] = min(lower, objective_grid_lower[1] - span)
        if upper > objective_grid_upper[1]:
            objective_grid_upper[1] = max(upper, objective_grid_upper[1] + span)
        objective_grid_expansions.append({
            "arm_index": int(arm_index), "evaluations": int(evaluations),
            "old_lower": old_lower.tolist(), "old_upper": old_upper.tolist(),
            "new_lower": objective_grid_lower.tolist(), "new_upper": objective_grid_upper.tolist(),
        })
        resolved_boundary_grids.clear()
        direction_boundary_tables.clear()
        online_value_cache.clear_radial_indices()

    def _prewarm_direction_boundaries(
        direction: Tuple[float, float],
        unfinished_arms: Sequence[int],
    ) -> None:
        """Build one direction's unique cold tables once per lambda stage."""
        nonlocal auto_jax_disabled, auto_jax_workload_skips
        if _direction_axis(direction) is not None:
            return
        if direction in direction_boundary_tables:
            return
        # Test doubles and third-party cache-like objects retain the historical
        # scalar `get` protocol. The optimized adapter intentionally relies on
        # the concrete cache's validated probe/publish machinery.
        if not optimized_boundary_cache:
            return

        arm_indices = tuple(int(arm_index) for arm_index in unfinished_arms)
        requests = tuple(
            RadialGittinsPrewarmRequest(
                direction=direction,
                effective_pull_cost=float(effective_pull_costs[arm_index]),
                initial_var=initial_var,
                obs_noise_var=noise_var,
                horizon=int(planning_horizons[arm_index]),
                grid=grid_for_arm(direction, arm_index),
            )
            for arm_index in arm_indices
        )
        explicit_jax = boundary_build_backend == "jax"
        unique_requests = tuple(dict.fromkeys(requests))
        estimated_cell_stages = sum(
            request.horizon * request.grid.state_size**2
            for request in unique_requests
        )
        auto_workload_eligible = (
            estimated_cell_stages >= _AUTO_JAX_MIN_CELL_STAGES
        )
        jax_available: bool | None
        if (
            boundary_build_backend == "scipy"
            or auto_jax_disabled
            or (
                boundary_build_backend == "auto"
                and not auto_workload_eligible
            )
        ):
            jax_available = False
        elif explicit_jax:
            jax_available = True
        else:
            jax_available = None
        result = _build_with_boundary_retries(
            direction,
            arm_indices,
            lambda: prewarm_radial_gittins_boundaries(
                tuple(
                    replace(request, grid=grid_for_arm(direction, arm_index))
                    for request, arm_index in zip(requests, arm_indices)
                ),
                cache=cache,
                jax_min_batch_size=(
                    1 if explicit_jax else boundary_jax_min_batch_size
                ),
                jax_available=jax_available,
                fallback_on_jax_error=boundary_build_backend == "auto",
            ),
        )
        if (
            boundary_build_backend == "auto"
            and not auto_jax_disabled
            and not auto_workload_eligible
            and result.stats.cold_misses
        ):
            auto_jax_workload_skips += 1
        if (
            boundary_build_backend == "auto"
            and result.stats.jax_fallback_groups
        ):
            # A backend/import/device failure is normally persistent for this
            # process. Avoid paying for the same failed JAX attempt on every
            # later direction; SciPy remains exact and available.
            auto_jax_disabled = True
        direction_boundary_tables[direction] = dict(
            zip(arm_indices, result.tables)
        )
        for name in _BOUNDARY_PREWARM_STAT_FIELDS:
            boundary_prewarm_stats[name] += int(getattr(result.stats, name))

    def _compute_radial_terminal_utility(
        direction_index: int,
        arm_index: int,
    ) -> float:
        direction = resolved_directions[int(direction_index)]
        posterior = posteriors[int(arm_index)]
        return terminal_expected_direction_utility(
            posterior.mean,
            posterior.var,
            direction,
            resolved_reference,
        )

    def _cached_radial_terminal_utility(
        direction_index: int,
        arm_index: int,
    ) -> float:
        direction_index = int(direction_index)

        def compute(current_arm: int) -> float:
            return _compute_radial_terminal_utility(
                direction_index,
                current_arm,
            )

        return online_value_cache.get_or_compute(
            ("radial_terminal", direction_index),
            arm_index,
            compute,
        )

    def _observed_endpoint_tiebreak(direction: Sequence[float], arm_index: int) -> float:
        if _direction_axis(direction) == 0:
            return (
                -float(np.mean(observed_costs[arm_index]))
                if observed_costs[arm_index]
                else float(posteriors[arm_index].mean[1])
            )
        return (
            float(np.mean(observed_scores[arm_index]))
            if observed_scores[arm_index]
            else float(posteriors[arm_index].mean[0])
        )

    trace: List[Dict[str, Any]] = []

    def _append_trace(event: Dict[str, Any]) -> None:
        if record_trace:
            trace.append(event)

    for arm_index in (range(n_arms) if not cold_start else ()):
        _append_trace(
            {
                "event": "warm_start",
                "current_lambda": current_lambda,
                "lambda_stage": lambda_stage,
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
                **({
                    "raw_posterior_mean_after": model_posteriors[arm_index].mean.tolist(),
                    "raw_posterior_var_after": model_posteriors[arm_index].var.tolist(),
                } if cost_model == "raw_mean" else {}),
                "cumulative_evaluations": (arm_index + 1) * batch_size,
                "cumulative_search_cost_usd": float(
                    np.sum(warm_costs[: arm_index + 1])
                ),
            }
        )

    direction_index = scheduler.direction_index
    visit_counts = np.zeros(len(resolved_directions), dtype=np.int64)
    global_step = 0
    stop_reason = "all_directions_gittins_stop"
    gittins_stop_evaluations: Optional[int] = None
    gittins_stop_cost_usd: Optional[float] = None
    recommendation_events: List[RecommendationEvent] = []
    recommendation_initial_event: Optional[RecommendationEvent] = None
    recommendation_final_event: Optional[RecommendationEvent] = None
    recommendation_recording = {
        "membership_checks": 0,
        "membership_wall_time_seconds": 0.0,
        "captured_events": 0,
        "event_capture_wall_time_seconds": 0.0,
        "diagnostic_materializations": 0,
        "diagnostic_wall_time_seconds": 0.0,
    }
    adaptive_checkpoint_pulls = 0
    past_gittins_stop = False
    lambda_stop_events: List[Dict[str, Any]] = []
    stage_timing_events: List[Dict[str, Any]] = []
    lambda_numerical_threshold: Optional[float] = None
    last_deployable_archive: set[int] = set()
    stopping_index_scale = 1.0

    # Full-data diagnostic inputs are computed once. Online membership and
    # event capture never read these arrays; retained checkpoint diagnostics
    # use them only after sampling (or on an explicit materialization request).
    truth_metric_start = time.perf_counter()
    raw_truth_vectors = _full_raw_objective_vectors(
        models,
        evaluation_datapoints,
        table,
    )
    truth_vectors, metric_cost_reference_usd = mean_cost_metric_vectors(raw_truth_vectors)
    metric_reference = (0.0, 0.0)
    truth_front = truth_vectors[nondominated_indices(truth_vectors)]
    ground_truth_hv = hypervolume_2d(truth_front, metric_reference)
    wall_start += time.perf_counter() - truth_metric_start

    def _append_recommendation_checkpoint(
        event: str,
        *,
        completed_arm_changed: bool = False,
    ) -> None:
        nonlocal last_deployable_archive
        nonlocal recommendation_initial_event, recommendation_final_event
        if not record_recommendation_trajectory:
            return
        if (
            not uncertain_recommendation and recommendation_changes_only
            and event == "adaptive_pull" and not completed_arm_changed
        ):
            # Completed measurements are immutable; without a new completion
            # the empirical frontier cannot change. No histories or DP reads.
            return
        ordinary_checkpoint_due = (
            event != "adaptive_pull"
            or (
                recommendation_checkpoint_interval is not None
                and adaptive_checkpoint_pulls
                % recommendation_checkpoint_interval
                == 0
            )
        )
        inspect_every_pull = recommendation_changes_only or uncertain_recommendation
        if (
            not ordinary_checkpoint_due
            and not inspect_every_pull
            and not completed_arm_changed
        ):
            return
        completed_at_checkpoint = tuple(
            i
            for i in range(n_arms)
            if adaptive_pulls[i] >= actual_horizons[i]
        )
        # Completed-only has the same empirical contract at every checkpoint.
        archive_scope = (
            FINITE_ARCHIVE_SCOPE if uncertain_recommendation else DEPLOYABLE_ARCHIVE_SCOPE
        )
        recommendation_recording["membership_checks"] += 1
        membership_started = time.perf_counter()
        selection = _recommendation_selection(
            posteriors=posteriors,
            directions=resolved_directions,
            reference_point=resolved_reference,
            stop_tolerance=stop_tolerance,
            observed_scores=observed_scores,
            observed_costs=observed_costs,
            completed_arms=completed_at_checkpoint,
            archive_scope=archive_scope,
            recommendation_rule=recommendation_rule,
            recommendation_beta=recommendation_beta,
            completed_raw_archive=completed_raw_archive_cache,
            finite_recommendation_cache=finite_recommendation_cache,
        )
        recommendation_recording["membership_wall_time_seconds"] += time.perf_counter() - membership_started
        archive_members = set(selection.selected_arm_indices)
        membership_changed = archive_members != last_deployable_archive
        added = tuple(sorted(archive_members - last_deployable_archive))
        removed = tuple(sorted(last_deployable_archive - archive_members))
        last_deployable_archive = archive_members
        retain = (
            membership_changed
            or (ordinary_checkpoint_due and not recommendation_changes_only)
        )
        # Warm/final preserve their own current evidence even if their set is
        # unchanged. Discarded LCB pulls never scan histories; no discarded
        # pull constructs diagnostic archives or metrics.
        if not retain and event not in {"after_warm_start", "initial_prior", "final"}:
            return
        capture_started = time.perf_counter()
        checkpoint = _capture_recommendation_event(
            selection,
            posteriors=posteriors,
            observed_scores=observed_scores,
            observed_costs=observed_costs,
            total_evaluations=total_evaluations,
            total_cost=total_cost,
            bruteforce_search_cost_usd=bruteforce_search_cost_usd,
            event=event,
            completed_arms=completed_at_checkpoint,
            archive_scope=archive_scope,
            current_lambda=current_lambda,
            lambda_stage=lambda_stage,
            direction_eta_multipliers=tuple(direction_eta_multipliers),
            direction_eta_stages=tuple(direction_eta_stages),
            recommendation_rule=recommendation_rule,
            recommendation_beta=recommendation_beta,
        )
        recommendation_recording["captured_events"] += 1
        recommendation_recording["event_capture_wall_time_seconds"] += time.perf_counter() - capture_started
        if event in {"after_warm_start", "initial_prior"}:
            recommendation_initial_event = checkpoint
        if event == "final":
            recommendation_final_event = checkpoint
        if not retain:
            return
        checkpoint = replace(
            checkpoint,
            added_arm_indices=added,
            removed_arm_indices=removed,
        )
        if added:
            checkpoint = replace(
                checkpoint,
                event=(
                    "recommendation_added" if event == "adaptive_pull" else event
                ),
                added_arm_indices=added,
            )
        elif membership_changed and event == "adaptive_pull":
            checkpoint = replace(checkpoint, event="recommendation_changed")
        recommendation_events.append(checkpoint)

    _append_recommendation_checkpoint(
        "initial_prior" if cold_start else "after_warm_start"
    )
    lambda_stage_wall_start = time.perf_counter()
    stage_timing_events.append({
        "event": "cold_start_complete" if cold_start else "warm_start_complete",
        "lambda_stage": 0,
        "current_lambda": current_lambda,
        "stage_wall_time_seconds": float(lambda_stage_wall_start - timing_origin),
        "run_wall_time_seconds": float(lambda_stage_wall_start - timing_origin),
        "cumulative_evaluations": int(total_evaluations),
        "cumulative_search_cost_usd": float(total_cost),
        "budget_fraction": float(total_cost) / float(bruteforce_search_cost_usd),
    })

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
        if total_evaluations >= question_cap:
            _append_trace(
                {
                    "event": "budget_stop",
                    "current_lambda": current_lambda,
                    "lambda_stage": lambda_stage,
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
            _append_trace(
                {
                    "event": "budget_stop",
                    "current_lambda": current_lambda,
                    "lambda_stage": lambda_stage,
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
        if independent_eta:
            active_eta_direction_index = direction_index
            current_lambda = direction_eta_multipliers[direction_index]
            lambda_stage = direction_eta_stages[direction_index]
            raw_effective_pull_costs = current_lambda * base_raw_effective_pull_costs
            effective_pull_costs = current_lambda * base_effective_pull_costs
        context = DirectionVisitContext(
            global_step=global_step,
            direction_index=direction_index,
            direction=direction,
            visit_count=int(visit_counts[direction_index]),
            completed_arms=completed,
            unfinished_arms=unfinished,
            adaptive_pulls=tuple(int(x) for x in adaptive_pulls),
            model_names=tuple(models),
            posteriors=MappingProxyType(posteriors),
            reference_point=resolved_reference,
            effective_pull_costs=tuple(
                float(x) for x in effective_pull_costs
            ),
            current_lambda=current_lambda,
            lambda_stage=lambda_stage,
            direction_eta_multipliers=tuple(direction_eta_multipliers),
            direction_eta_stages=tuple(direction_eta_stages),
        )
        visit_counts[direction_index] += 1

        unfinished_indices: Dict[int, float] = {}
        if index_provider is not None:
            # External providers may depend on global_step, visit_count,
            # other arms, or any other live context field. They therefore
            # remain intentionally uncached.
            for arm_index in unfinished:
                unfinished_indices[arm_index] = float(
                    index_provider(context, arm_index)
                )
        else:
            _prewarm_direction_boundaries(direction, unfinished)

            def compute_radial_index(arm_index: int) -> float:
                axis = _direction_axis(direction)
                if axis is not None:
                    # Endpoints use a scalar required-completion problem.
                    # Its table depends only on the active variance/noise,
                    # horizon, resolution and lambda-scaled cost; the inactive
                    # coordinate never enters acquisition or its grid.
                    axis_table = axis_cache.get(
                        effective_pull_cost=float(effective_pull_costs[arm_index]),
                        initial_var=float(initial_var[axis]),
                        obs_noise_var=float(noise_var[axis]),
                        horizon=int(planning_horizons[arm_index]),
                        grid_size=base_boundary_grid.state_size,
                    )
                    return axis_table.index(
                        int(adaptive_pulls[arm_index]),
                        float(posteriors[arm_index].mean[axis]),
                        reference=float(resolved_reference[axis]),
                    )
                table_for_arm = direction_boundary_tables.get(
                    direction, {}
                ).get(arm_index)
                if table_for_arm is None:
                    table_for_arm = _build_with_boundary_retries(
                        direction,
                        (arm_index,),
                        lambda: cache.get(
                            direction=direction,
                            effective_pull_cost=float(
                                effective_pull_costs[arm_index]
                            ),
                            initial_var=initial_var,
                            obs_noise_var=noise_var,
                            horizon=int(planning_horizons[arm_index]),
                            grid=grid_for_arm(direction, arm_index),
                        ),
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
                return u - boundary

            unfinished_indices = {
                arm_index: online_value_cache.get_or_compute(
                    ("radial_index", direction_index),
                    arm_index,
                    compute_radial_index,
                )
                for arm_index in unfinished
            }

        # Keep the ordered arm scan in evaluate_direction_status. Its
        # tolerance-aware tie rule is an ordered fold, so replacing it
        # with a plain max-heap could change the selected arm.
        completed_terminal_indices = {
            arm_index: _cached_radial_terminal_utility(
                direction_index,
                arm_index,
            )
            for arm_index in completed
        }
        status = evaluate_direction_status(
            direction=direction,
            posteriors=posteriors,
            completed_arms=completed,
            unfinished_indices=unfinished_indices,
            reference_point=resolved_reference,
            stop_tolerance=stop_tolerance,
            completed_terminal_indices=completed_terminal_indices,
            completed_tiebreak_values=(
                {
                    arm_index: _observed_endpoint_tiebreak(direction, arm_index)
                    for arm_index in completed
                }
                if _direction_axis(direction) is not None else None
            ),
        )
        visit_event: Dict[str, Any] = {
            "event": "direction_visit",
            "current_lambda": current_lambda,
            "lambda_stage": lambda_stage,
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
        } if record_trace or history is not None else {}

        if independent_eta and status.should_stop:
            _append_trace(visit_event)
            if history is not None:
                history.append(dict(visit_event))
            remaining_penalty = max(
                float(planning_horizons[i] - adaptive_pulls[i])
                * float(effective_pull_costs[i]) for i in unfinished
            )
            index_scale = max(1.0, abs(status.best_completed_index), abs(status.best_unfinished_index))
            lambda_numerical_threshold = max(
                stop_tolerance, float(np.finfo(np.float64).eps) * index_scale,
            )
            next_multiplier = current_lambda * lambda_decay
            next_raw_costs = next_multiplier * base_raw_effective_pull_costs
            next_effective_costs = next_multiplier * base_effective_pull_costs
            floor_reason = None
            if remaining_penalty <= lambda_numerical_threshold:
                floor_reason = "numerical_floor"
            elif (
                not 0.0 < next_multiplier < current_lambda
                or np.any(next_raw_costs <= 0.0)
                or np.any(next_effective_costs <= 0.0)
            ):
                floor_reason = "underflow"
            if floor_reason is None:
                direction_eta_multipliers[direction_index] = next_multiplier
                direction_eta_stages[direction_index] += 1
                direction_floor_stops.discard(direction_index)
                direction_boundary_tables.pop(direction, None)
                online_value_cache.clear_direction_radial_indices(direction_index)
            else:
                direction_floor_stops.add(direction_index)
                # Sequential endpoint schedules stay on one axis through its
                # eta decays and move to the other only after reaching the
                # numerical floor.  Round-robin policies are unaffected by
                # recording this stop because they have a single group.
                scheduler.record_stop()
            eta_event = {
                "event": "direction_eta_decay" if floor_reason is None else "direction_eta_floor_stop",
                "global_step": global_step, "direction_index": direction_index,
                "direction": list(direction), "current_lambda": current_lambda,
                "lambda_stage": lambda_stage,
                "next_lambda": next_multiplier if floor_reason is None else None,
                "next_stage": direction_eta_stages[direction_index],
                "direction_eta_multipliers": list(direction_eta_multipliers),
                "direction_eta_stages": list(direction_eta_stages),
                "floor_reason": floor_reason,
                "max_remaining_effective_penalty": remaining_penalty,
                "numerical_penalty_threshold": lambda_numerical_threshold,
                "cumulative_evaluations": int(total_evaluations),
                "cumulative_search_cost_usd": float(total_cost),
                "budget_fraction": total_cost / bruteforce_search_cost_usd,
            }
            direction_eta_events.append(eta_event)
            _append_trace(dict(eta_event))
            # This is a local event, not a common-price global lambda stop.
            # A local price change cannot change completed-only winners.
            # Record prices here; recommendation checkpoints are retained at
            # completions and at the final state, without duplicate archives.
            global_step += 1
            if len(direction_floor_stops) == len(resolved_directions):
                stop_reason = "direction_eta_numerical_floor"
                gittins_stop_evaluations = int(total_evaluations)
                gittins_stop_cost_usd = float(total_cost)
                break
            direction_index = scheduler.advance()
            continue

        if status.should_stop and not past_gittins_stop:
            scheduler.record_stop()
            stopping_index_scale = max(
                stopping_index_scale,
                abs(status.best_completed_index),
                abs(status.best_unfinished_index),
            )
            if scheduler.all_stopped:
                if gittins_stop_evaluations is None:
                    gittins_stop_evaluations = int(total_evaluations)
                    gittins_stop_cost_usd = float(total_cost)
                    if not anytime:
                        _append_recommendation_checkpoint("gittins_stop")
                if anytime:
                    # Stops from every direction since the last observation
                    # certify this stage's stop. Recompute after decay;
                    # none of the old skip decisions is valid at a new cost.
                    _append_trace(visit_event)
                    global_step += 1
                    remaining_penalty = max(
                        float(planning_horizons[i] - adaptive_pulls[i])
                        * float(effective_pull_costs[i])
                        for i in unfinished
                    )
                    lambda_numerical_threshold = max(
                        stop_tolerance,
                        float(np.finfo(np.float64).eps) * stopping_index_scale,
                    )
                    next_lambda = current_lambda * lambda_decay
                    next_raw_costs = next_lambda * base_raw_effective_pull_costs
                    next_effective_costs = next_lambda * base_effective_pull_costs
                    halt_reason: Optional[str] = None
                    if remaining_penalty <= lambda_numerical_threshold:
                        halt_reason = "lambda_numerical_floor"
                    elif (
                        not 0.0 < next_lambda < current_lambda
                        or np.any(next_raw_costs <= 0.0)
                        or np.any(next_effective_costs <= 0.0)
                    ):
                        halt_reason = "lambda_underflow"
                    previous_stop = (
                        lambda_stop_events[-1] if lambda_stop_events else None
                    )
                    same_observations = (
                        previous_stop is not None
                        and previous_stop["cumulative_evaluations"] == total_evaluations
                    )
                    lambda_stop_event = {
                        "event": "lambda_stop",
                        "global_step": global_step,
                        "current_lambda": current_lambda,
                        "lambda_stage": lambda_stage,
                        "next_lambda": next_lambda if halt_reason is None else None,
                        "continued": halt_reason is None,
                        "halt_reason": halt_reason,
                        "cumulative_evaluations": int(total_evaluations),
                        "cumulative_search_cost_usd": float(total_cost),
                        "budget_fraction": total_cost / bruteforce_search_cost_usd,
                        "max_remaining_effective_penalty": remaining_penalty,
                        "numerical_penalty_threshold": lambda_numerical_threshold,
                        "consecutive_stops_without_evaluation": (
                            int(previous_stop["consecutive_stops_without_evaluation"]) + 1
                            if same_observations else 0
                        ),
                    }
                    lambda_stop_events.append(lambda_stop_event)
                    _append_trace(dict(lambda_stop_event))
                    _append_recommendation_checkpoint("lambda_stop")
                    stage_stopped_at = time.perf_counter()
                    timing_fields = {
                        "stage_wall_time_seconds": float(
                            stage_stopped_at - lambda_stage_wall_start
                        ),
                        "run_wall_time_seconds": float(
                            stage_stopped_at - timing_origin
                        ),
                    }
                    lambda_stop_event.update(timing_fields)
                    if record_trace:
                        trace[-1].update(timing_fields)
                    stage_timing_events.append({
                        **lambda_stop_event,
                        "event": "lambda_stage_stop",
                    })
                    if halt_reason is not None:
                        stop_reason = halt_reason
                        break
                    current_lambda = next_lambda
                    lambda_stage += 1
                    direction_eta_multipliers[:] = [current_lambda] * len(resolved_directions)
                    direction_eta_stages[:] = [lambda_stage] * len(resolved_directions)
                    raw_effective_pull_costs = next_raw_costs
                    effective_pull_costs = next_effective_costs
                    direction_boundary_tables.clear()
                    online_value_cache.clear_radial_indices()
                    stopping_index_scale = 1.0
                    direction_index = scheduler.start_next_stage()
                    lambda_stage_wall_start = stage_stopped_at
                    continue
                if halt_on_gittins_stop:
                    _append_trace(visit_event)
                    global_step += 1
                    stop_reason = "all_directions_gittins_stop"
                    break
                past_gittins_stop = True
                # Fall through and force-pull under the remaining budget.
            else:
                _append_trace(visit_event)
                global_step += 1
                direction_index = scheduler.advance()
                continue

        selected_arm = status.best_unfinished_arm
        if selected_arm is None:
            _append_trace(visit_event)
            stop_reason = "all_arms_completed"
            break
        planned_batch_size = _next_batch_size(
            schedule.remaining(selected_arm),
            batch_size,
        )
        if planned_batch_size <= 0:
            raise RuntimeError("unfinished arm has no remaining questions")
        if total_evaluations + planned_batch_size > question_cap:
            _append_trace(visit_event)
            stop_reason = "question_budget"
            break
        batch_scale = float(planned_batch_size) / float(batch_size)
        reservation_cost = float(reservation_costs[selected_arm]) * batch_scale
        visit_event.update(
            {
                "candidate_arm": selected_arm,
                "candidate_model": models[selected_arm],
                "expected_batch_search_cost_usd": float(
                    expected_batch_costs[selected_arm]
                )
                * batch_scale,
                "budget_reservation_cost_usd": reservation_cost,
                "cost_budget_guard": cost_budget_guard,
                "raw_effective_pull_cost": float(
                    raw_effective_pull_costs[selected_arm]
                ),
                "quantized_effective_pull_cost": float(
                    effective_pull_costs[selected_arm]
                ),
                "forced_after_gittins_stop": past_gittins_stop,
                "planned_batch_size": planned_batch_size,
            }
        )
        if max_search_cost_usd is not None and (
            total_cost + reservation_cost > max_search_cost_usd
        ):
            _append_trace(visit_event)
            stop_reason = "search_cost_budget"
            break

        question_ids = schedule.next_batch(selected_arm, planned_batch_size)
        if len(question_ids) != planned_batch_size:
            raise RuntimeError("adaptive replay produced an unexpected batch size")
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
        guaranteed_bound = (
            None
            if guaranteed_batch_costs is None
            else float(guaranteed_batch_costs[selected_arm]) * batch_scale
        )
        if (
            guaranteed_bound is not None
            and realized_batch_cost > guaranteed_bound
        ):
            raise ValueError(
                "guaranteed_batch_cost_usd is violated by an adaptive batch for "
                f"model {models[selected_arm]!r}: observed "
                f"${realized_batch_cost:.6f}, bound "
                f"${guaranteed_bound:.6f}"
            )
        for question_id in question_ids:
            observed_cells.add((selected_arm, question_id))

        model_observation = calibration.posterior_observations(
            batch_scores,
            batch_costs,
        ).mean(axis=0)
        posterior = posteriors[selected_arm]
        mean_before = posterior.mean.copy()
        var_before = posterior.var.copy()
        model_posterior = model_posteriors[selected_arm]
        raw_mean_before = model_posterior.mean.copy()
        raw_var_before = model_posterior.var.copy()
        model_posterior.update(
            model_observation,
            _scaled_batch_noise(
                calibration.warm_obs_noise_var,
                full_batch_size=batch_size,
                actual_batch_size=planned_batch_size,
            ),
            batch_size=planned_batch_size,
        )
        if cost_model == "raw_mean":
            posterior = calibration.reward_posterior(model_posterior)
            posteriors[selected_arm] = posterior
        adaptive_pulls[selected_arm] += 1
        online_value_cache.invalidate(selected_arm)
        observed_scores[selected_arm].extend(batch_scores)
        observed_costs[selected_arm].extend(batch_costs)
        observed_latencies[selected_arm].extend(batch_latencies)
        if finite_recommendation_cache is not None:
            finite_recommendation_cache.update(
                selected_arm, model_posterior, observed_scores[selected_arm], observed_costs[selected_arm],
            )
        total_cost += realized_batch_cost
        total_evaluations += planned_batch_size
        _expand_observed_objective_grid(selected_arm, total_evaluations)

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
                **({
                    "raw_posterior_mean_before": raw_mean_before.tolist(),
                    "raw_posterior_mean_after": model_posterior.mean.tolist(),
                    "raw_posterior_var_before": raw_var_before.tolist(),
                    "raw_posterior_var_after": model_posterior.var.tolist(),
                } if cost_model == "raw_mean" else {}),
                "cumulative_evaluations": total_evaluations,
                "cumulative_search_cost_usd": total_cost,
                "cost_budget_overshoot_usd_after": (
                    max(0.0, total_cost - max_search_cost_usd)
                    if max_search_cost_usd is not None
                    else 0.0
                ),
            }
        )
        _append_trace(visit_event)
        if history is not None:
            history.append(dict(visit_event))
        adaptive_checkpoint_pulls += 1
        _append_recommendation_checkpoint(
            "adaptive_pull",
            completed_arm_changed=(
                adaptive_pulls[selected_arm] >= actual_horizons[selected_arm]
            ),
        )
        scheduler.record_observation()
        direction_floor_stops.clear()
        stopping_index_scale = 1.0
        global_step += 1
        direction_index = scheduler.advance(force_round_robin=past_gittins_stop)

    completed_final = [
        i for i in range(n_arms) if adaptive_pulls[i] >= actual_horizons[i]
    ]
    direction_winners: List[DirectionWinner] = []
    for direction_index_final, direction in enumerate(resolved_directions):
        if not completed_final:
            break
        utilities = {
            arm_index: _cached_radial_terminal_utility(
                direction_index_final,
                arm_index,
            )
            for arm_index in completed_final
        }
        winner_arm, utility = _best_direction_index(
            utilities,
            stop_tolerance,
            direction=direction,
            posteriors=posteriors,
            endpoint_secondary_values=(
                {
                    arm_index: _observed_endpoint_tiebreak(direction, arm_index)
                    for arm_index in completed_final
                }
                if _direction_axis(direction) is not None else None
            ),
        )
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
    if finite_recommendation:
        assert finite_recommendation_cache is not None
        archive_arms = list(finite_recommendation_cache.select().selected_arm_indices)
        online_raw_archive_arms = raw_archive_arm_indices(
            archive_arms, finite_recommendation_cache.observed_means[archive_arms],
        )
        oracle_raw_archive_arms = raw_archive_arm_indices(archive_arms, raw_truth_vectors[archive_arms])
    else:
        assert completed_raw_archive_cache is not None
        completed_raw_archive_cache.update(completed_final, observed_scores, observed_costs)
        archive_arms = sorted(completed_raw_archive_cache.frontier)
        online_raw_archive_arms = list(archive_arms)
        # Full-data values only describe a separately named diagnostic;
        # membership above depends exclusively on completed observed means.
        oracle_raw_archive_arms = raw_archive_arm_indices(
            completed_final,
            raw_truth_vectors[completed_final] if completed_final else np.empty((0, 2)),
        )
    selected_models = [models[i] for i in archive_arms]

    model_results: List[RadialArmSummary] = []
    winner_arm_set = set(unique_winner_arms)
    archive_arm_set = set(archive_arms)
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
                observed_accuracy=float(np.mean(scores)) if scores else None,
                observed_mean_cost_usd=float(np.mean(costs)) if costs else None,
                observed_total_cost_usd=float(sum(costs)),
                equivalent_posterior_cost_usd=float(model_posteriors[arm_index].mean[1]) if cost_model == "raw_mean" else _equivalent_cost(
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
                raw_posterior_mean=tuple(float(x) for x in model_posteriors[arm_index].mean) if cost_model == "raw_mean" else None,
                raw_posterior_var=tuple(float(x) for x in model_posteriors[arm_index].var) if cost_model == "raw_mean" else None,
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
    # Preserve the historical stopped_by_gittins diagnostic, which also counts
    # earlier stops in fixed-budget force-continuation runs. These explicit
    # fields distinguish a stage trigger from the reason this run actually ends.
    gittins_stop_triggered = gittins_stop_evaluations is not None or bool(direction_eta_events)
    halted_by_gittins = stop_reason in {
        "all_directions_gittins_stop", "lambda_numerical_floor", "lambda_underflow",
        "direction_eta_numerical_floor",
    }
    gittins_stop_budget_fraction = (
        float(gittins_stop_cost_usd) / float(bruteforce_search_cost_usd)
        if gittins_stop_cost_usd is not None
        else None
    )
    if record_recommendation_trajectory:
        _append_recommendation_checkpoint("final")
    run_completed_at = time.perf_counter()
    stage_timing_events.append({
        "event": "run_complete",
        "lambda_stage": lambda_stage,
        "current_lambda": current_lambda,
        "stop_reason": stop_reason,
        "stage_wall_time_seconds": float(
            run_completed_at - lambda_stage_wall_start
        ),
        "run_wall_time_seconds": float(run_completed_at - timing_origin),
        "cumulative_evaluations": int(total_evaluations),
        "cumulative_search_cost_usd": float(total_cost),
        "budget_fraction": float(total_cost) / float(bruteforce_search_cost_usd),
    })
    if independent_eta:
        stage_timing_events[-1].update({
            "lambda_scope": "visited_direction",
            "direction_eta_multipliers": list(direction_eta_multipliers),
            "direction_eta_stages": list(direction_eta_stages),
        })
    selected_truth = (
        truth_vectors[archive_arms]
        if archive_arms
        else np.empty((0, 2), dtype=np.float64)
    )
    quality = front_quality_metrics(
        selected_truth,
        truth_front,
        metric_reference,
        ground_truth_hv,
    )
    selected_hv = quality.hypervolume
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

    cache_stats = _boundary_cache_stats_delta(
        cache_stats_before,
        _boundary_cache_stats_snapshot(cache),
        fallback_builds=len(cache) - cache_size_before,
    )
    cache_dir = getattr(cache, "cache_dir", None)
    cache_disk_read = bool(getattr(cache, "disk_read", False))
    cache_disk_write = bool(getattr(cache, "disk_write", False))
    if cache_disk_read and cache_disk_write:
        boundary_cache_mode = "disk_read_write_and_memory"
    elif cache_disk_read:
        boundary_cache_mode = "disk_read_and_memory"
    elif cache_disk_write:
        boundary_cache_mode = "disk_write_and_memory"
    else:
        boundary_cache_mode = "memory"
    online_cache_stats_method = getattr(
        online_value_cache,
        "stats_snapshot",
        None,
    )
    online_cache_stats = (
        online_cache_stats_method()
        if callable(online_cache_stats_method)
        else None
    )

    params: Dict[str, Any] = {
        "batch_size": batch_size,
        "warm_start_batch_size": resolved_warm_start_batch_size,
        "warm_start_question_order": warm_start_question_order,
        "initialization": (
            "fixed_general_prior_without_warm_start"
            if cold_start
            else "empirical_bayes_uniform_shared_question_warm_start"
            if warm_start_question_order == "shared"
            else "empirical_bayes_uniform_independent_question_warm_start"
        ),
        "warm_start_question_ids": list(schedule.warm_start_question_ids),
        "warm_start_question_ids_by_arm": [
            list(schedule.warm_start_question_ids_by_arm[i])
            for i in range(n_arms)
        ],
        "cost_model": cost_model,
        "metric_space": "offline_common_affine_mean_usd",
        "metric_cost_reference_usd": metric_cost_reference_usd,
        "metric_reference_point": list(metric_reference),
        "metric_reference_source": "offline_full_mean_cost_max_times_1.05",
        "posterior_state_space": "normalized_accuracy_mean_cost_usd" if cost_model == "raw_mean" else "normalized_desirability",
        "posterior_summary_space": "reward_desirability",
        "cost_reference_source": "explicit" if cost_reference_usd is not None else (
            "max_warm_arm_mean_usd" if cost_model == "raw_mean" else "median_warm_arm_mean_usd"
        ),
        "reward_prior_mean": calibration.reward_posterior(GaussianVectorPosterior(calibration.prior_mean, calibration.prior_var)).mean.tolist(),
        "reward_prior_variance": calibration.reward_posterior(GaussianVectorPosterior(calibration.prior_mean, calibration.prior_var)).var.tolist(),
        "reward_obs_noise_variance": noise_var.tolist(),
        "raw_cost_question_noise_variance_usd2": float(calibration.warm_obs_noise_var[1] * batch_size) if cost_model == "raw_mean" else None,
        "raw_cost_prior_variance_estimate_usd2": calibration.raw_cost_prior_variance_estimate_usd2,
        "raw_cost_observation_variance_estimate_usd2": calibration.raw_cost_observation_variance_estimate_usd2,
        "raw_cost_variance_floor_usd2": calibration.raw_cost_variance_floor_usd2,
        "objective_grid_lower": objective_grid_lower.tolist(),
        "objective_grid_upper": objective_grid_upper.tolist(),
        "objective_grid_expansions": objective_grid_expansions,
        "objective_grid_evidence": "observed_posterior_with_six_std_padding" if cost_model == "raw_mean" else "bounded_normalized_objectives",
        "recommendation_space": (
            "finite_test_mean_accuracy_mean_usd" if recommendation_rule == "finite_mean" else
            "finite_test_mean_accuracy_lcb_mean_usd_ucb" if finite_recommendation else
            "observed_raw_accuracy_mean_cost_usd"
        ),
        "recommendation_rule": recommendation_rule,
        "recommendation_beta": recommendation_beta,
        "recommendation_min_samples": recommendation_min_samples,
        "recommendation_completed_std_penalty": 0.0,
        "recommendation_eligibility": (
            "observed_question_count_at_least_min_samples" if recommendation_min_samples
            else "all_arms" if uncertain_recommendation else "completed_only"
        ),
        "recommendation_direction_score": None,
        "recommendation_endpoint_score": None,
        "recommendation_endpoint_tie_break": None,
        "recommendation_filter": (
            "eligible_arms_finite_test_mean_raw_pareto" if recommendation_rule == "finite_mean" else
            "eligible_arms_finite_test_lcb_raw_pareto" if finite_recommendation and recommendation_min_samples else
            "all_arms_finite_test_lcb_raw_pareto" if finite_recommendation else
            "all_completed_empirical_raw_pareto"
        ),
        "recommendation_evidence_scope": (
            "selected_finite_test_mean_frontier" if recommendation_rule == "finite_mean" else
            "selected_finite_test_lcb_frontier" if finite_recommendation else
            "selected_completed_empirical_frontier"
        ),
        "direction_winners_role": "acquisition_diagnostic_only",
        **({
            "finite_target_question_counts": list(finite_recommendation_cache.n_total),
            "finite_target_question_noise_variance": finite_recommendation_cache.question_noise_var.tolist(),
            "finite_target_noise_space": "normalized_accuracy_mean_usd",
            "finite_target_mean_formula": "(observed_sum + (N-n)*posterior_mean)/N",
            "finite_target_variance_formula": "((N-n)^2*posterior_variance + (N-n)*question_noise_variance)/N^2",
            "finite_target_assumptions": "conditionally_independent_gaussian_questions_fixed_warm_noise",
            "recommendation_desirability_space": (
                "normalized_accuracy_mean_affine_cost_mean" if recommendation_rule == "finite_mean"
                else "normalized_accuracy_lcb_affine_cost_ucb"
            ),
            "recommendation_raw_space": (
                "accuracy_mean_mean_usd" if recommendation_rule == "finite_mean" else "accuracy_lcb_mean_usd_ucb"
            ),
            "recommendation_confidence_guarantee": (
                "none_minimum_sample_gate_is_not_a_confidence_guarantee" if recommendation_rule == "finite_mean"
                else "componentwise_model_based_heuristic_not_joint_or_anytime_coverage"
            ),
        } if finite_recommendation else {}),
        "stopping_eligibility": "completed_only",
        "stopping_value": "required_completion_gittins",
        "recommendation_changes_acquisition": False,
        "recommendation_changes_only": bool(recommendation_changes_only),
        "defer_recommendation_diagnostics": bool(defer_recommendation_diagnostics),
        "recommendation_recording": recommendation_recording,
        "posterior_archive_space": "normalized_posterior_mean_desirability",
        "oracle_raw_winner_archive_is_diagnostic": True,
        "directions": [list(x) for x in resolved_directions],
        "direction_scheduler": direction_scheduler,
        "eta_decay_schedule": eta_decay_schedule,
        "lambda_scope": "visited_direction" if independent_eta else "global",
        "lambda_direction_index": active_eta_direction_index if independent_eta else None,
        "direction_eta_multipliers": list(direction_eta_multipliers),
        "direction_eta_stages": list(direction_eta_stages),
        "direction_eta_event_count": len(direction_eta_events),
        "direction_scheduler_groups": [list(group) for group in scheduler.groups],
        "endpoint_direction_policy": {
            "scope": "acquisition_and_stopping",
            "directions": [
                list(direction) for direction in resolved_directions
                if _direction_axis(direction) is not None
            ],
            "terminal_utility": "active_posterior_mean_minus_reference",
            "index": "scalar_gaussian_required_completion_gittins",
            "tie_break": "other_observed_objective_then_arm_index",
            "grid_size": base_boundary_grid.state_size,
            "cache": "memory_keyed_by_cost_variance_noise_horizon_resolution",
            "cache_stats": axis_cache.stats_snapshot(),
        },
        "prior_variance": calibration.prior_var.tolist(),
        "obs_noise_variance": calibration.warm_obs_noise_var.tolist(),
        "cost_reference_usd": calibration.cost_reference_usd,
        "reference_point": list(resolved_reference),
        "search_cost_scale_eta": search_cost_scale_eta,
        "anytime": anytime,
        "lambda_initial": float(lambda_initial) if anytime else 1.0,
        "lambda_decay": float(lambda_decay) if anytime else None,
        "lambda_final": current_lambda,
        "lambda_stage_count": lambda_stage + 1,
        "lambda_stop_count": len(lambda_stop_events),
        "lambda_cost_quantization": "quantize_eta_cost_then_multiply_lambda",
        "lambda_numerical_floor_policy": (
            "max_remaining_planning_penalty_at_stopping_precision"
            if anytime else None
        ),
        "lambda_numerical_penalty_threshold": lambda_numerical_threshold,
        "gittins_stop_action": (
            "decay_direction_eta" if independent_eta else "decay_lambda" if anytime else (
                "halt" if halt_on_gittins_stop else "force_continuation"
            )
        ),
        "boundary_z_padding_extra": boundary_z_padding_extra,
        "boundary_minimum_z_padding": 2.0 if anytime else 1.0,
        "boundary_grid_expansions": boundary_grid_expansions,
        "boundary_max_widening_retries": (
            _ANYTIME_BOUNDARY_MAX_WIDENING_RETRIES if anytime or cost_model == "raw_mean" else 0
        ),
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
        "record_trace": record_trace,
        "recommendation_checkpoint_interval": (
            recommendation_checkpoint_interval
        ),
        "recommendation_checkpoint_interval_requested": (
            requested_checkpoint_interval
        ),
        "recommendation_checkpoint_target": recommendation_checkpoint_target,
        "planned_adaptive_pulls": planned_adaptive_pulls,
        "horizon_bin_width": horizon_bin_width,
        "actual_horizons": actual_horizons.tolist(),
        "planning_horizons": planning_horizons.tolist(),
        "question_universe": question_universe,
        "question_order": question_order,
        "question_order_semantics": (
            "global_random_order_with_independent_arm_cursors"
            if cold_start and question_order == "shared" else
            "independent_random_per_arm_orders"
            if cold_start else
            "shared_warm_then_global_random_tail_restricted_to_each_arm_with_independent_cursors"
            if warm_start_question_order == "shared" and question_order == "shared" else
            "shared_warm_then_independent_random_per_arm_tails"
            if warm_start_question_order == "shared" else
            "independent_warm_then_global_random_tail_restricted_to_each_arm_with_independent_cursors"
            if question_order == "shared" else
            "independent_warm_then_independent_random_per_arm_tails"
        ),
        "question_order_rng_scheme": (
            "shared_global_rng" if cold_start and question_order == "shared"
            else "spawned_per_arm_full_orders" if cold_start
            else "shared_warm_rng_continuation"
            if warm_start_question_order == "shared" and question_order == "shared"
            else "spawned_per_arm_tails"
            if warm_start_question_order == "shared"
            else "spawned_per_arm_warm_then_shared_global_tail"
            if question_order == "shared"
            else "spawned_per_arm_full_orders"
        ),
        "common_question_count": len(common_questions),
        "benchmark_question_count": (
            len(evaluation_datapoints)
            if question_universe == "common"
            else None
        ),
        "available_cells_in_universe": n_available,
        "bruteforce_search_cost_usd": bruteforce_search_cost_usd,
        "planned_partial_tail_cells": planned_partial_tail_cells,
        "ragged_tail_cells_excluded": 0,
        "unobserved_cells_at_stop": int(
            sum(schedule.remaining(i) for i in range(n_arms))
        ),
        "boundary_grid_mode": boundary_grid_mode,
        "boundary_cache": {
            "mode": boundary_cache_mode,
            "directory": str(cache_dir) if cache_dir is not None else None,
            "disk_read": cache_disk_read,
            "disk_write": cache_disk_write,
            "stats": dict(cache_stats),
        },
        "boundary_table_build": {
            "mode": (
                "disabled_external_index_provider"
                if index_provider is not None
                else (
                    "direction_lazy_hybrid"
                    if optimized_boundary_cache
                    else "scalar_cache_protocol"
                )
            ),
            "requested_backend": boundary_build_backend,
            "backend_scope": (
                "not_applicable"
                if index_provider is not None
                else (
                    "cold_cache_misses"
                    if optimized_boundary_cache
                    else "external_cache_protocol"
                )
            ),
            "cache_backend_policy": (
                "not_applicable"
                if index_provider is not None
                else (
                    "backend_neutral"
                    if optimized_boundary_cache
                    else "external_cache_defined"
                )
            ),
            "jax_min_batch_size": boundary_jax_min_batch_size,
            "effective_jax_min_batch_size": (
                1
                if boundary_build_backend == "jax"
                else boundary_jax_min_batch_size
            ),
            "auto_jax_disabled_after_failure": auto_jax_disabled,
            "auto_jax_min_cell_stages": _AUTO_JAX_MIN_CELL_STAGES,
            "auto_jax_workload_skips": auto_jax_workload_skips,
            "prewarmed_directions": len(direction_boundary_tables),
            "stats": dict(boundary_prewarm_stats),
        },
        "online_index_cache": {
            "mode": "per_arm_generation_lazy_invalidation",
            "stats": online_cache_stats,
        },
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
        "acquisition": (
            "radial_gittins_boundary_index"
            if index_provider is None
            else "external_index_provider"
        ),
    }
    if extra_params is not None:
        overlapping = sorted(set(extra_params).intersection(params))
        if overlapping:
            raise ValueError(
                "extra_params must not override replay params: "
                f"{overlapping}"
            )
        params.update(dict(extra_params))
    result = RadialSimulationResult(
        selector=(
            "radial_gittins_anytime"
            if anytime and selector_name == "radial_gittins"
            else str(selector_name)
        ),
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
        hypervolume_regret=quality.hypervolume_regret,
        generational_distance=quality.generational_distance,
        inverted_generational_distance=quality.inverted_generational_distance,
        contains_true_accuracy_best=bool(
            truth_accuracy_best.intersection(archive_arm_set)
        ),
        model_results=model_results,
        trace=trace,
        observed_cells=tuple(sorted(observed_cells)),
        recommendation_events=recommendation_events,
        recommendation_initial_event=recommendation_initial_event,
        recommendation_final_event=recommendation_final_event,
        gittins_stop_evaluations=gittins_stop_evaluations,
        gittins_stop_cost_usd=gittins_stop_cost_usd,
        gittins_stop_budget_fraction=gittins_stop_budget_fraction,
        current_lambda=current_lambda,
        lambda_stage=lambda_stage,
        lambda_stop_events=lambda_stop_events,
        direction_eta_multipliers=tuple(direction_eta_multipliers),
        direction_eta_stages=tuple(direction_eta_stages),
        direction_eta_events=direction_eta_events,
        stage_timing_events=stage_timing_events,
        gittins_stop_triggered=gittins_stop_triggered,
        halted_by_gittins=halted_by_gittins,
        truth_vectors=truth_vectors,
        raw_truth_vectors=raw_truth_vectors,
        posterior_archive_arm_indices=tuple(posterior_archive_arms),
        posterior_archive_models=[models[i] for i in posterior_archive_arms],
        online_raw_archive_arm_indices=tuple(online_raw_archive_arms),
        online_raw_archive_models=[models[i] for i in online_raw_archive_arms],
        oracle_raw_winner_archive_arm_indices=tuple(oracle_raw_archive_arms),
        oracle_raw_winner_archive_models=[
            models[i] for i in oracle_raw_archive_arms
        ],
    )
    if not defer_recommendation_diagnostics:
        materialize_recommendation_diagnostics(result)
    if run_metadata is not None:
        run_metadata.update(
            {
                "stop_reason": stop_reason,
                "stopped_by_gittins": stopped_by_gittins,
                "anytime": anytime,
                "current_lambda": current_lambda,
                "lambda_stage": lambda_stage,
                "lambda_stop_count": len(lambda_stop_events),
                "eta_decay_schedule": eta_decay_schedule,
                "recommendation_rule": recommendation_rule,
                "recommendation_beta": recommendation_beta,
                "recommendation_min_samples": recommendation_min_samples,
                "stopping_eligibility": "completed_only",
                "lambda_scope": "visited_direction" if independent_eta else "global",
                "direction_eta_multipliers": list(direction_eta_multipliers),
                "direction_eta_stages": list(direction_eta_stages),
                "direction_eta_events": list(direction_eta_events),
                "stage_timing_events": list(stage_timing_events),
                "gittins_stop_triggered": gittins_stop_triggered,
                "halted_by_gittins": halted_by_gittins,
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
                "boundary_tables_built": cache_stats["builds"],
                "boundary_tables_loaded_from_disk": cache_stats["disk_hits"],
                "boundary_cache_memory_hits": cache_stats["memory_hits"],
                "boundary_cache_disk_misses": cache_stats["disk_misses"],
                "boundary_cache_corruptions": cache_stats["corruptions"],
                "boundary_cache_read_failures": cache_stats["read_failures"],
                "boundary_cache_write_failures": cache_stats["write_failures"],
                "boundary_tables_cached_total": len(cache),
                "policy_wall_time_seconds": policy_wall_time,
            }
        )
    return result


def _json_mean(values: Sequence[float]) -> Optional[float]:
    mean = float(np.mean(np.asarray(list(values), dtype=np.float64)))
    return mean if math.isfinite(mean) else None


def summarize_radial_multi_seed(
    results: Sequence[RadialSimulationResult],
) -> Dict[str, Any]:
    if not results:
        raise ValueError("at least one result is required")
    selectors = {result.selector for result in results}
    if len(selectors) != 1:
        raise ValueError(
            f"cannot summarize a mixture of selectors: {sorted(selectors)}"
        )
    return {
        "selector": results[0].selector,
        "n_seeds": len(results),
        "mean_hypervolume_regret": float(
            np.mean([result.hypervolume_regret for result in results])
        ),
        "mean_generational_distance": _json_mean(
            [result.generational_distance for result in results]
        ),
        "mean_inverted_generational_distance": _json_mean(
            [result.inverted_generational_distance for result in results]
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
    print(f"{result.selector} (seed={result.seed})")
    print(f"stop={result.stop_reason}, C_ref=${result.cost_reference_usd:.6g}")
    print(
        f"evaluations={result.total_evaluations}, cost=${result.total_cost:.6f}, "
        f"HV regret={result.hypervolume_regret:.6f}, "
        f"GD={result.generational_distance:.6f}, "
        f"IGD={result.inverted_generational_distance:.6f}"
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
    print(f"recommendation ({result.params['recommendation_space']}): {result.selected_models}")
    print(f"posterior-desirability archive: {result.posterior_archive_models}")
    print(
        "offline oracle raw winner archive: "
        f"{result.oracle_raw_winner_archive_models}"
    )


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _jsonable_result(result: RadialSimulationResult) -> Dict[str, Any]:
    return _jsonable_value(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pickle", help="Path to a cached lookup pickle")
    source.add_argument("--jsonl", help="Path to a brute-force JSONL file")
    source.add_argument(
        "--scope",
        help="Path to a SCOPE benchmark directory containing matrix CSVs",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--question-order", choices=QUESTION_ORDERS, default="shared",
        help="Share the seeded adaptive tail across arms (default), or use independent per-arm tails",
    )
    parser.add_argument(
        "--warm-start-question-order", choices=QUESTION_ORDERS, default="shared",
        help="Use the same warm questions for every arm (default), or draw each arm's warm questions independently",
    )
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
    parser.add_argument("--cost-model", choices=("reciprocal", "raw_mean"), default="reciprocal")
    parser.add_argument("--cost-reference-usd", type=float, default=None,
                        help="Frozen cost scale/reference; raw_mean defaults to the largest warm arm mean")
    parser.add_argument(
        "--extra-direction",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("ACCURACY", "COST_DESIRABILITY"),
        help=(
            "Append a simplex direction once; defaults are nine interior "
            "directions, plus (1, 0) with --anytime"
        ),
    )
    parser.add_argument(
        "--anytime",
        action="store_true",
        help=(
            "Use ten directions including (1, 0), decay lambda at each global "
            "Gittins stop, and continue within the budget"
        ),
    )
    parser.add_argument(
        "--direction-scheduler",
        choices=(
            "round_robin",
            "accuracy_last",
            "quality_then_deployment",
            "deployment_then_quality",
        ),
        default="round_robin",
        help=(
            "Use balanced round-robin, defer (1, 0) with accuracy_last, or "
            "drain the exact axes sequentially in quality/deployment order"
        ),
    )
    parser.add_argument(
        "--eta-decay-schedule", choices=("global_stop", "direction_stop"),
        default="global_stop",
        help="Decay the shared eta after global stopping, or each direction's eta after its local stop",
    )
    parser.add_argument(
        "--recommendation-rule", choices=RECOMMENDATION_RULES, default="completed_only",
        help="completed_only empirical frontier; finite_lcb/finite_mean full-test predictive Pareto (requires raw_mean); stopping unchanged",
    )
    parser.add_argument(
        "--recommendation-beta", type=float, default=1.0,
        help="Nonnegative posterior-standard-deviation multiplier for LCB recommendations (default: 1)",
    )
    parser.add_argument(
        "--recommendation-min-samples", type=int, default=0,
        help="Minimum actual observed questions per eligible arm for finite_lcb/finite_mean (default: 0)",
    )
    parser.add_argument(
        "--lambda-initial", type=float, default=1.0,
        help="Initial continuation-cost multiplier in anytime mode (default: 1.0)",
    )
    parser.add_argument(
        "--lambda-decay", type=float, default=0.5,
        help="Multiplier after each anytime stopping trigger (default: 0.5)",
    )
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Record recommendation checkpoints (always enabled with --anytime)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not retain per-direction-visit diagnostic trace events",
    )
    parser.add_argument(
        "--recommendation-changes-only", action="store_true",
        help="Record only recommendation membership changes, checking after every pull",
    )
    parser.add_argument(
        "--defer-recommendation-diagnostics", action="store_true",
        help="Save lightweight recommendation events; compute checkpoint diagnostics later on demand",
    )
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
    parser.add_argument(
        "--boundary-cache-dir",
        type=Path,
        default=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR,
        help=(
            "Persistent DP-boundary cache directory (default: %(default)s; "
            "override with AGENTOPT_RADIAL_GITTINS_CACHE_DIR)"
        ),
    )
    parser.add_argument(
        "--no-boundary-disk-cache",
        action="store_true",
        help="Use only the in-process boundary cache",
    )
    parser.add_argument(
        "--boundary-build-backend",
        choices=("auto", "jax", "scipy"),
        default="auto",
        help=(
            "Cold boundary-table builder: auto batches large miss groups with "
            "JAX and falls back to SciPy (default: auto)"
        ),
    )
    parser.add_argument(
        "--boundary-jax-min-batch-size",
        type=int,
        default=4,
        help="Minimum cold miss group routed to JAX in auto mode (default: 4)",
    )
    parser.add_argument(
        "--trajectory-checkpoint-interval",
        type=int,
        default=None,
        help=(
            "Record an ordinary trajectory point every N adaptive pulls; "
            "by default only event checkpoints are kept"
        ),
    )
    parser.add_argument(
        "--trajectory-target-checkpoints",
        type=int,
        default=None,
        help=(
            "Derive the interval to retain approximately N ordinary trajectory "
            "points; mandatory event checkpoints remain"
        ),
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    if args.pickle:
        path = _require_data_path(args.pickle)
        models, datapoints, table = load_pickle(path)
    elif args.jsonl:
        path = _require_data_path(args.jsonl)
        models, datapoints, table = load_jsonl(path)
    else:
        path = args.scope
        models, datapoints, table = load_scope(path)
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
    cache = RadialGittinsBoundaryCache(
        cache_dir=(
            None if args.no_boundary_disk_cache else args.boundary_cache_dir
        )
    )
    for offset in range(args.seeds):
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            cost_model=args.cost_model,
            cost_reference_usd=args.cost_reference_usd,
            batch_size=args.batch_size,
            directions=_cli_directions(
                anytime=args.anytime, extra_directions=args.extra_direction
            ),
            direction_scheduler=args.direction_scheduler,
            eta_decay_schedule=args.eta_decay_schedule,
            recommendation_rule=args.recommendation_rule,
            recommendation_beta=args.recommendation_beta,
            recommendation_min_samples=args.recommendation_min_samples,
            observation_budget_fraction=args.budget_fraction,
            max_search_cost_usd=args.max_search_cost,
            guaranteed_batch_cost_usd=args.guaranteed_batch_cost,
            search_cost_scale_eta=args.eta,
            anytime=args.anytime,
            lambda_initial=args.lambda_initial,
            lambda_decay=args.lambda_decay,
            record_trace=not args.no_trace,
            record_recommendation_trajectory=(
                args.record_trajectory or args.anytime or args.recommendation_changes_only
            ),
            recommendation_changes_only=args.recommendation_changes_only,
            defer_recommendation_diagnostics=args.defer_recommendation_diagnostics,
            effective_cost_bin_ratio=args.effective_cost_bin_ratio,
            effective_cost_bin_anchor=args.effective_cost_bin_anchor,
            horizon_bin_width=args.horizon_bin_width,
            seed=args.base_seed + offset,
            question_order=args.question_order,
            warm_start_question_order=args.warm_start_question_order,
            boundary_grid=grid,
            boundary_cache=cache,
            boundary_build_backend=args.boundary_build_backend,
            boundary_jax_min_batch_size=args.boundary_jax_min_batch_size,
            question_universe=("per_arm" if args.ragged_diagnostic else "common"),
            recommendation_checkpoint_interval=(
                args.trajectory_checkpoint_interval
            ),
            recommendation_checkpoint_target=(
                args.trajectory_target_checkpoints
            ),
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
