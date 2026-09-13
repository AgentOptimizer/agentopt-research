"""Core utilities for cost-aware radial-Gittins model selection.

This module contains the dependency-light statistical pieces used by the
offline replay selector and its radial boundary dynamic program:

* fixed reciprocal or affine deployment-cost normalizers;
* a two-objective Gaussian posterior;
* the uniform empirical-Bayes warm-start protocol;
* reproducible shared or independent question orders with per-arm cursors; and
* radial scalarization helpers used by the boundary-table policy.

The warm-start protocol is two phase.  All raw observations must be collected
before :func:`fit_empirical_bayes_warm_start` is called.  The returned cost
reference and common plug-in prior are then frozen.  Initializing the mutable
arm posteriors applies each arm's warm batch exactly once as a likelihood.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


VectorLike = Union[float, Sequence[float], np.ndarray]
COST_MODELS: Tuple[str, ...] = ("reciprocal", "raw_mean")
QUESTION_ORDERS: Tuple[str, ...] = ("shared", "independent")


DEFAULT_DIRECTIONS: Tuple[Tuple[float, float], ...] = (
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
)

# The anytime replay also searches the accuracy endpoint with its scalar DP.
# Keep the interior-only set above for fixed runs and two-dimensional tables.
DEFAULT_ANYTIME_DIRECTIONS: Tuple[Tuple[float, float], ...] = (
    *DEFAULT_DIRECTIONS,
    (1.0, 0.0),
)


def _two_vector(value: VectorLike, name: str, *, positive: bool) -> np.ndarray:
    """Return *value* as a validated two-element float vector."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 2)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a length-2 vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array.copy()


def _validated_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional arm-by-question matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one arm and one question")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_integer(value: object, name: str, *, minimum: int) -> int:
    """Return an integer without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if resolved < minimum:
        qualifier = "nonnegative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return int(resolved)


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def default_batch_observation_noise(batch_size: int) -> np.ndarray:
    """Return the conservative variance bound for a bounded batch mean.

    Each normalized objective lies in ``[0, 1]``, whose maximum possible
    per-question variance is ``1/4``.  Averaging an actual batch of size
    ``B`` therefore gives the default ``tau^2 = 1/(4B)`` for both objectives.
    """
    resolved_batch_size = _validated_integer(
        batch_size, "batch_size", minimum=1
    )
    return np.full(2, 1.0 / (4.0 * resolved_batch_size), dtype=np.float64)


@dataclass(frozen=True)
class ObjectiveNormalizer:
    """Normalize accuracy and deployment cost to reward coordinates.

    Accuracy is affinely mapped from ``score_bounds`` to ``[0, 1]``.  The
    default bounds leave benchmark scores that are already in ``[0, 1]``
    unchanged.  Deployment cost uses the fixed reciprocal map

    ``cost_reference_usd / (cost_reference_usd + cost_usd)`` by default.
    ``cost_model='raw_mean'`` instead uses ``1 - cost_usd / cost_reference_usd``.
    The affine reward is not clipped, so averaging rewards preserves raw mean
    cost ordering even when an observed cost exceeds the reference.

    The reference is a scale anchor, not a maximum.  It must stay fixed after
    calibration so every observation measures the same latent objective.
    """

    cost_reference_usd: float
    score_bounds: Tuple[float, float] = (0.0, 1.0)
    cost_model: str = "reciprocal"

    def __post_init__(self) -> None:
        reference = float(self.cost_reference_usd)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("cost_reference_usd must be finite and strictly positive")
        if self.cost_model not in COST_MODELS:
            raise ValueError(f"cost_model must be one of {COST_MODELS}")
        low, high = (float(x) for x in self.score_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("score_bounds must be finite with upper > lower")
        object.__setattr__(self, "cost_reference_usd", reference)
        object.__setattr__(self, "score_bounds", (low, high))

    def normalize_score(self, score: float) -> float:
        score = float(score)
        low, high = self.score_bounds
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if score < low or score > high:
            raise ValueError(
                f"score {score} lies outside configured bounds [{low}, {high}]"
            )
        return (score - low) / (high - low)

    def normalize_deployment_cost(self, cost_usd: float) -> float:
        cost = float(cost_usd)
        if not math.isfinite(cost):
            raise ValueError("deployment cost must be finite")
        if self.cost_model == "raw_mean":
            # Gaussian latent predictions may be negative. Actual observed
            # costs remain nonnegative-validated by normalize_batch.
            return 1.0 - cost / self.cost_reference_usd
        if cost < 0.0:
            raise ValueError("deployment cost must be finite and nonnegative")
        return self.cost_reference_usd / (self.cost_reference_usd + cost)

    def inverse_deployment_cost(self, desirability: float) -> Optional[float]:
        """Invert a finite reward without clipping Gaussian predictions.

        Reciprocal rewards at or below zero have no finite cost inverse.
        Affine rewards, including negative values or values above one, retain
        their algebraic raw-cost inverse.
        """
        reward = float(desirability)
        if not math.isfinite(reward):
            raise ValueError("desirability must be finite")
        if self.cost_model == "raw_mean":
            return self.cost_reference_usd * (1.0 - reward)
        if reward <= 0.0:
            return None
        cost = self.cost_reference_usd * (1.0 - reward) / reward
        return cost if math.isfinite(cost) else None

    def normalize_batch(
        self,
        scores: Sequence[float],
        costs_usd: Sequence[float],
    ) -> np.ndarray:
        """Return per-question normalized observations with shape ``(B, 2)``."""
        scores_array = np.asarray(scores, dtype=np.float64)
        costs_array = np.asarray(costs_usd, dtype=np.float64)
        if scores_array.ndim != 1 or costs_array.ndim != 1:
            raise ValueError("scores and costs_usd must be one-dimensional")
        if scores_array.shape != costs_array.shape:
            raise ValueError("scores and costs_usd must have the same shape")
        if scores_array.size == 0:
            raise ValueError("a normalized batch must not be empty")
        if not np.all(np.isfinite(scores_array)):
            raise ValueError("scores must contain only finite values")
        if not np.all(np.isfinite(costs_array)) or np.any(costs_array < 0.0):
            raise ValueError("costs_usd must be finite and nonnegative")

        low, high = self.score_bounds
        if np.any(scores_array < low) or np.any(scores_array > high):
            raise ValueError(
                f"scores must lie inside configured bounds [{low}, {high}]"
            )
        normalized_scores = (scores_array - low) / (high - low)
        normalized_costs = (
            1.0 - costs_array / self.cost_reference_usd
            if self.cost_model == "raw_mean" else
            self.cost_reference_usd / (self.cost_reference_usd + costs_array)
        )
        return np.column_stack((normalized_scores, normalized_costs))


@dataclass
class GaussianVectorPosterior:
    """Independent two-objective Gaussian posterior for one configuration."""

    mean: np.ndarray
    var: np.ndarray
    n_batches: int = 0
    n_questions: int = 0

    def __post_init__(self) -> None:
        self.mean = _two_vector(self.mean, "mean", positive=False)
        self.var = _two_vector(self.var, "var", positive=True)
        self.n_batches = _validated_integer(
            self.n_batches, "n_batches", minimum=0
        )
        self.n_questions = _validated_integer(
            self.n_questions, "n_questions", minimum=0
        )

    def copy(self) -> "GaussianVectorPosterior":
        return GaussianVectorPosterior(
            mean=self.mean.copy(),
            var=self.var.copy(),
            n_batches=self.n_batches,
            n_questions=self.n_questions,
        )

    def update(
        self,
        observation: VectorLike,
        obs_noise_var: Optional[VectorLike] = None,
        *,
        batch_size: int,
    ) -> None:
        """Apply one conjugate batch-mean update in place.

        When ``obs_noise_var`` is omitted, it is derived from the *actual*
        ``batch_size``.  This avoids giving a smaller final batch the precision
        of an earlier full batch by accident.
        """
        resolved_batch_size = _validated_integer(
            batch_size, "batch_size", minimum=1
        )
        y = _two_vector(observation, "observation", positive=False)
        tau_sq = (
            default_batch_observation_noise(resolved_batch_size)
            if obs_noise_var is None
            else _two_vector(obs_noise_var, "obs_noise_var", positive=True)
        )
        posterior_var = 1.0 / (1.0 / self.var + 1.0 / tau_sq)
        posterior_mean = posterior_var * (
            self.mean / self.var + y / tau_sq
        )
        self.mean = posterior_mean
        self.var = posterior_var
        self.n_batches += 1
        self.n_questions += resolved_batch_size

    def updated(
        self,
        observation: VectorLike,
        obs_noise_var: Optional[VectorLike] = None,
        *,
        batch_size: int,
    ) -> "GaussianVectorPosterior":
        result = self.copy()
        result.update(observation, obs_noise_var, batch_size=batch_size)
        return result


@dataclass(frozen=True)
class WarmStartCalibration:
    """Immutable calibration fitted from one uniform batch over every arm.

    The mutable arm posteriors are intentionally created separately by
    :func:`initialize_empirical_bayes_posteriors`.  This prevents online
    updates from changing the fitted common prior or cost reference.
    """

    arm_ids: Tuple[Hashable, ...]
    question_ids: Tuple[int, ...]
    normalizer: ObjectiveNormalizer
    prior_mean: np.ndarray
    prior_var: np.ndarray
    warm_obs_noise_var: np.ndarray
    batch_observations: np.ndarray
    raw_cost_means_usd: np.ndarray
    raw_scores: np.ndarray
    raw_costs_usd: np.ndarray
    raw_cost_prior_variance_estimate_usd2: Optional[float] = None
    raw_cost_observation_variance_estimate_usd2: Optional[float] = None
    raw_cost_variance_floor_usd2: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_ids", tuple(self.arm_ids))
        object.__setattr__(self, "question_ids", tuple(self.question_ids))
        for field_name in (
            "prior_mean",
            "prior_var",
            "warm_obs_noise_var",
            "batch_observations",
            "raw_cost_means_usd",
            "raw_scores",
            "raw_costs_usd",
        ):
            object.__setattr__(
                self,
                field_name,
                _readonly_copy(getattr(self, field_name)),
            )

    @property
    def batch_size(self) -> int:
        return len(self.question_ids)

    @property
    def cost_reference_usd(self) -> float:
        return self.normalizer.cost_reference_usd

    @property
    def cost_model(self) -> str:
        return self.normalizer.cost_model

    @property
    def reward_obs_noise_var(self) -> np.ndarray:
        """Frozen warm-batch noise in the reward coordinates used by the DP."""
        noise = self.warm_obs_noise_var.copy()
        if self.cost_model == "raw_mean":
            noise[1] /= self.cost_reference_usd ** 2
        return _readonly_copy(noise)

    def posterior_observations(
        self, scores: Sequence[float], costs_usd: Sequence[float],
    ) -> np.ndarray:
        """Return per-question likelihood data in authoritative posterior units.

        Accuracy is normalized in both modes. The second coordinate is raw
        USD cost for raw_mean, and reciprocal desirability for the legacy mode.
        """
        observations = self.normalizer.normalize_batch(scores, costs_usd)
        if self.cost_model == "raw_mean":
            observations[:, 1] = np.asarray(costs_usd, dtype=np.float64)
        return observations

    def reward_posterior(self, posterior: GaussianVectorPosterior) -> GaussianVectorPosterior:
        """Copy a posterior into reward coordinates, preserving sample counts."""
        result = posterior.copy()
        if self.cost_model == "raw_mean":
            result.mean[1] = 1.0 - posterior.mean[1] / self.cost_reference_usd
            result.var[1] = posterior.var[1] / self.cost_reference_usd ** 2
        return result

    @property
    def n_arms(self) -> int:
        return len(self.arm_ids)

    @property
    def total_question_evaluations(self) -> int:
        return self.n_arms * self.batch_size

    @property
    def warm_start_cost_usd(self) -> float:
        """Sum of the observed per-question workflow costs in calibration."""
        return float(self.raw_costs_usd.sum())

    def initialize_posteriors(self) -> Dict[Hashable, GaussianVectorPosterior]:
        """Create independent mutable posteriors with each warm batch used once."""
        return initialize_empirical_bayes_posteriors(self)


def initialize_empirical_bayes_posteriors(
    calibration: WarmStartCalibration,
) -> Dict[Hashable, GaussianVectorPosterior]:
    """Initialize arm posteriors from a frozen warm-start or fixed prior."""
    if not isinstance(calibration, WarmStartCalibration):
        raise TypeError("calibration must be a WarmStartCalibration")

    posteriors: Dict[Hashable, GaussianVectorPosterior] = {}
    for arm_id, observation in zip(calibration.arm_ids, calibration.batch_observations):
        posterior = GaussianVectorPosterior(
            calibration.prior_mean.copy(),
            calibration.prior_var.copy(),
        )
        if calibration.batch_size:
            posterior.update(
                observation,
                calibration.warm_obs_noise_var,
                batch_size=calibration.batch_size,
            )
        posteriors[arm_id] = posterior
    return posteriors


def build_fixed_prior_calibration(
    arm_ids: Iterable[Hashable],
    *,
    observation_batch_size: int,
    cost_reference_usd: float,
    prior_mean: VectorLike = (0.5, 0.5),
    prior_variance: VectorLike = 0.04,
    obs_noise_variance: Optional[VectorLike] = None,
    score_bounds: Tuple[float, float] = (0.0, 1.0),
) -> WarmStartCalibration:
    """Build a zero-observation reciprocal-cost calibration.

    This is the explicit cold-start alternative to empirical-Bayes warm-up.
    Both prior coordinates are normalized desirabilities in ``[0, 1]``.
    The cost coordinate is inverted through the fixed reciprocal reference to
    obtain one generic per-question cost estimate for acquisition planning.
    No benchmark response-matrix cell is read or counted as observed.
    """
    resolved_arm_ids = tuple(arm_ids)
    if not resolved_arm_ids or len(set(resolved_arm_ids)) != len(resolved_arm_ids):
        raise ValueError("arm_ids must be nonempty and unique")
    resolved_batch_size = _validated_integer(
        observation_batch_size, "observation_batch_size", minimum=1
    )
    mean = _two_vector(prior_mean, "prior_mean", positive=False)
    if np.any(mean < 0.0) or np.any(mean > 1.0):
        raise ValueError("fixed reciprocal prior_mean must lie in [0, 1]")
    if mean[1] <= 0.0:
        raise ValueError(
            "fixed reciprocal cost prior mean must be positive so it has a finite cost"
        )
    variance = _two_vector(prior_variance, "prior_variance", positive=True)
    noise = (
        default_batch_observation_noise(resolved_batch_size)
        if obs_noise_variance is None
        else _two_vector(
            obs_noise_variance, "obs_noise_variance", positive=True
        )
    )
    normalizer = ObjectiveNormalizer(
        cost_reference_usd=cost_reference_usd,
        score_bounds=score_bounds,
        cost_model="reciprocal",
    )
    raw_cost_mean = normalizer.inverse_deployment_cost(float(mean[1]))
    if raw_cost_mean is None:
        raise AssertionError("positive reciprocal desirability must have a finite inverse")
    n_arms = len(resolved_arm_ids)
    return WarmStartCalibration(
        arm_ids=resolved_arm_ids,
        question_ids=(),
        normalizer=normalizer,
        prior_mean=mean,
        prior_var=variance,
        warm_obs_noise_var=noise,
        batch_observations=np.repeat(mean[None, :], n_arms, axis=0),
        raw_cost_means_usd=np.full(n_arms, raw_cost_mean, dtype=np.float64),
        raw_scores=np.empty((n_arms, 0), dtype=np.float64),
        raw_costs_usd=np.empty((n_arms, 0), dtype=np.float64),
    )


def fit_empirical_bayes_warm_start(
    scores: Sequence[Sequence[float]],
    costs_usd: Sequence[Sequence[float]],
    *,
    arm_ids: Optional[Iterable[Hashable]] = None,
    question_ids: Optional[Iterable[int]] = None,
    prior_variance: VectorLike = 0.04,
    obs_noise_variance: Optional[VectorLike] = None,
    cost_reference_usd: Optional[float] = None,
    score_bounds: Tuple[float, float] = (0.0, 1.0),
    cost_model: str = "reciprocal",
) -> WarmStartCalibration:
    """Fit and freeze the agreed common plug-in warm-start prior.

    ``scores`` and ``costs_usd`` must be dense ``(n_arms, batch_size)``
    matrices collected under a uniform initial design.  Cost normalization is
    performed per question before taking each arm's batch mean.  Every arm
    starts from the same plug-in prior and its warm batch is then applied once.

    The legacy reciprocal model uses the median warm arm cost as its reference
    by default. Raw_mean instead uses the maximum warm arm cost, and fits its
    authoritative Gaussian to normalized accuracy and raw mean USD cost.

    In raw_mean mode, cost entries of generic ``prior_variance`` and
    ``obs_noise_variance`` are overridden: cost prior variance is the across-arm
    sample variance of raw warm means, and per-question noise is the average
    within-arm sample variance. Both use ddof=1 and a floor of ``1e-12 * R**2``;
    batch noise divides the latter by the warm batch size. One-arm or one-cell
    variance estimates use zero before flooring. Accuracy entries are retained.
    Every fitted value stays frozen after the warm start.
    """
    if cost_model not in COST_MODELS:
        raise ValueError(f"cost_model must be one of {COST_MODELS}")
    score_matrix = _validated_matrix(scores, "scores")
    cost_matrix = _validated_matrix(costs_usd, "costs_usd")
    if score_matrix.shape != cost_matrix.shape:
        raise ValueError("scores and costs_usd must have identical shapes")
    if np.any(cost_matrix < 0.0):
        raise ValueError("costs_usd must be nonnegative")

    n_arms, batch_size = score_matrix.shape
    if arm_ids is None:
        resolved_arm_ids: Tuple[Hashable, ...] = tuple(range(n_arms))
    else:
        resolved_arm_ids = tuple(arm_ids)
        if len(resolved_arm_ids) != n_arms:
            raise ValueError("arm_ids length must equal the number of matrix rows")
        if len(set(resolved_arm_ids)) != len(resolved_arm_ids):
            raise ValueError("arm_ids must be unique")

    if question_ids is None:
        resolved_question_ids = tuple(range(batch_size))
    else:
        resolved_question_ids = tuple(
            _validated_integer(x, "question_id", minimum=0)
            for x in question_ids
        )
        if len(resolved_question_ids) != batch_size:
            raise ValueError(
                "question_ids length must equal the warm-start batch size"
            )
        if len(set(resolved_question_ids)) != len(resolved_question_ids):
            raise ValueError("question_ids must be unique")

    raw_cost_means = cost_matrix.mean(axis=1)
    if cost_reference_usd is None:
        reference = float(np.max(raw_cost_means) if cost_model == "raw_mean" else np.median(raw_cost_means))
        if reference <= 0.0:
            anchor = "maximum" if cost_model == "raw_mean" else "median"
            raise ValueError(
                f"cannot infer a positive cost reference from a zero {anchor}; "
                "provide cost_reference_usd explicitly or disable the cost objective"
            )
    else:
        reference = float(cost_reference_usd)

    normalizer = ObjectiveNormalizer(
        cost_reference_usd=reference,
        score_bounds=score_bounds,
        cost_model=cost_model,
    )
    normalized = np.empty((n_arms, batch_size, 2), dtype=np.float64)
    for arm_index in range(n_arms):
        normalized[arm_index] = normalizer.normalize_batch(
            score_matrix[arm_index], cost_matrix[arm_index]
        )
    if cost_model == "raw_mean":
        normalized[:, :, 1] = cost_matrix

    batch_observations = normalized.mean(axis=1)
    # Equal arm weighting is deliberate: ragged/adaptive observation counts
    # must not determine the benchmark-level plug-in prior.
    prior_mean = batch_observations.mean(axis=0)
    prior_var = _two_vector(prior_variance, "prior_variance", positive=True)
    if obs_noise_variance is None:
        obs_noise_var = default_batch_observation_noise(batch_size)
    else:
        obs_noise_var = _two_vector(
            obs_noise_variance, "obs_noise_variance", positive=True
        )

    raw_prior_variance = raw_observation_variance = raw_variance_floor = None
    if cost_model == "raw_mean":
        raw_variance_floor = 1e-12 * reference ** 2
        raw_prior_variance = float(np.var(raw_cost_means, ddof=1)) if n_arms > 1 else 0.0
        raw_observation_variance = float(np.var(cost_matrix, axis=1, ddof=1).mean()) if batch_size > 1 else 0.0
        prior_var[1] = max(raw_prior_variance, raw_variance_floor)
        obs_noise_var[1] = max(raw_observation_variance, raw_variance_floor) / batch_size

    return WarmStartCalibration(
        arm_ids=resolved_arm_ids,
        question_ids=resolved_question_ids,
        normalizer=normalizer,
        prior_mean=prior_mean.copy(),
        prior_var=prior_var.copy(),
        warm_obs_noise_var=obs_noise_var.copy(),
        batch_observations=batch_observations.copy(),
        raw_cost_means_usd=raw_cost_means.copy(),
        raw_scores=score_matrix.copy(),
        raw_costs_usd=cost_matrix.copy(),
        raw_cost_prior_variance_estimate_usd2=raw_prior_variance,
        raw_cost_observation_variance_estimate_usd2=raw_observation_variance,
        raw_cost_variance_floor_usd2=raw_variance_floor,
    )


@dataclass
class PerArmQuestionSchedule:
    """Seeded per-arm question orders with configurable warm-start pairing.

    The backward-compatible default gives every arm the same first
    ``warm_start_batch_size`` question IDs.  Setting
    ``warm_start_question_order='independent'`` instead draws each arm's warm
    prefix from its own seeded permutation. ``question_order='shared'`` uses
    one random tail, restricted to each arm's available questions;
    ``'independent'`` uses per-arm tails. Orders may be ragged when cached
    cells are missing.
    Calling :meth:`next_batch` advances an arm cursor immediately, so attempted cells
    are not selected again even if their physical evaluation fails.
    """

    orders: Dict[Hashable, Tuple[int, ...]]
    positions: Dict[Hashable, int] = field(default_factory=dict)
    warm_start_question_ids: Tuple[int, ...] = ()
    warm_start_question_ids_by_arm: Dict[Hashable, Tuple[int, ...]] = field(
        default_factory=dict
    )
    question_order: str = "independent"
    warm_start_question_order: str = "shared"

    def __post_init__(self) -> None:
        if self.question_order not in QUESTION_ORDERS:
            raise ValueError(f"question_order must be one of {QUESTION_ORDERS}")
        if self.warm_start_question_order not in QUESTION_ORDERS:
            raise ValueError(
                "warm_start_question_order must be one of "
                f"{QUESTION_ORDERS}"
            )
        if not self.orders:
            raise ValueError("question schedule requires at least one arm")
        normalized_orders: Dict[Hashable, Tuple[int, ...]] = {}
        for arm_id, order in self.orders.items():
            normalized_orders[arm_id] = tuple(
                _validated_integer(x, "question_id", minimum=0) for x in order
            )
        self.orders = normalized_orders

        for arm_id, order in self.orders.items():
            if not order:
                raise ValueError(f"question order for arm {arm_id!r} must not be empty")
            if len(set(order)) != len(order):
                raise ValueError(f"question order for arm {arm_id!r} has duplicates")

        self.warm_start_question_ids = tuple(
            _validated_integer(x, "warm_start_question_id", minimum=0)
            for x in self.warm_start_question_ids
        )
        if self.warm_start_question_ids_by_arm:
            if set(self.warm_start_question_ids_by_arm) != set(self.orders):
                raise ValueError(
                    "warm_start_question_ids_by_arm and orders must contain "
                    "the same arm IDs"
                )
            warm_by_arm = {
                arm_id: tuple(
                    _validated_integer(
                        x, "warm_start_question_id", minimum=0
                    )
                    for x in question_ids
                )
                for arm_id, question_ids in self.warm_start_question_ids_by_arm.items()
            }
        else:
            warm_by_arm = {
                arm_id: self.warm_start_question_ids for arm_id in self.orders
            }
        warm_sizes = {len(question_ids) for question_ids in warm_by_arm.values()}
        if len(warm_sizes) != 1:
            raise ValueError("every arm must have the same warm-start batch size")
        if self.warm_start_question_order == "shared":
            unique_warm_batches = set(warm_by_arm.values())
            if len(unique_warm_batches) != 1:
                raise ValueError(
                    "shared warm start requires the same question IDs for every arm"
                )
            shared_warm = next(iter(unique_warm_batches))
            if self.warm_start_question_ids and self.warm_start_question_ids != shared_warm:
                raise ValueError(
                    "warm_start_question_ids disagrees with the per-arm warm batches"
                )
            self.warm_start_question_ids = shared_warm
        elif self.warm_start_question_ids:
            raise ValueError(
                "independent warm start stores question IDs per arm, not in "
                "warm_start_question_ids"
            )
        self.warm_start_question_ids_by_arm = warm_by_arm
        for arm_id, order in self.orders.items():
            warm_ids = warm_by_arm[arm_id]
            if len(warm_ids) > len(order):
                raise ValueError(
                    f"warm-start batch exceeds the order for arm {arm_id!r}"
                )
            if order[: len(warm_ids)] != warm_ids:
                raise ValueError(
                    f"arm {arm_id!r} does not start with its warm batch"
                )

        if not self.positions:
            self.positions = {arm_id: 0 for arm_id in self.orders}
        elif set(self.positions) != set(self.orders):
            raise ValueError("positions and orders must contain the same arm IDs")
        else:
            self.positions = {
                arm_id: _validated_integer(position, "position", minimum=0)
                for arm_id, position in self.positions.items()
            }
            for arm_id, position in self.positions.items():
                if position > len(self.orders[arm_id]):
                    raise ValueError(
                        f"question cursor for arm {arm_id!r} lies beyond its order"
                    )

    @property
    def warm_start_batch_size(self) -> int:
        return len(next(iter(self.warm_start_question_ids_by_arm.values())))

    @classmethod
    def create(
        cls,
        arm_ids: Iterable[Hashable],
        *,
        n_questions: int,
        warm_start_batch_size: int,
        seed: int = 0,
        question_order: str = "independent",
        warm_start_question_order: str = "shared",
    ) -> "PerArmQuestionSchedule":
        if question_order not in QUESTION_ORDERS:
            raise ValueError(f"question_order must be one of {QUESTION_ORDERS}")
        if warm_start_question_order not in QUESTION_ORDERS:
            raise ValueError(
                "warm_start_question_order must be one of "
                f"{QUESTION_ORDERS}"
            )
        resolved_arm_ids = tuple(arm_ids)
        if not resolved_arm_ids or len(set(resolved_arm_ids)) != len(resolved_arm_ids):
            raise ValueError("arm_ids must be nonempty and unique")
        resolved_n_questions = _validated_integer(
            n_questions, "n_questions", minimum=1
        )
        resolved_warm_size = _validated_integer(
            warm_start_batch_size, "warm_start_batch_size", minimum=0
        )
        if resolved_warm_size > resolved_n_questions:
            raise ValueError(
                "warm_start_batch_size must lie in [1, n_questions]"
            )
        resolved_seed = _validated_integer(seed, "seed", minimum=0)

        seed_sequence = np.random.SeedSequence(resolved_seed)
        children = seed_sequence.spawn(len(resolved_arm_ids) + 1)
        orders: Dict[Hashable, Tuple[int, ...]] = {}
        warm_by_arm: Dict[Hashable, Tuple[int, ...]] = {}
        shared_rng = np.random.default_rng(children[0])
        if warm_start_question_order == "shared":
            shared_order = shared_rng.permutation(resolved_n_questions).tolist()
            warm_ids = tuple(int(x) for x in shared_order[:resolved_warm_size])
            remaining = np.asarray(
                shared_order[resolved_warm_size:], dtype=np.int64
            )
            warm_by_arm = {arm_id: warm_ids for arm_id in resolved_arm_ids}
            if question_order == "shared":
                # Continue the warm RNG only after the unchanged warm draw.
                # Sorting matches create_from_available for a complete universe.
                tail = tuple(
                    int(x) for x in shared_rng.permutation(np.sort(remaining))
                )
                common_order = warm_ids + tail
                orders = {arm_id: common_order for arm_id in resolved_arm_ids}
            else:
                for arm_id, child in zip(resolved_arm_ids, children[1:]):
                    arm_rng = np.random.default_rng(child)
                    tail = arm_rng.permutation(remaining).tolist()
                    orders[arm_id] = warm_ids + tuple(int(x) for x in tail)
        else:
            shared_tail = tuple(
                int(x) for x in shared_rng.permutation(resolved_n_questions)
            )
            for arm_id, child in zip(resolved_arm_ids, children[1:]):
                arm_rng = np.random.default_rng(child)
                arm_order = tuple(
                    int(x) for x in arm_rng.permutation(resolved_n_questions)
                )
                warm_ids = arm_order[:resolved_warm_size]
                warm_by_arm[arm_id] = warm_ids
                if question_order == "independent":
                    orders[arm_id] = arm_order
                else:
                    warm_set = set(warm_ids)
                    orders[arm_id] = warm_ids + tuple(
                        q for q in shared_tail if q not in warm_set
                    )
        return cls(
            orders=orders,
            warm_start_question_ids=(
                next(iter(warm_by_arm.values()))
                if warm_start_question_order == "shared"
                else ()
            ),
            warm_start_question_ids_by_arm=warm_by_arm,
            question_order=question_order,
            warm_start_question_order=warm_start_question_order,
        )

    @classmethod
    def create_from_available(
        cls,
        available_question_ids_by_arm: Mapping[Hashable, Iterable[int]],
        *,
        warm_start_batch_size: int,
        seed: int = 0,
        question_order: str = "independent",
        warm_start_question_order: str = "shared",
    ) -> "PerArmQuestionSchedule":
        """Create configurable warm prefixes followed by per-arm tails.

        In shared-warm mode, warm questions are sampled from the intersection
        of all arms' available IDs. In independent-warm mode, every arm draws
        from its own available IDs. Every other available ID remains eligible
        only for the arm that actually contains it; missing response-matrix
        cells are never represented as free or zero-valued observations.

        Shared mode permutes the union of remaining available IDs once and
        filters that order per arm. Common IDs retain the same relative order,
        while each cursor advances independently. No response values are used.
        """
        if question_order not in QUESTION_ORDERS:
            raise ValueError(f"question_order must be one of {QUESTION_ORDERS}")
        if warm_start_question_order not in QUESTION_ORDERS:
            raise ValueError(
                "warm_start_question_order must be one of "
                f"{QUESTION_ORDERS}"
            )
        if not available_question_ids_by_arm:
            raise ValueError("available questions require at least one arm")
        resolved_warm_size = _validated_integer(
            warm_start_batch_size, "warm_start_batch_size", minimum=0
        )
        resolved_seed = _validated_integer(seed, "seed", minimum=0)

        available: Dict[Hashable, Tuple[int, ...]] = {}
        for arm_id, question_ids in available_question_ids_by_arm.items():
            ids = tuple(
                _validated_integer(x, "question_id", minimum=0)
                for x in question_ids
            )
            if not ids:
                raise ValueError(f"arm {arm_id!r} has no available questions")
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"available questions for arm {arm_id!r} contain duplicates"
                )
            available[arm_id] = ids

        arm_ids = tuple(available)
        seed_sequence = np.random.SeedSequence(resolved_seed)
        children = seed_sequence.spawn(len(arm_ids) + 1)
        shared_rng = np.random.default_rng(children[0])
        orders: Dict[Hashable, Tuple[int, ...]] = {}
        warm_by_arm: Dict[Hashable, Tuple[int, ...]] = {}
        if warm_start_question_order == "shared":
            shared = set(next(iter(available.values())))
            for ids in available.values():
                shared.intersection_update(ids)
            if len(shared) < resolved_warm_size:
                raise ValueError(
                    "fewer shared questions than the requested warm-start batch: "
                    f"need {resolved_warm_size}, found {len(shared)}"
                )
            shared_candidates = np.asarray(sorted(shared), dtype=np.int64)
            warm_ids = tuple(
                int(x)
                for x in shared_rng.permutation(shared_candidates)[
                    :resolved_warm_size
                ]
            )
            warm_set = set(warm_ids)
            warm_by_arm = {arm_id: warm_ids for arm_id in arm_ids}
            if question_order == "shared":
                available_sets = {
                    arm_id: set(ids) for arm_id, ids in available.items()
                }
                tail_candidates = np.asarray(
                    sorted(set().union(*available_sets.values()) - warm_set),
                    dtype=np.int64,
                )
                shared_tail = tuple(
                    int(x) for x in shared_rng.permutation(tail_candidates)
                )
                for arm_id, available_set in available_sets.items():
                    orders[arm_id] = warm_ids + tuple(
                        q for q in shared_tail if q in available_set
                    )
            else:
                for arm_id, child in zip(arm_ids, children[1:]):
                    tail_candidates = np.asarray(
                        sorted(set(available[arm_id]) - warm_set),
                        dtype=np.int64,
                    )
                    arm_rng = np.random.default_rng(child)
                    tail = tuple(
                        int(x) for x in arm_rng.permutation(tail_candidates)
                    )
                    orders[arm_id] = warm_ids + tail
        else:
            for arm_id, ids in available.items():
                if len(ids) < resolved_warm_size:
                    raise ValueError(
                        f"arm {arm_id!r} has fewer questions than the requested "
                        f"warm-start batch: need {resolved_warm_size}, found {len(ids)}"
                    )
            available_sets = {arm_id: set(ids) for arm_id, ids in available.items()}
            shared_candidates = np.asarray(
                sorted(set().union(*available_sets.values())), dtype=np.int64
            )
            shared_tail = tuple(
                int(x) for x in shared_rng.permutation(shared_candidates)
            )
            for arm_id, child in zip(arm_ids, children[1:]):
                arm_rng = np.random.default_rng(child)
                arm_order = tuple(
                    int(x)
                    for x in arm_rng.permutation(
                        np.asarray(sorted(available[arm_id]), dtype=np.int64)
                    )
                )
                warm_ids = arm_order[:resolved_warm_size]
                warm_by_arm[arm_id] = warm_ids
                if question_order == "independent":
                    orders[arm_id] = arm_order
                else:
                    warm_set = set(warm_ids)
                    orders[arm_id] = warm_ids + tuple(
                        q
                        for q in shared_tail
                        if q in available_sets[arm_id] and q not in warm_set
                    )
        return cls(
            orders=orders,
            warm_start_question_ids=(
                next(iter(warm_by_arm.values()))
                if warm_start_question_order == "shared"
                else ()
            ),
            warm_start_question_ids_by_arm=warm_by_arm,
            question_order=question_order,
            warm_start_question_order=warm_start_question_order,
        )

    def next_batch(self, arm_id: Hashable, batch_size: int) -> Tuple[int, ...]:
        if arm_id not in self.orders:
            raise KeyError(f"unknown arm_id {arm_id!r}")
        resolved_batch_size = _validated_integer(
            batch_size, "batch_size", minimum=1
        )
        start = self.positions[arm_id]
        end = min(start + resolved_batch_size, len(self.orders[arm_id]))
        result = self.orders[arm_id][start:end]
        self.positions[arm_id] = end
        return result

    def take_uniform_warm_start(self) -> Dict[Hashable, Tuple[int, ...]]:
        """Consume and return one equal-sized warm-start batch per arm.

        A live or replay selector should call this once, then evaluate the
        returned cells before fitting :class:`WarmStartCalibration`.  Cursors
        advance immediately, so failed physical attempts cannot cause those
        cells to be selected again silently.
        """
        batch_size = self.warm_start_batch_size
        if batch_size == 0:
            return {arm_id: () for arm_id in self.orders}
        if any(position != 0 for position in self.positions.values()):
            raise RuntimeError(
                "uniform warm start must be consumed before any arm cursor advances"
            )
        return {
            arm_id: self.next_batch(arm_id, batch_size) for arm_id in self.orders
        }

    def attempted_question_ids(self, arm_id: Hashable) -> Tuple[int, ...]:
        if arm_id not in self.orders:
            raise KeyError(f"unknown arm_id {arm_id!r}")
        return self.orders[arm_id][: self.positions[arm_id]]

    def remaining(self, arm_id: Hashable) -> int:
        if arm_id not in self.orders:
            raise KeyError(f"unknown arm_id {arm_id!r}")
        return len(self.orders[arm_id]) - self.positions[arm_id]


def direction_scale(direction: VectorLike) -> float:
    direction_array = _two_vector(direction, "direction", positive=True)
    return float(np.max(direction_array))


def radial_utility(
    objective: VectorLike,
    direction: VectorLike,
    reference: VectorLike = (0.0, 0.0),
) -> float:
    """Return direction-normalized two-objective radial utility."""
    objective_array = _two_vector(objective, "objective", positive=False)
    direction_array = _two_vector(direction, "direction", positive=True)
    reference_array = _two_vector(reference, "reference", positive=False)
    if not math.isclose(float(direction_array.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("direction components must sum to one")
    scaled = np.max(direction_array) * (
        objective_array - reference_array
    ) / direction_array
    return float(np.min(scaled))


def expected_min_of_two_normals(
    m1: float,
    v1: float,
    m2: float,
    v2: float,
    cov12: float = 0.0,
    *,
    variance_tolerance: float = 1e-12,
) -> float:
    """Closed-form expectation of ``min(X1, X2)`` for two normals."""
    try:
        m1, v1, m2, v2, cov12 = (
            float(m1),
            float(v1),
            float(m2),
            float(v2),
            float(cov12),
        )
        variance_tolerance = float(variance_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("normal moments and tolerance must be real numbers") from exc
    if not all(math.isfinite(x) for x in (m1, v1, m2, v2, cov12)):
        raise ValueError("normal moments must be finite")
    if v1 < 0.0 or v2 < 0.0:
        raise ValueError("normal variances must be nonnegative")
    if not math.isfinite(variance_tolerance) or variance_tolerance < 0.0:
        raise ValueError("variance_tolerance must be finite and nonnegative")

    maximum_covariance = math.sqrt(v1) * math.sqrt(v2)
    if maximum_covariance == 0.0:
        invalid_covariance = cov12 != 0.0
    else:
        invalid_covariance = abs(cov12) > maximum_covariance * (
            1.0 + variance_tolerance
        )
    if invalid_covariance:
        raise ValueError("covariance does not define a positive-semidefinite matrix")

    difference_variance = v1 + v2 - 2.0 * cov12
    variance_scale = max(
        v1 + v2 + 2.0 * abs(cov12),
        np.finfo(np.float64).tiny,
    )
    numerical_tolerance = variance_tolerance * variance_scale
    if difference_variance < -numerical_tolerance:
        raise ValueError("covariance implies a negative variance for X1 - X2")
    if difference_variance <= numerical_tolerance:
        return float(min(m1, m2))

    s = math.sqrt(difference_variance)
    d = (m1 - m2) / s
    phi = math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    phi_cdf = 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))
    return m1 * (1.0 - phi_cdf) + m2 * phi_cdf - s * phi


__all__ = [
    "COST_MODELS",
    "QUESTION_ORDERS",
    "DEFAULT_ANYTIME_DIRECTIONS",
    "DEFAULT_DIRECTIONS",
    "GaussianVectorPosterior",
    "ObjectiveNormalizer",
    "PerArmQuestionSchedule",
    "WarmStartCalibration",
    "default_batch_observation_noise",
    "direction_scale",
    "expected_min_of_two_normals",
    "fit_empirical_bayes_warm_start",
    "initialize_empirical_bayes_posteriors",
    "radial_utility",
]
