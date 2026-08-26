"""Optimistic radial index for two-objective model selection.

This is the upper-confidence-bound counterpart of the radial-Gittins boundary
policy in :mod:`agentopt.model_selection.radial_gittins_dp`.  Both policies
reduce one shared two-objective Gaussian posterior to a single
direction-specific number for an unfinished configuration, so they share the
round-robin scheduler, the warm-start calibration, the required-completion
stopping convention, and the archive metrics.  They differ only in that number:

* radial-Gittins solves a finite-horizon retirement dynamic program and scores
  an arm by ``u - b_n(delta)``;
* radial-UCB scores an arm by the radial utility of its componentwise upper
  confidence limit, minus the effective pull cost.

The direction-normalized radial utility

``rho(y) = min_i a_lambda (y_i - r_i) / lambda_i``, ``a_lambda = max_i lambda_i``

is nondecreasing in every objective because each ``lambda_i`` is positive.  The
largest radial utility attainable inside the per-objective confidence box
``[mu_i - beta sigma_i, mu_i + beta sigma_i]`` is therefore attained at the
upper corner, which gives the index

``rho(mu + beta sigma) - c_eff``.

Optimism is applied to the raw objectives rather than to the direction-scaled
coordinates; the two are equivalent because direction scaling is a positive
linear map, so ``beta`` keeps one interpretation across directions.

The effective pull cost ``c_eff`` plays the same role as in the boundary
dynamic program: it converts the dollars spent on one more batch into the
normalized utility units of the radial objective.  Because a completed arm is
scored by its terminal expected radial utility, subtracting ``c_eff`` is what
lets a direction eventually prefer an already-finished arm.

No dynamic program, grid, or boundary cache is required, which makes this the
cheap baseline for the radial-Gittins search-cost ablations.  It is a heuristic
acquisition rule: unlike the one-direction Gittins index it carries no
optimal-stopping interpretation, and the optimism does not account for the
number of remaining batches an arm still owes before it becomes selectable.

The index is evaluated once per (arm, direction visit), which is the hottest
loop in a replay, so :meth:`RadialUCBPolicy.index` is a closed-form scalar path
with no array allocation.  The array-returning helpers below are the readable
statement of the same arithmetic and share its core.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import numpy as np


VectorLike = Union[float, Sequence[float], np.ndarray]

DEFAULT_EXPLORATION_BETA: float = 2.0

POSTERIOR_SD_BONUS: str = "posterior_sd"
BATCH_COUNT_BONUS: str = "batch_count"
BONUS_MODES: Tuple[str, ...] = (POSTERIOR_SD_BONUS, BATCH_COUNT_BONUS)


def _two_scalars(
    value: VectorLike,
    name: str,
    *,
    positive: bool,
) -> Tuple[float, float]:
    """Return *value* as a validated pair of floats.

    Scalar checks rather than numpy reductions, and no intermediate array for
    the common tuple/length-2 inputs: on a two-element vector an ``np.all``
    guard costs more than the closed-form utility it protects.
    """
    if isinstance(value, np.ndarray):
        if value.shape == (2,):
            first = float(value[0])
            second = float(value[1])
        elif value.shape == ():
            first = second = float(value)
        else:
            raise ValueError(f"{name} must be a scalar or a length-2 vector")
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or a length-2 vector")
        first = float(value[0])
        second = float(value[1])
    else:
        try:
            first = second = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a scalar or a length-2 vector"
            ) from exc
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and (first <= 0.0 or second <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return first, second


def _nonnegative_float(value: object, name: str) -> float:
    try:
        resolved = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return resolved


@lru_cache(maxsize=256)
def _direction_scaling_factors(
    direction: Tuple[float, float],
) -> Tuple[float, float]:
    """Return ``max(direction) / direction`` for an already-positive pair.

    Every arm scored during one direction visit shares the same direction, and
    a replay revisits the same fixed direction list thousands of times, so both
    the simplex check and the two divisions are memoized rather than repeated
    per arm.  Failures are not cached, so an invalid direction keeps raising.
    """
    total = direction[0] + direction[1]
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("direction components must sum to one")
    largest = max(direction[0], direction[1])
    return largest / direction[0], largest / direction[1]


def _posterior_sd_bonus_scalars(
    var: VectorLike,
    beta: float,
) -> Tuple[float, float]:
    first, second = _two_scalars(var, "var", positive=True)
    return beta * math.sqrt(first), beta * math.sqrt(second)


def _batch_count_bonus_scalar(n_batches: object, beta: float) -> float:
    if isinstance(n_batches, (bool, np.bool_)):
        raise ValueError("n_batches must be an integer")
    try:
        resolved = operator.index(n_batches)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("n_batches must be an integer") from exc
    if resolved < 1:
        raise ValueError(
            "the batch-count bonus is undefined before an arm is observed; "
            "the mandatory uniform warm start guarantees n_batches >= 1"
        )
    return beta / math.sqrt(resolved)


def posterior_sd_bonus(
    var: VectorLike,
    *,
    exploration_beta: float = DEFAULT_EXPLORATION_BETA,
) -> np.ndarray:
    """Return ``beta * sigma`` for each objective.

    The conjugate posterior variance already shrinks like ``1 / n`` in the
    number of observed batches, so a fixed ``beta`` is the Bayes-UCB style
    analogue of a decaying confidence radius.
    """
    beta = _nonnegative_float(exploration_beta, "exploration_beta")
    return np.asarray(
        _posterior_sd_bonus_scalars(var, beta),
        dtype=np.float64,
    )


def batch_count_bonus(
    n_batches: int,
    *,
    exploration_beta: float = DEFAULT_EXPLORATION_BETA,
) -> np.ndarray:
    """Return the UCB1-style ``beta / sqrt(n_batches)`` bonus per objective.

    This matches the plain matrix-UCB bound ``mu + sqrt(a / count)`` used by the
    single-objective selector with ``a = beta ** 2``, except that the count is
    the number of question batches folded into the posterior rather than the
    number of individual questions.
    """
    beta = _nonnegative_float(exploration_beta, "exploration_beta")
    return np.full(2, _batch_count_bonus_scalar(n_batches, beta), dtype=np.float64)


@dataclass(frozen=True)
class RadialUCBPolicy:
    """Direction-specific optimistic index over one shared vector posterior.

    ``bonus_mode`` selects how the confidence radius is formed:

    ``"posterior_sd"``
        ``beta * sqrt(var)``, which uses the actual per-objective posterior
        uncertainty and therefore reacts to unequal accuracy/cost precision.

    ``"batch_count"``
        ``beta / sqrt(n_batches)``, the frequentist UCB1 radius shared by both
        objectives.  Included so the baseline can be run without depending on
        the Gaussian variance schedule being well calibrated.

    Hyperparameters are validated once here so that scoring an arm does not
    revalidate them.
    """

    exploration_beta: float = DEFAULT_EXPLORATION_BETA
    bonus_mode: str = POSTERIOR_SD_BONUS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exploration_beta",
            _nonnegative_float(self.exploration_beta, "exploration_beta"),
        )
        if self.bonus_mode not in BONUS_MODES:
            raise ValueError(f"bonus_mode must be one of {BONUS_MODES}")

    def _bonus_scalars(
        self,
        var: VectorLike,
        n_batches: Optional[int],
    ) -> Tuple[float, float]:
        if self.bonus_mode == POSTERIOR_SD_BONUS:
            return _posterior_sd_bonus_scalars(var, self.exploration_beta)
        if n_batches is None:
            raise ValueError(
                f"bonus_mode {self.bonus_mode!r} requires n_batches"
            )
        bonus = _batch_count_bonus_scalar(n_batches, self.exploration_beta)
        return bonus, bonus

    def objective_bonus(
        self,
        var: VectorLike,
        *,
        n_batches: Optional[int] = None,
    ) -> np.ndarray:
        """Return the per-objective confidence radius in raw objective units."""
        return np.asarray(
            self._bonus_scalars(var, n_batches),
            dtype=np.float64,
        )

    def optimistic_objective(
        self,
        mean: VectorLike,
        var: VectorLike,
        *,
        n_batches: Optional[int] = None,
    ) -> np.ndarray:
        """Return the componentwise upper confidence limit of the posterior."""
        first, second = _two_scalars(mean, "mean", positive=False)
        bonus_first, bonus_second = self._bonus_scalars(var, n_batches)
        return np.asarray(
            (first + bonus_first, second + bonus_second),
            dtype=np.float64,
        )

    def index(
        self,
        mean: VectorLike,
        var: VectorLike,
        direction: VectorLike,
        reference: VectorLike = (0.0, 0.0),
        *,
        n_batches: Optional[int] = None,
        effective_pull_cost: float = 0.0,
    ) -> float:
        """Return the cost-adjusted optimistic radial index for one arm."""
        cost = _nonnegative_float(effective_pull_cost, "effective_pull_cost")
        mean_first, mean_second = _two_scalars(mean, "mean", positive=False)
        bonus_first, bonus_second = self._bonus_scalars(var, n_batches)
        reference_first, reference_second = _two_scalars(
            reference,
            "reference",
            positive=False,
        )
        factor_first, factor_second = _direction_scaling_factors(
            _two_scalars(direction, "direction", positive=True)
        )
        return (
            min(
                factor_first * (mean_first + bonus_first - reference_first),
                factor_second * (mean_second + bonus_second - reference_second),
            )
            - cost
        )


@lru_cache(maxsize=64)
def _cached_policy(exploration_beta: float, bonus_mode: str) -> RadialUCBPolicy:
    return RadialUCBPolicy(
        exploration_beta=exploration_beta,
        bonus_mode=bonus_mode,
    )


def radial_ucb_index(
    mean: VectorLike,
    var: VectorLike,
    direction: VectorLike,
    reference: VectorLike = (0.0, 0.0),
    *,
    exploration_beta: float = DEFAULT_EXPLORATION_BETA,
    bonus_mode: str = POSTERIOR_SD_BONUS,
    n_batches: Optional[int] = None,
    effective_pull_cost: float = 0.0,
) -> float:
    """Convenience wrapper around :meth:`RadialUCBPolicy.index`."""
    policy = _cached_policy(
        _nonnegative_float(exploration_beta, "exploration_beta"),
        bonus_mode,
    )
    return policy.index(
        mean,
        var,
        direction,
        reference,
        n_batches=n_batches,
        effective_pull_cost=effective_pull_cost,
    )


__all__ = [
    "BATCH_COUNT_BONUS",
    "BONUS_MODES",
    "DEFAULT_EXPLORATION_BETA",
    "POSTERIOR_SD_BONUS",
    "RadialUCBPolicy",
    "batch_count_bonus",
    "posterior_sd_bonus",
    "radial_ucb_index",
]
