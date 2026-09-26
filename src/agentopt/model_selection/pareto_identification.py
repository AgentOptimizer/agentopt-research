"""Fixed-budget and LUCB-style Pareto set identification rules.

These are the acquisition / elimination kernels for the multi-objective
multi-armed bandit baselines of Kone, Kaufmann, and Richert:

* EGE-SH / EGE-SR — Empirical Gap Elimination with Sequential Halving or
  Successive Rejects (AISTATS 2024). Fixed budget, no stopping time.
* APE — Adaptive Pareto Exploration sampling (NeurIPS 2023). The paper's
  APE-k rule also has a fixed-confidence stop ``|OPT_ε1(t)| ≥ k``; that stop
  is intentionally not implemented here.

Both papers work in a maximization geometry. Callers must already map
deployment cost onto a larger-is-better coordinate before scoring.

The functions below are pure: they consume empirical means (and, for APE,
pull counts) and return which arms to keep or which arm to pull next.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


EGE_SH = "ege_sh"
EGE_SR = "ege_sr"
APE_K = "ape_k"
EGE_VARIANTS = (EGE_SH, EGE_SR)

# Official EGE-SH code adds a tiny offset to empirically sub-optimal gaps so
# Sequential Halving's keep-half sort breaks ties in favour of the empirical
# Pareto set, matching the paper's "ties first add arms in S_r" rule.
_EGE_SUBOPTIMAL_GAP_TIEBREAK = 1e-7


def pairwise_m(means: np.ndarray) -> np.ndarray:
    """Return ``m(i, j) = min_d (θ_j^d − θ_i^d)`` for every pair.

    ``m(i, j) > 0`` iff ``i`` is strictly dominated by ``j``.
    """
    values = _finite_means(means)
    return (values[None, :, :] - values[:, None, :]).min(axis=-1)


def pairwise_M(means: np.ndarray) -> np.ndarray:
    """Return ``M(i, j) = max_d (θ_i^d − θ_j^d)`` for every pair.

    ``M(i, j) < 0`` iff ``i`` is strictly dominated by ``j``.
    """
    values = _finite_means(means)
    return (values[:, None, :] - values[None, :, :]).max(axis=-1)


def empirical_pareto_mask(means: np.ndarray) -> np.ndarray:
    """Return a boolean mask of empirically nondominated rows.

    Arm ``i`` is empirically Pareto iff ``M(i, j) > 0`` for every ``j ≠ i``,
    which is the definition used by Kone et al. (AISTATS 2024, §3).
    """
    values = _finite_means(means)
    n_arms = values.shape[0]
    if n_arms == 1:
        return np.ones(1, dtype=bool)
    scores = pairwise_M(values)
    np.fill_diagonal(scores, np.inf)
    return np.all(scores > 0.0, axis=1)


def empirical_gaps(means: np.ndarray) -> np.ndarray:
    """Return the EGE empirical gap ``Δ̂_i`` for every arm.

    For an empirically sub-optimal arm this is ``max_{j ≠ i} m(i, j)``.
    For an empirically Pareto arm it is Lemma 1 of Kone et al. (AISTATS 2024):

    ``min_{j ≠ i} [ M(i, j) ∧ (M(j, i)_+ + (Δ̂*_j)_+) ]``.
    """
    values = _finite_means(means)
    n_arms = values.shape[0]
    if n_arms == 1:
        return np.array([math.inf], dtype=np.float64)

    m_ij = pairwise_m(values)
    M_ij = pairwise_M(values)
    np.fill_diagonal(m_ij, -np.inf)
    np.fill_diagonal(M_ij, np.inf)

    delta_star = m_ij.max(axis=1)
    pareto = empirical_pareto_mask(values)
    competitor = np.minimum(M_ij, np.maximum(M_ij.T, 0.0) + np.maximum(delta_star, 0.0)[None, :])
    np.fill_diagonal(competitor, np.inf)
    optimal_gap = competitor.min(axis=1)

    gaps = np.where(pareto, optimal_gap, delta_star)
    # Tie-break: slightly inflate sub-optimal gaps so equal-gap sorts keep S_r.
    gaps = np.where(pareto, gaps, gaps + _EGE_SUBOPTIMAL_GAP_TIEBREAK)
    return np.asarray(gaps, dtype=np.float64)


def ege_successive_rejects_schedule(n_arms: int, budget: int) -> Tuple[int, ...]:
    """Return per-round *new* pulls per active arm for EGE-SR.

    ``n_r`` follows Audibert & Bubeck (2010), so round ``r`` (1-indexed)
    collects ``n_r − n_{r-1}`` additional samples from each of the
    ``K − r + 1`` surviving arms. ``log(K) = 1/2 + sum_{i=2}^K 1/i``.
    """
    n_arms = _positive_integer(n_arms, "n_arms")
    budget = _nonnegative_integer(budget, "budget")
    if n_arms == 1:
        return ()
    log_k = 0.5 + sum(1.0 / i for i in range(2, n_arms + 1))
    n_ks = [0]
    for r in range(1, n_arms):
        n_ks.append(int(math.ceil((budget - n_arms) / (log_k * (n_arms + 1 - r)))) if budget > n_arms else 0)
    pulls = []
    for r in range(1, n_arms):
        pulls.append(max(0, n_ks[r] - n_ks[r - 1]))
    return tuple(pulls)


def ege_sequential_halving_rounds(n_arms: int) -> int:
    """Return ``⌈log2(K)⌉``, the number of EGE-SH elimination rounds."""
    n_arms = _positive_integer(n_arms, "n_arms")
    if n_arms == 1:
        return 0
    return int(math.ceil(math.log2(n_arms)))


def ege_sequential_halving_pulls(n_active: int, n_arms: int, budget: int) -> int:
    """Return new pulls per active arm for one Sequential Halving round."""
    n_active = _positive_integer(n_active, "n_active")
    n_rounds = ege_sequential_halving_rounds(n_arms)
    budget = _nonnegative_integer(budget, "budget")
    if n_rounds == 0:
        return 0
    return int(math.floor(budget / (n_active * n_rounds)))


def ege_keep_count(n_active: int, variant: str) -> int:
    """Return how many arms stay active after one EGE elimination round."""
    n_active = _positive_integer(n_active, "n_active")
    if variant == EGE_SR:
        return max(1, n_active - 1)
    if variant == EGE_SH:
        return max(1, int(math.ceil(n_active / 2)))
    raise ValueError(f"variant must be one of {EGE_VARIANTS}, got {variant!r}")


def ege_select_survivors(
    means: np.ndarray,
    active: Sequence[int],
    n_keep: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Split ``active`` into (survivors, accepted, rejected) by empirical gap.

    Survivors are the ``n_keep`` active arms with the *smallest* gaps.
    Discarded arms that currently sit on the empirical Pareto set of the
    active pool are accepted as optimal; the rest are rejected as sub-optimal.
    This is Algorithm 1 of Kone et al. (AISTATS 2024).
    """
    active_idx = _unique_indices(active, "active")
    n_keep = int(n_keep)
    if n_keep < 1 or n_keep > len(active_idx):
        raise ValueError("n_keep must lie in [1, len(active)]")
    if means.shape[0] <= max(active_idx):
        raise ValueError("means must contain a row for every active arm")

    active_means = np.asarray(means, dtype=np.float64)[list(active_idx)]
    gaps = empirical_gaps(active_means)
    pareto = empirical_pareto_mask(active_means)
    # Smallest gap first; Pareto before non-Pareto when gaps tie.
    order = sorted(
        range(len(active_idx)),
        key=lambda i: (float(gaps[i]), 0 if pareto[i] else 1, active_idx[i]),
    )
    survivor_local = set(order[:n_keep])
    survivors = tuple(active_idx[i] for i in order[:n_keep])
    accepted: List[int] = []
    rejected: List[int] = []
    for local_index, arm in enumerate(active_idx):
        if local_index in survivor_local:
            continue
        if pareto[local_index]:
            accepted.append(int(arm))
        else:
            rejected.append(int(arm))
    return survivors, tuple(accepted), tuple(rejected)


def ape_pairwise_bonus(
    n_i: int,
    n_j: int,
    *,
    delta: float = 0.1,
    k1: float = 1.0,
) -> float:
    """Time-uniform pairwise bonus ``β_{i,j}(t)`` from Kone et al. (NeurIPS 2023, eq. 4).

    Their experiments replace the union-bound ``K1 = K(K−1)D/2`` with ``K1 = 1``.
    ``C_g(x) ≈ x + log(x)``.
    """
    pulls_i = _positive_integer(n_i, "n_i")
    pulls_j = _positive_integer(n_j, "n_j")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not math.isfinite(k1) or k1 <= 0.0:
        raise ValueError("k1 must be finite and positive")
    log_term = math.log(k1 / (delta * delta))
    log_term += math.log(4.0 + math.log(pulls_i)) + math.log(4.0 + math.log(pulls_j))
    log_term = max(log_term, 1e-12)
    calibration = log_term + math.log(log_term)
    return float(math.sqrt(max(calibration, 0.0) * (1.0 / pulls_i + 1.0 / pulls_j)))


def ape_opt_mask(
    means: np.ndarray,
    pulls: Sequence[int],
    *,
    epsilon1: float = 0.0,
    delta: float = 0.1,
    k1: float = 1.0,
) -> np.ndarray:
    """Return the ``OPT_ε1`` mask: arms whose lower CB already certifies ε1-optimality."""
    values = _finite_means(means)
    counts = _pull_counts(pulls, values.shape[0])
    if not math.isfinite(epsilon1) or epsilon1 < 0.0:
        raise ValueError("epsilon1 must be finite and nonnegative")
    if values.shape[1] == 2:
        lower, _competitors, _upper = _ape_pairwise_extrema_2d(
            values, counts, delta=delta, k1=k1,
        )
    else:
        M_ij, bonuses = _ape_pairwise_scores(
            values, counts, delta=delta, k1=k1,
        )
        lower_matrix = M_ij - bonuses
        np.fill_diagonal(lower_matrix, np.inf)
        lower = np.min(lower_matrix, axis=1)
    return lower + epsilon1 > 0.0


def ape_select_arm(
    means: np.ndarray,
    pulls: Sequence[int],
    *,
    epsilon1: float = 0.0,
    delta: float = 0.1,
    k1: float = 1.0,
) -> int:
    """Return the next APE arm: least-pulled of the LUCB-style pair ``(b_t, c_t)``.

    If every arm is already in ``OPT_ε1`` (the APE-k stop would have fired for
    ``k = K``), continue by pulling the globally least-sampled arm so a
    fixed-budget run can exhaust its remaining questions.
    """
    values = _finite_means(means)
    counts = _pull_counts(pulls, values.shape[0])
    if not math.isfinite(epsilon1) or epsilon1 < 0.0:
        raise ValueError("epsilon1 must be finite and nonnegative")
    if values.shape[1] == 2:
        lower, competitors, optimistic = _ape_pairwise_extrema_2d(
            values, counts, delta=delta, k1=k1,
        )
    else:
        M_ij, bonuses = _ape_pairwise_scores(
            values, counts, delta=delta, k1=k1,
        )
        lower_matrix = M_ij - bonuses
        np.fill_diagonal(lower_matrix, np.inf)
        lower = np.min(lower_matrix, axis=1)
        competitors = np.argmin(lower_matrix, axis=1)
        optimistic_matrix = M_ij + bonuses
        np.fill_diagonal(optimistic_matrix, np.inf)
        optimistic = np.min(optimistic_matrix, axis=1)
    opt = lower + epsilon1 > 0.0
    unfinished = np.flatnonzero(~opt)
    if unfinished.size == 0:
        return int(np.argmin(counts))

    # np.argmax returns the first maximum, matching the old ``-i`` tie-break
    # because ``unfinished`` is in increasing arm order.
    b_t = int(unfinished[int(np.argmax(optimistic[unfinished]))])
    c_t = int(competitors[b_t])
    return int(b_t if counts[b_t] <= counts[c_t] else c_t)


def _ape_pairwise_extrema_2d(
    values: np.ndarray,
    counts: np.ndarray,
    *,
    delta: float,
    k1: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact APE row extrema without materializing a ``K x K`` matrix.

    For two objectives, let ``a_i = x_i - y_i``. Then

    ``M(i,j) = x_i-x_j`` when ``a_j <= a_i``, and ``y_i-y_j`` otherwise.

    The APE bonus is constant for all competitors with the same pull count.
    For each count group we therefore answer every arm's minimum ``M(i,j)``
    using prefix maxima of x and suffix maxima of y. The result is identical
    to the all-pairs definition, including exclusion of self and lowest-index
    tie breaks, while typical work is ``O(K U log K)`` for ``U`` distinct
    pull counts and memory is linear in ``K``.
    """
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not math.isfinite(k1) or k1 <= 0.0:
        raise ValueError("k1 must be finite and positive")

    n_arms = int(values.shape[0])
    arm_ids = np.arange(n_arms, dtype=np.int64)
    coordinates = values[:, 0] - values[:, 1]
    lower = np.full(n_arms, np.inf, dtype=np.float64)
    lower_competitors = np.full(n_arms, -1, dtype=np.int64)
    upper = np.full(n_arms, np.inf, dtype=np.float64)
    float_counts = counts.astype(np.float64)

    for group_count in np.unique(counts):
        group_ids = np.flatnonzero(counts == group_count)
        group_coordinates = coordinates[group_ids]
        order = np.lexsort((group_ids, group_coordinates))
        group_ids = group_ids[order]
        group_coordinates = group_coordinates[order]

        prefix = _running_top_two(
            values[group_ids, 0], group_ids, reverse=False,
        )
        suffix = _running_top_two(
            values[group_ids, 1], group_ids, reverse=True,
        )
        left_values, left_ids = _query_running_maximum(
            prefix,
            np.searchsorted(group_coordinates, coordinates, side="right") - 1,
            arm_ids,
        )
        right_values, right_ids = _query_running_maximum(
            suffix,
            np.searchsorted(group_coordinates, coordinates, side="right"),
            arm_ids,
        )
        left_scores = values[:, 0] - left_values
        right_scores = values[:, 1] - right_values
        use_right = (right_scores < left_scores) | (
            (right_scores == left_scores)
            & (right_ids >= 0)
            & ((left_ids < 0) | (right_ids < left_ids))
        )
        group_minimum = np.where(use_right, right_scores, left_scores)
        group_competitors = np.where(use_right, right_ids, left_ids)
        bonuses = _ape_bonus_against_count(
            float_counts,
            int(group_count),
            delta=delta,
            k1=k1,
        )
        group_lower = group_minimum - bonuses
        replace = (group_lower < lower) | (
            (group_lower == lower)
            & (group_competitors >= 0)
            & (
                (lower_competitors < 0)
                | (group_competitors < lower_competitors)
            )
        )
        lower = np.where(replace, group_lower, lower)
        lower_competitors = np.where(
            replace, group_competitors, lower_competitors,
        )
        upper = np.minimum(upper, group_minimum + bonuses)
    return lower, lower_competitors, upper


def _running_top_two(
    values: np.ndarray,
    arm_ids: np.ndarray,
    *,
    reverse: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return best and second-best value/id at every prefix or suffix."""
    n_values = int(values.size)
    best_values = np.full(n_values, -np.inf, dtype=np.float64)
    best_ids = np.full(n_values, -1, dtype=np.int64)
    second_values = np.full(n_values, -np.inf, dtype=np.float64)
    second_ids = np.full(n_values, -1, dtype=np.int64)
    best_value = second_value = -math.inf
    best_id = second_id = -1
    positions = (
        range(n_values - 1, -1, -1) if reverse else range(n_values)
    )
    for position in positions:
        value = float(values[position])
        arm_id = int(arm_ids[position])
        if value > best_value or (value == best_value and arm_id < best_id):
            second_value, second_id = best_value, best_id
            best_value, best_id = value, arm_id
        elif value > second_value or (
            value == second_value and arm_id < second_id
        ):
            second_value, second_id = value, arm_id
        best_values[position], best_ids[position] = best_value, best_id
        second_values[position], second_ids[position] = second_value, second_id
    return best_values, best_ids, second_values, second_ids


def _query_running_maximum(
    running: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    positions: np.ndarray,
    query_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query running maxima, substituting the runner-up to exclude self."""
    best_values, best_ids, second_values, second_ids = running
    values = np.full(query_ids.size, -np.inf, dtype=np.float64)
    ids = np.full(query_ids.size, -1, dtype=np.int64)
    valid = (positions >= 0) & (positions < best_values.size)
    selected_positions = positions[valid]
    selected_ids = best_ids[selected_positions]
    selects_self = selected_ids == query_ids[valid]
    values[valid] = np.where(
        selects_self,
        second_values[selected_positions],
        best_values[selected_positions],
    )
    ids[valid] = np.where(
        selects_self,
        second_ids[selected_positions],
        selected_ids,
    )
    return values, ids


def _ape_bonus_against_count(
    counts: np.ndarray,
    other_count: int,
    *,
    delta: float,
    k1: float,
) -> np.ndarray:
    """Vectorized ``beta(n_i, other_count)`` with scalar-formula parity."""
    log_terms = (
        math.log(k1 / (delta * delta))
        + np.log(4.0 + np.log(counts))
        + math.log(4.0 + math.log(other_count))
    )
    np.maximum(log_terms, 1e-12, out=log_terms)
    calibration = log_terms + np.log(log_terms)
    return np.sqrt(
        np.maximum(calibration, 0.0)
        * (1.0 / counts + 1.0 / other_count)
    )


def _ape_pairwise_scores(
    values: np.ndarray,
    counts: np.ndarray,
    *,
    delta: float,
    k1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(M(i,j), beta(i,j))`` matrices for APE.

    This is algebraically identical to :func:`ape_pairwise_bonus`, but moves
    the quadratic work out of Python.  APE necessarily compares every arm
    pair; on 1,000+ arm datasets the old nested scalar loops dominated the
    entire replay by orders of magnitude.
    """
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not math.isfinite(k1) or k1 <= 0.0:
        raise ValueError("k1 must be finite and positive")

    first = values[:, 0, None] - values[None, :, 0]
    for objective in range(1, values.shape[1]):
        difference = values[:, objective, None] - values[None, :, objective]
        np.maximum(first, difference, out=first)

    count_logs = np.log(4.0 + np.log(counts.astype(np.float64)))
    log_terms = count_logs[:, None] + count_logs[None, :]
    log_terms += math.log(k1 / (delta * delta))
    np.maximum(log_terms, 1e-12, out=log_terms)
    calibration = log_terms + np.log(log_terms)
    np.maximum(calibration, 0.0, out=calibration)
    inverse_counts = 1.0 / counts
    calibration *= inverse_counts[:, None] + inverse_counts[None, :]
    np.sqrt(calibration, out=calibration)
    return first, calibration


def _finite_means(means: np.ndarray) -> np.ndarray:
    values = np.asarray(means, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("means must have shape (K, D) with K, D ≥ 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("means must be finite")
    return values


def _pull_counts(pulls: Sequence[int], n_arms: int) -> np.ndarray:
    counts = np.asarray(pulls, dtype=np.int64)
    if counts.shape != (n_arms,):
        raise ValueError(f"pulls must have length {n_arms}")
    if np.any(counts < 1):
        raise ValueError("APE scoring requires at least one pull per arm")
    return counts


def _positive_integer(value: int, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_integer(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _unique_indices(values: Sequence[int], name: str) -> Tuple[int, ...]:
    idx = tuple(int(v) for v in values)
    if not idx:
        raise ValueError(f"{name} must be nonempty")
    if any(v < 0 for v in idx):
        raise ValueError(f"{name} indices must be nonnegative")
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name} indices must be unique")
    return idx
