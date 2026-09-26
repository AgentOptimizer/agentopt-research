"""Offline Q/latency/deployment-cost CC-Gittins pilot.

The requested directions have at most two active coordinates. Exact scalar
endpoint DPs and the existing two-dimensional radial DP therefore suffice;
there is no three-dimensional grid approximation. A Q-L direction means a
radial ray, not the arithmetic weighted-sum reward (Q + L) / 2.

Each physical cell provides Q, L in seconds, and D in USD simultaneously.
Two independent-coordinate pair posteriors reuse the established raw-mean
calibration and update code. Their duplicate Q coordinate is checked for
exact agreement. Search is always charged the observed USD once, irrespective
of the active objective. Full-data truth is read only after sampling ends.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from experiments.combined_objective.offline_radial_gittins import (
    _best_index,
    _quantize_effective_costs,
    finite_test_mean_moments,
)
from experiments.combined_objective.three_objective_metrics import raw_pareto_indices
from agentopt.model_selection.axis_gittins_dp import AxisGittinsBoundaryCache
from agentopt.model_selection.radial_gittins import (
    PerArmQuestionSchedule,
    fit_empirical_bayes_warm_start,
)
from agentopt.model_selection.radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
    direction_aware_grid,
    radial_posterior_coordinates,
    terminal_expected_radial_utility,
)


DEFAULT_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _directions(values: Iterable[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    result = []
    for value in values:
        direction = np.asarray(value, dtype=float)
        if (direction.shape != (3,) or not np.all(np.isfinite(direction))
                or np.any(direction < 0.0) or not np.isclose(direction.sum(), 1.0)):
            raise ValueError("directions must be finite nonnegative Q,L,D triples summing to one")
        active = np.flatnonzero(direction)
        if len(active) > 1 and tuple(active) != (0, 1):
            raise ValueError("this pilot supports exact Q/L/D endpoints and the Q-L face only")
        result.append(tuple(map(float, direction)))
    if not result or len(result) != len(set(result)):
        raise ValueError("directions must be nonempty and unique")
    return tuple(result)


def _values(sample: Any) -> np.ndarray:
    values = np.asarray((sample.score, sample.latency_seconds, sample.cost), dtype=float)
    if (not np.all(np.isfinite(values)) or not 0.0 <= values[0] <= 1.0
            or np.any(values[1:] < 0.0)):
        raise ValueError("every sampled cell needs finite Q in [0,1], L seconds >= 0 and D USD >= 0")
    return values


def simulate_three_objective_gittins(
    models: Sequence[str],
    datapoints: Sequence[int],
    table: Mapping[str, Mapping[int, Any]],
    *,
    seed: int = 42,
    batch_size: int = 4,
    warm_start_batch_size: int = 4,
    question_order: str = "independent",
    warm_start_question_order: str = "independent",
    directions: Iterable[Sequence[float]] = DEFAULT_DIRECTIONS,
    recommendation_beta: float = 1.0,
    lambda_initial: float = 1.0,
    lambda_decay: float = 0.5,
    grid_size: int = 129,
    effective_cost_bin_ratio: Optional[float] = 2.0,
    effective_cost_bin_anchor: float = 1e-4,
    observation_budget_fraction: float = 1.0,
    max_total_question_evaluations: Optional[int] = None,
    max_search_cost_usd: Optional[float] = None,
    stop_tolerance: float = 1e-9,
    boundary_cache_dir: Optional[str] = None,
    index_provider: Optional[Callable[[dict[str, Any], int], float]] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run the four-direction pilot and return JSON-compatible evidence.

    Directions share a single observation schedule and use equal round-robin
    visits with independent eta decay. Warm and subsequent per-arm question
    orders are independent by default. Recommendation uses
    all-arm finite-target LCB Q and UCB L,D with strict three-objective Pareto
    filtering. Coordinates in every raw array are Q, L seconds, D USD.

    ``points`` records the warm state and every actual batch, with all-arm
    finite target moments (arrays indexed by global arm index). A USD cap is
    a soft expected-cost reservation and can be exceeded by a realized batch.
    Numerical-floor stopping can leave dominated arms unfinished.
    """
    started = time.perf_counter()
    seed = _integer(seed, "seed", 0)
    batch_size = _integer(batch_size, "batch_size")
    warm_start_batch_size = _integer(warm_start_batch_size, "warm_start_batch_size")
    grid_size = _integer(grid_size, "grid_size", 17)
    models = list(models)
    if not models or len(models) != len(set(models)):
        raise ValueError("models must be nonempty and unique")
    question_ids = list(datapoints)
    if not question_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError("datapoints must be nonempty and unique")
    common = set(question_ids)
    for model in models:
        common.intersection_update(table[model])
    question_ids = sorted(common)
    if len(question_ids) < warm_start_batch_size:
        raise ValueError("common question intersection cannot fund the warm start")
    resolved_directions = _directions(directions)
    for name, value, allow_zero in (
        ("recommendation_beta", recommendation_beta, True),
        ("lambda_initial", lambda_initial, False),
        ("stop_tolerance", stop_tolerance, True),
    ):
        if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(f"invalid {name}")
    if not math.isfinite(lambda_decay) or not 0 < lambda_decay < 1:
        raise ValueError("lambda_decay must lie strictly between zero and one")
    if not math.isfinite(observation_budget_fraction) or not 0 < observation_budget_fraction <= 1:
        raise ValueError("observation_budget_fraction must lie in (0,1]")
    if max_search_cost_usd is not None and (not math.isfinite(max_search_cost_usd) or max_search_cost_usd <= 0):
        raise ValueError("max_search_cost_usd must be finite and positive")
    n_arms, n_questions = len(models), len(question_ids)
    question_cap = int(math.ceil(n_arms * n_questions * observation_budget_fraction))
    if max_total_question_evaluations is not None:
        question_cap = min(question_cap, _integer(max_total_question_evaluations, "max_total_question_evaluations"))
    if question_cap < n_arms * warm_start_batch_size:
        raise ValueError("observation budget cannot fund the mandatory uniform warm start")
    schedule = PerArmQuestionSchedule.create_from_available(
        {arm: question_ids for arm in range(n_arms)},
        warm_start_batch_size=warm_start_batch_size,
        seed=seed, question_order=question_order,
        warm_start_question_order=warm_start_question_order,
    )
    warm_batches = schedule.take_uniform_warm_start()
    observed = [list() for _ in models]
    observed_cells: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    def observe(arm: int, questions: Sequence[int]) -> np.ndarray:
        batch = []
        for question in questions:
            cell = (arm, question)
            if cell in seen:
                raise RuntimeError(f"duplicate physical observation: {cell}")
            value = _values(table[models[arm]][question])
            batch.append(value)
            observed[arm].append(value)
            seen.add(cell)
            observed_cells.append([arm, int(question)])
        return np.asarray(batch)

    warm = np.asarray([observe(arm, warm_batches[arm]) for arm in range(n_arms)])
    total_cost = float(warm[:, :, 2].sum())
    if max_search_cost_usd is not None and total_cost > max_search_cost_usd:
        raise ValueError("USD budget cannot fund the mandatory uniform warm start")
    total_evaluations = n_arms * warm_start_batch_size
    counts = np.full(n_arms, warm_start_batch_size, dtype=int)
    sums = warm.sum(axis=1)
    calibrations = [
        fit_empirical_bayes_warm_start(
            warm[:, :, 0], warm[:, :, axis], arm_ids=range(n_arms),
            question_ids=schedule.warm_start_question_ids or None,
            cost_model="raw_mean",
        )
        for axis in (1, 2)
    ]
    raw_pairs = [calibration.initialize_posteriors() for calibration in calibrations]
    reward_pairs = [
        {arm: calibration.reward_posterior(posterior) for arm, posterior in pairs.items()}
        for calibration, pairs in zip(calibrations, raw_pairs)
    ]
    pair_question_noise = [cal.warm_obs_noise_var * warm_start_batch_size for cal in calibrations]
    pair_batch_noise = [noise / batch_size for noise in pair_question_noise]
    reward_batch_noise = [
        noise / np.asarray((1.0, cal.cost_reference_usd ** 2))
        for noise, cal in zip(pair_batch_noise, calibrations)
    ]
    initial_variances = [pairs[0].var.copy() for pairs in reward_pairs]
    expected_batch_costs = batch_size * warm[:, :, 2].mean(axis=1)
    if np.any(expected_batch_costs <= 0):
        raise ValueError("every arm requires a strictly positive warm-estimated USD continuation cost")
    base_effective_costs = _quantize_effective_costs(
        expected_batch_costs, anchor=effective_cost_bin_anchor,
        bin_ratio=effective_cost_bin_ratio,
    )
    horizons = np.full(n_arms, int(math.ceil((n_questions - warm_start_batch_size) / batch_size)), dtype=int)
    pulls = np.zeros(n_arms, dtype=int)
    etas = np.full(len(resolved_directions), lambda_initial, dtype=float)
    eta_stages = np.zeros(len(resolved_directions), dtype=int)
    floor_stops: set[int] = set()
    axis_cache = AxisGittinsBoundaryCache()
    radial_cache = RadialGittinsBoundaryCache(cache_dir=boundary_cache_dir)
    index_cache: dict[tuple[Any, ...], float] = {}
    trace: list[dict[str, Any]] = []
    eta_events: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    grid_expansions: list[dict[str, Any]] = []
    grid_retries: list[dict[str, Any]] = []
    grid_lower, grid_upper = np.zeros(2), np.ones(2)
    warm_reward_means = np.asarray([posterior.mean for posterior in reward_pairs[0].values()])
    warm_reward_std = np.sqrt(initial_variances[0])
    grid_lower[1] = min(0.0, float(warm_reward_means[:, 1].min() - 6 * warm_reward_std[1]))
    grid_upper[1] = max(1.0, float(warm_reward_means[:, 1].max() + 6 * warm_reward_std[1]))
    grid_version = 0
    base_grid = RadialGittinsGrid(z_size=grid_size, delta_size=grid_size, state_size=grid_size,
                                boundary_margin_cells=4)

    def raw_state() -> tuple[np.ndarray, np.ndarray]:
        means, variances = np.empty((n_arms, 3)), np.empty((n_arms, 3))
        for arm in range(n_arms):
            ql, qd = raw_pairs[0][arm], raw_pairs[1][arm]
            if ql.mean[0] != qd.mean[0] or ql.var[0] != qd.var[0] or ql.n_questions != qd.n_questions:
                raise RuntimeError("shared Q posterior diverged between Q-L and Q-D")
            means[arm] = (ql.mean[0], ql.mean[1], qd.mean[1])
            variances[arm] = (ql.var[0], ql.var[1], qd.var[1])
        return means, variances

    def checkpoint(event: str) -> None:
        means, variances = np.empty((n_arms, 3)), np.empty((n_arms, 3))
        observed_means = sums / counts[:, None]
        for arm in range(n_arms):
            for pair, axis in ((0, 1), (1, 2)):
                mean, variance = finite_test_mean_moments(
                    raw_pairs[pair][arm], sums[arm, [0, axis]], int(counts[arm]),
                    n_questions, pair_question_noise[pair],
                )
                means[arm, [0, axis]] = mean
                variances[arm, [0, axis]] = variance
            if counts[arm] == n_questions:
                # Canonical empirical mean avoids splitting exact raw ties
                # through a different grouping of batch additions.
                observed_means[arm] = np.asarray(observed[arm]).mean(axis=0)
                means[arm] = observed_means[arm]
        stds = np.sqrt(variances)
        conservative = means + recommendation_beta * stds * (-1, 1, 1)
        selected = raw_pareto_indices(conservative)
        point = {
            "event": event, "total_evaluations": int(total_evaluations),
            "cumulative_evaluations": int(total_evaluations),
            "total_cost": float(total_cost), "cumulative_search_cost_usd": float(total_cost),
            "counts": counts.tolist(), "completed_arm_indices": np.flatnonzero(counts == n_questions).tolist(),
            "selected_arm_indices": list(selected), "selected_models": [models[arm] for arm in selected],
            "finite_target_mean_vectors": means.tolist(), "finite_target_std_vectors": stds.tolist(),
            "recommendation_raw_vectors": conservative.tolist(), "observed_mean_vectors": observed_means.tolist(),
            "direction_eta_multipliers": etas.tolist(), "direction_eta_stages": eta_stages.tolist(),
        }
        points.append(point)
        if progress_callback is not None:
            progress_callback({"event": event, "total_evaluations": int(total_evaluations),
                               "total_cost": float(total_cost), "completed_arms": int(np.sum(counts == n_questions))})

    def radial_table(direction: tuple[float, ...], arm: int, cost: float):
        pair_direction = direction[:2]
        factors = max(pair_direction) / np.asarray(pair_direction)
        lower, upper = factors * grid_lower, factors * grid_upper
        delta_min, delta_max = float(lower[0] - upper[1]), float(upper[0] - lower[1])
        padding = max(2.0, max(1.0, float(horizons[arm]) * cost + 1.0) + 2.0)
        for attempt in range(5):
            envelope = replace(
                base_grid,
                z_min=min(base_grid.z_min, float(lower.mean()) - padding, -6 - padding),
                z_max=max(base_grid.z_max, float(upper.mean()) + padding, 6 + padding,
                          (max(abs(delta_min), abs(delta_max)) + .2) / 2 + padding),
                delta_min=min(base_grid.delta_min, delta_min - .2),
                delta_max=max(base_grid.delta_max, delta_max + .2),
            )
            grid = direction_aware_grid(
                pair_direction, base_grid=envelope, objective_lower=grid_lower,
                objective_upper=grid_upper, z_padding=padding,
            )
            try:
                return radial_cache.get(
                    direction=pair_direction, effective_pull_cost=cost,
                    initial_var=initial_variances[0], obs_noise_var=reward_batch_noise[0],
                    horizon=int(horizons[arm]), grid=grid,
                )
            except BoundaryGridError as error:
                if attempt == 4:
                    raise BoundaryGridError(f"Q-L pilot exhausted grid retries: {error}") from error
                grid_retries.append({"direction": list(direction), "arm_index": arm,
                                     "effective_pull_cost": cost, "old_padding": padding,
                                     "new_padding": 2 * padding, "reason": str(error)})
                padding *= 2
        raise AssertionError("unreachable")

    def direction_value(direction_index: int, arm: int, completed: bool) -> float:
        direction = resolved_directions[direction_index]
        active = np.flatnonzero(direction)
        if len(active) == 1:
            axis = int(active[0])
            pair, local_axis = (1, 1) if axis == 2 else (0, axis)
            posterior = reward_pairs[pair][arm]
            if completed:
                return float(posterior.mean[local_axis])
            table_for_arm = axis_cache.get(
                effective_pull_cost=float(etas[direction_index] * base_effective_costs[arm]),
                initial_var=float(initial_variances[pair][local_axis]),
                obs_noise_var=float(reward_batch_noise[pair][local_axis]),
                horizon=int(horizons[arm]), grid_size=grid_size,
            )
            return table_for_arm.index(int(pulls[arm]), float(posterior.mean[local_axis]))
        posterior = reward_pairs[0][arm]
        if completed:
            return terminal_expected_radial_utility(posterior.mean, posterior.var, direction[:2])
        table_for_arm = radial_table(direction, arm, float(etas[direction_index] * base_effective_costs[arm]))
        u, delta, _ = radial_posterior_coordinates(posterior.mean, posterior.var, direction[:2])
        return float(u - table_for_arm.boundary(int(pulls[arm]), delta))

    for arm in range(n_arms):
        raw_means, raw_variances = raw_state()
        trace.append({
            "event": "warm_start", "arm_index": arm, "selected_arm": arm,
            "question_ids": list(warm_batches[arm]), "batch_raw_mean": warm[arm].mean(axis=0).tolist(),
            "actual_batch_search_cost_usd": float(warm[arm, :, 2].sum()),
            "raw_posterior_mean_after": raw_means[arm].tolist(),
            "raw_posterior_var_after": raw_variances[arm].tolist(),
            "cumulative_evaluations": (arm + 1) * warm_start_batch_size,
            "cumulative_search_cost_usd": float(warm[:arm + 1, :, 2].sum()),
        })
    checkpoint("warm_start")
    visit = 0
    stop_reason = "all_arms_completed"
    while np.any(counts < n_questions):
        if total_evaluations >= question_cap:
            stop_reason = "observation_budget"
            break
        direction_index = visit % len(resolved_directions)
        visit += 1
        if direction_index in floor_stops:
            continue
        direction = resolved_directions[direction_index]
        unfinished = np.flatnonzero(counts < n_questions).tolist()
        completed = np.flatnonzero(counts == n_questions).tolist()
        context = {
            "direction_index": direction_index, "direction": direction,
            "global_step": visit - 1, "current_lambda": float(etas[direction_index]),
            "lambda_stage": int(eta_stages[direction_index]), "counts": counts.copy(),
            "adaptive_pulls": pulls.copy(), "completed_arms": tuple(completed),
            "unfinished_arms": tuple(unfinished), "expected_batch_costs_usd": expected_batch_costs.copy(),
            "effective_pull_costs": etas[direction_index] * base_effective_costs,
        }
        if index_provider is not None:
            context["raw_posterior_means"], context["raw_posterior_variances"] = raw_state()
            unfinished_indices = {arm: float(index_provider(context, arm)) for arm in unfinished}
        else:
            unfinished_indices = {}
            for arm in unfinished:
                key = (direction_index, arm, int(pulls[arm]), float(etas[direction_index]), grid_version)
                if key not in index_cache:
                    index_cache[key] = direction_value(direction_index, arm, False)
                unfinished_indices[arm] = index_cache[key]
        completed_indices = {arm: direction_value(direction_index, arm, True) for arm in completed}
        best_arm, best_value = _best_index(unfinished_indices, stop_tolerance)
        best_completed_arm, best_completed_value = _best_index(completed_indices, stop_tolerance)
        should_stop = best_completed_arm is not None and best_completed_value >= best_value - stop_tolerance
        event = {
            "event": "direction_visit", "global_step": visit - 1,
            "direction_index": direction_index, "direction": list(direction),
            "current_lambda": float(etas[direction_index]), "lambda_stage": int(eta_stages[direction_index]),
            "direction_should_stop": bool(should_stop), "best_completed_arm": best_completed_arm,
            "best_completed_terminal_index": best_completed_value if completed else None,
            "best_unfinished_arm": best_arm, "best_unfinished_gittins_index": best_value,
            "selected_arm": None, "question_ids": [], "actual_batch_search_cost_usd": 0.0,
            "cumulative_evaluations": int(total_evaluations), "cumulative_search_cost_usd": float(total_cost),
        }
        if should_stop:
            trace.append(event)
            remaining_penalty = max(float((horizons[arm] - pulls[arm]) * etas[direction_index] * base_effective_costs[arm]) for arm in unfinished)
            threshold = max(stop_tolerance, np.finfo(float).eps * max(1.0, abs(best_value), abs(best_completed_value)))
            old_eta = float(etas[direction_index])
            next_eta = old_eta * lambda_decay
            floor_reason = None
            if remaining_penalty <= threshold:
                floor_reason = "numerical_floor"
            elif not 0 < next_eta < old_eta or np.any(next_eta * base_effective_costs <= 0):
                floor_reason = "underflow"
            if floor_reason is None:
                etas[direction_index] = next_eta
                eta_stages[direction_index] += 1
            else:
                floor_stops.add(direction_index)
            eta_event = dict(event, event="direction_eta_decay" if floor_reason is None else "direction_eta_floor_stop",
                             next_lambda=next_eta if floor_reason is None else None,
                             direction_eta_multipliers=etas.tolist(), direction_eta_stages=eta_stages.tolist(),
                             floor_reason=floor_reason, max_remaining_effective_penalty=remaining_penalty,
                             numerical_penalty_threshold=float(threshold))
            eta_events.append(eta_event)
            trace.append(eta_event)
            if len(floor_stops) == len(resolved_directions):
                stop_reason = "direction_eta_numerical_floor"
                break
            continue
        assert best_arm is not None
        n_batch = min(batch_size, n_questions - int(counts[best_arm]))
        if total_evaluations + n_batch > question_cap:
            stop_reason = "observation_budget"
            break
        reservation = float(expected_batch_costs[best_arm]) * n_batch / batch_size
        if max_search_cost_usd is not None and total_cost + reservation > max_search_cost_usd:
            stop_reason = "search_cost_budget"
            break
        questions = schedule.next_batch(best_arm, n_batch)
        batch = observe(best_arm, questions)
        before_means, before_vars = raw_state()
        for pair, axis in ((0, 1), (1, 2)):
            raw_pairs[pair][best_arm].update(
                batch[:, [0, axis]].mean(axis=0), pair_question_noise[pair] / n_batch,
                batch_size=n_batch,
            )
            reward_pairs[pair][best_arm] = calibrations[pair].reward_posterior(raw_pairs[pair][best_arm])
        after_means, after_vars = raw_state()
        counts[best_arm] += n_batch
        pulls[best_arm] += 1
        sums[best_arm] += batch.sum(axis=0)
        total_evaluations += n_batch
        batch_cost = float(batch[:, 2].sum())
        total_cost += batch_cost
        floor_stops.clear()
        posterior = reward_pairs[0][best_arm]
        radius = 6 * math.sqrt(float(posterior.var[1]))
        lower, upper = float(posterior.mean[1] - radius), float(posterior.mean[1] + radius)
        if lower < grid_lower[1] or upper > grid_upper[1]:
            old_lower, old_upper = grid_lower.copy(), grid_upper.copy()
            span = grid_upper[1] - grid_lower[1]
            if lower < grid_lower[1]:
                grid_lower[1] = min(lower, grid_lower[1] - span)
            if upper > grid_upper[1]:
                grid_upper[1] = max(upper, grid_upper[1] + span)
            grid_expansions.append({"arm_index": best_arm, "evaluations": total_evaluations,
                                    "old_lower": old_lower.tolist(), "old_upper": old_upper.tolist(),
                                    "new_lower": grid_lower.tolist(), "new_upper": grid_upper.tolist()})
            grid_version += 1
        event.update({
            "selected_arm": best_arm, "selected_model": models[best_arm], "question_ids": list(questions),
            "batch_raw_mean": batch.mean(axis=0).tolist(), "actual_batch_search_cost_usd": batch_cost,
            "expected_batch_search_cost_usd": float(expected_batch_costs[best_arm]),
            "raw_effective_pull_cost": float(etas[direction_index] * expected_batch_costs[best_arm]),
            "quantized_effective_pull_cost": float(etas[direction_index] * base_effective_costs[best_arm]),
            "raw_posterior_mean_before": before_means[best_arm].tolist(),
            "raw_posterior_var_before": before_vars[best_arm].tolist(),
            "raw_posterior_mean_after": after_means[best_arm].tolist(),
            "raw_posterior_var_after": after_vars[best_arm].tolist(),
            "cumulative_evaluations": int(total_evaluations), "cumulative_search_cost_usd": float(total_cost),
        })
        trace.append(event)
        checkpoint("pull")
    sampling_seconds = time.perf_counter() - started
    # Final metadata is independent of checkpoint timing or recommendation changes.
    points[-1]["final"] = True
    points[-1]["direction_eta_multipliers"] = etas.tolist()
    points[-1]["direction_eta_stages"] = eta_stages.tolist()
    raw_means, raw_variances = raw_state()
    # Oracle diagnostics begin here. None of these cells, means, or totals
    # participates in the online prior, grid, acquisition, or stopping.
    full_rows = [np.asarray([_values(table[model][q]) for q in question_ids]) for model in models]
    truth = np.asarray([row.mean(axis=0) for row in full_rows])
    bruteforce_cost = float(sum(row[:, 2].sum() for row in full_rows))
    parameters = {
        "seed": seed, "batch_size": batch_size, "warm_start_batch_size": warm_start_batch_size,
        "question_order": question_order, "warm_start_question_order": warm_start_question_order,
        "directions": [list(direction) for direction in resolved_directions],
        "direction_labels": ["Q" if direction == (1., 0., 0.) else "L" if direction == (0., 1., 0.) else "D" if direction == (0., 0., 1.) else "(Q+L)/2" for direction in resolved_directions],
        "direction_scheduler": "round_robin", "eta_decay_schedule": "direction_stop",
        "lambda_initial": float(lambda_initial), "lambda_decay": float(lambda_decay),
        "recommendation_rule": "finite_lcb", "recommendation_beta": float(recommendation_beta),
        "cost_model": "raw_mean", "latency_model": "raw_mean", "objective_order": ["Q", "L", "D"],
        "objective_units": ["accuracy", "seconds", "USD"],
        "direction_utility": "support-restricted radial; Q-L balanced ray uses min(Q_reward,L_reward)",
        "search_cost_units": "USD", "grid_size": grid_size, "boundary_build_backend": "scipy",
        "boundary_margin_cells": 4, "boundary_z_padding_extra": 2.0,
        "planning_batch_semantics": "DP stages assume full batches; realized final partial batch uses per-question noise divided by its actual size",
        "effective_cost_bin_ratio": effective_cost_bin_ratio, "effective_cost_bin_anchor": effective_cost_bin_anchor,
        "observation_budget_fraction": float(observation_budget_fraction),
        "max_total_question_evaluations": max_total_question_evaluations, "question_cap": question_cap,
        "max_search_cost_usd": max_search_cost_usd, "cost_budget_guard": "soft_expected_cost",
        "latency_reference_seconds": calibrations[0].cost_reference_usd,
        "cost_reference_usd": calibrations[1].cost_reference_usd,
        "raw_prior_mean": [float(calibrations[0].prior_mean[0]), float(calibrations[0].prior_mean[1]), float(calibrations[1].prior_mean[1])],
        "raw_prior_variance": [float(calibrations[0].prior_var[0]), float(calibrations[0].prior_var[1]), float(calibrations[1].prior_var[1])],
        "raw_question_noise_variance": [float(pair_question_noise[0][0]), float(pair_question_noise[0][1]), float(pair_question_noise[1][1])],
        "expected_batch_costs_usd": expected_batch_costs.tolist(), "base_effective_pull_costs": base_effective_costs.tolist(),
        "objective_grid_lower": grid_lower.tolist(), "objective_grid_upper": grid_upper.tolist(),
        "objective_grid_expansions": grid_expansions, "boundary_grid_retries": grid_retries,
        "posterior_independence_assumption": "independent Gaussian coordinates; shared physical Q,L,D observations",
        "terminal_completion_semantics": "acquisition terminal value is latent posterior utility, matching two-objective protocol",
    }
    return {
        "selector": "three_objective_cc_gittins" if index_provider is None else "three_objective_custom_index",
        "parameters": parameters, "params": parameters, "models": models, "model_names": models, "question_ids": question_ids,
        "n_arms": n_arms, "n_questions": n_questions, "raw_truth_vectors": truth.tolist(),
        "true_frontier_indices": raw_pareto_indices(truth), "full_data_pareto_arm_indices": raw_pareto_indices(truth),
        "bruteforce_search_cost_usd": bruteforce_cost, "cost_fraction": float(total_cost / bruteforce_cost),
        "points": points, "trace": trace, "eta_events": eta_events, "observed_cells": observed_cells,
        "stop_reason": stop_reason, "total_cost": float(total_cost), "search_cost_usd": float(total_cost), "cumulative_search_cost_usd": float(total_cost),
        "total_evaluations": int(total_evaluations), "counts": counts.tolist(),
        "selected_arm_indices": points[-1]["selected_arm_indices"],
        "raw_posterior_means": raw_means.tolist(), "raw_posterior_variances": raw_variances.tolist(),
        "direction_eta_multipliers": etas.tolist(), "direction_eta_stages": eta_stages.tolist(),
        "sampling_wall_seconds": sampling_seconds,
        "axis_boundary_cache_stats": axis_cache.stats_snapshot(),
        "radial_boundary_cache_stats": asdict(radial_cache.stats_snapshot()),
    }
