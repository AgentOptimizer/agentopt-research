"""Matched cost-aware acquisition trajectories and finite-pool readouts."""

import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule
from experiments.combined_objective.cost_aware_purchases import (
    COHV_UCB, SCALARIZATION_UCB, cohv_purchases, scalarization_ucb_purchases,
)
from experiments.combined_objective.offline_radial_gittins import (
    RadialGittinsBoundaryCache, RadialGittinsGrid, _FiniteRecommendationCache,
    _scaled_batch_noise, fit_empirical_bayes_warm_start, raw_pareto_front_indices,
    simulate_radial_gittins,
)

CC_GITTINS = "cc_gittins"
GITTINS_ACQUISITION_TARGET = "finite_benchmark_mean"
RANDOM_QUESTIONS = "random_questions"
DEFAULT_METHODS = (CC_GITTINS, RANDOM_QUESTIONS, COHV_UCB)
METHODS = (*DEFAULT_METHODS, SCALARIZATION_UCB)
PURCHASES = {COHV_UCB: cohv_purchases, SCALARIZATION_UCB: scalarization_ucb_purchases}
TARGETS = (0.10, 0.30)
READOUT = "finite_lcb_beta1"
READOUTS = ("empirical_means", "finite_mean_beta0", READOUT)
BATCH_SIZE = 4
COST_TOLERANCE = 1e-10


class AtReportingLimit(Exception):
    """End a replay after observing a physical purchase past the last target."""


class PurchasePrefix:
    def __init__(self, limit):
        self.limit = limit
        self.visits = []

    def append(self, event):
        if (event["event"] != "direction_visit" or not event.get("question_ids")
                or event.get("selected_arm") is None):
            return
        if event["cumulative_search_cost_usd"] > self.limit + COST_TOLERANCE:
            raise AtReportingLimit
        self.visits.append((
            [(int(event["selected_arm"]), event["question_ids"])],
            (event["raw_posterior_mean_after"], event["raw_posterior_var_after"]),
        ))


def refresh_archive(conservative, archive, changed_arm):
    """Reuse the frontier when an updated nonmember is still dominated."""
    if changed_arm is not None and changed_arm not in archive:
        score, cost = conservative[changed_arm]
        for member in archive:
            member_score, member_cost = conservative[member]
            if (member_score >= score and member_cost <= cost
                    and (member_score > score or member_cost < cost)):
                return archive
    return raw_pareto_front_indices(conservative)


def replay_search(
    models, questions, table, full_cost, seed, *, methods=DEFAULT_METHODS,
    save_curve=False, targets=TARGETS, readouts=None,
):
    """Freeze readouts from the specified lookup table at complete purchases."""
    methods, targets = tuple(methods), tuple(targets)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("methods must be distinct supported policies")
    if (
        not targets or targets != tuple(sorted(set(targets))) or targets[0] < 0
        or not all(math.isfinite(target) for target in targets) or targets[-1] != TARGETS[-1]
    ):
        raise ValueError("targets must be strictly increasing from a nonnegative cap to 0.30")
    if readouts is not None:
        readouts = tuple(readouts)
        if not readouts or len(set(readouts)) != len(readouts) or not set(readouts) <= set(READOUTS):
            raise ValueError("readouts must be distinct supported recommendation rules")
    started = time.perf_counter()
    n, k, batch = len(questions), len(models), BATCH_SIZE
    if n < batch or k < 2 or not math.isfinite(full_cost) or full_cost <= 0:
        raise ValueError("replay needs at least four questions, two arms, and positive exhaustive cost")
    reporting_limit = TARGETS[-1] * full_cost
    ids = dict.fromkeys(range(k), questions)
    # Adaptive policies use independent four-question warm batches.
    independent = PerArmQuestionSchedule.create_from_available(
        ids, warm_start_batch_size=batch, seed=seed,
        question_order="independent", warm_start_question_order="independent",
    )
    independent_warm = tuple(tuple(independent.orders[arm][:batch]) for arm in range(k))
    random_order = tuple(map(int, np.random.default_rng(seed).permutation(questions)))
    warm_start_by_method = {
        method: (random_order[:batch],) * k if method == RANDOM_QUESTIONS else independent_warm
        for method in methods
    }
    warm_cost_fraction_by_method = {}
    quality_prior_variance_by_method = {}

    def calibrate(warm_batches):
        warm_scores = np.array([[table[m][q].score for q in qs] for m, qs in zip(models, warm_batches)])
        warm_costs = np.array([[table[m][q].cost for q in qs] for m, qs in zip(models, warm_batches)])
        quality_prior = max(float(np.var(warm_scores.mean(axis=1), ddof=1)), 1e-12)
        calibration = fit_empirical_bayes_warm_start(
            warm_scores, warm_costs, arm_ids=range(k),
            prior_variance=quality_prior, obs_noise_variance=None, cost_reference_usd=None,
            cost_model="raw_mean",
        )
        return warm_scores, warm_costs, calibration

    adaptive_calibration = calibrate(independent_warm)
    calibration = adaptive_calibration[2]
    prefix = PurchasePrefix(reporting_limit)
    cc_stop = cc_seconds = None
    if CC_GITTINS in methods:
        try:
            result = simulate_radial_gittins(
                models, questions, table, directions=((0., 1.), (1., 0.)),
                anytime=True, eta_decay_schedule="direction_stop", cost_model="raw_mean",
                recommendation_rule="completed_only",
                acquisition_target=GITTINS_ACQUISITION_TARGET,
                prior_variance=calibration.prior_var,
                question_order="independent", warm_start_question_order="independent",
                seed=seed, batch_size=batch, lambda_initial=1., lambda_decay=.5,
                boundary_z_padding_extra=2.,
                boundary_grid=RadialGittinsGrid(z_size=129, delta_size=129, state_size=129,
                                              boundary_margin_cells=4),
                boundary_cache=RadialGittinsBoundaryCache(),
                boundary_build_backend="scipy", record_trace=False, history=prefix,
                record_recommendation_trajectory=False,
            )
            cc_stop = result.stop_reason
            np.testing.assert_allclose(calibration.prior_var, result.params["prior_variance"])
            np.testing.assert_allclose(calibration.warm_obs_noise_var, result.params["obs_noise_variance"])
        except AtReportingLimit:
            cc_stop = "crossed_reporting_limit"
        cc_seconds = time.perf_counter() - started
    # Random questions use the same seed's shared order, evaluating every
    # configuration and publishing once each question is complete.
    rand_visits = (
        ([(arm, (question,)) for arm in range(k)], None)
        for question in random_order[batch:]
    )
    rows = []
    unavailable = []
    curve_by_method = {}
    visits_by_method = {CC_GITTINS: prefix.visits, RANDOM_QUESTIONS: rand_visits}
    for method in methods:
        warm_scores, warm_costs, calibration = (
            calibrate(warm_start_by_method[method]) if method == RANDOM_QUESTIONS
            else adaptive_calibration
        )
        posteriors = calibration.initialize_posteriors()
        finite = _FiniteRecommendationCache(
            [n] * k, calibration.warm_obs_noise_var * batch, calibration.normalizer, 1.0,
        )
        scores, costs = warm_scores.tolist(), warm_costs.tolist()
        for arm in range(k):
            finite.update(arm, posteriors[arm], scores[arm], costs[arm])
        spend = math.fsum(warm_costs.ravel())
        warm_cost_fraction_by_method[method] = spend / full_cost
        quality_prior_variance_by_method[method] = float(calibration.prior_var[0])
        evaluations = k * batch
        remaining_targets = list(targets)

        def record(target):
            if spend > target * full_cost + COST_TOLERANCE:
                unavailable.append({
                    "method": method, "target_cost_fraction": target,
                    "warm_cost_fraction": spend / full_cost,
                })
                return
            for rule in readouts or (READOUT,):
                if rule == "empirical_means":
                    selected = raw_pareto_front_indices(finite.observed_means)
                elif rule == "finite_mean_beta0":
                    selected = raw_pareto_front_indices(finite.means)
                else:
                    selected = finite.select().selected_arm_indices
                rows.append({
                    "method": method, "target_cost_fraction": target,
                    "cost_fraction": spend / full_cost, "actual_usd": spend,
                    "evaluations": evaluations, "recommendations": selected,
                    **({"readout": rule} if readouts is not None else {}),
                })

        archive = ()

        def snapshot(changed_arm=None):
            nonlocal archive
            archive = refresh_archive(finite.conservative, archive, changed_arm)
            return {
                "actual_usd": spend, "cost_fraction": spend / full_cost,
                "evaluations": evaluations, "recommendations": archive,
            }

        if save_curve:
            curve = [snapshot()] if spend <= reporting_limit + COST_TOLERANCE else []
            curve_by_method[method] = curve

        if method in PURCHASES:
            visits = (
                (group, None) for group in PURCHASES[method](
                    models, questions, table, independent, reporting_limit, batch
                )
            )
        else:
            visits = visits_by_method[method]
        for group, event in visits:
            selected_samples = [
                (arm, [table[models[arm]][q] for q in qs]) for arm, qs in group
            ]
            prices = [math.fsum(s.cost for s in samples) for _, samples in selected_samples]
            next_spend = spend
            for price in prices:
                next_spend += price
            while remaining_targets and next_spend > remaining_targets[0] * full_cost + COST_TOLERANCE:
                record(remaining_targets.pop(0))
            if not remaining_targets:
                break
            for (arm, samples), price in zip(selected_samples, prices):
                new_scores = [s.score for s in samples]
                new_costs = [s.cost for s in samples]
                count = len(samples)
                posteriors[arm].update(
                    calibration.posterior_observations(np.asarray(new_scores), np.asarray(new_costs)).mean(axis=0),
                    _scaled_batch_noise(calibration.warm_obs_noise_var,
                                        full_batch_size=batch, actual_batch_size=count),
                    batch_size=count,
                )
                scores[arm].extend(new_scores)
                costs[arm].extend(new_costs)
                finite.update(arm, posteriors[arm], scores[arm], costs[arm])
                spend += price
                evaluations += count
                if event and (evaluations % 1024 < batch or evaluations == k * batch + count):
                    np.testing.assert_allclose(posteriors[arm].mean, event[0], rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(posteriors[arm].var, event[1], rtol=1e-10, atol=1e-12)
            if save_curve:
                curve.append(snapshot(group[0][0] if len(group) == 1 else None))

        for target in remaining_targets:
            record(target)
    return {
        **({"curve_by_method": curve_by_method} if save_curve else {}),
        **({"unavailable_checkpoints": unavailable} if readouts is not None else {}),
        "rows": rows, "warm_start_by_method": warm_start_by_method,
        "warm_cost_fraction_by_method": warm_cost_fraction_by_method,
        "quality_prior_variance_by_method": quality_prior_variance_by_method,
        "cc_gittins_stop": cc_stop,
        "cc_gittins_acquisition_target": GITTINS_ACQUISITION_TARGET if CC_GITTINS in methods else None,
        "cohv_ucb_budget_fraction": TARGETS[-1] if COHV_UCB in methods else None,
        "cc_gittins_seconds": cc_seconds, "wall_seconds": time.perf_counter() - started,
    }


def write_json_once(destination: Path, result: dict) -> bool:
    """Publish complete JSON atomically, without overwriting another replay."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, prefix=".replay-", suffix=".tmp",
            encoding="utf-8", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, allow_nan=False)
        try:
            if os.name == "nt":
                # Windows rename refuses to overwrite an existing destination.
                temporary.rename(destination)
            else:
                os.link(temporary, destination)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
