"""Finite-benchmark acquisition resolves the uncertainty of the reported payoff."""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from agentopt.model_selection.axis_gittins_dp import build_axis_gittins_boundary_table
from agentopt.model_selection.finite_benchmark_gittins import FiniteBenchmarkGittinsCache
from agentopt.model_selection.gittins_schedules import finite_benchmark_schedule
from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments.combined_objective.offline_radial_gittins import simulate_radial_gittins
from experiments.combined_objective.offline_three_objective_gittins import simulate_three_objective_gittins
from experiments.single_objective.offline_selector_sim import SampleResult


def problem(n=5):
    models = ["a", "b"]
    questions = list(range(n))
    table = {
        model: {q: SampleResult(score=(.25, .75)[arm] + .03 * (q % 3),
                               cost=(.001, .004)[arm] * (1 + .1 * q),
                               latency_seconds=(2., 4.)[arm] + .2 * q,
                               input_tokens={}, output_tokens={})
                for q in questions}
        for arm, model in enumerate(models)
    }
    return models, questions, table


def two_run(**kwargs):
    options = dict(batch_size=4, warm_start_batch_size=4, cost_model="raw_mean",
                   acquisition_target="finite_benchmark_mean", recommendation_rule="finite_lcb",
                   directions=((1., 0.),), expected_batch_cost_usd=.01,
                   effective_cost_bin_ratio=None, seed=42, record_trace=True,
                   record_recommendation_trajectory=True)
    options.update(kwargs)
    return simulate_radial_gittins(*problem(), **options)


def test_partial_batches_partition_all_remaining_benchmark_uncertainty():
    initial = np.array([.04, .01])
    noise = np.array([.25, .05])
    variance, batches = finite_benchmark_schedule(initial, noise, n_total=13, n_observed=4, batch_size=4)
    np.testing.assert_array_equal(batches, [4, 4, 1])
    np.testing.assert_array_equal(variance[-1], [0., 0.])
    np.testing.assert_allclose((variance[:-1] - variance[1:]).sum(axis=0), variance[0])
    # Independently condition on a future batch mean. Cov(F,Y)=((N-n)*v+noise)/N.
    for stage, count in enumerate((4, 8, 12)):
        v = 1 / (1 / initial + (count - 4) / noise)
        covariance = ((13 - count) * v + noise) / 13
        expected_reduction = covariance**2 / (v + noise / batches[stage])
        np.testing.assert_allclose(variance[stage] - variance[stage + 1], expected_reduction)


def test_one_step_boundary_charges_the_actual_partial_batch():
    cache = FiniteBenchmarkGittinsCache()
    table = cache.get(direction=(1., 0.), effective_pull_cost=.02,
                      initial_var=(.04, .01), question_noise_var=(.25, .01),
                      n_total=5, n_observed=4, batch_size=4)
    std = np.sqrt((.04 + .25) / 25)
    expected = brentq(lambda x: x * norm.cdf(x / std) + std * norm.pdf(x / std) - .02 / 4, -2, 2)
    assert table.boundary(0) == pytest.approx(expected, abs=1e-11)
    assert table.boundary(1) == 0


def test_scalar_roots_match_the_affine_payoff_identity_at_fixed_cost():
    initial, noise, warm, n, batch = .04, .25, 4, 20, 4
    slope = (n - warm + noise / initial) / n
    finite = FiniteBenchmarkGittinsCache().get(
        direction=(1., 0.), effective_pull_cost=.01, initial_var=(initial, .01),
        question_noise_var=(noise, .01), n_total=n, n_observed=warm, batch_size=batch,
    )
    latent = build_axis_gittins_boundary_table(
        effective_pull_cost=.01 / slope, initial_var=initial,
        obs_noise_var=noise / batch, horizon=4, grid_size=finite.grid_size,
    )
    np.testing.assert_allclose(finite.boundaries, slope * latent.boundaries, atol=1e-11)


def test_radial_terminal_reward_has_no_unrevealed_uncertainty_and_cache_separates_targets():
    cache = FiniteBenchmarkGittinsCache()
    kwargs = dict(direction=(.5, .5), effective_pull_cost=.01,
                  initial_var=(.04, .03), question_noise_var=(.25, .2),
                  n_total=5, n_observed=4, batch_size=4,
                  grid=RadialGittinsGrid(z_size=65, delta_size=65, state_size=65))
    table = cache.get(**kwargs)
    np.testing.assert_array_equal(table.boundaries[-1], np.abs(table.delta_grid) / 2)
    assert cache.get(**kwargs) is table
    assert cache.get(**(kwargs | {"n_total": 6})) is not table
    assert cache.get(**(kwargs | {"n_observed": 3})) is not table


def test_replay_uses_full_benchmark_mean_in_the_index_and_completed_incumbent():
    run = two_run(max_total_question_evaluations=9)
    warm = [e for e in run.trace if e["event"] == "warm_start"]
    noise = run.params["obs_noise_variance"][0] * 4
    std = np.sqrt((warm[0]["raw_posterior_var_after"][0] + noise) / 25)
    root = brentq(lambda x: x * norm.cdf(x / std) + std * norm.pdf(x / std) - .01 / 4, -2, 2)
    means = [(4 * e["batch_score_mean"] + e["raw_posterior_mean_after"][0]) / 5 for e in warm]
    first = next(e for e in run.trace if e["event"] == "direction_visit")
    assert first["best_unfinished_gittins_index"] == pytest.approx(max(means) - root, abs=1e-10)
    models, questions, table = problem()
    winner = run.direction_winners[0]
    expected = np.mean([table[models[winner.arm_index]][q].score for q in questions])
    assert winner.terminal_utility == expected
    assert run.params["acquisition_target"] == "finite_benchmark_mean"


def test_three_objective_completed_incumbent_uses_observed_minimum():
    models, questions, table = problem()
    run = simulate_three_objective_gittins(
        models, questions, table, acquisition_target="finite_benchmark_mean",
        directions=((.5, .5, 0.),), grid_size=65,
        effective_cost_bin_ratio=None,
    )
    visits = [e for e in run["trace"] if e["event"] == "direction_visit" and e["best_completed_arm"] is not None]
    assert visits
    scale = run["params"]["latency_reference_seconds"]
    for event in visits:
        arm = event["best_completed_arm"]
        quality = np.mean([table[models[arm]][q].score for q in questions])
        latency = np.mean([table[models[arm]][q].latency_seconds for q in questions])
        assert event["best_completed_terminal_index"] == min(quality, 1 - latency / scale)


def test_historical_result_cannot_be_reused_as_benchmark_mean_acquisition():
    from experiments.combined_objective import run_two_direction_ablation as runner
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "exact_axes/mathqa/result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"search_cost_checkpoint_10pct": {}, "config": {}}))
        with patch.object(runner, "benchmark_input_hashes", return_value={}):
            with pytest.raises(ValueError, match="acquisition_target"):
                runner.run_one("mathqa", "exact_axes", seed=42, outdir=Path(directory),
                               observation_budget_fraction=1.)


def test_invalid_target_schedule_and_unsupported_modes_are_rejected():
    with pytest.raises(ValueError, match="acquisition_target"):
        two_run(acquisition_target="unknown")
    with pytest.raises(ValueError, match="raw_mean"):
        two_run(cost_model="reciprocal")
    with pytest.raises(ValueError, match="horizon_bin_width"):
        two_run(horizon_bin_width=2)
    with pytest.raises(ValueError, match="n_observed"):
        finite_benchmark_schedule([.1], [.2], n_total=5, n_observed=6, batch_size=4)
