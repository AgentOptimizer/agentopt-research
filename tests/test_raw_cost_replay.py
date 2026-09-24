"""Raw mean USD calibration must reach recommendations and the native DP."""

import json
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule
from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


WARM_COSTS = np.asarray(((.1, .2, .3, .4), (.02, .04, .08, .06)))


def _problem(cost_rows=None, *, scale=1.0):
    """Put specified warm/tail values at the actual seeded question positions."""
    if cost_rows is None:
        cost_rows = ((.1, .2, .3, .4, .2, .2, .2, .2),
                     (.02, .04, .08, .06, 1., 1., 1., 1.))
    models = ["accurate_cheap", "expensive_tail"]
    questions = list(range(len(cost_rows[0])))
    schedule = PerArmQuestionSchedule.create_from_available(
        {arm: questions for arm in range(2)}, warm_start_batch_size=4, seed=42,
        question_order="shared",
    )
    table = {
        model: {
            question: SampleResult(
                score=(.95, .7)[arm], cost=float(cost) * scale,
                latency_seconds=0., input_tokens={}, output_tokens={},
            )
            for question, cost in zip(schedule.orders[arm], cost_rows[arm])
        }
        for arm, model in enumerate(models)
    }
    return models, questions, table


def _run(problem=None, *, cost_model="raw_mean", **overrides):
    options = dict(
        seed=42, batch_size=4, anytime=True,
        eta_decay_schedule="direction_stop",
        recommendation_rule="completed_only", recommendation_beta=1.,
        directions=((.4, .6), (1., 0.)),
        effective_cost_bin_ratio=None,
        index_provider=lambda context, arm: 100. - arm,
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True,
    )
    if cost_model is not None:
        options["cost_model"] = cost_model
    options.update(overrides)
    return replay.simulate_radial_gittins(*(_problem() if problem is None else problem), **options)


def _physical_events(result):
    return [event for event in result.trace
            if event["event"] == "warm_start" or event.get("selected_arm") is not None]


def _event_arm(event):
    return event["arm_index"] if event["event"] == "warm_start" else event["selected_arm"]


class RawCostReplayTests(unittest.TestCase):
    def test_default_and_explicit_reciprocal_preserve_legacy_behavior(self):
        default = _run(cost_model=None)
        explicit = _run(cost_model="reciprocal")
        self.assertEqual(default.params["cost_model"], "reciprocal")
        self.assertEqual(default.trace, explicit.trace)
        self.assertEqual(default.model_results, explicit.model_results)
        self.assertEqual(default.selected_models, explicit.selected_models)
        self.assertEqual(default.recommendation_events, explicit.recommendation_events)
        self.assertEqual(default.direction_eta_events, explicit.direction_eta_events)

    def test_warm_calibration_and_every_update_use_raw_usd_precision(self):
        # A one-question final batch also checks the batch-noise B/n adjustment.
        problem = _problem((
            (.1, .2, .3, .4, .2, .2, .2, .2, .6),
            (.02, .04, .08, .06, 1., 1., 1., 1., .5),
        ))
        result = _run(problem)
        legacy = _run(problem, cost_model="reciprocal")
        reference = float(WARM_COSTS.mean(axis=1).max())
        prior_mean = float(WARM_COSTS.mean())
        prior_var = float(np.var(WARM_COSTS.mean(axis=1), ddof=1))
        question_noise = float(np.var(WARM_COSTS, axis=1, ddof=1).mean())
        self.assertAlmostEqual(result.cost_reference_usd, reference)
        self.assertAlmostEqual(result.prior_mean[1], prior_mean)
        self.assertAlmostEqual(result.prior_variance[1], prior_var)
        self.assertAlmostEqual(result.params["raw_cost_question_noise_variance_usd2"], question_noise)
        self.assertAlmostEqual(result.params["obs_noise_variance"][1], question_noise / 4)
        self.assertAlmostEqual(result.params["raw_cost_variance_floor_usd2"], 1e-12 * reference ** 2)
        self.assertEqual(result.prior_mean[0], legacy.prior_mean[0])
        accuracy_prior_variance = 0.03125  # Sample variance of (.95, .7).
        self.assertAlmostEqual(result.prior_variance[0], accuracy_prior_variance)
        self.assertEqual(legacy.prior_variance[0], 0.04)
        self.assertEqual(result.params["obs_noise_variance"][0], legacy.params["obs_noise_variance"][0])
        np.testing.assert_allclose(result.params["reward_prior_mean"], (result.prior_mean[0], 1 - prior_mean / reference))
        np.testing.assert_allclose(result.params["reward_prior_variance"], (result.prior_variance[0], prior_var / reference ** 2))
        self.assertAlmostEqual(result.params["reward_obs_noise_variance"][1], question_noise / (4 * reference ** 2))

        counts, sums = np.zeros(2, dtype=int), np.zeros(2)
        events = _physical_events(result)
        self.assertTrue(any(len(event["question_ids"]) == 1 for event in events))
        for event in events:
            arm = _event_arm(event)
            n = len(event["question_ids"])
            counts[arm] += n
            sums[arm] += n * event["batch_deployment_cost_mean_usd"]
            expected_var = 1 / (1 / prior_var + counts[arm] / question_noise)
            expected_mean = expected_var * (prior_mean / prior_var + sums[arm] / question_noise)
            np.testing.assert_allclose(event["raw_posterior_mean_after"][1], expected_mean, rtol=1e-12)
            np.testing.assert_allclose(event["raw_posterior_var_after"][1], expected_var, rtol=1e-12)
            np.testing.assert_allclose(event["posterior_mean_after"][1], 1 - expected_mean / reference, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(event["posterior_var_after"][1], expected_var / reference ** 2, rtol=1e-12)
            accuracy_var = 1 / (1 / accuracy_prior_variance + counts[arm] / 0.25)
            accuracy_mean = accuracy_var * (
                0.825 / accuracy_prior_variance + counts[arm] * (.95, .7)[arm] / 0.25
            )
            self.assertAlmostEqual(event["posterior_mean_after"][0], accuracy_mean)
            self.assertAlmostEqual(event["posterior_var_after"][0], accuracy_var)
        self.assertEqual(counts.tolist(), [9, 9])
        for summary in result.model_results:
            self.assertAlmostEqual(summary.equivalent_posterior_cost_usd, summary.raw_posterior_mean[1])
            self.assertAlmostEqual(summary.posterior_mean[1], 1 - summary.raw_posterior_mean[1] / reference)

    def test_unobserved_tail_does_not_change_calibration_or_objective_grid(self):
        first = _run(max_total_question_evaluations=12)
        changed = _run(_problem((
            (.1, .2, .3, .4, .2, .2, .2, .2),
            (.02, .04, .08, .06, 1e6, 1e6, 1e6, 1e6),
        )), max_total_question_evaluations=12)
        # Only arm zero's tail is sampled at this budget. Oracle diagnostics
        # may differ, but neither calibration nor the online grid may use it.
        self.assertEqual(first.observed_cells, changed.observed_cells)
        self.assertEqual(_physical_events(first), _physical_events(changed))
        self.assertEqual(first.cost_reference_usd, changed.cost_reference_usd)
        self.assertEqual(first.prior_mean, changed.prior_mean)
        self.assertEqual(first.prior_variance, changed.prior_variance)
        for key in ("obs_noise_variance", "objective_grid_lower", "objective_grid_upper", "objective_grid_expansions"):
            self.assertEqual(first.params[key], changed.params[key])
        self.assertFalse(np.array_equal(first.truth_vectors, changed.truth_vectors))

    def test_changing_usd_unit_scales_raw_moments_and_preserves_reward_moments(self):
        first = _run()
        factor = 1e-3
        scaled = _run(_problem(scale=factor))
        self.assertAlmostEqual(scaled.cost_reference_usd, first.cost_reference_usd * factor)
        self.assertAlmostEqual(scaled.total_cost, first.total_cost * factor)
        self.assertEqual(first.selected_models, scaled.selected_models)
        for left, right in zip(first.model_results, scaled.model_results):
            np.testing.assert_allclose(left.posterior_mean, right.posterior_mean, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(left.posterior_var, right.posterior_var, rtol=1e-12)
            np.testing.assert_allclose(right.raw_posterior_mean, np.asarray(left.raw_posterior_mean) * (1., factor), rtol=1e-12)
            np.testing.assert_allclose(right.raw_posterior_var, np.asarray(left.raw_posterior_var) * (1., factor ** 2), rtol=1e-12)

    def test_deferred_and_json_restored_checkpoints_equal_live_output(self):
        live = _run()
        deferred = _run(defer_recommendation_diagnostics=True)
        self.assertEqual(live.trace, deferred.trace)
        self.assertIsNone(deferred.recommendation_final_snapshot)
        saved = json.loads(json.dumps(replay._jsonable_result(deferred), allow_nan=False))
        replay.materialize_saved_recommendation_diagnostics(saved)
        replay.materialize_recommendation_diagnostics(deferred)
        self.assertEqual(live.recommendation_trajectory, deferred.recommendation_trajectory)
        self.assertEqual(live.recommendation_final_snapshot, deferred.recommendation_final_snapshot)
        live_json = replay._jsonable_result(live)
        for key in ("recommendation_trajectory", "recommendation_initial_snapshot", "recommendation_final_snapshot"):
            self.assertEqual(saved[key], live_json[key])
        self.assertEqual(saved["params"]["cost_model"], "raw_mean")
        self.assertIn("raw_posterior_mean", saved["model_results"][0])

    def test_native_dp_expands_for_observed_negative_cost_reward(self):
        problem = _problem((
            (.1, .2, .3, .4, 2., 2., 2., 2., 2.),
            (.02, .04, .08, .06, .05, .05, .05, .05, .05),
        ))
        result = _run(
            problem, index_provider=None,
            directions=((.5, .5), (1., 0.)),
            anytime=False, eta_decay_schedule="global_stop",
            halt_on_gittins_stop=False, search_cost_scale_eta=.01,
            boundary_build_backend="scipy",
            boundary_grid=RadialGittinsGrid(z_size=65, delta_size=65, state_size=65),
        )
        self.assertEqual(result.total_evaluations, 18)
        self.assertEqual(len(set(result.observed_cells)), 18)
        self.assertLess(result.model_results[0].posterior_mean[1], -1.)
        # Completion may expand the envelope once more and clear the live
        # grids, so verify actual table construction through cumulative stats.
        self.assertGreater(result.params["boundary_cache"]["stats"]["builds"], 0)
        expansions = result.params["objective_grid_expansions"]
        self.assertTrue(expansions)
        self.assertTrue(any(event["new_lower"][1] < event["old_lower"][1] for event in expansions))
        self.assertTrue(all(event["evaluations"] > 8 for event in expansions))
        for event in expansions:
            self.assertTrue(np.all(np.asarray(event["new_lower"]) <= event["old_lower"]))
            self.assertTrue(np.all(np.asarray(event["new_upper"]) >= event["old_upper"]))
        for event in _physical_events(result):
            mean = np.asarray(event["posterior_mean_after"])
            std = np.sqrt(event["posterior_var_after"])
            # Accuracy keeps its existing bounded objective box; only the raw
            # cost coordinate needs support outside the old [0, 1] envelope.
            self.assertLessEqual(result.params["objective_grid_lower"][1], mean[1] - 6 * std[1] + 1e-12)
            self.assertGreaterEqual(result.params["objective_grid_upper"][1], mean[1] + 6 * std[1] - 1e-12)
        self.assertTrue(np.isfinite(result.hypervolume))
        # The expensive arm is the accuracy extreme. Offline metrics must
        # retain its positive contribution even though its online reward is
        # below zero under the fixed warm reference.
        self.assertEqual(set(replay.nondominated_indices(result.truth_vectors)), {0, 1})
        self.assertTrue(np.all(result.truth_vectors[:, 1] > 0.))
        self.assertGreater(result.ground_truth_hypervolume, np.max(np.prod(result.truth_vectors, axis=1)))


if __name__ == "__main__":
    unittest.main()
