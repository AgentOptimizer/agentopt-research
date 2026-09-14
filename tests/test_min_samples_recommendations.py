"""A sample-count gate precedes direct finite-mean Pareto recommendation."""

import json
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import GaussianVectorPosterior, ObjectiveNormalizer
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


_DEFAULT = object()

def _cache(totals, *, minimum, beta=0.):
    return replay._FiniteRecommendationCache(
        totals, [.25, .0001], ObjectiveNormalizer(1., cost_model="raw_mean"), beta,
        min_samples=minimum,
    )


def _problem(points=((.3, .001), (.6, .01), (.9, .1)), n_questions=5):
    models, questions = [f"arm_{i}" for i in range(len(points))], list(range(n_questions))
    table = {
        model: {
            q: SampleResult(score=score, cost=cost, latency_seconds=0.,
                            input_tokens={}, output_tokens={})
            for q in questions
        }
        for model, (score, cost) in zip(models, points)
    }
    return models, questions, table


def _run(problem=None, **overrides):
    options = dict(
        seed=42, batch_size=2, anytime=True,
        eta_decay_schedule="direction_stop", directions=((.3, .7), (1., 0.)),
        cost_model="raw_mean", recommendation_rule="finite_mean",
        recommendation_min_samples=4, effective_cost_bin_ratio=None,
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True,
        index_provider=lambda context, arm: (
            100. - arm if context.current_lambda <= .5 else -100.
        ),
    )
    options.update(overrides)
    if options["recommendation_min_samples"] is _DEFAULT:
        options.pop("recommendation_min_samples")
    return replay.simulate_radial_gittins(*(_problem() if problem is None else problem), **options)


class MinimumSampleCacheTests(unittest.TestCase):
    def test_31_rows_are_ineligible_and_32_rows_are_eligible(self):
        cache = _cache((40,), minimum=32)
        posterior = GaussianVectorPosterior([.8, .1], [.04, .001])
        cache.update(0, posterior, [.8] * 31, [.1] * 31)
        empty = cache.select()
        for field in (
            "selected_arm_indices", "direction_winner_arm_indices",
            "estimated_raw_winner_vectors", "finite_target_mean_vectors",
            "finite_target_std_vectors", "recommendation_raw_vectors",
            "recommendation_desirability_vectors",
        ):
            self.assertEqual(getattr(empty, field), (), field)
        cache.update(0, posterior, [.8] * 32, [.1] * 32)
        selected = cache.select()
        self.assertEqual(selected.selected_arm_indices, (0,))
        np.testing.assert_allclose(selected.recommendation_raw_vectors, [[.8, .1]])
        self.assertGreater(selected.finite_target_std_vectors[0][0], 0.)

    def test_ineligible_superior_arm_cannot_filter_out_eligible_arm(self):
        cache = _cache((8, 8), minimum=4)
        cache.update(0, GaussianVectorPosterior([.6, .2], [.01, .001]), [.6] * 4, [.2] * 4)
        cache.update(1, GaussianVectorPosterior([.9, .1], [.01, .001]), [.9] * 3, [.1] * 3)
        self.assertEqual(cache.select().selected_arm_indices, (0,))
        cache.update(1, GaussianVectorPosterior([.9, .1], [.01, .001]), [.9] * 4, [.1] * 4)
        self.assertEqual(cache.select().selected_arm_indices, (1,))

    def test_mean_pareto_ignores_uncertainty_that_changes_lcb_membership(self):
        mean_cache, lcb_cache = _cache((8, 8), minimum=4), _cache((8, 8), minimum=4, beta=1.)
        for cache in (mean_cache, lcb_cache):
            cache.update(0, GaussianVectorPosterior([.8, .1], [.16, .0001]), [.8] * 4, [.1] * 4)
            cache.update(1, GaussianVectorPosterior([.7, .1], [.0001, .0001]), [.7] * 4, [.1] * 4)
        self.assertEqual(mean_cache.select().selected_arm_indices, (0,))
        self.assertEqual(lcb_cache.select().selected_arm_indices, (1,))

    def test_completion_does_not_bypass_the_explicit_sample_threshold(self):
        cache = _cache((3,), minimum=4)
        cache.update(0, GaussianVectorPosterior([.8, .1], [.04, .001]), [.8] * 3, [.1] * 3)
        self.assertEqual(cache.select().selected_arm_indices, ())
        np.testing.assert_array_equal(cache.stds, [[0., 0.]])


class MinimumSampleReplayTests(unittest.TestCase):
    def test_warm_gate_is_empty_then_records_actual_eligibility_crossing(self):
        result = _run()
        self.assertEqual(result.recommendation_initial_event.selected_arm_indices, ())
        self.assertEqual(result.recommendation_initial_event.cumulative_evaluations, 6)
        first = result.recommendation_events[0]
        self.assertEqual(first.cumulative_evaluations, 8)
        self.assertEqual(first.selected_arm_indices, (0,))
        self.assertEqual(first.direction_winner_sample_counts, (4,))
        self.assertEqual(first.added_arm_indices, (0,))
        self.assertEqual(first.completed_arm_indices, ())
        for event in result.recommendation_events:
            self.assertTrue(all(n >= 4 for n in event.direction_winner_sample_counts))
        self.assertEqual(result.selected_models, ["arm_0", "arm_1", "arm_2"])

    def test_smaller_last_batch_counts_actual_questions_not_planned_batches(self):
        result = _run(_problem(points=((.8, .1),)), recommendation_min_samples=5)
        self.assertEqual(result.total_evaluations, 5)
        self.assertEqual([event.cumulative_evaluations for event in result.recommendation_events], [5])
        self.assertEqual(result.recommendation_events[0].direction_winner_sample_counts, (5,))
        pulls = [event for event in result.trace if event.get("selected_arm") is not None]
        self.assertEqual([len(event["question_ids"]) for event in pulls], [2, 1])
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, (0,))

    def test_finite_lcb_can_use_the_same_gate_without_changing_acquisition(self):
        gated = _run(recommendation_rule="finite_lcb")
        ungated = _run(recommendation_rule="finite_lcb", recommendation_min_samples=0)
        self.assertEqual(gated.recommendation_initial_event.selected_arm_indices, ())
        self.assertTrue(ungated.recommendation_initial_event.selected_arm_indices)
        self.assertEqual(gated.trace, ungated.trace)
        self.assertEqual(gated.direction_eta_events, ungated.direction_eta_events)
        self.assertTrue(gated.recommendation_events)
        self.assertTrue(all(n >= 4 for event in gated.recommendation_events
                            for n in event.direction_winner_sample_counts))

    def test_finite_mean_has_no_penalty_even_if_legacy_beta_argument_is_nonzero(self):
        ordinary = _run()
        configured = _run(recommendation_beta=99.)
        self.assertEqual(ordinary.recommendation_events, configured.recommendation_events)
        self.assertEqual(ordinary.selected_models, configured.selected_models)
        self.assertEqual(configured.params["recommendation_beta"], 0.)
        partial_found = False
        for event in configured.recommendation_events:
            self.assertEqual(event.recommendation_beta, 0.)
            np.testing.assert_allclose(event.recommendation_raw_vectors, event.finite_target_mean_vectors)
            partial_found |= any(n < 5 for n in event.direction_winner_sample_counts)
        self.assertTrue(partial_found)

    def test_completed_finite_means_equal_empirical_frontier(self):
        result = _run()
        completed = _run(recommendation_rule="completed_only", recommendation_min_samples=0)
        self.assertEqual(result.selected_models, completed.selected_models)
        final = result.recommendation_final_snapshot
        for arm, mean in zip(final.direction_winner_arm_indices, final.finite_target_mean_vectors):
            np.testing.assert_allclose(mean, result.raw_truth_vectors[arm])
        np.testing.assert_array_equal(final.finite_target_std_vectors, np.zeros((3, 2)))
        # Recommendation mean becomes empirical without altering the Gaussian
        # acquisition state, which still has positive posterior variance.
        self.assertTrue(all(arm.raw_posterior_var[0] > 0 for arm in result.model_results))

    def test_gate_and_mean_rule_preserve_sampling_and_asynchronous_eta(self):
        result = _run()
        for rule in ("finite_lcb", "completed_only"):
            with self.subTest(rule=rule):
                baseline = _run(recommendation_rule=rule, recommendation_min_samples=0)
                self.assertEqual(result.trace, baseline.trace)
                self.assertEqual(result.observed_cells, baseline.observed_cells)
                self.assertEqual(result.model_results, baseline.model_results)
                self.assertEqual(result.total_cost, baseline.total_cost)
                self.assertEqual(result.stop_reason, baseline.stop_reason)
                self.assertEqual(result.direction_eta_events, baseline.direction_eta_events)
                self.assertEqual(result.direction_eta_stages, baseline.direction_eta_stages)
        self.assertTrue(result.direction_eta_events)

    def test_deferred_json_retains_empty_prefix_crossing_and_gate_metadata(self):
        live = _run()
        deferred = _run(defer_recommendation_diagnostics=True)
        self.assertEqual(live.recommendation_events, deferred.recommendation_events)
        self.assertEqual(deferred.recommendation_trajectory, [])
        saved = json.loads(json.dumps(replay._jsonable_result(deferred), allow_nan=False))
        self.assertEqual(saved["params"]["recommendation_min_samples"], 4)
        self.assertEqual(saved["params"]["recommendation_beta"], 0.)
        self.assertEqual(saved["params"]["recommendation_rule"], "finite_mean")
        replay.materialize_saved_recommendation_diagnostics(saved)
        replay.materialize_recommendation_diagnostics(deferred)
        self.assertEqual(deferred.recommendation_trajectory, live.recommendation_trajectory)
        self.assertEqual(deferred.recommendation_initial_snapshot, live.recommendation_initial_snapshot)
        self.assertEqual(deferred.recommendation_final_snapshot, live.recommendation_final_snapshot)
        live_json = replay._jsonable_result(live)
        for field in ("recommendation_trajectory", "recommendation_initial_snapshot", "recommendation_final_snapshot"):
            self.assertEqual(saved[field], live_json[field])
        self.assertEqual(saved["recommendation_initial_snapshot"]["selected_arm_indices"], [])

    def test_completed_but_below_threshold_produces_empty_final_recommendation(self):
        result = _run(recommendation_min_samples=6)
        self.assertTrue(all(arm.completed for arm in result.model_results))
        self.assertEqual(result.selected_models, [])
        self.assertEqual(result.recommendation_events, [])
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, ())

    def test_supported_rules_default_gate_remains_zero_and_preserves_explicit_zero_behavior(self):
        problem = _problem()
        for rule in ("completed_only", "finite_lcb", "finite_mean"):
            with self.subTest(rule=rule):
                explicit = _run(problem, recommendation_rule=rule, recommendation_min_samples=0)
                # Omit the argument entirely to exercise the public default.
                implicit = _run(problem, recommendation_rule=rule,
                                recommendation_min_samples=_DEFAULT)
                self.assertEqual(implicit.params["recommendation_min_samples"], 0)
                self.assertEqual(implicit.recommendation_events, explicit.recommendation_events)
                self.assertEqual(implicit.selected_models, explicit.selected_models)
                self.assertEqual(implicit.trace, explicit.trace)

    def test_invalid_sample_thresholds_and_nonfinite_rule_gates_are_rejected(self):
        for minimum in (True, False, -1, .5, "4"):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "recommendation_min_samples|min_samples"):
                    _run(recommendation_min_samples=minimum)
        for rule in ("completed_only",):
            with self.subTest(rule=rule):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _run(recommendation_rule=rule, recommendation_min_samples=4)

    def test_removed_lcb_rules_are_rejected(self):
        for rule in ("lcb", "hybrid_lcb"):
            with self.subTest(rule=rule):
                with self.assertRaisesRegex(ValueError, "recommendation_rule"):
                    _run(recommendation_rule=rule)


if __name__ == "__main__":
    unittest.main()
