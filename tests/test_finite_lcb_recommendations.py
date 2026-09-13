"""Finite-test recommendations predict the remaining rows, not another population mean."""

import json
import unittest
from unittest import mock

import numpy as np

from agentopt.model_selection.radial_gittins import (
    GaussianVectorPosterior,
    ObjectiveNormalizer,
    PerArmQuestionSchedule,
)
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(points=((.3, .001), (.6, .01), (.9, .1), (.6, .01), (.2, .2)),
             *, n_questions=9, cost_rows=None):
    models = [f"arm_{arm}" for arm in range(len(points))]
    questions = list(range(n_questions))
    schedule = PerArmQuestionSchedule.create_from_available(
        {arm: questions for arm in range(len(models))}, warm_start_batch_size=4, seed=42,
        question_order="shared",
    )
    table = {}
    for arm, (model, (score, cost)) in enumerate(zip(models, points)):
        costs = [cost] * n_questions if cost_rows is None else cost_rows[arm]
        table[model] = {
            q: SampleResult(score=score, cost=float(value), latency_seconds=0.,
                            input_tokens={}, output_tokens={})
            for q, value in zip(schedule.orders[arm], costs)
        }
    return models, questions, table


def _run(problem=None, **overrides):
    options = dict(
        seed=42, batch_size=4, anytime=True,
        eta_decay_schedule="direction_stop", directions=((.3, .7), (1., 0.)),
        cost_model="raw_mean", recommendation_rule="finite_lcb",
        effective_cost_bin_ratio=None,
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True,
        index_provider=lambda context, arm: (
            100. - arm if context.current_lambda <= .5 else -100.
        ),
    )
    options.update(overrides)
    return replay.simulate_radial_gittins(*(_problem() if problem is None else problem), **options)


def _physical_events(result):
    return [event for event in result.trace
            if event["event"] == "warm_start" or event.get("selected_arm") is not None]


class FiniteTestMeanMomentsTests(unittest.TestCase):
    def test_informative_posterior_only_predicts_unobserved_rows(self):
        posterior = GaussianVectorPosterior([.3, 2.], [.04, .25])
        mean, variance = replay.finite_test_mean_moments(
            posterior, np.array([3.5, 20.]), 5, 10, np.array([.25, 1.]),
        )
        # Observed means are [.7, 4], distinct from the posterior [.3, 2].
        # Five known rows and five posterior-predicted rows yield [.5, 3].
        np.testing.assert_allclose(mean, [.5, 3.])
        np.testing.assert_allclose(variance, [.0225, .1125])
        np.testing.assert_array_equal(posterior.mean, [.3, 2.])
        np.testing.assert_array_equal(posterior.var, [.04, .25])

    def test_remaining_observation_noise_survives_nearly_known_population_mean(self):
        posterior = GaussianVectorPosterior([.5, 2.], [1e-20, 1e-20])
        _, variance = replay.finite_test_mean_moments(
            posterior, np.array([2.5, 10.]), 5, 10, np.array([.25, 1.]),
        )
        # Conditioning on the population mean cannot reveal the five future
        # realized rows: their independent noise still contributes m*tau^2/N^2.
        np.testing.assert_allclose(variance, [.0125, .05], rtol=1e-12)

    def test_completed_limit_is_empirical_with_exactly_zero_uncertainty(self):
        posterior = GaussianVectorPosterior([.01, 1000.], [100., 10000.])
        mean, variance = replay.finite_test_mean_moments(
            posterior, np.array([7., 3.]), 10, 10, np.array([.25, 1.]),
        )
        np.testing.assert_array_equal(mean, [.7, .3])
        np.testing.assert_array_equal(variance, [0., 0.])
        np.testing.assert_array_equal(posterior.mean, [.01, 1000.])
        np.testing.assert_array_equal(posterior.var, [100., 10000.])

    def test_no_observations_has_prior_and_finite_future_noise(self):
        posterior = GaussianVectorPosterior([.3, 2.], [.04, .25])
        mean, variance = replay.finite_test_mean_moments(
            posterior, np.zeros(2), 0, 10, np.array([.25, 1.]),
        )
        np.testing.assert_array_equal(mean, posterior.mean)
        np.testing.assert_allclose(variance, [.065, .35])

    def test_changing_currency_units_scales_moments_without_changing_accuracy(self):
        factor = np.array([1., 100.])
        mean, variance = replay.finite_test_mean_moments(
            GaussianVectorPosterior([.3, 2.], [.04, .25]),
            np.array([3.5, 20.]), 5, 10, np.array([.25, 1.]),
        )
        scaled_mean, scaled_variance = replay.finite_test_mean_moments(
            GaussianVectorPosterior(np.array([.3, 2.]) * factor,
                                    np.array([.04, .25]) * factor ** 2),
            np.array([3.5, 20.]) * factor, 5, 10, np.array([.25, 1.]) * factor ** 2,
        )
        np.testing.assert_allclose(scaled_mean, mean * factor)
        np.testing.assert_allclose(scaled_variance, variance * factor ** 2)


class FiniteFrontierCacheTests(unittest.TestCase):
    def test_cost_group_sweep_obeys_strict_dominance_and_keeps_duplicate_rows(self):
        points = np.array([
            [.8, .2],  # Dominated by the cheaper equal-accuracy arm 4.
            [.6, .1],  # Dominated by arm 4 at exactly the same cost.
            [.9, .3],  # Frontier.
            [.9, .3],  # Exact duplicate, retained as its own arm.
            [.8, .1],  # Frontier.
            [.7, .3],  # Dominated within its cost group.
            [.9, .4],  # Dominated by equal accuracy at lower cost.
            [.95, .5], # Frontier.
        ])
        self.assertEqual(replay.raw_pareto_front_indices(points), (2, 3, 4, 7))
        self.assertEqual(replay.raw_pareto_front_indices(np.empty((0, 2))), ())

    def test_cache_reads_only_new_samples_and_never_rereads_unchanged_histories(self):
        class SuffixOnly:
            def __init__(self, values, first_readable):
                self.values, self.first_readable = values, first_readable

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                if not isinstance(index, slice) or index.start < self.first_readable:
                    raise AssertionError("reread old observation history")
                return self.values[index]

        cache = replay._FiniteRecommendationCache(
            (4,), [.25, .01], ObjectiveNormalizer(1., cost_model="raw_mean"), 1.,
        )
        posterior = GaussianVectorPosterior([.5, .2], [.04, .01])
        scores, costs = [.5, .6], [.1, .2]
        cache.update(0, posterior, scores, costs)
        scores.append(.7)
        costs.append(.3)
        cache.update(0, posterior, SuffixOnly(scores, 2), SuffixOnly(costs, 2))
        np.testing.assert_allclose(cache.means, [[.575, .2]])
        cache.update(0, posterior, SuffixOnly(scores, 3), SuffixOnly(costs, 3))
        scores.append(.8)
        costs.append(.4)
        cache.update(0, posterior, scores, costs)
        with mock.patch.object(replay.np, "mean", side_effect=AssertionError("reread completion")):
            cache.update(0, posterior, SuffixOnly(scores, 4), SuffixOnly(costs, 4))
            selection = cache.select()
        np.testing.assert_allclose(selection.finite_target_mean_vectors, [[.65, .25]])
        np.testing.assert_array_equal(selection.finite_target_std_vectors, [[0., 0.]])

    def test_raw_accuracy_units_and_affine_display_units_are_consistent(self):
        cache = replay._FiniteRecommendationCache(
            (4,), [.25, 1.], ObjectiveNormalizer(10., score_bounds=(10., 110.), cost_model="raw_mean"), 1.,
        )
        cache.update(0, GaussianVectorPosterior([.3, 2.], [.04, .25]), [80., 90.], [3., 4.])
        selection = cache.select()
        # The normalized observed sum is 1.5; two remaining rows contribute .6.
        mean = np.array([62.5, 2.75])
        std = np.sqrt([.66 / 16, 3. / 16]) * [100., 1.]
        conservative = mean + std * [-1., 1.]
        np.testing.assert_allclose(selection.finite_target_mean_vectors, [mean])
        np.testing.assert_allclose(selection.finite_target_std_vectors, [std])
        np.testing.assert_allclose(selection.recommendation_raw_vectors, [conservative])
        np.testing.assert_allclose(selection.recommendation_desirability_vectors,
                                   [[(conservative[0] - 10.) / 100., 1. - conservative[1] / 10.]])


class FiniteLcbReplayTests(unittest.TestCase):
    def test_warm_partial_frontier_has_no_direction_filter_and_preserves_ties(self):
        result = _run(directions=((1., 0.),), max_total_question_evaluations=20)
        warm = result.recommendation_initial_snapshot
        self.assertEqual(warm.completed_arm_indices, ())
        self.assertEqual(warm.selected_arm_indices, (0, 1, 2, 3))
        self.assertEqual(result.selected_models, ["arm_0", "arm_1", "arm_2", "arm_3"])
        self.assertEqual(warm.selected_arm_indices, result.recommendation_final_snapshot.selected_arm_indices)
        self.assertTrue(all(not arm.completed for arm in result.model_results))
        self.assertEqual(warm.recommendation_beta, 1.)

    def test_score_uses_finite_mean_and_full_per_question_noise_in_raw_units(self):
        problem = _problem(points=((.8, 1.),), n_questions=9,
                           cost_rows=((.1, .2, .3, .4, .5, .6, .7, .8, .9),))
        for observed in (4, 8):
            with self.subTest(observed=observed):
                result = _run(problem, obs_noise_variance=(.025, 1.),
                              max_total_question_evaluations=observed)
                last_observation = _physical_events(result)[-1]
                snapshot = result.recommendation_final_snapshot
                posterior_mean = np.asarray(last_observation["raw_posterior_mean_after"])
                posterior_var = np.asarray(last_observation["raw_posterior_var_after"])
                # Accuracy input noise is batch-mean noise; raw cost noise is
                # fitted from within-warm-arm variance. Predicting even ONE
                # remaining row needs per-question noise, not batch noise.
                question_noise = np.array([.1, np.var([.1, .2, .3, .4], ddof=1)])
                np.testing.assert_allclose(np.asarray(result.params["obs_noise_variance"]) * 4,
                                           question_noise)
                remaining = 9 - observed
                observed_sum = np.array([.8 * observed, .1 * observed * (observed + 1) / 2])
                expected_mean = (observed_sum + remaining * posterior_mean) / 9
                expected_variance = (remaining ** 2 * posterior_var + remaining * question_noise) / 81
                expected_std = np.sqrt(expected_variance)
                expected_conservative = expected_mean + expected_std * [-1., 1.]
                np.testing.assert_allclose(snapshot.finite_target_mean_vectors, [expected_mean])
                np.testing.assert_allclose(snapshot.finite_target_std_vectors, [expected_std])
                np.testing.assert_allclose(snapshot.recommendation_raw_vectors,
                                           [expected_conservative])
                np.testing.assert_allclose(snapshot.recommendation_desirability_vectors,
                                           [[expected_conservative[0],
                                             1 - expected_conservative[1] / result.cost_reference_usd]])
                # Existing diagnostic slots continue to mean observed samples.
                np.testing.assert_allclose(snapshot.estimated_raw_winner_vectors,
                                           [observed_sum / observed])

    def test_completion_reduces_to_measured_raw_frontier_despite_posterior_shrinkage(self):
        finite = _run()
        completed = _run(recommendation_rule="completed_only")
        self.assertEqual(finite.selected_models, completed.selected_models)
        self.assertEqual(finite.selected_models, ["arm_0", "arm_1", "arm_2", "arm_3"])
        final = finite.recommendation_final_snapshot
        self.assertEqual(final.completed_arm_indices, (0, 1, 2, 3, 4))
        self.assertTrue(any(len(event["question_ids"]) == 1 for event in _physical_events(finite)))
        for arm, conservative, prediction in zip(
            final.direction_winner_arm_indices, final.recommendation_raw_vectors,
            final.finite_target_mean_vectors,
        ):
            empirical = finite.raw_truth_vectors[arm]
            np.testing.assert_allclose(prediction, empirical)
            np.testing.assert_allclose(conservative, empirical)
            self.assertGreater(finite.model_results[arm].raw_posterior_var[0], 0.)
        np.testing.assert_array_equal(final.finite_target_std_vectors, np.zeros((4, 2)))
        # The acquisition posterior still shrinks accuracy after every row is
        # known; the finite-target mean uses observed sums and no such shrinkage.
        self.assertNotAlmostEqual(finite.model_results[0].raw_posterior_mean[0],
                                  finite.raw_truth_vectors[0, 0])

    def test_recommendation_rule_preserves_acquisition_and_independent_eta(self):
        finite = _run()
        completed = _run(recommendation_rule="completed_only")
        self.assertEqual(finite.trace, completed.trace)
        self.assertEqual(finite.observed_cells, completed.observed_cells)
        self.assertEqual(finite.model_results, completed.model_results)
        self.assertEqual(finite.total_cost, completed.total_cost)
        self.assertEqual(finite.stop_reason, completed.stop_reason)
        self.assertEqual(finite.direction_eta_events, completed.direction_eta_events)
        self.assertTrue(finite.direction_eta_events)
        self.assertEqual(finite.direction_eta_stages, completed.direction_eta_stages)

    def test_deferred_and_saved_evidence_materialize_to_identical_snapshots(self):
        live = _run()
        deferred = _run(defer_recommendation_diagnostics=True)
        self.assertEqual(live.trace, deferred.trace)
        self.assertEqual(live.recommendation_events, deferred.recommendation_events)
        self.assertEqual(deferred.recommendation_trajectory, [])
        self.assertIsNone(deferred.recommendation_final_snapshot)
        saved = json.loads(json.dumps(replay._jsonable_result(deferred), allow_nan=False))
        replay.materialize_saved_recommendation_diagnostics(saved)
        replay.materialize_recommendation_diagnostics(deferred)
        self.assertEqual(live.recommendation_trajectory, deferred.recommendation_trajectory)
        self.assertEqual(live.recommendation_initial_snapshot, deferred.recommendation_initial_snapshot)
        self.assertEqual(live.recommendation_final_snapshot, deferred.recommendation_final_snapshot)
        live_saved = replay._jsonable_result(live)
        for field in ("recommendation_trajectory", "recommendation_initial_snapshot", "recommendation_final_snapshot"):
            self.assertEqual(saved[field], live_saved[field])

    def test_unobserved_tail_cannot_change_finite_recommendation(self):
        kwargs = dict(points=((.8, 1.),), n_questions=9)
        ordinary = _problem(**kwargs, cost_rows=((.1, .2, .3, .4, .5, .6, .7, .8, .9),))
        expensive = _problem(**kwargs, cost_rows=((.1, .2, .3, .4, 1e6, 1e6, 1e6, 1e6, 1e6),))
        first = _run(ordinary, max_total_question_evaluations=4)
        second = _run(expensive, max_total_question_evaluations=4)
        a, b = first.recommendation_initial_snapshot, second.recommendation_initial_snapshot
        self.assertEqual(first.trace, second.trace)
        self.assertEqual(a.selected_arm_indices, b.selected_arm_indices)
        self.assertEqual(a.recommendation_desirability_vectors, b.recommendation_desirability_vectors)
        self.assertEqual(a.estimated_raw_winner_vectors, b.estimated_raw_winner_vectors)
        self.assertEqual(a.finite_target_mean_vectors, b.finite_target_mean_vectors)
        self.assertEqual(a.finite_target_std_vectors, b.finite_target_std_vectors)
        self.assertEqual(a.recommendation_raw_vectors, b.recommendation_raw_vectors)
        self.assertFalse(np.array_equal(first.raw_truth_vectors, second.raw_truth_vectors))

    def test_non_raw_cost_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw_mean"):
            _run(cost_model="reciprocal")

    def test_per_arm_test_sizes_use_actual_remaining_counts_and_small_final_batches(self):
        problem = _problem(points=((.3, .01), (.9, .1)), n_questions=9)
        # Keep a shared four-row warm pool, with five versus nine total rows.
        problem[2]["arm_0"] = {q: row for q, row in problem[2]["arm_0"].items() if q < 5}
        result = _run(problem, question_universe="per_arm")
        initial = result.recommendation_initial_snapshot
        warm_events = {event["arm_index"]: event for event in result.trace
                       if event["event"] == "warm_start"}
        noise = np.asarray(result.params["obs_noise_variance"]) * 4
        self.assertEqual(initial.selected_arm_indices, (0, 1))
        for arm, mean, std in zip(initial.direction_winner_arm_indices,
                                  initial.finite_target_mean_vectors,
                                  initial.finite_target_std_vectors):
            total = (5, 9)[arm]
            remaining = total - 4
            event = warm_events[arm]
            observed_sum = np.array([event["batch_score_mean"],
                                     event["batch_deployment_cost_mean_usd"]]) * 4
            expected_mean = (observed_sum + remaining * np.asarray(event["raw_posterior_mean_after"])) / total
            expected_var = (remaining ** 2 * np.asarray(event["raw_posterior_var_after"])
                            + remaining * noise) / total ** 2
            np.testing.assert_allclose(mean, expected_mean)
            np.testing.assert_allclose(std, np.sqrt(expected_var))
        self.assertEqual(result.total_evaluations, 14)
        self.assertEqual(len(set(result.observed_cells)), 14)
        self.assertEqual([arm.n_samples_evaluated for arm in result.model_results], [5, 9])
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, (0, 1))
        np.testing.assert_array_equal(result.recommendation_final_snapshot.finite_target_std_vectors,
                                      [[0., 0.], [0., 0.]])


if __name__ == "__main__":
    unittest.main()
