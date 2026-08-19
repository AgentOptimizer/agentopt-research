import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from agentopt.model_selection.radial_gittins import (
    GaussianVectorPosterior,
    ObjectiveNormalizer,
    PerArmQuestionSchedule,
    default_batch_observation_noise,
    direction_scale,
    expected_min_of_two_normals,
    fit_empirical_bayes_warm_start,
    radial_utility,
)


class ObjectiveNormalizerTests(unittest.TestCase):
    def test_reciprocal_cost_reference_is_the_midpoint_not_a_maximum(self):
        normalizer = ObjectiveNormalizer(cost_reference_usd=0.2)

        self.assertEqual(normalizer.normalize_deployment_cost(0.0), 1.0)
        self.assertEqual(normalizer.normalize_deployment_cost(0.2), 0.5)
        self.assertEqual(normalizer.normalize_deployment_cost(0.6), 0.25)

    def test_score_bounds_are_configurable(self):
        normalizer = ObjectiveNormalizer(
            cost_reference_usd=1.0,
            score_bounds=(-1.0, 1.0),
        )

        self.assertEqual(normalizer.normalize_score(-1.0), 0.0)
        self.assertEqual(normalizer.normalize_score(0.0), 0.5)
        self.assertEqual(normalizer.normalize_score(1.0), 1.0)

    def test_invalid_normalizer_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            ObjectiveNormalizer(cost_reference_usd=0.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ObjectiveNormalizer(1.0).normalize_deployment_cost(-0.01)
        with self.assertRaisesRegex(ValueError, "outside configured bounds"):
            ObjectiveNormalizer(1.0).normalize_score(1.01)


class GaussianVectorPosteriorTests(unittest.TestCase):
    def test_one_batch_matches_two_scalar_conjugate_updates(self):
        posterior = GaussianVectorPosterior(
            mean=np.array([0.5, 0.5]),
            var=np.array([0.04, 0.04]),
        )
        posterior.update(
            observation=np.array([1.0, 0.0]),
            obs_noise_var=np.array([0.0625, 0.0625]),
            batch_size=4,
        )

        gain = 0.04 / (0.04 + 0.0625)
        expected_mean = np.array([0.5 + gain * 0.5, 0.5 - gain * 0.5])
        expected_var = np.full(2, 1.0 / (1.0 / 0.04 + 1.0 / 0.0625))
        np.testing.assert_allclose(posterior.mean, expected_mean)
        np.testing.assert_allclose(posterior.var, expected_var)
        self.assertEqual(posterior.n_batches, 1)
        self.assertEqual(posterior.n_questions, 4)

    def test_updated_returns_a_copy_and_variance_decreases(self):
        original = GaussianVectorPosterior([0.5, 0.5], [0.04, 0.04])
        updated = original.updated([0.7, 0.4], [0.0625, 0.0625], batch_size=4)

        np.testing.assert_array_equal(original.mean, [0.5, 0.5])
        np.testing.assert_array_equal(original.var, [0.04, 0.04])
        self.assertTrue(np.all(updated.var < original.var))
        self.assertEqual(original.n_batches, 0)
        self.assertEqual(updated.n_batches, 1)

    def test_noninteger_batch_size_is_not_silently_truncated(self):
        posterior = GaussianVectorPosterior([0.5, 0.5], [0.04, 0.04])
        with self.assertRaisesRegex(ValueError, "batch_size must be an integer"):
            posterior.update([0.5, 0.5], [0.1, 0.1], batch_size=1.5)

    def test_default_noise_uses_the_actual_partial_batch_size(self):
        posterior = GaussianVectorPosterior([0.5, 0.5], [0.04, 0.04])
        posterior.update([1.0, 0.0], batch_size=2)

        expected_tau_sq = 1.0 / 8.0
        expected_var = 1.0 / (1.0 / 0.04 + 1.0 / expected_tau_sq)
        np.testing.assert_allclose(posterior.var, np.full(2, expected_var))
        np.testing.assert_array_equal(
            default_batch_observation_noise(2),
            np.full(2, expected_tau_sq),
        )

    def test_repeated_updates_decrease_variance_and_do_not_alias_arms(self):
        calibration = fit_empirical_bayes_warm_start(
            np.full((2, 4), 0.5),
            np.array([[1.0] * 4, [2.0] * 4]),
        )
        posteriors = calibration.initialize_posteriors()
        arm_zero_before = posteriors[0].mean.copy()
        arm_one_var_before = posteriors[1].var.copy()

        posteriors[1].update([0.8, 0.4], batch_size=4)

        np.testing.assert_array_equal(posteriors[0].mean, arm_zero_before)
        self.assertTrue(np.all(posteriors[1].var < arm_one_var_before))
        self.assertEqual(posteriors[1].n_batches, 2)
        self.assertEqual(posteriors[1].n_questions, 8)


class EmpiricalBayesWarmStartTests(unittest.TestCase):
    def test_reference_is_median_of_arm_batch_means(self):
        # Arm means are 50, 2, and 3, so C_ref is 3.  The median over all
        # individual cells would instead be 2.5.
        scores = np.full((3, 2), 0.5)
        costs = np.array([[0.0, 100.0], [2.0, 2.0], [3.0, 3.0]])

        calibration = fit_empirical_bayes_warm_start(
            scores,
            costs,
            arm_ids=("a", "b", "c"),
            question_ids=(7, 11),
        )

        self.assertEqual(calibration.cost_reference_usd, 3.0)
        np.testing.assert_array_equal(
            calibration.raw_cost_means_usd,
            np.array([50.0, 2.0, 3.0]),
        )
        self.assertEqual(calibration.question_ids, (7, 11))

    def test_cost_is_transformed_per_question_before_batch_averaging(self):
        scores = np.full((3, 2), 0.5)
        costs = np.array([[0.0, 100.0], [2.0, 2.0], [3.0, 3.0]])

        calibration = fit_empirical_bayes_warm_start(scores, costs)
        observed = calibration.batch_observations[0, 1]
        expected = np.mean(3.0 / (3.0 + costs[0]))
        incorrect_transform_of_mean = 3.0 / (3.0 + costs[0].mean())

        self.assertAlmostEqual(observed, expected)
        self.assertNotAlmostEqual(observed, incorrect_transform_of_mean)

    def test_common_plugin_mean_and_fixed_variance_seed_every_arm(self):
        scores = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        costs = np.array([[1.0, 3.0], [4.0, 4.0], [7.0, 9.0]])

        calibration = fit_empirical_bayes_warm_start(
            scores,
            costs,
            arm_ids=("a", "b", "c"),
        )

        expected_cost_observations = np.mean(
            4.0 / (4.0 + costs), axis=1
        )
        expected_batch_observations = np.column_stack(
            (np.full(3, 0.5), expected_cost_observations)
        )
        expected_prior_mean = expected_batch_observations.mean(axis=0)
        np.testing.assert_allclose(
            calibration.batch_observations,
            expected_batch_observations,
        )
        np.testing.assert_allclose(calibration.prior_mean, expected_prior_mean)
        np.testing.assert_array_equal(calibration.prior_var, [0.04, 0.04])

        tau_sq = 1.0 / 8.0
        expected_posterior_var = 1.0 / (1.0 / 0.04 + 1.0 / tau_sq)
        posteriors = calibration.initialize_posteriors()
        for arm_index, arm_id in enumerate(calibration.arm_ids):
            posterior = posteriors[arm_id]
            expected_mean = expected_prior_mean + (
                0.04 / (0.04 + tau_sq)
            ) * (expected_batch_observations[arm_index] - expected_prior_mean)
            np.testing.assert_allclose(posterior.mean, expected_mean)
            np.testing.assert_allclose(
                posterior.var,
                np.full(2, expected_posterior_var),
            )
            # The arm's warm batch is a likelihood exactly once.  Its role in
            # estimating the shared plug-in hypermean is the chosen EB model.
            self.assertEqual(posterior.n_batches, 1)
            self.assertEqual(posterior.n_questions, 2)

    def test_default_noise_and_first_update_for_batch_sizes_four_and_eight(self):
        expected = {
            4: (0.0625, 0.024390243902439025),
            8: (0.03125, 0.017543859649122806),
        }
        for batch_size, (tau_sq, posterior_var) in expected.items():
            with self.subTest(batch_size=batch_size):
                calibration = fit_empirical_bayes_warm_start(
                    np.full((3, batch_size), 0.5),
                    np.ones((3, batch_size)),
                )
                np.testing.assert_allclose(
                    calibration.warm_obs_noise_var,
                    np.full(2, tau_sq),
                )
                for posterior in calibration.initialize_posteriors().values():
                    np.testing.assert_allclose(
                        posterior.var,
                        np.full(2, posterior_var),
                    )
                    self.assertEqual(posterior.n_batches, 1)

    def test_calibrated_reference_is_reused_for_later_outliers(self):
        calibration = fit_empirical_bayes_warm_start(
            np.full((3, 4), 0.5),
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                ]
            ),
        )

        self.assertEqual(calibration.cost_reference_usd, 2.0)
        desirability = calibration.normalizer.normalize_deployment_cost(1e9)
        self.assertAlmostEqual(desirability, 2.0 / (2.0 + 1e9))
        self.assertEqual(calibration.cost_reference_usd, 2.0)

    def test_calibration_hyperparameters_and_audit_arrays_are_read_only(self):
        calibration = fit_empirical_bayes_warm_start(
            np.full((2, 4), 0.5),
            np.array([[1.0] * 4, [2.0] * 4]),
        )

        with self.assertRaises(FrozenInstanceError):
            calibration.normalizer = ObjectiveNormalizer(99.0)
        with self.assertRaisesRegex(ValueError, "read-only"):
            calibration.prior_mean[0] = 0.9
        with self.assertRaisesRegex(ValueError, "read-only"):
            calibration.raw_costs_usd[0, 0] = 99.0

    def test_raw_warm_data_and_once_only_cost_accounting_are_retained(self):
        scores = np.array([[0.1, 0.2], [0.3, 0.4]])
        costs = np.array([[0.01, 0.02], [0.03, 0.04]])
        calibration = fit_empirical_bayes_warm_start(scores, costs)

        np.testing.assert_array_equal(calibration.raw_scores, scores)
        np.testing.assert_array_equal(calibration.raw_costs_usd, costs)
        self.assertEqual(calibration.total_question_evaluations, 4)
        self.assertAlmostEqual(calibration.warm_start_cost_usd, 0.1)

    def test_explicit_reference_allows_zero_median_cost(self):
        calibration = fit_empirical_bayes_warm_start(
            [[0.25, 0.75]],
            [[0.0, 0.0]],
            cost_reference_usd=0.5,
        )

        self.assertEqual(calibration.cost_reference_usd, 0.5)
        self.assertEqual(calibration.prior_mean[1], 1.0)

    def test_custom_score_bounds_feed_the_common_plugin_mean(self):
        calibration = fit_empirical_bayes_warm_start(
            [[-1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            score_bounds=(-1.0, 1.0),
        )

        self.assertAlmostEqual(calibration.prior_mean[0], 0.5)

    def test_invalid_warm_start_data_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            fit_empirical_bayes_warm_start([[0.5, 0.5]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            fit_empirical_bayes_warm_start([[0.5]], [[-1.0]])
        with self.assertRaisesRegex(ValueError, "finite values"):
            fit_empirical_bayes_warm_start([[math.nan]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "configured bounds"):
            fit_empirical_bayes_warm_start([[1.1]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "zero median"):
            fit_empirical_bayes_warm_start([[0.5]], [[0.0]])
        with self.assertRaisesRegex(ValueError, "arm_ids must be unique"):
            fit_empirical_bayes_warm_start(
                [[0.5], [0.5]],
                [[1.0], [1.0]],
                arm_ids=("same", "same"),
            )
        with self.assertRaisesRegex(ValueError, "question_id must be an integer"):
            fit_empirical_bayes_warm_start(
                [[0.5]],
                [[1.0]],
                question_ids=(1.5,),
            )


class PerArmQuestionScheduleTests(unittest.TestCase):
    def test_ragged_available_constructor_uses_shared_prefix_and_arm_tails(self):
        available = {
            "a": (0, 1, 2, 3, 4, 5),
            "b": (0, 1, 2, 3, 4),
            "c": (0, 1, 2, 3),
        }
        schedule = PerArmQuestionSchedule.create_from_available(
            available,
            warm_start_batch_size=2,
            seed=7,
        )

        self.assertEqual([len(schedule.orders[x]) for x in ("a", "b", "c")], [6, 5, 4])
        warm_batches = schedule.take_uniform_warm_start()
        self.assertTrue(
            all(batch == schedule.warm_start_question_ids for batch in warm_batches.values())
        )
        for arm_id, order in schedule.orders.items():
            self.assertEqual(set(order), set(available[arm_id]))

    def test_uniform_warm_start_advances_every_cursor_exactly_once(self):
        schedule = PerArmQuestionSchedule.create(
            ("a", "b", "c"),
            n_questions=12,
            warm_start_batch_size=4,
            seed=17,
        )

        batches = schedule.take_uniform_warm_start()

        self.assertEqual(set(batches), {"a", "b", "c"})
        self.assertTrue(
            all(batch == schedule.warm_start_question_ids for batch in batches.values())
        )
        self.assertTrue(all(position == 4 for position in schedule.positions.values()))
        for arm_id in schedule.orders:
            next_batch = schedule.next_batch(arm_id, 4)
            self.assertTrue(
                set(next_batch).isdisjoint(schedule.warm_start_question_ids)
            )
        with self.assertRaisesRegex(RuntimeError, "before any arm cursor advances"):
            schedule.take_uniform_warm_start()

    def test_first_batch_is_shared_and_each_arm_cursor_is_independent(self):
        schedule = PerArmQuestionSchedule.create(
            ("a", "b", "c"),
            n_questions=12,
            warm_start_batch_size=4,
            seed=17,
        )

        first_a = schedule.next_batch("a", 4)
        # Advancing a again represents an attempted physical batch.  It must
        # not be offered again even if its evaluator later reports failure.
        second_a = schedule.next_batch("a", 3)
        first_b = schedule.next_batch("b", 4)

        self.assertEqual(first_a, schedule.warm_start_question_ids)
        self.assertEqual(first_b, schedule.warm_start_question_ids)
        self.assertEqual(len(set(first_a + second_a)), 7)
        self.assertEqual(schedule.attempted_question_ids("a"), first_a + second_a)
        self.assertEqual(schedule.remaining("a"), 5)
        self.assertEqual(schedule.remaining("b"), 8)

    def test_schedule_is_seed_reproducible_and_covers_every_question(self):
        first = PerArmQuestionSchedule.create(
            (0, 1), n_questions=20, warm_start_batch_size=8, seed=2026
        )
        second = PerArmQuestionSchedule.create(
            (0, 1), n_questions=20, warm_start_batch_size=8, seed=2026
        )

        self.assertEqual(first.orders, second.orders)
        self.assertEqual(first.warm_start_question_ids, second.warm_start_question_ids)
        for order in first.orders.values():
            self.assertEqual(set(order), set(range(20)))

    def test_invalid_sizes_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "n_questions must be an integer"):
            PerArmQuestionSchedule.create(
                ("a",),
                n_questions=10.5,
                warm_start_batch_size=4,
            )
        schedule = PerArmQuestionSchedule.create(
            ("a",), n_questions=10, warm_start_batch_size=4
        )
        with self.assertRaisesRegex(ValueError, "batch_size must be an integer"):
            schedule.next_batch("a", 2.5)


class RadialMathTests(unittest.TestCase):
    def test_balanced_direction_selects_unsupported_middle_point(self):
        direction = (0.5, 0.5)
        utilities = [
            radial_utility(point, direction)
            for point in ((1.0, 0.0), (0.4, 0.4), (0.0, 1.0))
        ]

        self.assertEqual(int(np.argmax(utilities)), 1)
        self.assertAlmostEqual(utilities[1], 0.4)

    def test_direction_scale_and_reference_point(self):
        self.assertEqual(direction_scale((0.1, 0.9)), 0.9)
        self.assertAlmostEqual(
            radial_utility((0.6, 0.8), (0.5, 0.5), reference=(0.1, 0.2)),
            0.5,
        )

    def test_expected_min_has_known_symmetric_value(self):
        observed = expected_min_of_two_normals(0.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(observed, -1.0 / math.sqrt(math.pi), places=12)

    def test_expected_min_handles_deterministic_difference(self):
        self.assertEqual(
            expected_min_of_two_normals(1.0, 1.0, 2.0, 1.0, cov12=1.0),
            1.0,
        )

    def test_expected_min_rejects_invalid_covariance(self):
        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            expected_min_of_two_normals(0.0, 1.0, 0.0, 1.0, cov12=2.0)

    def test_expected_min_uses_scale_relative_covariance_validation(self):
        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            expected_min_of_two_normals(
                0.0,
                1e-20,
                0.0,
                1e-20,
                cov12=-1e-10,
            )

        observed = expected_min_of_two_normals(0.0, 1e-20, 0.0, 1e-20)
        self.assertAlmostEqual(observed, -1e-10 / math.sqrt(math.pi), places=20)


if __name__ == "__main__":
    unittest.main()
