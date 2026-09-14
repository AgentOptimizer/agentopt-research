"""Raw-cost inference and affine rewards must describe the same Gaussian state."""
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import (
    GaussianVectorPosterior,
    ObjectiveNormalizer,
    fit_empirical_bayes_warm_start,
)


SCORES = np.asarray(((0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 0, 1)), dtype=float)
COSTS = np.asarray(((1, 2, 3, 4), (2, 4, 6, 8), (5, 4, 3, 2)), dtype=float)


def calibration(costs=COSTS, **options):
    options.setdefault("cost_model", "raw_mean")
    return fit_empirical_bayes_warm_start(SCORES, costs, **options)


class RawCostCalibrationTests(unittest.TestCase):
    def test_calibration_uses_raw_sample_variances_and_preserves_accuracy_options(self):
        fitted = calibration(prior_variance=(0.08, 99), obs_noise_variance=(0.125, 77))
        self.assertEqual(fitted.cost_model, "raw_mean")
        self.assertEqual(fitted.cost_reference_usd, 5)
        np.testing.assert_allclose(fitted.batch_observations, ((0.5, 2.5), (0.5, 5), (0.25, 3.5)))
        np.testing.assert_allclose(fitted.prior_mean, (5 / 12, 11 / 3))
        np.testing.assert_allclose(fitted.prior_var, (0.08, 19 / 12))
        np.testing.assert_allclose(fitted.warm_obs_noise_var, (0.125, 5 / 6))
        self.assertAlmostEqual(fitted.raw_cost_prior_variance_estimate_usd2, 19 / 12)
        self.assertAlmostEqual(fitted.raw_cost_observation_variance_estimate_usd2, 10 / 3)
        self.assertAlmostEqual(fitted.raw_cost_variance_floor_usd2, 25e-12)
        posterior = fitted.initialize_posteriors()[0]
        self.assertAlmostEqual(posterior.mean[1], 505 / 174)
        self.assertAlmostEqual(posterior.var[1], 95 / 174)
        self.assertEqual((posterior.n_batches, posterior.n_questions), (1, 4))

    def test_reward_posterior_matches_a_conjugate_update_in_affine_coordinates(self):
        fitted = calibration()
        for arm, posterior in fitted.initialize_posteriors().items():
            reference = fitted.cost_reference_usd
            reward_prior = GaussianVectorPosterior(
                mean=(fitted.prior_mean[0], 1 - fitted.prior_mean[1] / reference),
                var=(fitted.prior_var[0], fitted.prior_var[1] / reference ** 2),
            )
            reward_prior.update(
                fitted.normalizer.normalize_batch(SCORES[arm], COSTS[arm]).mean(axis=0),
                fitted.reward_obs_noise_var, batch_size=4,
            )
            reward = fitted.reward_posterior(posterior)
            np.testing.assert_allclose(reward.mean, reward_prior.mean, atol=1e-15)
            np.testing.assert_allclose(reward.var, reward_prior.var)
            self.assertEqual((reward.n_batches, reward.n_questions), (1, 4))
            old_raw_mean = posterior.mean.copy()
            reward.mean[:] = 99
            np.testing.assert_array_equal(posterior.mean, old_raw_mean)

    def test_later_batch_partitions_have_identical_raw_and_reward_posteriors(self):
        fitted = calibration()
        scores = np.asarray((0.1, 0.9, 0.2, 0.8, 0.5, 0.7))
        costs = np.asarray((1, 20, 2, 4, 9, 5))
        observations = fitted.posterior_observations(scores, costs)
        np.testing.assert_array_equal(observations[:, 1], costs)
        per_question_noise = fitted.warm_obs_noise_var * fitted.batch_size
        posterior_one = fitted.initialize_posteriors()[0]
        posterior_one.update(observations.mean(axis=0), per_question_noise / 6, batch_size=6)
        posterior_split = fitted.initialize_posteriors()[0]
        reward_split = fitted.reward_posterior(posterior_split)
        offset = 0
        for count in (2, 1, 3):
            section = slice(offset, offset + count)
            posterior_split.update(observations[section].mean(axis=0), per_question_noise / count,
                                   batch_size=count)
            reward_split.update(
                fitted.normalizer.normalize_batch(scores[section], costs[section]).mean(axis=0),
                fitted.reward_obs_noise_var * fitted.batch_size / count, batch_size=count,
            )
            offset += count
        np.testing.assert_allclose(posterior_split.mean, posterior_one.mean, atol=1e-14)
        np.testing.assert_allclose(posterior_split.var, posterior_one.var)
        np.testing.assert_allclose(reward_split.mean, fitted.reward_posterior(posterior_split).mean, atol=1e-14)
        np.testing.assert_allclose(reward_split.var, fitted.reward_posterior(posterior_split).var)
        self.assertEqual(posterior_one.n_questions, posterior_split.n_questions)
        self.assertEqual(fitted.cost_reference_usd, 5)
        np.testing.assert_allclose(fitted.prior_var[1], 19 / 12)
        with self.assertRaisesRegex(ValueError, "read-only"):
            fitted.warm_obs_noise_var[1] = 999

    def test_currency_unit_change_preserves_reward_inference(self):
        dollars, cents = calibration(), calibration(COSTS * 100)
        self.assertEqual(cents.cost_reference_usd, dollars.cost_reference_usd * 100)
        self.assertAlmostEqual(cents.prior_mean[1], dollars.prior_mean[1] * 100)
        self.assertAlmostEqual(cents.prior_var[1], dollars.prior_var[1] * 100 ** 2)
        self.assertAlmostEqual(cents.warm_obs_noise_var[1], dollars.warm_obs_noise_var[1] * 100 ** 2)
        self.assertAlmostEqual(cents.raw_cost_variance_floor_usd2, dollars.raw_cost_variance_floor_usd2 * 100 ** 2)
        for arm, usd in dollars.initialize_posteriors().items():
            cent = cents.initialize_posteriors()[arm]
            np.testing.assert_allclose(dollars.reward_posterior(usd).mean, cents.reward_posterior(cent).mean)
            np.testing.assert_allclose(dollars.reward_posterior(usd).var, cents.reward_posterior(cent).var)
        np.testing.assert_allclose(dollars.reward_obs_noise_var, cents.reward_obs_noise_var)

    def test_affine_reward_repairs_nonlinear_cost_order_reversal_without_clipping(self):
        scores = np.full((2, 4), 0.8)
        costs = np.asarray(((0, 0, 0, 10), (1, 1, 1, 1)), dtype=float)
        old = fit_empirical_bayes_warm_start(scores, costs)
        raw = fit_empirical_bayes_warm_start(scores, costs, cost_model="raw_mean")
        self.assertGreater(costs[0].mean(), costs[1].mean())
        self.assertGreater(old.batch_observations[0, 1], old.batch_observations[1, 1])
        rewards = [raw.normalizer.normalize_batch(scores[arm], costs[arm]) for arm in (0, 1)]
        np.testing.assert_allclose(rewards[0][:, 1], (1, 1, 1, -3))
        self.assertLess(rewards[0][:, 1].mean(), rewards[1][:, 1].mean())
        posteriors = raw.initialize_posteriors()
        self.assertLess(raw.reward_posterior(posteriors[0]).mean[1], raw.reward_posterior(posteriors[1]).mean[1])
        self.assertEqual(raw.normalizer.inverse_deployment_cost(-3), 10)
        self.assertEqual(raw.normalizer.normalize_deployment_cost(-2.5), 2)
        self.assertEqual(raw.normalizer.inverse_deployment_cost(2), -2.5)
        predicted = GaussianVectorPosterior(mean=(0.5, -2.5), var=(0.04, 0.1), n_batches=7, n_questions=21)
        reward = raw.reward_posterior(predicted)
        self.assertEqual(reward.mean[1], 2)
        self.assertEqual((reward.n_batches, reward.n_questions), (7, 21))

    def test_variance_floor_handles_constant_costs_and_one_arm_or_one_question(self):
        for costs in (np.full((1, 1), 2.0), np.full((2, 4), 2.0),
                      np.asarray(((1, 2, 3, 4),)), np.asarray(((2,), (4,)))):
            with self.subTest(shape=costs.shape):
                fitted = fit_empirical_bayes_warm_start(np.full_like(costs, 0.5), costs, cost_model="raw_mean")
                floor = 1e-12 * fitted.cost_reference_usd ** 2
                if len(costs) == 1 or np.all(costs == costs.flat[0]):
                    self.assertEqual(fitted.prior_var[1], floor)
                if costs.shape[1] == 1 or np.all(costs == costs.flat[0]):
                    self.assertEqual(fitted.warm_obs_noise_var[1], floor / costs.shape[1])
                self.assertGreater(fitted.prior_var[1], 0)
                self.assertGreater(fitted.warm_obs_noise_var[1], 0)
                for posterior in fitted.initialize_posteriors().values():
                    self.assertTrue(np.all(np.isfinite(posterior.mean)))
                    self.assertTrue(np.all(posterior.var > 0))

    def test_legacy_helpers_keep_reciprocal_semantics_and_independent_copies(self):
        fitted = fit_empirical_bayes_warm_start(SCORES, COSTS)
        self.assertEqual(fitted.cost_model, "reciprocal")
        self.assertEqual(fitted.cost_reference_usd, 3.5)
        np.testing.assert_array_equal(fitted.posterior_observations(SCORES[0], COSTS[0]),
                                      fitted.normalizer.normalize_batch(SCORES[0], COSTS[0]))
        posterior = fitted.initialize_posteriors()[0]
        copied = fitted.reward_posterior(posterior)
        np.testing.assert_array_equal(copied.mean, posterior.mean)
        np.testing.assert_array_equal(copied.var, posterior.var)
        np.testing.assert_array_equal(fitted.reward_obs_noise_var, fitted.warm_obs_noise_var)
        self.assertFalse(np.shares_memory(copied.mean, posterior.mean))
        self.assertEqual(fitted.normalizer.inverse_deployment_cost(0.5), 3.5)
        self.assertIsNone(fitted.normalizer.inverse_deployment_cost(0))
        self.assertIsNone(fitted.normalizer.inverse_deployment_cost(-0.1))

    def test_zero_costs_require_explicit_positive_scale_and_invalid_models_fail(self):
        with self.assertRaisesRegex(ValueError, "zero maximum"):
            fit_empirical_bayes_warm_start([[0.5]], [[0]], cost_model="raw_mean")
        fitted = fit_empirical_bayes_warm_start([[0.5]], [[0]], cost_model="raw_mean", cost_reference_usd=1)
        self.assertEqual(fitted.prior_var[1], 1e-12)
        with self.assertRaisesRegex(ValueError, "cost_model"):
            ObjectiveNormalizer(1, cost_model="unknown")
        with self.assertRaisesRegex(ValueError, "cost_model"):
            calibration(cost_model="unknown")


if __name__ == "__main__":
    unittest.main()
