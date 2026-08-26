import math
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import radial_utility
from agentopt.model_selection.radial_gittins_dp import (
    terminal_expected_radial_utility,
)
from agentopt.model_selection.radial_ucb import (
    BATCH_COUNT_BONUS,
    POSTERIOR_SD_BONUS,
    RadialUCBPolicy,
    batch_count_bonus,
    posterior_sd_bonus,
    radial_ucb_index,
)


class BonusTests(unittest.TestCase):
    def test_posterior_bonus_is_beta_standard_deviations(self):
        np.testing.assert_allclose(
            posterior_sd_bonus((0.04, 0.09), exploration_beta=2.0),
            (0.4, 0.6),
        )

    def test_batch_count_bonus_matches_matrix_ucb_radius(self):
        # matrix_ucb uses ``mu + sqrt(a / count)``; beta**2 plays the role of a.
        beta = 3.0
        np.testing.assert_allclose(
            batch_count_bonus(9, exploration_beta=beta),
            np.full(2, math.sqrt(beta**2 / 9.0)),
        )

    def test_batch_count_bonus_rejects_an_unobserved_arm(self):
        with self.assertRaises(ValueError):
            batch_count_bonus(0)

    def test_zero_beta_removes_optimism(self):
        policy = RadialUCBPolicy(exploration_beta=0.0)
        np.testing.assert_allclose(
            policy.optimistic_objective((0.3, 0.7), (0.04, 0.04)),
            (0.3, 0.7),
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            RadialUCBPolicy(bonus_mode="thompson")
        with self.assertRaises(ValueError):
            RadialUCBPolicy(exploration_beta=-1.0)
        with self.assertRaises(ValueError):
            RadialUCBPolicy(bonus_mode=BATCH_COUNT_BONUS).index(
                (0.3, 0.7), (0.04, 0.04), (0.5, 0.5)
            )


class RadialIndexTests(unittest.TestCase):
    def test_index_is_the_radial_utility_of_the_upper_confidence_limit(self):
        # The index uses a closed-form scalar path instead of calling
        # radial_utility, so this pins the two definitions together.
        policy = RadialUCBPolicy(exploration_beta=1.5)
        for direction in ((0.1, 0.9), (0.2, 0.8), (0.5, 0.5), (0.9, 0.1)):
            for reference in ((0.0, 0.0), (0.1, 0.25)):
                for mean, var in (
                    ((0.3, 0.7), (0.04, 0.01)),
                    ((0.85, 0.2), (0.0025, 0.09)),
                    ((0.5, 0.5), (0.02, 0.02)),
                ):
                    with self.subTest(
                        direction=direction,
                        reference=reference,
                        mean=mean,
                    ):
                        expected = radial_utility(
                            np.asarray(mean) + 1.5 * np.sqrt(var),
                            direction,
                            reference,
                        )
                        self.assertAlmostEqual(
                            policy.index(mean, var, direction, reference),
                            expected,
                        )

    def test_index_accepts_arrays_scalars_and_sequences_alike(self):
        policy = RadialUCBPolicy(exploration_beta=1.0)
        expected = policy.index((0.4, 0.4), (0.01, 0.01), (0.5, 0.5))
        for mean, var in (
            (np.asarray((0.4, 0.4)), np.asarray((0.01, 0.01))),
            ([0.4, 0.4], [0.01, 0.01]),
            (0.4, 0.01),
        ):
            with self.subTest(mean=mean):
                self.assertAlmostEqual(
                    policy.index(mean, var, (0.5, 0.5)),
                    expected,
                )

    def test_malformed_vectors_are_rejected(self):
        policy = RadialUCBPolicy()
        with self.assertRaises(ValueError):
            policy.index((0.1, 0.2, 0.3), (0.01, 0.01), (0.5, 0.5))
        with self.assertRaises(ValueError):
            policy.index((0.4, float("nan")), (0.01, 0.01), (0.5, 0.5))
        with self.assertRaises(ValueError):
            policy.index((0.4, 0.4), (0.01, 0.0), (0.5, 0.5))
        with self.assertRaises(ValueError):
            policy.index((0.4, 0.4), (0.01, 0.01), (0.5, 0.5), (0.0, 0.0),
                         effective_pull_cost=-1.0)

    def test_index_is_the_box_maximum_of_the_radial_utility(self):
        mean = np.asarray((0.45, 0.55))
        var = np.asarray((0.01, 0.04))
        direction = (0.3, 0.7)
        beta = 2.0
        index = radial_ucb_index(
            mean,
            var,
            direction,
            exploration_beta=beta,
        )
        radius = beta * np.sqrt(var)
        grid = [
            np.asarray((mean[0] + a * radius[0], mean[1] + b * radius[1]))
            for a in np.linspace(-1.0, 1.0, 17)
            for b in np.linspace(-1.0, 1.0, 17)
        ]
        for point in grid:
            self.assertLessEqual(
                radial_utility(point, direction),
                index + 1e-12,
            )
        self.assertAlmostEqual(
            index,
            max(radial_utility(point, direction) for point in grid),
        )

    def test_index_dominates_the_terminal_value_of_the_same_posterior(self):
        # A completed arm is scored by E[min(X1, X2)], which never exceeds the
        # optimistic index; otherwise a direction could stop before an
        # unfinished arm with an identical posterior was ever compared.
        for direction in ((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)):
            with self.subTest(direction=direction):
                mean = (0.4, 0.6)
                var = (0.02, 0.03)
                self.assertGreater(
                    radial_ucb_index(mean, var, direction),
                    terminal_expected_radial_utility(mean, var, direction),
                )

    def test_index_increases_with_beta_and_decreases_with_pull_cost(self):
        mean = (0.4, 0.6)
        var = (0.02, 0.02)
        direction = (0.5, 0.5)
        indices = [
            radial_ucb_index(mean, var, direction, exploration_beta=beta)
            for beta in (0.0, 1.0, 2.0)
        ]
        self.assertEqual(indices, sorted(indices))
        penalized = radial_ucb_index(
            mean,
            var,
            direction,
            exploration_beta=1.0,
            effective_pull_cost=0.25,
        )
        self.assertAlmostEqual(penalized, indices[1] - 0.25)

    def test_direction_scaling_multiplies_each_confidence_radius(self):
        direction = np.asarray((0.1, 0.9))
        mean = np.asarray((0.1, 0.9))
        var = np.asarray((0.01, 0.0025))
        beta = 2.0
        factors = np.max(direction) / direction
        self.assertAlmostEqual(
            radial_ucb_index(mean, var, direction, exploration_beta=beta),
            float(np.min(factors * (mean + beta * np.sqrt(var)))),
        )

    def test_only_the_binding_scaled_coordinate_drives_optimism(self):
        # Under direction (0.1, 0.9) the first objective is scaled by 9, so it
        # sits far above the minimum and its uncertainty cannot move the index.
        direction = (0.1, 0.9)
        mean = (0.5, 0.5)
        base = radial_ucb_index(mean, (1e-8, 1e-8), direction)
        first_uncertain = radial_ucb_index(mean, (0.04, 1e-8), direction)
        second_uncertain = radial_ucb_index(mean, (1e-8, 0.04), direction)
        self.assertAlmostEqual(first_uncertain, base, places=3)
        self.assertGreater(second_uncertain, base + 0.3)

    def test_reference_point_shifts_the_index(self):
        mean = (0.6, 0.6)
        var = (0.01, 0.01)
        direction = (0.5, 0.5)
        self.assertAlmostEqual(
            radial_ucb_index(mean, var, direction, (0.2, 0.2)),
            radial_ucb_index(mean, var, direction) - 0.2,
        )

    def test_malformed_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            radial_ucb_index((0.5, 0.5), (0.01, 0.01), (0.4, 0.4))
        with self.assertRaises(ValueError):
            radial_ucb_index((0.5, 0.5), (0.01, 0.01), (0.0, 1.0))

    def test_batch_count_mode_ignores_the_posterior_variance(self):
        direction = (0.4, 0.6)
        first = radial_ucb_index(
            (0.3, 0.7),
            (0.04, 0.04),
            direction,
            bonus_mode=BATCH_COUNT_BONUS,
            n_batches=4,
        )
        second = radial_ucb_index(
            (0.3, 0.7),
            (1e-6, 1e-6),
            direction,
            bonus_mode=BATCH_COUNT_BONUS,
            n_batches=4,
        )
        self.assertAlmostEqual(first, second)
        self.assertGreater(
            first,
            radial_ucb_index(
                (0.3, 0.7),
                (0.04, 0.04),
                direction,
                bonus_mode=BATCH_COUNT_BONUS,
                n_batches=16,
            ),
        )

    def test_default_mode_is_the_posterior_standard_deviation(self):
        self.assertEqual(RadialUCBPolicy().bonus_mode, POSTERIOR_SD_BONUS)


if __name__ == "__main__":
    unittest.main()
