import unittest

import numpy as np

from agentopt.model_selection.pareto_identification import (
    EGE_SH,
    EGE_SR,
    ape_opt_mask,
    ape_pairwise_bonus,
    ape_select_arm,
    ege_keep_count,
    ege_select_survivors,
    ege_sequential_halving_pulls,
    ege_successive_rejects_schedule,
    empirical_gaps,
    empirical_pareto_mask,
    pairwise_M,
    pairwise_m,
)
from agentopt.model_selection.qnehvi import encode_configuration_features


class PairwiseGapTests(unittest.TestCase):
    def test_strict_dominance_signs(self):
        means = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        m_ij = pairwise_m(means)
        M_ij = pairwise_M(means)
        self.assertGreater(m_ij[0, 1], 0.0)
        self.assertLess(M_ij[0, 1], 0.0)
        self.assertLess(m_ij[1, 0], 0.0)
        self.assertGreater(M_ij[1, 0], 0.0)

    def test_empirical_pareto_is_the_nondominated_set(self):
        means = np.array(
            [
                [0.9, 0.3],
                [0.7, 0.5],
                [0.5, 0.8],
                [0.4, 0.2],
            ],
            dtype=np.float64,
        )
        mask = empirical_pareto_mask(means)
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_suboptimal_gap_is_the_largest_dominance_margin(self):
        means = np.array([[1.0, 1.0], [0.4, 0.7]], dtype=np.float64)
        gaps = empirical_gaps(means)
        # Arm 1 is strictly dominated by 0.3 and 0.3; m(1, 0) = 0.3.
        self.assertAlmostEqual(gaps[1] - 1e-7, 0.3)
        self.assertGreater(gaps[0], 0.0)


class EGEScheduleTests(unittest.TestCase):
    def test_successive_rejects_uses_k_minus_one_rounds(self):
        schedule = ege_successive_rejects_schedule(5, 40)
        self.assertEqual(len(schedule), 4)
        self.assertTrue(all(n >= 0 for n in schedule))

    def test_sequential_halving_keeps_half(self):
        self.assertEqual(ege_keep_count(81, EGE_SH), 41)
        self.assertEqual(ege_keep_count(5, EGE_SR), 4)
        self.assertGreater(ege_sequential_halving_pulls(81, 81, 15390), 0)

    def test_survivors_discard_the_largest_gap_first(self):
        means = np.array(
            [
                [0.9, 0.3],
                [0.7, 0.5],
                [0.5, 0.8],
                [0.2, 0.1],
            ],
            dtype=np.float64,
        )
        survivors, accepted, rejected = ege_select_survivors(
            means, [0, 1, 2, 3], n_keep=3
        )
        self.assertEqual(set(survivors), {0, 1, 2})
        self.assertEqual(rejected, (3,))
        self.assertEqual(accepted, ())


class APESamplingTests(unittest.TestCase):
    def test_bonus_decreases_with_more_pulls(self):
        wide = ape_pairwise_bonus(2, 2, delta=0.1, k1=1.0)
        tight = ape_pairwise_bonus(50, 50, delta=0.1, k1=1.0)
        self.assertGreater(wide, tight)
        self.assertGreater(tight, 0.0)

    def test_well_separated_front_is_eventually_certified(self):
        means = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.1, 0.1],
            ],
            dtype=np.float64,
        )
        pulls = [400, 400, 400]
        certified = ape_opt_mask(means, pulls, epsilon1=0.0, delta=0.1, k1=1.0)
        self.assertTrue(certified[0])
        self.assertTrue(certified[1])
        self.assertFalse(certified[2])

    def test_select_arm_returns_a_valid_index(self):
        means = np.array(
            [[0.9, 0.2], [0.6, 0.6], [0.2, 0.9], [0.3, 0.3]],
            dtype=np.float64,
        )
        pulls = [3, 3, 3, 2]
        arm = ape_select_arm(means, pulls)
        self.assertIn(arm, range(4))


class ConfigurationFeatureTests(unittest.TestCase):
    def test_two_role_names_become_two_categoricals(self):
        names = [
            "planner=A + solver=X",
            "planner=A + solver=Y",
            "planner=B + solver=X",
        ]
        features, dims = encode_configuration_features(names)
        self.assertEqual(features.shape, (3, 2))
        self.assertEqual(dims, [0, 1])
        self.assertEqual(features[0, 0], features[1, 0])
        self.assertNotEqual(features[0, 1], features[1, 1])

    def test_flat_model_names_use_one_categorical(self):
        features, dims = encode_configuration_features(["haiku", "opus", "haiku"])
        self.assertEqual(features.shape, (3, 1))
        self.assertEqual(dims, [0])
        self.assertEqual(features[0, 0], features[2, 0])


if __name__ == "__main__":
    unittest.main()
