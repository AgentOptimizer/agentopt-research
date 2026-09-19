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
from agentopt.model_selection.qnehvi import (
    encode_configuration_features,
    select_qnehvi_index,
)


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

    def test_vectorized_ape_matches_scalar_definition(self):
        rng = np.random.default_rng(321)
        for n_arms in (2, 3, 9, 25):
            for _ in range(5):
                means = rng.normal(size=(n_arms, 2))
                pulls = rng.integers(1, 100, size=n_arms)
                epsilon1 = float(rng.choice([0.0, 0.05]))
                M_ij = pairwise_M(means)
                bonuses = np.asarray(
                    [
                        [
                            ape_pairwise_bonus(
                                int(pulls[i]), int(pulls[j]),
                                delta=0.1, k1=1.0,
                            )
                            for j in range(n_arms)
                        ]
                        for i in range(n_arms)
                    ]
                )
                lower = M_ij - bonuses
                np.fill_diagonal(lower, np.inf)
                expected_mask = np.min(lower, axis=1) + epsilon1 > 0.0
                np.testing.assert_array_equal(
                    ape_opt_mask(means, pulls, epsilon1=epsilon1),
                    expected_mask,
                )

                unfinished = np.flatnonzero(~expected_mask)
                if unfinished.size == 0:
                    expected_arm = int(np.argmin(pulls))
                else:
                    upper = M_ij + bonuses
                    np.fill_diagonal(upper, np.inf)
                    scores = np.min(upper[unfinished], axis=1)
                    b_t = int(unfinished[int(np.argmax(scores))])
                    c_t = int(np.argmin(lower[b_t]))
                    expected_arm = b_t if pulls[b_t] <= pulls[c_t] else c_t
                self.assertEqual(
                    ape_select_arm(means, pulls, epsilon1=epsilon1),
                    expected_arm,
                )


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

    def test_qnehvi_candidate_batching_preserves_selection(self):
        try:
            import botorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("botorch is not installed")
        features = np.arange(6, dtype=np.float64).reshape(-1, 1)
        objectives = np.asarray(
            [[0.15, 0.95], [0.35, 0.75], [0.55, 0.58],
             [0.72, 0.42], [0.86, 0.27], [0.96, 0.12]],
            dtype=np.float64,
        )
        common = dict(
            categorical_dims=[0],
            mc_samples=8,
            seed=19,
        )
        unbatched = select_qnehvi_index(
            features,
            objectives,
            features,
            candidate_batch_size=None,
            **common,
        )
        batched = select_qnehvi_index(
            features,
            objectives,
            features,
            candidate_batch_size=2,
            **common,
        )
        self.assertEqual(batched, unbatched)


if __name__ == "__main__":
    unittest.main()
