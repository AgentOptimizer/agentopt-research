import unittest

import numpy as np

from experiments.combined_objective.plot_20seed_eight_dataset_curves import (
    RunTrajectory,
    aggregate_random_checkpoints,
    aggregate_random_trajectories,
)


class RandomStartAlignmentTest(unittest.TestCase):
    def test_checkpoint_aggregation_uses_mean_realized_cost_and_all_y_values(self):
        runs = (
            RunTrajectory(
                1,
                np.asarray([0.10, 0.25, 1.0]),
                {"hv_regret": np.asarray([10.0, 8.0, 0.0])},
            ),
            RunTrajectory(
                2,
                np.asarray([0.20, 0.35, 1.0]),
                {"hv_regret": np.asarray([20.0, 12.0, 0.0])},
            ),
            RunTrajectory(
                3,
                np.asarray([0.30, 0.45, 1.0]),
                {"hv_regret": np.asarray([30.0, 18.0, 0.0])},
            ),
        )

        xs, means, _, counts = aggregate_random_checkpoints(runs, "hv_regret")

        np.testing.assert_allclose(xs, [0.20, 0.35, 1.0])
        np.testing.assert_allclose(means, [20.0, 38.0 / 3.0, 0.0])
        np.testing.assert_array_equal(counts, [3, 3, 3])

    def test_mean_start_uses_all_runs_with_left_extension(self):
        runs = (
            RunTrajectory(
                1,
                np.asarray([0.10, 0.20, 1.0]),
                {"hv_regret": np.asarray([10.0, 8.0, 0.0])},
            ),
            RunTrajectory(
                2,
                np.asarray([0.20, 0.30, 1.0]),
                {"hv_regret": np.asarray([20.0, 12.0, 0.0])},
            ),
            RunTrajectory(
                3,
                np.asarray([0.30, 0.40, 1.0]),
                {"hv_regret": np.asarray([30.0, 18.0, 0.0])},
            ),
        )

        xs, means, _, counts, start = aggregate_random_trajectories(
            runs, "hv_regret", grid_step=0.10
        )

        self.assertAlmostEqual(start, 0.20)
        self.assertAlmostEqual(xs[0], 0.20)
        # Run 1 uses its second saved value, run 2 its first, and the later
        # run 3 extends its first value left to the shared mean start.
        self.assertAlmostEqual(means[0], (8.0 + 20.0 + 30.0) / 3.0)
        np.testing.assert_array_equal(counts, np.full(len(xs), 3))

    def test_start_can_fall_between_regular_grid_points(self):
        runs = (
            RunTrajectory(
                1,
                np.asarray([0.11, 1.0]),
                {"hv_regret": np.asarray([1.0, 0.0])},
            ),
            RunTrajectory(
                2,
                np.asarray([0.12, 1.0]),
                {"hv_regret": np.asarray([3.0, 0.0])},
            ),
        )

        xs, means, _, counts, start = aggregate_random_trajectories(
            runs, "hv_regret", grid_step=0.05
        )

        self.assertAlmostEqual(start, 0.115)
        self.assertAlmostEqual(xs[0], 0.115)
        self.assertAlmostEqual(xs[1], 0.15)
        self.assertAlmostEqual(means[0], 2.0)
        self.assertTrue(np.all(counts == 2))


if __name__ == "__main__":
    unittest.main()
