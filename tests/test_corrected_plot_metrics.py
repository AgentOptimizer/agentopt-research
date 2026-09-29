"""Regression tests for complete-returned-set plotting metrics."""

import unittest

from experiments.combined_objective.plot_three_objective_corrected_gd_igd import (
    recompute_corrected_distances,
)


class CorrectedPlotMetricTests(unittest.TestCase):
    def test_three_objective_distances_keep_dominated_returned_points(self):
        run = {
            "raw_truth_vectors": [
                [0.2, 0.0, 1.0],
                [1.0, 4.0, 1.0],
                [0.2, 4.0, 1.0],
            ],
            "points": [{"selected_arm_indices": [0, 2]}],
        }

        recompute_corrected_distances(run)

        point = run["points"][0]
        self.assertAlmostEqual(point["generational_distance"], 0.25)
        self.assertAlmostEqual(point["inverted_generational_distance"], 0.4)


if __name__ == "__main__":
    unittest.main()
