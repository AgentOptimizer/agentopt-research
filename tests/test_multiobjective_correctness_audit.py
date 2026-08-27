import unittest

import numpy as np

from experiments.combined_objective.audit_multiobjective_results import (
    dominates,
    nearest_front_distance,
)
from experiments.combined_objective.offline_multiobjective_random_search import (
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_radial_gittins import (
    generational_distance,
    hypervolume_2d,
    inverted_generational_distance,
)


class MultiObjectiveCorrectnessAuditTests(unittest.TestCase):
    def test_dominance_requires_weak_better_both_and_strict_better_one(self):
        self.assertTrue(dominates(np.array([0.8, 0.1]), np.array([0.7, 0.2])))
        self.assertTrue(dominates(np.array([0.7, 0.1]), np.array([0.7, 0.2])))
        self.assertFalse(dominates(np.array([0.7, 0.2]), np.array([0.7, 0.2])))
        self.assertFalse(dominates(np.array([0.6, 0.1]), np.array([0.7, 0.2])))

    def test_nondominated_set_excludes_known_dominated_point(self):
        points = np.array([[0.9, 0.4], [0.8, 0.2], [0.7, 0.3], [0.6, 0.1]])
        self.assertEqual(pareto_min_cost_indices(points), [0, 1, 3])

    def test_distance_to_front_is_zero_for_front_point(self):
        front = np.array([[0.6, 0.1], [0.8, 0.2]])
        ranges = np.array([0.2, 0.1])
        self.assertAlmostEqual(nearest_front_distance(front[0], front, ranges), 0.0)
        self.assertGreater(nearest_front_distance(np.array([0.5, 0.2]), front, ranges), 0.0)

    def test_hypervolume_gap_is_zero_for_identical_front(self):
        front = np.array([[0.5, 0.9], [0.8, 0.6]])
        true_hv = hypervolume_2d(front)
        returned_hv = hypervolume_2d(front.copy())
        self.assertAlmostEqual(true_hv - returned_hv, 0.0)
        self.assertAlmostEqual(generational_distance(front, front), 0.0)
        self.assertAlmostEqual(inverted_generational_distance(front, front), 0.0)


if __name__ == "__main__":
    unittest.main()
