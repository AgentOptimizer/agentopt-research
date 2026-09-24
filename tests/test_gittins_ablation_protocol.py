"""Protocol coverage for the eight-group Gittins ablation."""

import math
import unittest

from experiments.combined_objective import offline_radial_gittins as replay
from experiments.combined_objective.run_gittins_ablation_slurm_task import (
    BENCHMARKS,
    RUN_CONFIGURATIONS,
    SEEDS,
    task_mapping,
)
from experiments.combined_objective.run_two_direction_ablation import (
    ABLATION_CONFIGS,
    PAIRS,
)


class GittinsAblationProtocolTests(unittest.TestCase):
    def test_protocol_has_one_existing_baseline_and_seven_new_groups(self):
        self.assertEqual(len(ABLATION_CONFIGS), 8)
        self.assertEqual(len(RUN_CONFIGURATIONS), 7)
        self.assertNotIn("g0_gauss_radau", RUN_CONFIGURATIONS)
        self.assertEqual(len(BENCHMARKS) * len(SEEDS), 160)

    def test_gauss_legendre_nodes_are_exact_two_point_nodes(self):
        left = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))
        self.assertEqual(
            PAIRS["gauss_legendre_two_point"],
            ((left, 1.0 - left), (1.0 - left, left)),
        )

    def test_cost_endpoint_pair_mirrors_the_primary_gauss_radau_pair(self):
        endpoint, interior = PAIRS["gauss_radau_cost_endpoint"]
        self.assertEqual(endpoint, (0.0, 1.0))
        self.assertAlmostEqual(interior[0], 2.0 / 3.0)
        self.assertAlmostEqual(interior[1], 1.0 / 3.0)

    def test_dense_grid_has_nine_interiors_and_accuracy_endpoint(self):
        directions = PAIRS["dense_grid_10"]
        self.assertEqual(len(directions), 10)
        self.assertEqual(directions[0], (0.1, 0.9))
        self.assertEqual(directions[-1], (1.0, 0.0))

    def test_array_mapping_covers_each_benchmark_and_seed_once(self):
        mappings = [task_mapping(task_id) for task_id in range(160)]
        self.assertEqual(len(set(mappings)), 160)
        self.assertEqual(set(mappings), {(b, s) for b in BENCHMARKS for s in SEEDS})

    def test_weighted_scheduler_visits_interior_three_times_then_endpoint(self):
        scheduler = replay._DirectionScheduler(
            ((1.0 / 3.0, 2.0 / 3.0), (1.0, 0.0)),
            "weighted_round_robin_3_to_1",
        )
        visits = [scheduler.direction_index]
        for _ in range(7):
            visits.append(scheduler.advance())
        self.assertEqual(visits, [0, 0, 0, 1, 0, 0, 0, 1])

    def test_weighted_scheduler_rejects_non_two_direction_design(self):
        with self.assertRaisesRegex(ValueError, "exactly two directions"):
            replay._DirectionScheduler(
                ((0.2, 0.8), (0.5, 0.5), (1.0, 0.0)),
                "weighted_round_robin_3_to_1",
            )


if __name__ == "__main__":
    unittest.main()
