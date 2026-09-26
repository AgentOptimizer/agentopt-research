"""Protocol coverage for the G0-centered Gittins ablation."""
import unittest

from experiments.combined_objective.gittins_ablation_v2 import (
    BENCHMARKS,
    CONFIGURATIONS,
    FAMILIES,
    G0_CONFIGURATION,
    RUN_CONFIGURATIONS,
    SEEDS,
)
from experiments.combined_objective.run_gittins_ablation_v2_slurm_task import (
    task_mapping,
)
from experiments.combined_objective.run_two_direction_ablation import DATASETS, PAIRS


class GittinsAblationV2ProtocolTests(unittest.TestCase):
    def test_protocol_has_one_reused_g0_and_nine_new_groups(self):
        self.assertEqual(len(CONFIGURATIONS), 10)
        self.assertEqual(len(RUN_CONFIGURATIONS), 9)
        self.assertNotIn(G0_CONFIGURATION, RUN_CONFIGURATIONS)
        self.assertEqual(len(BENCHMARKS) * len(SEEDS), 160)

    def test_g0_is_the_historical_g2_protocol(self):
        settings = CONFIGURATIONS[G0_CONFIGURATION]
        self.assertEqual(settings["source_configuration"], "g2_exact_axes")
        self.assertEqual(PAIRS[settings["pair_name"]], ((0.0, 1.0), (1.0, 0.0)))
        self.assertEqual(settings["acquisition_cost_mode"], "real")
        self.assertEqual(settings["continuation_mode"], "eta_decay")
        self.assertEqual(settings["eta_decay_schedule"], "direction_stop")
        self.assertEqual(settings["direction_scheduler"], "round_robin")

    def test_requested_direction_sets_are_exact(self):
        self.assertEqual(PAIRS["quality_only"], ((1.0, 0.0),))
        self.assertEqual(PAIRS["deployment_only"], ((0.0, 1.0),))
        self.assertEqual(
            PAIRS["axes_midpoint"],
            ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)),
        )
        self.assertEqual(
            PAIRS["five_directions"],
            (
                (1.0, 0.0),
                (0.75, 0.25),
                (0.5, 0.5),
                (0.25, 0.75),
                (0.0, 1.0),
            ),
        )

    def test_overlapping_controls_are_reused_across_families(self):
        self.assertIn(G0_CONFIGURATION, FAMILIES["cost_mechanism"])
        self.assertIn(G0_CONFIGURATION, FAMILIES["directions"])
        self.assertIn(G0_CONFIGURATION, FAMILIES["continuation"])
        self.assertIn(G0_CONFIGURATION, FAMILIES["scheduler"])
        self.assertIn("g1_q_only_real_cost", FAMILIES["cost_mechanism"])
        self.assertIn("g1_q_only_real_cost", FAMILIES["directions"])
        self.assertEqual(
            set(name for family in FAMILIES.values() for name in family),
            set(CONFIGURATIONS),
        )

    def test_array_mapping_covers_each_dataset_seed_once(self):
        mappings = [task_mapping(task_id) for task_id in range(160)]
        self.assertEqual(len(set(mappings)), 160)
        self.assertEqual(set(mappings), {(b, s) for b in BENCHMARKS for s in SEEDS})

    def test_matrix_benchmarks_use_the_g0_ten_by_ten_sources(self):
        self.assertEqual(DATASETS["hotpotqa"], ("scope", "data/hotpotqa"))
        self.assertEqual(DATASETS["mathqa"], ("scope", "data/mathqa"))


if __name__ == "__main__":
    unittest.main()
