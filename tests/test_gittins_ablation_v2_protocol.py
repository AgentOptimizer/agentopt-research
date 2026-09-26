"""Protocol coverage for the G0-centered Gittins ablation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.combined_objective import gittins_ablation_v2 as protocol
from experiments.combined_objective import run_gittins_ablation_v2_slurm_task as task
from experiments.combined_objective import run_two_direction_ablation as runner

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
    def test_protocol_can_run_g0_and_all_nine_ablation_groups(self):
        self.assertEqual(len(CONFIGURATIONS), 10)
        self.assertEqual(len(RUN_CONFIGURATIONS), 10)
        self.assertIn(G0_CONFIGURATION, RUN_CONFIGURATIONS)
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

    def test_only_current_ablation_direction_sets_remain(self):
        self.assertEqual(set(PAIRS), {settings["pair_name"] for settings in CONFIGURATIONS.values()})
        self.assertEqual(runner.PRIMARY_PAIR, "exact_axes")

    def test_g0_results_prefer_new_runs_and_fall_back_to_historical_g2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / G0_CONFIGURATION / "seed-43/exact_axes/mathqa/result.json"
            historical = root / "historical/seed-43/exact_axes/mathqa/result.json"
            with mock.patch.object(protocol, "G0_SOURCE_ROOT", root / "historical"):
                self.assertEqual(protocol.result_path(G0_CONFIGURATION, "mathqa", 43, output_root=root), historical)
                local.parent.mkdir(parents=True)
                local.write_text("{}")
                self.assertEqual(protocol.result_path(G0_CONFIGURATION, "mathqa", 43, output_root=root), local)

    def test_all_protocol_commands_are_accepted_by_the_current_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, settings in CONFIGURATIONS.items():
                with self.subTest(configuration=name):
                    output = io.StringIO()
                    argv = ["task", "--configuration", name, "--task-id", "159",
                            "--output-root", directory, "--dry-run"]
                    with mock.patch("sys.argv", argv), contextlib.redirect_stdout(output):
                        task.main()
                    mapping = json.loads(output.getvalue())
                    self.assertEqual(mapping["benchmark"], BENCHMARKS[-1])
                    self.assertEqual(mapping["seed"], SEEDS[-1])
                    command = mapping["command"]
                    self.assertNotIn("--eta-decay-schedule", command)
                    with mock.patch("sys.argv", command[2:]), mock.patch.object(runner, "run_one") as run:
                        runner.main()
                    options = run.call_args.kwargs
                    self.assertEqual(run.call_args.args[1], settings["pair_name"])
                    for key in ("direction_scheduler", "acquisition_cost_mode", "continuation_mode"):
                        self.assertEqual(options[key], settings[key])

    def test_g0_can_run_an_arbitrary_seed_without_an_external_g0_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / G0_CONFIGURATION / "seed-47/exact_axes/restaurant_test/result.json"

            def run_command(command, **kwargs):
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"config": {"wall_time_seconds": 0.1}, "run": {}}))
                return SimpleNamespace(returncode=0)

            argv = ["task", "--configuration", G0_CONFIGURATION,
                    "--task-id", "5", "--output-root", directory]
            with mock.patch("sys.argv", argv), mock.patch.object(task.subprocess, "run", side_effect=run_command), \
                    mock.patch.object(task, "git_output", return_value="test"), \
                    mock.patch.object(task.platform, "platform", return_value="test-platform"), \
                    mock.patch.object(task, "protocol_result_path", side_effect=AssertionError("G0 must bootstrap itself")), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                task.main()
            self.assertEqual(stopped.exception.code, 0)
            metadata = json.loads(result_path.with_name("slurm_task_metadata.json").read_text())
            self.assertEqual(metadata["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
