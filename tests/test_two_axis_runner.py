"""The current runner preserves checkpoint evidence and rejects stale reuse."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agentopt.model_selection.radial_gittins_dp import RadialGittinsBoundaryCache
from experiments.combined_objective import run_two_direction_ablation as runner
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(question_count=40):
    models = ["cheap", "accurate"]
    questions = list(range(question_count))
    table = {
        model: {
            question: SampleResult(
                score=(0.6, 0.9)[arm], cost=(0.01, 0.02)[arm],
                latency_seconds=0.0, input_tokens={}, output_tokens={},
            )
            for question in questions
        }
        for arm, model in enumerate(models)
    }
    return models, questions, table, {"matrix": "current-input"}


class TwoAxisRunnerTests(unittest.TestCase):
    def _run(self, directory, *, stop_early=False, **settings):
        simulate = runner.simulate_radial_gittins

        def run_with_index(*args, **kwargs):
            kwargs["index_provider"] = lambda context, arm: -1e8 if stop_early else 1e8 - arm
            return simulate(*args, **kwargs)

        with mock.patch.object(runner, "load_benchmark", return_value=_problem(100 if stop_early else 40)), \
                mock.patch.object(runner, "RadialGittinsBoundaryCache", return_value=RadialGittinsBoundaryCache()), \
                mock.patch.object(runner, "simulate_radial_gittins", side_effect=run_with_index), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.run_one("mathqa", "exact_axes", seed=42,
                                  outdir=Path(directory),
                                  observation_budget_fraction=0.04 if stop_early else 1.0, **settings)

    def test_real_and_unit_runs_keep_actual_dollar_checkpoints(self):
        for mode in ("real", "unit"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                payload = self._run(directory, acquisition_cost_mode=mode)
                checkpoint = payload["search_cost_checkpoint_10pct"]
                self.assertIsNotNone(checkpoint)
                self.assertAlmostEqual(checkpoint["actual_search_cost_usd"], checkpoint["estimated_search_cost_usd"])
                self.assertEqual(len(payload["summary"]["direction_eta"]), 2)
                self.assertIsNone(payload["search_cost_checkpoint_10pct_unavailable_reason"])
                if mode == "unit":
                    self.assertEqual(payload["parameters"]["expected_batch_costs_usd"], [1.0, 1.0])

    def test_early_stop_retains_the_run_without_fabricating_a_ten_percent_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run(directory, stop_early=True)
            self.assertIsNone(payload["search_cost_checkpoint_10pct"])
            self.assertIn("never reached", payload["search_cost_checkpoint_10pct_unavailable_reason"])
            runner.export_combined(Path(directory), ["mathqa"], ["exact_axes"])
            summary = json.loads((Path(directory) / "summary.json").read_text())[0]
            self.assertIsNone(summary["actual_search_cost_percent_at_10pct"])

    def test_unrelated_checkpoint_errors_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner, "actual_cost_checkpoint", side_effect=ValueError("corrupt trace")), \
                self.assertRaisesRegex(ValueError, "corrupt trace"):
            self._run(directory)

    def test_existing_result_reuse_requires_the_same_protocol_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run(directory)
            output = Path(directory) / "exact_axes/mathqa/result.json"
            original = output.read_text()
            with mock.patch.object(runner, "benchmark_input_hashes", return_value={"matrix": "current-input"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                reused = runner.run_one("mathqa", "exact_axes", seed=42, outdir=Path(directory),
                                        observation_budget_fraction=1.0)
                self.assertEqual(reused, payload)
                changes = {
                    "seed": 43,
                    "ablation_name": "g0_current_gittins",
                    "acquisition_cost_mode": "unit",
                    "continuation_mode": "fixed_eta_no_stop",
                    "eta_decay_schedule": "global_stop",
                    "direction_scheduler": "quality_then_deployment",
                    "observation_budget_fraction": 0.5,
                }
                for key, value in changes.items():
                    with self.subTest(changed=key):
                        options = {"seed": 42, "observation_budget_fraction": 1.0, key: value}
                        with self.assertRaisesRegex(ValueError, "configuration differs"):
                            runner.run_one("mathqa", "exact_axes", outdir=Path(directory), **options)
                        self.assertEqual(output.read_text(), original)
                for field, value in (("pair_name", "quality_only"), ("directions", [[1.0, 0.0]]),
                                     ("question_order", "shared"), ("warm_start_question_order", "shared"),
                                     ("cost_model", "reciprocal"), ("recommendation_rule", "finite_mean"),
                                     ("recommendation_beta", 0.0), ("batch_size", 8),
                                     ("input_sha256", {"matrix": "stale-input"})):
                    with self.subTest(changed=field):
                        changed = json.loads(original)
                        changed["config"][field] = value
                        output.write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError, field):
                            runner.run_one("mathqa", "exact_axes", seed=42, outdir=Path(directory),
                                           observation_budget_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
