"""Three-objective Pareto runner shares post-run scoring and actual USD budgets."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.combined_objective import compare_three_objective_baselines as comparison
from experiments.combined_objective import run_three_objective_pareto_baselines as runner
from experiments.combined_objective.offline_pareto_baselines import simulate_pareto_baseline
from experiments.combined_objective.three_objective_metrics import (
    evaluation_space, raw_pareto_indices, raw_truth_vectors, score_selection,
)
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem():
    models = ["accurate", "balanced", "cheap", "fast", "dominated"]
    questions = [3, 6, 9, 12, 15, 18]
    values = [(.90, 10., .09), (.70, 2., .04), (.50, 1., .01),
              (.65, .1, .07), (.40, 20., .08)]
    table = {
        model: {
            question: SampleResult(score=quality, latency_seconds=latency + .01 * i,
                                   cost=cost + .001 * i, input_tokens={}, output_tokens={})
            for i, question in enumerate(questions)
        }
        for model, (quality, latency, cost) in zip(models, values)
    }
    return models, questions, table


def _write_matrices(directory: Path, problem) -> None:
    models, questions, table = problem
    for filename, attribute in (("accuracy_matrix.csv", "score"),
                                ("cost_matrix_usd.csv", "cost"),
                                ("latency_matrix_seconds.csv", "latency_seconds")):
        with (directory / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_name", *[f"question_{q}" for q in questions]])
            for model in models:
                writer.writerow([model, *[getattr(table[model][q], attribute) for q in questions]])


class ThreeObjectiveParetoRunnerTests(unittest.TestCase):
    def test_replay_scored_afterward_in_shared_three_objective_space(self):
        models, questions, table = _problem()
        result = simulate_pareto_baseline(
            models, questions, table, method="ege_sh", seed=7, batch_size=2,
            observation_budget_fraction=1., objectives=("Q", "L", "D"),
        )
        run = runner._result_as_run(result, models, questions)
        truth = raw_truth_vectors(models, questions, table)
        self.assertEqual(raw_pareto_indices(truth), [0, 1, 2, 3])
        self.assertEqual(run["full_data_pareto_arm_indices"], [0, 1, 2, 3])
        self.assertEqual(run["points"][-1]["selected_arm_indices"], [0, 1, 2, 3])
        self.assertEqual(run["points"][-1]["pareto_recall"], 1.)
        self.assertAlmostEqual(run["cost_fraction"], 1.)
        self.assertAlmostEqual(run["search_cost_usd"],
                               sum(sample.cost for row in table.values() for sample in row.values()))
        metric, true_front, _, _, truth_hv = evaluation_space(truth)
        first = run["points"][0]
        expected = score_selection(first["selected_arm_indices"], metric, true_front, truth_hv)
        for key in ("hypervolume", "relative_hv_regret", "generational_distance",
                    "inverted_generational_distance"):
            self.assertAlmostEqual(first[key], expected[key])
        self.assertTrue(all(point["cost_fraction"] <= next_point["cost_fraction"]
                            for point, next_point in zip(run["points"], run["points"][1:])))

    def test_saved_replay_is_strict_json_and_comparison_loads_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "toy"
            benchmark.mkdir()
            _write_matrices(benchmark, _problem())
            output = root / "results"
            saved = runner.run_one(benchmark, method="ege_sh", seed=7, outdir=output,
                                   batch_size=2, observation_budget_fraction=.1)
            path = output / "seed_7" / "toy" / "ege_sh" / "result.json"
            self.assertEqual(json.loads(path.read_text()), saved)
            self.assertEqual(saved["config"]["objective_order"], ["Q", "L", "D"])
            self.assertEqual(saved["run"]["params"]["objectives"], ["Q", "L", "D"])
            self.assertLess(saved["run"]["total_evaluations"], 30)
            self.assertIn(None, np.asarray(saved["baseline_result"]["estimated_raw_vectors"],
                                           dtype=object).ravel().tolist())
            points = saved["run"]["points"]
            target = (points[0]["cost_fraction"] + points[1]["cost_fraction"]) / 2
            checkpoint = runner._latest_at_or_below(saved["run"], target)
            self.assertEqual(checkpoint["total_evaluations"], points[0]["total_evaluations"])
            self.assertEqual(checkpoint["cumulative_search_cost_usd"],
                             points[0]["cumulative_search_cost_usd"])
            self.assertEqual(runner._latest_at_or_below(saved["run"], .5)["reason"],
                             "target_not_reached")
            held = comparison.aggregate_at([saved["run"]], .5)
            self.assertEqual(held["n_available"], 1)
            self.assertEqual(held["n_stopped_before_target"], 1)
            self.assertAlmostEqual(held["actual_cost_fraction_mean"], saved["run"]["cost_fraction"])
            with mock.patch.object(runner, "simulate_pareto_baseline", side_effect=AssertionError("reran")):
                self.assertEqual(runner.run_one(benchmark, method="ege_sh", seed=7,
                                                outdir=output, batch_size=2,
                                                observation_budget_fraction=.1), saved)
            with mock.patch.object(comparison, "PARETO_METHODS", ("ege_sh",)):
                groups = comparison.pareto_runs("toy", [7], output,
                                                saved["config"]["input_sha256"])
            self.assertEqual(groups["ege_sh"][0]["points"], saved["run"]["points"])
            with mock.patch.object(comparison, "PARETO_METHODS", ("ege_sh",)):
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    comparison.pareto_runs("toy", [7], output, {"different": "hash"})


if __name__ == "__main__":
    unittest.main()
