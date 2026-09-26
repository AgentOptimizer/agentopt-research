"""Dimension and scoring checks for Pareto-search baselines."""

import unittest
from unittest import mock

import numpy as np

from experiments.combined_objective.offline_pareto_baselines import (
    simulate_pareto_baseline,
)
from experiments.combined_objective.three_objective_metrics import (
    evaluation_space,
    raw_pareto_indices,
    score_selection,
)
from experiments.single_objective.offline_selector_sim import SampleResult


def _sample(q, l, d):
    return SampleResult(score=q, latency_seconds=l, cost=d,
                        input_tokens={}, output_tokens={})


def _problem(n_questions=8):
    models = ["accurate", "balanced", "cheap_fast", "dominated"]
    questions = list(range(n_questions))
    values = {
        "accurate": (.9, 10., .09),
        "balanced": (.7, 4., .05),
        "cheap_fast": (.5, 1., .01),
        "dominated": (.3, 20., .20),
    }
    table = {model: {q: _sample(*point) for q in questions}
             for model, point in values.items()}
    return models, questions, table


class ThreeObjectiveParetoBaselineTests(unittest.TestCase):
    def test_ege_and_ape_full_budget_recover_3d_front_and_shared_metrics(self):
        models, questions, table = _problem()
        expected_cost = sum(sample.cost for row in table.values() for sample in row.values())
        for method in ("ege_sh", "ege_sr", "ape_k"):
            with self.subTest(method=method):
                result = simulate_pareto_baseline(
                    models, questions, table, method=method, objectives=("Q", "L", "D"),
                    seed=7, batch_size=2,
                )
                self.assertEqual(result.params["objectives"], ["Q", "L", "D"])
                self.assertEqual(result.estimated_raw_vectors.shape, (4, 3))
                self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
                self.assertEqual(result.total_evaluations, 4 * len(questions))
                self.assertAlmostEqual(result.total_search_cost_usd, expected_cost)
                self.assertAlmostEqual(result.hypervolume_regret, 0.)
                metric, front, _, _, true_hv = evaluation_space(result.truth_raw_vectors)
                for checkpoint in result.recommendation_trajectory:
                    expected = score_selection(checkpoint.selected_arm_indices, metric, front, true_hv)
                    self.assertAlmostEqual(checkpoint.hypervolume, expected["hypervolume"])
                    self.assertAlmostEqual(checkpoint.hypervolume_regret, expected["hv_regret"])
                    self.assertAlmostEqual(checkpoint.generational_distance,
                                           expected["generational_distance"])
                    self.assertAlmostEqual(checkpoint.inverted_generational_distance,
                                           expected["inverted_generational_distance"])
                self.assertEqual(raw_pareto_indices(result.truth_raw_vectors), [0, 1, 2])

    def test_qnehvi_runner_passes_three_outcomes_and_reference(self):
        models, questions, table = _problem()
        with mock.patch("agentopt.model_selection.qnehvi.select_qnehvi_index", return_value=0) as select:
            result = simulate_pareto_baseline(
                models, questions, table, method="qnehvi", objectives=("Q", "L", "D"),
                seed=3, batch_size=2, qnehvi_refit_every=1,
            )
        self.assertTrue(select.called)
        self.assertEqual(select.call_args.args[1].shape, (4, 3))
        self.assertEqual(len(select.call_args.kwargs["reference_point"]), 3)
        self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})

    def test_qnehvi_three_objective_failure_is_not_silent_random_sampling(self):
        models, questions, table = _problem()
        with mock.patch("agentopt.model_selection.qnehvi.select_qnehvi_index",
                        side_effect=ValueError("acquisition failed")):
            with self.assertRaisesRegex(RuntimeError, "three-objective qNEHVI acquisition failed"):
                simulate_pareto_baseline(
                    models, questions, table, method="qnehvi",
                    objectives=("Q", "L", "D"), seed=3, batch_size=2,
                )

    def test_zero_warm_axes_use_fixed_fallback_not_full_data_scales(self):
        models, questions, table = _problem()
        seed = 19
        rng = np.random.default_rng(seed)
        first_questions = [int(rng.permutation(questions)[-1]) for _ in models]
        for model, first in zip(models, first_questions):
            previous = table[model][first]
            table[model][first] = _sample(previous.score, 0., 0.)
        result = simulate_pareto_baseline(
            models, questions, table, method="ape_k", objectives=("Q", "L", "D"),
            seed=seed, observation_budget_fraction=1 / len(questions),
        )
        self.assertEqual(result.total_evaluations, len(models))
        self.assertEqual(result.params["algorithm_latency_reference_seconds"], 1.)
        self.assertEqual(result.params["algorithm_cost_reference_usd"], 1.)
        self.assertNotEqual(result.params["evaluation_latency_reference_seconds"], 1.)
        self.assertNotEqual(result.params["evaluation_cost_reference_usd"], 1.)

    def test_three_objective_mode_rejects_missing_cell_and_wrong_reference_dimension(self):
        models, questions, table = _problem()
        del table[models[0]][questions[0]]
        with self.assertRaisesRegex(ValueError, "complete aligned matrix"):
            simulate_pareto_baseline(models, questions, table, method="ege_sh",
                                     objectives=("Q", "L", "D"))
        models, questions, table = _problem()
        with self.assertRaisesRegex(ValueError, "match the objective dimension"):
            simulate_pareto_baseline(models, questions, table, method="ege_sh",
                                     objectives=("Q", "L", "D"), reference_point=(0., 0.))

    def test_default_two_objective_mode_matches_explicit_qd(self):
        models, questions, table = _problem()
        common = dict(method="ege_sh", seed=11, batch_size=2,
                      observation_budget_fraction=.5)
        default = simulate_pareto_baseline(models, questions, table, **common)
        explicit = simulate_pareto_baseline(models, questions, table,
                                            objectives=("Q", "D"), **common)
        np.testing.assert_array_equal(default.estimated_raw_vectors,
                                      explicit.estimated_raw_vectors)
        self.assertEqual(default.selected_arm_indices, explicit.selected_arm_indices)
        self.assertEqual(default.recommendation_trajectory,
                         explicit.recommendation_trajectory)


if __name__ == "__main__":
    unittest.main()
