import unittest

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_pareto_baselines import (
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.single_objective.offline_selector_sim import SampleResult


def _sample(score, cost):
    return SampleResult(
        score=score,
        latency_seconds=0.1,
        input_tokens={},
        output_tokens={},
        cost=cost,
    )


def _toy_table(n_questions: int = 16):
    models = ["accurate", "balanced", "cheap", "dominated"]
    datapoints = list(range(n_questions))
    values = {
        "accurate": (0.9, 0.09),
        "balanced": (0.7, 0.04),
        "cheap": (0.5, 0.01),
        "dominated": (0.4, 0.08),
    }
    table = {
        model: {q: _sample(score, cost) for q in datapoints}
        for model, (score, cost) in values.items()
    }
    return models, datapoints, table


class ParetoBaselineReplayTests(unittest.TestCase):
    def test_ege_sh_full_budget_recovers_true_front(self):
        models, datapoints, table = _toy_table()
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=EGE_SH,
            seed=0,
            batch_size=2,
            observation_budget_fraction=1.0,
        )
        self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
        self.assertAlmostEqual(result.hypervolume_regret, 0.0)
        self.assertAlmostEqual(result.generational_distance, 0.0)
        self.assertAlmostEqual(result.inverted_generational_distance, 0.0)
        self.assertEqual(result.total_evaluations, 64)
        self.assertGreaterEqual(len(result.recommendation_trajectory), 2)
        self.assertFalse(result.params["halt_on_identification_stop"])

    def test_ege_sr_full_budget_recovers_true_front(self):
        models, datapoints, table = _toy_table()
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=EGE_SR,
            seed=1,
            batch_size=2,
        )
        self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
        self.assertAlmostEqual(result.hypervolume_regret, 0.0)

    def test_ape_runs_full_budget_without_stopping(self):
        models, datapoints, table = _toy_table()
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=APE_K,
            seed=2,
            batch_size=2,
            ape_k=2,
        )
        self.assertEqual(result.total_evaluations, 64)
        self.assertEqual(result.stop_reason, "question_budget")
        self.assertFalse(result.params["ape_stopping"])
        self.assertEqual(result.params["ape_k"], 2)
        self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
        self.assertAlmostEqual(result.hypervolume_regret, 0.0)

    def test_partial_budget_does_not_evaluate_the_full_matrix(self):
        models, datapoints, table = _toy_table()
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=EGE_SH,
            seed=3,
            batch_size=2,
            observation_budget_fraction=0.5,
        )
        self.assertLessEqual(result.total_evaluations, 32)
        self.assertGreater(result.total_evaluations, 0)
        self.assertTrue(np.isnan(result.estimated_raw_vectors).any() or result.total_evaluations < 64)


class QNEHVIReplayTests(unittest.TestCase):
    def test_qnehvi_full_budget_recovers_true_front_when_botorch_is_available(self):
        try:
            import botorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("botorch is not installed")
        models, datapoints, table = _toy_table(n_questions=8)
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=QNEHVI,
            seed=4,
            batch_size=2,
            qnehvi_mc_samples=16,
            qnehvi_refit_every=4,
        )
        self.assertEqual(result.selector, QNEHVI)
        self.assertEqual(result.total_evaluations, 32)
        self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
        self.assertAlmostEqual(result.hypervolume_regret, 0.0)


if __name__ == "__main__":
    unittest.main()
