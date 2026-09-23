import unittest
from unittest.mock import patch

import numpy as np

from agentopt.model_selection.pareto_identification import APE_K, EGE_SH, EGE_SR
from experiments.combined_objective.offline_pareto_baselines import (
    QNEHVI,
    simulate_pareto_baseline,
)
from experiments.combined_objective.offline_cost_budget_random_search import (
    RANDOM_CONFIGURATIONS,
    RANDOM_QUESTIONS,
    simulate_cost_budget_random_search,
)
from experiments.combined_objective.build_usd_checkpoint_dataset import (
    _trajectory_interval,
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
    def test_checkpoint_downsampling_does_not_change_policy(self):
        models, datapoints, table = _toy_table(n_questions=32)
        dense = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=APE_K,
            seed=9,
            batch_size=2,
            recommendation_checkpoint_interval=1,
        )
        sparse = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=APE_K,
            seed=9,
            batch_size=2,
            recommendation_checkpoint_interval=10,
        )
        self.assertEqual(dense.selected_arm_indices, sparse.selected_arm_indices)
        np.testing.assert_array_equal(
            dense.estimated_raw_vectors,
            sparse.estimated_raw_vectors,
        )
        self.assertEqual(dense.total_evaluations, sparse.total_evaluations)
        self.assertLess(len(sparse.recommendation_trajectory), len(dense.recommendation_trajectory))

    def test_usd_checkpoints_and_membership_changes_are_always_retained(self):
        models, datapoints, table = _toy_table(n_questions=32)
        result = simulate_pareto_baseline(
            models,
            datapoints,
            table,
            method=APE_K,
            seed=9,
            batch_size=2,
            recommendation_checkpoint_interval=1000,
            recommendation_cost_checkpoint_fractions=(0.10, 0.30),
        )
        events = [point.event for point in result.recommendation_trajectory]
        self.assertEqual(events.count("cost_checkpoint_10pct"), 1)
        self.assertEqual(events.count("cost_checkpoint_30pct"), 1)
        self.assertIn("recommendation_initial", events)
        for target in (0.10, 0.30):
            point = next(
                point
                for point in result.recommendation_trajectory
                if point.event == f"cost_checkpoint_{int(100 * target)}pct"
            )
            self.assertLessEqual(point.budget_fraction, target + 1e-12)
        self.assertEqual(
            result.params["recommendation_checkpoint_schema_version"], 3
        )
        self.assertEqual(
            result.params["recommendation_cost_checkpoint_semantics"],
            "latest_completed_policy_update_at_or_below_realized_usd_fraction",
        )

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
    def test_qnehvi_can_stop_after_target_recommendation_changes(self):
        models, datapoints, table = _toy_table(n_questions=20)

        def recommendation(puller):
            return (0,) if puller.total_evaluations < 9 else (1,)

        with (
            patch(
                "agentopt.model_selection.qnehvi.select_qnehvi_index",
                return_value=0,
            ),
            patch(
                "experiments.combined_objective.offline_pareto_baselines."
                "empirical_raw_pareto_arms",
                side_effect=recommendation,
            ),
        ):
            result = simulate_pareto_baseline(
                models,
                datapoints,
                table,
                method=QNEHVI,
                seed=5,
                batch_size=1,
                qnehvi_refit_every=32,
                recommendation_cost_checkpoint_fractions=(0.10, 0.30),
                stop_after_recommendation_interval_fraction=0.10,
            )

        self.assertEqual(result.stop_reason, "recommendation_interval_complete")
        self.assertLess(result.total_evaluations, len(models) * len(datapoints))
        events = [point.event for point in result.recommendation_trajectory]
        self.assertEqual(events.count("cost_checkpoint_10pct"), 1)
        self.assertEqual(events[-1], "recommendation_changed")
        self.assertNotIn("terminal", events)

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


class CostBudgetRandomReplayTests(unittest.TestCase):
    def test_random_methods_store_realized_usd_checkpoints(self):
        models, datapoints, table = _toy_table(n_questions=16)
        for method in (RANDOM_CONFIGURATIONS, RANDOM_QUESTIONS):
            with self.subTest(method=method):
                result = simulate_cost_budget_random_search(
                    models,
                    datapoints,
                    table,
                    method=method,
                    seed=5,
                )
                events = [row["event"] for row in result.trajectory]
                self.assertEqual(events.count("cost_checkpoint_10pct"), 1)
                self.assertEqual(events.count("cost_checkpoint_30pct"), 1)
                self.assertEqual(events[-1], "terminal")
                self.assertEqual(
                    result.summary["params"][
                        "recommendation_checkpoint_schema_version"
                    ],
                    3,
                )
                for target in (0.10, 0.30):
                    point = next(
                        row
                        for row in result.trajectory
                        if row["event"]
                        == f"cost_checkpoint_{int(100 * target)}pct"
                    )
                    self.assertLessEqual(
                        float(point["budget_fraction"]), target + 1e-12
                    )

    def test_checkpoint_interval_uses_contiguous_membership_segment(self):
        def row(event, fraction, evaluations, selected):
            return {
                "event": event,
                "budget_fraction": str(fraction),
                "cumulative_search_cost_usd": str(10 * fraction),
                "cumulative_evaluations": str(evaluations),
                "selected_arm_indices": selected,
                "selected_models": selected,
            }

        rows = [
            row("recommendation_initial", 0.02, 2, "1"),
            row("recommendation_changed", 0.18, 18, "1;2"),
            row("cost_checkpoint_30pct", 0.299, 30, "1;2"),
            row("recommendation_changed", 0.44, 44, "2"),
            row("terminal", 1.0, 100, "2"),
        ]
        interval = _trajectory_interval(rows, 0.30)
        self.assertAlmostEqual(
            interval["recommendation_start_cost_fraction"], 0.18
        )
        self.assertAlmostEqual(
            interval["recommendation_end_cost_fraction"], 0.44
        )
        self.assertFalse(interval["recommendation_end_censored_at_terminal"])


if __name__ == "__main__":
    unittest.main()
