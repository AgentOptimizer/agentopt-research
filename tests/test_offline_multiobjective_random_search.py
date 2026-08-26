import unittest

import numpy as np

from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    mean_raw_vectors,
    pareto_min_cost_indices,
    run_budget_sweep,
    simulate_multiobjective_random_search,
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


def _toy_table():
    models = ["accurate", "balanced", "cheap", "dominated"]
    datapoints = list(range(10))
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


class MultiObjectiveRandomSearchTests(unittest.TestCase):
    def test_pareto_front_uses_accuracy_up_and_cost_down(self):
        points = np.asarray([[0.9, 0.09], [0.7, 0.04], [0.5, 0.01], [0.4, 0.08]])
        self.assertEqual(pareto_min_cost_indices(points), [0, 1, 2])

    def test_common_questions_exclude_any_missing_cell(self):
        models, datapoints, table = _toy_table()
        del table["cheap"][7]
        self.assertEqual(common_question_ids(models, datapoints, table), tuple(range(7)) + (8, 9))

    def test_mean_raw_vectors(self):
        models, datapoints, table = _toy_table()
        vectors = mean_raw_vectors(models, datapoints, table)
        np.testing.assert_allclose(
            vectors,
            [[0.9, 0.09], [0.7, 0.04], [0.5, 0.01], [0.4, 0.08]],
        )

    def test_random_configurations_observes_full_rows(self):
        models, datapoints, table = _toy_table()
        result = simulate_multiobjective_random_search(
            models,
            datapoints,
            table,
            version="random_configurations",
            budget_fraction=0.5,
            seed=3,
        )
        self.assertEqual(len(result.sampled_arm_indices), 2)
        self.assertEqual(result.sampled_question_ids, tuple(datapoints))
        self.assertEqual(result.total_evaluations, 20)
        self.assertEqual(np.isnan(result.estimated_vectors[:, 0]).sum(), 2)

    def test_random_questions_observes_shared_partial_columns(self):
        models, datapoints, table = _toy_table()
        result = simulate_multiobjective_random_search(
            models,
            datapoints,
            table,
            version="random_questions",
            budget_fraction=0.5,
            seed=3,
        )
        self.assertEqual(result.sampled_arm_indices, tuple(range(4)))
        self.assertEqual(len(result.sampled_question_ids), 5)
        self.assertEqual(result.total_evaluations, 20)
        self.assertFalse(np.isnan(result.estimated_vectors).any())

    def test_full_budget_recovers_true_front_for_both_versions(self):
        models, datapoints, table = _toy_table()
        for version in ("random_configurations", "random_questions"):
            result = simulate_multiobjective_random_search(
                models,
                datapoints,
                table,
                version=version,
                budget_fraction=1.0,
                seed=7,
            )
            self.assertEqual(set(result.selected_arm_indices), {0, 1, 2})
            self.assertAlmostEqual(result.hypervolume_regret, 0.0)
            self.assertAlmostEqual(result.true_front_recall, 1.0)
            self.assertAlmostEqual(result.recommendation_precision, 1.0)
            self.assertEqual(result.false_positive_count, 0)

    def test_random_questions_reports_false_positive_recommendations(self):
        models = ["accurate", "cheap", "noisy"]
        datapoints = list(range(10))
        table = {
            "accurate": {q: _sample(0.9, 0.09) for q in datapoints},
            "cheap": {q: _sample(0.5, 0.01) for q in datapoints},
            "noisy": {
                q: _sample(1.0 if q == 0 else 0.0, 0.08)
                for q in datapoints
            },
        }
        # Find a deterministic seed whose one-question prefix observes q=0.
        seed = next(
            candidate
            for candidate in range(100)
            if np.random.default_rng(candidate).permutation(datapoints)[0] == 0
        )
        result = simulate_multiobjective_random_search(
            models, datapoints, table, version="random_questions",
            budget_fraction=0.1, seed=seed,
        )
        self.assertIn(2, result.selected_arm_indices)
        self.assertEqual(result.false_positive_count, 1)
        self.assertLess(result.recommendation_precision, 1.0)

    def test_sweep_has_every_version_seed_and_fraction(self):
        models, datapoints, table = _toy_table()
        results = run_budget_sweep(
            models,
            datapoints,
            table,
            budget_fractions=(0.1, 0.2),
            seeds=(10, 11),
        )
        self.assertEqual(len(results), 8)

    def test_budget_sweep_uses_nested_random_prefixes(self):
        models, datapoints, table = _toy_table()
        results = run_budget_sweep(
            models,
            datapoints,
            table,
            budget_fractions=(0.2, 0.5, 0.8),
            seeds=(13,),
        )
        for version in ("random_configurations", "random_questions"):
            version_results = [result for result in results if result.version == version]
            for earlier, later in zip(version_results, version_results[1:]):
                self.assertTrue(
                    set(earlier.sampled_arm_indices).issubset(later.sampled_arm_indices)
                )
                self.assertTrue(
                    set(earlier.sampled_question_ids).issubset(
                        later.sampled_question_ids
                    )
                )

    def test_invalid_version_and_fraction_fail(self):
        models, datapoints, table = _toy_table()
        with self.assertRaises(ValueError):
            simulate_multiobjective_random_search(
                models, datapoints, table, version="unknown", budget_fraction=0.5
            )
        with self.assertRaises(ValueError):
            simulate_multiobjective_random_search(
                models,
                datapoints,
                table,
                version="random_questions",
                budget_fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main()
