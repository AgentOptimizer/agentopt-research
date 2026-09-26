"""The 10% dollar checkpoint must preserve online estimates and true spend."""

from types import SimpleNamespace
import unittest

from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule
from experiments.combined_objective.compare_lcb_recommendations import compact_lcb_run
from experiments.combined_objective.offline_radial_gittins import simulate_radial_gittins
from experiments.combined_objective.search_cost_checkpoint import (
    CheckpointNotReachedError,
    actual_cost_checkpoint,
)
from experiments.single_objective.offline_selector_sim import SampleResult


def _pull(event, arm, scores, costs, cumulative_evaluations, cumulative_cost):
    return {
        "event": event,
        "arm_index" if event == "warm_start" else "selected_arm": arm,
        "question_ids": list(range(len(costs))),
        "batch_score_mean": sum(scores) / len(scores),
        "batch_deployment_cost_mean_usd": sum(costs) / len(costs),
        "actual_batch_search_cost_usd": sum(costs),
        "expected_batch_search_cost_usd": 0.4 if arm == 0 else 0.2,
        "cumulative_evaluations": cumulative_evaluations,
        "cumulative_search_cost_usd": cumulative_cost,
    }


def _result():
    return SimpleNamespace(
        params={
            "record_trace": True,
            "record_recommendation_trajectory": True,
            "recommendation_changes_only": True,
            "bruteforce_search_cost_usd": 10.0,
            "batch_size": 2,
            "expected_batch_costs_usd": [0.4, 0.2],
        },
        model_results=[SimpleNamespace(model_name="arm_0"), SimpleNamespace(model_name="arm_1")],
        raw_truth_vectors=[[0.75, 0.55], [0.5, 0.2]],
        recommendation_initial_snapshot=SimpleNamespace(
            cumulative_evaluations=4, selected_arm_indices=(1,)
        ),
        recommendation_trajectory=[SimpleNamespace(
            cumulative_evaluations=6, selected_arm_indices=(0,)
        )],
        trace=[
            _pull("warm_start", 0, [0.4, 0.4], [0.2, 0.2], 2, 0.4),
            _pull("warm_start", 1, [0.5, 0.5], [0.1, 0.1], 4, 0.6),
            _pull("direction_visit", 0, [0.8, 0.8], [0.7, 0.7], 6, 2.0),
            # This later pull must not alter the 10% checkpoint.
            _pull("direction_visit", 1, [0.9, 0.9], [0.25, 0.25], 8, 2.5),
        ],
        total_cost=2.5,
    )


class SearchCostCheckpointTests(unittest.TestCase):
    def test_replay_uses_the_first_batch_above_ten_percent(self):
        questions = list(range(24))
        schedule = PerArmQuestionSchedule.create_from_available(
            {0: questions}, warm_start_batch_size=2, seed=7,
            question_order="independent", warm_start_question_order="independent",
        )
        costs = {
            question: (0.01 if position < 2 else 0.05)
            for position, question in enumerate(schedule.orders[0])
        }
        table = {"arm_0": {
            question: SampleResult(
                score=0.6, cost=costs[question], latency_seconds=0.0,
                input_tokens={}, output_tokens={},
            )
            for question in questions
        }}
        result = simulate_radial_gittins(
            ["arm_0"], questions, table, seed=7, batch_size=2,
            warm_start_batch_size=2, question_order="independent",
            warm_start_question_order="independent", directions=((1.0, 0.0),),
            anytime=True, cost_model="raw_mean", recommendation_rule="finite_lcb",
            index_provider=lambda context, arm: 100.0,
            record_trace=True, record_recommendation_trajectory=True,
            recommendation_changes_only=True,
            defer_recommendation_diagnostics=True,
        )
        compact_lcb_run(result)
        checkpoint = actual_cost_checkpoint(result)
        self.assertEqual(checkpoint["cumulative_evaluations"], 4)
        self.assertAlmostEqual(checkpoint["actual_search_cost_usd"], 0.12)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_usd"], 0.04)
        self.assertEqual(checkpoint["selected_arm_indices"], [0])

    def test_first_crossing_uses_warm_estimate_and_exact_checkpoint_coordinates(self):
        checkpoint = actual_cost_checkpoint(_result())
        self.assertEqual(checkpoint["cumulative_evaluations"], 6)
        self.assertEqual(checkpoint["selected_arm_indices"], [0])
        self.assertEqual(checkpoint["selected_sample_counts"], [4])
        self.assertAlmostEqual(checkpoint["warm_start_search_cost_usd"], 0.6)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_usd"], 1.0)
        self.assertAlmostEqual(checkpoint["actual_search_cost_usd"], 2.0)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_percent"], 10.0)
        self.assertAlmostEqual(checkpoint["actual_search_cost_percent"], 20.0)
        self.assertAlmostEqual(checkpoint["estimated_raw_archive_vectors"][0][0], 0.6)
        self.assertAlmostEqual(checkpoint["estimated_raw_archive_vectors"][0][1], 0.45)
        self.assertEqual(checkpoint["offline_raw_selected_vectors"], [[0.75, 0.55]])

    def test_below_threshold_run_is_rejected_instead_of_using_a_sparse_snapshot(self):
        result = _result()
        result.trace = result.trace[:2]
        result.total_cost = 0.6
        with self.assertRaisesRegex(CheckpointNotReachedError, "never reached"):
            actual_cost_checkpoint(result)

    def test_unit_acquisition_uses_warm_dollars_for_the_cost_estimate(self):
        result = _result()
        result.params["expected_batch_costs_usd"] = [1.0, 1.0]
        for event in result.trace:
            event["expected_batch_search_cost_usd"] = 1.0
        checkpoint = actual_cost_checkpoint(result, acquisition_cost_mode="unit")
        self.assertAlmostEqual(checkpoint["actual_search_cost_usd"], 2.0)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_usd"], 1.0)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_percent"], 10.0)
        result.trace[2]["expected_batch_search_cost_usd"] = 0.9
        with self.assertRaisesRegex(ValueError, "frozen estimate"):
            actual_cost_checkpoint(result, acquisition_cost_mode="unit")

    def test_threshold_within_warm_start_uses_first_available_recommendation(self):
        checkpoint = actual_cost_checkpoint(_result(), target_fraction=0.05)
        self.assertEqual(checkpoint["cumulative_evaluations"], 4)
        self.assertEqual(checkpoint["selected_arm_indices"], [1])
        self.assertAlmostEqual(checkpoint["actual_search_cost_percent"], 6.0)
        self.assertAlmostEqual(checkpoint["estimated_search_cost_percent"], 6.0)

    def test_explicit_cost_override_cannot_be_mislabelled_as_warm_estimate(self):
        result = _result()
        result.params["expected_batch_costs_usd"][0] = 0.9
        with self.assertRaisesRegex(ValueError, "warm-start means"):
            actual_cost_checkpoint(result)

    def test_frozen_prediction_is_checked_against_the_recorded_batch(self):
        result = _result()
        result.trace[2]["expected_batch_search_cost_usd"] = 0.9
        with self.assertRaisesRegex(ValueError, "frozen estimate"):
            actual_cost_checkpoint(result)


if __name__ == "__main__":
    unittest.main()
