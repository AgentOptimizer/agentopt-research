"""Exact objective endpoints participate in replay, costs, and recommendations."""

import unittest
from unittest import mock

import numpy as np

from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(values=((0.6, 0.001), (0.95, 1.0)), n_questions=3):
    models = [f"arm_{index}" for index in range(len(values))]
    questions = list(range(n_questions))
    table = {
        model: {
            question: SampleResult(
                score=accuracy,
                cost=cost,
                latency_seconds=0.1,
                input_tokens={},
                output_tokens={},
            )
            for question in questions
        }
        for model, (accuracy, cost) in zip(models, values)
    }
    return models, questions, table


def _run(problem=None, **overrides):
    options = {
        "batch_size": 1,
        "directions": ((1.0, 0.0),),
        "cost_reference_usd": 0.001,
        "prior_variance": 0.04,
        "obs_noise_variance": 1e-5,
        "effective_cost_bin_ratio": None,
        "boundary_grid": RadialGittinsGrid(
            z_size=65, delta_size=65, state_size=65,
        ),
        "boundary_build_backend": "scipy",
        "anytime": True,
        "record_recommendation_trajectory": True,
        "recommendation_checkpoint_interval": 100,
        "seed": 42,
    }
    options.update(overrides)
    return replay.simulate_radial_gittins(
        *(_problem() if problem is None else problem), **options,
    )


class AxisRadialReplayTests(unittest.TestCase):
    def test_accuracy_endpoint_ignores_inactive_deployment_objective(self):
        for deployment_mean, deployment_var in ((0.01, 0.001), (0.99, 100.0)):
            with self.subTest(deployment_mean=deployment_mean):
                utility = replay.terminal_expected_direction_utility(
                    (0.8, deployment_mean),
                    (0.02, deployment_var),
                    (1.0, 0.0),
                    (0.1, 0.3),
                )
                self.assertAlmostEqual(utility, 0.7)
        self.assertAlmostEqual(
            replay.terminal_expected_direction_utility(
                (0.8, 0.6), (0.02, 0.01), (0.0, 1.0), (0.1, 0.3),
            ),
            0.3,
        )

    def test_real_axis_dp_recovers_expensive_accuracy_best_after_lambda_decay(self):
        # At lambda=1 the expensive arm's remaining search penalty makes the
        # cheap arm worth completing first. Lower penalties eventually justify
        # evaluating and recommending the expensive accuracy endpoint.
        with mock.patch.object(
            replay,
            "prewarm_radial_gittins_boundaries",
            side_effect=AssertionError("an exact axis must use its scalar DP"),
        ):
            result = _run()

        self.assertEqual(result.selected_models, ["arm_1"])
        self.assertTrue(result.contains_true_accuracy_best)
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertGreater(result.lambda_stage, 0)
        self.assertTrue(result.lambda_stop_events)
        self.assertTrue(all(arm.completed for arm in result.model_results))
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(len(result.observed_cells), 6)
        self.assertEqual(len(set(result.observed_cells)), 6)
        pulls = [event for event in result.trace if event.get("selected_arm") is not None]
        self.assertEqual(pulls[0]["selected_arm"], 0)
        expensive_pulls = [event for event in pulls if event["selected_arm"] == 1]
        self.assertTrue(expensive_pulls)
        self.assertLess(expensive_pulls[0]["current_lambda"], 1.0)
        for event in pulls:
            self.assertEqual(event["direction"], [1.0, 0.0])
            expected_cost = (0.001, 1.0)[event["selected_arm"]]
            self.assertAlmostEqual(
                event["raw_effective_pull_cost"],
                event["current_lambda"] * expected_cost,
            )
            self.assertFalse(event["forced_after_gittins_stop"])
        for checkpoint in result.recommendation_trajectory:
            self.assertTrue(checkpoint.is_deployable)
            self.assertTrue(
                set(checkpoint.selected_arm_indices).issubset(checkpoint.completed_arm_indices)
            )

    def test_mixed_radial_and_accuracy_directions_keep_both_frontier_extremes(self):
        result = _run(directions=((0.9, 0.1), (1.0, 0.0)))

        self.assertEqual(set(result.selected_models), {"arm_0", "arm_1"})
        self.assertTrue(result.contains_true_accuracy_best)
        winners = {winner.direction: winner.model_name for winner in result.direction_winners}
        self.assertEqual(winners[(0.9, 0.1)], "arm_0")
        self.assertEqual(winners[(1.0, 0.0)], "arm_1")
        self.assertAlmostEqual(result.hypervolume, result.ground_truth_hypervolume)
        self.assertAlmostEqual(result.generational_distance, 0.0)
        self.assertAlmostEqual(result.inverted_generational_distance, 0.0)

    def test_endpoint_participates_in_stopping_instead_of_postprocessing(self):
        def index(context, arm_index):
            if 0 not in context.completed_arms:
                return 100.0 if arm_index == 0 else -100.0
            if context.direction == (1.0, 0.0):
                return 100.0
            return -100.0

        result = _run(
            directions=((0.5, 0.5), (1.0, 0.0)),
            index_provider=index,
        )

        self.assertEqual(result.lambda_stage, 0)
        self.assertFalse(result.lambda_stop_events)
        visits = [event for event in result.trace if event["event"] == "direction_visit"]
        self.assertTrue(any(
            event["direction"] == [0.5, 0.5] and event["direction_should_stop"]
            for event in visits
        ))
        expensive_pulls = [event for event in visits if event.get("selected_arm") == 1]
        self.assertEqual(len(expensive_pulls), 2)
        self.assertTrue(all(event["direction"] == [1.0, 0.0] for event in expensive_pulls))
        self.assertTrue(result.contains_true_accuracy_best)

    def test_accuracy_tie_recommends_cheapest_observed_combination(self):
        result = _run(_problem(values=((0.9, 3.0), (0.9, 1.0)), n_questions=1))

        self.assertEqual(result.selected_models, ["arm_1"])
        self.assertEqual(result.direction_winners[0].model_name, "arm_1")
        self.assertEqual(result.recommendation_trajectory[-1].selected_models, ("arm_1",))
        self.assertTrue(result.contains_true_accuracy_best)

    def test_cost_endpoint_tie_recommends_more_accurate_combination(self):
        result = _run(
            _problem(values=((0.6, 0.001), (0.9, 0.001)), n_questions=1),
            directions=((0.0, 1.0),),
        )

        self.assertEqual(result.selected_models, ["arm_1"])
        self.assertEqual(result.direction_winners[0].model_name, "arm_1")
        self.assertTrue(np.isfinite(result.hypervolume))


if __name__ == "__main__":
    unittest.main()
