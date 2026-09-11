"""Behavioral checks for completed-only, changes-only recommendations."""

import unittest

import numpy as np

from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(values=((0.8, 0.1),), n_questions=8):
    models = [f"arm_{index}" for index in range(len(values))]
    questions = list(range(n_questions))
    table = {
        model: {
            question: SampleResult(
                score=score,
                latency_seconds=0.1,
                input_tokens={},
                output_tokens={},
                cost=cost,
            )
            for question in questions
        }
        for model, (score, cost) in zip(models, values)
    }
    return models, questions, table


def _staged_index(context, arm_index):
    target = 0 if context.current_lambda > 0.5 else (
        1 if context.current_lambda > 0.25 else 2
    )
    return 100.0 if arm_index == target else -100.0


def _run(problem=None, **kwargs):
    options = {
        "batch_size": 1,
        "directions": ((0.5, 0.5),),
        "cost_reference_usd": 1.0,
        "prior_variance": 1.0,
        "obs_noise_variance": 1.0,
        "effective_cost_bin_ratio": None,
        "anytime": True,
        "record_recommendation_trajectory": True,
        "recommendation_checkpoint_interval": 100,
        "recommendation_changes_only": True,
        "index_provider": lambda context, arm_index: 100.0,
        "seed": 1,
    }
    options.update(kwargs)
    return replay.simulate_radial_gittins(
        *(_problem() if problem is None else problem), **options
    )


def _without_timing(events):
    timing_fields = {"stage_wall_time_seconds", "run_wall_time_seconds"}
    return [
        {key: value for key, value in event.items() if key not in timing_fields}
        for event in events
    ]


class CompletedRecommendationTests(unittest.TestCase):
    def test_small_posterior_variance_does_not_admit_unfinished_arm(self):
        # With unit prior and batch noise, v(n) = 1 / (1+n). Reaching or
        # passing a quarter of the prior cannot replace full evaluation.
        for observed in (3, 7):
            with self.subTest(observed=observed):
                result = _run(max_total_question_evaluations=observed)

                self.assertEqual(result.stop_reason, "question_budget")
                self.assertEqual(result.total_evaluations, observed)
                self.assertFalse(result.model_results[0].completed)
                np.testing.assert_allclose(
                    result.model_results[0].posterior_var,
                    (1.0 / (1 + observed), 1.0 / (1 + observed)),
                )
                self.assertEqual(result.selected_models, [])
                self.assertEqual(result.direction_winners, [])
                self.assertEqual(result.recommendation_trajectory, [])
                self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, ())
                self.assertEqual(result.params["recommendation_eligibility"], "completed_only")

        completed = _run()
        self.assertTrue(completed.model_results[0].completed)
        self.assertEqual(completed.selected_models, ["arm_0"])
        self.assertEqual(
            [point.cumulative_evaluations for point in completed.recommendation_trajectory],
            [8],
        )
        point = completed.recommendation_trajectory[0]
        self.assertEqual(point.completed_arm_indices, (0,))
        self.assertEqual(point.selected_arm_indices, (0,))
        self.assertEqual(point.archive_scope, "deployable")
        self.assertTrue(point.is_deployable)

    def test_completion_suffices_despite_large_posterior_variance(self):
        result = _run(_problem(n_questions=2), obs_noise_variance=100.0)

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.selected_models, ["arm_0"])
        self.assertTrue(np.all(np.asarray(result.model_results[0].posterior_var) > 0.25))
        self.assertEqual(len(result.recommendation_trajectory), 1)
        point = result.recommendation_trajectory[0]
        self.assertEqual(point.cumulative_evaluations, 2)
        self.assertEqual(point.completed_arm_indices, (0,))

    def test_changes_only_retains_completed_warmup_without_final_duplicate(self):
        result = _run(_problem(n_questions=1), recommendation_checkpoint_interval=1)

        self.assertEqual(result.total_evaluations, 1)
        self.assertEqual(len(result.recommendation_trajectory), 1)
        point = result.recommendation_trajectory[0]
        self.assertEqual(point.cumulative_evaluations, 1)
        self.assertEqual(point.selected_arm_indices, (0,))
        self.assertEqual(point.completed_arm_indices, (0,))
        self.assertEqual(point.added_arm_indices, (0,))
        self.assertEqual(result.selected_models, list(point.selected_models))
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, (0,))

    def test_removal_only_membership_changes_are_retained_without_lambda_duplicates(self):
        models, questions, table = _problem(
            values=((0.3, 1.0), (0.8, 10.0), (0.4, 100.0)), n_questions=3
        )
        # The last arm wins the normalized cost direction, but its raw cost
        # leaves it dominated by arm 1. This removes arm 0 without adding
        # a new recommendation after arm 2 completes.
        for question, cost in zip(questions, (0.0, 0.0, 300.0)):
            table[models[2]][question].cost = cost
        result = _run(
            (models, questions, table),
            directions=((0.1, 0.9), (0.9, 0.1)),
            expected_batch_cost_usd=(1.0, 10.0, 100.0),
            prior_variance=0.001,
            obs_noise_variance=1e-9,
            index_provider=_staged_index,
        )

        self.assertEqual(result.lambda_stage, 2)
        trajectory = result.recommendation_trajectory
        self.assertEqual([point.cumulative_evaluations for point in trajectory], [5, 7, 9])
        self.assertEqual(
            [set(point.selected_arm_indices) for point in trajectory],
            [{0}, {0, 1}, {1}],
        )
        self.assertEqual(trajectory[-1].added_arm_indices, ())
        self.assertEqual(trajectory[-1].removed_arm_indices, (0,))
        self.assertEqual(trajectory[-1].removed_models, ("arm_0",))
        self.assertEqual(result.selected_models, ["arm_1"])

    def test_changes_only_preserves_sampling_cost_and_lambda_decay(self):
        problem = _problem(values=((0.9, 9.0), (0.6, 1.0), (0.2, 0.1)), n_questions=3)
        options = {
            "directions": ((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
            "prior_variance": 0.001,
            "obs_noise_variance": 1e-9,
            "index_provider": _staged_index,
        }
        baseline = _run(
            problem,
            recommendation_changes_only=False,
            **options,
        )
        variant = _run(problem, **options)

        self.assertEqual(_without_timing(variant.trace), _without_timing(baseline.trace))
        self.assertEqual(variant.observed_cells, baseline.observed_cells)
        self.assertEqual(variant.total_evaluations, baseline.total_evaluations)
        self.assertEqual(variant.total_cost, baseline.total_cost)
        self.assertEqual(variant.stop_reason, baseline.stop_reason)
        self.assertEqual(variant.current_lambda, baseline.current_lambda)
        self.assertEqual(variant.lambda_stage, baseline.lambda_stage)
        self.assertEqual(variant.lambda_stage, 2)
        memberships = [set(point.selected_arm_indices) for point in variant.recommendation_trajectory]
        self.assertTrue(all(left != right for left, right in zip([set()] + memberships, memberships)))



if __name__ == "__main__":
    unittest.main()
