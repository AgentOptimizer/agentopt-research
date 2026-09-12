"""Conservative recommendation scores must not alter acquisition or stopping."""

import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import GaussianVectorPosterior
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _posterior(mean, std):
    return GaussianVectorPosterior(
        mean=np.asarray(mean, dtype=float),
        var=np.square(std),
        n_batches=1,
        n_questions=1,
    )


def _checkpoint(posteriors, *, directions=((1.0, 0.0),), beta=1.0, **overrides):
    arms = tuple(sorted(posteriors))
    raw = np.asarray([(0.9 - 0.1 * i, 0.1 + i) for i in arms])
    truth = np.column_stack((raw[:, 0], 1.0 / (1.0 + raw[:, 1])))
    options = dict(
        posteriors=posteriors,
        directions=directions,
        models=[f"arm_{i}" for i in arms],
        reference_point=(0.0, 0.0),
        stop_tolerance=1e-9,
        truth_vectors=truth,
        truth_front=truth[replay.nondominated_indices(truth)],
        raw_truth_vectors=raw,
        observed_scores={i: [raw[i, 0]] for i in arms},
        observed_costs={i: [raw[i, 1]] for i in arms},
        ground_truth_hv=replay.hypervolume_2d(truth),
        total_evaluations=len(arms),
        total_cost=1.0,
        bruteforce_search_cost_usd=20.0,
        event="after_warm_start",
        completed_arms=(),
        archive_scope="lcb",
        recommendation_rule="lcb",
        recommendation_beta=beta,
    )
    options.update(overrides)
    return replay._recommendation_checkpoint(**options)


def _problem(values=((0.8, 0.1), (0.7, 0.1)), n_questions=8):
    models = [f"arm_{i}" for i in range(len(values))]
    questions = list(range(n_questions))
    table = {
        model: {
            q: SampleResult(
                score=score, cost=cost, latency_seconds=0.1,
                input_tokens={}, output_tokens={},
            )
            for q in questions
        }
        for model, (score, cost) in zip(models, values)
    }
    return models, questions, table


def _run(problem=None, **overrides):
    options = dict(
        batch_size=1,
        directions=((1.0, 0.0),),
        anytime=True,
        recommendation_rule="lcb",
        recommendation_beta=1.0,
        record_recommendation_trajectory=True,
        recommendation_checkpoint_interval=100,
        recommendation_changes_only=True,
        prior_variance=1.0,
        obs_noise_variance=1.0,
        cost_reference_usd=1.0,
        effective_cost_bin_ratio=None,
        index_provider=lambda context, arm: 100.0 if arm == 1 else -100.0,
        seed=42,
    )
    options.update(overrides)
    return replay.simulate_radial_gittins(
        *(_problem() if problem is None else problem), **options,
    )


def _without_timing(events):
    timing_fields = {"stage_wall_time_seconds", "run_wall_time_seconds"}
    return [{key: value for key, value in event.items() if key not in timing_fields}
            for event in events]


class LcbRecommendationTests(unittest.TestCase):
    def test_uncertain_high_mean_loses_to_more_certain_arm_and_beta_zero_reverses_it(self):
        posteriors = {
            0: _posterior((0.9, 0.7), (0.4, 0.01)),
            1: _posterior((0.75, 0.7), (0.01, 0.01)),
        }
        conservative = _checkpoint(posteriors)
        mean_only = _checkpoint(posteriors, beta=0.0)

        self.assertEqual(conservative.selected_arm_indices, (1,))
        self.assertEqual(mean_only.selected_arm_indices, (0,))
        self.assertEqual(conservative.completed_arm_indices, ())
        self.assertEqual(conservative.archive_scope, "lcb")
        self.assertEqual(conservative.recommendation_rule, "lcb")
        self.assertEqual(conservative.recommendation_beta, 1.0)
        np.testing.assert_allclose(
            conservative.recommendation_desirability_vectors, ((0.74, 0.69),),
        )

    def test_cost_desirability_is_penalized_downward(self):
        posteriors = {
            0: _posterior((0.8, 0.8), (0.01, 0.3)),
            1: _posterior((0.8, 0.6), (0.01, 0.01)),
        }
        point = _checkpoint(posteriors, directions=((0.0, 1.0),))

        self.assertEqual(point.selected_arm_indices, (1,))
        np.testing.assert_allclose(
            point.recommendation_desirability_vectors, ((0.79, 0.59),),
        )

    def test_endpoint_ties_use_conservative_inactive_coordinate_then_arm_index(self):
        posteriors = {
            0: _posterior((0.7, 0.8), (0.1, 0.4)),
            1: _posterior((0.7, 0.7), (0.1, 0.1)),
            2: _posterior((0.7, 0.7), (0.1, 0.1)),
        }
        # Observed raw cost favors arm zero; the conservative secondary
        # coordinate favors arms one and two, with the stable tie going to one.
        point = _checkpoint(posteriors)
        self.assertEqual(point.direction_winner_arm_indices, (1,))

    def test_componentwise_scores_are_not_clipped_or_replaced_by_expected_radial_utility(self):
        posteriors = {
            0: _posterior((0.1, 0.2), (0.2, 0.3)),
            1: _posterior((0.9, 0.8), (0.1, 0.3)),
        }
        vectors = replay.recommendation_lcb_vectors(posteriors, (1, 0), 1.0)
        np.testing.assert_allclose(vectors, ((0.8, 0.5), (-0.1, -0.1)))
        scores = replay.recommendation_lcb_utilities(vectors, (0.5, 0.5), (0.0, 0.0))
        np.testing.assert_allclose(scores, (0.5, -0.1))
        endpoint_scores = replay.recommendation_lcb_utilities(vectors, (1.0, 0.0), (0.2, 0.0))
        np.testing.assert_allclose(endpoint_scores, (0.6, -0.3))

    def test_raw_and_oracle_dominance_do_not_replace_conservative_frontier(self):
        posteriors = {
            0: _posterior((0.9, 0.3), (0.1, 0.1)),
            1: _posterior((0.3, 0.9), (0.1, 0.1)),
        }
        point = _checkpoint(posteriors, directions=((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(point.selected_arm_indices, (0, 1))
        self.assertEqual(point.online_raw_archive_arm_indices, (0,))
        self.assertEqual(point.oracle_raw_winner_archive_arm_indices, (0,))
        np.testing.assert_allclose(
            point.recommendation_desirability_vectors, ((0.8, 0.2), (0.2, 0.8)),
        )

        reversed_truth = np.asarray(((0.1, 9.0), (0.95, 0.01)))
        changed = _checkpoint(
            posteriors, directions=((1.0, 0.0), (0.0, 1.0)),
            raw_truth_vectors=reversed_truth,
            observed_scores={0: [0.1], 1: [0.95]},
            observed_costs={0: [9.0], 1: [0.01]},
        )
        self.assertEqual(changed.selected_arm_indices, point.selected_arm_indices)
        self.assertEqual(changed.online_raw_archive_arm_indices, (1,))
        self.assertEqual(changed.oracle_raw_winner_archive_arm_indices, (1,))

    def test_deduplicated_direction_winners_are_filtered_in_lcb_space(self):
        posteriors = {
            0: _posterior((0.7, 0.7), (0.1, 0.1)),
            1: _posterior((0.7, 0.9), (0.1, 0.1)),
        }
        # The accuracy-limited interior ties and chooses arm zero. The cost
        # endpoint chooses arm one, which dominates zero in conservative space.
        point = _checkpoint(posteriors, directions=((0.9, 0.1), (0.0, 1.0), (1.0, 0.0)))
        self.assertEqual(point.direction_winner_arm_indices, (0, 1))
        self.assertEqual(point.selected_arm_indices, (1,))
        self.assertEqual(point.online_raw_archive_arm_indices, (0,))

    def test_partial_recommendation_changes_before_completion_and_matches_final_result(self):
        result = _run(max_total_question_evaluations=4)

        self.assertEqual(result.stop_reason, "question_budget")
        self.assertFalse(any(arm.completed for arm in result.model_results))
        points = result.recommendation_trajectory
        self.assertEqual([point.cumulative_evaluations for point in points], [2, 3])
        self.assertEqual([point.selected_arm_indices for point in points], [(0,), (1,)])
        self.assertEqual(points[-1].added_arm_indices, (1,))
        self.assertEqual(points[-1].removed_arm_indices, (0,))
        self.assertTrue(all(point.archive_scope == "lcb" for point in points))
        self.assertTrue(all(point.completed_arm_indices == () for point in points))
        self.assertEqual(result.selected_models, ["arm_1"])
        self.assertEqual(result.recommendation_final_snapshot.selected_models, ("arm_1",))
        self.assertEqual(result.recommendation_final_snapshot.cumulative_evaluations, 4)
        self.assertEqual([winner.arm_index for winner in result.direction_winners], [1])

    def test_lcb_output_preserves_exact_sampling_costs_and_eta_events(self):
        models, questions, table = _problem(((0.5, 0.1),) * 3, n_questions=3)
        table[models[0]] = {questions[0]: table[models[0]][questions[0]]}

        def staged_index(context, arm):
            target = 1 if context.direction_index == 1 else (
                2 if context.current_lambda <= 0.5 else None
            )
            return 100.0 if arm == target else -100.0

        settings = dict(
            directions=((0.2, 0.8), (0.8, 0.2)),
            question_universe="per_arm",
            eta_decay_schedule="direction_stop",
            index_provider=staged_index,
            prior_variance=0.001,
            obs_noise_variance=1e-9,
        )
        baseline = _run((models, questions, table), recommendation_rule="completed_only", **settings)
        self.assertTrue(baseline.direction_eta_events)
        for rule in ("lcb", "hybrid_lcb"):
            with self.subTest(rule=rule):
                conservative = _run(
                    (models, questions, table), recommendation_rule=rule, **settings,
                )
                self.assertEqual(_without_timing(conservative.trace), _without_timing(baseline.trace))
                self.assertEqual(conservative.direction_eta_events, baseline.direction_eta_events)
                self.assertEqual(conservative.lambda_stop_events, baseline.lambda_stop_events)
                self.assertEqual(conservative.direction_eta_multipliers, baseline.direction_eta_multipliers)
                self.assertEqual(conservative.direction_eta_stages, baseline.direction_eta_stages)
                self.assertEqual(conservative.observed_cells, baseline.observed_cells)
                self.assertEqual(len(set(conservative.observed_cells)), conservative.total_evaluations)
                self.assertEqual(conservative.total_evaluations, baseline.total_evaluations)
                self.assertEqual(conservative.total_cost, baseline.total_cost)
                self.assertEqual(conservative.stop_reason, baseline.stop_reason)

    def test_hybrid_completed_mean_prevents_c61_uncertainty_penalty_reversal(self):
        # C61: partial 13813's uniform LCB beats completed 13814's LCB,
        # but not its posterior mean. Raw/oracle coordinates are not used.
        posteriors = {
            0: _posterior((0.4444045911, 0.7), (0.03787770095, 0.02)),
            1: _posterior((0.4158010063, 0.8), (0.01274014573, 0.03)),
        }
        before = {arm: (p.mean.copy(), p.var.copy()) for arm, p in posteriors.items()}
        original = _checkpoint(posteriors, completed_arms=(1,))
        hybrid = _checkpoint(posteriors, completed_arms=(1,), recommendation_rule="hybrid_lcb")

        self.assertEqual(original.selected_arm_indices, (0,))
        self.assertEqual(hybrid.selected_arm_indices, (1,))
        self.assertEqual(hybrid.recommendation_rule, "hybrid_lcb")
        np.testing.assert_allclose(hybrid.recommendation_desirability_vectors, ((0.4158010063, 0.8),))
        for arm, posterior in posteriors.items():
            np.testing.assert_array_equal(posterior.mean, before[arm][0])
            np.testing.assert_array_equal(posterior.var, before[arm][1])

    def test_hybrid_mixed_ties_and_pareto_filter_use_same_vectors(self):
        posteriors = {
            0: _posterior((0.7, 0.7), (0.1, 0.1)),
            1: _posterior((0.6, 0.8), (0.1, 0.4)),
        }
        point = _checkpoint(
            posteriors, recommendation_rule="hybrid_lcb", completed_arms=(1,),
            directions=((0.9, 0.1), (0.0, 1.0), (1.0, 0.0)),
        )
        self.assertEqual(point.direction_winner_arm_indices, (0, 1))
        self.assertEqual(point.selected_arm_indices, (1,))
        np.testing.assert_allclose(point.recommendation_desirability_vectors, ((0.6, 0.6), (0.6, 0.8)))
        endpoint = _checkpoint(posteriors, recommendation_rule="hybrid_lcb", completed_arms=(1,))
        self.assertEqual(endpoint.selected_arm_indices, (1,))

    def test_hybrid_switches_on_completion_batch_and_final_result_agrees(self):
        problem = _problem(((0.9, 0.1), (0.77, 0.1)), n_questions=3)
        settings = dict(max_total_question_evaluations=4, prior_variance=0.04, obs_noise_variance=0.04)
        original = _run(problem, **settings)
        hybrid = _run(problem, recommendation_rule="hybrid_lcb", **settings)

        self.assertEqual(original.selected_models, ["arm_0"])
        self.assertEqual(
            [(p.cumulative_evaluations, p.selected_arm_indices, p.completed_arm_indices)
             for p in hybrid.recommendation_trajectory],
            [(2, (0,), ()), (4, (1,), (1,))],
        )
        self.assertEqual(hybrid.selected_models, ["arm_1"])
        self.assertEqual(hybrid.recommendation_final_snapshot.selected_models, ("arm_1",))
        self.assertEqual([w.arm_index for w in hybrid.direction_winners], [1])
        np.testing.assert_allclose(
            hybrid.recommendation_final_snapshot.recommendation_desirability_vectors,
            [hybrid.model_results[1].posterior_mean],
        )
        self.assertEqual(hybrid.params["recommendation_completed_std_penalty"], 0.0)
        self.assertEqual(hybrid.params["stopping_eligibility"], "completed_only")

    def test_hybrid_uses_per_arm_completion_including_warm_start(self):
        models, questions, table = _problem(((0.77, 0.1), (0.9, 0.1)), n_questions=3)
        table[models[0]] = {questions[0]: table[models[0]][questions[0]]}
        settings = dict(
            question_universe="per_arm", max_total_question_evaluations=2,
            prior_variance=0.04, obs_noise_variance=0.04,
        )
        original = _run((models, questions, table), **settings)
        hybrid = _run((models, questions, table), recommendation_rule="hybrid_lcb", **settings)
        warm = hybrid.recommendation_initial_snapshot
        self.assertEqual([a.n_samples_evaluated for a in hybrid.model_results], [1, 1])
        self.assertEqual(warm.completed_arm_indices, (0,))
        self.assertEqual(original.selected_models, ["arm_1"])
        self.assertEqual(warm.selected_arm_indices, (0,))
        self.assertEqual(hybrid.selected_models, ["arm_0"])

    def test_invalid_rule_and_beta_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "recommendation_rule"):
            _run(recommendation_rule="unknown")
        for beta in (-1.0, float("inf"), float("nan")):
            with self.subTest(beta=beta):
                with self.assertRaisesRegex(ValueError, "recommendation_beta|beta"):
                    _run(recommendation_beta=beta)
        with self.assertRaisesRegex(ValueError, "recommendation_beta|beta"):
            _checkpoint({0: _posterior((0.5, 0.5), (0.1, 0.1))}, beta=-0.1)


if __name__ == "__main__":
    unittest.main()
