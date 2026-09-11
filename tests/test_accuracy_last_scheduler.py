"""Behavioral coverage for deferring the accuracy endpoint in each lambda stage."""

import unittest

from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


ACCURACY = (1.0, 0.0)
PRIMARY = (0.2, 0.8)
OTHER_PRIMARY = (0.8, 0.2)


def _problem(n_arms=4, n_questions=3, values=None):
    models = [f"arm_{index}" for index in range(n_arms)]
    questions = list(range(n_questions))
    values = values or [(0.5, 0.1)] * n_arms
    table = {
        model: {
            question: SampleResult(
                score=score, cost=cost, latency_seconds=0.1,
                input_tokens={}, output_tokens={},
            )
            for question in questions
        }
        for model, (score, cost) in zip(models, values)
    }
    return models, questions, table


def _target_index(context, arm_index):
    target = {PRIMARY: 0, OTHER_PRIMARY: 1, ACCURACY: 2}[context.direction]
    return 100.0 if arm_index == target else -100.0


def _run(problem=None, scheduler="accuracy_last", **overrides):
    options = dict(
        batch_size=1,
        directions=(ACCURACY, PRIMARY, OTHER_PRIMARY),
        cost_reference_usd=1.0,
        prior_variance=0.001,
        obs_noise_variance=1e-9,
        effective_cost_bin_ratio=None,
        anytime=False,
        index_provider=_target_index,
        seed=42,
    )
    if scheduler is not None:
        options["direction_scheduler"] = scheduler
    options.update(overrides)
    return replay.simulate_radial_gittins(
        *(_problem() if problem is None else problem), **options,
    )


def _visits(result):
    return [event for event in result.trace if event["event"] == "direction_visit"]


def _without_timing(events):
    timing_fields = {"stage_wall_time_seconds", "run_wall_time_seconds"}
    return [
        {key: value for key, value in event.items() if key not in timing_fields}
        for event in events
    ]


class AccuracyLastSchedulerTests(unittest.TestCase):
    def test_default_preserves_global_round_robin_in_custom_order(self):
        implicit = _run(scheduler=None)
        explicit = _run(scheduler="round_robin")
        self.assertEqual(_without_timing(implicit.trace), _without_timing(explicit.trace))
        self.assertEqual(implicit.observed_cells, explicit.observed_cells)
        self.assertEqual(implicit.params["direction_scheduler"], "round_robin")
        self.assertEqual(
            [event["direction_index"] for event in _visits(implicit)[:6]],
            [0, 1, 2, 0, 1, 2],
        )

    def test_custom_endpoint_order_waits_for_primary_stops_then_runs_contiguously(self):
        result = _run()
        visits = _visits(result)
        first_endpoint = next(
            index for index, event in enumerate(visits)
            if tuple(event["direction"]) == ACCURACY
        )
        primary_visits = visits[:first_endpoint]
        self.assertEqual(
            [event["direction_index"] for event in primary_visits],
            [1, 2, 1, 2, 1, 2],
        )
        self.assertTrue(all(event["direction_should_stop"] for event in primary_visits[-2:]))
        self.assertEqual(
            [event["selected_arm"] for event in visits[first_endpoint:first_endpoint + 2]],
            [2, 2],
        )
        self.assertTrue(all(
            tuple(event["direction"]) == ACCURACY
            for event in visits[first_endpoint:first_endpoint + 3]
        ))
        self.assertEqual(result.stop_reason, "all_directions_gittins_stop")
        self.assertEqual(result.total_evaluations, 10)
        self.assertFalse(result.lambda_stop_events)

    def test_endpoint_observations_reactivate_previously_stopped_primary(self):
        def index(context, arm_index):
            if context.direction == ACCURACY:
                target = 1
            else:
                target = 2 if 1 in context.completed_arms else 0
            return 100.0 if arm_index == target else -100.0

        result = _run(directions=(ACCURACY, PRIMARY), index_provider=index)
        visits = _visits(result)
        first_endpoint = next(
            index for index, event in enumerate(visits) if event["selected_arm"] == 1
        )
        self.assertTrue(visits[first_endpoint - 1]["direction_should_stop"])
        self.assertEqual(
            [event["selected_arm"] for event in visits if event["selected_arm"] is not None],
            [0, 0, 1, 1, 2, 2],
        )
        resumed_primary = [event for event in visits if event["selected_arm"] == 2]
        self.assertTrue(all(tuple(event["direction"]) == PRIMARY for event in resumed_primary))
        self.assertTrue(all(event["current_lambda"] == 1.0 for event in resumed_primary))
        self.assertEqual(result.stop_reason, "all_directions_gittins_stop")

    def test_primary_observation_invalidates_another_primarys_stop(self):
        models, questions, table = _problem()
        table[models[0]] = {question: table[models[0]][question] for question in questions[:2]}
        result = _run((models, questions, table), question_universe="per_arm")
        visits = _visits(result)
        self.assertEqual(tuple(visits[2]["direction"]), PRIMARY)
        self.assertTrue(visits[2]["direction_should_stop"])
        self.assertEqual(visits[3]["selected_arm"], 1)
        # The intervening observation requires a fresh primary stop before
        # the scheduler may pass control to the endpoint.
        self.assertEqual(tuple(visits[4]["direction"]), PRIMARY)
        self.assertTrue(visits[4]["direction_should_stop"])
        self.assertEqual(tuple(visits[5]["direction"]), OTHER_PRIMARY)
        self.assertTrue(visits[5]["direction_should_stop"])
        self.assertEqual(tuple(visits[6]["direction"]), ACCURACY)

    def test_lambda_decay_requires_fresh_stops_and_restarts_with_primary(self):
        def index(context, arm_index):
            target = 2 * context.lambda_stage + int(context.direction == ACCURACY)
            return 100.0 if arm_index == target else -100.0

        result = _run(
            directions=(ACCURACY, PRIMARY), anytime=True, index_provider=index,
        )
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.lambda_stage, 1)
        self.assertEqual(result.current_lambda, 0.5)
        self.assertEqual(len(result.lambda_stop_events), 1)
        stop_position = next(
            index for index, event in enumerate(result.trace) if event["event"] == "lambda_stop"
        )
        last_pull_position = max(
            index for index, event in enumerate(result.trace[:stop_position])
            if event.get("selected_arm") is not None
        )
        certificate = result.trace[last_pull_position + 1:stop_position]
        self.assertEqual({tuple(event["direction"]) for event in certificate}, {PRIMARY, ACCURACY})
        self.assertTrue(all(event["direction_should_stop"] for event in certificate))
        self.assertTrue(all(event["selected_arm"] is None for event in certificate))
        next_stage = [event for event in _visits(result) if event["lambda_stage"] == 1]
        self.assertEqual(tuple(next_stage[0]["direction"]), PRIMARY)
        self.assertEqual(next_stage[0]["selected_arm"], 2)
        self.assertEqual(len(result.observed_cells), 12)

    def test_without_accuracy_endpoint_matches_round_robin(self):
        for anytime in (False, True):
            with self.subTest(anytime=anytime):
                options = dict(
                    directions=(OTHER_PRIMARY, PRIMARY), anytime=anytime,
                )
                baseline = _run(scheduler="round_robin", **options)
                deferred = _run(**options)
                self.assertEqual(_without_timing(deferred.trace), _without_timing(baseline.trace))
                self.assertEqual(deferred.observed_cells, baseline.observed_cells)
                self.assertEqual(deferred.stop_reason, baseline.stop_reason)
                if anytime:
                    self.assertGreater(deferred.lambda_stage, 0)

    def test_endpoint_reuses_completed_arms_and_resumes_only_unseen_questions(self):
        models, questions, table = _problem(n_arms=3, n_questions=4)
        table[models[0]] = {questions[0]: table[models[0]][questions[0]]}
        scored_arms = []

        def index(context, arm_index):
            scored_arms.append((arm_index, context.completed_arms))
            wants_partial = context.direction == ACCURACY or context.adaptive_pulls[1] == 0
            return 100.0 if wants_partial and arm_index == 1 else -100.0

        result = _run(
            (models, questions, table), directions=(PRIMARY, ACCURACY),
            question_universe="per_arm", index_provider=index,
        )
        pulls = [event for event in _visits(result) if event["selected_arm"] is not None]
        self.assertEqual([tuple(event["direction"]) for event in pulls], [PRIMARY, ACCURACY, ACCURACY])
        self.assertTrue(all(event["selected_arm"] == 1 for event in pulls))
        self.assertTrue(all(arm not in completed for arm, completed in scored_arms))
        self.assertTrue(all(arm != 0 for arm, _ in scored_arms))
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(len(result.observed_cells), 6)
        self.assertEqual({question for arm, question in result.observed_cells if arm == 1}, set(questions))
        self.assertEqual(result.model_results[0].n_samples_evaluated, 1)
        self.assertEqual(result.model_results[1].n_samples_evaluated, 4)

    def test_question_and_hard_dollar_budgets_apply_inside_endpoint_phase(self):
        def index(context, arm_index):
            target = int(context.direction == ACCURACY)
            return 100.0 if arm_index == target else -100.0

        for budget, reason in (
            ({"max_total_question_evaluations": 7}, "question_budget"),
            ({"max_search_cost_usd": 0.75, "guaranteed_batch_cost_usd": 0.1}, "search_cost_budget"),
        ):
            with self.subTest(reason=reason):
                result = _run(
                    directions=(ACCURACY, PRIMARY), index_provider=index,
                    anytime=True, **budget,
                )
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(result.total_evaluations, 7)
                self.assertEqual(len(result.observed_cells), 7)
                self.assertAlmostEqual(result.total_cost, 0.7)
                pulls = [event for event in _visits(result) if event["selected_arm"] is not None]
                self.assertEqual(tuple(pulls[-1]["direction"]), ACCURACY)
                self.assertFalse(result.model_results[1].completed)
                self.assertEqual(result.lambda_stage, 0)
                if reason == "search_cost_budget":
                    self.assertEqual(result.cost_budget_overshoot_usd, 0.0)

    def test_endpoint_only_real_scalar_dp_preserves_replay_and_decay(self):
        problem = _problem(n_arms=2, values=[(0.6, 0.001), (0.95, 1.0)])
        options = dict(
            directions=(ACCURACY,), anytime=True, index_provider=None,
            cost_reference_usd=0.001, prior_variance=0.04,
            obs_noise_variance=1e-5, boundary_build_backend="scipy",
            boundary_grid=RadialGittinsGrid(z_size=65, delta_size=65, state_size=65),
        )
        baseline = _run(problem, scheduler="round_robin", **options)
        deferred = _run(problem, **options)
        self.assertEqual(_without_timing(deferred.trace), _without_timing(baseline.trace))
        self.assertEqual(deferred.observed_cells, baseline.observed_cells)
        self.assertEqual(deferred.stop_reason, "all_arms_completed")
        self.assertGreater(deferred.lambda_stage, 0)
        self.assertEqual(deferred.selected_models, ["arm_1"])

    def test_fixed_lambda_forced_continuation_returns_to_global_round_robin(self):
        result = _run(
            halt_on_gittins_stop=False, max_total_question_evaluations=15,
            problem=_problem(n_questions=4),
        )
        forced = [event for event in _visits(result) if event.get("forced_after_gittins_stop")]
        # The scripted targets finish after nine adaptive pulls; use one
        # remaining arm to keep the diagnostic continuation observable.
        self.assertTrue(result.gittins_stop_triggered)
        self.assertFalse(result.halted_by_gittins)
        self.assertEqual(len(forced), 2)
        for before, after in zip(forced, forced[1:]):
            self.assertEqual(after["direction_index"], (before["direction_index"] + 1) % 3)

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "direction_scheduler"):
            _run(scheduler="unknown")

    def test_untraced_scheduler_preserves_changes_with_checkpoint_target(self):
        def index(context, arm_index):
            target = 2 * context.lambda_stage + int(context.direction == ACCURACY)
            return 100.0 if arm_index == target else -100.0

        problem = _problem(values=[(0.3, 0.001), (0.6, 0.01), (0.8, 0.1), (0.95, 1.0)])
        options = dict(
            directions=(ACCURACY, PRIMARY), anytime=True, index_provider=index,
            record_recommendation_trajectory=True, recommendation_changes_only=True,
            recommendation_checkpoint_target=2,
        )
        traced = _run(problem, **options)
        untraced = _run(problem, record_trace=False, **options)

        self.assertTrue(traced.trace)
        self.assertEqual(untraced.trace, [])
        self.assertEqual(untraced.observed_cells, traced.observed_cells)
        self.assertEqual(untraced.selected_models, traced.selected_models)
        self.assertEqual(untraced.total_cost, traced.total_cost)
        self.assertEqual(untraced.stop_reason, "all_arms_completed")
        self.assertEqual(untraced.lambda_stage, 1)
        self.assertEqual(
            _without_timing(untraced.lambda_stop_events),
            _without_timing(traced.lambda_stop_events),
        )
        self.assertEqual(untraced.recommendation_trajectory, traced.recommendation_trajectory)
        self.assertEqual(untraced.params["recommendation_checkpoint_interval"], 4)
        self.assertEqual(
            [point.cumulative_evaluations for point in untraced.recommendation_trajectory],
            [6, 8, 10, 12],
        )
        self.assertTrue(all(
            set(point.selected_arm_indices) <= set(point.completed_arm_indices)
            for point in untraced.recommendation_trajectory
        ))
        self.assertEqual(
            untraced.recommendation_final_snapshot.selected_arm_indices,
            traced.recommendation_final_snapshot.selected_arm_indices,
        )


if __name__ == "__main__":
    unittest.main()
