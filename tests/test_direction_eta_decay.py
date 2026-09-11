"""Independent direction costs retain strict rotation and shared observations."""

import unittest

import numpy as np

from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


DIRECTIONS = ((0.2, 0.8), (0.8, 0.2))


def _problem(values=((0.5, 0.1), (0.5, 0.1), (0.5, 0.1)), n_questions=3, completed_warm=True):
    models = [f"arm_{index}" for index in range(len(values))]
    questions = list(range(n_questions))
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
    if completed_warm:
        table[models[0]] = {questions[0]: table[models[0]][questions[0]]}
    return models, questions, table


def _independent_index(context, arm_index):
    target = 1 if context.direction_index == 1 else (
        2 if context.current_lambda <= 0.5 else None
    )
    return 100.0 if arm_index == target else -100.0


def _run(problem=None, schedule="direction_stop", **overrides):
    options = dict(
        batch_size=1, directions=DIRECTIONS, question_universe="per_arm",
        cost_reference_usd=1.0, prior_variance=0.001, obs_noise_variance=1e-9,
        effective_cost_bin_ratio=None, anytime=True,
        direction_scheduler="round_robin", index_provider=_independent_index,
        record_recommendation_trajectory=True, recommendation_changes_only=True,
        seed=42,
    )
    if schedule is not None:
        options["eta_decay_schedule"] = schedule
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


class DirectionEtaDecayTests(unittest.TestCase):
    def test_default_global_schedule_preserves_existing_replay(self):
        implicit = _run(schedule=None)
        explicit = _run(schedule="global_stop")
        self.assertEqual(_without_timing(implicit.trace), _without_timing(explicit.trace))
        self.assertEqual(implicit.observed_cells, explicit.observed_cells)
        self.assertEqual(implicit.total_cost, explicit.total_cost)
        self.assertEqual(implicit.selected_models, explicit.selected_models)
        self.assertEqual(implicit.params["eta_decay_schedule"], "global_stop")
        self.assertTrue(implicit.lambda_stop_events)

    def test_stopping_decays_only_that_direction_and_waits_until_its_next_turn(self):
        contexts = {}

        def index(context, arm_index):
            contexts[context.global_step] = (
                context.direction_index, context.current_lambda,
                tuple(context.direction_eta_multipliers),
                tuple(context.direction_eta_stages), tuple(context.effective_pull_costs),
            )
            return _independent_index(context, arm_index)

        result = _run(index_provider=index)
        visits = _visits(result)
        self.assertEqual([event["direction_index"] for event in visits], [0, 1, 0, 1, 0])
        self.assertEqual([event["selected_arm"] for event in visits], [None, 1, 2, 1, 2])
        self.assertEqual([event["current_lambda"] for event in visits], [1.0, 1.0, 0.5, 1.0, 0.5])
        self.assertEqual(contexts[0][2:4], ((1.0, 1.0), (0, 0)))
        self.assertEqual(contexts[1][2:4], ((0.5, 1.0), (1, 0)))
        np.testing.assert_allclose(contexts[1][4], (0.1, 0.1, 0.1))
        np.testing.assert_allclose(contexts[2][4], (0.05, 0.05, 0.05))
        self.assertEqual(tuple(result.direction_eta_multipliers), (0.5, 1.0))
        self.assertEqual(tuple(result.direction_eta_stages), (1, 0))
        self.assertEqual(result.lambda_stop_events, [])
        self.assertEqual([event["event"] for event in result.direction_eta_events], ["direction_eta_decay"])
        self.assertEqual(result.direction_eta_events[0]["direction_index"], 0)
        self.assertEqual(result.direction_eta_events[0]["cumulative_evaluations"], 3)
        self.assertEqual(result.stop_reason, "all_arms_completed")

    def test_floor_stop_reactivates_after_another_direction_updates_shared_state(self):
        primary_variances = []

        def index(context, arm_index):
            if context.direction_index == 0:
                if arm_index == 1:
                    primary_variances.append(float(context.posteriors[1].var[0]))
                target = 2 if context.adaptive_pulls[1] else None
            else:
                target = 1
            return 100.0 if arm_index == target else -100.0

        result = _run(index_provider=index, lambda_initial=1e-20, stop_tolerance=0.0)
        visits = _visits(result)
        self.assertEqual([event["direction_index"] for event in visits], [0, 1, 0, 1, 0])
        self.assertEqual([event["selected_arm"] for event in visits], [None, 1, 2, 1, 2])
        self.assertLess(primary_variances[1], primary_variances[0])
        self.assertTrue(all(event["current_lambda"] == 1e-20 for event in visits))
        self.assertEqual(result.direction_eta_events[0]["event"], "direction_eta_floor_stop")
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 7)

    def test_floor_termination_requires_every_direction_without_new_observation(self):
        result = _run(
            index_provider=lambda context, arm_index: -100.0,
            lambda_initial=1e-20, stop_tolerance=0.0,
        )
        visits = _visits(result)
        self.assertEqual([event["direction_index"] for event in visits], [0, 1])
        self.assertTrue(all(event["direction_should_stop"] for event in visits))
        self.assertTrue(all(event["selected_arm"] is None for event in visits))
        self.assertEqual(result.stop_reason, "direction_eta_numerical_floor")
        self.assertEqual(result.total_evaluations, 3)
        self.assertEqual(len(result.direction_eta_events), 2)
        self.assertEqual({event["direction_index"] for event in result.direction_eta_events}, {0, 1})
        self.assertTrue(all(event["event"] == "direction_eta_floor_stop" for event in result.direction_eta_events))

    def test_repeated_independent_decay_reaches_a_bounded_numerical_floor(self):
        result = _run(index_provider=lambda context, arm_index: -100.0)
        visits = _visits(result)
        self.assertEqual(result.stop_reason, "direction_eta_numerical_floor")
        self.assertEqual(result.total_evaluations, 3)
        self.assertGreater(len(visits), 2)
        self.assertLess(len(visits), 120)
        self.assertTrue(all(event["selected_arm"] is None for event in visits))
        self.assertEqual([event["direction_index"] for event in visits], [index % 2 for index in range(len(visits))])
        self.assertEqual(result.direction_eta_stages[0], result.direction_eta_stages[1])
        self.assertGreater(result.direction_eta_stages[0], 0)

    def test_completed_only_changes_and_unique_cells_survive_untraced_replay(self):
        problem = _problem(values=((0.2, 0.001), (0.6, 0.01), (0.9, 0.1)))
        traced = _run(problem)
        untraced = _run(problem, record_trace=False)
        self.assertEqual(untraced.trace, [])
        self.assertEqual(untraced.observed_cells, traced.observed_cells)
        self.assertEqual(len(untraced.observed_cells), untraced.total_evaluations)
        self.assertEqual(len(set(untraced.observed_cells)), 7)
        self.assertEqual(untraced.recommendation_trajectory, traced.recommendation_trajectory)
        self.assertEqual(
            _without_timing(untraced.direction_eta_events),
            _without_timing(traced.direction_eta_events),
        )
        self.assertEqual(untraced.total_cost, traced.total_cost)
        trajectory = untraced.recommendation_trajectory
        self.assertEqual([point.cumulative_evaluations for point in trajectory], [3, 6, 7])
        self.assertTrue(all(
            set(point.selected_arm_indices) <= set(point.completed_arm_indices)
            for point in trajectory
        ))
        self.assertEqual(trajectory[0].selected_arm_indices, (0,))
        self.assertEqual(tuple(trajectory[-1].direction_eta_multipliers), (0.5, 1.0))
        memberships = [set(point.selected_arm_indices) for point in trajectory]
        self.assertTrue(all(left != right for left, right in zip(memberships, memberships[1:])))

    def test_question_and_hard_cost_budgets_are_not_discounted_with_eta(self):
        for budget, reason in (
            ({"max_total_question_evaluations": 5}, "question_budget"),
            ({"max_search_cost_usd": 0.55, "guaranteed_batch_cost_usd": 0.1}, "search_cost_budget"),
        ):
            with self.subTest(reason=reason):
                result = _run(**budget)
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(result.total_evaluations, 5)
                self.assertEqual(len(result.observed_cells), 5)
                self.assertAlmostEqual(result.total_cost, 0.5)
                self.assertEqual(result.selected_models, ["arm_0"])
                self.assertEqual(tuple(result.direction_eta_multipliers), (0.5, 1.0))
                self.assertEqual(result.cost_budget_overshoot_usd, 0.0)

    def test_decay_invalidates_only_its_direction_and_keeps_other_costs_current(self):
        class Boundary:
            def __init__(self, penalty):
                self.penalty = penalty

            def boundary(self, *args):
                return self.penalty

        class RecordingCache:
            def __init__(self):
                self.requests = []

            def __len__(self):
                return 0

            def get(self, **kwargs):
                direction = tuple(kwargs["direction"])
                cost = kwargs["effective_pull_cost"]
                self.requests.append((direction, cost))
                return Boundary(100.0 if direction == DIRECTIONS[0] and cost >= 0.0375 else -100.0)

        cache = RecordingCache()
        result = _run(index_provider=None, boundary_cache=cache)
        primary_costs = [cost for direction, cost in cache.requests if direction == DIRECTIONS[0]]
        other_costs = [cost for direction, cost in cache.requests if direction == DIRECTIONS[1]]
        np.testing.assert_allclose(primary_costs, (0.1, 0.1, 0.05, 0.05, 0.025))
        # The untouched remaining arm retains its index across the second
        # primary decay. Clearing both namespaces would add a fifth lookup.
        np.testing.assert_allclose(other_costs, (0.1, 0.1, 0.1, 0.1))
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(tuple(result.direction_eta_multipliers), (0.25, 1.0))
        self.assertEqual(result.total_evaluations, 7)

    def test_single_direction_matches_global_schedule_with_real_axis_dp(self):
        problem = _problem(values=((0.6, 0.001), (0.95, 1.0)), completed_warm=False)
        options = dict(
            directions=((1.0, 0.0),), index_provider=None,
            cost_reference_usd=0.001, prior_variance=0.04, obs_noise_variance=1e-5,
            boundary_build_backend="scipy",
            boundary_grid=RadialGittinsGrid(z_size=65, delta_size=65, state_size=65),
        )
        global_run = _run(problem, schedule="global_stop", **options)
        local_run = _run(problem, **options)
        self.assertEqual(local_run.observed_cells, global_run.observed_cells)
        self.assertEqual(local_run.total_cost, global_run.total_cost)
        self.assertEqual(local_run.selected_models, global_run.selected_models)
        self.assertEqual(local_run.stop_reason, "all_arms_completed")
        self.assertEqual(local_run.total_evaluations, 6)
        self.assertGreater(local_run.direction_eta_stages[0], 0)
        pull_fields = ("selected_arm", "question_ids", "current_lambda", "raw_effective_pull_cost")
        pulls = lambda result: [
            tuple(event[field] for field in pull_fields)
            for event in _visits(result) if event["selected_arm"] is not None
        ]
        self.assertEqual(pulls(local_run), pulls(global_run))

    def test_invalid_schedule_combinations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "eta_decay_schedule"):
            _run(schedule="unknown")
        with self.assertRaisesRegex(ValueError, "anytime"):
            _run(anytime=False)
        with self.assertRaisesRegex(ValueError, "round_robin|direction_scheduler"):
            _run(direction_scheduler="accuracy_last")


if __name__ == "__main__":
    unittest.main()
