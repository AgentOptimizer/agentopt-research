"""Behavioral coverage for the stateful, decaying-lambda replay."""

import json
import math
import unittest
from unittest import mock

import numpy as np

from agentopt.model_selection.radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryCache,
)
from agentopt.model_selection.radial_gittins_prewarm import (
    RadialGittinsPrewarmResult,
    RadialGittinsPrewarmStats,
)
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(values=((0.9, 9.0), (0.6, 1.0), (0.2, 0.1)), n_questions=3):
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
        "directions": ((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        "cost_reference_usd": 1.0,
        "prior_variance": 0.001,
        "obs_noise_variance": 1e-9,
        "effective_cost_bin_ratio": None,
        "anytime": True,
        "record_recommendation_trajectory": True,
        "recommendation_checkpoint_interval": 100,
        "index_provider": _staged_index,
        "seed": 1,
    }
    options.update(kwargs)
    return replay.simulate_radial_gittins(*(_problem() if problem is None else problem), **options)


class _CostDependentBoundary:
    def __init__(self, cost):
        self.cost = cost

    def boundary(self, *args):
        # At lambda=1 all unfinished indices are below completed utility;
        # at lambda=1/2 the remaining arm becomes worth completing.
        return 100.0 if self.cost >= 0.075 else -100.0


class AnytimeRadialGittinsTests(unittest.TestCase):
    def test_trace_can_be_disabled_without_changing_results_or_checkpoints(self):
        traced = _run()
        untraced = _run(record_trace=False)

        self.assertTrue(traced.trace)
        self.assertEqual(untraced.trace, [])
        self.assertEqual(traced.selected_models, untraced.selected_models)
        self.assertEqual(traced.stop_reason, untraced.stop_reason)
        self.assertEqual(traced.total_evaluations, untraced.total_evaluations)
        self.assertEqual(traced.total_cost, untraced.total_cost)
        self.assertEqual(
            traced.recommendation_trajectory,
            untraced.recommendation_trajectory,
        )
        timing_fields = {"stage_wall_time_seconds", "run_wall_time_seconds"}
        strip_timing = lambda events: [
            {key: value for key, value in event.items() if key not in timing_fields}
            for event in events
        ]
        self.assertEqual(
            strip_timing(traced.lambda_stop_events),
            strip_timing(untraced.lambda_stop_events),
        )

    def test_stage_timing_records_warm_start_stops_and_completion(self):
        result = _run()

        events = result.stage_timing_events
        self.assertEqual(events[0]["event"], "warm_start_complete")
        self.assertEqual(events[-1]["event"], "run_complete")
        stop_timings = [x for x in events if x["event"] == "lambda_stage_stop"]
        self.assertEqual(len(stop_timings), len(result.lambda_stop_events))
        self.assertTrue(all(x["stage_wall_time_seconds"] >= 0.0 for x in events))
        self.assertEqual(
            [x["run_wall_time_seconds"] for x in events],
            sorted(x["run_wall_time_seconds"] for x in events),
        )
        self.assertTrue(all(
            "stage_wall_time_seconds" in stop
            and "run_wall_time_seconds" in stop
            for stop in result.lambda_stop_events
        ))

    def test_omitted_directions_use_ten_for_anytime_and_nine_for_fixed(self):
        for anytime, expected in (
            (True, replay.DEFAULT_ANYTIME_DIRECTIONS),
            (False, replay.DEFAULT_DIRECTIONS),
        ):
            with self.subTest(anytime=anytime):
                result = replay.simulate_radial_gittins(
                    *_problem(),
                    batch_size=1,
                    anytime=anytime,
                    index_provider=_staged_index,
                    cost_reference_usd=1.0,
                    obs_noise_variance=1e-9,
                    effective_cost_bin_ratio=None,
                )
                self.assertEqual(result.params["directions"], [list(x) for x in expected])
                visited = {
                    tuple(event["direction"])
                    for event in result.trace if event["event"] == "direction_visit"
                }
                self.assertEqual(visited, set(expected))
                self.assertEqual((1.0, 0.0) in visited, anytime)
                self.assertEqual(len(expected), 10 if anytime else 9)

    def test_explicit_custom_directions_replace_mode_defaults(self):
        custom = ((0.8, 0.2), (0.0, 1.0))
        for anytime in (True, False):
            with self.subTest(anytime=anytime):
                result = _run(anytime=anytime, directions=custom)
                self.assertEqual(result.params["directions"], [list(x) for x in custom])
                self.assertEqual(
                    {tuple(event["direction"]) for event in result.trace if event["event"] == "direction_visit"},
                    set(custom),
                )

    def test_none_selects_defaults_and_cli_extra_directions_are_idempotent(self):
        result = _run(directions=None)
        self.assertEqual(
            result.params["directions"], [list(x) for x in replay.DEFAULT_ANYTIME_DIRECTIONS]
        )
        self.assertEqual(
            replay._cli_directions(anytime=True, extra_directions=((1.0, 0.0), (1.0, 0.0))),
            replay.DEFAULT_ANYTIME_DIRECTIONS,
        )
        self.assertEqual(
            replay._cli_directions(anytime=False, extra_directions=((1.0, 0.0),)),
            replay.DEFAULT_ANYTIME_DIRECTIONS,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            _run(directions=((1.0, 0.0), (1.0, 0.0)))

    def test_endpoint_ties_stay_within_tolerance_of_the_true_maximum(self):
        utilities = {0: 0.5, 1: 0.509, 2: 0.491}
        secondary = {0: -3.0, 1: -2.0, 2: -1.0}
        winner, utility = replay._best_direction_index(
            utilities,
            0.01,
            direction=(1.0, 0.0),
            posteriors={},
            endpoint_secondary_values=secondary,
        )
        # Arm two is close to the ordered incumbent (arm zero), but it is
        # outside tolerance of the true accuracy maximum (arm one).
        self.assertEqual((winner, utility), (1, 0.509))
        self.assertEqual(
            replay._best_direction_index(
                utilities, 0.01, direction=(0.5, 0.5), posteriors={}
            ),
            (0, 0.5),
        )

    def test_halves_after_a_complete_stopping_sweep_and_resumes(self):
        contexts = []

        def recording_index(context, arm_index):
            contexts.append((context.current_lambda, context.lambda_stage))
            return _staged_index(context, arm_index)

        result = _run(index_provider=recording_index)

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.current_lambda, 0.25)
        self.assertEqual(result.lambda_stage, 2)
        self.assertEqual(set(contexts), {(1.0, 0), (0.5, 1), (0.25, 2)})
        stop_positions = [
            index for index, event in enumerate(result.trace)
            if event["event"] == "lambda_stop"
        ]
        self.assertEqual(len(stop_positions), 2)
        for position in stop_positions:
            sweep = result.trace[position - 3:position]
            self.assertEqual(len({event["direction_index"] for event in sweep}), 3)
            self.assertTrue(all(event["direction_should_stop"] for event in sweep))
            self.assertTrue(all(event["selected_arm"] is None for event in sweep))
        pulls = [event for event in result.trace if event.get("selected_arm") is not None]
        self.assertEqual([event["selected_arm"] for event in pulls], [0, 0, 1, 1, 2, 2])
        self.assertFalse(any(event["forced_after_gittins_stop"] for event in pulls))

    def test_a_single_stopped_direction_does_not_reduce_lambda(self):
        models, questions, table = _problem(n_questions=6)
        # Arm zero needs one adaptive batch; the other arms need two.
        table[models[0]] = {question: table[models[0]][question] for question in questions[:4]}

        def direction_index(context, arm_index):
            target = context.direction_index if context.current_lambda == 1.0 else 2
            return 100.0 if arm_index == target else -100.0

        result = _run(
            (models, questions, table),
            batch_size=2,
            directions=((0.1, 0.9), (0.9, 0.1)),
            question_universe="per_arm",
            index_provider=direction_index,
        )
        visits = [event for event in result.trace if event["event"] == "direction_visit"]
        isolated_stop = next(
            index for index, event in enumerate(visits[:-1])
            if event["direction_should_stop"] and visits[index + 1]["selected_arm"] == 1
        )
        self.assertEqual(visits[isolated_stop]["current_lambda"], 1.0)
        self.assertEqual(visits[isolated_stop + 1]["current_lambda"], 1.0)
        self.assertEqual(result.lambda_stage, 1)

    def test_continuation_retains_calibration_posteriors_and_unique_observations(self):
        with mock.patch.object(
            replay, "fit_empirical_bayes_warm_start",
            wraps=replay.fit_empirical_bayes_warm_start,
        ) as calibrate:
            result = _run()

        self.assertEqual(calibrate.call_count, 1)
        self.assertEqual(result.total_evaluations, 9)
        self.assertEqual(len(result.observed_cells), 9)
        self.assertEqual(len(set(result.observed_cells)), 9)
        self.assertEqual(sum(event["event"] == "warm_start" for event in result.trace), 3)
        warm = {
            event["arm_index"]: event
            for event in result.trace if event["event"] == "warm_start"
        }
        for arm_index in (1, 2):
            first_pull = next(event for event in result.trace if event.get("selected_arm") == arm_index)
            np.testing.assert_array_equal(first_pull["posterior_mean_before"], warm[arm_index]["posterior_mean_after"])
            np.testing.assert_array_equal(first_pull["posterior_var_before"], warm[arm_index]["posterior_var_after"])
        for summary in result.model_results:
            self.assertEqual(summary.n_batches, 3)
            self.assertEqual(summary.n_samples_evaluated, 3)
            self.assertTrue(summary.completed)
        self.assertAlmostEqual(result.total_cost, 30.3)

    def test_new_recommendations_survive_downsampling_and_have_cost_percentages(self):
        result = _run()
        trajectory = result.recommendation_trajectory
        self.assertTrue(all(point.is_deployable for point in trajectory))
        self.assertEqual(trajectory[0].selected_arm_indices, ())
        self.assertEqual(trajectory[0].hypervolume, 0.0)
        self.assertTrue(math.isinf(trajectory[0].generational_distance))
        self.assertTrue(math.isinf(trajectory[0].inverted_generational_distance))

        additions = [point for point in trajectory if point.event == "recommendation_added"]
        self.assertEqual([point.cumulative_evaluations for point in additions], [5, 7, 9])
        self.assertEqual([point.added_arm_indices for point in additions], [(0,), (1,), (2,)])
        self.assertEqual([point.added_models for point in additions], [("arm_0",), ("arm_1",), ("arm_2",)])
        np.testing.assert_allclose(
            [point.budget_fraction for point in additions], [28.1 / 30.3, 30.1 / 30.3, 1.0]
        )
        self.assertNotAlmostEqual(additions[0].budget_fraction, additions[0].cumulative_evaluations / 9)
        for point in trajectory:
            self.assertLessEqual(set(point.selected_arm_indices), set(point.completed_arm_indices))
            self.assertAlmostEqual(point.budget_fraction, point.cumulative_search_cost_usd / 30.3)
        self.assertAlmostEqual(additions[0].hypervolume, 0.09)
        self.assertTrue(all(point.generational_distance == 0 for point in additions))
        self.assertGreater(additions[0].inverted_generational_distance, 0)
        self.assertAlmostEqual(trajectory[-1].inverted_generational_distance, 0)
        self.assertAlmostEqual(trajectory[-1].hypervolume, result.ground_truth_hypervolume)

    def test_replacing_a_recommendation_records_the_new_identity(self):
        result = _run(
            _problem(values=((0.3, 0.1), (0.9, 0.05))),
            directions=((0.5, 0.5),),
        )
        additions = [point for point in result.recommendation_trajectory if point.event == "recommendation_added"]
        self.assertEqual([point.selected_arm_indices for point in additions], [(0,), (1,)])
        self.assertEqual([point.added_arm_indices for point in additions], [(0,), (1,)])

    def test_removal_only_archive_changes_survive_downsampling(self):
        models, questions, table = _problem(values=((0.3, 1.0), (0.8, 10.0), (0.4, 100.0)))
        # Arm two eventually wins the cost direction because of its two
        # free questions, but its high raw mean cost makes arm one dominate
        # it. Its completion removes arm zero without adding a recommendation.
        for question, cost in zip(questions, (0.0, 0.0, 300.0)):
            table[models[2]][question].cost = cost

        result = _run(
            (models, questions, table),
            directions=((0.1, 0.9), (0.9, 0.1)),
            expected_batch_cost_usd=(1.0, 10.0, 100.0),
        )
        trajectory = result.recommendation_trajectory
        changes = [point for point in trajectory if point.event == "recommendation_changed"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].cumulative_evaluations, 9)
        self.assertEqual(changes[0].selected_arm_indices, (1,))
        self.assertEqual(changes[0].added_arm_indices, ())
        self.assertEqual(changes[0].added_models, ())
        previous_addition = [point for point in trajectory if point.event == "recommendation_added"][-1]
        self.assertEqual(set(previous_addition.selected_arm_indices), {0, 1})
        self.assertLess(changes[0].hypervolume, previous_addition.hypervolume)
        self.assertEqual(result.selected_models, [models[1]])

    def test_question_budget_applies_across_lambda_stages(self):
        result = _run(max_total_question_evaluations=6)
        self.assertEqual(result.stop_reason, "question_budget")
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(result.current_lambda, 0.5)
        self.assertEqual(result.selected_models, ["arm_0"])
        self.assertEqual(result.recommendation_trajectory[-1].completed_arm_indices, (0,))
        self.assertEqual(len(set(result.observed_cells)), 6)
        self.assertTrue(result.gittins_stop_triggered)
        self.assertFalse(result.halted_by_gittins)

    def test_hard_dollar_budget_is_not_discounted_when_lambda_halves(self):
        result = _run(
            max_search_cost_usd=29.25,
            guaranteed_batch_cost_usd=(9.0, 1.0, 0.1),
        )
        self.assertEqual(result.stop_reason, "search_cost_budget")
        self.assertEqual(result.total_evaluations, 6)
        self.assertAlmostEqual(result.total_cost, 29.1)
        self.assertEqual(result.current_lambda, 0.5)
        self.assertEqual(result.cost_budget_guard, "hard_guaranteed_bound")
        self.assertEqual(result.cost_budget_overshoot_usd, 0.0)
        self.assertLessEqual(result.total_cost, 29.25)

    def test_cost_changes_refresh_cached_indices_for_untouched_arms(self):
        class RecordingCache:
            def __init__(self):
                self.costs = []

            def __len__(self):
                return 0

            def get(self, **kwargs):
                self.costs.append(kwargs["effective_pull_cost"])
                return _CostDependentBoundary(kwargs["effective_pull_cost"])

        cache = RecordingCache()
        result = _run(
            _problem(values=((0.5, 0.1), (0.5, 0.1))),
            directions=((0.5, 0.5),),
            index_provider=None,
            boundary_cache=cache,
        )
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(result.current_lambda, 0.5)
        self.assertEqual(set(cache.costs), {0.1, 0.05})

    def test_optimized_prewarm_rebuilds_tables_for_the_new_cost(self):
        requests_by_stage = []

        def prewarm(requests, **kwargs):
            requests = tuple(requests)
            requests_by_stage.append(tuple(request.effective_pull_cost for request in requests))
            return RadialGittinsPrewarmResult(
                tables=tuple(_CostDependentBoundary(request.effective_pull_cost) for request in requests),
                stats=RadialGittinsPrewarmStats(requested=len(requests)),
            )

        with mock.patch.object(replay, "prewarm_radial_gittins_boundaries", side_effect=prewarm):
            result = _run(
                _problem(values=((0.5, 0.1), (0.5, 0.1))),
                directions=((0.5, 0.5),),
                index_provider=None,
                boundary_cache=RadialGittinsBoundaryCache(),
            )

        self.assertEqual(requests_by_stage, [(0.1, 0.1), (0.05,)])
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 6)

    def test_lambda_halves_the_frozen_eta_scaled_cost_bins(self):
        costs_by_lambda = {}

        def record_costs(context, arm_index):
            costs_by_lambda[context.current_lambda] = context.effective_pull_costs
            return _staged_index(context, arm_index)

        result = _run(
            search_cost_scale_eta=3.0,
            effective_cost_bin_ratio=3.0,
            effective_cost_bin_anchor=0.1,
            index_provider=record_costs,
        )
        np.testing.assert_allclose(costs_by_lambda[1.0], (24.3, 2.7, 0.3))
        for current_lambda in (0.5, 0.25):
            np.testing.assert_allclose(
                costs_by_lambda[current_lambda],
                np.asarray(costs_by_lambda[1.0]) * current_lambda,
            )
        # Lambda discounts the acquisition penalty, never realized dollars.
        self.assertAlmostEqual(result.total_cost, 30.3)

    def test_scalar_boundary_failure_widens_grid_without_replaying_observations(self):
        attempted_grids = []

        class FailingOnceCache:
            def __len__(self):
                return 0

            def get(self, **kwargs):
                if kwargs["effective_pull_cost"] == 0.05:
                    attempted_grids.append(kwargs["grid"])
                    if len(attempted_grids) == 1:
                        raise BoundaryGridError("scripted root outside grid")
                return _CostDependentBoundary(kwargs["effective_pull_cost"])

        result = _run(
            _problem(values=((0.5, 0.1), (0.5, 0.1))),
            directions=((0.5, 0.5),),
            index_provider=None,
            boundary_cache=FailingOnceCache(),
        )

        self.assertLess(attempted_grids[1].z_min, attempted_grids[0].z_min)
        self.assertGreater(attempted_grids[1].z_max, attempted_grids[0].z_max)
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(len(set(result.observed_cells)), 6)
        self.assertEqual(sum(event["event"] == "warm_start" for event in result.trace), 2)
        expansions = result.params["boundary_grid_expansions"]
        self.assertEqual(len(expansions), 1)
        self.assertEqual(expansions[0]["lambda_stage"], 1)
        self.assertEqual(expansions[0]["current_lambda"], 0.5)
        self.assertEqual(expansions[0]["new_z_padding"], 2 * expansions[0]["previous_z_padding"])
        self.assertEqual(expansions[0]["reason"], "scripted root outside grid")

    def test_prewarm_boundary_failure_retries_with_wider_requests(self):
        attempted_requests = []

        def failing_once_prewarm(requests, **kwargs):
            requests = tuple(requests)
            if requests[0].effective_pull_cost == 0.05:
                attempted_requests.append(requests)
                if len(attempted_requests) == 1:
                    raise BoundaryGridError("scripted prewarm grid failure")
            return RadialGittinsPrewarmResult(
                tables=tuple(_CostDependentBoundary(request.effective_pull_cost) for request in requests),
                stats=RadialGittinsPrewarmStats(requested=len(requests)),
            )

        with mock.patch.object(replay, "prewarm_radial_gittins_boundaries", side_effect=failing_once_prewarm):
            result = _run(
                _problem(values=((0.5, 0.1), (0.5, 0.1))),
                directions=((0.5, 0.5),),
                index_provider=None,
                boundary_cache=RadialGittinsBoundaryCache(),
            )

        self.assertEqual(len(attempted_requests), 2)
        original, retried = attempted_requests[0][0], attempted_requests[1][0]
        self.assertEqual(original.effective_pull_cost, retried.effective_pull_cost)
        self.assertEqual(original.horizon, retried.horizon)
        self.assertLess(retried.grid.z_min, original.grid.z_min)
        self.assertGreater(retried.grid.z_max, original.grid.z_max)
        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 6)
        self.assertEqual(len(set(result.observed_cells)), 6)
        self.assertEqual(sum(event["event"] == "warm_start" for event in result.trace), 2)
        self.assertEqual(len(result.params["boundary_grid_expansions"]), 1)

    def test_permanent_grid_failure_has_bounded_retries_and_fixed_mode_fails_immediately(self):
        for optimized in (False, True):
            for anytime, expected_attempts in ((True, 5), (False, 1)):
                with self.subTest(optimized_prewarm=optimized, anytime=anytime):
                    attempted_grids = []

                    class FailingCache:
                        def __len__(self):
                            return 0

                        def get(self, **kwargs):
                            attempted_grids.append(kwargs["grid"])
                            raise BoundaryGridError("persistent grid failure")

                    def failing_prewarm(requests, **kwargs):
                        attempted_grids.append(tuple(requests)[0].grid)
                        raise BoundaryGridError("persistent grid failure")

                    cache = RadialGittinsBoundaryCache() if optimized else FailingCache()
                    with mock.patch.object(replay, "prewarm_radial_gittins_boundaries", side_effect=failing_prewarm):
                        with self.assertRaisesRegex(BoundaryGridError, "persistent grid failure"):
                            _run(
                                _problem(values=((0.5, 0.1), (0.5, 0.1))),
                                directions=((0.5, 0.5),),
                                index_provider=None,
                                boundary_cache=cache,
                                anytime=anytime,
                            )

                    self.assertEqual(len(attempted_grids), expected_attempts)
                    for original, retried in zip(attempted_grids, attempted_grids[1:]):
                        self.assertLess(retried.z_min, original.z_min)
                        self.assertGreater(retried.z_max, original.z_max)

    def test_anytime_disabled_preserves_fixed_lambda_stopping(self):
        kwargs = {
            "batch_size": 1,
            "directions": ((0.5, 0.5),),
            "index_provider": lambda context, arm_index: 100.0 if arm_index == 0 else -100.0,
            "seed": 1,
        }
        default = replay.simulate_radial_gittins(*_problem(), **kwargs)
        explicit_fixed = replay.simulate_radial_gittins(*_problem(), anytime=False, **kwargs)
        self.assertEqual(default.trace, explicit_fixed.trace)
        self.assertEqual(default.observed_cells, explicit_fixed.observed_cells)
        self.assertEqual(default.stop_reason, "all_directions_gittins_stop")
        self.assertEqual(default.total_evaluations, 5)
        self.assertEqual(default.lambda_stage, 0)
        self.assertFalse(any(event["event"] == "lambda_stop" for event in default.trace))

    def test_numerically_negligible_lambda_terminates_without_forced_pulls(self):
        result = _run(
            lambda_initial=1e-20,
            stop_tolerance=0.0,
            index_provider=lambda context, arm_index: 100.0 if arm_index == 0 else -100.0,
        )
        self.assertEqual(result.stop_reason, "lambda_numerical_floor")
        self.assertEqual(result.total_evaluations, 5)
        self.assertEqual(result.lambda_stage, 0)
        self.assertEqual(result.current_lambda, 1e-20)
        self.assertLess(len(result.trace), 20)
        self.assertEqual(result.selected_models, ["arm_0"])
        # Empty initial archives have infinite GD/IGD; exports must still be
        # valid JSON for plotting and checkpoint consumers.
        json.dumps(replay._jsonable_result(result), allow_nan=False)

    def test_invalid_lambda_schedule_is_rejected(self):
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(lambda_initial=value):
                with self.assertRaisesRegex(ValueError, "lambda_initial"):
                    _run(lambda_initial=value)
        for value in (0.0, -0.5, 1.0, 2.0, float("inf"), float("nan")):
            with self.subTest(lambda_decay=value):
                with self.assertRaisesRegex(ValueError, "lambda_decay"):
                    _run(lambda_decay=value)


if __name__ == "__main__":
    unittest.main()
