import json
import unittest
from unittest import mock

import numpy as np

from agentopt.model_selection.radial_gittins_dp import RadialGittinsGrid
from experiments import offline_radial_gittins as radial_replay
from experiments.offline_radial_gittins import (
    _jsonable_result,
    hypervolume_2d,
    nondominated_indices,
    raw_archive_arm_indices,
    raw_nondominated_indices,
    simulate_radial_gittins,
    summarize_radial_multi_seed,
)
from experiments.offline_selector_sim_v2 import SampleResult


def _sample(score, cost):
    return SampleResult(
        score=score,
        latency_seconds=0.1,
        input_tokens={},
        output_tokens={},
        cost=cost,
    )


def _toy_frontier():
    models = ["A", "B", "C", "D"]
    datapoints = list(range(6))
    values = {
        "A": (0.9, 0.03),
        "B": (0.7, 0.01),
        "C": (0.3, 0.0025),
        "D": (0.5, 0.01),
    }
    table = {
        model: {
            question_id: _sample(score, cost)
            for question_id in datapoints
        }
        for model, (score, cost) in values.items()
    }
    return models, datapoints, table


def _target_provider(context, arm_index):
    targets = {0: 2, 1: 1, 2: 0}
    return 100.0 if arm_index == targets[context.direction_index] else -100.0


class ParetoMetricTests(unittest.TestCase):
    def test_nondominance_uses_maximize_desirability_convention(self):
        points = np.array(
            [
                [0.9, 0.25],
                [0.7, 0.5],
                [0.3, 0.8],
                [0.5, 0.5],
            ]
        )
        self.assertEqual(nondominated_indices(points), [0, 1, 2])

    def test_hypervolume_is_exact_for_two_rectangles(self):
        points = np.array([[0.5, 1.0], [1.0, 0.5]])
        self.assertAlmostEqual(hypervolume_2d(points), 0.75)

    def test_raw_nondominance_maximizes_accuracy_and_minimizes_cost(self):
        points = np.array(
            [
                [0.8, 2.0],
                [0.9, 1.5],
                [0.9, 2.0],
                [0.7, 1.0],
                [0.9, 1.5],
            ]
        )
        self.assertEqual(raw_nondominated_indices(points), [1, 3, 4])
        self.assertEqual(
            raw_archive_arm_indices([8, 2, 5, 4, 9], points),
            [2, 4, 9],
        )


class OfflineRoundRobinTests(unittest.TestCase):
    def test_scripted_round_robin_reuses_posteriors_and_returns_full_archive(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
            seed=3,
            index_provider=_target_provider,
        )

        self.assertEqual(result.stop_reason, "all_directions_gittins_stop")
        self.assertTrue(result.stopped_by_gittins)
        self.assertEqual(result.total_evaluations, 20)
        self.assertAlmostEqual(result.total_cost, 0.275)
        self.assertAlmostEqual(result.cost_reference_usd, 0.01)
        np.testing.assert_allclose(result.prior_mean, (0.6, 0.5125))
        self.assertEqual(result.selected_models, ["C", "B", "A"])
        self.assertEqual(
            [winner.model_name for winner in result.direction_winners],
            ["C", "B", "A"],
        )

        visits = [event for event in result.trace if event["event"] == "direction_visit"]
        self.assertEqual(
            [event["direction_index"] for event in visits],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
        )
        self.assertEqual(
            [event["selected_model"] for event in visits],
            ["C", "B", "A", "C", "B", "A", None, None, None],
        )
        self.assertEqual(len(result.observed_cells), result.total_evaluations)
        self.assertEqual(len(set(result.observed_cells)), result.total_evaluations)

        summaries = {summary.model_name: summary for summary in result.model_results}
        for model in ("A", "B", "C"):
            self.assertEqual(summaries[model].n_batches, 3)
            self.assertEqual(summaries[model].n_samples_evaluated, 6)
            self.assertTrue(summaries[model].completed)
        self.assertEqual(summaries["D"].n_batches, 1)
        self.assertEqual(summaries["D"].n_samples_evaluated, 2)
        self.assertFalse(summaries["D"].completed)

    def test_raw_filter_removes_transform_reversal_from_final_archive(self):
        models = ["A", "B"]
        datapoints = [0, 1]
        table = {
            "A": {0: _sample(0.8, 0.01), 1: _sample(0.8, 4.01)},
            "B": {0: _sample(0.9, 1.5), 1: _sample(0.9, 1.5)},
        }

        def direction_target(context, arm_index):
            return 100.0 if arm_index == context.direction_index else -100.0

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.1, 0.9), (0.9, 0.1)),
            cost_reference_usd=1.0,
            prior_variance=0.001,
            obs_noise_variance=1e-6,
            seed=1,
            index_provider=direction_target,
        )

        self.assertEqual(
            [winner.model_name for winner in result.direction_winners],
            ["A", "B"],
        )
        self.assertEqual(result.posterior_archive_models, ["A", "B"])
        self.assertEqual(result.selected_models, ["B"])
        self.assertEqual(result.online_raw_archive_models, ["B"])
        self.assertEqual(result.oracle_raw_winner_archive_models, ["B"])

    def test_oracle_tail_changes_do_not_leak_into_online_raw_archive(self):
        models = ["A", "B"]
        datapoints = [0, 1, 2]
        common = {
            "A": {0: _sample(0.8, 0.5), 1: _sample(0.8, 0.5)},
            "B": {0: _sample(0.9, 1.0), 1: _sample(0.9, 1.0)},
        }
        tables = []
        for a_tail, b_tail in (
            ((0.0, 10.0), (1.0, 0.1)),
            ((1.0, 0.1), (0.0, 10.0)),
        ):
            table = {model: dict(row) for model, row in common.items()}
            table["A"][2] = _sample(*a_tail)
            table["B"][2] = _sample(*b_tail)
            tables.append(table)

        results = [
            simulate_radial_gittins(
                models,
                datapoints,
                table,
                batch_size=2,
                directions=((0.1, 0.9), (0.9, 0.1)),
                cost_reference_usd=1.0,
                prior_variance=0.001,
                obs_noise_variance=1e-6,
                max_total_question_evaluations=4,
                seed=0,
                index_provider=lambda context, arm_index: 1.0,
                record_recommendation_trajectory=True,
            )
            for table in tables
        ]
        checkpoints = [result.recommendation_trajectory[0] for result in results]

        self.assertEqual(results[0].observed_cells, results[1].observed_cells)
        self.assertEqual(results[0].trace, results[1].trace)
        self.assertEqual(
            checkpoints[0].estimated_raw_winner_vectors,
            checkpoints[1].estimated_raw_winner_vectors,
        )
        self.assertEqual(checkpoints[0].selected_models, ("A", "B"))
        self.assertEqual(checkpoints[1].selected_models, ("A", "B"))
        self.assertEqual(
            checkpoints[0].selected_arm_indices,
            checkpoints[0].online_raw_archive_arm_indices,
        )
        self.assertEqual(
            checkpoints[0].oracle_raw_winner_archive_models,
            ("B",),
        )
        self.assertEqual(
            checkpoints[1].oracle_raw_winner_archive_models,
            ("A",),
        )
        for checkpoint, result in zip(checkpoints, results):
            self.assertEqual(checkpoint.completed_arm_indices, ())
            self.assertEqual(
                checkpoint.deployable_direction_winner_arm_indices,
                (),
            )
            self.assertEqual(
                checkpoint.deployable_online_raw_archive_models,
                (),
            )
            self.assertEqual(checkpoint.deployable_hypervolume, 0.0)
            self.assertEqual(
                checkpoint.deployable_hypervolume_regret,
                result.ground_truth_hypervolume,
            )

    def test_gittins_stop_checkpoint_excludes_unfinished_provisional_winner(self):
        models = ["A", "B"]
        datapoints = [0, 1, 2]
        table = {
            "A": {
                question_id: _sample(0.8, 0.1)
                for question_id in datapoints
            },
            # With seed=1, question 1 is the shared warm-start question.  B
            # initially looks attractive, but its full-data vector is truly
            # dominated by A and it remains unfinished at the Gittins stop.
            "B": {
                0: _sample(0.0, 10.0),
                1: _sample(1.0, 1.0),
                2: _sample(0.0, 10.0),
            },
        }

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.1, 0.9), (0.9, 0.1)),
            cost_reference_usd=1.0,
            prior_variance=0.001,
            obs_noise_variance=1e-6,
            seed=1,
            index_provider=lambda context, arm_index: (
                100.0 if arm_index == 0 else -100.0
            ),
            halt_on_gittins_stop=False,
            record_recommendation_trajectory=True,
        )
        stop = next(
            checkpoint
            for checkpoint in result.recommendation_trajectory
            if checkpoint.event == "gittins_stop"
        )

        self.assertEqual(stop.cumulative_evaluations, 4)
        self.assertEqual(stop.completed_arm_indices, (0,))
        self.assertEqual(stop.direction_winner_arm_indices, (0, 1))
        self.assertEqual(stop.online_raw_archive_models, ("A", "B"))
        self.assertEqual(stop.oracle_raw_winner_archive_models, ("A",))
        self.assertEqual(
            stop.deployable_direction_winner_arm_indices,
            (0,),
        )
        self.assertEqual(
            stop.deployable_online_raw_archive_models,
            ("A",),
        )
        self.assertEqual(
            stop.deployable_oracle_raw_winner_archive_models,
            ("A",),
        )
        self.assertLessEqual(
            set(stop.deployable_online_raw_archive_arm_indices),
            set(stop.completed_arm_indices),
        )

        final = result.recommendation_trajectory[-1]
        self.assertEqual(final.event, "final")
        self.assertEqual(final.completed_arm_indices, (0, 1))
        self.assertEqual(
            final.direction_winner_arm_indices,
            final.deployable_direction_winner_arm_indices,
        )
        self.assertEqual(
            final.online_raw_archive_arm_indices,
            final.deployable_online_raw_archive_arm_indices,
        )
        self.assertEqual(result.selected_models, ["A"])
        # The saved stop snapshot remains the completed-only A recommendation
        # after the diagnostic replay continues and finishes both arms.
        self.assertEqual(stop.completed_arm_indices, (0,))
        self.assertEqual(stop.deployable_online_raw_archive_models, ("A",))

        terminal_result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.1, 0.9), (0.9, 0.1)),
            cost_reference_usd=1.0,
            prior_variance=0.001,
            obs_noise_variance=1e-6,
            seed=1,
            index_provider=lambda context, arm_index: (
                100.0 if arm_index == 0 else -100.0
            ),
            halt_on_gittins_stop=True,
            record_recommendation_trajectory=True,
        )
        terminal_stop = next(
            checkpoint
            for checkpoint in terminal_result.recommendation_trajectory
            if checkpoint.event == "gittins_stop"
        )
        self.assertEqual(
            terminal_result.selected_models,
            list(terminal_stop.deployable_online_raw_archive_models),
        )
        self.assertEqual(terminal_result.selected_models, ["A"])
        self.assertEqual(terminal_result.total_evaluations, 4)

    def test_question_budget_counts_warm_start_and_stops_before_partial_batch(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
            max_total_question_evaluations=11,
            seed=3,
            index_provider=_target_provider,
        )

        # Eight warm cells plus one full two-question batch. Remaining budget
        # of one cell is not turned into a different DP action.
        self.assertEqual(result.stop_reason, "question_budget")
        self.assertEqual(result.total_evaluations, 10)
        self.assertAlmostEqual(result.total_cost, 0.11)
        self.assertEqual(result.selected_models, [])

    def test_dollar_budget_reserves_calibrated_batch_cost(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
            max_search_cost_usd=0.131,
            seed=3,
            index_provider=_target_provider,
        )

        self.assertEqual(result.stop_reason, "search_cost_budget")
        self.assertEqual(result.total_evaluations, 12)
        self.assertAlmostEqual(result.total_cost, 0.13)

    def test_default_universe_is_common_complete_question_intersection(self):
        models = ["A", "B", "C"]
        datapoints = list(range(7))
        available = {"A": range(7), "B": range(6), "C": range(5)}
        table = {
            model: {question_id: _sample(0.5, 0.01) for question_id in ids}
            for model, ids in available.items()
        }

        def first_arm(context, arm_index):
            return 10.0 if arm_index == 0 else -10.0

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=8,
            seed=9,
            index_provider=first_arm,
        )

        self.assertEqual(result.total_evaluations, 8)
        for arm_index, question_id in result.observed_cells:
            self.assertIn(question_id, range(5))
        self.assertEqual(result.params["question_universe"], "common")
        self.assertEqual(result.params["common_question_count"], 5)
        self.assertEqual(result.params["available_cells_in_universe"], 15)
        self.assertEqual(result.params["actual_horizons"], [2, 2, 2])
        self.assertEqual(result.params["ragged_tail_cells_excluded"], 0)
        self.assertEqual(result.params["planned_partial_tail_cells"], 3)

    def test_per_arm_question_universe_is_explicit_ragged_diagnostic(self):
        models = ["A", "B", "C"]
        datapoints = list(range(7))
        available = {"A": range(7), "B": range(6), "C": range(5)}
        table = {
            model: {question_id: _sample(0.5, 0.01) for question_id in ids}
            for model, ids in available.items()
        }

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=8,
            seed=9,
            index_provider=lambda context, arm_index: (
                10.0 if arm_index == 0 else -10.0
            ),
            question_universe="per_arm",
        )

        self.assertEqual(result.params["question_universe"], "per_arm")
        self.assertEqual(result.params["available_cells_in_universe"], 18)
        self.assertEqual(result.params["actual_horizons"], [3, 2, 2])
        self.assertEqual(result.params["ragged_tail_cells_excluded"], 0)
        self.assertEqual(result.params["planned_partial_tail_cells"], 2)

    def test_last_partial_batch_consumes_remaining_questions(self):
        models = ["A"]
        datapoints = list(range(5))
        table = {"A": {question_id: _sample(0.6, 0.1) for question_id in datapoints}}
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.5, 0.5),),
            seed=1,
            index_provider=lambda context, arm_index: 10.0,
        )

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 5)
        self.assertEqual(result.params["actual_horizons"], [2])
        self.assertEqual(result.params["planned_partial_tail_cells"], 1)
        self.assertEqual(result.params["ragged_tail_cells_excluded"], 0)
        self.assertEqual(result.params["unobserved_cells_at_stop"], 0)
        self.assertEqual(result.model_results[0].n_samples_evaluated, 5)
        self.assertEqual(result.model_results[0].n_batches, 3)
        self.assertEqual(result.online_raw_archive_models, ["A"])
        self.assertEqual(result.oracle_raw_winner_archive_models, ["A"])
        pulls = [
            event
            for event in result.trace
            if event["event"] == "direction_visit" and event["selected_arm"] is not None
        ]
        self.assertEqual([len(event["question_ids"]) for event in pulls], [2, 1])
        self.assertEqual(pulls[-1]["planned_batch_size"], 1)

    def test_expected_cost_vector_and_mapping_are_resolved_per_arm(self):
        models = ["A", "B"]
        datapoints = [0, 1]
        table = {
            model: {question_id: _sample(0.5, 0.1) for question_id in datapoints}
            for model in models
        }
        supplied = (
            [0.2, 0.3],
            {"A": 0.4, "B": 0.5},
        )
        expected = ([0.2, 0.3], [0.4, 0.5])
        for value, wanted in zip(supplied, expected):
            with self.subTest(value=value):
                result = simulate_radial_gittins(
                    models,
                    datapoints,
                    table,
                    batch_size=1,
                    directions=((0.5, 0.5),),
                    max_total_question_evaluations=2,
                    expected_batch_cost_usd=value,
                    index_provider=lambda context, arm_index: 1.0,
                )
                np.testing.assert_allclose(
                    result.params["expected_batch_costs_usd"], wanted
                )

    def test_soft_expected_cost_guard_reports_realized_overshoot(self):
        models = ["A"]
        datapoints = [0, 1, 2]
        # With seed=1, question 1 is warm and question 2 is the first tail.
        table = {
            "A": {
                0: _sample(0.5, 0.1),
                1: _sample(0.5, 0.1),
                2: _sample(0.5, 0.5),
            }
        }
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            max_search_cost_usd=0.25,
            seed=1,
            index_provider=lambda context, arm_index: 10.0,
        )

        self.assertEqual(result.stop_reason, "search_cost_budget")
        self.assertEqual(result.cost_budget_guard, "soft_expected_cost")
        self.assertAlmostEqual(result.total_cost, 0.6)
        self.assertAlmostEqual(result.cost_budget_overshoot_usd, 0.35)
        budget_stop = [x for x in result.trace if x["event"] == "budget_stop"]
        self.assertEqual(budget_stop[-1]["reason"], "search_cost_budget")

    def test_guaranteed_cost_bound_makes_budget_guard_hard(self):
        models = ["A"]
        datapoints = [0, 1, 2]
        table = {
            "A": {
                0: _sample(0.5, 0.1),
                1: _sample(0.5, 0.1),
                2: _sample(0.5, 0.5),
            }
        }
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            guaranteed_batch_cost_usd={"A": 0.5},
            max_search_cost_usd=0.6,
            seed=1,
            index_provider=lambda context, arm_index: 10.0,
        )

        self.assertEqual(result.cost_budget_guard, "hard_guaranteed_bound")
        self.assertAlmostEqual(result.total_cost, 0.6)
        self.assertEqual(result.cost_budget_overshoot_usd, 0.0)

    def test_cost_bins_share_close_cache_keys_but_separate_distant_arms(self):
        class FakeBoundary:
            def boundary(self, pull, delta):
                return 0.0

        class RecordingCache:
            def __init__(self):
                self.keys = set()
                self.grids = []

            def __len__(self):
                return len(self.keys)

            def get(self, **kwargs):
                self.grids.append(kwargs["grid"])
                self.keys.add(
                    (
                        kwargs["direction"],
                        kwargs["effective_pull_cost"],
                        kwargs["horizon"],
                    )
                )
                return FakeBoundary()

        models = ["A", "B", "C"]
        datapoints = [0, 1, 2]
        costs = {"A": 1.0, "B": 1.1, "C": 4.0}
        table = {
            model: {question_id: _sample(0.5, cost) for question_id in datapoints}
            for model, cost in costs.items()
        }
        cache = RecordingCache()
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=4,
            effective_cost_bin_ratio=2.0,
            seed=1,
            boundary_cache=cache,
        )

        raw = result.params["raw_effective_pull_costs"]
        binned = result.params["quantized_effective_pull_costs"]
        np.testing.assert_allclose(raw, [1.0, 1.1, 4.0])
        self.assertAlmostEqual(binned[0], binned[1])
        self.assertNotAlmostEqual(binned[1], binned[2])
        self.assertEqual(len({key[1] for key in cache.keys}), 2)
        self.assertEqual(result.params["boundary_grid_mode"], "direction_aware_default")
        self.assertTrue(all(grid.z_size == 513 for grid in cache.grids))
        self.assertTrue(all(grid.boundary_margin_cells == 4 for grid in cache.grids))

    def test_custom_reference_is_used_for_reported_hypervolume(self):
        models = ["A"]
        datapoints = [0, 1]
        table = {"A": {q: _sample(0.6, 1.0) for q in datapoints}}
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            reference_point=(0.5, 0.5),
            index_provider=lambda context, arm_index: 10.0,
        )

        self.assertEqual(result.selected_models, ["A"])
        self.assertEqual(result.hypervolume, 0.0)
        self.assertEqual(result.ground_truth_hypervolume, 0.0)

    def test_any_tied_accuracy_best_counts_as_contained(self):
        models = ["A", "B"]
        datapoints = [0, 1]
        table = {
            "A": {q: _sample(0.8, 1.0) for q in datapoints},
            "B": {q: _sample(0.8, 0.1) for q in datapoints},
        }
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            seed=1,
            index_provider=lambda context, arm_index: (
                100.0 if arm_index == 1 else -100.0
            ),
        )

        self.assertEqual(result.selected_models, ["B"])
        self.assertTrue(result.contains_true_accuracy_best)

    def test_result_is_strict_json_even_when_a_direction_side_is_empty(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=10,
            index_provider=lambda context, arm_index: 1.0,
        )

        payload = _jsonable_result(result)
        json.dumps(payload, allow_nan=False)
        first_visit = next(x for x in payload["trace"] if x["event"] == "direction_visit")
        self.assertIsNone(first_visit["best_completed_terminal_index"])

    def test_policy_timing_stops_before_truth_metric_evaluation(self):
        models = ["A"]
        datapoints = [0, 1]
        table = {"A": {q: _sample(0.6, 1.0) for q in datapoints}}
        truth_started = False
        original_truth = radial_replay._full_truth_vectors

        def timed_truth(*args, **kwargs):
            nonlocal truth_started
            truth_started = True
            return original_truth(*args, **kwargs)

        def fake_clock():
            return 100.0 if not truth_started else 999.0

        with mock.patch.object(radial_replay, "_full_truth_vectors", timed_truth), mock.patch.object(
            radial_replay.time, "perf_counter", fake_clock
        ):
            result = simulate_radial_gittins(
                models,
                datapoints,
                table,
                batch_size=1,
                directions=((0.5, 0.5),),
                index_provider=lambda context, arm_index: 10.0,
            )

        self.assertEqual(result.policy_wall_time_seconds, 0.0)

    def test_multi_seed_summary_keeps_each_fitted_parameter_set(self):
        models = ["A"]
        datapoints = [0, 1]
        table = {"A": {0: _sample(0.5, 0.1), 1: _sample(0.5, 0.2)}}
        results = [
            simulate_radial_gittins(
                models,
                datapoints,
                table,
                batch_size=1,
                directions=((0.5, 0.5),),
                max_total_question_evaluations=1,
                expected_batch_cost_usd=0.15,
                seed=seed,
                index_provider=lambda context, arm_index: 1.0,
            )
            for seed in (1, 3)
        ]
        summary = summarize_radial_multi_seed(results)

        self.assertNotIn("params", summary)
        self.assertEqual(len(summary["params_by_seed"]), 2)
        fitted_refs = [
            run["params"]["cost_reference_usd"]
            for run in summary["params_by_seed"]
        ]
        self.assertEqual(fitted_refs, [0.2, 0.1])
        anchors = [
            run["params"]["effective_cost_bin_anchor"]
            for run in summary["params_by_seed"]
        ]
        binned_costs = [
            run["params"]["quantized_effective_pull_costs"]
            for run in summary["params_by_seed"]
        ]
        self.assertEqual(anchors, [1e-4, 1e-4])
        self.assertEqual(binned_costs[0], binned_costs[1])


class OfflineActualDPTests(unittest.TestCase):
    def test_high_cost_required_completion_expands_root_search_band(self):
        models = ["A"]
        datapoints = [0, 1, 2, 3]
        table = {
            "A": {question_id: _sample(0.6, 1.0) for question_id in datapoints}
        }
        compact_base = RadialGittinsGrid(
            z_min=-3.0,
            z_max=3.0,
            z_size=65,
            delta_min=-3.0,
            delta_max=3.0,
            delta_size=65,
            state_size=65,
            state_halo=2.0,
            boundary_margin_cells=2,
        )

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            seed=1,
            boundary_grid=compact_base,
        )

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 4)
        # H*c displaces the finite-horizon stopping root beyond the original
        # z_max=3 envelope; the replay must expand it rather than clip/fail.
        self.assertGreater(
            result.params["boundary_grids"][0]["grid"]["z_max"],
            compact_base.z_max,
        )

    def test_one_arm_actual_dp_replay_completes_without_duplicate_cells(self):
        models = ["A"]
        datapoints = [0, 1]
        table = {"A": {question_id: _sample(0.6, 1.0) for question_id in datapoints}}
        grid = RadialGittinsGrid(
            z_min=-3.0,
            z_max=3.0,
            z_size=65,
            delta_min=-3.0,
            delta_max=3.0,
            delta_size=65,
            state_size=65,
            state_halo=4.0,
            boundary_margin_cells=2,
        )

        result = simulate_radial_gittins(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            search_cost_scale_eta=0.1,
            seed=1,
            boundary_grid=grid,
        )

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 2)
        self.assertEqual(result.total_cost, 2.0)
        self.assertEqual(result.selected_models, ["A"])
        self.assertEqual(len(result.observed_cells), 2)
        self.assertEqual(
            result.params["boundary_grid_mode"],
            "direction_aware_custom_base",
        )
        self.assertEqual(result.params["boundary_grids"][0]["grid"]["z_size"], 65)


if __name__ == "__main__":
    unittest.main()
