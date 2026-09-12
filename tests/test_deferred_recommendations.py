"""Recommendation diagnostics must be detached from the adaptive sampling loop."""

from dataclasses import FrozenInstanceError, asdict
import json
import unittest
from unittest import mock

from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem(values=((0.8, 0.1), (0.7, 0.1)), n_questions=8):
    models = [f"arm_{i}" for i in range(len(values))]
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
    return models, questions, table


def _run(problem=None, **overrides):
    options = dict(
        batch_size=1,
        directions=((1.0, 0.0),),
        anytime=True,
        record_recommendation_trajectory=True,
        recommendation_checkpoint_interval=100,
        recommendation_changes_only=True,
        prior_variance=1.0,
        obs_noise_variance=1.0,
        cost_reference_usd=1.0,
        effective_cost_bin_ratio=None,
        index_provider=lambda context, arm: 100.0,
        seed=42,
    )
    options.update(overrides)
    return replay.simulate_radial_gittins(
        *(_problem() if problem is None else problem), **options,
    )


def _without_timing(trace):
    return [
        {key: value for key, value in event.items()
         if key not in {"stage_wall_time_seconds", "run_wall_time_seconds"}}
        for event in trace
    ]


def _snapshots(result):
    return (
        [asdict(point) for point in result.recommendation_trajectory],
        asdict(result.recommendation_initial_snapshot),
        asdict(result.recommendation_final_snapshot),
    )


class DeferredRecommendationTests(unittest.TestCase):
    def test_default_materializes_after_acquisition_and_deferred_never_does(self):
        for defer in (False, True):
            with self.subTest(defer=defer):
                provider_calls = []
                with mock.patch.object(
                    replay, "_materialize_recommendation_event",
                    wraps=replay._materialize_recommendation_event,
                ) as materialize:
                    def index(context, arm):
                        provider_calls.append((context.global_step, arm))
                        # No warm-start, completion, or eta event may calculate
                        # recommendation diagnostics before acquisition ends.
                        self.assertEqual(materialize.call_count, 0)
                        return 100.0 if arm == 1 else -100.0

                    result = _run(
                        max_total_question_evaluations=11,
                        index_provider=index,
                        defer_recommendation_diagnostics=defer,
                    )
                    self.assertTrue(provider_calls)
                    self.assertTrue(result.recommendation_events)
                    if defer:
                        self.assertEqual(materialize.call_count, 0)
                        self.assertEqual(result.recommendation_trajectory, [])
                        self.assertIsNone(result.recommendation_initial_snapshot)
                        self.assertIsNone(result.recommendation_final_snapshot)
                    else:
                        self.assertGreater(materialize.call_count, 0)
                        self.assertTrue(result.recommendation_trajectory)

    def test_unchanged_pulls_do_not_create_or_materialize_per_pull_events(self):
        # The recommendation stays empty until the sole arm completes. Its
        # 23 adaptive pulls must produce only one membership change, plus the
        # independent warm-start and final snapshots.
        with mock.patch.object(
            replay, "_capture_recommendation_event",
            wraps=replay._capture_recommendation_event,
        ) as capture, mock.patch.object(
            replay, "_materialize_recommendation_event",
            wraps=replay._materialize_recommendation_event,
        ) as materialize:
            result = _run(_problem(((0.8, 0.1),), n_questions=24))

        self.assertEqual(result.total_evaluations, 24)
        self.assertEqual(len(result.recommendation_events), 1)
        self.assertEqual(len(result.recommendation_trajectory), 1)
        self.assertLessEqual(capture.call_count, 3)
        self.assertLessEqual(materialize.call_count, 3)
        self.assertGreater(materialize.call_count, 0)
        self.assertEqual(result.recommendation_initial_event.cumulative_evaluations, 1)
        self.assertEqual(result.recommendation_initial_event.selected_arm_indices, ())
        self.assertEqual(result.recommendation_events[0].cumulative_evaluations, 24)
        self.assertEqual(result.recommendation_events[0].selected_arm_indices, (0,))
        self.assertEqual(result.recommendation_final_event.cumulative_evaluations, 24)

    def test_explicit_materialization_matches_default_and_is_idempotent(self):
        for schedule in ("global_stop", "direction_stop"):
            with self.subTest(schedule=schedule):
                options = dict(
                    directions=((0.2, 0.8), (0.8, 0.2), (1.0, 0.0)),
                    eta_decay_schedule=schedule,
                )
                problem = _problem(((0.8, 0.4), (0.6, 0.1)), n_questions=5)
                immediate = _run(problem, **options)
                deferred = _run(problem, defer_recommendation_diagnostics=True, **options)

                self.assertEqual(deferred.recommendation_trajectory, [])
                self.assertEqual(deferred.recommendation_events, immediate.recommendation_events)
                self.assertEqual(deferred.observed_cells, immediate.observed_cells)
                self.assertEqual(deferred.total_cost, immediate.total_cost)
                self.assertEqual(deferred.selected_models, immediate.selected_models)
                self.assertEqual(deferred.direction_eta_events, immediate.direction_eta_events)
                self.assertEqual(_without_timing(deferred.trace), _without_timing(immediate.trace))
                replay.materialize_recommendation_diagnostics(deferred)
                self.assertEqual(_snapshots(deferred), _snapshots(immediate))

                with mock.patch.object(
                    replay, "_materialize_recommendation_event",
                    wraps=replay._materialize_recommendation_event,
                ) as materialize:
                    replay.materialize_recommendation_diagnostics(deferred)
                    materialize.assert_not_called()
                self.assertEqual(_snapshots(deferred), _snapshots(immediate))

    def test_events_and_materialized_diagnostics_do_not_depend_on_trace(self):
        options = dict(defer_recommendation_diagnostics=True)
        with_trace = _run(**options)
        without_trace = _run(record_trace=False, **options)

        self.assertTrue(with_trace.trace)
        self.assertEqual(without_trace.trace, [])
        self.assertEqual(without_trace.recommendation_events, with_trace.recommendation_events)
        self.assertEqual(without_trace.recommendation_initial_event, with_trace.recommendation_initial_event)
        self.assertEqual(without_trace.recommendation_final_event, with_trace.recommendation_final_event)
        replay.materialize_recommendation_diagnostics(with_trace)
        replay.materialize_recommendation_diagnostics(without_trace)
        self.assertEqual(_snapshots(without_trace), _snapshots(with_trace))

    def test_saved_events_round_trip_to_legacy_json_without_replay_or_lookup(self):
        checkpoint_fields = (
            "recommendation_trajectory",
            "recommendation_initial_snapshot",
            "recommendation_final_snapshot",
        )
        # Cover completed-only archives, an always-empty completed archive,
        # and partial fixed-budget diagnostic evidence with tracing disabled.
        for options in ({}, {"max_total_question_evaluations": 4}, {"anytime": False}):
            with self.subTest(options=options):
                problem = _problem(n_questions=5)
                expected = replay._jsonable_result(_run(problem, **options))
                deferred = _run(
                    problem, defer_recommendation_diagnostics=True,
                    record_trace=False, **options,
                )
                saved = json.loads(json.dumps(replay._jsonable_result(deferred), allow_nan=False))
                self.assertEqual(saved["recommendation_trajectory"], [])
                self.assertIsNone(saved["recommendation_initial_snapshot"])
                self.assertEqual(saved["trace"], [])
                # Serialized evidence and truth vectors must suffice after
                # both the live simulation and original lookup are gone.
                del deferred
                problem[2].clear()
                with mock.patch.object(
                    replay, "simulate_radial_gittins", side_effect=AssertionError("replayed acquisition"),
                ), mock.patch.object(
                    replay, "_full_truth_vectors", side_effect=AssertionError("rescanned lookup"),
                ), mock.patch.object(
                    replay, "_full_raw_objective_vectors", side_effect=AssertionError("rescanned lookup"),
                ), mock.patch(
                    "builtins.open", side_effect=AssertionError("read or wrote a file"),
                ):
                    replay.materialize_saved_recommendation_diagnostics(saved)

                for field in checkpoint_fields:
                    self.assertEqual(saved[field], expected[field])
                materialized_json = json.dumps(saved, allow_nan=False, sort_keys=True)
                with mock.patch.object(
                    replay, "_materialize_recommendation_event",
                    wraps=replay._materialize_recommendation_event,
                ) as materialize:
                    replay.materialize_saved_recommendation_diagnostics(saved)
                    materialize.assert_not_called()
                self.assertEqual(json.dumps(saved, allow_nan=False, sort_keys=True), materialized_json)

    def test_partial_event_freezes_evidence_and_final_refreshes_unchanged_membership(self):
        problem = _problem(((0.8, 0.1),), n_questions=12)
        for question in problem[1]:
            problem[2]["arm_0"][question] = SampleResult(
                score=float(question >= 6), cost=0.1, latency_seconds=0.1,
                input_tokens={}, output_tokens={},
            )
        # Fixed-budget runs already expose provisional partial diagnostics;
        # they need no additional recommendation rule to exercise this case.
        result = _run(problem, anytime=False, defer_recommendation_diagnostics=True)
        warm = result.recommendation_initial_event
        final = result.recommendation_final_event
        self.assertEqual(warm.selected_arm_indices, final.selected_arm_indices)
        self.assertEqual(warm.direction_winner_sample_counts, (1,))
        self.assertEqual(final.direction_winner_sample_counts, (12,))
        self.assertEqual(warm.completed_arm_indices, ())
        self.assertEqual(final.completed_arm_indices, (0,))
        self.assertNotEqual(warm.winner_posterior_means, final.winner_posterior_means)
        self.assertNotEqual(warm.estimated_raw_winner_vectors, final.estimated_raw_winner_vectors)
        self.assertIsInstance(warm.winner_posterior_means, tuple)
        self.assertIsInstance(warm.winner_posterior_means[0], tuple)
        with self.assertRaises(FrozenInstanceError):
            warm.cumulative_evaluations = 999

        frozen_evidence = asdict(warm)
        replay.materialize_recommendation_diagnostics(result)
        self.assertEqual(asdict(warm), frozen_evidence)
        self.assertEqual(result.recommendation_initial_snapshot.direction_winner_sample_counts, (1,))
        self.assertEqual(result.recommendation_final_snapshot.direction_winner_sample_counts, (12,))
        self.assertEqual(
            result.recommendation_initial_snapshot.estimated_raw_winner_vectors,
            warm.estimated_raw_winner_vectors,
        )
        self.assertEqual(
            result.recommendation_final_snapshot.estimated_raw_winner_vectors,
            final.estimated_raw_winner_vectors,
        )

    def test_warm_and_final_survive_when_completed_recommendation_is_always_empty(self):
        result = _run(
            max_total_question_evaluations=4,
            defer_recommendation_diagnostics=True,
        )
        self.assertEqual(result.recommendation_events, [])
        self.assertEqual(result.recommendation_initial_event.selected_arm_indices, ())
        self.assertEqual(result.recommendation_final_event.selected_arm_indices, ())
        self.assertEqual(result.recommendation_initial_event.cumulative_evaluations, 2)
        self.assertEqual(result.recommendation_final_event.cumulative_evaluations, 4)

        replay.materialize_recommendation_diagnostics(result)
        self.assertEqual(result.recommendation_trajectory, [])
        self.assertEqual(result.recommendation_initial_snapshot.selected_arm_indices, ())
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices, ())
        self.assertEqual(result.recommendation_final_snapshot.cumulative_evaluations, 4)


if __name__ == "__main__":
    unittest.main()
