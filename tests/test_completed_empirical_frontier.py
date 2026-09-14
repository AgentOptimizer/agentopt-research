"""Completed recommendations report measured tradeoffs, independently of exploration."""

import unittest
from unittest import mock

import numpy as np

from agentopt.model_selection.radial_gittins import GaussianVectorPosterior
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def _select(points, completed=None, **overrides):
    # Deliberately make the posterior favor the last arm rather than the
    # empirical frontier. Completed recommendations must ignore these values.
    options = dict(
        posteriors={
            arm: GaussianVectorPosterior([.1 + arm / 100, .9], [100., 100.])
            for arm in points
        },
        directions=((1., 0.),),
        reference_point=(0., 0.), stop_tolerance=1e-9,
        observed_scores={arm: [point[0]] for arm, point in points.items()},
        observed_costs={arm: [point[1]] for arm, point in points.items()},
        completed_arms=tuple(points) if completed is None else completed,
        archive_scope=replay.DEPLOYABLE_ARCHIVE_SCOPE,
        recommendation_rule="completed_only", recommendation_beta=1.,
    )
    options.update(overrides)
    return replay._recommendation_selection(**options)


def _run(points=((.3, .001), (.6, .01), (.9, .1), (.2, .2)), **overrides):
    models, questions = [f"arm_{i}" for i in range(len(points))], list(range(4))
    table = {
        model: {
            q: SampleResult(score=score, cost=cost, latency_seconds=0.,
                            input_tokens={}, output_tokens={})
            for q in questions
        }
        for model, (score, cost) in zip(models, points)
    }
    options = dict(
        seed=42, batch_size=1, anytime=True,
        directions=((.2, .8), (.8, .2), (1., 0.)),
        eta_decay_schedule="direction_stop",
        cost_reference_usd=1., effective_cost_bin_ratio=None,
        prior_variance=.001, obs_noise_variance=1e-9,
        recommendation_rule="completed_only",
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True,
        index_provider=lambda context, arm: (
            100. - arm if context.current_lambda <= .5 else -100.
        ),
    )
    options.update(overrides)
    return replay.simulate_radial_gittins(models, questions, table, **options)


class CompletedEmpiricalSelectionTests(unittest.TestCase):
    def test_returns_entire_empirical_frontier_despite_misleading_posteriors(self):
        points = {0: (.3, .001), 1: (.6, .01), 2: (.9, .1), 3: (.2, .2), 4: (1., 0.)}
        selection = _select(points, completed=(3, 2, 1, 0))

        self.assertEqual(selection.selected_arm_indices, (0, 1, 2))
        self.assertNotIn(4, selection.selected_arm_indices)
        np.testing.assert_allclose(
            selection.estimated_raw_winner_vectors,
            [points[arm] for arm in selection.direction_winner_arm_indices],
        )

    def test_directions_reference_beta_and_uncertainty_cannot_change_membership(self):
        points = {0: (.3, 1e-12), 1: (.6, 1.), 2: (.9, 1e6)}
        for directions, reference, beta in (
            (((1., 0.),), (0., 0.), 0.),
            (((.01, .99), (.99, .01)), (-100., 100.), 1000.),
        ):
            with self.subTest(directions=directions, reference=reference, beta=beta):
                selection = _select(
                    points, directions=directions, reference_point=reference,
                    recommendation_beta=beta,
                )
                self.assertEqual(selection.selected_arm_indices, (0, 1, 2))
                self.assertEqual(selection.estimated_raw_winner_vectors,
                                 tuple(points[arm] for arm in (0, 1, 2)))

    def test_no_incomplete_recommendation_even_with_provisional_archive_scope(self):
        for scope in (replay.DEPLOYABLE_ARCHIVE_SCOPE, replay.PROVISIONAL_ARCHIVE_SCOPE):
            with self.subTest(scope=scope):
                selection = _select({0: (1., 0.)}, completed=(), archive_scope=scope)
                self.assertEqual(selection.selected_arm_indices, ())
                self.assertEqual(selection.direction_winner_arm_indices, ())
                self.assertEqual(selection.estimated_raw_winner_vectors, ())

    def test_strict_dominance_keeps_exact_ties_in_deterministic_arm_order(self):
        points = {8: (.8, .2), 4: (.7, .2), 2: (.8, .2), 7: (.8, .3), 0: (.6, .1)}
        for completed in (tuple(points), tuple(reversed(points))):
            with self.subTest(completed=completed):
                selection = _select(points, completed=completed)
                self.assertEqual(selection.selected_arm_indices, (0, 2, 8))

    def test_averages_raw_question_costs_before_testing_dominance(self):
        # Averaging reciprocal rewards reverses the cost order of these rows:
        # A costs [1, 1], while B costs [0, 3]. Their raw means are 1 and 1.5.
        selection = _select(
            {0: (.8, 1.), 1: (.8, 1.5)},
            observed_scores={0: [.6, 1.], 1: [.8, .8]},
            observed_costs={0: [1., 1.], 1: [0., 3.]},
        )
        self.assertEqual(selection.selected_arm_indices, (0,))
        self.assertEqual(selection.estimated_raw_winner_vectors, ((.8, 1.),))

    def test_cached_completed_histories_need_not_be_read_again(self):
        archive = replay._CompletedRawParetoArchive()
        archive.update((0,), {0: [.3, .5]}, {0: [.1, .1]})
        # A new arm can replace a cached old arm without access to old rows.
        archive.update((0, 1), {1: [.8, .8]}, {1: [.05, .05]})
        with mock.patch.object(replay.np, "mean", side_effect=AssertionError("reread history")):
            archive.update((0, 1), {}, {})
        self.assertEqual(archive.frontier, {1: (.8, .05)})


class CompletedEmpiricalReplayTests(unittest.TestCase):
    def test_frontier_events_remove_only_dominated_members_and_preserve_ties(self):
        result = _run(points=((.3, .1), (.8, .05), (.2, .2), (.8, .05)))
        events = result.recommendation_events
        self.assertEqual([event.selected_arm_indices for event in events],
                         [(0,), (1,), (1, 3)])
        self.assertEqual([event.cumulative_evaluations for event in events], [7, 10, 16])
        self.assertEqual(events[1].added_arm_indices, (1,))
        self.assertEqual(events[1].removed_arm_indices, (0,))
        self.assertEqual(events[2].removed_arm_indices, ())
        self.assertEqual(result.selected_models, ["arm_1", "arm_3"])

    def test_final_frontier_can_exceed_number_of_exploration_directions(self):
        for record in (False, True):
            with self.subTest(record=record):
                result = _run(directions=((1., 0.),), record_recommendation_trajectory=record)
                self.assertEqual(result.selected_models, ["arm_0", "arm_1", "arm_2"])
                self.assertTrue(all(arm.completed for arm in result.model_results))
                if record:
                    self.assertEqual(result.recommendation_final_event.selected_arm_indices, (0, 1, 2))
                    self.assertEqual(result.recommendation_final_snapshot.selected_models,
                                     tuple(result.selected_models))

    def test_recommendation_is_independent_of_cost_model_after_same_rows_complete(self):
        for model, reference in (("reciprocal", .001), ("reciprocal", 100.), ("raw_mean", 1.)):
            with self.subTest(model=model, reference=reference):
                result = _run(cost_model=model, cost_reference_usd=reference)
                self.assertEqual(result.selected_models, ["arm_0", "arm_1", "arm_2"])

    def test_finite_lcb_preserves_sampling_and_independent_eta_decay(self):
        completed = _run(cost_model="raw_mean")
        finite = _run(recommendation_rule="finite_lcb", cost_model="raw_mean")
        self.assertEqual(completed.trace, finite.trace)
        self.assertEqual(completed.observed_cells, finite.observed_cells)
        self.assertEqual(completed.total_cost, finite.total_cost)
        self.assertEqual(completed.stop_reason, finite.stop_reason)
        self.assertEqual(completed.direction_eta_events, finite.direction_eta_events)
        self.assertTrue(completed.direction_eta_events)
        self.assertEqual(completed.direction_eta_stages, finite.direction_eta_stages)

    def test_fixed_lambda_partial_run_never_recommends_unfinished_rows(self):
        result = _run(
            anytime=False, eta_decay_schedule="global_stop",
            max_total_question_evaluations=6,
            index_provider=lambda context, arm: 100. - arm,
        )
        self.assertEqual(result.total_evaluations, 6)
        self.assertTrue(all(not arm.completed for arm in result.model_results))
        self.assertEqual(result.selected_models, [])
        self.assertEqual(result.recommendation_initial_event.selected_arm_indices, ())
        self.assertEqual(result.recommendation_final_event.selected_arm_indices, ())
        self.assertTrue(all(not event.selected_arm_indices for event in result.recommendation_events))


if __name__ == "__main__":
    unittest.main()
