import json
import unittest

import numpy as np

from agentopt.model_selection.radial_ucb import (
    BATCH_COUNT_BONUS,
    RadialUCBPolicy,
)
from experiments.combined_objective.offline_radial_gittins import (
    _jsonable_result,
    raw_nondominated_indices,
    simulate_radial_gittins,
    summarize_radial_multi_seed,
)
from experiments.combined_objective.offline_radial_ucb import simulate_radial_ucb
from experiments.single_objective.offline_selector_sim import SampleResult


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


def _pull_events(result):
    return [
        event
        for event in result.trace
        if event["event"] == "direction_visit"
        and event["selected_arm"] is not None
    ]


class RadialUCBReplayTests(unittest.TestCase):
    def test_result_is_labelled_as_radial_ucb_with_its_hyperparameters(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
            exploration_beta=1.5,
            seed=3,
        )

        self.assertEqual(result.selector, "radial_ucb")
        self.assertEqual(result.params["acquisition"], "external_index_provider")
        self.assertEqual(result.params["exploration_beta"], 1.5)
        self.assertEqual(result.params["bonus_mode"], "posterior_sd")
        self.assertTrue(result.params["cost_aware_index"])
        # No boundary table is built, so no DP grid is ever resolved.
        self.assertEqual(result.params["boundary_grids"], [])
        json.dumps(_jsonable_result(result), allow_nan=False)

    def test_recorded_index_matches_the_policy_on_the_selected_posterior(self):
        models, datapoints, table = _toy_frontier()
        policy = RadialUCBPolicy(exploration_beta=2.0)
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
            exploration_beta=2.0,
            seed=3,
        )
        pull_costs = result.params["quantized_effective_pull_costs"]

        events = _pull_events(result)
        self.assertGreater(len(events), 0)
        for event in events:
            expected = policy.index(
                event["posterior_mean_before"],
                event["posterior_var_before"],
                event["direction"],
                result.params["reference_point"],
                effective_pull_cost=pull_costs[event["selected_arm"]],
            )
            self.assertAlmostEqual(
                event["best_unfinished_gittins_index"],
                expected,
            )

    def test_search_cost_penalty_changes_which_arm_is_pulled(self):
        # Identical lookup cells keep both posteriors on the same cost
        # desirability, so only the frozen per-batch search cost differs.
        models = ["cheap_search", "costly_search"]
        datapoints = [0, 1, 2]
        table = {
            "cheap_search": {q: _sample(0.3, 0.01) for q in datapoints},
            "costly_search": {q: _sample(0.4, 0.01) for q in datapoints},
        }
        selected = {}
        for cost_aware in (True, False):
            result = simulate_radial_ucb(
                models,
                datapoints,
                table,
                batch_size=1,
                directions=((0.5, 0.5),),
                prior_variance=0.001,
                obs_noise_variance=1e-6,
                expected_batch_cost_usd={
                    "cheap_search": 1e-6,
                    "costly_search": 10.0,
                },
                cost_aware=cost_aware,
                max_total_question_evaluations=3,
                seed=0,
            )
            self.assertEqual(result.params["cost_aware_index"], cost_aware)
            selected[cost_aware] = _pull_events(result)[0]["selected_model"]

        self.assertEqual(selected[False], "costly_search")
        self.assertEqual(selected[True], "cheap_search")

    def test_cost_blind_index_ignores_eta(self):
        models, datapoints, table = _toy_frontier()
        results = [
            simulate_radial_ucb(
                models,
                datapoints,
                table,
                batch_size=2,
                directions=((0.5, 0.5),),
                cost_aware=False,
                search_cost_scale_eta=eta,
                seed=5,
            )
            for eta in (0.01, 100.0)
        ]
        # Only the recorded cost provenance may differ, never a decision.
        self.assertEqual(results[0].observed_cells, results[1].observed_cells)
        self.assertEqual(
            [event.get("selected_arm") for event in results[0].trace],
            [event.get("selected_arm") for event in results[1].trace],
        )
        self.assertEqual(results[0].selected_models, results[1].selected_models)

    def test_recommendation_is_nondominated_in_raw_space(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=2,
            seed=7,
        )

        self.assertEqual(result.stop_reason, "all_arms_completed")
        self.assertEqual(result.total_evaluations, 24)
        self.assertEqual(len(set(result.observed_cells)), 24)
        self.assertTrue(result.selected_models)
        archive = list(result.online_raw_archive_arm_indices)
        raw = np.asarray(
            [
                (
                    result.model_results[i].observed_accuracy,
                    result.model_results[i].observed_mean_cost_usd,
                )
                for i in archive
            ]
        )
        self.assertEqual(
            raw_nondominated_indices(raw),
            list(range(len(archive))),
        )

    def test_greedy_beta_zero_still_completes_and_recommends(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=2,
            exploration_beta=0.0,
            seed=11,
        )

        self.assertEqual(result.params["exploration_beta"], 0.0)
        self.assertTrue(result.selected_models)

    def test_batch_count_bonus_mode_runs_end_to_end(self):
        models, datapoints, table = _toy_frontier()
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=2,
            directions=((0.3, 0.7), (0.7, 0.3)),
            bonus_mode=BATCH_COUNT_BONUS,
            exploration_beta=0.5,
            seed=13,
        )

        self.assertEqual(result.params["bonus_mode"], BATCH_COUNT_BONUS)
        self.assertTrue(_pull_events(result))

    def test_effective_pull_costs_are_unquantized_by_default(self):
        models = ["A", "B"]
        datapoints = [0, 1]
        table = {
            "A": {q: _sample(0.5, 1.0) for q in datapoints},
            "B": {q: _sample(0.5, 1.1) for q in datapoints},
        }
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=1,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=2,
            seed=1,
        )

        self.assertIsNone(result.params["effective_cost_bin_ratio"])
        np.testing.assert_allclose(
            result.params["quantized_effective_pull_costs"],
            result.params["raw_effective_pull_costs"],
        )

    def test_replay_protocol_is_shared_with_radial_gittins(self):
        models, datapoints, table = _toy_frontier()
        shared = dict(
            batch_size=2,
            directions=((0.5, 0.5),),
            max_total_question_evaluations=10,
            seed=3,
        )
        ucb = simulate_radial_ucb(models, datapoints, table, **shared)
        scripted = simulate_radial_gittins(
            models,
            datapoints,
            table,
            index_provider=lambda context, arm_index: 1.0,
            **shared,
        )

        # The warm start, universe, horizons, and calibration must not depend
        # on which acquisition rule scores the unfinished arms.
        for key in (
            "cost_reference_usd",
            "actual_horizons",
            "available_cells_in_universe",
            "bruteforce_search_cost_usd",
            "common_question_count",
        ):
            self.assertEqual(ucb.params[key], scripted.params[key], msg=key)
        self.assertEqual(ucb.prior_mean, scripted.prior_mean)
        self.assertEqual(
            [event["question_ids"] for event in ucb.trace[:4]],
            [event["question_ids"] for event in scripted.trace[:4]],
        )

    def test_multi_seed_summary_reports_the_radial_ucb_selector(self):
        models, datapoints, table = _toy_frontier()
        results = [
            simulate_radial_ucb(
                models,
                datapoints,
                table,
                batch_size=2,
                directions=((0.5, 0.5),),
                seed=seed,
            )
            for seed in (1, 2)
        ]
        summary = summarize_radial_multi_seed(results)

        self.assertEqual(summary["selector"], "radial_ucb")
        self.assertEqual(summary["n_seeds"], 2)

    def test_summary_refuses_to_mix_selectors(self):
        models, datapoints, table = _toy_frontier()
        shared = dict(batch_size=2, directions=((0.5, 0.5),), seed=1)
        with self.assertRaises(ValueError):
            summarize_radial_multi_seed(
                [
                    simulate_radial_ucb(models, datapoints, table, **shared),
                    simulate_radial_gittins(
                        models,
                        datapoints,
                        table,
                        index_provider=lambda context, arm_index: 1.0,
                        **shared,
                    ),
                ]
            )


if __name__ == "__main__":
    unittest.main()
