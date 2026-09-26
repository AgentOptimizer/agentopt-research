"""Compare nested random baselines using the measured USD they actually spend."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.combined_objective import three_objective_metrics as metrics
from experiments.single_objective.offline_selector_sim import SampleResult


def _problem():
    models = ["accurate", "fast", "cheap", "noisy", "dominated"]
    questions = [2, 7, 11, 16, 23, 31, 42]
    table = {}
    for arm, model in enumerate(models):
        table[model] = {}
        for i, question in enumerate(questions):
            quality = [.9, .6 + .02 * (i % 3), .4 + .01 * (i % 4), 1. if i == 0 else .2, .1][arm]
            latency = [10. + i, 1. + .2 * i, 6. + .1 * i, 4. + .2 * i, 20. + i][arm]
            cost = [.06 + .002 * i, .04 + .001 * i, .005 + .0001 * i, .05 + .005 * i, .2 + .01 * i][arm]
            table[model][question] = SampleResult(
                score=quality, latency_seconds=latency, cost=cost,
                input_tokens={}, output_tokens={},
            )
    return models, questions, table


def _run(version, seed=17, problem=None):
    from experiments.combined_objective.offline_three_objective_random_search import (
        simulate_three_objective_random_search,
    )
    return simulate_three_objective_random_search(*(_problem() if problem is None else problem),
                                                 version=version, seed=seed)


def _mean_vectors(models, questions, table, arms):
    return np.asarray([
        np.mean([[table[models[arm]][q].score, table[models[arm]][q].latency_seconds,
                  table[models[arm]][q].cost] for q in questions], axis=0)
        for arm in arms
    ])


class ThreeObjectiveRandomBaselineTests(unittest.TestCase):
    def _check_point(self, point, arms, questions, problem):
        models, _, table = problem
        observed = _mean_vectors(models, questions, table, arms)
        expected_selection = sorted(arms[i] for i in metrics.raw_pareto_indices(observed))
        self.assertEqual(sorted(point["selected_arm_indices"]), expected_selection)
        self.assertEqual(point["sampled_arm_count"], len(arms))
        self.assertEqual(point["sampled_question_count"], len(questions))
        self.assertEqual(point["total_evaluations"], len(arms) * len(questions))
        actual_cost = sum(table[models[arm]][q].cost for arm in arms for q in questions)
        self.assertAlmostEqual(point["cumulative_search_cost_usd"], actual_cost)
        exhaustive_cost = sum(sample.cost for row in table.values() for sample in row.values())
        self.assertAlmostEqual(point["cost_fraction"], actual_cost / exhaustive_cost)

    def test_question_path_uses_a_shared_nested_prefix_for_every_configuration(self):
        problem = _problem()
        models, questions, _ = problem
        result = _run("random_questions", problem=problem)
        order = result["sampled_question_order"]
        self.assertEqual(sorted(order), sorted(questions))
        self.assertEqual(len(set(order)), len(questions))
        self.assertEqual(len(result["points"]), len(questions))
        for step, point in enumerate(result["points"], start=1):
            self.assertEqual(point["step"], step)
            self._check_point(point, list(range(len(models))), order[:step], problem)
            expected_completed = list(range(len(models))) if step == len(questions) else []
            self.assertEqual(sorted(point["completed_arm_indices"]), expected_completed)
        # One unit purchases an entire column, not one randomly chosen cell.
        self.assertEqual(result["points"][0]["total_evaluations"], len(models))
        increments = np.diff([0, *[p["total_evaluations"] for p in result["points"]]])
        np.testing.assert_array_equal(increments, np.full(len(questions), len(models)))

    def test_configuration_path_adds_complete_rows_and_recommends_only_observed_arms(self):
        problem = _problem()
        models, questions, _ = problem
        result = _run("random_configurations", problem=problem)
        order = result["sampled_arm_order"]
        self.assertEqual(sorted(order), list(range(len(models))))
        self.assertEqual(len(set(order)), len(models))
        self.assertEqual(len(result["points"]), len(models))
        for step, point in enumerate(result["points"], start=1):
            arms = order[:step]
            self.assertEqual(point["step"], step)
            self._check_point(point, arms, questions, problem)
            self.assertEqual(set(point["completed_arm_indices"]), set(arms))
            self.assertTrue(set(point["selected_arm_indices"]).issubset(arms))
        increments = np.diff([0, *[p["total_evaluations"] for p in result["points"]]])
        np.testing.assert_array_equal(increments, np.full(len(models), len(questions)))

    def test_partial_question_frontier_uses_observed_means_not_full_data_truth(self):
        models, questions, table = _problem()
        seed = next(seed for seed in range(100)
                    if np.random.default_rng(seed).permutation(questions)[0] == questions[0])
        result = _run("random_questions", seed=seed)
        self.assertIn(models.index("noisy"), result["points"][0]["selected_arm_indices"])
        self.assertNotIn(models.index("noisy"), result["full_data_pareto_arm_indices"])
        changed = copy.deepcopy(table)
        for model in models:
            for question in result["sampled_question_order"][1:]:
                old = changed[model][question]
                changed[model][question] = SampleResult(
                    score=1. - old.score, latency_seconds=100. * old.latency_seconds,
                    cost=100. * old.cost, input_tokens={}, output_tokens={},
                )
        perturbed = _run("random_questions", seed=seed, problem=(models, questions, changed))
        for field in ("selected_arm_indices", "total_evaluations", "cumulative_search_cost_usd"):
            self.assertEqual(perturbed["points"][0][field], result["points"][0][field])
        self.assertNotEqual(perturbed["raw_truth_vectors"], result["raw_truth_vectors"])

    def test_full_budget_recovers_exact_three_objective_frontier_and_cost(self):
        problem = _problem()
        truth = metrics.raw_truth_vectors(*problem)
        true_front = metrics.raw_pareto_indices(truth)
        expected_cost = sum(sample.cost for row in problem[2].values() for sample in row.values())
        for version in ("random_questions", "random_configurations"):
            with self.subTest(version=version):
                result = _run(version, problem=problem)
                final = result["points"][-1]
                self.assertEqual(sorted(final["selected_arm_indices"]), true_front)
                self.assertEqual(result["full_data_pareto_arm_indices"], true_front)
                self.assertEqual(result["total_evaluations"], len(problem[0]) * len(problem[1]))
                self.assertAlmostEqual(result["search_cost_usd"], expected_cost)
                self.assertAlmostEqual(result["bruteforce_search_cost_usd"], expected_cost)
                self.assertAlmostEqual(final["cost_fraction"], 1.)
                for field in ("hv_regret", "relative_hv_regret", "generational_distance", "inverted_generational_distance"):
                    self.assertAlmostEqual(final[field], 0.)
                self.assertEqual(final["pareto_precision"], 1.)
                self.assertEqual(final["pareto_recall"], 1.)
                self.assertTrue(final["exact_frontier"])

    def test_seed_reproduces_the_entire_nested_path(self):
        for version, order_field in (("random_questions", "sampled_question_order"),
                                      ("random_configurations", "sampled_arm_order")):
            with self.subTest(version=version):
                first, repeated, different = _run(version, 17), _run(version, 17), _run(version, 18)
                self.assertEqual(first[order_field], repeated[order_field])
                self.assertEqual(first["points"], repeated["points"])
                self.assertNotEqual(first[order_field], different[order_field])

    def test_latency_unit_conversion_never_changes_usd_spend_or_frontier(self):
        models, questions, table = _problem()
        scaled = copy.deepcopy(table)
        for model in models:
            for q in questions:
                old = scaled[model][q]
                scaled[model][q] = SampleResult(score=old.score, cost=old.cost,
                                                latency_seconds=old.latency_seconds * 1000.,
                                                input_tokens={}, output_tokens={})
        for version in ("random_questions", "random_configurations"):
            with self.subTest(version=version):
                seconds = _run(version)
                milliseconds = _run(version, problem=(models, questions, scaled))
                for original, converted in zip(seconds["points"], milliseconds["points"]):
                    self.assertEqual(original["selected_arm_indices"], converted["selected_arm_indices"])
                    self.assertEqual(original["cumulative_search_cost_usd"], converted["cumulative_search_cost_usd"])
                    self.assertEqual(original["cost_fraction"], converted["cost_fraction"])
                    self.assertAlmostEqual(original["relative_hv_regret"], converted["relative_hv_regret"])

    def test_each_observed_cell_is_read_once_for_the_whole_nested_path(self):
        class CountedRow(dict):
            def __init__(self, values):
                super().__init__(values)
                self.read_counts = {key: 0 for key in values}

            def __getitem__(self, key):
                self.read_counts[key] += 1
                if self.read_counts[key] > 1:
                    raise AssertionError("an observation was reread")
                return super().__getitem__(key)

        for version in ("random_questions", "random_configurations"):
            with self.subTest(version=version):
                models, questions, original = _problem()
                counted = {model: CountedRow(row) for model, row in original.items()}
                result = _run(version, problem=(models, questions, counted))
                self.assertEqual(result["total_evaluations"], len(models) * len(questions))
                for row in counted.values():
                    self.assertEqual(set(row.read_counts.values()), {1})


class LatestUnderCheckpointTests(unittest.TestCase):
    def _checkpoint(self, target):
        from experiments.combined_objective.offline_three_objective_random_search import latest_under_checkpoint
        run = {"cost_fraction": .55, "points": [
            {"cost_fraction": cost, "selected_arm_indices": [i], "total_evaluations": i + 1}
            for i, cost in enumerate((.08, .14, .31, .55))
        ]}
        return latest_under_checkpoint(run, target)

    def test_uses_last_complete_unit_under_actual_usd_target_without_interpolation(self):
        checkpoint = self._checkpoint(.30)
        self.assertTrue(checkpoint["available"])
        self.assertEqual(checkpoint["cost_fraction"], .14)
        self.assertEqual(checkpoint["selected_arm_indices"], [1])
        self.assertEqual(checkpoint["target_cost_fraction"], .30)
        exact = self._checkpoint(.31)
        self.assertEqual(exact["cost_fraction"], .31)
        self.assertEqual(exact["selected_arm_indices"], [2])

    def test_no_checkpoint_before_first_unit_or_beyond_reached_terminal_budget(self):
        self.assertFalse(self._checkpoint(.05)["available"])
        self.assertFalse(self._checkpoint(.60)["available"])
        self.assertTrue(self._checkpoint(.55)["available"])


class ComparisonAggregationTests(unittest.TestCase):
    @staticmethod
    def _runs():
        return [{"cost_fraction": 1., "points": [{
            "cost_fraction": cost, "hypervolume": volume, "relative_hv_regret": regret,
            "generational_distance": distance, "inverted_generational_distance": distance * 2,
        }]} for cost, volume, regret, distance in ((.08, .4, .2, .2), (.10, .6, .1, .4))]

    def test_seed_aggregation_uses_latest_under_states_and_seed_percentiles(self):
        from experiments.combined_objective.compare_three_objective_baselines import aggregate_at
        result = aggregate_at(self._runs(), .11)
        self.assertEqual(result["n_seeds"], 2)
        self.assertEqual(result["n_available"], 2)
        self.assertAlmostEqual(result["hypervolume"]["mean"], .5)
        self.assertAlmostEqual(result["hypervolume"]["p10"], .42)
        self.assertAlmostEqual(result["hypervolume"]["p90"], .58)
        self.assertAlmostEqual(result["generational_distance"]["mean"], .3)
        self.assertAlmostEqual(result["actual_cost_fraction_mean"], .09)

    def test_unavailable_seed_is_not_silently_dropped_from_comparison(self):
        from experiments.combined_objective.compare_three_objective_baselines import aggregate_at
        result = aggregate_at(self._runs(), .09)
        self.assertEqual(result["n_seeds"], 2)
        self.assertEqual(result["n_available"], 1)
        self.assertAlmostEqual(result["hypervolume"]["mean"], .2)
        self.assertAlmostEqual(result["relative_hv_regret"]["mean"], .6)
        for distance in ("generational_distance", "inverted_generational_distance"):
            self.assertIsNone(result[distance]["mean"])
            self.assertIsNone(result[distance]["p10"])
            self.assertIsNone(result[distance]["p90"])
            self.assertEqual(result[distance]["n_finite"], 1)

    def test_a_stopped_policy_keeps_its_terminal_recommendation_without_extra_spend(self):
        from experiments.combined_objective.compare_three_objective_baselines import aggregate_at
        run = self._runs()[0]
        run.update(cost_fraction=.08, stop_reason="direction_eta_numerical_floor")
        result = aggregate_at([run], .50)
        self.assertEqual(result["n_available"], 1)
        self.assertEqual(result["n_stopped_before_target"], 1)
        self.assertEqual(result["actual_cost_fraction_mean"], .08)
        self.assertEqual(result["hypervolume"]["mean"], .4)
        self.assertEqual(result["generational_distance"]["mean"], .2)

    def test_budget_limited_or_incomplete_runs_are_not_extrapolated(self):
        from experiments.combined_objective.compare_three_objective_baselines import aggregate_at
        for reason in ("observation_budget", "search_cost_budget"):
            with self.subTest(reason=reason):
                run = self._runs()[0]
                run.update(cost_fraction=.08, stop_reason=reason)
                result = aggregate_at([run], .50)
                self.assertEqual(result["n_available"], 0)
                self.assertEqual(result["n_stopped_before_target"], 0)
                self.assertIsNone(result["actual_cost_fraction_mean"])
                self.assertEqual(result["hypervolume"]["mean"], 0.)
                self.assertIsNone(result["generational_distance"]["mean"])


class CompactSavedGittinsTests(unittest.TestCase):
    @staticmethod
    def _payload(directions=None):
        from experiments.combined_objective.offline_three_objective_gittins import simulate_three_objective_gittins
        options = {} if directions is None else {"directions": directions}
        run = simulate_three_objective_gittins(
            *_problem(), seed=17, grid_size=33, index_provider=lambda context, arm: 100. - arm,
            **options,
        )
        metrics.enrich_run(run)
        return {
            "config": {"seed": 17, "benchmark": "toy", "input_sha256": {"latency": "test-matrix"},
                       "source_sha256": {"engine": "test-source"},
                       "directions": run["parameters"]["directions"],
                       "direction_labels": run["parameters"]["direction_labels"]},
            "run": run, "summary": {"total_evaluations": run["total_evaluations"]},
            "timing": {"simulation_seconds": run["sampling_wall_seconds"]},
        }

    def test_compact_saved_run_preserves_every_comparison_metric_and_checkpoint(self):
        from experiments.combined_objective.run_three_objective_gittins_multiseed import compact_payload
        from experiments.combined_objective.compare_three_objective_baselines import compact_gittins, aggregate_at
        dense = self._payload()
        original = copy.deepcopy(dense)
        compact = compact_payload(dense)
        self.assertEqual(dense, original)
        self.assertEqual(compact["config"], dense["config"])
        self.assertEqual(compact["summary"], dense["summary"])
        self.assertNotIn("trace", compact["run"])
        self.assertNotIn("observed_cells", compact["run"])
        for point in compact["run"]["points"]:
            self.assertNotIn("finite_target_mean_vectors", point)
            self.assertNotIn("finite_target_std_vectors", point)
        with tempfile.TemporaryDirectory() as directory:
            dense_path, compact_path = Path(directory) / "dense.json", Path(directory) / "compact.json"
            dense_path.write_text(json.dumps(dense, allow_nan=False))
            compact_path.write_text(json.dumps(compact, allow_nan=False))
            loaded_dense = compact_gittins(dense_path, dense["config"]["input_sha256"])
            loaded_compact = compact_gittins(compact_path, dense["config"]["input_sha256"])
        for key in loaded_dense:
            if key != "source_result":
                self.assertEqual(loaded_compact[key], loaded_dense[key], key)
        for budget in (.05, .10, .20, .50, .75, 1.):
            self.assertEqual(aggregate_at([loaded_compact], budget), aggregate_at([loaded_dense], budget))

    def test_compact_saved_run_still_rejects_input_matrix_hash_mismatches(self):
        from experiments.combined_objective.run_three_objective_gittins_multiseed import compact_payload
        from experiments.combined_objective.compare_three_objective_baselines import compact_gittins
        payload = compact_payload(self._payload())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.json"
            path.write_text(json.dumps(payload, allow_nan=False))
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                compact_gittins(path, {"latency": "changed-matrix"})

    def test_resume_validation_rejects_changed_protocol_and_inconsistent_final_totals(self):
        from experiments.combined_objective.run_three_objective_gittins_multiseed import compact_payload, validate_config
        saved = compact_payload(self._payload())
        expected = copy.deepcopy(saved["config"])
        validate_config(saved, expected, "test-result.json")
        for field, value in (("seed", 18), ("input_sha256", {"latency": "changed"}),
                             ("source_sha256", {"engine": "changed"})):
            with self.subTest(field=field):
                changed = copy.deepcopy(saved)
                changed["config"][field] = value
                with self.assertRaisesRegex(ValueError, "changed"):
                    validate_config(changed, expected, "test-result.json")
        changed = copy.deepcopy(saved)
        changed["run"]["total_evaluations"] += 1
        with self.assertRaisesRegex(ValueError, "observation totals disagree"):
            validate_config(changed, expected, "test-result.json")


class DynamicProgramDirectionRoutingTests(unittest.TestCase):
    def test_ql_midpoint_uses_radial_dp_and_pure_endpoints_use_scalar_dp(self):
        from experiments.combined_objective import offline_three_objective_gittins as replay
        directions = ((1., 0., 0.), (.5, .5, 0.), (0., 1., 0.), (0., 0., 1.))
        for direction in directions:
            with self.subTest(direction=direction):
                radial = replay.RadialGittinsBoundaryCache()
                axis = replay.AxisGittinsBoundaryCache()
                with mock.patch.object(replay, "RadialGittinsBoundaryCache", return_value=radial), \
                        mock.patch.object(replay, "AxisGittinsBoundaryCache", return_value=axis), \
                        mock.patch.object(radial, "get", wraps=radial.get) as radial_get, \
                        mock.patch.object(axis, "get", wraps=axis.get) as axis_get:
                    result = replay.simulate_three_objective_gittins(
                        *_problem(), seed=17, directions=(direction,), grid_size=33,
                        max_total_question_evaluations=23,
                    )
                self.assertEqual(result["total_evaluations"], 23)
                if direction == (.5, .5, 0.):
                    self.assertGreater(radial_get.call_count, 0)
                    self.assertEqual(axis_get.call_count, 0)
                    for call in radial_get.call_args_list:
                        self.assertEqual(call.kwargs["direction"], (.5, .5))
                else:
                    self.assertGreater(axis_get.call_count, 0)
                    self.assertEqual(radial_get.call_count, 0)

    def test_combined_three_axes_visit_q_l_d_round_robin_without_radial_tables(self):
        from experiments.combined_objective import offline_three_objective_gittins as replay
        axes = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        radial = replay.RadialGittinsBoundaryCache()
        axis = replay.AxisGittinsBoundaryCache()
        with mock.patch.object(replay, "RadialGittinsBoundaryCache", return_value=radial), \
                mock.patch.object(replay, "AxisGittinsBoundaryCache", return_value=axis), \
                mock.patch.object(radial, "get", wraps=radial.get) as radial_get, \
                mock.patch.object(axis, "get", wraps=axis.get) as axis_get:
            result = replay.simulate_three_objective_gittins(
                *_problem(), seed=17, directions=axes, grid_size=33,
                max_total_question_evaluations=32,
            )
        visits = [event for event in result["trace"] if event["event"] == "direction_visit"]
        self.assertGreaterEqual(len(visits), 3)
        self.assertEqual([event["direction_index"] for event in visits[:3]], [0, 1, 2])
        for event in visits:
            self.assertEqual(event["direction_index"], event["global_step"] % 3)
            self.assertEqual(event["direction"], list(axes[event["direction_index"]]))
        self.assertEqual(result["parameters"]["directions"], [list(d) for d in axes])
        self.assertEqual(result["parameters"]["direction_labels"], ["Q", "L", "D"])
        self.assertEqual(len(result["direction_eta_multipliers"]), 3)
        self.assertGreater(axis_get.call_count, 0)
        self.assertEqual(radial_get.call_count, 0)
        self.assertGreater(result["axis_boundary_cache_stats"]["builds"], 0)
        self.assertEqual(result["radial_boundary_cache_stats"]["builds"], 0)


class AxesOnlyRunnerTests(unittest.TestCase):
    def test_axes_config_preserves_protocol_and_explicitly_changes_only_directions(self):
        from experiments.combined_objective import run_three_objective_gittins_multiseed as runner
        hashes = {"latency": "test-matrix"}
        original = runner.base_config("mathqa", 42, hashes)
        explicit_original = runner.base_config("mathqa", 42, hashes, variant="four_directions")
        axes = runner.base_config("mathqa", 42, hashes, variant="axes_only")
        self.assertEqual(original, explicit_original)
        self.assertNotIn("variant", original)
        self.assertEqual(axes["variant"], "axes_only")
        self.assertEqual(axes["directions"], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(axes["direction_labels"], ["Q", "L", "D"])
        changed_fields = {key for key in set(original) | set(axes) if original.get(key) != axes.get(key)}
        self.assertEqual(changed_fields, {"directions", "direction_labels", "variant"})
        self.assertEqual(axes["acquisition_cost_unit"], "USD")
        self.assertNotEqual(runner.DEFAULT_OUTDIR, runner.DEFAULT_AXES_OUTDIR)

    def test_axes_seed42_is_simulated_even_when_four_direction_seed42_exists(self):
        from experiments.combined_objective import run_three_objective_gittins_multiseed as runner
        models, questions, table = _problem()
        hashes = {"latency": "test-matrix"}
        original_payload = CompactSavedGittinsTests._payload()
        original_payload["config"] = runner.base_config("mathqa", 42, hashes)
        simulate = runner.simulate_three_objective_gittins

        def tiny_simulation(*args, **kwargs):
            kwargs["grid_size"] = 33
            kwargs["index_provider"] = lambda context, arm: 100. - arm
            return simulate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            original_root, axes_root = Path(directory) / "four", Path(directory) / "axes"
            original_path = original_root / "seed_42/mathqa/result.json"
            original_path.parent.mkdir(parents=True)
            original_text = json.dumps(original_payload, allow_nan=False)
            original_path.write_text(original_text)
            with mock.patch.object(runner, "load_three_objective_benchmark", return_value=(models, questions, table, hashes)), \
                    mock.patch.object(runner, "simulate_three_objective_gittins", side_effect=tiny_simulation) as simulated, \
                    contextlib.redirect_stdout(io.StringIO()):
                receipt = runner.run_one("mathqa", 42, outdir=axes_root,
                                         original_root=original_root, variant="axes_only")
            self.assertEqual(simulated.call_count, 1)
            self.assertEqual(simulated.call_args.kwargs["directions"],
                             ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
            self.assertEqual(receipt["status"], "completed")
            saved = json.loads((axes_root / "seed_42/mathqa/result.json").read_text())
            self.assertEqual(saved["config"]["variant"], "axes_only")
            self.assertEqual(saved["run"]["parameters"]["direction_labels"], ["Q", "L", "D"])
            self.assertNotIn("source_result", saved["provenance"])
            self.assertEqual(original_path.read_text(), original_text)

    def test_axes_resume_rejects_a_four_direction_result_in_the_same_output_path(self):
        from experiments.combined_objective import run_three_objective_gittins_multiseed as runner
        payload = CompactSavedGittinsTests._payload()
        hashes = payload["config"]["input_sha256"]
        payload["config"] = runner.base_config("mathqa", 42, hashes, variant="four_directions")
        expected = runner.base_config("mathqa", 42, hashes, variant="axes_only")
        with self.assertRaisesRegex(ValueError, "directions"):
            runner.validate_config(payload, expected, "axes/seed_42/mathqa/result.json")


class ComparisonVariantValidationTests(unittest.TestCase):
    @staticmethod
    def _load(payload, method):
        from experiments.combined_objective.compare_three_objective_baselines import compact_gittins
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload, allow_nan=False))
            return compact_gittins(path, payload["config"]["input_sha256"],
                                   method=method, validate_directions=True)

    def test_correct_variants_load_but_cannot_be_mislabeled_as_the_other_method(self):
        cases = ((None, "cc_gittins", "cc_gittins_axes"),
                 (((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "cc_gittins_axes", "cc_gittins"))
        for directions, correct, incorrect in cases:
            with self.subTest(method=correct):
                payload = CompactSavedGittinsTests._payload(directions=directions)
                loaded = self._load(payload, correct)
                self.assertEqual(loaded["selector"], correct)
                self.assertEqual(loaded["protocol"]["directions"], payload["config"]["directions"])
                with self.assertRaisesRegex(ValueError, "directions differ"):
                    self._load(payload, incorrect)

    def test_comparison_checks_both_declared_and_actual_run_directions(self):
        axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        for location in ("config", "parameters"):
            with self.subTest(location=location):
                payload = CompactSavedGittinsTests._payload()
                if location == "config":
                    payload["config"]["directions"] = axes
                else:
                    payload["run"]["parameters"]["directions"] = axes
                with self.assertRaisesRegex(ValueError, "directions differ"):
                    self._load(payload, "cc_gittins_axes")

    def test_axes_comparison_rejects_any_radial_cache_activity_including_hits(self):
        axes = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        original = CompactSavedGittinsTests._payload(directions=axes)
        for counter in ("builds", "memory_hits", "disk_hits"):
            with self.subTest(counter=counter):
                changed = copy.deepcopy(original)
                changed["run"]["radial_boundary_cache_stats"][counter] = 1
                with self.assertRaisesRegex(ValueError, "unexpectedly used radial DP"):
                    self._load(changed, "cc_gittins_axes")


if __name__ == "__main__":
    unittest.main()
