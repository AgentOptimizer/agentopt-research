"""Latency participates in objectives; the observations still cost dollars."""

import csv
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.combined_objective import three_objective_metrics as metrics
from experiments.single_objective.offline_selector_sim import SampleResult


ROOT = Path(__file__).resolve().parents[1]


def _write_matrix(directory, filename, rows, questions=(2, 7)):
    with (Path(directory) / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", *(f"question_{q}" for q in questions)])
        writer.writerows(rows)


def _matrix_fixture(directory):
    _write_matrix(directory, "accuracy_matrix.csv", [["a", 1, .5], ["b", 0, 1]])
    _write_matrix(directory, "latency_matrix_seconds.csv", [["a", 12, 34], ["b", 5, 8]])
    _write_matrix(directory, "cost_matrix_usd.csv", [["a", .01, .03], ["b", .02, .04]])


def _problem(n_questions=9, latency_scale=1.):
    models = ["accurate", "fast", "cheap"]
    questions = list(range(n_questions))
    table = {}
    for arm, model in enumerate(models):
        table[model] = {
            q: SampleResult(
                score=(.85, .55, .35)[arm] + .01 * (q % 3),
                latency_seconds=latency_scale * ((20., 2., 10.)[arm] + .1 * q),
                cost=(.03, .02, .005)[arm] + .0001 * q,
                input_tokens={}, output_tokens={},
            )
            for q in questions
        }
    return models, questions, table


def _run(problem=None, **options):
    from experiments.combined_objective.offline_three_objective_gittins import (
        simulate_three_objective_gittins,
    )
    arguments = dict(
        seed=42, batch_size=4, warm_start_batch_size=4,
        question_order="independent", warm_start_question_order="independent",
        grid_size=33, index_provider=lambda context, arm: 100. - arm,
    )
    arguments.update(options)
    return simulate_three_objective_gittins(*(_problem() if problem is None else problem), **arguments)


class ThreeObjectiveDataTests(unittest.TestCase):
    def test_loads_latency_seconds_and_cost_dollars_from_matching_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            _matrix_fixture(directory)
            models, questions, table, hashes = metrics.load_three_objective_benchmark(directory)
            self.assertEqual(models, ["a", "b"])
            self.assertEqual(questions, [2, 7])
            self.assertEqual(table["a"][7].latency_seconds, 34.)
            self.assertEqual(table["b"][2].cost, .02)
            np.testing.assert_allclose(
                metrics.raw_truth_vectors(models, questions, table),
                [[.75, 23., .02], [.5, 6.5, .03]],
            )
            self.assertEqual(len(hashes), 3)
            before = hashes.copy()
            _write_matrix(directory, "latency_matrix_seconds.csv", [["a", 12, 35], ["b", 5, 8]])
            after = metrics.load_three_objective_benchmark(directory)[3]
            changed = [key for key in before if before[key] != after[key]]
            self.assertEqual(len(changed), 1)
            self.assertTrue(changed[0].endswith("latency_matrix_seconds.csv"))

    def test_rejects_latency_rows_columns_missing_cells_and_invalid_values(self):
        cases = [
            ([["b", 5, 8], ["a", 12, 34]], (2, 7)),
            ([["a", 12, 34], ["b", 5, 8]], (7, 2)),
            ([["a", 12, ""], ["b", 5, 8]], (2, 7)),
            ([["a", -1, 34], ["b", 5, 8]], (2, 7)),
            ([["a", "nan", 34], ["b", 5, 8]], (2, 7)),
        ]
        for rows, questions in cases:
            with self.subTest(rows=rows, questions=questions), tempfile.TemporaryDirectory() as directory:
                _matrix_fixture(directory)
                _write_matrix(directory, "latency_matrix_seconds.csv", rows, questions)
                with self.assertRaises(ValueError):
                    metrics.load_three_objective_benchmark(directory)

    def test_requires_latency_file_instead_of_silently_using_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_matrix(directory, "accuracy_matrix.csv", [["a", 1, .5]])
            _write_matrix(directory, "cost_matrix_usd.csv", [["a", .01, .03]])
            with self.assertRaises((ValueError, FileNotFoundError)):
                metrics.load_three_objective_benchmark(directory)

    def test_real_benchmarks_keep_nonzero_measured_latencies(self):
        for benchmark in ("mathqa", "hotpotqa"):
            with self.subTest(benchmark=benchmark):
                path = ROOT / "data" / benchmark
                if not (path / "latency_matrix_seconds.csv").is_file():
                    self.skipTest(f"local {benchmark} matrices are unavailable")
                models, questions, table, hashes = metrics.load_three_objective_benchmark(path)
                self.assertGreater(len(models), 1)
                self.assertGreater(len(questions), 1)
                self.assertTrue(any(key.endswith("latency_matrix_seconds.csv") for key in hashes))
                for filename, attribute in (("latency_matrix_seconds.csv", "latency_seconds"),
                                            ("cost_matrix_usd.csv", "cost")):
                    with (path / filename).open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.reader(handle))
                    self.assertEqual(rows[1][0], models[0])
                    self.assertAlmostEqual(getattr(table[models[0]][questions[0]], attribute),
                                           float(rows[1][1]))
                self.assertTrue(all(sample.latency_seconds > 0.
                                    for model in models for sample in table[model].values()))


class ThreeObjectiveMetricTests(unittest.TestCase):
    def test_raw_pareto_maximizes_quality_minimizes_latency_and_dollars_with_ties(self):
        points = np.array([
            [.8, 2., .02], [.9, 2., .02], [.9, 3., .02],
            [.9, 2., .03], [.7, 1., .01], [.9, 2., .02], [.6, 4., .04],
        ])
        self.assertEqual(list(metrics.raw_pareto_indices(points)), [1, 4, 5])
        self.assertEqual(list(metrics.raw_pareto_indices(np.empty((0, 3)))), [])

    def test_three_dimensional_nondominance_matches_strict_brute_force(self):
        rng = np.random.default_rng(123)
        for size in (0, 1, 2, 20, 100):
            points = rng.integers(0, 7, size=(size, 3)).astype(float)
            expected = [i for i in range(size)
                        if not any(np.all(points[j] >= points[i]) and np.any(points[j] > points[i])
                                   for j in range(size) if j != i)]
            self.assertEqual(list(metrics.nondominated_indices(points)), expected)

    def test_hypervolume_is_exact_union_of_three_dimensional_boxes(self):
        points = np.array([[.5, 1., 1.], [1., .5, 1.], [1., 1., .5]])
        self.assertAlmostEqual(metrics.hypervolume_3d(points), .875)
        self.assertAlmostEqual(metrics.hypervolume_3d(np.vstack([points, points[0], [.1, .1, .1]])), .875)
        self.assertEqual(metrics.hypervolume_3d(np.empty((0, 3))), 0.)
        self.assertEqual(metrics.hypervolume_3d([[0., 1., 1.], [-1., 2., 2.]]), 0.)

    def test_hypervolume_matches_inclusion_exclusion_with_translated_reference(self):
        rng = np.random.default_rng(41)
        reference = np.array([-.2, .3, -.7])
        for _ in range(5):
            points = rng.uniform(.1, 1., size=(5, 3))
            expected = 0.
            for size in range(1, len(points) + 1):
                for subset in combinations(points, size):
                    expected += (-1) ** (size + 1) * float(np.prod(np.min(subset, axis=0)))
            self.assertAlmostEqual(metrics.hypervolume_3d(points + reference, reference=reference), expected)

    def test_metric_coordinates_preserve_raw_frontier_and_physical_unit_changes(self):
        raw = np.array([[.9, 20., .03], [.6, 2., .02], [.4, 10., .005], [.3, 30., .04]])
        space, front, latency_ref, cost_ref, volume = metrics.evaluation_space(raw)
        self.assertEqual(list(metrics.nondominated_indices(space)), metrics.raw_pareto_indices(raw))
        scaled_space, scaled_front, scaled_latency_ref, scaled_cost_ref, scaled_volume = (
            metrics.evaluation_space(raw * [1., 1000., 100.])
        )
        np.testing.assert_allclose(scaled_space, space)
        np.testing.assert_allclose(scaled_front, front)
        self.assertAlmostEqual(scaled_latency_ref, latency_ref * 1000.)
        self.assertAlmostEqual(scaled_cost_ref, cost_ref * 100.)
        self.assertAlmostEqual(scaled_volume, volume)
        scores = metrics.score_selection(metrics.raw_pareto_indices(raw), space, front, volume)
        self.assertAlmostEqual(scores["hv_regret"], 0.)
        self.assertAlmostEqual(scores["generational_distance"], 0.)
        self.assertAlmostEqual(scores["inverted_generational_distance"], 0.)


class ThreeObjectiveReplayTests(unittest.TestCase):
    def test_completed_finite_targets_equal_exact_full_data_means(self):
        result = _run()
        final = result["points"][-1]
        expected = metrics.raw_truth_vectors(*_problem())
        self.assertEqual(result["total_evaluations"], 27)
        self.assertEqual(result["counts"], [9, 9, 9])
        np.testing.assert_allclose(final["finite_target_mean_vectors"], expected, rtol=1e-12)
        np.testing.assert_array_equal(final["finite_target_std_vectors"], np.zeros((3, 3)))
        np.testing.assert_allclose(final["recommendation_raw_vectors"], expected, rtol=1e-12)
        self.assertEqual(final["selected_arm_indices"], metrics.raw_pareto_indices(expected))
        expected_cost = sum(sample.cost for row in _problem()[2].values() for sample in row.values())
        self.assertAlmostEqual(result["total_cost"], expected_cost)

    def test_finite_lcb_is_conservative_in_each_objectives_raw_units(self):
        result = _run(max_total_question_evaluations=16)
        self.assertEqual(result["total_evaluations"], 16)
        self.assertEqual(sum(result["counts"]), 16)
        for point in result["points"]:
            mean = np.asarray(point["finite_target_mean_vectors"])
            std = np.asarray(point["finite_target_std_vectors"])
            np.testing.assert_allclose(point["recommendation_raw_vectors"], mean + std * [-1., 1., 1.])
            self.assertTrue(np.all(std >= 0.))
            self.assertTrue(np.all(np.isfinite(mean)))

    def test_rescaling_latency_preserves_dollar_acquisition_penalties(self):
        captured = []
        def record(context, arm):
            captured.append({
                key: np.asarray(context[key]).copy()
                for key in ("expected_batch_costs_usd", "effective_pull_costs")
            })
            return 100. - arm
        seconds = _run(index_provider=record, max_total_question_evaluations=20)
        first_contexts = captured.copy()
        captured.clear()
        milliseconds = _run(_problem(latency_scale=1000.), index_provider=record,
                            max_total_question_evaluations=20)
        self.assertEqual(len(first_contexts), len(captured))
        self.assertGreater(len(captured), 0)
        for expected, changed in zip(first_contexts, captured):
            for key in expected:
                np.testing.assert_allclose(changed[key], expected[key], rtol=1e-12)
        self.assertAlmostEqual(seconds["total_cost"], milliseconds["total_cost"])
        self.assertEqual(seconds["counts"], milliseconds["counts"])
        for original, scaled in zip(seconds["points"], milliseconds["points"]):
            self.assertEqual(original["selected_arm_indices"], scaled["selected_arm_indices"])
            for key in ("finite_target_mean_vectors", "finite_target_std_vectors", "recommendation_raw_vectors"):
                np.testing.assert_allclose(scaled[key], np.asarray(original[key]) * [1., 1000., 1.], rtol=1e-10)

    def test_all_directions_share_every_update_and_each_cell_is_charged_once(self):
        contexts = {}
        def record(context, arm):
            contexts.setdefault(context["global_step"], {
                key: np.asarray(context[key]).copy()
                for key in ("raw_posterior_means", "raw_posterior_variances", "counts")
            })
            return 100. - arm
        result = _run(index_provider=record)
        state_mean, state_var = np.zeros((3, 3)), np.zeros((3, 3))
        counts = np.zeros(3, dtype=int)
        seen, directions, cost = set(), set(), 0.
        table, models = _problem()[2], _problem()[0]
        for event in result["trace"]:
            if event["event"] == "direction_visit":
                context = contexts[event["global_step"]]
                np.testing.assert_allclose(context["raw_posterior_means"], state_mean)
                np.testing.assert_allclose(context["raw_posterior_variances"], state_var)
                np.testing.assert_array_equal(context["counts"], counts)
                directions.add(event["direction_index"])
            arm = event.get("selected_arm")
            if arm is None:
                continue
            batch_cost = 0.
            for question in event["question_ids"]:
                cell = (arm, question)
                self.assertNotIn(cell, seen)
                seen.add(cell)
                batch_cost += table[models[arm]][question].cost
            cost += batch_cost
            self.assertAlmostEqual(event["actual_batch_search_cost_usd"], batch_cost)
            self.assertAlmostEqual(event["cumulative_search_cost_usd"], cost)
            counts[arm] += len(event["question_ids"])
            state_mean[arm] = event["raw_posterior_mean_after"]
            state_var[arm] = event["raw_posterior_var_after"]
        self.assertEqual(directions, {0, 1, 2, 3})
        self.assertEqual(len(seen), result["total_evaluations"])
        self.assertEqual(seen, {tuple(cell) for cell in result["observed_cells"]})
        self.assertAlmostEqual(cost, result["total_cost"])

    def test_partial_final_batch_uses_its_actual_sample_size_for_all_posteriors(self):
        result = _run()
        noise = np.asarray(result["parameters"]["raw_question_noise_variance"])
        sizes = []
        for event in result["trace"]:
            if event["event"] != "direction_visit" or event["selected_arm"] is None:
                continue
            size = len(event["question_ids"])
            sizes.append(size)
            before_mean = np.asarray(event["raw_posterior_mean_before"])
            before_var = np.asarray(event["raw_posterior_var_before"])
            expected_var = 1. / (1. / before_var + size / noise)
            expected_mean = expected_var * (before_mean / before_var + size * np.asarray(event["batch_raw_mean"]) / noise)
            np.testing.assert_allclose(event["raw_posterior_var_after"], expected_var, rtol=1e-12)
            np.testing.assert_allclose(event["raw_posterior_mean_after"], expected_mean, rtol=1e-12)
        self.assertIn(1, sizes)
        self.assertIn(4, sizes)

    def test_unobserved_rows_do_not_change_calibration_acquisition_or_recommendation(self):
        baseline = _run(max_total_question_evaluations=16)
        observed = {tuple(cell) for cell in baseline["observed_cells"]}
        models, questions, changed = _problem()
        for arm, model in enumerate(models):
            for q in questions:
                if (arm, q) not in observed:
                    previous = changed[model][q]
                    changed[model][q] = SampleResult(
                        score=1. - previous.score,
                        latency_seconds=100. * previous.latency_seconds + 1000.,
                        cost=100. * previous.cost + 10., input_tokens={}, output_tokens={},
                    )
        perturbed = _run((models, questions, changed), max_total_question_evaluations=16)
        self.assertNotEqual(perturbed["raw_truth_vectors"], baseline["raw_truth_vectors"])
        self.assertNotEqual(perturbed["bruteforce_search_cost_usd"], baseline["bruteforce_search_cost_usd"])
        for key in ("parameters", "observed_cells", "trace", "points", "raw_posterior_means",
                    "raw_posterior_variances", "total_cost", "stop_reason"):
            self.assertEqual(perturbed[key], baseline[key], key)

    def test_round_robin_decay_changes_only_the_stopped_direction(self):
        def provider(context, arm):
            if context["direction_index"] == 2 and context["current_lambda"] > .5:
                return -100.
            return 100. - arm
        result = _run(index_provider=provider)
        visits = [event for event in result["trace"] if event["event"] == "direction_visit"]
        self.assertEqual([event["direction_index"] for event in visits], [0, 1, 2, 3, 0, 1, 2])
        self.assertEqual(len(result["eta_events"]), 1)
        decay = result["eta_events"][0]
        self.assertEqual(decay["event"], "direction_eta_decay")
        self.assertEqual(decay["direction_index"], 2)
        self.assertEqual(decay["direction_eta_multipliers"], [1., 1., .5, 1.])
        self.assertEqual(result["direction_eta_stages"], [0, 0, 1, 0])
        self.assertEqual(result["counts"], [9, 9, 9])

    def test_shared_observation_reactivates_a_direction_at_its_numerical_floor(self):
        def provider(context, arm):
            if context["direction_index"] == 2 and context["counts"][1] == 4:
                return -100.
            return 100. - arm
        result = _run(index_provider=provider, lambda_initial=1e-6, stop_tolerance=1e-6)
        self.assertEqual(len(result["eta_events"]), 1)
        floor = result["eta_events"][0]
        self.assertEqual(floor["event"], "direction_eta_floor_stop")
        self.assertEqual(floor["direction_index"], 2)
        self.assertEqual(floor["floor_reason"], "numerical_floor")
        later_pulls = [event for event in result["trace"]
                       if event["event"] == "direction_visit" and event["selected_arm"] is not None
                       and event["global_step"] > floor["global_step"]]
        self.assertEqual(later_pulls[0]["direction_index"], 3)
        self.assertTrue(any(event["direction_index"] == 2 for event in later_pulls))
        self.assertEqual(result["counts"], [9, 9, 9])

    def test_small_real_dynamic_program_run_is_deterministic_and_respects_budget(self):
        first = _run(index_provider=None, max_total_question_evaluations=20)
        second = _run(index_provider=None, max_total_question_evaluations=20)
        self.assertEqual(first["stop_reason"], "observation_budget")
        self.assertLessEqual(first["total_evaluations"], 20)
        self.assertGreater(first["total_evaluations"], 12)
        self.assertEqual(first["observed_cells"], second["observed_cells"])
        self.assertEqual(first["points"], second["points"])
        self.assertEqual(first["trace"], second["trace"])
        self.assertGreater(first["axis_boundary_cache_stats"]["builds"], 0)
        self.assertGreater(first["radial_boundary_cache_stats"]["builds"], 0)


if __name__ == "__main__":
    unittest.main()
