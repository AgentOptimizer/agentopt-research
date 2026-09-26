"""Historical replay helpers preserve honest budget snapshots without legacy CLIs."""
import csv
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments.combined_objective import run_lcb_benchmarks as runner
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


def toy_problem():
    models, questions = ["arm_0"], list(range(4))
    table = {"arm_0": {
        question: SampleResult(
            score=float(question != 0), cost=0.1, latency_seconds=0.1,
            input_tokens={}, output_tokens={},
        ) for question in questions
    }}
    return models, questions, table


def toy_result(rule="finite_lcb", *, cost_model=None, recommendation_min_samples=0, question_order="shared"):
    return replay.simulate_radial_gittins(
        *toy_problem(), seed=42, batch_size=1, anytime=True,
        eta_decay_schedule="direction_stop", directions=((1.0, 0.0),),
        recommendation_rule=rule, recommendation_beta=1.0,
        recommendation_min_samples=recommendation_min_samples, question_order=question_order,
        cost_model=cost_model or ("raw_mean" if rule in ("finite_lcb", "finite_mean") else "reciprocal"),
        cost_reference_usd=1.0, prior_variance=1.0, obs_noise_variance=1.0,
        effective_cost_bin_ratio=None, index_provider=lambda context, arm: 100.0,
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True, defer_recommendation_diagnostics=True,
    )


class FiniteLcbBenchmarkTests(unittest.TestCase):
    def test_source_mapping_dispatches_correct_loader_and_hashes_exact_files(self):
        expected = {
            "hotpotqa": ("pickle", "experiments/data/lookup/hotpotqa_lookup.pkl"),
            "mathqa": ("pickle", "experiments/data/lookup/mathqa_lookup.pkl"),
            "restaurant_test": ("scope", "data/scope/restaurant_test"),
            "stackoverflow": ("scope", "data/scope/stackoverflow"),
            "restaurant_valid": ("scope", "data/scope/restaurant_valid"),
            "bird_dev": ("scope", "data/scope/bird_dev"),
        }
        self.assertEqual(runner.DATASETS, expected)
        self.assertEqual(runner.DEFAULT_BENCHMARKS, ("hotpotqa", "mathqa", "restaurant_test", "stackoverflow"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for benchmark, (kind, relative) in expected.items():
                with self.subTest(benchmark=benchmark):
                    path = root / relative
                    files = [path] if kind == "pickle" else [path / name for name in (
                        "accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json",
                    )]
                    hashes = {}
                    for index, file in enumerate(files):
                        file.parent.mkdir(parents=True, exist_ok=True)
                        content = f"{benchmark}:{index}".encode()
                        file.write_bytes(content)
                        hashes[str(file.relative_to(root))] = hashlib.sha256(content).hexdigest()
                    problem = toy_problem()
                    with mock.patch.object(runner, "ROOT", root), mock.patch.object(
                        runner, "load_pickle", return_value=problem,
                    ) as pickle_loader, mock.patch.object(
                        runner, "load_scope", return_value=problem,
                    ) as scope_loader:
                        loaded = runner.load_benchmark(benchmark)
                    active, inactive = (pickle_loader, scope_loader) if kind == "pickle" else (scope_loader, pickle_loader)
                    active.assert_called_once_with(str(path))
                    inactive.assert_not_called()
                    self.assertEqual(loaded[:3], problem)
                    self.assertEqual(loaded[3], hashes)

    def test_legacy_modules_keep_library_helpers_without_command_entrypoints(self):
        from experiments.combined_objective import compare_lcb_recommendations as reports
        from experiments.combined_objective.plot import plot_lcb_recommendations as plots

        for module in (runner, reports, plots):
            with self.subTest(module=module.__name__):
                self.assertFalse(hasattr(module, "main"))
        self.assertTrue(callable(reports.compact_lcb_run))
        self.assertTrue(callable(plots.with_reference))

    def test_simulation_dispatch_keeps_seed42_async_finite_lcb_and_deferred_diagnostics(self):
        problem, cache = toy_problem(), object()
        result = SimpleNamespace(total_evaluations=4, total_cost=0.4)
        with mock.patch.object(runner, "simulate_radial_gittins", return_value=result) as simulate:
            actual, elapsed = runner.simulate_rule("hotpotqa", "finite_lcb", *problem, cache, 1.0)
        self.assertIs(actual, result)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(simulate.call_count, 1)
        self.assertEqual(simulate.call_args.args, problem)
        kwargs = simulate.call_args.kwargs
        expected = {
            "seed": 42, "batch_size": 4, "anytime": True, "question_order": "shared",
            "direction_scheduler": "round_robin", "eta_decay_schedule": "direction_stop",
            "recommendation_rule": "finite_lcb", "recommendation_beta": 1.0,
            "lambda_initial": 1.0, "lambda_decay": 0.5,
            "search_cost_scale_eta": 1.0, "observation_budget_fraction": 1.0,
            "boundary_z_padding_extra": 2.0, "effective_cost_bin_ratio": 2.0,
            "record_trace": True, "record_recommendation_trajectory": True,
            "recommendation_changes_only": True, "defer_recommendation_diagnostics": True,
            "recommendation_checkpoint_interval": None,
        }
        for key, value in expected.items():
            self.assertEqual(kwargs[key], value, key)
        self.assertIs(kwargs["boundary_cache"], cache)
        grid = kwargs["boundary_grid"]
        self.assertEqual((grid.z_size, grid.delta_size, grid.state_size, grid.boundary_margin_cells), (129, 129, 129, 4))

    def test_one_finite_lcb_run_serializes_and_final_refreshes_unchanged_membership(self):
        result = toy_result()
        with tempfile.TemporaryDirectory() as temporary:
            outdir, cache = Path(temporary), object()
            problem = toy_problem()
            with mock.patch.object(runner, "load_benchmark", return_value=(*problem, {"fixture": "sha"})), mock.patch.object(
                runner, "RadialGittinsBoundaryCache", return_value=cache,
            ), mock.patch.object(runner, "simulate_rule", return_value=(result, 1.25)) as simulate:
                saved = runner.run_benchmark("hotpotqa", outdir, 1.0)
            simulate.assert_called_once_with("hotpotqa", "finite_lcb", *problem, cache, 1.0,
                                             cost_model="raw_mean", cost_reference_usd=None,
                                             recommendation_min_samples=0, question_order="shared")
            self.assertEqual(set(saved["runs"]), {"finite_lcb"})
            self.assertEqual(saved["config"]["recommendation_rule"], "finite_lcb")
            self.assertEqual(saved["config"]["policy_wall_time_seconds"], 1.25)
            self.assertEqual(saved["config"]["lookup_sha256"], {"fixture": "sha"})
            folder = outdir / "hotpotqa"
            self.assertEqual(json.loads((folder / "comparison.json").read_text()), json.loads(json.dumps(saved, allow_nan=False)))
            with gzip.open(folder / "finite_lcb_trace.json.gz", "rt") as handle:
                self.assertEqual(json.load(handle), result.trace)
            self.assertEqual({file.name for file in folder.iterdir()}, {"comparison.json", "finite_lcb_trace.json.gz"})
            points = saved["runs"]["finite_lcb"]["points"]
            self.assertEqual([point["snapshot_role"] for point in points], ["warm_start", "final"])
            self.assertEqual([point["selected_sample_counts"] for point in points], [[1], [4]])
            self.assertEqual([point["partial_recommended_count"] for point in points], [1, 0])
            self.assertEqual(points[0]["selected_arm_indices"], points[-1]["selected_arm_indices"])

    def test_completed_helper_keeps_acquisition_settings(self):
        result = SimpleNamespace(total_evaluations=4, total_cost=.4)
        with mock.patch.object(runner, "simulate_radial_gittins", return_value=result) as simulate:
            runner.simulate_rule("restaurant_valid", "completed_only", *toy_problem(), object(), 1., cost_model="raw_mean")
        self.assertEqual(simulate.call_args.kwargs["direction_scheduler"], "round_robin")
        self.assertEqual(simulate.call_args.kwargs["eta_decay_schedule"], "direction_stop")
        self.assertEqual(simulate.call_args.kwargs["recommendation_rule"], "completed_only")

    def test_completed_rule_serializes_and_summarizes_under_its_own_name(self):
        result = toy_result("completed_only", cost_model="raw_mean")
        with tempfile.TemporaryDirectory() as temporary:
            outdir, cache = Path(temporary), object()
            problem = toy_problem()
            with mock.patch.object(runner, "load_benchmark", return_value=(*problem, {"fixture": "sha"})), mock.patch.object(
                runner, "RadialGittinsBoundaryCache", return_value=cache,
            ), mock.patch.object(runner, "simulate_rule", return_value=(result, 1.25)) as simulate:
                saved = runner.run_benchmark("hotpotqa", outdir, 1.0, recommendation_rule="completed_only")
            simulate.assert_called_once_with("hotpotqa", "completed_only", *problem, cache, 1.0,
                                             cost_model="raw_mean", cost_reference_usd=None,
                                             recommendation_min_samples=0, question_order="shared")
            self.assertEqual(set(saved["runs"]), {"completed_only"})
            self.assertEqual(saved["config"]["recommendation_rule"], "completed_only")
            self.assertIsNone(saved["config"]["recommendation_beta"])
            folder = outdir / "hotpotqa"
            self.assertEqual({file.name for file in folder.iterdir()}, {"comparison.json", "completed_only_trace.json.gz"})
            with gzip.open(folder / "completed_only_trace.json.gz", "rt") as handle:
                self.assertEqual(json.load(handle), result.trace)
            run = saved["runs"]["completed_only"]
            self.assertEqual(run["recommendation_rule"], "completed_only")
            self.assertEqual(run["points"][0]["selected_arm_indices"], [])
            self.assertEqual(run["points"][-1]["selected_arm_indices"], [0])
            self.assertTrue(all(point["partial_recommended_count"] == 0 for point in run["points"]))
            summaries, matched = runner.export_summary(outdir, ["hotpotqa"])
            self.assertEqual(summaries[0]["method"], "completed_only")
            self.assertTrue(all(point["method"] == "completed_only" for point in matched))

    def test_finite_helper_preserves_beta_and_exploration(self):
        result = SimpleNamespace(total_evaluations=4, total_cost=.4)
        with mock.patch.object(runner, "simulate_radial_gittins", return_value=result) as simulate:
            runner.simulate_rule("bird_dev", "finite_lcb", *toy_problem(), object(), 1.5, cost_model="raw_mean")
        self.assertEqual(simulate.call_args.kwargs["recommendation_beta"], 1.5)
        self.assertEqual(simulate.call_args.kwargs["direction_scheduler"], "round_robin")
        self.assertEqual(simulate.call_args.kwargs["eta_decay_schedule"], "direction_stop")

    def test_finite_rule_rejects_incompatible_cost_model(self):
        with mock.patch.object(runner, "load_benchmark") as load, self.assertRaises(ValueError):
            runner.run_benchmark("bird_dev", Path("unused"), 1.0,
                                 recommendation_rule="finite_lcb", cost_model="reciprocal")
        load.assert_not_called()

    def test_finite_rule_serializes_target_moments_and_completed_zero_uncertainty(self):
        result = toy_result("finite_lcb")
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            with mock.patch.object(runner, "load_benchmark", return_value=(*toy_problem(), {"fixture": "sha"})), mock.patch.object(
                runner, "RadialGittinsBoundaryCache", return_value=object(),
            ), mock.patch.object(runner, "simulate_rule", return_value=(result, 1.25)):
                saved = runner.run_benchmark("hotpotqa", outdir, 1.0, recommendation_rule="finite_lcb", cost_model="raw_mean")
            self.assertEqual(set(saved["runs"]), {"finite_lcb"})
            self.assertEqual(saved["config"]["recommendation_beta"], 1.0)
            self.assertEqual(saved["config"]["recommendation_semantics"], "finite_test_mean_accuracy_lcb_mean_usd_ucb_pareto_of_all_arms")
            self.assertTrue((outdir / "hotpotqa" / "finite_lcb_trace.json.gz").is_file())
            restored = json.loads((outdir / "hotpotqa" / "comparison.json").read_text())
            run = restored["runs"]["finite_lcb"]
            initial, final = run["points"][0], run["points"][-1]
            self.assertEqual(initial["selected_sample_counts"], [1])
            self.assertEqual(initial["partial_recommended_count"], 1)
            self.assertEqual(final["selected_sample_counts"], [4])
            self.assertEqual(final["finite_target_std_vectors"], [[0.0, 0.0]])
            self.assertEqual(final["finite_target_mean_vectors"], final["estimated_raw_archive_vectors"])
            self.assertEqual(final["recommendation_raw_vectors"], final["finite_target_mean_vectors"])
            self.assertGreater(initial["finite_target_std_vectors"][0][0], 0.0)
            self.assertEqual(run["recommendation_filter"], "all_arms_finite_test_lcb_raw_pareto")
            self.assertEqual(run["recommendation_evidence_scope"], "selected_finite_test_lcb_frontier")
            summaries, _ = runner.export_summary(outdir, ["hotpotqa"])
            self.assertEqual(summaries[0]["method"], "finite_lcb")

    def test_finite_mean_helper_passes_observation_gate_and_forces_zero_beta(self):
        with mock.patch.object(runner, "simulate_radial_gittins", return_value=SimpleNamespace(total_evaluations=4, total_cost=.4)) as simulate:
            runner.simulate_rule("bird_dev", "finite_mean", *toy_problem(), object(), 9.,
                                 cost_model="raw_mean", recommendation_min_samples=32)
        kwargs = simulate.call_args.kwargs
        self.assertEqual(kwargs["recommendation_beta"], 0.0)
        self.assertEqual(kwargs["recommendation_min_samples"], 32)
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["eta_decay_schedule"], "direction_stop")

    def test_sample_gate_rejects_invalid_counts_and_incompatible_rules_before_loading(self):
        for rule, count in (("finite_mean", -1), ("finite_mean", 1.5), ("finite_lcb", True),
                            ("completed_only", 32), ("hybrid_lcb", 0), ("lcb", 0)):
            with self.subTest(rule=rule, count=count), mock.patch.object(runner, "load_benchmark") as load, self.assertRaises(ValueError):
                runner.run_benchmark("hotpotqa", Path("unused"), 1., recommendation_rule=rule,
                                     cost_model="raw_mean", recommendation_min_samples=count)
            load.assert_not_called()
        with mock.patch.object(runner, "load_benchmark") as load, self.assertRaises(ValueError):
            runner.run_benchmark("hotpotqa", Path("unused"), 1., recommendation_rule="finite_mean",
                                 cost_model="reciprocal", recommendation_min_samples=32)
        load.assert_not_called()

    def test_finite_mean_serializes_gated_mean_evidence_and_zero_effective_beta(self):
        result = toy_result("finite_mean", recommendation_min_samples=2)
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            with mock.patch.object(runner, "load_benchmark", return_value=(*toy_problem(), {"fixture": "sha"})), mock.patch.object(
                runner, "RadialGittinsBoundaryCache", return_value=object(),
            ), mock.patch.object(runner, "simulate_rule", return_value=(result, 1.25)) as simulate:
                saved = runner.run_benchmark("hotpotqa", outdir, 9.0, recommendation_rule="finite_mean",
                                             cost_model="raw_mean", recommendation_min_samples=2)
            self.assertEqual(simulate.call_args.args[-1], 0.0)
            self.assertEqual(simulate.call_args.kwargs["recommendation_min_samples"], 2)
            config = saved["config"]
            self.assertEqual(config["recommendation_beta"], 0.0)
            self.assertEqual(config["recommendation_min_samples"], 2)
            self.assertIn("without_std_penalty", config["recommendation_semantics"])
            restored = json.loads((outdir / "hotpotqa" / "comparison.json").read_text())
            run = restored["runs"]["finite_mean"]
            self.assertTrue((outdir / "hotpotqa" / "finite_mean_trace.json.gz").exists())
            self.assertEqual(run["recommendation_min_samples"], 2)
            self.assertEqual(run["recommendation_beta"], 0.0)
            self.assertEqual(run["recommendation_filter"], "eligible_arms_finite_test_mean_raw_pareto")
            self.assertEqual(run["points"][0]["selected_arm_indices"], [])
            first = next(point for point in run["points"] if point["selected_arm_indices"])
            self.assertEqual(first["selected_sample_counts"], [2])
            self.assertGreater(first["finite_target_std_vectors"][0][0], 0.0)
            for point in run["points"]:
                self.assertEqual(point["recommendation_beta"], 0.0)
                self.assertEqual(point["recommendation_min_samples"], 2)
                self.assertEqual(point["recommendation_raw_vectors"], point["finite_target_mean_vectors"])
                self.assertTrue(all(count >= 2 for count in point["selected_sample_counts"]))
            summaries, _ = runner.export_summary(outdir, ["hotpotqa"])
            self.assertEqual(summaries[0]["method"], "finite_mean")

    def test_finite_plot_overlay_accepts_matching_completed_reference_and_checks_provenance(self):
        from experiments.combined_objective.plot.plot_lcb_recommendations import with_completed_reference

        config = {"recommendation_rule": "finite_lcb", "lookup_sha256": {"fixture.pkl": "same"},
                  "acquisition_validation": {"exact_physical_sample_sequence": True}}
        saved = {"config": config, "runs": {"finite_lcb": runner.compact_lcb_run(toy_result("finite_lcb"))}}
        reference = {"config": {"lookup_sha256": {"fixture.pkl": "same"}},
                     "runs": {"completed_only": runner.compact_lcb_run(toy_result("completed_only", cost_model="raw_mean"))}}
        merged = with_completed_reference(saved, reference)
        self.assertEqual(list(merged["runs"]), ["completed_only", "finite_lcb"])
        self.assertEqual(merged["config"]["recommendation_rule"], "finite_lcb")
        self.assertNotIn("acquisition_validation", merged["config"])
        self.assertIn("acquisition_validation", saved["config"])
        for key, value in (("seed", 43), ("metric_cost_reference_usd", 123.0),
                           ("recommendation_filter", "historical_direction_winners")):
            with self.subTest(key=key):
                mismatched = copy.deepcopy(reference)
                mismatched["runs"]["completed_only"][key] = value
                with self.assertRaises(ValueError):
                    with_completed_reference(saved, mismatched)
        mismatched = copy.deepcopy(reference)
        mismatched["config"]["lookup_sha256"]["fixture.pkl"] = "different"
        with self.assertRaises(ValueError):
            with_completed_reference(saved, mismatched)

    def test_summary_uses_last_available_event_without_replaying_or_claiming_current_counts(self):
        run = runner.compact_lcb_run(toy_result())
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            folder = outdir / "hotpotqa"
            folder.mkdir()
            runner.save_json(folder / "comparison.json", {"runs": {"finite_lcb": run}})
            with mock.patch.object(
                runner, "run_benchmark", side_effect=AssertionError("replayed benchmark"),
            ), mock.patch.object(runner, "load_benchmark", side_effect=AssertionError("loaded source")):
                runner.export_summary(outdir, ["hotpotqa"])
            with (outdir / "matched_budgets.csv").open(newline="") as handle:
                rows = {float(row["budget_fraction"]): row for row in csv.DictReader(handle)}
            self.assertEqual(json.loads(rows[0.20]["selected_arm_indices"]), [])
            self.assertEqual(rows[0.20]["recommendation_event_cost_fraction"], "")
            self.assertEqual(float(rows[0.20]["relative_hv_regret"]), 1.0)
            # The arm is partially observed at the event even if further
            # samples were acquired before this later matched budget.
            for fraction in (0.25, 0.50, 0.75):
                row = rows[fraction]
                self.assertEqual(float(row["recommendation_event_cost_fraction"]), 0.25)
                self.assertEqual(float(row["budget_usd"]), fraction * 0.4)
                self.assertEqual(row["partial_recommended_count_at_recommendation_event"], "1")
                self.assertNotIn("partial_recommended_count", row)
                self.assertEqual(json.loads(row["selected_arm_indices"]), [0])
            self.assertEqual(rows[1.0]["partial_recommended_count_at_recommendation_event"], "0")
            self.assertEqual(float(rows[1.0]["recommendation_event_cost_fraction"]), 1.0)
            with (outdir / "summary.csv").open(newline="") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["method"], "finite_lcb")
            self.assertEqual(summaries[0]["membership_change_count"], "0")

    def test_repeatable_plot_references_merge_three_rules_and_keep_primary_alias(self):
        from experiments.combined_objective.plot import plot_lcb_recommendations as plot

        def payload(rule, minimum=0):
            return {"config": {"recommendation_rule": rule, "lookup_sha256": {"fixture.pkl": "same"}},
                    "runs": {rule: runner.compact_lcb_run(toy_result(rule, cost_model="raw_mean",
                                                                     recommendation_min_samples=minimum))}}

        primary, finite, completed = payload("finite_mean", 2), payload("finite_lcb"), payload("completed_only")
        merged = plot.with_reference(primary, completed)
        merged = plot.with_reference(merged, finite)
        self.assertEqual(list(merged["runs"]), ["completed_only", "finite_lcb", "finite_mean"])
        self.assertEqual(merged["config"]["recommendation_rule"], "finite_mean")
        self.assertEqual(list(primary["runs"]), ["finite_mean"])
        self.assertEqual(plot._method_label("finite_mean", {"recommendation_min_samples": 32}),
                         "Full-test mean Pareto (n≥32)")
        self.assertIn("finite_mean", plot.PARTIAL_RULES)
        self.assertNotIn("finite_mean", plot.PENALIZED_RULES)
        with self.assertRaises(ValueError):
            plot.with_reference(merged, finite)
        mismatched = copy.deepcopy(finite)
        mismatched["runs"]["finite_lcb"]["seed"] = 43
        with self.assertRaises(ValueError):
            plot.with_reference(primary, mismatched)

    def test_independent_question_order_is_explicit_in_helper(self):
        with mock.patch.object(runner, "simulate_radial_gittins", return_value=SimpleNamespace(total_evaluations=4,total_cost=.4)) as simulate:
            runner.simulate_rule("stackoverflow", "finite_mean", *toy_problem(), object(), 9., cost_model="raw_mean",
                                 recommendation_min_samples=32, question_order="independent")
        self.assertEqual(simulate.call_args.kwargs["question_order"], "independent")
        self.assertEqual(simulate.call_args.kwargs["recommendation_beta"], 0.)
        with mock.patch.object(runner, "load_benchmark") as load, self.assertRaises(ValueError):
            runner.run_benchmark("hotpotqa", Path("unused"), 1., question_order="invalid")
        load.assert_not_called()

    def test_question_order_is_exported_and_historical_missing_means_independent(self):
        for order in ("shared", "independent"):
            result = toy_result("finite_mean", recommendation_min_samples=2, question_order=order)
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temporary:
                with mock.patch.object(runner, "load_benchmark", return_value=(*toy_problem(), {})), mock.patch.object(
                    runner, "RadialGittinsBoundaryCache", return_value=object(),
                ), mock.patch.object(runner, "simulate_rule", return_value=(result, 0.1)) as simulate:
                    saved = runner.run_benchmark("hotpotqa", Path(temporary), 1., recommendation_rule="finite_mean",
                                                 cost_model="raw_mean", recommendation_min_samples=2, question_order=order)
                self.assertEqual(saved["config"]["question_order"], order)
                self.assertEqual(saved["runs"]["finite_mean"]["question_order"], order)
                self.assertEqual(saved["runs"]["finite_mean"]["question_order_semantics"], result.params["question_order_semantics"])
                self.assertEqual(simulate.call_args.kwargs["question_order"], order)
            result.params.pop("question_order")
            legacy = runner.compact_lcb_run(result)
            self.assertEqual(legacy["question_order"], "independent")
            legacy.pop("question_order")
            self.assertEqual(runner.summarize("finite_mean", legacy)["question_order"], "independent")

    def test_existing_other_question_order_is_protected_before_loading_or_sampling(self):
        existing_values = (
            {"config": {}, "runs": {"finite_mean": {}}},
            {"config": {"question_order": "independent"}, "runs": {}},
            {"runs": {"finite_mean": {"question_order": "independent"}}},
        )
        for existing in existing_values:
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "hotpotqa" / "comparison.json"
                path.parent.mkdir()
                path.write_text(json.dumps(existing))
                original = path.read_bytes()
                with mock.patch.object(runner, "load_benchmark") as load, mock.patch.object(runner, "simulate_rule") as simulate, self.assertRaisesRegex(ValueError, "new --outdir"):
                    runner.run_benchmark("hotpotqa", Path(temporary), 1.)
                load.assert_not_called()
                simulate.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                with mock.patch.object(runner, "load_benchmark", side_effect=RuntimeError("loading allowed")), self.assertRaisesRegex(RuntimeError, "loading allowed"):
                    runner.run_benchmark("hotpotqa", Path(temporary), 1., question_order="independent")

    def test_plot_reference_allows_different_orders_and_keeps_legacy_reference_independent(self):
        from experiments.combined_objective.plot import plot_lcb_recommendations as plot
        primary = {"config": {"question_order": "shared", "lookup_sha256": {"fixture": "same"},
                              "acquisition_validation": {"exact_physical_sample_sequence": True}},
                   "runs": {"finite_mean": runner.compact_lcb_run(toy_result("finite_mean"))}}
        reference = {"config": {"lookup_sha256": {"fixture": "same"}},
                     "runs": {"completed_only": runner.compact_lcb_run(toy_result("completed_only", cost_model="raw_mean", question_order="independent"))}}
        reference["runs"]["completed_only"].pop("question_order")
        merged = plot.with_reference(primary, reference)
        self.assertEqual(merged["runs"]["finite_mean"]["question_order"], "shared")
        self.assertEqual(merged["runs"]["completed_only"]["question_order"], "independent")
        self.assertEqual(plot._plot_context(merged["config"], merged["runs"]["completed_only"])["question_order"], "independent")
        self.assertNotIn("acquisition_validation", merged["config"])
        self.assertNotIn("question_order", reference["runs"]["completed_only"])
        self.assertIn("[shared]", plot._method_label("finite_mean", merged["runs"]["finite_mean"], show_question_order=True))


if __name__ == "__main__":
    unittest.main()
