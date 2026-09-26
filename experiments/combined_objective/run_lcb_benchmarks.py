"""Library helpers for validating and reading historical recommendation replays.

The standalone benchmark command is retired. Maintained experiment commands live
in the ablation runners; these helpers remain available for saved-result checks.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]

from agentopt.model_selection.radial_gittins_dp import RadialGittinsBoundaryCache, RadialGittinsGrid
from experiments.combined_objective.compare_lcb_recommendations import (
    at_budget, compact_lcb_run, summarize, write_csv,
)
from experiments.combined_objective.offline_radial_gittins import (
    DEFAULT_RADIAL_BOUNDARY_CACHE_DIR, load_pickle, load_scope,
    simulate_radial_gittins,
)

DATASETS = {
    "hotpotqa": ("pickle", "experiments/data/lookup/hotpotqa_lookup.pkl"),
    "mathqa": ("pickle", "experiments/data/lookup/mathqa_lookup.pkl"),
    "restaurant_test": ("scope", "data/scope/restaurant_test"),
    "stackoverflow": ("scope", "data/scope/stackoverflow"),
    "restaurant_valid": ("scope", "data/scope/restaurant_valid"),
    "bird_dev": ("scope", "data/scope/bird_dev"),
}
DEFAULT_BENCHMARKS = ("hotpotqa", "mathqa", "restaurant_test", "stackoverflow")
DISPLAY_NAMES = {"hotpotqa": "HotpotQA", "mathqa": "MathQA",
                 "restaurant_test": "Restaurant test", "stackoverflow": "Stack Overflow",
                 "restaurant_valid": "Restaurant validation", "bird_dev": "BIRD Dev"}
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)


def validate_question_order(question_order):
    if question_order not in ("shared", "independent"):
        raise ValueError("question_order must be 'shared' or 'independent'")


def validate_recommendation_options(rule, cost_model, recommendation_min_samples=0):
    if rule not in ("finite_lcb", "finite_mean", "completed_only"):
        raise ValueError("recommendation_rule must be finite_lcb, finite_mean, or completed_only")
    if rule in ("finite_lcb", "finite_mean") and cost_model != "raw_mean":
        raise ValueError(f"{rule} requires cost_model='raw_mean'")
    if isinstance(recommendation_min_samples, bool) or not isinstance(recommendation_min_samples, int) or recommendation_min_samples < 0:
        raise ValueError("recommendation_min_samples must be a nonnegative integer")
    if recommendation_min_samples and rule not in ("finite_lcb", "finite_mean"):
        raise ValueError("recommendation_min_samples only applies to finite_lcb and finite_mean")


def load_benchmark(benchmark):
    kind, relative = DATASETS[benchmark]
    path = ROOT / relative
    files = [path] if kind == "pickle" else [path / name for name in
        ("accuracy_matrix.csv", "cost_matrix_usd.csv", "metadata.json")]
    hashes = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    models, questions, table = (load_pickle if kind == "pickle" else load_scope)(str(path))
    return models, questions, table, hashes


def simulate_rule(benchmark, rule, models, questions, table, cache, beta, *,
                  cost_model="raw_mean", cost_reference_usd=None, recommendation_min_samples=0,
                  question_order="shared"):
    validate_recommendation_options(rule, cost_model, recommendation_min_samples)
    validate_question_order(question_order)
    beta = 0.0 if rule == "finite_mean" else beta
    started, finished = time.perf_counter(), threading.Event()

    def progress():
        while not finished.wait(30):
            stats = cache.stats_snapshot()
            print(f"{benchmark} {rule}: {time.perf_counter() - started:.0f}s; "
                  f"cache builds={stats.builds}, disk hits={stats.disk_hits}", flush=True)

    reporter = threading.Thread(target=progress, daemon=True)
    reporter.start()
    penalty = "" if rule in ("completed_only", "finite_mean") else f", beta={beta:g}"
    if recommendation_min_samples:
        penalty += f", n>={recommendation_min_samples} observed questions"
    print(f"Running {benchmark}: {rule}, cost={cost_model}, seed42, {question_order} question order, asynchronous eta{penalty}", flush=True)
    try:
        result = simulate_radial_gittins(
            models, questions, table, anytime=True, direction_scheduler="round_robin",
            eta_decay_schedule="direction_stop", recommendation_rule=rule, recommendation_beta=beta,
            recommendation_min_samples=recommendation_min_samples,
            question_order=question_order,
            cost_model=cost_model, cost_reference_usd=cost_reference_usd,
            seed=42, batch_size=4, lambda_initial=1.0, lambda_decay=0.5,
            search_cost_scale_eta=1.0, observation_budget_fraction=1.0,
            boundary_z_padding_extra=2.0, effective_cost_bin_ratio=2.0,
            boundary_grid=RadialGittinsGrid(z_size=129, delta_size=129, state_size=129,
                                          boundary_margin_cells=4),
            boundary_cache=cache, record_trace=True, record_recommendation_trajectory=True,
            recommendation_checkpoint_interval=None, recommendation_changes_only=True,
            defer_recommendation_diagnostics=True,
        )
    finally:
        finished.set()
        reporter.join()
    elapsed = time.perf_counter() - started
    print(f"Finished {benchmark} {rule}: {result.total_evaluations:,} cells, "
          f"${result.total_cost:.6f}, {elapsed:.1f}s", flush=True)
    return result, elapsed


def save_json(path, value):
    # One bulk write after a run; allow_nan=False catches invalid exported metrics.
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_benchmark(benchmark, outdir, beta, *, recommendation_rule="finite_lcb",
                  cost_model="raw_mean", cost_reference_usd=None, recommendation_min_samples=0,
                  question_order="shared"):
    validate_recommendation_options(recommendation_rule, cost_model, recommendation_min_samples)
    validate_question_order(question_order)
    existing_path = outdir / benchmark / "comparison.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        existing_order = existing.get("config", {}).get("question_order")
        if existing_order is None:
            orders = {run.get("question_order", "independent") for run in existing.get("runs", {}).values()}
            existing_order = next(iter(orders)) if len(orders) == 1 else "mixed" if orders else "independent"
        if existing_order != question_order:
            raise ValueError(f"Existing {existing_path} uses {existing_order} question order; requested {question_order}. "
                             "Choose a new --outdir to preserve the saved run.")
    beta = 0.0 if recommendation_rule == "finite_mean" else beta
    models, questions, table, hashes = load_benchmark(benchmark)
    folder = outdir / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    cache = RadialGittinsBoundaryCache(cache_dir=DEFAULT_RADIAL_BOUNDARY_CACHE_DIR)
    config = {
        "benchmark": benchmark, "seed": 42,
        "question_order": question_order,
        "recommendation_beta": None if recommendation_rule == "completed_only" else beta,
        "recommendation_rule": recommendation_rule,
        "recommendation_min_samples": recommendation_min_samples,
        "recommendation_eligibility": (
            "observed_question_count_at_least_min_samples" if recommendation_min_samples else "no_minimum_sample_gate"
        ),
        "cost_model": cost_model, "cost_reference_usd_requested": cost_reference_usd,
        "algorithm": "Radial-Gittins-decay", "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin", "batch_size": 4, "grid_size": 129,
        "eta_initial": 1.0, "eta_decay": 0.5, "search_cost_scale_eta": 1.0,
        "boundary_z_padding_extra": 2.0, "effective_cost_bin_ratio": 2.0,
        "observation_budget_fraction": 1.0, "question_universe": "common",
        "recommendation_changes_only": True, "defer_recommendation_diagnostics": True,
        "lookup_sha256": hashes,
        "replay_source_sha256": hashlib.sha256(
            (ROOT / "experiments/combined_objective/offline_radial_gittins.py").read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "budget_alignment": "Latest available recommendation at or below actual cumulative USD budget",
        "metric_space": (
            "Full-data quality of actual recommended arms in the separate offline mean-USD metric scale"
        ),
        "recommendation_semantics": (
            "empirical_mean_accuracy_mean_usd_pareto_of_all_completed_arms"
            if recommendation_rule == "completed_only" else
            "finite_test_mean_accuracy_lcb_mean_usd_ucb_pareto_of_all_arms"
            if recommendation_rule == "finite_lcb" and not recommendation_min_samples else
            "finite_test_mean_accuracy_lcb_mean_usd_ucb_pareto_of_eligible_arms"
            if recommendation_rule == "finite_lcb" else
            "finite_test_mean_accuracy_mean_usd_pareto_of_eligible_arms_without_std_penalty"
            if recommendation_rule == "finite_mean" else "posterior_direction_winners"
        ),
        "estimate_snapshot_semantics": "Estimates and sample/completion counts are frozen at membership events; they are not current at a later matched budget",
        "runtime_note": "Shared boundary cache; timings are descriptive, not a controlled speed comparison",
    }
    result, seconds = simulate_rule(
        benchmark, recommendation_rule, models, questions, table, cache, beta,
        cost_model=cost_model, cost_reference_usd=cost_reference_usd,
        recommendation_min_samples=recommendation_min_samples,
        question_order=question_order,
    )
    run = compact_lcb_run(result)
    config["policy_wall_time_seconds"] = seconds
    with gzip.open(folder / f"{recommendation_rule}_trace.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(result.trace, handle)
    saved = {"config": config, "runs": {recommendation_rule: run}, "parameters": result.params,
             "direction_eta_events": result.direction_eta_events}
    save_json(folder / "comparison.json", saved)
    print(f"Saved {benchmark} {recommendation_rule} result: {folder}", flush=True)
    return saved


def export_summary(outdir, benchmarks):
    summaries, matched = [], []
    for benchmark in benchmarks:
        path = outdir / benchmark / "comparison.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        for method, run in saved["runs"].items():
            summaries.append({"benchmark": benchmark, **summarize(method, run)})
            for budget in BUDGETS:
                point = at_budget(run, budget)
                matched.append({
                    "benchmark": benchmark, "method": method, "budget_fraction": budget,
                    "question_order": run.get("question_order", "independent"),
                    "budget_usd": budget * run["bruteforce_search_cost_usd"],
                    "recommendation_event_cost_fraction": point["cost_fraction"] if point else None,
                    "best_recommended_accuracy": point["best_recommended_accuracy"] if point else None,
                    "accuracy_gap": point["accuracy_gap"] if point else None,
                    "relative_hv_regret": point["relative_hv_regret"] if point else 1.0,
                    "contains_true_accuracy_best": point["contains_true_accuracy_best"] if point else False,
                    "dominated_recommendation_count": point["offline_dominated_selected_count"] if point else 0,
                    "partial_recommended_count_at_recommendation_event": point["partial_recommended_count"] if point else 0,
                    "selected_arm_indices": json.dumps(point["selected_arm_indices"] if point else []),
                })
    write_csv(outdir / "summary.csv", summaries)
    write_csv(outdir / "matched_budgets.csv", matched)
    if summaries:
        print("\nHighest-accuracy recommendation (fraction of full USD cost):", flush=True)
        for row in summaries:
            first, sustained = row["first_best_accuracy_cost_fraction"], row["sustained_best_accuracy_cost_fraction"]
            print(f"  {row['benchmark']:16s} {row['method']:22s} "
                  f"first={'never' if first is None else format(first, '.2%'):>8s} "
                  f"sustained={'never' if sustained is None else format(sustained, '.2%'):>8s} "
                  f"retractions={row['highest_accuracy_retraction_count']}", flush=True)
    return summaries, matched
