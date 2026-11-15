"""Search one SCOPE cache with CC-Gittins and score frozen sets on another."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Sequence

from experiments.combined_objective.offline_multiobjective_random_search import mean_raw_vectors
from experiments.combined_objective.offline_radial_gittins import (
    RadialSimulationResult,
    front_quality_metrics,
    hypervolume_2d,
    mean_cost_metric_vectors,
    raw_pareto_front_indices,
    simulate_radial_gittins,
)
from experiments.combined_objective.run_two_direction_ablation import PAIRS, ROOT
from experiments.single_objective.offline_selector_sim import LookupTable, load_scope


PAIRS_BY_NAME = {
    "restaurant": ("restaurant_valid", "restaurant_test"),
    "data_transformation": ("bing_querylogs", "stackoverflow"),
    "bird": ("bird_mini_dev", "bird_dev"),
}
DEFAULT_BUDGETS = (0.10, 0.30)
DEFAULT_OUTDIR = ROOT / "experiments/combined_objective/results/heldout_scope"
POLICY = {
    "acquisition_target": "finite_benchmark_mean",
    "recommendation_rule": "finite_lcb",
    "recommendation_beta": 1.0,
    "directions": PAIRS["exact_axes"],
    "batch_size": 4,
    "warm_start_batch_size": 4,
    "question_order": "independent",
    "warm_start_question_order": "independent",
    "cost_model": "raw_mean",
    "anytime": True,
    "direction_scheduler": "round_robin",
    "eta_decay_schedule": "direction_stop",
    "lambda_initial": 1.0,
    "lambda_decay": 0.5,
}

Roles = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ScopeCache:
    questions: list[int]
    complete_models: dict[Roles, str]
    table: LookupTable
    input_sha256: dict[str, str]


def load_cache(directory: Path) -> ScopeCache:
    """Match a workflow by the complete role-to-model assignment."""
    models, questions, table = load_scope(str(directory))
    names_to_roles = {}
    manifest = (directory / "configs.json").read_bytes()
    for entry in json.loads(manifest):
        name, roles = entry["model_name"], entry["config"]
        if name in names_to_roles or not isinstance(roles, dict) or not roles:
            raise ValueError(f"{directory}: invalid or duplicate config entry for {name!r}")
        if any(not isinstance(role, str) or not isinstance(model, str)
               for role, model in roles.items()):
            raise ValueError(f"{directory}: roles and models must be strings for {name!r}")
        names_to_roles[name] = tuple(sorted(roles.items()))

    complete_models = {}
    for name in models:
        if name not in names_to_roles:
            raise ValueError(f"{directory}: no role assignment for {name!r}")
        if len(table[name]) != len(questions):
            continue
        roles = names_to_roles[name]
        if roles in complete_models:
            raise ValueError(f"{directory}: ambiguous complete rows for the role assignment {dict(roles)}")
        complete_models[roles] = name

    hashes = {"configs.json": hashlib.sha256(manifest).hexdigest()}
    for name in ("accuracy_matrix.csv", "cost_matrix_usd.csv"):
        hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return ScopeCache(questions, complete_models, table, hashes)


def exclude_bird_search_questions(
    path: Path, search_ids: Sequence[int], dev_ids: Sequence[int],
) -> tuple[list[int], dict]:
    """Exclude every mapped Dev occurrence of each Mini Dev search question."""
    contents = path.read_bytes()
    mapping = json.loads(contents)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: expected an object with source and mini_to_dev fields")
    source, mini_to_dev = mapping.get("source"), mapping.get("mini_to_dev")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{path}: supply the provenance of the cache-aligned question mapping")
    if not isinstance(mini_to_dev, dict) or set(mini_to_dev) != {str(q) for q in search_ids}:
        raise ValueError(f"{path}: mini_to_dev must cover exactly the Mini Dev cache question IDs")
    dev_ids_set = set(dev_ids)
    excluded = set()
    for mini_id, matches in mini_to_dev.items():
        if (not isinstance(matches, list) or not matches
                or any(type(q) is not int or q not in dev_ids_set for q in matches)
                or len(matches) != len(set(matches))):
            raise ValueError(f"{path}: Mini Dev question {mini_id} needs valid, distinct Dev matches")
        excluded.update(matches)
    test_ids = [q for q in dev_ids if q not in excluded]
    if not test_ids:
        raise ValueError(f"{path}: no Dev questions remain for held-out scoring")
    return test_ids, {
        "source": source,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "excluded_dev_question_ids": sorted(excluded),
    }


def freeze_at_budgets(
    result: RadialSimulationResult, full_cost: float, budgets: Sequence[float],
) -> list[dict]:
    """Take the latest completed recommendation within each realized USD cap."""
    if result.recommendation_initial_event is None or result.recommendation_final_event is None:
        raise ValueError("The replay did not record its initial and final recommendations")
    events = [
        result.recommendation_initial_event,
        *result.recommendation_events,
        result.recommendation_final_event,
    ]
    costs = [event.cumulative_search_cost_usd for event in events]
    checkpoints = []
    for budget in budgets:
        cap = budget * full_cost
        index = bisect_right(costs, cap + cap * 1e-12) - 1
        row = {"budget_fraction": budget, "budget_usd": cap, "available": index >= 0}
        if index >= 0:
            event = events[index]
            row.update(
                search_cost_usd=event.cumulative_search_cost_usd,
                search_evaluations=event.cumulative_evaluations,
                selected_indices=list(event.selected_arm_indices),
            )
        else:
            row["reason"] = "warm_start_exceeds_budget"
        checkpoints.append(row)
    return checkpoints


def score_on_test(
    checkpoints: Sequence[dict], test_models: list[str], test_questions: list[int], test_table: LookupTable,
) -> dict:
    """Score every frozen recommendation in the same pool's test metric space."""
    objectives = mean_raw_vectors(test_models, test_questions, test_table)
    pareto = raw_pareto_front_indices(objectives)
    vectors, scale = mean_cost_metric_vectors(objectives)
    frontier = vectors[list(pareto)]
    pareto_ids = set(pareto)
    reference = (0.0, 0.0)
    oracle_hv = hypervolume_2d(frontier, reference)
    scored = []
    for checkpoint in checkpoints:
        if not checkpoint["available"]:
            scored.append(dict(checkpoint))
            continue
        selected = checkpoint["selected_indices"]
        result = front_quality_metrics(vectors[selected], frontier, reference, oracle_hv)
        matches = len(pareto_ids.intersection(selected))
        scored.append({
            **checkpoint,
            "test_metrics": {
                "hypervolume": result.hypervolume,
                "hv_regret": result.hypervolume_regret,
                "gd": result.generational_distance,
                "igd": result.inverted_generational_distance,
                "precision": matches / len(selected),
                "recall": matches / len(pareto),
            },
        })
    return {
        "checkpoints": scored,
        "test_objectives": objectives.tolist(),
        "test_objective_columns": ["mean_quality", "mean_deployment_cost_usd"],
        "test_pareto_indices": list(pareto),
        "test_oracle_hypervolume": oracle_hv,
        "test_cost_reference_usd": scale,
    }


def run_pair(
    pair: str,
    seeds: Sequence[int],
    budgets: Sequence[float],
    outdir: Path,
    *,
    bird_question_map: Path | None = None,
    dry_run: bool = False,
) -> None:
    search_name, test_name = PAIRS_BY_NAME[pair]
    if (pair == "bird") != (bird_question_map is not None):
        raise ValueError("BIRD requires --bird-question-map, which applies only to BIRD")
    if not budgets or any(not math.isfinite(b) or not 0 < b <= 1 for b in budgets):
        raise ValueError("Budget fractions must be in (0, 1]")
    search, test = (load_cache(ROOT / "data" / "scope" / name) for name in (search_name, test_name))
    shared_roles = sorted(search.complete_models.keys() & test.complete_models.keys())
    if len(shared_roles) < 2:
        raise ValueError(f"{pair}: the two caches need at least two common complete configurations")

    test_questions = test.questions
    overlap = None
    if bird_question_map is not None:
        test_questions, overlap = exclude_bird_search_questions(
            bird_question_map, search.questions, test.questions,
        )
    print(f"{search_name} → {test_name}: {len(shared_roles)} common complete configurations, "
          f"{len(search.questions)} search questions, {len(test_questions)} test questions", flush=True)
    if dry_run:
        return

    search_models = [search.complete_models[roles] for roles in shared_roles]
    test_models = [test.complete_models[roles] for roles in shared_roles]
    search_table = {name: search.table[name] for name in search_models}
    full_cost = math.fsum(sample.cost for row in search_table.values() for sample in row.values())
    if not math.isfinite(full_cost) or full_cost <= 0:
        raise ValueError(f"{pair}: exhaustive search cost must be finite and positive")
    configs = [
        {"roles": dict(roles), "search_model_name": left, "test_model_name": right}
        for roles, left, right in zip(shared_roles, search_models, test_models)
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = simulate_radial_gittins(
            search_models, search.questions, search_table, seed=seed, **POLICY,
            record_trace=False, record_recommendation_trajectory=True,
            recommendation_checkpoint_interval=1, defer_recommendation_diagnostics=True,
        )
        checkpoints = freeze_at_budgets(result, full_cost, sorted(set(budgets)))
        scored = score_on_test(checkpoints, test_models, test_questions, test.table)
        payload = {
            "schema_version": 1,
            "pair": pair,
            "seed": seed,
            "search_benchmark": search_name,
            "test_benchmark": test_name,
            "search_question_ids": search.questions,
            "test_question_ids": test_questions,
            "bird_overlap": overlap,
            "input_sha256": {search_name: search.input_sha256, test_name: test.input_sha256},
            "configurations": configs,
            "policy": POLICY,
            "exhaustive_search_cost_usd": full_cost,
            "search_stop_reason": result.stop_reason,
            **scored,
        }
        destination = outdir / f"{pair}-seed{seed}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=outdir, suffix=".tmp", delete=False, encoding="utf-8",
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Saved {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=PAIRS_BY_NAME, default="restaurant")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS,
                        help="fractions of exhaustive search cost, including warm start")
    parser.add_argument("--bird-question-map", type=Path,
                        help="cache-aligned JSON mapping every BIRD Mini Dev ID to matching Dev IDs")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true", help="validate pairing without replaying")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique nonnegative integers")
    try:
        run_pair(args.pair, args.seeds, args.budgets, args.outdir,
                 bird_question_map=args.bird_question_map, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
