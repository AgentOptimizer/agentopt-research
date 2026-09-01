#!/usr/bin/env python3
"""Offline multi-objective random-search baselines on frozen response matrices.

Two equal-cell-budget protocols are implemented:

``random_configurations``
    Sample a fraction of configurations and evaluate every common question for
    those configurations.

``random_questions``
    Sample a shared fraction of questions and evaluate every configuration on
    those questions.

Both protocols recommend the nondominated set under maximum accuracy and
minimum mean deployment cost. Full-matrix values are used only after selection
to evaluate the recommendation against the brute-force Pareto frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.single_objective.offline_selector_sim import (  # noqa: E402
    LookupTable,
    load_pickle,
    load_scope,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    front_quality_metrics,
    hypervolume_2d,
    nondominated_indices,
)


VERSIONS = ("random_configurations", "random_questions")
DEFAULT_BUDGET_FRACTIONS = tuple(i / 10 for i in range(1, 11))


@dataclass(frozen=True)
class MultiObjectiveRandomSearchResult:
    version: str
    seed: int
    budget_fraction: float
    selected_arm_indices: Tuple[int, ...]
    selected_models: Tuple[str, ...]
    completed_arm_indices: Tuple[int, ...]
    completed_models: Tuple[str, ...]
    completed_pareto_arm_indices: Tuple[int, ...]
    completed_pareto_models: Tuple[str, ...]
    sampled_arm_indices: Tuple[int, ...]
    sampled_question_ids: Tuple[int, ...]
    total_evaluations: int
    total_search_cost_usd: float
    hypervolume: float
    ground_truth_hypervolume: float
    hypervolume_regret: float
    generational_distance: float
    inverted_generational_distance: float
    true_front_recall: float
    recommendation_precision: float
    false_positive_count: int
    estimated_vectors: np.ndarray
    truth_vectors: np.ndarray
    cost_reference_usd: float


def common_question_ids(
    models: Sequence[str], datapoints: Sequence[int], table: LookupTable,
) -> Tuple[int, ...]:
    """Return questions observed for every configuration."""
    return tuple(
        int(question_id)
        for question_id in datapoints
        if all(question_id in table.get(model, {}) for model in models)
    )


def mean_raw_vectors(
    models: Sequence[str], question_ids: Sequence[int], table: LookupTable,
) -> np.ndarray:
    """Return columns ``(mean accuracy, mean deployment cost USD)``."""
    if not question_ids:
        raise ValueError("question_ids must not be empty")
    vectors = np.empty((len(models), 2), dtype=np.float64)
    for arm_index, model in enumerate(models):
        samples = table.get(model, {})
        selected = [samples[q] for q in question_ids if q in samples]
        if len(selected) != len(question_ids):
            raise ValueError(f"model {model!r} is missing a requested question")
        vectors[arm_index, 0] = float(np.mean([sample.score for sample in selected]))
        vectors[arm_index, 1] = float(np.mean([sample.cost for sample in selected]))
    return vectors


def pareto_min_cost_indices(vectors: np.ndarray) -> List[int]:
    """Return nondominated indices for accuracy up / cost down."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("vectors must have shape (n, 2)")
    maximization_vectors = np.column_stack([vectors[:, 0], -vectors[:, 1]])
    return nondominated_indices(maximization_vectors)


def normalized_truth_vectors(
    raw_vectors: np.ndarray, cost_reference_usd: float,
) -> np.ndarray:
    """Map raw accuracy/cost to two maximization objectives in ``[0, 1]``."""
    if not math.isfinite(cost_reference_usd) or cost_reference_usd <= 0.0:
        raise ValueError("cost_reference_usd must be finite and positive")
    return np.column_stack(
        [
            raw_vectors[:, 0],
            cost_reference_usd / (cost_reference_usd + raw_vectors[:, 1]),
        ]
    )


def _validate_fraction(value: float) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("budget_fraction must lie in (0, 1]")
    return fraction


def simulate_multiobjective_random_search(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    version: str,
    budget_fraction: float,
    seed: int = 42,
    evaluation_question_ids: Sequence[int] | None = None,
    complete_only: bool = False,
) -> MultiObjectiveRandomSearchResult:
    """Run one random-search protocol at one observed-cell budget."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    fraction = _validate_fraction(budget_fraction)
    questions = tuple(
        evaluation_question_ids
        if evaluation_question_ids is not None
        else common_question_ids(models, datapoints, table)
    )
    if not models or not questions:
        raise ValueError("models and the common question universe must be nonempty")

    truth_raw = mean_raw_vectors(models, questions, table)
    positive_costs = truth_raw[:, 1][truth_raw[:, 1] > 0.0]
    if positive_costs.size == 0:
        raise ValueError("at least one configuration must have positive mean cost")
    cost_reference = float(np.median(positive_costs))
    truth_normalized = normalized_truth_vectors(truth_raw, cost_reference)
    true_front = set(pareto_min_cost_indices(truth_raw))
    truth_front_points = truth_normalized[list(true_front)]
    ground_truth_hv = hypervolume_2d(truth_front_points)

    rng = np.random.default_rng(seed)
    n_arms = len(models)
    n_questions = len(questions)
    estimated = np.full((n_arms, 2), np.nan, dtype=np.float64)

    if version == "random_configurations":
        n_sampled_arms = min(n_arms, max(1, int(math.ceil(fraction * n_arms))))
        sampled_arms = tuple(
            sorted(int(i) for i in rng.permutation(n_arms)[:n_sampled_arms])
        )
        sampled_questions = questions
        estimated[list(sampled_arms)] = truth_raw[list(sampled_arms)]
        local_front = pareto_min_cost_indices(estimated[list(sampled_arms)])
        selected_arms = tuple(sampled_arms[position] for position in local_front)
    else:
        n_sampled_questions = min(
            n_questions, max(1, int(math.ceil(fraction * n_questions)))
        )
        sampled_questions = tuple(
            sorted(
                int(q)
                for q in rng.permutation(questions)[:n_sampled_questions]
            )
        )
        sampled_arms = tuple(range(n_arms))
        estimated[:] = mean_raw_vectors(models, sampled_questions, table)
        selected_arms = tuple(pareto_min_cost_indices(estimated))

    completed_arms = tuple(
        arm for arm in sampled_arms if len(sampled_questions) == len(questions)
    )
    if completed_arms:
        local_complete = pareto_min_cost_indices(estimated[list(completed_arms)])
        completed_pareto = tuple(completed_arms[i] for i in local_complete)
    else:
        completed_pareto = ()
    if complete_only:
        selected_arms = completed_pareto

    total_cost = 0.0
    for arm_index in sampled_arms:
        samples = table[models[arm_index]]
        total_cost += sum(samples[q].cost for q in sampled_questions)
    total_evaluations = len(sampled_arms) * len(sampled_questions)
    selected_points = (
        truth_normalized[list(selected_arms)]
        if selected_arms
        else np.empty((0, 2), dtype=np.float64)
    )
    quality = front_quality_metrics(
        selected_points,
        truth_front_points,
        (0.0, 0.0),
        ground_truth_hv,
    )
    recalled = len(true_front.intersection(selected_arms))
    recall = recalled / len(true_front) if true_front else 1.0
    false_positive_count = len(set(selected_arms) - true_front)
    precision = recalled / len(selected_arms) if selected_arms else 1.0

    return MultiObjectiveRandomSearchResult(
        version=version,
        seed=int(seed),
        budget_fraction=fraction,
        selected_arm_indices=selected_arms,
        selected_models=tuple(models[i] for i in selected_arms),
        completed_arm_indices=completed_arms,
        completed_models=tuple(models[i] for i in completed_arms),
        completed_pareto_arm_indices=completed_pareto,
        completed_pareto_models=tuple(models[i] for i in completed_pareto),
        sampled_arm_indices=sampled_arms,
        sampled_question_ids=sampled_questions,
        total_evaluations=total_evaluations,
        total_search_cost_usd=float(total_cost),
        hypervolume=quality.hypervolume,
        ground_truth_hypervolume=float(ground_truth_hv),
        hypervolume_regret=quality.hypervolume_regret,
        generational_distance=quality.generational_distance,
        inverted_generational_distance=quality.inverted_generational_distance,
        true_front_recall=float(recall),
        recommendation_precision=float(precision),
        false_positive_count=int(false_positive_count),
        estimated_vectors=estimated,
        truth_vectors=truth_raw,
        cost_reference_usd=cost_reference,
    )


def run_budget_sweep(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    versions: Iterable[str] = VERSIONS,
    budget_fractions: Iterable[float] = DEFAULT_BUDGET_FRACTIONS,
    seeds: Iterable[int] = (42,),
) -> List[MultiObjectiveRandomSearchResult]:
    questions = common_question_ids(models, datapoints, table)
    return [
        simulate_multiobjective_random_search(
            models,
            datapoints,
            table,
            version=version,
            budget_fraction=fraction,
            seed=seed,
            evaluation_question_ids=questions,
        )
        for version in versions
        for seed in seeds
        for fraction in budget_fractions
    ]


def write_results_csv(
    results: Sequence[MultiObjectiveRandomSearchResult], path: Path,
) -> None:
    fields = [
        "version", "seed", "budget_fraction", "total_evaluations",
        "total_search_cost_usd", "n_recommended", "hypervolume",
        "ground_truth_hypervolume", "hypervolume_regret",
        "generational_distance", "inverted_generational_distance",
        "true_front_recall",
        "recommendation_precision", "false_positive_count", "selected_models",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "version": result.version,
                    "seed": result.seed,
                    "budget_fraction": result.budget_fraction,
                    "total_evaluations": result.total_evaluations,
                    "total_search_cost_usd": result.total_search_cost_usd,
                    "n_recommended": len(result.selected_arm_indices),
                    "hypervolume": result.hypervolume,
                    "ground_truth_hypervolume": result.ground_truth_hypervolume,
                    "hypervolume_regret": result.hypervolume_regret,
                    "generational_distance": result.generational_distance,
                    "inverted_generational_distance": (
                        result.inverted_generational_distance
                    ),
                    "true_front_recall": result.true_front_recall,
                    "recommendation_precision": result.recommendation_precision,
                    "false_positive_count": result.false_positive_count,
                    "selected_models": json.dumps(result.selected_models),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pickle", help="Path to a cached lookup pickle")
    source.add_argument(
        "--scope",
        help="Path to a SCOPE benchmark directory containing matrix CSVs",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.pickle:
        models, datapoints, table = load_pickle(args.pickle)
    else:
        models, datapoints, table = load_scope(args.scope)
    results = run_budget_sweep(
        models,
        datapoints,
        table,
        seeds=range(args.base_seed, args.base_seed + args.seeds),
    )
    write_results_csv(results, Path(args.output))
    print(f"wrote {args.output} ({len(results)} runs)")


if __name__ == "__main__":
    main()
