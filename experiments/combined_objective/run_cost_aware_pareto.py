"""Replay cost-aware Pareto baselines on the committed quality/USD matrices.

The budget is a fraction of exhaustive USD spend, including a shared warm
start. Policies observe each selected batch's charge after selection. Scoring
uses the last complete batch at or below budget, and the output separately
records the charge for a batch that crossed it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.combined_objective.offline_pareto_baselines import (
    LookupTable,
    _QuestionPuller,
    _algorithm_references,
    _checkpoint,
    empirical_raw_pareto_arms,
)
from experiments.combined_objective.offline_multiobjective_random_search import (
    common_question_ids,
    mean_raw_vectors,
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_radial_gittins import (
    hypervolume_2d,
    mean_cost_metric_vectors,
)
from experiments.combined_objective.run_two_direction_ablation import DATASETS, load_benchmark
from agentopt.model_selection.pareto_identification import pairwise_M
from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule

COST_AWARE_QNEHVI = "cost_aware_qnehvi"
COPSI = "copsi"
COHV_UCB = "cohv_ucb"
METHODS = (COST_AWARE_QNEHVI, COPSI, COHV_UCB)


def _copsi_classify(means: np.ndarray) -> tuple[int, bool]:
    """Algorithm 2's largest classification gap, favoring rejection on ties."""
    margins = pairwise_M(means)
    np.fill_diagonal(margins, np.inf)
    dominated_gap = -margins.min(axis=1)
    pareto = dominated_gap <= 0.0  # the paper requires strict improvement in every coordinate
    rivals = np.minimum(
        margins, np.maximum(margins.T, 0.0) + np.maximum(dominated_gap, 0.0)
    )
    np.fill_diagonal(rivals, np.inf)
    gaps = np.where(pareto, rivals.min(axis=1), dominated_gap)
    tied = np.flatnonzero(np.isclose(gaps, gaps.max(), rtol=1e-10, atol=1e-12))
    rejected = tied[~pareto[tied]]
    index = int(rejected[0] if len(rejected) else tied[0])
    return index, bool(pareto[index])


def replay(
    models: Sequence[str], datapoints: Sequence[int], table: LookupTable, *,
    method: str, seed: int, budget_fraction: float,
    batch_size: int = 4, qnehvi_mc_samples: int = 64, qnehvi_refit_every: int = 8,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if not 0 < budget_fraction <= 1 or min(batch_size, qnehvi_mc_samples, qnehvi_refit_every) < 1:
        raise ValueError("invalid budget fraction, batch size, MC count, or refit interval")
    questions = common_question_ids(models, datapoints, table)
    truth = mean_raw_vectors(models, questions, table)
    exhaustive = math.fsum(float(table[model][q].cost) for model in models for q in questions)
    if not math.isfinite(exhaustive) or exhaustive <= 0:
        raise ValueError("exhaustive search cost must be positive and finite")
    cap = budget_fraction * exhaustive
    reference = (0.0, 0.0)
    truth_front = pareto_min_cost_indices(truth)
    endpoint = max(truth_front, key=lambda i: (truth[i, 0], -truth[i, 1]))
    truth_normalized, eval_scale = mean_cost_metric_vectors(truth)
    front_vectors = truth_normalized[truth_front]
    oracle_hv = hypervolume_2d(front_vectors, reference)
    schedule = PerArmQuestionSchedule.create_from_available(
        dict.fromkeys(range(len(models)), questions), seed=seed,
        warm_start_batch_size=min(batch_size, len(questions)),
        question_order="independent", warm_start_question_order="independent",
    )
    puller = _QuestionPuller(
        models, questions, table, seed=seed,
        question_orders=[schedule.orders[i] for i in range(len(models))],
    )
    checkpoints = []
    started = time.perf_counter()

    def record(event, selected=None):
        if puller.total_cost_usd <= cap + 1e-10:
            selection = empirical_raw_pareto_arms(puller) if selected is None else selected
            point = _checkpoint(
                puller, selected=selection, truth_normalized=truth_normalized,
                true_front_vectors=front_vectors, ground_truth_hv=oracle_hv,
                bruteforce_cost=exhaustive, event=event, reference=reference,
            )
            if checkpoints and checkpoints[-1].cumulative_evaluations == point.cumulative_evaluations:
                checkpoints[-1] = point
            else:
                checkpoints.append(point)

    def pull(arm, count, event, selected=None):
        puller.pull(arm, count)
        if puller.total_cost_usd > cap + 1e-10:
            return False
        record(event, None if selected is None else selected())
        return True

    stop = "budget_crossed"
    params = {
        "batch_size": batch_size,
        "metric_space": "offline_common_affine_mean_usd",
        "metric_cost_reference_usd": eval_scale,
        "question_order_rng_scheme": "spawned_per_arm_full_orders",
    }
    for arm in range(len(models)):
        puller.pull(arm, batch_size)
        if puller.total_cost_usd > cap + 1e-10:
            break
    else:
        record("warm_start")
        references = _algorithm_references(puller, (1.0, 1.0))
        warm_prices = np.maximum(puller.raw_means()[:, 1], 1e-12)
        params["deployment_cost_scale_usd"] = float(references[1])
        if method == COPSI:
            stop = _run_copsi(puller, warm_prices, cap, batch_size, pull, record, params)
        else:
            stop = _run_online(
                puller, method, references, warm_prices, cap, batch_size, pull,
                seed, qnehvi_mc_samples, qnehvi_refit_every, params,
            )

    points = []
    truth_set = set(truth_front)
    for point in checkpoints:
        selected = point.selected_arm_indices
        matched = len(truth_set.intersection(selected))
        points.append({
            "event": point.event,
            "evaluations": point.cumulative_evaluations,
            "cost_usd": point.cumulative_search_cost_usd,
            "cost_fraction": point.budget_fraction,
            "selected_arm_indices": selected,
            "pareto_recall": matched / len(truth_front),
            "pareto_precision": matched / len(selected) if selected else 0.0,
            "pareto_f1": 2 * matched / (len(truth_front) + len(selected)),
            "accuracy_endpoint_found": endpoint in selected,
            "hypervolume_regret": point.hypervolume_regret,
            "relative_hv_regret": point.hypervolume_regret / oracle_hv if oracle_hv > 0 else 0.0,
            "generational_distance": point.generational_distance,
            "inverted_generational_distance": point.inverted_generational_distance,
        })
    return {
        "method": method, "seed": seed, "budget_fraction": budget_fraction,
        "budget_usd": cap, "bruteforce_search_cost_usd": exhaustive, "params": params,
        "n_models": len(models), "n_questions": len(questions), "models": models,
        "truth_raw_vectors": truth.tolist(), "truth_pareto_arm_indices": truth_front,
        "ground_truth_hypervolume": oracle_hv,
        "accuracy_endpoint": int(endpoint), "stop_reason": stop,
        "charged_cost_usd": puller.total_cost_usd, "charged_evaluations": puller.total_evaluations,
        "wall_seconds": time.perf_counter() - started,
        "final": points[-1] if points else None, "checkpoints": points,
    }


def _run_copsi(puller, prices, cap, batch_size, pull, record, params):
    active = list(range(puller.n_arms))
    accepted = []
    cost_floor = prices.min() / prices.max()
    schedule_factor = np.sum(1.0 / (1.0 + np.arange(len(active)) * cost_floor))
    params["schedule_factor"] = float(schedule_factor)
    params["cost_estimator"] = "frozen_warm_mean"

    def recommendation():
        means = puller.raw_means()[active]
        return sorted(accepted + [active[i] for i in pareto_min_cost_indices(means)])

    references = (1.0, params["deployment_cost_scale_usd"])
    while len(active) > 1:
        target = min(len(puller.questions), math.floor(cap / (schedule_factor * prices[active].sum())))
        # Interleave active arms up to the phase target. The completed phase
        # has the same observations regardless of this ordering.
        for start in range(int(puller.n_pulls[active].min()), target, batch_size):
            for arm in active:
                count = min(start + batch_size, target) - int(puller.n_pulls[arm])
                if count > 0 and not pull(arm, count, "copsi_pull", recommendation):
                    return "budget_crossed"
        index, pareto = _copsi_classify(puller.maximization_means(references)[active])
        classified = active.pop(index)
        if pareto:
            accepted.append(classified)
        record("copsi_classify", recommendation())
    record("copsi_terminal", sorted(accepted + active))
    return "identification_complete"


def _run_online(puller, method, references, warm_prices, cap, batch_size, pull,
                seed, mc_samples, refit_every, params):
    price_floor = float(warm_prices.min())
    price_scale = float(warm_prices.max())
    log_horizon = math.log(math.ceil(cap / price_floor) + 1)
    if method == COHV_UCB:
        params.update({"radius_scale": 0.01, "price_floor_usd": price_floor,
                       "price_scale_usd": price_scale})
    if method == COST_AWARE_QNEHVI:
        from agentopt.model_selection.qnehvi import encode_configuration_features, select_qnehvi_index

        features, cat_dims = encode_configuration_features(puller.models)
        params.update({"mc_samples": mc_samples, "refit_every_batches": refit_every,
                       "incumbent": "fully_evaluated_arms"})
    sticky = None
    step = 0
    while puller.has_remaining() and puller.total_cost_usd < cap:
        candidates = np.array([i for i in range(puller.n_arms) if puller.has_remaining(i)])
        sizes = np.array([min(batch_size, puller.remaining_count(i)) for i in candidates])
        raw = puller.raw_means()
        means = puller.maximization_means(references)
        prices = np.maximum(raw[candidates, 1], 1e-12) * sizes
        counts = puller.n_pulls[candidates]
        if method == COST_AWARE_QNEHVI:
            if sticky not in candidates or step % refit_every == 0:
                completed = puller.n_pulls == len(puller.questions)
                local = select_qnehvi_index(
                    features, means, features[candidates], categorical_dims=cat_dims,
                    baseline_features=features[completed], candidate_costs=prices,
                    mc_samples=mc_samples, seed=seed + puller.total_evaluations,
                )
                sticky = int(candidates[local])
            arm = sticky
        else:
            radius = 0.01 * np.sqrt(2 * log_horizon / counts)
            upper = np.minimum(1, means[candidates] + radius[:, None])
            lower_prices = np.maximum(price_floor / 2, raw[candidates, 1] - radius * price_scale)
            scores = upper.prod(axis=1) / (lower_prices * sizes)
            arm = int(candidates[np.lexsort((candidates, counts, -scores))[0]])
        if not pull(arm, batch_size, method):
            return "budget_crossed"
        step += 1
    return "all_arms_exhausted" if not puller.has_remaining() else "budget_reached"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", choices=tuple(DATASETS), default="mathqa",
                        help="Committed benchmark matrix")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                        help="Methods to replay; cost_aware_qnehvi requires the bayesian extra")
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                        help="One or more question-order seeds")
    parser.add_argument("--budget-fraction", type=float, default=0.10,
                        help="Fraction of exhaustive evaluation cost in USD")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--qnehvi-mc-samples", type=int, default=64)
    parser.add_argument("--qnehvi-refit-every", type=int, default=8)
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory for one JSON result per benchmark, method, and seed")
    args = parser.parse_args()
    jobs = [
        (seed, method, args.outdir / f"{args.benchmark}_{method}_seed{seed}.json")
        for seed in args.seed
        for method in args.methods
    ]
    if len({output for _, _, output in jobs}) != len(jobs):
        parser.error("each seed and method must be listed only once")
    for _, _, output in jobs:
        if output.exists():
            parser.error(f"result already exists: {output}. Choose a fresh --outdir.")

    if COST_AWARE_QNEHVI in args.methods:
        from agentopt.model_selection.qnehvi import _require_botorch

        try:
            _require_botorch()
        except ImportError as error:
            parser.error(str(error))
        import torch

        torch.set_num_threads(1)
    models, questions, table, hashes = load_benchmark(args.benchmark)
    for seed, method, output in jobs:
        result = replay(
            models, questions, table, method=method, seed=seed,
            budget_fraction=args.budget_fraction, batch_size=args.batch_size,
            qnehvi_mc_samples=args.qnehvi_mc_samples, qnehvi_refit_every=args.qnehvi_refit_every,
        )
        result.update(benchmark=args.benchmark, input_sha256=hashes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"{method} seed={seed} stop={result['stop_reason']} "
              f"final={result['final']} wall={result['wall_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
