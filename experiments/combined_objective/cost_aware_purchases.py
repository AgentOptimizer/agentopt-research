"""Post-warm purchases for the two lightweight cost-aware UCB policies."""

from collections.abc import Iterator, Sequence
import heapq
import math

import numpy as np

from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule
from experiments.combined_objective.offline_pareto_baselines import (
    LookupTable,
    _QuestionPuller,
    _algorithm_references,
)


COHV_UCB = "cohv_ucb"
SCALARIZATION_UCB = "cost_aware_scalarization_ucb"
PurchaseGroup = list[tuple[int, tuple[int, ...]]]


def _warm_puller(models, questions, table, schedule, budget_usd, batch_size):
    puller = _QuestionPuller(
        models, questions, table, seed=0,
        question_orders=[schedule.orders[arm] for arm in range(len(models))],
    )
    for arm in range(puller.n_arms):
        puller.pull(arm, batch_size)
        if puller.total_cost_usd > budget_usd + 1e-10:
            return None
    return puller


def cohv_purchases(
    models: Sequence[str],
    questions: Sequence[int],
    table: LookupTable,
    schedule: PerArmQuestionSchedule,
    budget_usd: float,
    batch_size: int,
) -> Iterator[PurchaseGroup]:
    """Yield optimistic volume per lower-bound dollar purchases through the cap."""
    puller = _warm_puller(models, questions, table, schedule, budget_usd, batch_size)
    if puller is None:
        return

    cost_reference = float(_algorithm_references(puller, (1.0, 1.0))[1])
    warm_prices = np.maximum(puller.raw_means()[:, 1], 1e-12)
    price_floor = float(warm_prices.min())
    price_scale = float(warm_prices.max())
    log_horizon = math.log(math.ceil(budget_usd / price_floor) + 1)
    max_count = len(questions)
    radii = [0.0] + (0.01 * np.sqrt(2 * log_horizon / np.arange(1, max_count + 1))).tolist()

    def priority(arm):
        count = int(puller.n_pulls[arm])
        size = min(batch_size, puller.remaining_count(arm))
        quality_sum, cost_sum = puller.sums[arm]
        quality = float(quality_sum) / count
        cost = float(cost_sum) / count
        radius = radii[count]
        cost_desirability = cost_reference / (cost_reference + cost)
        upper_quality = min(1.0, quality + radius)
        upper_cost_desirability = min(1.0, cost_desirability + radius)
        lower_price = max(price_floor / 2, cost - radius * price_scale)
        score = (upper_quality * upper_cost_desirability) / (lower_price * size)
        return -score, count, arm, size

    # Constants are fixed at warm-up, so only the purchased arm needs a new score.
    candidates = [priority(arm) for arm in range(puller.n_arms) if puller.has_remaining(arm)]
    heapq.heapify(candidates)
    while candidates and puller.total_cost_usd < budget_usd:
        _, _, arm, size = candidates[0]
        question_ids = tuple(reversed(puller.remaining[arm][-size:]))
        puller.pull(arm, size)
        if puller.has_remaining(arm):
            heapq.heapreplace(candidates, priority(arm))
        else:
            heapq.heappop(candidates)
        yield [(arm, question_ids)]


def scalarization_ucb_purchases(
    models: Sequence[str],
    questions: Sequence[int],
    table: LookupTable,
    schedule: PerArmQuestionSchedule,
    budget_usd: float,
    batch_size: int,
) -> Iterator[PurchaseGroup]:
    """Yield rotating weighted UCB per mean batch dollar purchases through the cap."""
    puller = _warm_puller(models, questions, table, schedule, budget_usd, batch_size)
    if puller is None:
        return

    references = _algorithm_references(puller, (1.0, 1.0))
    step = 0
    while puller.has_remaining() and puller.total_cost_usd < budget_usd:
        candidates = np.array([
            arm for arm in range(puller.n_arms) if puller.has_remaining(arm)
        ])
        sizes = np.array([
            min(batch_size, puller.remaining_count(arm)) for arm in candidates
        ])
        raw = puller.raw_means()
        means = puller.maximization_means(references)
        prices = np.maximum(raw[candidates, 1], 1e-12) * sizes
        counts = puller.n_pulls[candidates]
        radius = 0.5 * np.sqrt(math.log1p(puller.total_evaluations) / counts)
        upper = np.minimum(1, means[candidates] + radius[:, None])
        cost_weight = (step % 11) / 10
        scores = (upper @ np.array([1 - cost_weight, cost_weight])) / prices
        position = np.lexsort((candidates, counts, -scores))[0]
        arm = int(candidates[position])
        size = int(sizes[position])
        question_ids = tuple(reversed(puller.remaining[arm][-size:]))
        puller.pull(arm, size)
        yield [(arm, question_ids)]
        step += 1
