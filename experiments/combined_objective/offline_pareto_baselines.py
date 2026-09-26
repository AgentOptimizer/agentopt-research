#!/usr/bin/env python3
"""Offline Pareto-set baselines: EGE-SH/SR, APE-k sampling, and qNEHVI.

These methods identify a set of configurations under maximize-accuracy /
minimize-deployment-cost, optionally including mean latency as a third
objective.  They do not use the radial-Gittins direction
scheduler.  Each pull reads one batch of frozen lookup-table questions for
one configuration.  There is no endogenous stop: a run continues until the
cell budget is exhausted (or every remaining question has been used).

EGE-SH / EGE-SR (Kone et al., AISTATS 2024) are fixed-budget elimination
rules.  APE-k (Kone et al., NeurIPS 2023) is a LUCB-style sampler; its
fixed-confidence ``|OPT_ε1| ≥ k`` stop is omitted, so the sampling rule
simply runs for the full budget.  qNEHVI is sequential noisy expected
hypervolume improvement over categorical configuration features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from experiments.single_objective import offline_selector_sim as _offline_sim
except ImportError:
    from single_objective import offline_selector_sim as _offline_sim  # type: ignore[no-redef]

sys.modules.setdefault("offline_selector_sim_v2", _offline_sim)
LookupTable = _offline_sim.LookupTable
_require_data_path = _offline_sim._require_data_path
load_pickle = _offline_sim.load_pickle

from agentopt.model_selection.pareto_identification import (  # noqa: E402
    APE_K,
    EGE_SH,
    EGE_SR,
    ape_select_arm,
    ege_keep_count,
    ege_select_survivors,
    ege_sequential_halving_pulls,
    ege_sequential_halving_rounds,
    ege_successive_rejects_schedule,
)
from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
    normalized_truth_vectors,
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    _sample_values,
    generational_distance,
    hypervolume_2d,
    inverted_generational_distance,
)
from experiments.combined_objective.three_objective_metrics import (  # noqa: E402
    evaluation_space as evaluation_space_3d,
    front_distance as front_distance_3d,
    hypervolume_3d,
    raw_pareto_indices as raw_pareto_indices_3d,
    raw_truth_vectors as raw_truth_vectors_3d,
)


QNEHVI = "qnehvi"
METHODS = (EGE_SH, EGE_SR, APE_K, QNEHVI)
DEFAULT_BATCH_SIZE = 4
DEFAULT_RECOMMENDATION_CHECKPOINT_INTERVAL = 10
DEFAULT_OBJECTIVES = ("Q", "D")
THREE_OBJECTIVES = ("Q", "L", "D")


@dataclass(frozen=True)
class BaselineCheckpoint:
    cumulative_evaluations: int
    cumulative_search_cost_usd: float
    budget_fraction: float
    selected_arm_indices: Tuple[int, ...]
    selected_models: Tuple[str, ...]
    hypervolume: float
    hypervolume_regret: float
    generational_distance: float
    inverted_generational_distance: float
    event: str


@dataclass
class ParetoBaselineResult:
    selector: str
    seed: int
    params: Dict[str, object]
    selected_arm_indices: Tuple[int, ...]
    selected_models: Tuple[str, ...]
    completed_arm_indices: Tuple[int, ...]
    completed_models: Tuple[str, ...]
    completed_pareto_arm_indices: Tuple[int, ...]
    completed_pareto_models: Tuple[str, ...]
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
    stop_reason: str
    cost_reference_usd: float
    policy_wall_time_seconds: float
    recommendation_trajectory: List[BaselineCheckpoint] = field(default_factory=list)
    estimated_raw_vectors: Optional[np.ndarray] = None
    truth_raw_vectors: Optional[np.ndarray] = None
    truth_vectors: Optional[np.ndarray] = None


class _QuestionPuller:
    """Consume unused lookup-table questions for one configuration at a time."""

    def __init__(
        self,
        models: Sequence[str],
        questions: Sequence[int],
        table: LookupTable,
        *,
        seed: int,
        objectives: Sequence[str] = DEFAULT_OBJECTIVES,
    ) -> None:
        self.models = tuple(models)
        self.questions = tuple(int(q) for q in questions)
        self.table = table
        self.objectives = tuple(objectives)
        rng = np.random.default_rng(seed)
        self.remaining: List[List[int]] = [
            [int(q) for q in rng.permutation(self.questions)]
            for _ in self.models
        ]
        n_arms = len(self.models)
        self.n_pulls = np.zeros(n_arms, dtype=np.int64)
        self.sums = np.zeros((n_arms, len(self.objectives)), dtype=np.float64)
        self.total_evaluations = 0
        self.total_cost_usd = 0.0

    @property
    def n_arms(self) -> int:
        return len(self.models)

    def remaining_count(self, arm: int) -> int:
        return len(self.remaining[int(arm)])

    def has_remaining(self, arm: Optional[int] = None) -> bool:
        if arm is None:
            return any(self.remaining)
        return bool(self.remaining[int(arm)])

    def raw_means(self) -> np.ndarray:
        means = np.full((self.n_arms, len(self.objectives)), np.nan, dtype=np.float64)
        pulled = self.n_pulls > 0
        means[pulled] = self.sums[pulled] / self.n_pulls[pulled, None]
        return means

    def maximization_means(self, references: Sequence[float]) -> np.ndarray:
        raw = self.raw_means()
        if not np.all(self.n_pulls > 0):
            raise ValueError("maximization_means requires at least one pull per arm")
        scales = np.asarray(references, dtype=np.float64)
        if scales.shape != (len(self.objectives),) or np.any(scales <= 0.0):
            raise ValueError("positive algorithm references must match objectives")
        if self.objectives == DEFAULT_OBJECTIVES:
            # Preserve the historical two-axis acquisition geometry.
            return normalized_truth_vectors(raw, float(scales[1]))
        values = raw.copy()
        for column, objective in enumerate(self.objectives):
            if objective != "Q":
                # Affine transformation preserves the ordering of mean L/D.
                values[:, column] = 1.0 - raw[:, column] / scales[column]
        return values

    def pull(self, arm: int, n_questions: int) -> int:
        arm_index = int(arm)
        take = min(int(n_questions), self.remaining_count(arm_index))
        if take <= 0:
            return 0
        model = self.models[arm_index]
        samples = self.table[model]
        for _ in range(take):
            question_id = self.remaining[arm_index].pop()
            score, cost, latency = _sample_values(samples[question_id])
            observed = {"Q": score, "L": latency, "D": cost}
            self.sums[arm_index] += [observed[name] for name in self.objectives]
            self.total_cost_usd += cost
        self.n_pulls[arm_index] += take
        self.total_evaluations += take
        return take


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _validate_fraction(value: float) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("observation_budget_fraction must lie in (0, 1]")
    return fraction


def _score_selection(
    selected: Sequence[int],
    truth_normalized: np.ndarray,
    true_front_vectors: np.ndarray,
    reference: Sequence[float],
) -> tuple[float, float, float]:
    dimensions = truth_normalized.shape[1]
    obtained = (truth_normalized[list(selected)] if selected
                else np.empty((0, dimensions), dtype=np.float64))
    if dimensions == 2:
        hv = hypervolume_2d(obtained, reference) if selected else 0.0
        gd = generational_distance(obtained, true_front_vectors)
        igd = inverted_generational_distance(obtained, true_front_vectors)
    elif dimensions == 3:
        hv = hypervolume_3d(obtained, reference)
        gd = front_distance_3d(obtained, true_front_vectors)
        igd = front_distance_3d(true_front_vectors, obtained)
    else:
        raise ValueError("Pareto scoring requires two or three objectives")
    return float(hv), float(gd), float(igd)


def _checkpoint(
    puller: _QuestionPuller,
    *,
    selected: Sequence[int],
    truth_normalized: np.ndarray,
    true_front_vectors: np.ndarray,
    ground_truth_hv: float,
    bruteforce_cost: float,
    event: str,
    reference: Sequence[float],
) -> BaselineCheckpoint:
    arms = tuple(int(i) for i in selected)
    selected_hv, selected_gd, selected_igd = _score_selection(
        arms, truth_normalized, true_front_vectors, reference,
    )
    return BaselineCheckpoint(
        cumulative_evaluations=int(puller.total_evaluations),
        cumulative_search_cost_usd=float(puller.total_cost_usd),
        budget_fraction=float(puller.total_cost_usd) / float(bruteforce_cost),
        selected_arm_indices=arms,
        selected_models=tuple(puller.models[i] for i in arms),
        hypervolume=float(selected_hv),
        hypervolume_regret=float(max(0.0, ground_truth_hv - selected_hv)),
        generational_distance=selected_gd,
        inverted_generational_distance=selected_igd,
        event=event,
    )


def _raw_pareto_indices(raw: np.ndarray, objectives: Sequence[str]) -> Sequence[int]:
    if tuple(objectives) == DEFAULT_OBJECTIVES:
        return pareto_min_cost_indices(raw)
    if tuple(objectives) == THREE_OBJECTIVES:
        return raw_pareto_indices_3d(raw)
    raise ValueError("unsupported objective order")


def empirical_raw_pareto_arms(puller: _QuestionPuller) -> Tuple[int, ...]:
    """Recommend the observed raw Pareto set of sampled arms."""
    raw = puller.raw_means()
    sampled = [i for i in range(puller.n_arms) if puller.n_pulls[i] > 0]
    if not sampled:
        return ()
    local = _raw_pareto_indices(raw[sampled], puller.objectives)
    return tuple(sampled[i] for i in local)


def completed_raw_pareto_arms(puller: _QuestionPuller) -> Tuple[int, ...]:
    """Return the empirical Pareto set restricted to fully evaluated arms."""
    completed = [
        i for i in range(puller.n_arms)
        if int(puller.n_pulls[i]) == len(puller.questions)
    ]
    if not completed:
        return ()
    local = _raw_pareto_indices(puller.raw_means()[completed], puller.objectives)
    return tuple(completed[i] for i in local)


def ege_recommended_arms(
    accepted: Sequence[int],
    active: Sequence[int],
) -> Tuple[int, ...]:
    return tuple(sorted(set(int(i) for i in accepted) | set(int(i) for i in active)))


def _warm_start_all(
    puller: _QuestionPuller,
    *,
    batch_size: int,
    cell_budget: int,
) -> int:
    used = 0
    for arm in range(puller.n_arms):
        remaining_budget = cell_budget - puller.total_evaluations
        if remaining_budget <= 0:
            break
        used += puller.pull(arm, min(batch_size, remaining_budget))
    return used


def _algorithm_references(
    puller: _QuestionPuller,
    fallback: Sequence[float],
) -> np.ndarray:
    """Scale minimization axes using sampled means only; never oracle means."""
    references = np.asarray(fallback, dtype=np.float64).copy()
    observed = puller.raw_means()
    for column, objective in enumerate(puller.objectives):
        if objective == "Q":
            references[column] = 1.0
            continue
        sampled = observed[:, column][puller.n_pulls > 0]
        positive = sampled[np.isfinite(sampled) & (sampled > 0.0)]
        if positive.size:
            references[column] = float(np.median(positive))
    return references


def _run_ege(
    puller: _QuestionPuller,
    *,
    variant: str,
    batch_size: int,
    cell_budget: int,
    fallback_references: Sequence[float],
    record: Callable[[str, Optional[Sequence[int]]], None],
) -> Tuple[Tuple[int, ...], str, np.ndarray]:
    n_arms = puller.n_arms
    active: Tuple[int, ...] = tuple(range(n_arms))
    accepted: List[int] = []
    references = np.asarray(fallback_references, dtype=np.float64).copy()
    sr_schedule = (
        ege_successive_rejects_schedule(n_arms, cell_budget)
        if variant == EGE_SR
        else ()
    )
    n_rounds = (
        len(sr_schedule) if variant == EGE_SR else ege_sequential_halving_rounds(n_arms)
    )
    for round_index in range(n_rounds):
        if puller.total_evaluations >= cell_budget or not active:
            break
        if variant == EGE_SR:
            per_arm = sr_schedule[round_index]
        else:
            per_arm = ege_sequential_halving_pulls(len(active), n_arms, cell_budget)
        # A round with a floored-zero allocation still needs one sample before
        # gaps can be computed, matching the EGE invariant that active arms
        # are comparable.
        per_arm = max(per_arm, 1)
        for arm in active:
            remaining_for_arm = per_arm
            while remaining_for_arm > 0 and puller.has_remaining(arm):
                if puller.total_evaluations >= cell_budget:
                    break
                take = min(
                    batch_size,
                    remaining_for_arm,
                    cell_budget - puller.total_evaluations,
                    puller.remaining_count(arm),
                )
                pulled = puller.pull(arm, take)
                if pulled <= 0:
                    break
                remaining_for_arm -= pulled
                record("ege_pull", None)
        if not np.all(puller.n_pulls[list(active)] > 0):
            break
        references = _algorithm_references(puller, fallback_references)
        if len(active) <= 1:
            break
        n_keep = ege_keep_count(len(active), variant)
        if n_keep >= len(active):
            continue
        means = puller.maximization_means(references)
        survivors, newly_accepted, _rejected = ege_select_survivors(
            means, active, n_keep
        )
        accepted.extend(newly_accepted)
        active = survivors
        record("ege_eliminate", None)

    # Leftover budget: keep sampling surviving / accepted arms, then anyone else.
    leftover_order = list(ege_recommended_arms(accepted, active)) or list(range(n_arms))
    cursor = 0
    idle_passes = 0
    while puller.total_evaluations < cell_budget and puller.has_remaining():
        arm = leftover_order[cursor % len(leftover_order)]
        cursor += 1
        if not puller.has_remaining(arm):
            idle_passes += 1
            if idle_passes >= len(leftover_order):
                leftover_order = [i for i in range(n_arms) if puller.has_remaining(i)]
                cursor = 0
                idle_passes = 0
                if not leftover_order:
                    break
            continue
        idle_passes = 0
        puller.pull(
            arm,
            min(batch_size, cell_budget - puller.total_evaluations, puller.remaining_count(arm)),
        )
        record("ege_leftover", None)

    recommended = empirical_raw_pareto_arms(puller)
    stop = (
        "question_budget"
        if puller.total_evaluations >= cell_budget
        else "all_arms_exhausted"
    )
    return recommended, stop, references


def _run_ape(
    puller: _QuestionPuller,
    *,
    batch_size: int,
    cell_budget: int,
    fallback_references: Sequence[float],
    epsilon1: float,
    delta: float,
    k1: float,
    record: Callable[[str, Optional[Sequence[int]]], None],
) -> Tuple[Tuple[int, ...], str, np.ndarray]:
    _warm_start_all(puller, batch_size=1, cell_budget=cell_budget)
    references = _algorithm_references(puller, fallback_references)
    record("ape_warm_start", None)
    while puller.total_evaluations < cell_budget and puller.has_remaining():
        means = puller.maximization_means(references)
        arm = ape_select_arm(
            means,
            puller.n_pulls,
            epsilon1=epsilon1,
            delta=delta,
            k1=k1,
        )
        if not puller.has_remaining(arm):
            remaining_arms = [i for i in range(puller.n_arms) if puller.has_remaining(i)]
            if not remaining_arms:
                break
            arm = min(remaining_arms, key=lambda i: (int(puller.n_pulls[i]), i))
        puller.pull(
            arm,
            min(batch_size, cell_budget - puller.total_evaluations, puller.remaining_count(arm)),
        )
        record("ape_pull", None)
    stop = (
        "question_budget"
        if puller.total_evaluations >= cell_budget
        else "all_arms_exhausted"
    )
    return empirical_raw_pareto_arms(puller), stop, references


def _run_qnehvi(
    puller: _QuestionPuller,
    *,
    batch_size: int,
    cell_budget: int,
    fallback_references: Sequence[float],
    features: np.ndarray,
    categorical_dims: Sequence[int],
    reference_point: Sequence[float],
    mc_samples: int,
    refit_every: int,
    candidate_batch_size: Optional[int],
    seed: int,
    record: Callable[[str, Optional[Sequence[int]]], None],
) -> Tuple[Tuple[int, ...], str, np.ndarray]:
    from agentopt.model_selection.qnehvi import select_qnehvi_index

    _warm_start_all(puller, batch_size=batch_size, cell_budget=cell_budget)
    references = _algorithm_references(puller, fallback_references)
    record("qnehvi_warm_start", None)
    sticky_arm: Optional[int] = None
    steps_since_refit = refit_every
    while puller.total_evaluations < cell_budget and puller.has_remaining():
        remaining_arms = [i for i in range(puller.n_arms) if puller.has_remaining(i)]
        if not remaining_arms:
            break
        steps_since_refit += 1
        need_refit = (
            sticky_arm is None
            or not puller.has_remaining(sticky_arm)
            or steps_since_refit >= refit_every
        )
        if need_refit:
            train_y = puller.maximization_means(references)
            try:
                local = select_qnehvi_index(
                    features,
                    train_y,
                    features[remaining_arms],
                    categorical_dims=categorical_dims,
                    reference_point=reference_point,
                    mc_samples=mc_samples,
                    seed=seed + puller.total_evaluations,
                    candidate_batch_size=candidate_batch_size,
                )
                arm = remaining_arms[local]
            except ImportError:
                raise
            except Exception as error:
                if puller.objectives == THREE_OBJECTIVES:
                    raise RuntimeError("three-objective qNEHVI acquisition failed") from error
                arm = min(remaining_arms, key=lambda i: (int(puller.n_pulls[i]), i))
            sticky_arm = arm
            steps_since_refit = 0
        else:
            arm = int(sticky_arm)
        puller.pull(
            arm,
            min(batch_size, cell_budget - puller.total_evaluations, puller.remaining_count(arm)),
        )
        record("qnehvi_pull", None)
    stop = (
        "question_budget"
        if puller.total_evaluations >= cell_budget
        else "all_arms_exhausted"
    )
    return empirical_raw_pareto_arms(puller), stop, references


def simulate_pareto_baseline(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    method: str,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    observation_budget_fraction: float = 1.0,
    record_recommendation_trajectory: bool = True,
    recommendation_checkpoint_interval: int = (
        DEFAULT_RECOMMENDATION_CHECKPOINT_INTERVAL
    ),
    epsilon1: float = 0.0,
    ape_delta: float = 0.1,
    ape_k1: float = 1.0,
    ape_k: int = 3,
    qnehvi_mc_samples: int = 64,
    qnehvi_refit_every: int = 8,
    qnehvi_candidate_batch_size: Optional[int] = 64,
    reference_point: Optional[Sequence[float]] = None,
    evaluation_question_ids: Optional[Sequence[int]] = None,
    complete_only: bool = False,
    objectives: Sequence[str] = DEFAULT_OBJECTIVES,
) -> ParetoBaselineResult:
    """Run a Pareto baseline on Q/D or Q/L/D observations.

    The default retains the historical two-objective protocol.  The explicit
    three-objective protocol requires a complete aligned question matrix.
    Recommendation diagnostics are retained every
    ``recommendation_checkpoint_interval`` ordinary policy batches; warm
    start, elimination, and terminal events are always retained.  The
    interval changes trajectory resolution only, not acquisition decisions.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    objective_order = tuple(objectives)
    if objective_order not in (DEFAULT_OBJECTIVES, THREE_OBJECTIVES):
        raise ValueError("objectives must be ('Q', 'D') or ('Q', 'L', 'D')")
    batch_size = _positive_int(batch_size, "batch_size")
    recommendation_checkpoint_interval = _positive_int(
        recommendation_checkpoint_interval,
        "recommendation_checkpoint_interval",
    )
    fraction = _validate_fraction(observation_budget_fraction)
    questions = tuple(
        evaluation_question_ids
        if evaluation_question_ids is not None
        else (datapoints if objective_order == THREE_OBJECTIVES
              else common_question_ids(models, datapoints, table))
    )
    if not models or not questions:
        raise ValueError("models and the common question universe must be nonempty")
    if objective_order == THREE_OBJECTIVES:
        if len(set(questions)) != len(questions):
            raise ValueError("three-objective question IDs must be unique")
        for model in models:
            if model not in table or any(q not in table[model] for q in questions):
                raise ValueError("three-objective baselines require a complete aligned matrix")

    reference = (tuple(0.0 for _ in objective_order) if reference_point is None
                 else tuple(float(x) for x in reference_point))
    if len(reference) != len(objective_order) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_point must match the objective dimension and be finite")
    if objective_order == THREE_OBJECTIVES:
        truth_raw = raw_truth_vectors_3d(models, questions, table)
        (truth_normalized, true_front_vectors, eval_latency_reference,
         eval_cost_reference, _) = evaluation_space_3d(truth_raw)
        true_front = set(raw_pareto_indices_3d(truth_raw))
        ground_truth_hv = hypervolume_3d(true_front_vectors, reference)
    else:
        truth_raw = mean_raw_vectors(models, questions, table)
        positive_costs = truth_raw[:, 1][truth_raw[:, 1] > 0.0]
        if positive_costs.size == 0:
            raise ValueError("at least one configuration must have positive mean cost")
        eval_cost_reference = float(np.median(positive_costs))
        eval_latency_reference = None
        truth_normalized = normalized_truth_vectors(truth_raw, eval_cost_reference)
        true_front = set(pareto_min_cost_indices(truth_raw))
        true_front_vectors = truth_normalized[list(true_front)]
        ground_truth_hv = hypervolume_2d(true_front_vectors, reference)
    bruteforce_cost = 0.0
    for model in models:
        samples = table[model]
        bruteforce_cost += sum(float(samples[q].cost) for q in questions)
    if not math.isfinite(bruteforce_cost) or bruteforce_cost <= 0.0:
        raise ValueError("brute-force search cost must be finite and positive")

    cell_budget = max(1, int(math.floor(fraction * len(models) * len(questions))))
    puller = _QuestionPuller(models, questions, table, seed=seed, objectives=objective_order)
    fallback_references = np.ones(len(objective_order), dtype=np.float64)
    trajectory: List[BaselineCheckpoint] = []
    ordinary_checkpoint_events = 0
    skipped_checkpoint_events = 0

    def record(event: str, selected: Optional[Sequence[int]]) -> None:
        nonlocal ordinary_checkpoint_events, skipped_checkpoint_events
        if not record_recommendation_trajectory:
            return
        if event in {"ege_pull", "ege_leftover", "ape_pull", "qnehvi_pull"}:
            ordinary_checkpoint_events += 1
            if ordinary_checkpoint_events % recommendation_checkpoint_interval != 0:
                skipped_checkpoint_events += 1
                return
        if trajectory and trajectory[-1].cumulative_evaluations == puller.total_evaluations:
            return
        if selected is None:
            selected = empirical_raw_pareto_arms(puller)
        trajectory.append(
            _checkpoint(
                puller,
                selected=selected,
                truth_normalized=truth_normalized,
                true_front_vectors=true_front_vectors,
                ground_truth_hv=ground_truth_hv,
                bruteforce_cost=bruteforce_cost,
                event=event,
                reference=reference,
            )
        )

    wall_start = time.perf_counter()
    if method in (EGE_SH, EGE_SR):
        selected, stop_reason, algo_references = _run_ege(
            puller,
            variant=method,
            batch_size=batch_size,
            cell_budget=cell_budget,
            fallback_references=fallback_references,
            record=record,
        )
    elif method == APE_K:
        selected, stop_reason, algo_references = _run_ape(
            puller,
            batch_size=batch_size,
            cell_budget=cell_budget,
            fallback_references=fallback_references,
            epsilon1=epsilon1,
            delta=ape_delta,
            k1=ape_k1,
            record=record,
        )
    else:
        from agentopt.model_selection.qnehvi import encode_configuration_features

        features, cat_dims = encode_configuration_features(models)
        selected, stop_reason, algo_references = _run_qnehvi(
            puller,
            batch_size=batch_size,
            cell_budget=cell_budget,
            fallback_references=fallback_references,
            features=features,
            categorical_dims=cat_dims,
            reference_point=reference,
            mc_samples=qnehvi_mc_samples,
            refit_every=_positive_int(qnehvi_refit_every, "qnehvi_refit_every"),
            candidate_batch_size=qnehvi_candidate_batch_size,
            seed=seed,
            record=record,
        )
    wall_time = time.perf_counter() - wall_start
    if not selected and not complete_only:
        selected = empirical_raw_pareto_arms(puller)
    completed = tuple(
        i for i in range(puller.n_arms)
        if int(puller.n_pulls[i]) == len(questions)
    )
    completed_pareto = completed_raw_pareto_arms(puller)
    if complete_only:
        selected = completed_pareto
    record("terminal", selected)

    selected_hv, selected_gd, selected_igd = _score_selection(
        selected, truth_normalized, true_front_vectors, reference,
    )
    recalled = len(true_front.intersection(selected))
    params: Dict[str, object] = {
        "method": method,
        "batch_size": batch_size,
        "observation_budget_fraction": fraction,
        "cell_budget": cell_budget,
        "n_models": len(models),
        "n_questions": len(questions),
        "algorithm_cost_reference_usd": float(algo_references[objective_order.index("D")]),
        "evaluation_cost_reference_usd": eval_cost_reference,
        "bruteforce_search_cost_usd": bruteforce_cost,
        "reference_point": list(reference),
        "objectives": list(objective_order),
        "halt_on_identification_stop": False,
        "complete_only": bool(complete_only),
        "recommendation_checkpoint_interval": recommendation_checkpoint_interval,
        "recommendation_checkpoint_events_seen": ordinary_checkpoint_events,
        "recommendation_checkpoint_events_skipped": skipped_checkpoint_events,
    }
    if objective_order == THREE_OBJECTIVES:
        params["algorithm_latency_reference_seconds"] = float(algo_references[1])
        params["evaluation_latency_reference_seconds"] = float(eval_latency_reference)
    if method == APE_K:
        params.update(
            {
                "epsilon1": epsilon1,
                "ape_delta": ape_delta,
                "ape_k1": ape_k1,
                "ape_k": int(ape_k),
                "ape_stopping": False,
            }
        )
    if method == QNEHVI:
        params.update(
            {
                "qnehvi_mc_samples": int(qnehvi_mc_samples),
                "qnehvi_refit_every": int(qnehvi_refit_every),
                "qnehvi_candidate_batch_size": qnehvi_candidate_batch_size,
            }
        )

    return ParetoBaselineResult(
        selector=method,
        seed=int(seed),
        params=params,
        selected_arm_indices=tuple(int(i) for i in selected),
        selected_models=tuple(models[i] for i in selected),
        completed_arm_indices=completed,
        completed_models=tuple(models[i] for i in completed),
        completed_pareto_arm_indices=completed_pareto,
        completed_pareto_models=tuple(models[i] for i in completed_pareto),
        total_evaluations=int(puller.total_evaluations),
        total_search_cost_usd=float(puller.total_cost_usd),
        hypervolume=float(selected_hv),
        ground_truth_hypervolume=float(ground_truth_hv),
        hypervolume_regret=float(max(0.0, ground_truth_hv - selected_hv)),
        generational_distance=float(selected_gd),
        inverted_generational_distance=float(selected_igd),
        true_front_recall=float(recalled / len(true_front) if true_front else 1.0),
        recommendation_precision=float(
            recalled / len(selected) if selected else 1.0
        ),
        false_positive_count=int(len(set(selected) - true_front)),
        stop_reason=stop_reason,
        cost_reference_usd=eval_cost_reference,
        policy_wall_time_seconds=float(wall_time),
        recommendation_trajectory=trajectory,
        estimated_raw_vectors=puller.raw_means(),
        truth_raw_vectors=truth_raw,
        truth_vectors=truth_normalized,
    )


def write_trajectory_csv(
    rows: Sequence[Mapping[str, object]], path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[EGE_SH, APE_K, QNEHVI],
        choices=list(METHODS),
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--ape-k", type=int, default=3)
    parser.add_argument("--qnehvi-mc-samples", type=int, default=64)
    parser.add_argument("--qnehvi-refit-every", type=int, default=8)
    parser.add_argument("--qnehvi-candidate-batch-size", type=int, default=64)
    parser.add_argument(
        "--trajectory-checkpoint-interval",
        type=int,
        default=DEFAULT_RECOMMENDATION_CHECKPOINT_INTERVAL,
        help=(
            "Retain one ordinary recommendation checkpoint every N policy "
            "batches (default: %(default)s; use 1 for every batch)"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/combined_objective/results/pareto_baselines"),
    )
    args = parser.parse_args()
    pickle_path = _require_data_path(args.pickle)
    models, datapoints, table = load_pickle(pickle_path)
    outdir = args.outdir if args.outdir.is_absolute() else _REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}
    trajectory_rows: List[Dict[str, object]] = []
    for method in args.methods:
        method_runs = []
        for offset in range(args.seeds):
            seed = args.base_seed + offset
            print(f"\n=== {method} seed={seed} ===")
            result = simulate_pareto_baseline(
                models,
                datapoints,
                table,
                method=method,
                seed=seed,
                batch_size=args.batch_size,
                observation_budget_fraction=args.budget_fraction,
                ape_k=args.ape_k,
                qnehvi_mc_samples=args.qnehvi_mc_samples,
                qnehvi_refit_every=args.qnehvi_refit_every,
                qnehvi_candidate_batch_size=args.qnehvi_candidate_batch_size,
                recommendation_checkpoint_interval=(
                    args.trajectory_checkpoint_interval
                ),
            )
            method_runs.append(result)
            print(
                f"stop={result.stop_reason} evals={result.total_evaluations} "
                f"cost_frac={result.total_search_cost_usd / float(result.params['bruteforce_search_cost_usd']):.3f} "
                f"hv_regret={result.hypervolume_regret:.5f} "
                f"gd={result.generational_distance:.5f} "
                f"igd={result.inverted_generational_distance:.5f} "
                f"archive={len(result.selected_arm_indices)} "
                f"wall={result.policy_wall_time_seconds:.1f}s"
            )
            for point in result.recommendation_trajectory:
                trajectory_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "event": point.event,
                        "budget_fraction": point.budget_fraction,
                        "cumulative_evaluations": point.cumulative_evaluations,
                        "cumulative_search_cost_usd": point.cumulative_search_cost_usd,
                        "hv_regret": point.hypervolume_regret,
                        "generational_distance": point.generational_distance,
                        "inverted_generational_distance": point.inverted_generational_distance,
                        "archive_size": len(point.selected_arm_indices),
                    }
                )
        last = method_runs[-1]
        summary[method] = {
            "selector": method,
            "n_seeds": args.seeds,
            "stop_reason": last.stop_reason,
            "mean_hv_regret": float(np.mean([r.hypervolume_regret for r in method_runs])),
            "mean_generational_distance": float(
                np.mean([r.generational_distance for r in method_runs])
            ),
            "mean_inverted_generational_distance": float(
                np.mean([r.inverted_generational_distance for r in method_runs])
            ),
            "mean_evaluations": float(np.mean([r.total_evaluations for r in method_runs])),
            "mean_cost_usd": float(np.mean([r.total_search_cost_usd for r in method_runs])),
            "mean_archive_size": float(np.mean([len(r.selected_arm_indices) for r in method_runs])),
            "mean_wall_seconds": float(np.mean([r.policy_wall_time_seconds for r in method_runs])),
            "final_models": list(last.selected_models),
            "params": last.params,
        }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    if trajectory_rows:
        write_trajectory_csv(trajectory_rows, outdir / "cost_trajectory.csv")
        print(f"wrote {outdir / 'cost_trajectory.csv'}")


if __name__ == "__main__":
    main()
