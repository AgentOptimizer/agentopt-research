#!/usr/bin/env python3
"""Offline replay for the cost-aware two-objective radial-UCB selector."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins import DEFAULT_DIRECTIONS
from agentopt.model_selection.radial_ucb import (
    BONUS_MODES,
    DEFAULT_EXPLORATION_BETA,
    POSTERIOR_SD_BONUS,
    RadialUCBPolicy,
)
from experiments.combined_objective import offline_radial_gittins as _radial


LookupTable = _radial.LookupTable
RadialSimulationResult = _radial.RadialSimulationResult
DirectionVisitContext = _radial.DirectionVisitContext
SELECTOR_NAME = "radial_ucb"


def simulate_radial_ucb(
    models: List[str],
    datapoints: List[int],
    table: LookupTable,
    *,
    batch_size: int = 4,
    directions: Iterable[Sequence[float]] = DEFAULT_DIRECTIONS,
    exploration_beta: float = DEFAULT_EXPLORATION_BETA,
    bonus_mode: str = POSTERIOR_SD_BONUS,
    cost_aware: bool = True,
    search_cost_scale_eta: float = 1.0,
    prior_variance: Sequence[float] | float = 0.04,
    obs_noise_variance: Optional[Sequence[float] | float] = None,
    cost_reference_usd: Optional[float] = None,
    reference_point: Sequence[float] = (0.0, 0.0),
    expected_batch_cost_usd: Optional[float | Sequence[float] | Mapping[Any, float]] = None,
    guaranteed_batch_cost_usd: Optional[float | Sequence[float] | Mapping[Any, float]] = None,
    effective_cost_bin_ratio: Optional[float] = None,
    effective_cost_bin_anchor: float = 1e-4,
    observation_budget_fraction: float = 1.0,
    max_total_question_evaluations: Optional[int] = None,
    max_search_cost_usd: Optional[float] = None,
    stop_tolerance: float = 1e-9,
    seed: int = 42,
    history: Optional[List[Dict[str, Any]]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    question_universe: str = "common",
    halt_on_index_stop: bool = True,
    record_recommendation_trajectory: bool = False,
) -> RadialSimulationResult:
    """Replay radial UCB using the shared radial multi-objective engine."""
    if bonus_mode not in BONUS_MODES:
        raise ValueError(f"bonus_mode must be one of {BONUS_MODES}")
    policy = RadialUCBPolicy(exploration_beta=exploration_beta, bonus_mode=bonus_mode)

    def ucb_index(context: DirectionVisitContext, arm_index: int) -> float:
        posterior = context.posteriors[arm_index]
        return policy.index(
            posterior.mean,
            posterior.var,
            context.direction,
            context.reference_point,
            n_batches=posterior.n_batches,
            effective_pull_cost=(
                context.effective_pull_costs[arm_index] if cost_aware else 0.0
            ),
        )

    return _radial.simulate_radial_gittins(
        models,
        datapoints,
        table,
        batch_size=batch_size,
        directions=directions,
        prior_variance=prior_variance,
        obs_noise_variance=obs_noise_variance,
        cost_reference_usd=cost_reference_usd,
        reference_point=reference_point,
        search_cost_scale_eta=search_cost_scale_eta,
        expected_batch_cost_usd=expected_batch_cost_usd,
        guaranteed_batch_cost_usd=guaranteed_batch_cost_usd,
        effective_cost_bin_ratio=effective_cost_bin_ratio,
        effective_cost_bin_anchor=effective_cost_bin_anchor,
        observation_budget_fraction=observation_budget_fraction,
        max_total_question_evaluations=max_total_question_evaluations,
        max_search_cost_usd=max_search_cost_usd,
        stop_tolerance=stop_tolerance,
        seed=seed,
        history=history,
        run_metadata=run_metadata,
        index_provider=ucb_index,
        question_universe=question_universe,
        halt_on_gittins_stop=halt_on_index_stop,
        record_recommendation_trajectory=record_recommendation_trajectory,
        selector_name=SELECTOR_NAME,
        extra_params={
            "exploration_beta": policy.exploration_beta,
            "bonus_mode": policy.bonus_mode,
            "cost_aware_index": bool(cost_aware),
        },
    )


__all__ = ["simulate_radial_ucb"]
