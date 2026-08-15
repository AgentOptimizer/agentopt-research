"""
Matrix UCB exploration for model selection (banditeval-style).

Each **combination** (model combo) × **datapoint** (dataset item / question) cell is a
possible evaluation. These selectors pick *which datapoint index* to run next via UCB
over the partially observed matrix—unlike ordered-dataset bandit selectors that advance through datapoints in order.

Plain UCB follows :func:`upper_confidence_bound_exploration` from banditeval.

``select_best(..., max_concurrent=k)`` caps how many **(combination, datapoint)** cells
are chosen per UCB step and how many of those evaluations run at once. The
``parallel`` flag on ``select_best`` is **ignored** for these selectors (unlike others,
where it toggles multi-combo parallelism and ``max_concurrent`` splits combo vs
datapoint slots).

``observation_budget_fraction`` (default ``1.0``) limits how much of the grid is
observed; below ``1.0`` the run stops once that fraction (ceiling) of cells is filled.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


def _resolve_observation_budget_fraction(
    observation_budget_fraction: float, sample_fraction: Optional[float],
) -> float:
    """Match Bayesian: ``sample_fraction`` overrides ``observation_budget_fraction``."""
    if sample_fraction is not None:
        s = float(sample_fraction)
        if not 0 < s <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")
        return min(1.0, s)
    o = float(observation_budget_fraction)
    if o <= 0:
        raise ValueError("observation_budget_fraction must be positive")
    return min(1.0, o)



def _target_observation_count(
    n_combos: int, n_datapoints: int, observation_budget_fraction: float,
) -> int:
    """Stop after this many observed cells (ceiling of fraction × grid size; full grid if ≥ 1)."""
    total = n_combos * n_datapoints
    if observation_budget_fraction >= 1.0:
        return total
    return max(1, int(math.ceil(observation_budget_fraction * total)))


def _np_filled_count(observed: np.ndarray) -> int:
    return int(np.sum(~np.isnan(observed)))


def _ucb_plain_next_batch(
    observed: np.ndarray, a: float, max_cells: int, rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Next batch of matrix cells: shape ``(2, k)`` rows [combo_idx, datapoint_idx], or ``None``."""
    n_combos, n_datapoints = observed.shape
    bounds = np.full(n_combos, np.inf, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mus = np.nanmean(observed, axis=1)
    counts = np.sum(~np.isnan(observed), axis=1)
    mask = counts > 0
    bounds[mask] = mus[mask] + np.sqrt(a / counts[mask])
    fully_observed_combo = counts == n_datapoints
    bounds[fully_observed_combo] = -np.inf
    if bool(np.all(fully_observed_combo)):
        return None
    best_combo = int(np.argmax(bounds))
    unobserved_dp = np.where(np.isnan(observed[best_combo]))[0]
    n_unobserved = int(unobserved_dp.size)
    if n_unobserved <= 0:
        return None
    k = min(max(max_cells, 1), n_unobserved)
    pick = rng.permutation(n_unobserved)[:k]
    dps = unobserved_dp[pick]
    combos = np.full(k, best_combo, dtype=np.int64)
    return np.stack([combos, dps.astype(np.int64)])


def _build_selection_results(
    selector: BaseModelSelector,
    all_combos: List[Dict[str, ModelCandidate]],
    cell_data: Dict[Tuple[int, int], Tuple[float, float, Optional[float], str]],
    n_datapoints: int,
) -> SelectionResults:
    all_results: List[ModelResult] = []
    for combo_idx, combo in enumerate(all_combos):
        combo_name = selector._combo_name(combo)
        scores: List[float] = []
        latencies: List[float] = []
        costs: List[Optional[float]] = []
        dp_ids: List[str] = []
        for datapoint_idx in range(n_datapoints):
            t = cell_data.get((combo_idx, datapoint_idx))
            if t:
                scores.append(t[0])
                latencies.append(t[1])
                costs.append(t[2])
                dp_ids.append(t[3])
        if scores:
            all_results.append(
                selector._build_combo_result(
                    combo_name, scores, latencies, dp_ids, costs=costs,
                )
            )
        else:
            all_results.append(
                selector._make_result(
                    model_name=combo_name,
                    accuracy=0.0,
                    latency_seconds=0.0,
                    input_tokens={},
                    output_tokens={},
                    attribute="combination",
                    is_best=False,
                )
            )

    selector._finalize_combined_objectives(all_results)
    best_info = selector._find_best(all_results)
    if best_info is not None:
        best_name, _ = best_info
        for result in all_results:
            if result.model_name == best_name:
                result.is_best = True
                break
    else:
        print("\n  No combinations succeeded.")

    return SelectionResults(results=all_results)


def _record_cells(
    cell_data: Dict[Tuple[int, int], Tuple[float, float, Optional[float], str]],
    combo_idx: int,
    datapoint_idx: int,
    scores: List[float],
    latencies: List[float],
    costs: List[Optional[float]],
    dp_ids: List[str],
) -> None:
    if not scores:
        cell_data[(combo_idx, datapoint_idx)] = (
            0.0,
            0.0,
            None,
            f"missing::{combo_idx}:{datapoint_idx}",
        )
        return
    cost = costs[0] if costs else None
    cell_data[(combo_idx, datapoint_idx)] = (
        scores[0], latencies[0], cost, dp_ids[0],
    )


def _refresh_observed_np(
    selector: BaseModelSelector,
    observed: np.ndarray,
    cell_data: Dict[Tuple[int, int], Tuple[float, float, Optional[float], str]],
) -> None:
    """Rewrite ``observed`` from ``cell_data`` against the selector's current normalizer.

    Called after absorbing new observations so plain-UCB row means use the
    latest combined objective rather than stale values. No-op (preserves prior
    contents) when no lambdas are configured.
    """
    if not selector._has_combined_objective:
        return
    for (ci, di), (sc, lat, cost, dp_id) in cell_data.items():
        if dp_id.startswith("missing::"):
            continue
        observed[ci, di] = selector._combined_objective(sc, lat, cost)




class MatrixUCBModelSelector(BaseModelSelector):
    """UCB on the full **combination × datapoint** matrix (row means + exploration bonus).

    Selection always proceeds in batches of matrix cells; only
    ``select_best(..., max_concurrent=...)`` matters (``parallel`` is ignored).

    ``observation_budget_fraction`` or, equivalently, ``sample_fraction`` (same meaning
    as in Bayesian optimization: fraction of the search budget —
    here, **fraction of matrix cells** to observe) caps evaluations. ``1.0`` fills the
    full grid; ``0.1`` stops after about 10% of cells. If both are passed, ``sample_fraction``
    wins.
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        a: float = 1.0,
        observation_budget_fraction: float = 1.0,
        sample_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
        lambda_cost: float = 0.0,
        lambda_latency: float = 0.0,
    ) -> None:
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
            lambda_cost=lambda_cost,
            lambda_latency=lambda_latency,
        )
        self.a = float(a)
        self.observation_budget_fraction = _resolve_observation_budget_fraction(
            observation_budget_fraction, sample_fraction,
        )
        self._rng = np.random.default_rng(seed)

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Run matrix UCB.

        The ``parallel`` argument is ignored. ``max_concurrent`` limits each
        step's (combination, datapoint) batch size and how many evaluations run
        at once.
        """
        return super().select_best(parallel=False, max_concurrent=max_concurrent)

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        return asyncio.run(self._select_async(max_concurrent))

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_datapoints = len(dataset_list)
        n_combos = len(all_combos)
        observed = np.full((n_combos, n_datapoints), np.nan, dtype=np.float64)
        cell_data: Dict[Tuple[int, int], Tuple[float, float, Optional[float], str]] = {}
        mc = max(max_concurrent, 1)
        target_n = _target_observation_count(
            n_combos, n_datapoints, self.observation_budget_fraction,
        )
        total_cells = n_combos * n_datapoints

        print(f"\n{'='*60}")
        print(
            f"Matrix UCB (async): {n_combos} combinations × {n_datapoints} datapoints, "
            f"a={self.a}, max {mc} concurrent"
            + (
                f", observe up to {target_n}/{total_cells} cells "
                f"({self.observation_budget_fraction:.0%} budget)"
                if target_n < total_cells
                else ""
            )
        )
        print(f"{'='*60}\n")

        sem = asyncio.Semaphore(mc)

        async def _one(
            combo_i: int, dp_i: int,
        ) -> Tuple[int, int, List[float], List[float], List[str]]:
            async with sem:
                combo = all_combos[combo_i]
                name = self._combo_name(combo)
                return (
                    combo_i,
                    dp_i,
                    *await self._evaluate_combo_async(
                        combo,
                        [dataset_list[dp_i]],
                        label=name,
                        max_concurrent=1,
                        dp_offset=dp_i,
                    ),
                )

        step = 0
        while True:
            filled = _np_filled_count(observed)
            if filled >= target_n:
                break
            mc_step = min(mc, target_n - filled)
            if mc_step <= 0:
                break
            batch = _ucb_plain_next_batch(observed, self.a, mc_step, self._rng)
            if batch is None:
                break
            combo_row, dp_row = batch[0], batch[1]
            step += 1
            print(
                f"Step {step}: {len(dp_row)} cells — "
                f"combination {int(combo_row[0])}, datapoint indices {dp_row.tolist()}"
            )
            outs = await asyncio.gather(
                *[_one(int(ci), int(di)) for ci, di in zip(combo_row, dp_row)],
                return_exceptions=True,
            )
            for res in outs:
                if isinstance(res, Exception):
                    logger.warning("Matrix UCB async cell error: %s", res)
                    continue
                combo_i, dp_i, sc, lat, ids = res
                costs = self._observe_combo(sc, lat, ids) if sc else []
                _record_cells(cell_data, combo_i, dp_i, sc, lat, costs, ids)
                if sc:
                    observed[combo_i, dp_i] = self._combined_objective(
                        sc[0], lat[0], costs[0],
                    )
            _refresh_observed_np(self, observed, cell_data)

        return _build_selection_results(self, all_combos, cell_data, n_datapoints)
