"""
Gittins-index exploration for model selection (matrix bandit).

Same **combination × datapoint** observation matrix as :class:`MatrixUCBModelSelector`,
but next cells are chosen by Gittins indices
(:func:`gittins_index_exploration`) instead of plain UCB.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, SelectionResults
from .gittins_lookup import compute_roots_lookup_table
from .gittins_policy import gittins_index_exploration, gittins_post_pull_update
from .gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior
from .matrix_ucb import (
    _build_selection_results,
    _np_filled_count,
    _record_cells,
    _refresh_observed_np,
    _resolve_observation_budget_fraction,
    _target_observation_count,
)

logger = logging.getLogger(__name__)


class GittinsModelSelector(BaseModelSelector):
    """Gittins index on the full **combination × datapoint** matrix.

    Each step selects one combination (arm) with the highest Gittins index, then
    evaluates up to ``max_concurrent`` unevaluated datapoints on that combination
    (same batch layout as Matrix UCB / banditeval).

    ``observation_budget_fraction`` / ``sample_fraction`` caps how many matrix
    cells are observed (same semantics as Matrix UCB). The policy may also stop
    early when the top-scoring arm is fully observed (``allow_early_stop``).
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        prior_mean: float = 0.5,
        prior_variance: float = 0.04,
        obs_noise_variance: Optional[float] = None,
        cost_per_transition: Union[float, Sequence[float]] = 1.0,
        cost_scaling_factor: float = 1e-4,
        n_gittins_grid_points: int = 2**10 + 1,
        use_batch_mean_gittins_dp: bool = False,
        force_per_observation_dp: bool = False,
        allow_early_stop: bool = True,
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
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)
        self._obs_noise_variance_override = (
            None if obs_noise_variance is None else float(obs_noise_variance)
        )
        self.cost_per_transition = cost_per_transition
        self.cost_scaling_factor = float(cost_scaling_factor)
        self.n_gittins_grid_points = int(n_gittins_grid_points)
        self.use_batch_mean_gittins_dp = bool(use_batch_mean_gittins_dp)
        self.force_per_observation_dp = bool(force_per_observation_dp)
        self.allow_early_stop = bool(allow_early_stop)
        self.observation_budget_fraction = _resolve_observation_budget_fraction(
            observation_budget_fraction, sample_fraction,
        )
        self._seed = seed
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Run Gittins matrix exploration.

        ``parallel`` is ignored. ``max_concurrent`` is the per-step example batch
        size on the chosen combination (and the async concurrency cap).
        """
        return super().select_best(parallel=False, max_concurrent=max_concurrent)

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        return asyncio.run(self._select_async(max_concurrent))

    def _obs_noise_variance(self, batch_size: int) -> float:
        if self._obs_noise_variance_override is not None:
            return self._obs_noise_variance_override
        # Worst-case [0, 1] bound τ² ≈ 1/(4B) with B = batch size
        return 1.0 / (4.0 * max(int(batch_size), 1))

    def _build_roots_lookup(
        self, n_examples: int, n_arms: int, tau_sq_cell: float,
    ) -> Optional[torch.Tensor]:
        """Precompute root table for the default (fast) per-cell lookup path."""
        if self.use_batch_mean_gittins_dp or self.force_per_observation_dp:
            return None
        import jax
        import jax.numpy as jnp

        transition_stds = transition_stds_shrinking_gaussian_posterior(
            jnp.float32(self.prior_variance),
            jnp.float32(tau_sq_cell),
            n_examples,
        )
        cost = float(self.cost_per_transition) if isinstance(
            self.cost_per_transition, (int, float),
        ) else float(self.cost_per_transition[0])  # type: ignore[index]
        cost *= self.cost_scaling_factor
        roots_all = compute_roots_lookup_table(
            transition_stds=transition_stds,
            costs_per_arm=jnp.float32(cost),
            n_points=self.n_gittins_grid_points,
        )
        table = torch.from_numpy(np.array(jax.device_get(roots_all), copy=True)).to(torch.float32)
        if table.ndim == 1:
            table = table.unsqueeze(0)
        # Broadcast to n_arms rows if needed (homogeneous cost)
        if table.shape[0] == 1 and n_arms > 1:
            table = table.expand(n_arms, -1).contiguous()
        return table

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_datapoints = len(dataset_list)
        n_combos = len(all_combos)
        observed_np = np.full((n_combos, n_datapoints), np.nan, dtype=np.float64)
        observed_t = torch.full(
            (n_combos, n_datapoints), float("nan"), dtype=torch.float32,
        )
        cell_data: Dict[Tuple[int, int], Tuple[float, float, Optional[float], str]] = {}
        mc = max(int(max_concurrent), 1)
        target_n = _target_observation_count(
            n_combos, n_datapoints, self.observation_budget_fraction,
        )
        total_cells = n_combos * n_datapoints
        tau_sq = self._obs_noise_variance(mc)

        cached_scores = torch.full((n_combos,), float("inf"), dtype=torch.float32)
        roots_lookup_table = self._build_roots_lookup(n_datapoints, n_combos, tau_sq)
        natural_stop: List[Optional[int]] = [None]
        rec_aware_stop: List[Optional[int]] = [None]
        last_pulled: Optional[List[int]] = None
        sim_cum_eval = 0

        print(f"\n{'='*60}")
        print(
            f"Gittins (async): {n_combos} combinations × {n_datapoints} datapoints, "
            f"batch≤{mc}, τ²={tau_sq:.6f}"
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
            filled = _np_filled_count(observed_np)
            if filled >= target_n:
                break

            remaining = target_n - filled
            batch_size = min(mc, remaining)
            if batch_size <= 0:
                break

            out = gittins_index_exploration(
                observed_t,
                prior_mean=self.prior_mean,
                prior_variance=self.prior_variance,
                obs_noise_variance=tau_sq,
                cost_per_transition=self.cost_per_transition,
                cost_scaling_factor=self.cost_scaling_factor,
                n_gittins_grid_points=self.n_gittins_grid_points,
                batch_size=batch_size,
                cached_scores=cached_scores,
                recompute_arms=None if last_pulled is None else last_pulled,
                use_batch_mean_gittins_dp=self.use_batch_mean_gittins_dp,
                force_per_observation_dp=self.force_per_observation_dp,
                allow_early_stop=self.allow_early_stop,
                roots_lookup_table=roots_lookup_table,
            )
            if out is None:
                break

            combo_row = out[0].tolist()
            dp_row = out[1].tolist()
            if not dp_row:
                break

            step += 1
            print(
                f"Step {step}: {len(dp_row)} cells — "
                f"combination {int(combo_row[0])}, datapoint indices {dp_row}"
            )

            outs = await asyncio.gather(
                *[_one(int(ci), int(di)) for ci, di in zip(combo_row, dp_row)],
                return_exceptions=True,
            )
            pulled_arms = set()
            for res in outs:
                if isinstance(res, Exception):
                    logger.warning("Gittins async cell error: %s", res)
                    continue
                combo_i, dp_i, sc, lat, ids = res
                costs = self._observe_combo(sc, lat, ids) if sc else []
                _record_cells(cell_data, combo_i, dp_i, sc, lat, costs, ids)
                if sc:
                    val = self._combined_objective(sc[0], lat[0], costs[0])
                    observed_np[combo_i, dp_i] = val
                    observed_t[combo_i, dp_i] = float(val)
                    sim_cum_eval += 1
                    pulled_arms.add(int(combo_i))
            _refresh_observed_np(self, observed_np, cell_data)
            # Keep torch matrix in sync when combined-objective normalizer refreshes
            if self._has_combined_objective:
                observed_t = torch.from_numpy(observed_np.astype(np.float32))

            last_pulled = sorted(pulled_arms) if pulled_arms else last_pulled
            if last_pulled:
                _, cached_scores = gittins_post_pull_update(
                    observed_t,
                    cached_scores=cached_scores,
                    recompute_arms=last_pulled,
                    prior_mean=self.prior_mean,
                    prior_variance=self.prior_variance,
                    obs_noise_variance=tau_sq,
                    cost_per_transition=self.cost_per_transition,
                    cost_scaling_factor=self.cost_scaling_factor,
                    n_gittins_grid_points=self.n_gittins_grid_points,
                    batch_size=batch_size,
                    use_batch_mean_gittins_dp=self.use_batch_mean_gittins_dp,
                    force_per_observation_dp=self.force_per_observation_dp,
                    roots_lookup_table=roots_lookup_table,
                    sim_cum_eval=sim_cum_eval,
                    natural_stop_cum_eval_holder=natural_stop,
                    recommendation_aware_stop_cum_eval_holder=rec_aware_stop,
                )

            if (
                self.allow_early_stop
                and natural_stop[0] is not None
                and self.observation_budget_fraction >= 1.0
            ):
                # Natural Gittins stop: top arm fully observed
                break

        return _build_selection_results(self, all_combos, cell_data, n_datapoints)
