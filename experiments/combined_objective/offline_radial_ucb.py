#!/usr/bin/env python3
"""Offline replay for the cost-aware two-objective radial-UCB selector.

Radial-UCB is the optimism-based baseline for radial-Gittins.  It reuses the
entire replay protocol of :mod:`experiments.combined_objective.offline_radial_gittins`
-- the uniform warm start and its frozen calibration, one shared Gaussian
posterior per configuration, round-robin over the fixed radial directions,
required completion before an arm may be recommended, the budget guards, and
the raw-space archive metrics -- and replaces only the index of an unfinished
arm:

* radial-Gittins scores an arm by ``u - b_n(delta)`` from a boundary dynamic
  program;
* radial-UCB scores an arm by ``rho_lambda(mu + beta sigma) - c_eff``.

Everything else being identical is the point: a difference in hypervolume
regret or in endogenous stopping cost is attributable to the acquisition rule
rather than to the calibration, the question schedule, or the archive
convention.  No boundary tables, grids, or caches are built, so a run is
dramatically cheaper than radial-Gittins.

The result reuses :class:`RadialSimulationResult`.  Its ``gittins_stop_*`` and
``stopped_by_gittins`` fields are the shared engine's generic endogenous-stop
fields; under radial-UCB they record the budget at which a full direction cycle
first performed no evaluation, which happens once every direction prefers an
already-completed arm to the best optimistic unfinished one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from agentopt.model_selection.radial_gittins import DEFAULT_DIRECTIONS
from agentopt.model_selection.radial_ucb import (
    BONUS_MODES,
    DEFAULT_EXPLORATION_BETA,
    POSTERIOR_SD_BONUS,
    RadialUCBPolicy,
)

try:
    from experiments.combined_objective import offline_radial_gittins as _radial
except ImportError:  # Direct ``python combined_objective/offline_radial_ucb.py``.
    from combined_objective import offline_radial_gittins as _radial  # type: ignore[no-redef]

DirectionVisitContext = _radial.DirectionVisitContext
LookupTable = _radial.LookupTable
RadialSimulationResult = _radial.RadialSimulationResult
_jsonable_result = _radial._jsonable_result
_require_data_path = _radial._require_data_path
load_jsonl = _radial.load_jsonl
load_pickle = _radial.load_pickle
print_radial_result = _radial.print_radial_result
simulate_radial_gittins = _radial.simulate_radial_gittins
summarize_radial_multi_seed = _radial.summarize_radial_multi_seed


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
    expected_batch_cost_usd: Optional[
        float | Sequence[float] | Mapping[Any, float]
    ] = None,
    guaranteed_batch_cost_usd: Optional[
        float | Sequence[float] | Mapping[Any, float]
    ] = None,
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
    """Replay the warm-start + round-robin radial-UCB policy.

    ``exploration_beta`` and ``bonus_mode`` define the confidence radius; see
    :class:`agentopt.model_selection.radial_ucb.RadialUCBPolicy`.

    ``search_cost_scale_eta`` keeps the radial-Gittins meaning of converting one
    batch of dollars into normalized utility, so an eta sweep is directly
    comparable across the two policies.  ``cost_aware=False`` is the cost-blind
    ablation: eta is still recorded for provenance but the index drops the
    penalty, leaving pure optimism.  Unlike radial-Gittins, effective pull costs
    are not quantized by default, because there is no boundary table whose reuse
    the cost bins exist to enable.
    """
    if bonus_mode not in BONUS_MODES:
        raise ValueError(f"bonus_mode must be one of {BONUS_MODES}")
    policy = RadialUCBPolicy(
        exploration_beta=exploration_beta,
        bonus_mode=bonus_mode,
    )

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

    return simulate_radial_gittins(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pickle", help="Path to a cached lookup pickle")
    source.add_argument("--jsonl", help="Path to a brute-force JSONL file")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=1.0,
        help="Fraction of common-universe question cells that may be replayed",
    )
    parser.add_argument(
        "--max-search-cost",
        type=float,
        default=None,
        help=(
            "Dollar guard; soft with warm-start expected costs, hard when "
            "--guaranteed-batch-cost is supplied"
        ),
    )
    parser.add_argument(
        "--guaranteed-batch-cost",
        type=float,
        default=None,
        help="Optional per-batch upper bound in USD, shared by every arm",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_EXPLORATION_BETA,
        help="Exploration coefficient of the confidence radius",
    )
    parser.add_argument(
        "--bonus-mode",
        choices=BONUS_MODES,
        default=POSTERIOR_SD_BONUS,
        help="Confidence radius: posterior standard deviation or UCB1 counts",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="USD-to-normalized-utility search-cost scale",
    )
    parser.add_argument(
        "--cost-blind",
        action="store_true",
        help="Drop the search-cost penalty from the index (pure optimism)",
    )
    parser.add_argument(
        "--ragged-diagnostic",
        action="store_true",
        help="Use arm-specific available tails instead of the common universe",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.pickle:
        path = _require_data_path(args.pickle)
        models, datapoints, table = load_pickle(path)
    else:
        path = _require_data_path(args.jsonl)
        models, datapoints, table = load_jsonl(path)
    print(
        f"Loaded {len(models)} models, {len(datapoints)} questions, "
        f"{sum(len(row) for row in table.values())} cells from {path}"
    )

    results: List[RadialSimulationResult] = []
    for offset in range(args.seeds):
        result = simulate_radial_ucb(
            models,
            datapoints,
            table,
            batch_size=args.batch_size,
            exploration_beta=args.beta,
            bonus_mode=args.bonus_mode,
            cost_aware=not args.cost_blind,
            search_cost_scale_eta=args.eta,
            observation_budget_fraction=args.budget_fraction,
            max_search_cost_usd=args.max_search_cost,
            guaranteed_batch_cost_usd=args.guaranteed_batch_cost,
            seed=args.base_seed + offset,
            question_universe=("per_arm" if args.ragged_diagnostic else "common"),
        )
        print_radial_result(result)
        results.append(result)

    summary = summarize_radial_multi_seed(results)
    if len(results) > 1:
        print(f"\nSummary: {json.dumps(summary, indent=2, allow_nan=False)}")
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".csv":
            row = dict(summary)
            for key, value in tuple(row.items()):
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(value, allow_nan=False)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        else:
            payload = {
                "summary": summary,
                "results": [_jsonable_result(result) for result in results],
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
