#!/usr/bin/env python3
"""Run a resumable Radial-Gittins eta sensitivity sweep.

Each benchmark/eta/seed trajectory is stored as an independent compressed
NumPy file. Plotting is intentionally handled by
``plot_radial_gittins_eta_sweep.py`` so visual changes never require rerunning
the policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentopt.model_selection.radial_gittins_dp import (  # noqa: E402
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    _sample_values,
    load_pickle,
    simulate_radial_gittins,
)


PICKLES = {
    "hotpotqa": ROOT / "experiments/data/lookup/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/data/lookup/mathqa_lookup.pkl",
}
DEFAULT_ETAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def _eta_slug(eta: float) -> str:
    return f"{eta:.8g}".replace(".", "p").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/vs/gittins_eta_sweep"))
    parser.add_argument("--etas", type=float, nargs="+", default=DEFAULT_ETAS)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=tuple(PICKLES),
        default=tuple(PICKLES),
        help="Benchmarks to run; defaults to both.",
    )
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, choices=(4, 8))
    parser.add_argument("--boundary-z-padding-min", type=float, default=2.0)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Optional fixed grid override; default uses production direction-aware grids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-after-stop",
        action="store_true",
        help="Continue replay to the full matrix after recording the natural stop.",
    )
    args = parser.parse_args()

    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    if any(not np.isfinite(eta) or eta <= 0 for eta in args.etas):
        raise SystemExit("every eta must be finite and positive")

    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    grid = (
        RadialGittinsGrid(
            z_size=args.grid_size,
            delta_size=args.grid_size,
            state_size=args.grid_size,
            boundary_margin_cells=max(2, min(4, args.grid_size // 32)),
        )
        if args.grid_size is not None
        else None
    )
    cache = RadialGittinsBoundaryCache()

    selected_pickles = {
        benchmark: PICKLES[benchmark] for benchmark in args.benchmarks
    }
    total = len(selected_pickles) * len(args.etas) * args.seeds
    completed = 0
    for benchmark, pickle_path in selected_pickles.items():
        models, datapoints, table = load_pickle(pickle_path)
        common_questions = sorted(
            set.intersection(
                *[
                    {question for question in datapoints if question in table.get(model, {})}
                    for model in models
                ]
            )
        )
        full_cost = float(
            sum(
                _sample_values(table[model][question])[1]
                for model in models
                for question in common_questions
            )
        )
        for eta in args.etas:
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                completed += 1
                destination = raw_dir / f"{benchmark}_eta-{_eta_slug(eta)}_seed-{seed}.npz"
                if destination.exists() and not args.overwrite:
                    print(f"[{completed}/{total}] cached {destination.name}", flush=True)
                    continue
                print(
                    f"[{completed}/{total}] run {benchmark} eta={eta:g} seed={seed}",
                    flush=True,
                )
                result = simulate_radial_gittins(
                    models,
                    datapoints,
                    table,
                    batch_size=args.batch_size,
                    observation_budget_fraction=1.0,
                    seed=seed,
                    boundary_grid=grid,
                    boundary_cache=cache,
                    search_cost_scale_eta=eta,
                    boundary_z_padding_min=args.boundary_z_padding_min,
                    halt_on_gittins_stop=not args.continue_after_stop,
                    record_recommendation_trajectory=True,
                    question_universe="common",
                )
                trajectory = result.recommendation_trajectory
                if not trajectory:
                    raise RuntimeError("Radial-Gittins returned an empty trajectory")
                cost_fraction = np.asarray(
                    [point.cumulative_search_cost_usd / full_cost for point in trajectory],
                    dtype=np.float64,
                )
                stop_cost_fraction = (
                    float(result.gittins_stop_cost_usd) / full_cost
                    if result.gittins_stop_cost_usd is not None
                    else np.nan
                )
                stop_points = [point for point in trajectory if point.event == "gittins_stop"]
                stop_point = stop_points[0] if stop_points else trajectory[-1]
                np.savez_compressed(
                    destination,
                    benchmark=benchmark,
                    eta=float(eta),
                    seed=int(seed),
                    grid_size=(int(args.grid_size) if args.grid_size is not None else -1),
                    batch_size=int(args.batch_size),
                    boundary_z_padding_min=float(args.boundary_z_padding_min),
                    cost_fraction=cost_fraction,
                    cell_fraction=np.asarray(
                        [point.budget_fraction for point in trajectory], dtype=np.float64
                    ),
                    deployable_hv_regret=np.asarray(
                        [point.deployable_hypervolume_regret for point in trajectory],
                        dtype=np.float64,
                    ),
                    provisional_hv_regret=np.asarray(
                        [point.hypervolume_regret for point in trajectory], dtype=np.float64
                    ),
                    stop_deployable_hv_regret=float(
                        stop_point.deployable_hypervolume_regret
                    ),
                    stop_provisional_hv_regret=float(stop_point.hypervolume_regret),
                    stop_cost_fraction=stop_cost_fraction,
                    stop_cell_fraction=(
                        float(result.gittins_stop_budget_fraction)
                        if result.gittins_stop_budget_fraction is not None
                        else np.nan
                    ),
                    stop_cost_usd=(
                        float(result.gittins_stop_cost_usd)
                        if result.gittins_stop_cost_usd is not None
                        else np.nan
                    ),
                    full_cost_usd=full_cost,
                    halted_at_gittins_stop=not args.continue_after_stop,
                )
                print(
                    f"    stop_cost={stop_cost_fraction:.3%}, "
                    f"stop_cells={result.gittins_stop_budget_fraction:.3%}",
                    flush=True,
                )

    print(f"wrote raw sweep runs under {raw_dir}")


if __name__ == "__main__":
    main()
