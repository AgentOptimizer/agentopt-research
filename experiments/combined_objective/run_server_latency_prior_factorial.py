"""Run matched Q/L/D prior-variance controls on server-latency data.

The original G0 uses fixed Q variance and warm L/D variances. The factorial
also evaluates the all-warm case and the other fixed/warm combinations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from experiments.combined_objective.offline_three_objective_gittins import simulate_three_objective_gittins
from experiments.combined_objective.run_three_objective_gittins import summarize
from experiments.combined_objective.run_three_objective_gittins_multiseed import compact_payload, dump_json
from experiments.combined_objective.three_objective_ablation import CONFIGURATIONS
from experiments.combined_objective.three_objective_metrics import enrich_run, load_three_objective_benchmark


ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(42, 62))
DATASETS = ("hotpotqa", "mathqa")
VARIANTS = (
    ("fixed", "warm", "warm"),
    ("warm", "warm", "fixed"),
    ("fixed", "warm", "fixed"),
    ("warm", "fixed", "warm"),
    ("fixed", "fixed", "warm"),
    ("warm", "fixed", "fixed"),
    ("fixed", "fixed", "fixed"),
    ("warm", "warm", "warm"),
)
DIRECTIONS = CONFIGURATIONS["four_direction_real_cost"]["directions"]


def variant_name(modes: tuple[str, str, str]) -> str:
    return "q_{}_l_{}_d_{}".format(*modes)


def run_one(modes: tuple[str, str, str], dataset: str, seed: int, output_root: Path) -> dict:
    output = output_root / variant_name(modes) / f"seed-{seed}" / dataset / "result.json"
    models, questions, table, input_hashes = load_three_objective_benchmark(dataset)
    source = ROOT / "experiments/combined_objective/offline_three_objective_gittins.py"
    config = {
        "variant": variant_name(modes), "benchmark": dataset, "seed": seed,
        "prior_variance_modes": list(modes), "observation_budget_fraction": 1.0,
        "input_sha256": input_hashes,
        "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "directions": [list(value) for value in DIRECTIONS],
    }
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["config"] != config:
            raise ValueError(f"Incompatible prior factorial result: {output}")
        return {"status": "reused", "variant": variant_name(modes), "dataset": dataset, "seed": seed}
    started = time.perf_counter()
    run = simulate_three_objective_gittins(
        models, questions, table, seed=seed, directions=DIRECTIONS,
        batch_size=4, warm_start_batch_size=4, grid_size=129,
        question_order="independent", warm_start_question_order="independent",
        lambda_initial=1., lambda_decay=.5, acquisition_cost_mode="real",
        continuation_mode="eta_decay", recommendation_beta=1.,
        quality_prior_variance_mode=modes[0], latency_prior_variance_mode=modes[1],
        cost_prior_variance_mode=modes[2], observation_budget_fraction=1.0,
    )
    run = enrich_run(run)
    run["selector"] = "three_objective_cc_gittins_" + variant_name(modes)
    payload = compact_payload({
        "config": config, "run": run, "summary": summarize(run),
        "timing": {"simulation_seconds": time.perf_counter() - started},
    })
    dump_json(output, payload)
    return {"status": "completed", "variant": variant_name(modes), "dataset": dataset,
            "seed": seed, "seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = [(modes, dataset, seed, args.output_root) for modes in VARIANTS
             for dataset in DATASETS for seed in SEEDS]
    print(f"Running {len(tasks)} prior-variance cases", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{index}/{len(tasks)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
