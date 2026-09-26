#!/usr/bin/env python3
"""Replay Q, (Q+L)/2, L, D CC-Gittins on the complete QA matrices."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

from experiments.combined_objective.offline_three_objective_gittins import (
    simulate_three_objective_gittins,
)
from experiments.combined_objective.three_objective_metrics import (
    enrich_run,
    load_three_objective_benchmark,
)

DIRECTIONS = ((1., 0., 0.), (.5, .5, 0.), (0., 1., 0.), (0., 0., 1.))
DIRECTION_LABELS = ("Q", "(Q+L)/2", "L", "D")
DEFAULT_OUTDIR = ROOT / "experiments/combined_objective/results/three_objective_ql_midpoint_d"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint(run: dict, target: float) -> dict:
    """Latest recorded physical batch at/below target, only if reached."""
    terminal = float(run["cost_fraction"])
    eligible = [p for p in run["points"] if p["cost_fraction"] <= target + 1e-12]
    if terminal + 1e-12 < target or not eligible:
        return {"target_cost_fraction": target, "available": False,
                "terminal_cost_fraction": terminal}
    return {"target_cost_fraction": target, "available": True, **eligible[-1]}


def summarize(run: dict) -> dict:
    last = run["points"][-1]
    return {
        "stop_reason": run["stop_reason"],
        "cost_fraction": run["cost_fraction"],
        "search_cost_usd": run["search_cost_usd"],
        "bruteforce_search_cost_usd": run["bruteforce_search_cost_usd"],
        "total_evaluations": run["total_evaluations"],
        "true_pareto_count": len(run["full_data_pareto_arm_indices"]),
        "selected_count": len(last["selected_arm_indices"]),
        **{key: last[key] for key in (
            "pareto_precision", "pareto_recall", "relative_hv_regret",
            "pareto_false_positive_count", "pareto_false_negative_count", "exact_frontier",
        )},
    }


def source_hashes() -> dict:
    files = [Path(__file__), *[
        ROOT / "experiments/combined_objective" / name for name in (
            "offline_three_objective_gittins.py", "three_objective_metrics.py",
            "offline_radial_gittins.py",
        )], *[
        ROOT / "src/agentopt/model_selection" / name for name in (
            "radial_gittins.py", "radial_gittins_dp.py", "axis_gittins_dp.py",
        )]]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run_one(benchmark: str, *, seed: int, outdir: Path,
            observation_budget_fraction: float = 1., grid_size: int = 129) -> dict:
    models, questions, table, hashes = load_three_objective_benchmark(benchmark)
    path = outdir / f"seed_{seed}" / benchmark / "result.json"
    config = {
        "schema_version": 1,
        "benchmark": benchmark, "seed": seed,
        "objective_order": ["Q", "L", "D"],
        "raw_objectives": ["accuracy_max", "mean_latency_seconds_min", "mean_cost_usd_min"],
        "directions": [list(d) for d in DIRECTIONS],
        "direction_labels": list(DIRECTION_LABELS),
        "direction_scheduler": "round_robin", "eta_decay_schedule": "direction_stop",
        "eta_initial": 1., "eta_decay": .5,
        "batch_size": 4, "warm_start_batch_size": 4,
        "question_order": "independent", "warm_start_question_order": "independent",
        "recommendation_rule": "finite_lcb", "recommendation_beta": 1.,
        "latency_model": "raw_mean", "cost_model": "raw_mean",
        "acquisition_cost_unit": "USD", "grid_size": grid_size,
        "observation_budget_fraction": observation_budget_fraction,
        "input_sha256": hashes, "source_sha256": source_hashes(),
    }
    if path.exists():
        saved = json.loads(path.read_text())
        mismatches = [k for k, v in config.items() if saved.get("config", {}).get(k) != v]
        if mismatches:
            raise ValueError(f"Refusing to reuse {path}; changed {mismatches}. Use a new --outdir.")
        print(f"Reusing {path}", flush=True)
        return saved

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"Running {benchmark}, seed {seed}: Q, (Q+L)/2, L, D; "
          f"{len(models)} configurations x {len(questions)} questions", flush=True)
    last_progress = started

    def progress(event: dict) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if now - last_progress >= 15.:
            print(f"  {benchmark}: {event['total_evaluations']} observations, "
                  f"{event['completed_arms']} complete configurations, "
                  f"${event['total_cost']:.4f} search spend, {now-started:.0f}s", flush=True)
            last_progress = now

    run = simulate_three_objective_gittins(
        models, questions, table, seed=seed, directions=DIRECTIONS,
        batch_size=4, warm_start_batch_size=4, grid_size=grid_size,
        question_order="independent", warm_start_question_order="independent",
        observation_budget_fraction=observation_budget_fraction,
        progress_callback=progress,
    )
    simulation_seconds = time.perf_counter() - started
    run = enrich_run(run)
    payload = {
        "config": config,
        "timing": {"started_at_utc": started_utc, "simulation_seconds": simulation_seconds,
                   "total_seconds": time.perf_counter() - started},
        "summary": summarize(run),
        "checkpoints": [checkpoint(run, target) for target in (.05, .10, .20, .30, .50)],
        "run": run,
    }
    write_json(path, payload)
    print(f"Finished {benchmark}: {run['cost_fraction']:.2%} exhaustive USD, "
          f"recall={payload['summary']['pareto_recall']:.1%}, "
          f"HV regret={payload['summary']['relative_hv_regret']:.2%}, "
          f"{simulation_seconds:.1f}s; {path}", flush=True)
    return payload


def plot_results(payloads: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(len(payloads), 3, figsize=(15, 4 * len(payloads)), squeeze=False)
    for row, payload in enumerate(payloads):
        run = payload["run"]
        truth = np.asarray(run["raw_truth_vectors"])
        front = np.asarray(run["full_data_pareto_arm_indices"], dtype=int)
        selected = np.asarray(run["points"][-1]["selected_arm_indices"], dtype=int)
        for col, (xcoord, xlabel) in enumerate(((2, "Mean deployment cost (USD)"),
                                               (1, "Mean latency (seconds)"))):
            ax = axes[row, col]
            ax.scatter(truth[:, xcoord], truth[:, 0], s=18, c="#ccd2db", label="All configurations")
            ax.scatter(truth[front, xcoord], truth[front, 0], s=60, facecolors="none",
                       edgecolors="#c88400", linewidths=1.5, label="True 3D Pareto set")
            ax.scatter(truth[selected, xcoord], truth[selected, 0], s=32, marker="x",
                       c="#007f84", label="Final recommendation")
            ax.set(xlabel=xlabel, ylabel="Accuracy", title=f"{payload['config']['benchmark']} · 3D set projected")
            ax.grid(alpha=.18)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        ax = axes[row, 2]
        points = run["points"]
        x = [p["cost_fraction"] for p in points]
        for field, label, color in (("pareto_recall", "Recall", "#007f84"),
                                     ("pareto_precision", "Precision", "#4d63b3"),
                                     ("relative_hv_regret", "3D HV regret", "#c05b28")):
            ax.step(x, [p[field] for p in points], where="post", label=label, color=color)
        ax.set(xlabel="Actual search cost / exhaustive USD", ylabel="Fraction", ylim=(-.025, 1.025),
               title=f"Seed {payload['config']['seed']} · Q, (Q+L)/2, L, D")
        ax.xaxis.set_major_formatter(PercentFormatter(1.))
        ax.yaxis.set_major_formatter(PercentFormatter(1.))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "overview.png")
    fig.savefig(output / "overview.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(7 * len(payloads), 5.5))
    for col, payload in enumerate(payloads):
        ax = fig.add_subplot(1, len(payloads), col + 1, projection="3d")
        run = payload["run"]
        raw = np.asarray(run["raw_truth_vectors"])
        front = run["full_data_pareto_arm_indices"]
        selected = run["points"][-1]["selected_arm_indices"]
        ax.scatter(raw[:, 2], raw[:, 1], raw[:, 0], c="#bdc5d0", s=16, alpha=.45)
        ax.scatter(raw[front, 2], raw[front, 1], raw[front, 0], facecolors="none",
                   edgecolors="#c88400", s=65, label="True Pareto set")
        ax.scatter(raw[selected, 2], raw[selected, 1], raw[selected, 0], c="#007f84",
                   marker="x", s=40, label="Final recommendation")
        ax.set(xlabel="Mean USD ↓", ylabel="Mean seconds ↓", zlabel="Accuracy ↑",
               title=payload["config"]["benchmark"])
        ax.view_init(elev=23, azim=42)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "frontiers_3d.png")
    plt.close(fig)


def export_results(payloads: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [{"benchmark": p["config"]["benchmark"], "seed": p["config"]["seed"],
             **p["summary"], "simulation_seconds": p["timing"]["simulation_seconds"]} for p in payloads]
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(payloads, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=("mathqa", "hotpotqa"), default=["mathqa", "hotpotqa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--observation-budget-fraction", type=float, default=1.)
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="For independent benchmark workers.")
    args = parser.parse_args()
    if not 0. < args.observation_budget_fraction <= 1.:
        parser.error("--observation-budget-fraction must lie in (0, 1]")
    if args.grid_size < 17:
        parser.error("--grid-size must be at least 17")
    if args.plot_only:
        payloads = [json.loads((args.outdir / f"seed_{args.seed}" / b / "result.json").read_text()) for b in args.benchmarks]
    else:
        payloads = [run_one(b, seed=args.seed, outdir=args.outdir,
                           observation_budget_fraction=args.observation_budget_fraction,
                           grid_size=args.grid_size) for b in args.benchmarks]
    if not args.skip_export:
        export_results(payloads, args.outdir / f"seed_{args.seed}")


if __name__ == "__main__":
    main()
