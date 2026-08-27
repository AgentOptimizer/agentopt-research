#!/usr/bin/env python3
"""Aggregate and plot the paper comparison for continuous seeds 42--61."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.plot_paper_20seed_comparison import (  # noqa: E402
    PICKLES,
    _baseline_curves,
    _checkpoint_metric_trajectory,
    plot_frontier_grid,
    write_hv_comparison,
)
from experiments.combined_objective.plot_multiobjective_method_comparison import (  # noqa: E402
    _radial_regret_series,
    _random_regret_series,
    write_comparison_figure,
)
from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    run_budget_sweep,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


SEEDS = tuple(range(42, 62))
BENCHMARKS = ("hotpotqa", "mathqa")
METHODS = ("ege_sh", "ape_k")
LABELS = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}
PAPER_DATA = ROOT / "analysis" / "paper_20seed_method_comparison"
METHOD_STYLES = {
    "radial_gittins_deployable": {"color": "tab:orange", "label": "Radial-Gittins deployable"},
    "radial_gittins_provisional": {"color": "tab:orange", "label": "Radial-Gittins provisional"},
    "ege_sh": {"color": "tab:olive", "label": "EGE-SH"},
    "ape_k": {"color": "tab:blue", "label": "APE-k"},
    "qnehvi": {"color": "tab:pink", "label": "qNEHVI"},
    "random_questions": {"color": "tab:purple", "label": "Random shared questions"},
    "random_configurations": {"color": "tab:brown", "label": "Random configurations"},
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_ege_ape(data_dir: Path, fraction_dir: str) -> None:
    root = data_dir / fraction_dir
    trajectory: list[dict[str, object]] = []
    recommendations: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for seed in SEEDS:
                if seed == 58:
                    run_dir = root / "seed58_runs" / benchmark
                else:
                    run_dir = root / "raw" / benchmark / method / f"seed-{seed}"
                rows = [
                    row for row in _read_rows(run_dir / "cost_trajectory.csv")
                    if row["method"] == method and int(row["seed"]) == seed
                ]
                if not rows:
                    raise ValueError(f"missing {benchmark} {method} seed {seed}")
                trajectory.extend({"benchmark": LABELS[benchmark], **row} for row in rows)
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                result = summary[method]
                recommendations.append({
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "budget_fraction": result["params"]["observation_budget_fraction"],
                    "actual_cost_fraction": (
                        result["mean_cost_usd"]
                        / result["params"]["bruteforce_search_cost_usd"]
                    ),
                    "selected_models": json.dumps(result["final_models"]),
                })
    _write_rows(root / "cost_trajectory.csv", trajectory)
    _write_rows(root / "recommendations.csv", recommendations)


def _aggregate_qnehvi(data_dir: Path) -> None:
    root = data_dir / "qnehvi"
    full_rows: list[dict[str, object]] = []
    recommendation_rows: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for seed in SEEDS:
            full_dir = root / "full" / "raw" / benchmark / f"seed-{seed}"
            full_rows.extend(
                {"benchmark": LABELS[benchmark], **row}
                for row in _read_rows(full_dir / "cost_trajectory.csv")
            )
            short_dir = root / "10pct" / "raw" / benchmark / f"seed-{seed}"
            summary = json.loads((short_dir / "summary.json").read_text(encoding="utf-8"))
            result = summary["qnehvi"]
            recommendation_rows.append({
                "benchmark": benchmark,
                "seed": seed,
                "actual_cost_fraction": (
                    result["mean_cost_usd"]
                    / result["params"]["bruteforce_search_cost_usd"]
                ),
                "selected_models": json.dumps(result["final_models"]),
            })
    _write_rows(root / "cost_trajectory.csv", full_rows)
    _write_rows(root / "recommendations_10pct.csv", recommendation_rows)


def _prepare_seed_level_data(data_dir: Path) -> None:
    old_gittins = _read_rows(PAPER_DATA / "gittins_seed_results.csv")
    new_gittins = _read_rows(
        data_dir.parent / "seed58_validation" / "gittins_seed_results.csv"
    )
    gittins_rows = [
        row for row in old_gittins if int(row["seed"]) in SEEDS and int(row["seed"]) != 58
    ] + new_gittins
    gittins_rows.sort(key=lambda row: (row["benchmark"], int(row["seed"])))
    _write_rows(data_dir / "gittins_seed_results.csv", gittins_rows)

    for benchmark in BENCHMARKS:
        old_random = [
            row for row in _read_rows(PAPER_DATA / f"{benchmark}_random_questions.csv")
            if int(row["seed"]) in SEEDS and int(row["seed"]) != 58
        ]
        models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
        fresh = run_budget_sweep(
            models,
            datapoints,
            table,
            versions=("random_questions",),
            budget_fractions=(0.1, 0.4),
            seeds=(58,),
        )
        for result in fresh:
            old_random.append({
                "version": result.version,
                "seed": result.seed,
                "budget_fraction": result.budget_fraction,
                "total_evaluations": result.total_evaluations,
                "total_search_cost_usd": result.total_search_cost_usd,
                "n_recommended": len(result.selected_models),
                "hypervolume": result.hypervolume,
                "ground_truth_hypervolume": result.ground_truth_hypervolume,
                "hypervolume_regret": result.hypervolume_regret,
                "generational_distance": result.generational_distance,
                "inverted_generational_distance": result.inverted_generational_distance,
                "true_front_recall": result.true_front_recall,
                "recommendation_precision": result.recommendation_precision,
                "false_positive_count": result.false_positive_count,
                "selected_models": json.dumps(result.selected_models),
            })
        old_random.sort(key=lambda row: (int(row["seed"]), float(row["budget_fraction"])))
        _write_rows(data_dir / f"{benchmark}_random_questions.csv", old_random)


def _write_distances_once(data_dir: Path, figures: Path) -> None:
    baseline_rows = _read_rows(data_dir / "pareto_baselines" / "cost_trajectory.csv")
    baseline_rows.extend(_read_rows(data_dir / "qnehvi" / "cost_trajectory.csv"))
    specs = (
        ("generational_distance", "Generational distance (GD)", "all_methods_20seed_gd.png"),
        ("inverted_generational_distance", "Inverted generational distance (IGD)", "all_methods_20seed_igd.png"),
    )
    panels = {field: [] for field, _, _ in specs}
    summary: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        trajectories = {
            field: {"deployable": [], "provisional": []} for field, _, _ in specs
        }
        template = None
        stop_fractions = []
        for seed in SEEDS:
            path = data_dir / "raw_gittins" / f"{benchmark}_seed-{seed}.pkl"
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            run = saved[0] if isinstance(saved, tuple) else saved
            if template is None:
                template = run
            stop_fractions.append(
                run.gittins_stop_cost_usd
                / run.recommendation_trajectory[-1].cumulative_search_cost_usd
            )
            for field, _, _ in specs:
                trajectories[field]["deployable"].append(
                    _checkpoint_metric_trajectory(
                        run, field, "online_raw_archive_arm_indices"
                    )
                )
                trajectories[field]["provisional"].append(
                    _checkpoint_metric_trajectory(
                        run, field, "posterior_archive_arm_indices"
                    )
                )
            del run, saved
        assert template is not None
        models, datapoints, table = load_pickle(str(PICKLES[benchmark]))
        random_results = run_budget_sweep(models, datapoints, table, seeds=SEEDS)
        radial_by_seed = {seed: template for seed in SEEDS}
        for field, _, _ in specs:
            deployable = _radial_regret_series(trajectories[field]["deployable"])
            provisional = _radial_regret_series(trajectories[field]["provisional"])
            random = [
                (
                    version,
                    *_random_regret_series(
                        random_results,
                        version=version,
                        radial_by_seed=radial_by_seed,
                        x_axis="cost",
                        field=field,
                    ),
                )
                for version in ("random_questions", "random_configurations")
            ]
            base = _baseline_curves(baseline_rows, benchmark, field)
            panels[field].append({
                "name": LABELS[benchmark],
                "stop_mean": float(np.mean(stop_fractions)),
                "deployable": deployable,
                "provisional": provisional,
                "random": random,
                "baselines": base,
            })
            for method, series in (
                ("radial_gittins_deployable", deployable),
                ("radial_gittins_provisional", provisional),
            ):
                for x, mean, spread, count in zip(*series):
                    summary.append({
                        "benchmark": LABELS[benchmark], "metric": field,
                        "method": method, "cost_fraction": x, "mean": mean,
                        "two_se": spread, "n_runs": int(count),
                    })
            for method, xs, means, spreads in base + random:
                for x, mean, spread in zip(xs, means, spreads):
                    summary.append({
                        "benchmark": LABELS[benchmark], "metric": field,
                        "method": method, "cost_fraction": x, "mean": mean,
                        "two_se": spread, "n_runs": len(SEEDS),
                    })
        del template, radial_by_seed
    for field, ylabel, filename in specs:
        write_comparison_figure(
            out_path=figures / filename,
            title="",
            panels=panels[field],
            seeds=len(SEEDS),
            seed=SEEDS[0],
            x_axis="cost",
            ylabel=ylabel,
        )
        print(f"wrote {figures / filename}", flush=True)
    _write_rows(data_dir / "all_method_gd_igd_summary.csv", summary)


def _write_hv_plot_data(data_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for row in _read_rows(data_dir / "all_method_hv_regret_summary.csv"):
        rows.append({
            "benchmark": row["benchmark"], "metric": "hypervolume_regret",
            "method": row["method"], "cost_fraction": row["budget_fraction"],
            "mean": row["mean_hv_regret"],
            "two_se": float(row["ci95_half_width"]) * 2.0 / 1.96,
            "n_runs": row["n_runs"],
        })
    baseline_rows = _read_rows(data_dir / "pareto_baselines" / "cost_trajectory.csv")
    baseline_rows.extend(_read_rows(data_dir / "qnehvi" / "cost_trajectory.csv"))
    for benchmark in BENCHMARKS:
        for method, xs, means, spreads in _baseline_curves(
            baseline_rows, benchmark, "hv_regret"
        ):
            for x, mean, spread in zip(xs, means, spreads):
                rows.append({
                    "benchmark": LABELS[benchmark], "metric": "hypervolume_regret",
                    "method": method, "cost_fraction": x, "mean": mean,
                    "two_se": spread, "n_runs": len(SEEDS),
                })
    _write_rows(data_dir / "plot_data_hv.csv", rows)


def _redraw_saved_distances(data_dir: Path, figures: Path) -> None:
    stop_rows = _read_rows(data_dir / "gittins_seed_results.csv")
    specs = (
        ("generational_distance", "Generational distance (GD)", "all_methods_20seed_gd.png"),
        ("inverted_generational_distance", "Inverted generational distance (IGD)", "all_methods_20seed_igd.png"),
    )
    for field, ylabel, filename in specs:
        grouped: dict[tuple[str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
        with (data_dir / "all_method_gd_igd_summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if row["metric"] != field:
                    continue
                grouped[(row["benchmark"], row["method"])].append((
                    float(row["cost_fraction"]), float(row["mean"]),
                    float(row["two_se"]), int(row["n_runs"]),
                ))
        panels = []
        for benchmark in BENCHMARKS:
            label = LABELS[benchmark]

            def series(method: str, *, counts: bool = False):
                method_rows = sorted(grouped[(label, method)])
                values = (
                    np.asarray([row[0] for row in method_rows]),
                    np.asarray([row[1] for row in method_rows]),
                    np.asarray([row[2] for row in method_rows]),
                )
                if counts:
                    return (*values, np.asarray([row[3] for row in method_rows]))
                return values

            stop_mean = float(np.mean([
                float(row["gittins_stop_cost_fraction"])
                for row in stop_rows if row["benchmark"].lower() == benchmark
            ]))
            panels.append({
                "name": label,
                "stop_mean": stop_mean,
                "deployable": series("radial_gittins_deployable", counts=True),
                "provisional": series("radial_gittins_provisional", counts=True),
                "random": [
                    (method, *series(method))
                    for method in ("random_questions", "random_configurations")
                ],
                "baselines": [
                    (method, *series(method))
                    for method in ("ege_sh", "ape_k", "qnehvi")
                ],
            })
        write_comparison_figure(
            out_path=figures / filename, title="", panels=panels,
            seeds=len(SEEDS), seed=SEEDS[0], x_axis="cost", ylabel=ylabel,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "hotpotqa", "mathqa", "hv", "distances", "redraw-distances", "all"),
        default="all",
    )
    args = parser.parse_args()
    data_dir = ROOT / "analysis" / "continuous_seeds_42_61" / "data"
    figures = ROOT / "analysis" / "continuous_seeds_42_61" / "figures"
    gittins_run = data_dir / "gittins_run"
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    (data_dir.parent / "run_metadata.json").write_text(
        json.dumps({"seeds": list(SEEDS), "matched": True, "excluded_seeds": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (data_dir.parent / "method_styles.json").write_text(
        json.dumps(METHOD_STYLES, indent=2) + "\n", encoding="utf-8"
    )

    if args.stage in ("prepare", "all"):
        _aggregate_ege_ape(data_dir, "pareto_baselines")
        _aggregate_ege_ape(data_dir, "pareto_baselines_10pct")
        _aggregate_qnehvi(data_dir)
        _prepare_seed_level_data(data_dir)
        if args.stage == "prepare":
            return

    if args.stage in BENCHMARKS:
        plot_frontier_grid(
            args.stage,
            data_dir,
            figures / f"{args.stage}_2x6_frontier_comparison",
        )
        return

    shutil.copy2(gittins_run / "gittins_seed_results.csv", data_dir / "gittins_seed_results.csv")
    shutil.copy2(
        gittins_run / "method_hv_regret_summary.csv",
        data_dir / "gittins_random_hv_regret_summary.csv",
    )
    (data_dir / "small_gittins_run").mkdir(exist_ok=True)
    shutil.copy2(
        gittins_run / "method_hv_regret_summary.csv",
        data_dir / "small_gittins_run" / "method_hv_regret_summary.csv",
    )
    if args.stage == "hv":
        write_hv_comparison(
            data_dir, figures / "all_methods_20seed_hv_regret.png", BENCHMARKS
        )
        _write_hv_plot_data(data_dir)
        return
    if args.stage == "distances":
        _write_distances_once(data_dir, figures)
        return
    if args.stage == "redraw-distances":
        _redraw_saved_distances(data_dir, figures)
        return

    for benchmark in BENCHMARKS:
        plot_frontier_grid(
            benchmark,
            data_dir,
            figures / f"{benchmark}_2x6_frontier_comparison",
        )
    write_hv_comparison(data_dir, figures / "all_methods_20seed_hv_regret.png", BENCHMARKS)
    _write_distances_once(data_dir, figures)


if __name__ == "__main__":
    main()
