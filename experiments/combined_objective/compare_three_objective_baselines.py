#!/usr/bin/env python3
"""3D HV/GD/IGD comparison of saved CC-Gittins variants and random baselines."""
from __future__ import annotations

import argparse
import csv
import gc
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

from experiments.combined_objective.offline_three_objective_random_search import (
    VERSIONS, latest_under_checkpoint, simulate_three_objective_random_search,
)
from experiments.combined_objective.offline_pareto_baselines import METHODS as PARETO_METHODS
from experiments.combined_objective.three_objective_metrics import load_three_objective_benchmark
from experiments.combined_objective.anonymous_metadata import artifact_reference, shareable_provenance

RESULTS = ROOT / "experiments/combined_objective/results/three_objective_20seed"
METRICS = ("hypervolume", "relative_hv_regret", "generational_distance", "inverted_generational_distance")
GITTINS_DIRECTIONS = {
    "cc_gittins": [[1., 0., 0.], [.5, .5, 0.], [0., 1., 0.], [0., 0., 1.]],
    "cc_gittins_axes": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
}
LABELS = {"cc_gittins": "CC-Gittins · Q, (Q+L)/2, L, D",
          "cc_gittins_axes": "CC-Gittins · Q, L, D", "random_questions": "Random questions",
          "random_configurations": "Random configurations",
          "ege_sh": "EGE-SH", "ege_sr": "EGE-SR", "ape_k": "APE-k", "qnehvi": "qNEHVI"}
COLORS = {"cc_gittins": "#e87500", "cc_gittins_axes": "#2977b8",
          "random_questions": "#6553a6", "random_configurations": "#007f84",
          "ege_sh": "#b44445", "ege_sr": "#d16f82", "ape_k": "#4d7f32",
          "qnehvi": "#8c5e3c"}
LINESTYLES = {"cc_gittins_axes": "--"}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compact_gittins(path: Path, input_hashes: dict, *, method: str = "cc_gittins",
                    validate_directions: bool = False) -> dict:
    saved = json.loads(path.read_text())
    if saved["config"]["input_sha256"] != input_hashes:
        raise ValueError(f"Saved Gittins inputs differ: {path}")
    run = saved["run"]
    if validate_directions:
        expected = GITTINS_DIRECTIONS[method]
        if (saved["config"].get("directions") != expected or
                run.get("parameters", {}).get("directions") != expected):
            raise ValueError(f"Saved Gittins directions differ from {method}: {path}")
        if method == "cc_gittins_axes" and any(run["radial_boundary_cache_stats"].values()):
            raise ValueError(f"Axes-only Gittins unexpectedly used radial DP: {path}")
    fields = ("cost_fraction", "cumulative_search_cost_usd", "total_evaluations", "selected_arm_indices", *METRICS)
    result = {key: run[key] for key in ("model_names", "raw_truth_vectors", "bruteforce_search_cost_usd",
              "cost_fraction", "ground_truth_hypervolume", "total_evaluations", "stop_reason")}
    result.update(seed=saved["config"]["seed"], selector=method,
                  protocol=saved["config"],
                  question_ids=run.get("question_ids"),
                  provenance=shareable_provenance(saved.get("provenance", {}), root=ROOT),
                  source_result=artifact_reference(path, root=ROOT),
                  source_sha256=saved["config"]["source_sha256"],
                  warm_start={key: run["points"][0].get(key) for key in (
                      "counts", "total_evaluations", "cumulative_search_cost_usd", "selected_arm_indices")},
                  calibration={key: run.get("parameters", {}).get(key) for key in (
                      "latency_reference_seconds", "cost_reference_usd", "raw_prior_mean",
                      "raw_prior_variance", "raw_question_noise_variance", "expected_batch_costs_usd",
                      "base_effective_pull_costs")},
                  points=[{key: p[key] for key in fields} for p in run["points"]])
    del saved, run
    gc.collect()
    return result


def aggregate_at(runs: list[dict], target: float) -> dict:
    """Aggregate latest completed units, never interpolate or drop empty HVs."""
    snapshots = []
    for run in runs:
        if run["cost_fraction"] < target and run["stop_reason"] in (
            "direction_eta_numerical_floor", "all_arms_completed", "all_cells_observed",
            "question_budget", "all_arms_exhausted",
        ):
            # A stopped method's final recommendation remains available at a
            # larger budget. Hold its output, without adding fictitious spend.
            snapshots.append({**run["points"][-1], "available": True, "stopped_before_target": True})
        else:
            snapshots.append(latest_under_checkpoint(run, target))
    row = {"target_cost_fraction": target, "n_seeds": len(runs),
           "n_available": sum(bool(p["available"]) for p in snapshots),
           "n_stopped_before_target": sum(bool(p.get("stopped_before_target")) for p in snapshots)}
    for metric in METRICS:
        values = []
        for point in snapshots:
            if point["available"]:
                value = point[metric]
            elif metric == "hypervolume":
                value = 0.0
            elif metric == "relative_hv_regret":
                value = 1.0
            else:
                value = None
            values.append(value)
        finite = [float(v) for v in values if v is not None and np.isfinite(v)]
        # Undefined empty-front distances remain undefined for the aggregate;
        # dropping those seeds would make the baseline appear too favorable.
        valid = len(finite) == len(runs)
        row[metric] = {"mean": float(np.mean(finite)) if valid else None,
                       "p10": float(np.quantile(finite, .1)) if valid else None,
                       "p90": float(np.quantile(finite, .9)) if valid else None,
                       "n_finite": len(finite)}
    fractions = [p["cost_fraction"] for p in snapshots if p["available"]]
    row["actual_cost_fraction_mean"] = float(np.mean(fractions)) if fractions else None
    return row


def baseline_runs(benchmark: str, seeds: list[int], output: Path) -> tuple[dict, dict]:
    models, questions, table, hashes = load_three_objective_benchmark(benchmark)
    source_files = [ROOT / "experiments/combined_objective" / name for name in (
        "offline_three_objective_random_search.py", "three_objective_metrics.py")]
    config = {"seeds": seeds, "input_sha256": hashes,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    cache_path = output / benchmark / "baselines.json"
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["config"] != config:
            raise ValueError(f"Baseline cache inputs/protocol changed: {cache_path}")
        print(f"Reusing {cache_path}", flush=True)
        return saved["groups"], hashes
    groups = {}
    started = time.perf_counter()
    for version in VERSIONS:
        runs = []
        for baseline_seed in seeds:
            run = simulate_three_objective_random_search(models, questions, table,
                                                        version=version, seed=baseline_seed)
            runs.append(run)
        groups[version] = runs
        print(f"{benchmark}: {version}, {len(seeds)} seeds complete ({time.perf_counter()-started:.1f}s)", flush=True)
    dump(cache_path, {"config": config, "groups": groups, "wall_seconds": time.perf_counter()-started})
    return groups, hashes


def pareto_runs(benchmark: str, seeds: list[int], root: Path,
                input_hashes: dict[str, str],
                methods: tuple[str, ...] | None = None) -> dict[str, list[dict]]:
    """Load the optional Q/L/D EGE, APE, and qNEHVI runs for this benchmark."""
    metric_path = ROOT / "experiments/combined_objective/three_objective_metrics.py"
    metric_key = str(metric_path.relative_to(ROOT))
    metric_sha = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    groups: dict[str, list[dict]] = {}
    for method in PARETO_METHODS if methods is None else methods:
        runs = []
        protocol_without_seed = None
        for seed in seeds:
            path = root / f"seed_{seed}" / benchmark / method / "result.json"
            saved = json.loads(path.read_text())
            config, run = saved["config"], saved["run"]
            if config["input_sha256"] != input_hashes:
                raise ValueError(f"Saved Pareto inputs differ: {path}")
            if config["source_sha256"].get(metric_key) != metric_sha:
                raise ValueError(f"Saved Pareto scoring code differs: {path}")
            if (config["method"] != method or config["seed"] != seed or
                    config["objective_order"] != ["Q", "L", "D"] or
                    run["selector"] != method or run["seed"] != seed):
                raise ValueError(f"Saved Pareto method, seed, or objectives differ: {path}")
            current_protocol = {key: value for key, value in config.items() if key != "seed"}
            if protocol_without_seed is not None and current_protocol != protocol_without_seed:
                raise ValueError(f"Saved Pareto protocol differs across seeds: {path}")
            protocol_without_seed = current_protocol
            if not run["points"] or run["selected_arm_indices"] != run["points"][-1]["selected_arm_indices"]:
                raise ValueError(f"Saved Pareto trajectory lacks its final recommendation: {path}")
            compact = {key: run[key] for key in (
                "selector", "seed", "model_names", "question_ids", "raw_truth_vectors",
                "bruteforce_search_cost_usd", "cost_fraction", "ground_truth_hypervolume",
                "total_evaluations", "stop_reason", "points",
            )}
            compact.update(protocol=config, source_result=artifact_reference(path, root=ROOT),
                           source_sha256=config["source_sha256"])
            runs.append(compact)
        groups[method] = runs
    return groups


def prepare_benchmark(benchmark: str, *, gittins_root: Path, axes_root: Path | None = None,
                      pareto_root: Path | None = None,
                      pareto_methods: tuple[str, ...] | None = None,
                      seeds: list[int], output: Path) -> dict:
    baseline_groups, hashes = baseline_runs(benchmark, seeds, output)
    gittins_groups = {
        method: [compact_gittins(root / f"seed_{seed}" / benchmark / "result.json", hashes,
                                method=method, validate_directions=True) for seed in seeds]
        for method, root in (("cc_gittins", gittins_root), ("cc_gittins_axes", axes_root)) if root is not None
    }
    cc_runs = gittins_groups["cc_gittins"]
    groups = {**gittins_groups, **baseline_groups}
    if pareto_root is not None:
        groups.update(pareto_runs(benchmark, seeds, pareto_root, hashes, pareto_methods))
    cc = cc_runs[0]
    metric_path = ROOT / "experiments/combined_objective/three_objective_metrics.py"
    metric_sha = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    for runs in gittins_groups.values():
        for requested_seed, reference, run in zip(seeds, cc_runs, runs):
            if run["seed"] != requested_seed:
                raise ValueError("Saved CC-Gittins seed differs from its requested result path")
            if run["source_sha256"] != cc["source_sha256"]:
                raise ValueError("CC-Gittins seeds/variants have different source fingerprints")
            if run["source_sha256"].get(str(metric_path.relative_to(ROOT))) != metric_sha:
                raise ValueError("Saved CC-Gittins metrics differ from baseline scoring code")
            paired_protocols = [{key: value for key, value in item["protocol"].items()
                                 if key not in ("directions", "direction_labels", "variant")}
                                for item in (reference, run)]
            if paired_protocols[0] != paired_protocols[1]:
                raise ValueError("Gittins variants differ beyond acquisition directions")
            if run["calibration"] != reference["calibration"]:
                raise ValueError("Paired Gittins variants have different warm calibration")
            if run["warm_start"] != reference["warm_start"]:
                raise ValueError("Paired Gittins variants have different warm observations or recommendations")
            if run["question_ids"] != reference["question_ids"]:
                raise ValueError("Paired Gittins variants have different question order")
    for runs in groups.values():
        if [run["seed"] for run in runs] != seeds:
            raise ValueError("Method seeds differ from the requested matched seed list")
        for run in runs:
            if run["model_names"] != cc["model_names"]:
                raise ValueError("Saved model order differs")
            np.testing.assert_allclose(run["raw_truth_vectors"], cc["raw_truth_vectors"], rtol=1e-12, atol=1e-14)
            if not np.isclose(run["bruteforce_search_cost_usd"], cc["bruteforce_search_cost_usd"], rtol=1e-12):
                raise ValueError("Baseline and Gittins exhaustive costs differ")
            if not np.isclose(run["ground_truth_hypervolume"], cc["ground_truth_hypervolume"], rtol=1e-12):
                raise ValueError("Baseline and Gittins three-objective scoring differ")
    grid = np.arange(1, 161, dtype=float) / 200.0  # Shared 0.5%..80% USD budget grid.
    curves = {method: [aggregate_at(runs, float(x)) for x in grid] for method, runs in groups.items()}
    checkpoints = {method: [aggregate_at(runs, x) for x in (.05, .10, .20, .30, .50)]
                   for method, runs in groups.items()}
    calibration_ranges = {}
    for key in ("latency_reference_seconds", "cost_reference_usd"):
        values = np.asarray([run["calibration"][key] for run in cc_runs], dtype=float)
        calibration_ranges[key] = {"min": float(values.min()), "median": float(np.median(values)),
                                   "max": float(values.max()), "by_seed": dict(zip(map(str, seeds), values.tolist()))}
    payload = {"benchmark": benchmark, "gittins_seeds": seeds, "baseline_seeds": seeds,
               "input_sha256": hashes, "ground_truth_hypervolume": cc["ground_truth_hypervolume"],
               "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "baseline_cache_sha256": hashlib.sha256((output / benchmark / "baselines.json").read_bytes()).hexdigest(),
               "metric_source_sha256": metric_sha,
               "metric_definition": "3D Q, reciprocal mean L, reciprocal mean USD; GD/IGD on truth-reevaluated nondominated returned subset; mean Euclidean distances",
               "gittins_directions": {method: GITTINS_DIRECTIONS[method] for method in gittins_groups},
               "paired_warm_calibration_verified": len(gittins_groups) > 1,
               "recommendations": {**{method: "finite LCB beta=1" for method in gittins_groups},
                                   "random_questions": "observed empirical 3D Pareto",
                                   "random_configurations": "empirical 3D Pareto over completely evaluated configurations",
                                   **{method: "observed empirical 3D Pareto" for method in PARETO_METHODS if method in groups}},
               "budget_alignment": "latest completed unit at/below target actual USD; hold final output after policy stop at its actual spend; no interpolation or partial question/configuration units",
               "uncertainty_band": "10th to 90th seed percentiles; not a confidence interval",
               "warm_calibration_ranges": calibration_ranges,
               "groups": groups, "curves": curves, "checkpoints": checkpoints}
    dump(output / benchmark / "comparison.json", payload)
    return payload


def plot(payloads: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    metrics = (("hypervolume", "HV / full-data HV ↑"), ("generational_distance", "GD ↓"),
               ("inverted_generational_distance", "IGD ↓"))
    fig, axes = plt.subplots(len(payloads), 3, figsize=(14, 4 * len(payloads)), squeeze=False)
    for row, p in enumerate(payloads):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for method, points in p["curves"].items():
                scale = p["ground_truth_hypervolume"] if metric == "hypervolume" else 1.
                x = np.asarray([point["target_cost_fraction"] for point in points])
                arrays = [np.asarray([np.nan if point[metric][key] is None else point[metric][key] / scale
                                      for point in points]) for key in ("mean", "p10", "p90")]
                ax.step(x, arrays[0], where="post", color=COLORS[method], label=LABELS[method],
                        linestyle=LINESTYLES.get(method, "-"), linewidth=1.8)
                ax.fill_between(x, arrays[1], arrays[2], color=COLORS[method], alpha=.13, step="post")
            ax.set(title=f"{p['benchmark']} · {label}", xlabel="Available budget / exhaustive USD", xlim=(0, .5))
            ax.xaxis.set_major_formatter(PercentFormatter(1.))
            ax.grid(alpha=.2)
            if metric == "hypervolume":
                ax.set_ylim(0, 1.025)
            else:
                ax.set_yscale("symlog", linthresh=1e-4)
                ax.set_ylim(bottom=0)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Three objectives · {len(payloads[0]['baseline_seeds'])} seeds per method · bands: 10–90% seed range", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "hv_gd_igd.png")
    fig.savefig(output / "hv_gd_igd.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(payloads), figsize=(7 * len(payloads), 4), squeeze=False)
    for ax, p in zip(axes[0], payloads):
        for method, points in p["curves"].items():
            x = [point["target_cost_fraction"] for point in points]
            values = [[point["relative_hv_regret"][key] for point in points] for key in ("mean", "p10", "p90")]
            ax.step(x, values[0], where="post", color=COLORS[method], label=LABELS[method],
                    linestyle=LINESTYLES.get(method, "-"), linewidth=1.8)
            ax.fill_between(x, values[1], values[2], color=COLORS[method], alpha=.13, step="post")
        ax.set(title=p["benchmark"], xlabel="Available budget / exhaustive USD", ylabel="Relative HV regret ↓", xlim=(0, .5))
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "hv_regret_detail.png")
    plt.close(fig)


def export(payloads: list[dict], output: Path) -> None:
    rows = []
    for p in payloads:
        for method, checkpoints in p["checkpoints"].items():
            for point in checkpoints:
                rows.append({"benchmark": p["benchmark"], "method": method,
                             **{key: value for key, value in point.items() if key not in METRICS},
                             **{f"{metric}_{stat}": value for metric in METRICS for stat, value in point[metric].items()}})
    dump(output / "checkpoints.json", rows)
    with (output / "checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Three-objective Gittins variants and baselines", "",
              "All methods use seeds " + ", ".join(map(str, payloads[0]["baseline_seeds"])) + ".",
              "Random questions evaluates a shared question prefix on every configuration. Random configurations fully evaluates each chosen configuration. Both recommend empirical 3D Pareto sets; CC-Gittins retains finite-LCB recommendations.",
              "When included, EGE-SH/SR, APE-k, and qNEHVI use their sampling rules with empirical 3D Pareto recommendations. Their checkpoints are recomputed in the same 3D evaluation space after replay; oracle diagnostics do not feed acquisition.",
              "Actual USD budgets use the latest complete unit at/below the target, with no interpolation. If a policy stops early, its final output remains available at larger budgets, without adding spend. This is a comparison of the full methods, including their recommendation rules.", "",
              "| Dataset | Budget | Method | HV regret | GD | IGD |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        if row["target_cost_fraction"] not in (.1, .2):
            continue
        def fmt(value):
            return "undefined" if value is None else f"{value:.5f}"
        report.append(f"| {row['benchmark']} | {row['target_cost_fraction']:.0%} | {LABELS[row['method']]} | {row['relative_hv_regret_mean']:.4%} | {fmt(row['generational_distance_mean'])} | {fmt(row['inverted_generational_distance_mean'])} |")
    report.extend(["", f"Values are means across the same {len(payloads[0]['baseline_seeds'])} seeds for every method. Plot bands show 10th–90th seed percentiles, not confidence intervals.",
                   "", "All methods use the same 3D metric coordinates: Q, R_L/(R_L+mean_L), R_D/(R_D+mean_D), with shared full-data positive median scales for offline scoring only. HV uses origin reference. GD/IGD follow the repository convention: recompute the nondominated subset of returned configurations in true evaluation space, then use mean nearest-point Euclidean distance. GD can be zero for an incomplete subset of the true front; IGD measures missing coverage.",
                   "", "The orange CC-Gittins uses Q, (Q+L)/2, L, D. When included, the blue dashed variant uses only Q, L, D. Paired variants share warm observations and calibration, question orders, USD penalties, batch sizes, eta schedules, DP grid, and recommendation rules; only the direction list changes. Consequently the round-robin scheduler gives each axis one in three visits instead of one in four.",
                   "", "(Q+L)/2 uses the existing two-dimensional radial DP on Q and affine latency reward. Its utility is the direction-scaled minimum, not the arithmetic average. The three pure axes use scalar DP; the axes-only variant never invokes the radial DP. USD continuation penalties apply on every direction.",
                   "", "Latency and cost calibration use each configuration's four warm questions: a common prior mean, across-configuration variance of warm means, and mean within-configuration observation variance. The scale is the largest configuration warm mean, frozen after warm-up. The authoritative latency posterior remains in seconds; the DP sees Q and 1-L/R_L, with variance/noise divided by R_L squared. The affine reward is not clipped. Balanced direction weights therefore depend on this warm-derived scale, rather than expressing an invariant practical tradeoff.",
                   "", "| Dataset | Warm latency scale min (s) | Median (s) | Max (s) |", "|---|---:|---:|---:|"])
    for p in payloads:
        scale = p["warm_calibration_ranges"]["latency_reference_seconds"]
        report.append(f"| {p['benchmark']} | {scale['min']:.2f} | {scale['median']:.2f} | {scale['max']:.2f} |")
    report.extend(["", "![HV, GD, IGD](hv_gd_igd.png)", "", "![HV regret detail](hv_regret_detail.png)", "",
                   "[All checkpoints](checkpoints.csv)"])
    (output / "report.md").write_text("\n".join(report) + "\n")
    plot(payloads, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=("mathqa", "hotpotqa"), default=["mathqa", "hotpotqa"])
    parser.add_argument("--gittins-root", type=Path, default=RESULTS)
    parser.add_argument("--axes-root", type=Path, default=None,
                        help="Include Q,L,D-only Gittins, e.g. results/three_objective_axes_20seed")
    parser.add_argument("--pareto-root", type=Path, default=None,
                        help="Include saved Q,L,D EGE-SH/SR, APE-k, qNEHVI runs")
    parser.add_argument("--pareto-methods", nargs="+", choices=PARETO_METHODS,
                        default=list(PARETO_METHODS),
                        help="Which methods to load from --pareto-root (default: all four)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--outdir", type=Path, default=RESULTS / "comparison")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("baseline seeds must be unique")
    if len(set(args.pareto_methods)) != len(args.pareto_methods):
        parser.error("Pareto methods must be unique")
    if args.baselines_only:
        for benchmark in args.benchmarks:
            baseline_runs(benchmark, args.seeds, args.outdir)
        return
    if args.plot_only:
        payloads = [json.loads((args.outdir / b / "comparison.json").read_text()) for b in args.benchmarks]
    else:
        payloads = [prepare_benchmark(b, gittins_root=args.gittins_root, axes_root=args.axes_root,
                                      pareto_root=args.pareto_root,
                                      pareto_methods=tuple(args.pareto_methods),
                                      seeds=args.seeds, output=args.outdir) for b in args.benchmarks]
    export(payloads, args.outdir)
    print(f"Comparison saved: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
