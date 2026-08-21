#!/usr/bin/env python3
"""Read-only correctness audit for saved multi-objective search results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from offline_radial_gittins import hypervolume_2d  # noqa: E402
from offline_multiobjective_random_search import (  # noqa: E402
    normalized_truth_vectors,
    pareto_min_cost_indices,
)
from offline_selector_sim_v2 import load_pickle  # noqa: E402


DATASETS = {
    "hotpotqa": ROOT / "experiments/results/cache_db_results/hotpotqa_lookup.pkl",
    "mathqa": ROOT / "experiments/results/cache_db_results/mathqa_lookup.pkl",
}
KEY_BUDGETS = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_components(dataset: str, config: str) -> tuple[str, str, str]:
    fields = {}
    for item in config.split(" + "):
        key, _, value = item.partition("=")
        fields[key.strip()] = value.strip()
    if dataset == "hotpotqa":
        return fields.get("planner", ""), fields.get("solver", ""), ""
    return fields.get("answer", ""), fields.get("critic", ""), ""


def dominates(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(
        a[0] >= b[0] - tol
        and a[1] <= b[1] + tol
        and (a[0] > b[0] + tol or a[1] < b[1] - tol)
    )


def nearest_front_distance(point: np.ndarray, front: np.ndarray, ranges: np.ndarray) -> float:
    normalized = (front - point) / ranges
    return float(np.min(np.sqrt(np.sum(normalized * normalized, axis=1))))


def load_truth(dataset: str, pickle_path: Path) -> dict:
    models, tasks, table = load_pickle(str(pickle_path))
    common = [q for q in tasks if all(q in table.get(model, {}) for model in models)]
    raw = np.asarray(
        [
            [
                np.mean([table[m][q].score for q in common]),
                np.mean([table[m][q].cost for q in common]),
            ]
            for m in models
        ],
        dtype=float,
    )
    front_idx = pareto_min_cost_indices(raw)
    positive = raw[:, 1][raw[:, 1] > 0]
    cost_ref = float(np.median(positive))
    normalized = normalized_truth_vectors(raw, cost_ref)
    gt_hv = float(hypervolume_2d(normalized[front_idx]))
    model_index = {m: i for i, m in enumerate(models)}
    return {
        "dataset": dataset,
        "models": models,
        "tasks": tasks,
        "table": table,
        "common": common,
        "raw": raw,
        "front_idx": front_idx,
        "front_set": set(front_idx),
        "normalized": normalized,
        "gt_hv": gt_hv,
        "cost_ref": cost_ref,
        "model_index": model_index,
    }


def audit_missing(ctx: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    cells = []
    tasks = ctx["tasks"]
    n_tasks = len(tasks)
    for model in ctx["models"]:
        observed = set(ctx["table"].get(model, {}))
        missing = [q for q in tasks if q not in observed]
        suffix = 0
        for q in reversed(tasks):
            if q in observed:
                break
            suffix += 1
        c1, c2, c3 = parse_components(ctx["dataset"], model)
        observed_costs = [ctx["table"][model][q].cost for q in observed]
        rows.append(
            {
                "dataset": ctx["dataset"],
                "configuration": model,
                "component_1": c1,
                "component_2": c2,
                "component_3": c3,
                "total_tasks": n_tasks,
                "observed_tasks": len(observed),
                "missing_tasks": len(missing),
                "missing_rate": len(missing) / n_tasks,
                "missing_task_ids": json.dumps(missing),
                "trailing_missing_count": suffix,
                "missing_concentrated_at_end": bool(missing and suffix == len(missing)),
                "observed_mean_cost_usd": float(np.mean(observed_costs)) if observed_costs else math.nan,
                "inferred_missing_reason": (
                    "experiment_stopped_early_or_interrupted_likely"
                    if missing and suffix == len(missing)
                    else "unknown_no_failure_metadata"
                ),
                "reason_evidence": (
                    "all missing task IDs form a trailing suffix; lookup has no explicit failure/status field"
                    if missing and suffix == len(missing)
                    else "lookup table stores successful cells only; no timeout/status/module-invocation field"
                ),
            }
        )
        for q in missing:
            cells.append(
                {
                    "dataset": ctx["dataset"],
                    "configuration": model,
                    "task_id": q,
                    "component_1": c1,
                    "component_2": c2,
                    "component_3": c3,
                    "is_trailing_task_for_configuration": q in tasks[-suffix:] if suffix else False,
                    "inferred_missing_reason": (
                        "experiment_stopped_early_or_interrupted_likely"
                        if suffix == len(missing) else "unknown_no_failure_metadata"
                    ),
                }
            )
    return rows, cells


def audit_saved_results(ctx: dict, result_csv: Path) -> tuple[list[dict], list[dict], list[dict]]:
    with result_csv.open(encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    metrics = []
    returned = []
    violations = []
    raw = ctx["raw"]
    front = raw[ctx["front_idx"]]
    ranges = np.maximum(np.ptp(raw, axis=0), 1e-12)
    for record in saved:
        method = record["version"]
        seed = int(record["seed"])
        budget = float(record["budget_fraction"])
        names = list(json.loads(record["selected_models"]))
        indices = [ctx["model_index"][name] for name in names]
        selected_set = set(indices)
        true_positive = len(selected_set & ctx["front_set"])
        precision = true_positive / len(indices) if indices else 1.0
        recall = true_positive / len(ctx["front_idx"]) if ctx["front_idx"] else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        hv = float(hypervolume_2d(ctx["normalized"][indices])) if indices else 0.0
        distances = [nearest_front_distance(raw[i], front, ranges) for i in indices]
        violation_count = 0
        for i, name in zip(indices, names):
            dominators = [j for j in range(len(raw)) if dominates(raw[j], raw[i])]
            is_violation = bool(dominators)
            violation_count += int(is_violation)
            best_dom = min(dominators, key=lambda j: (raw[j, 1], -raw[j, 0])) if dominators else None
            c1, c2, c3 = parse_components(ctx["dataset"], name)
            row = {
                "dataset": ctx["dataset"], "method": method, "seed": seed,
                "budget": budget, "returned_config": name,
                "planner_or_answer": c1, "executor_or_critic": c2, "verifier": c3,
                "accuracy": raw[i, 0], "cost_usd": raw[i, 1],
                "true_pareto": i in ctx["front_set"],
                "dominated": is_violation,
                "distance_to_true_frontier": distances[-1],
                "dominating_config": ctx["models"][best_dom] if best_dom is not None else "",
                "dominating_cost_usd": raw[best_dom, 1] if best_dom is not None else "",
                "dominating_accuracy": raw[best_dom, 0] if best_dom is not None else "",
            }
            if budget in KEY_BUDGETS:
                returned.append(row)
            if is_violation:
                violations.append(row)
        metrics.append(
            {
                "dataset": ctx["dataset"], "method": method, "seed": seed,
                "budget": budget, "returned_count": len(indices),
                "true_frontier_size": len(ctx["front_idx"]),
                "recovered_true_points": true_positive,
                "pareto_precision": precision, "pareto_recall": recall, "pareto_f1": f1,
                "hypervolume": hv, "ground_truth_hypervolume": ctx["gt_hv"],
                "hypervolume_regret": max(0.0, ctx["gt_hv"] - hv),
                "hypervolume_fraction": hv / ctx["gt_hv"] if ctx["gt_hv"] else 1.0,
                "mean_distance_to_true_frontier": float(np.mean(distances)) if distances else math.nan,
                "max_distance_to_true_frontier": float(np.max(distances)) if distances else math.nan,
                "dominance_violation_count": violation_count,
                "dominance_violation_rate": violation_count / len(indices) if indices else 0.0,
            }
        )
    return metrics, returned, violations


def ground_truth_rows(ctx: dict) -> list[dict]:
    ordered = sorted(ctx["front_idx"], key=lambda i: ctx["raw"][i, 1])
    rows = []
    previous = None
    for rank, i in enumerate(ordered, 1):
        acc, cost = ctx["raw"][i]
        c1, c2, c3 = parse_components(ctx["dataset"], ctx["models"][i])
        dc = cost - previous[1] if previous else math.nan
        da = acc - previous[0] if previous else math.nan
        rows.append(
            {
                "dataset": ctx["dataset"], "frontier_rank_by_cost": rank,
                "configuration": ctx["models"][i], "component_1": c1,
                "component_2": c2, "component_3": c3,
                "accuracy": acc, "cost_usd": cost,
                "delta_accuracy_from_previous": da,
                "delta_cost_from_previous": dc,
                "marginal_accuracy_per_usd": da / dc if previous and dc > 0 else math.nan,
            }
        )
        previous = (acc, cost)
    return rows


def component_effect_rows(ctx: dict) -> list[dict]:
    rows = []
    components = [parse_components(ctx["dataset"], m) for m in ctx["models"]]
    labels = ("planner_or_answer", "executor_or_critic", "verifier")
    for changed in (0, 1):
        other = 1 - changed
        groups = defaultdict(list)
        for i, parts in enumerate(components):
            groups[parts[other]].append(i)
        for fixed_value, indices in groups.items():
            for pos, i in enumerate(indices):
                for j in indices[pos + 1:]:
                    if components[i][changed] == components[j][changed]:
                        continue
                    for src, dst in ((i, j), (j, i)):
                        da = ctx["raw"][dst, 0] - ctx["raw"][src, 0]
                        dc = ctx["raw"][dst, 1] - ctx["raw"][src, 1]
                        rows.append(
                            {
                                "dataset": ctx["dataset"], "changed_component": labels[changed],
                                "fixed_component": labels[other], "fixed_model": fixed_value,
                                "from_model": components[src][changed], "to_model": components[dst][changed],
                                "from_configuration": ctx["models"][src], "to_configuration": ctx["models"][dst],
                                "delta_accuracy": da, "delta_cost_usd": dc,
                                "marginal_accuracy_per_usd": da / dc if abs(dc) > 1e-15 else math.nan,
                            }
                        )
    return rows


def aggregate_curves(metrics: list[dict], outdir: Path) -> None:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["dataset"], row["method"], row["budget"])].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = {"random_configurations": "#7b61a8", "random_questions": "#2a9d8f"}
    for col, dataset in enumerate(DATASETS):
        for method in colors:
            keys = sorted(k for k in grouped if k[0] == dataset and k[1] == method)
            xs = np.array([k[2] for k in keys])
            recall = np.array([np.mean([r["pareto_recall"] for r in grouped[k]]) for k in keys])
            precision = np.array([np.mean([r["pareto_precision"] for r in grouped[k]]) for k in keys])
            regret = np.array([np.mean([r["hypervolume_regret"] for r in grouped[k]]) for k in keys])
            axes[0, col].plot(xs, regret, marker="o", color=colors[method], label=method)
            axes[1, col].plot(xs, recall, marker="o", color=colors[method], label=f"{method} recall")
            axes[1, col].plot(xs, precision, marker="x", linestyle="--", color=colors[method], label=f"{method} precision")
        axes[0, col].set_title(dataset)
        axes[0, col].set_ylabel("Mean HV regret")
        axes[1, col].set_ylabel("Mean precision / recall")
        axes[1, col].set_xlabel("Evaluation budget fraction")
        axes[1, col].set_ylim(-0.02, 1.02)
        for ax in axes[:, col]: ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "frontier_recovery_and_hv.png", dpi=170)
    plt.close(fig)


def landscape_plot(contexts: dict[str, dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (dataset, ctx) in zip(axes, contexts.items()):
        raw = ctx["raw"]
        ordered = sorted(ctx["front_idx"], key=lambda i: raw[i, 1])
        ax.scatter(raw[:, 1], raw[:, 0], alpha=0.28, s=22, label="all configurations")
        ax.plot(raw[ordered, 1], raw[ordered, 0], "o-", color="#1f4e79", label="ground-truth frontier")
        ax.set(title=dataset, xlabel="Mean deployment cost (USD)", ylabel="Mean accuracy")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "full_landscape_ground_truth_frontier.png", dpi=170); plt.close(fig)


def missing_plot(missing_rows: list[dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, dataset in zip(axes, DATASETS):
        rows = [r for r in missing_rows if r["dataset"] == dataset]
        ax.scatter([r["observed_mean_cost_usd"] for r in rows], [r["missing_rate"] for r in rows], alpha=.65)
        ax.set(title=dataset, xlabel="Observed mean cost (USD)", ylabel="Missing rate")
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(outdir / "missing_rate_vs_configuration_cost.png", dpi=170); plt.close(fig)


def missing_heatmap(contexts: dict[str, dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (dataset, ctx) in zip(axes, contexts.items()):
        present = np.asarray(
            [[q in ctx["table"].get(model, {}) for q in ctx["tasks"]] for model in ctx["models"]],
            dtype=float,
        )
        ax.imshow(present, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax.set(title=f"{dataset}: observed-cell map", xlabel="Task index", ylabel="Configuration index")
    fig.tight_layout(); fig.savefig(outdir / "missing_cell_heatmaps.png", dpi=170); plt.close(fig)


def returned_overlay(contexts: dict[str, dict], returned: list[dict], outdir: Path) -> None:
    budgets = (0.2, 0.4, 0.6, 1.0)
    for dataset, ctx in contexts.items():
        raw = ctx["raw"]
        ordered = sorted(ctx["front_idx"], key=lambda i: raw[i, 1])
        for method in ("random_configurations", "random_questions"):
            fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharex=True, sharey=True)
            for ax, budget in zip(axes, budgets):
                rows = [r for r in returned if r["dataset"] == dataset and r["method"] == method and r["seed"] == 42 and abs(r["budget"] - budget) < 1e-12]
                ax.scatter(raw[:, 1], raw[:, 0], color="#9aa7b6", alpha=.22, s=18)
                ax.plot(raw[ordered, 1], raw[ordered, 0], "o-", color="#5f6b7a", linewidth=1, markersize=3)
                if rows:
                    ax.plot([float(r["cost_usd"]) for r in rows], [float(r["accuracy"]) for r in rows], "s-", color="#1f4e79", markersize=4)
                ax.set_title(f"{budget:.0%}"); ax.grid(alpha=.2)
            axes[0].set_ylabel("Accuracy")
            for ax in axes: ax.set_xlabel("Cost (USD)")
            fig.suptitle(f"{dataset} — {method}, seed=42")
            fig.tight_layout(); fig.savefig(outdir / f"{dataset}_{method}_returned_overlays.png", dpi=170); plt.close(fig)


def component_plot(contexts: dict[str, dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (dataset, ctx) in zip(axes, contexts.items()):
        grouped = defaultdict(list)
        for i, model in enumerate(ctx["models"]):
            c1, _, _ = parse_components(dataset, model)
            grouped[c1].append(ctx["raw"][i])
        for component, values in grouped.items():
            point = np.mean(np.asarray(values), axis=0)
            ax.scatter(point[1], point[0], s=35)
            ax.annotate(component, (point[1], point[0]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set(title=f"{dataset}: planner/answer average effect", xlabel="Mean cost across counterparts", ylabel="Mean accuracy across counterparts")
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(outdir / "component_cost_benefit.png", dpi=170); plt.close(fig)


def sample_efficiency(metrics: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["dataset"], row["method"], row["seed"])].append(row)
    rows = []
    targets = {
        "budget_at_80pct_recall": ("pareto_recall", .8),
        "budget_at_90pct_recall": ("pareto_recall", .9),
        "budget_at_95pct_recall": ("pareto_recall", .95),
        "budget_at_90pct_hypervolume": ("hypervolume_fraction", .9),
        "budget_at_95pct_hypervolume": ("hypervolume_fraction", .95),
    }
    for (dataset, method, seed), values in groups.items():
        values.sort(key=lambda r: r["budget"])
        row = {"dataset": dataset, "method": method, "seed": seed}
        for name, (metric, threshold) in targets.items():
            reached = [r["budget"] for r in values if r[metric] >= threshold]
            row[name] = min(reached) if reached else "not_reached"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "analysis/correctness_audit")
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    diagnostics = outdir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    contexts = {ds: load_truth(ds, path) for ds, path in DATASETS.items()}

    all_metrics, all_returned, all_violations = [], [], []
    all_missing, missing_cells, gt_rows, effects = [], [], [], []
    for dataset, ctx in contexts.items():
        missing, cells = audit_missing(ctx)
        all_missing.extend(missing); missing_cells.extend(cells)
        gt_rows.extend(ground_truth_rows(ctx)); effects.extend(component_effect_rows(ctx))
        result_path = ROOT / f"analysis/random_search_pareto/{dataset}/multi_seed_results.csv"
        metrics, returned, violations = audit_saved_results(ctx, result_path)
        all_metrics.extend(metrics); all_returned.extend(returned); all_violations.extend(violations)

    write_csv(outdir / "MISSING_CELL_AUDIT.csv", all_missing)
    write_csv(outdir / "MISSING_CELL_DETAILS.csv", missing_cells)
    write_csv(outdir / "RETURNED_CONFIGS.csv", all_returned)
    write_csv(outdir / "FRONTIER_METRICS.csv", all_metrics)
    write_csv(outdir / "DOMINANCE_VIOLATIONS.csv", all_violations)
    write_csv(outdir / "GROUND_TRUTH_FRONTIERS.csv", gt_rows)
    write_csv(outdir / "COMPONENT_PAIRWISE_EFFECTS.csv", effects)
    efficiency = sample_efficiency(all_metrics)
    write_csv(outdir / "SAMPLE_EFFICIENCY_BY_SEED.csv", efficiency)
    aggregate_curves(all_metrics, diagnostics)
    landscape_plot(contexts, diagnostics)
    missing_plot(all_missing, diagnostics)
    missing_heatmap(contexts, diagnostics)
    returned_overlay(contexts, all_returned, diagnostics)
    component_plot(contexts, diagnostics)

    summary = {
        "scope": {
            "methods": sorted({r["method"] for r in all_metrics}),
            "datasets": sorted(contexts),
            "seeds": sorted({r["seed"] for r in all_metrics}),
            "metric_records": len(all_metrics),
            "returned_point_records_at_key_budgets": len(all_returned),
        },
        "ground_truth": {
            ds: {
                "configurations": len(ctx["models"]), "total_tasks": len(ctx["tasks"]),
                "common_complete_tasks": len(ctx["common"]), "frontier_size": len(ctx["front_idx"]),
            } for ds, ctx in contexts.items()
        },
        "dominance": {
            "violation_count": len(all_violations),
            "returned_point_count_all_budgets": sum(r["returned_count"] for r in all_metrics),
            "violation_rate": len(all_violations) / sum(r["returned_count"] for r in all_metrics),
            "affected_seed_count": len({r["seed"] for r in all_violations}),
            "affected_method_seed_count": len({(r["method"], r["seed"]) for r in all_violations}),
            "affected_dataset_count": len({r["dataset"] for r in all_violations}),
        },
        "missing": {
            ds: {
                "missing_cells": sum(r["missing_tasks"] for r in all_missing if r["dataset"] == ds),
                "affected_configurations": sum(r["missing_tasks"] > 0 for r in all_missing if r["dataset"] == ds),
            } for ds in contexts
        },
    }
    (outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
