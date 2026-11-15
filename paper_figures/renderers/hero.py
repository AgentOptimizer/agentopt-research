from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, LogLocator
from common import DATA, OUT


from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import FixedLocator, NullLocator
HERE=DATA/'hero'
args=SimpleNamespace(summary=HERE/'case_summary.json',title='BIRD Dev - Monetary Savings',
    stem='bird_dev_cost_frontier_20seed_10pct',frontier_panel=True,seed=None,aggregate_seeds=True,
    orange=True,budget_fraction=.1,show_quality=True,hide_savings=False,budget_as_savings_label=True,font_scale=1.2)
data = json.loads(args.summary.read_text())
full = data["exhaustive_cost_usd"]
ours = data["cc_gittins_mean_cost_usd"]
low, high = data["cc_gittins_mean_cost_95pct_bootstrap_ci_usd"]
saving = data["cost_savings_percent"]
if args.frontier_panel:
    dataset = data["dataset"]
    truth = json.loads((HERE / "truth_reference.json").read_text(encoding="utf-8"))[dataset]
    runs = json.loads((HERE / "high_cost_runs_compact.json").read_text(encoding="utf-8"))[dataset]["runs"]
    budget = None if args.budget_fraction is None else full * args.budget_fraction
    checkpoint_rows = {}
    if budget is not None:
        with (HERE / "frontier_checkpoint_data.csv").open(newline="", encoding="utf-8") as f:
            checkpoint_rows = {int(row["seed"]): row for row in csv.DictReader(f)
                               if row["dataset"] == dataset and
                               np.isclose(float(row["target_cost_fraction"]), args.budget_fraction)}
        assert len(runs) == 20 and len(checkpoint_rows) in (0, 20)
    def recorded_point(seed_run):
        if budget is None:
            return seed_run["points"][-1]
        # Natural terminal recommendations are carried to later budget checkpoints.
        assert all(a["cost_usd"] <= b["cost_usd"] for a, b in zip(seed_run["points"], seed_run["points"][1:]))
        point = [p for p in seed_run["points"] if p["cost_usd"] <= budget][-1]
        if checkpoint_rows:
            row = checkpoint_rows[seed_run["seed"]]
            assert float(row["snapshot_cost_usd"]) <= budget < float(row["membership_end_cost_usd"])
            assert set(point["selected_arm_indices"]) == set(json.loads(row["selected_arm_indices"]))
        return point
    median_cost = float(np.median([r["actual_total_search_cost_usd"] for r in runs]))
    run = (next(r for r in runs if r["seed"] == args.seed) if args.seed is not None else
           min(runs, key=lambda r: (abs(r["actual_total_search_cost_usd"]-median_cost), r["seed"])))
    raw = np.asarray(truth["raw_truth"], dtype=float)
    front_ids = set(truth["true_front_indices"])
    final = recorded_point(run)
    selected_ids = set(final["selected_arm_indices"])
    assert raw.shape == (truth["n_configurations"], 2)
    assert np.all(np.isfinite(raw)) and np.all(raw[:, 1] > 0)
    # Independently verify maximize-accuracy/minimize-cost truth membership.
    computed_front = {i for i, (q, c) in enumerate(raw)
                      if not np.any((raw[:, 0] >= q) & (raw[:, 1] <= c) &
                                    ((raw[:, 0] > q) | (raw[:, 1] < c)))}
    assert computed_front == front_ids
    assert len(selected_ids - front_ids) == final["pareto_false_positive_count"]
    assert len(front_ids - selected_ids) == final["pareto_false_negative_count"]
    if budget is None:
        assert np.isclose(final["cost_usd"], run["actual_total_search_cost_usd"])
    assert np.isclose(full, truth["full_search_cost_usd"])
    assert len(runs) == data["n_seeds"] == 20
    assert np.isclose(ours, np.mean([r["actual_total_search_cost_usd"] for r in runs]))
    if budget is not None:
        ours = low = high = budget
        saving = 100 * (1 - args.budget_fraction)
    return_counts = np.zeros(len(raw), dtype=int)
    precisions, recalls, false_positives = [], [], []
    assert sorted(r["seed"] for r in runs) == list(range(42, 62))
    for seed_run in runs:
        seed_final = recorded_point(seed_run)
        seed_selected = set(seed_final["selected_arm_indices"])
        assert len(seed_selected-front_ids) == seed_final["pareto_false_positive_count"]
        assert len(front_ids-seed_selected) == seed_final["pareto_false_negative_count"]
        if budget is None:
            assert np.isclose(seed_final["cost_usd"], seed_run["actual_total_search_cost_usd"])
        return_counts[sorted(seed_selected)] += 1
        precisions.append(len(seed_selected & front_ids)/len(seed_selected))
        recalls.append(len(seed_selected & front_ids)/len(front_ids))
        false_positives.append(len(seed_selected-front_ids))
    if args.aggregate_seeds:
        # Include every returned configuration, including any false positives.
        selected_ids = set(map(int, np.flatnonzero(return_counts)))
    ordered_ids = sorted(front_ids, key=lambda i: raw[i, 1])
    frontier_raw = raw[ordered_ids]
    draw_ids = sorted(selected_ids, key=lambda i: (return_counts[i], i))
    selected_raw = raw[draw_ids]
    with (OUT / f"{args.stem}_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration_index", "configuration", "accuracy", "mean_cost_per_question_usd", "true_frontier", "returned_by_cc_gittins", "returned_seed_count", "returned_seed_fraction"])
        writer.writerows((i, truth["models"][i], q, c, i in front_ids, i in selected_ids, int(return_counts[i]), return_counts[i]/len(runs))
                         for i, (q, c) in enumerate(raw))
    provenance = {
        "dataset": dataset, "seed": run["seed"],
        "seed_selection": "User-specified" if args.seed is not None else "Closest terminal evaluation cost to median of 20 runs; ties by seed",
        "left_panel": {"n_seeds": len(runs), "mean_evaluation_cost_usd": ours,
                       "mean_cost_95pct_bootstrap_ci_usd": [low, high],
                       "exhaustive_evaluation_cost_usd": full, "cost_savings_percent": saving},
        "right_panel_evaluation_cost_usd": float(final["cost_usd"]),
        "checkpoint": "Recorded run termination" if budget is None else f"{100*args.budget_fraction:g}% of exhaustive evaluation USD",
        "true_frontier_indices": sorted(front_ids), "returned_indices": sorted(selected_ids),
        "precision": len(selected_ids & front_ids)/len(selected_ids),
        "recall": len(selected_ids & front_ids)/len(front_ids),
        "false_positives": len(selected_ids-front_ids),
        "false_negatives": len(front_ids-selected_ids),
        "coordinate_source": f"truth_reference.json: {dataset}.raw_truth, exhaustive frozen evaluation matrix",
        "recommendation_source": f"high_cost_runs_compact.json: {dataset}.runs, final recorded recommendation",
        "evaluation_mode": "Offline replay with frozen evaluation USD charges",
        "frontier_x_axis": "Mean workflow USD per question (log scale)",
        "frontier_y_axis": "Exhaustive mean accuracy (%)",
        "coordinate_jitter": False,
    }
    if args.aggregate_seeds:
        for key in ("seed", "seed_selection", "right_panel_evaluation_cost_usd", "precision", "recall", "false_positives", "false_negatives"):
            provenance.pop(key)
        provenance.update({
            "right_panel": {"n_seeds": len(runs), "seeds": sorted(r["seed"] for r in runs),
                            "color_encoding": "Number of returned sets at the selected checkpoint containing each configuration, out of 20",
                            "returned_seed_counts": {str(i): int(return_counts[i]) for i in sorted(selected_ids)},
                            "recommendation_filtering": "None; all returned configurations included"},
            "returned_indices_scope": "Union over all 20 seeds, not a single returned set",
            "mean_precision": float(np.mean(precisions)), "mean_recall": float(np.mean(recalls)),
            "total_false_positives": int(sum(false_positives)),
            "zero_false_positive_runs": sum(n == 0 for n in false_positives),
            "exact_frontier_runs": sum(p == 1 and r == 1 for p, r in zip(precisions, recalls)),
        })
    if budget is not None:
        provenance["left_panel"] = {
            "n_seeds": len(runs), "budget_fraction": args.budget_fraction,
            "evaluation_budget_usd_per_seed": budget,
            "exhaustive_evaluation_cost_usd": full, "budget_reduction_percent": saving,
            "cost_semantics": "Allocated common budget, not the cost when recommendations last changed",
        }
        provenance["recommendation_source"] = ("frontier_checkpoint_data.csv, independently matched against saved per-seed trajectories" if checkpoint_rows else
                                                f"high_cost_runs_compact.json: {dataset}.runs, latest recorded recommendation at or below budget")
        provenance["checkpoint_rule"] = "Latest returned set at or below each USD budget; same right-continuous recommendation rule as downstream figures"
        provenance["seed_checkpoints"] = [{
            "seed": r["seed"], "target_cost_usd": budget,
            "last_recommendation_update_cost_usd": recorded_point(r)["cost_usd"],
            "selected_indices": recorded_point(r)["selected_arm_indices"],
        } for r in runs]
    (OUT / f"{args.stem}_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

font = FontProperties(fname=findfont('Times New Roman',fallback_to_default=False))
def fs(points):
    return points * args.font_scale
accent = "#F4A259" if args.orange else "#87CEFA"
accent_dark = "#B84B0A" if args.orange else "#17649A"
np.random.seed(42)

with plt.xkcd(scale=0.7, length=110, randomness=2):
    plt.rcParams.update({"font.family": font.get_name(), "pdf.fonttype": 42, "svg.fonttype": "none"})
    if args.frontier_panel:
        fig, (ax, frontier_ax) = plt.subplots(
            1, 2, figsize=(10.9, 3.7), facecolor="white",
            gridspec_kw={"width_ratios": [1.15, 1]})
        fig.subplots_adjust(left=.10 if args.font_scale > 1 else .08,
                            right=.97, bottom=.255 if args.font_scale > 1 else .19,
                            top=.82, wspace=.38 if args.font_scale > 1 else .36)
    else:
        fig, ax = plt.subplots(figsize=(6.2, 3.7), facecolor="white")
        fig.subplots_adjust(left=0.14, right=0.965, bottom=0.19, top=0.82)
    ax.bar([0, 1], [full, ours], width=0.67,
           color=["#D3D3D3", accent], edgecolor="#333333", linewidth=0.8)
    if args.budget_fraction is None:
        ax.errorbar(1, ours, yerr=np.array([[ours-low], [high-ours]]),
                    fmt="none", color="#333333", capsize=3, elinewidth=0.8, capthick=0.8)
    title_size = fs(13.5) if args.font_scale > 1 else 10
    ax.set_title(args.title, pad=17, fontproperties=FontProperties(fname=font.get_file(), size=title_size))
    ax.set_ylabel("USD", fontsize=fs(13), fontproperties=font)
    ours_label = ("CC-Gittins (ours)" if args.budget_as_savings_label else
                  f"CC-Gittins\n({100*args.budget_fraction:g}% budget)" if args.budget_fraction is not None else
                  "CC-Gittins\n(20-seed mean)" if args.frontier_panel else "CC-Gittins")
    ax.set_xticks([0, 1], ["Brute Force", ours_label], fontsize=fs(12))
    if full < 20:
        ax.set_yticks([0, 5, 10, 15])
        ax.set_ylim(0, 17)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_ylim(0, full * 1.145)
    ax.set_xlim(-0.53, 1.55)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=4, labelsize=fs(11))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.text(0, full+0.044*full, f"${full:,.2f}", ha="center", va="bottom", fontsize=fs(12), fontproperties=font)
    cost_annotation = f"${ours:,.2f}"
    if not args.hide_savings or args.budget_as_savings_label:
        cost_annotation += "\n" + r"$\downarrow$" + f" {saving:.1f}%"
    ax.text(1, high+0.037*full, cost_annotation,
            ha="center", va="bottom", fontsize=fs(12), fontproperties=font, linespacing=1.18)
    if args.frontier_panel:
        background = frontier_ax.scatter(raw[:, 1], 100*raw[:, 0], s=22,
                                         color="#D3D3D3", zorder=2)
        line, = frontier_ax.plot(frontier_raw[:, 1], 100*frontier_raw[:, 0],
                                 color="#8C8C8C", linewidth=1.0, zorder=2)
        true_points = frontier_ax.scatter(frontier_raw[:, 1], 100*frontier_raw[:, 0],
                                          s=65, facecolor="white", edgecolor="#8C8C8C",
                                          linewidth=1.1, zorder=3, label="True frontier")
        if args.aggregate_seeds:
            frequency_cmap = LinearSegmentedColormap.from_list("recovery_frequency", ["#FFFFFF", accent, accent_dark])
            frequency_norm = Normalize(vmin=0, vmax=len(runs))
            selected_is_true = np.asarray([i in front_ids for i in draw_ids])
            found_points = frontier_ax.scatter(
                selected_raw[selected_is_true, 1], 100*selected_raw[selected_is_true, 0], s=65,
                c=return_counts[draw_ids][selected_is_true], cmap=frequency_cmap, norm=frequency_norm,
                edgecolor="#333333", linewidth=.8, zorder=4)
            if not np.all(selected_is_true):
                off_frontier_points = frontier_ax.scatter(
                    selected_raw[~selected_is_true, 1], 100*selected_raw[~selected_is_true, 0], s=58,
                    c=return_counts[draw_ids][~selected_is_true],
                    cmap=frequency_cmap, norm=frequency_norm, marker="D",
                    edgecolor="#333333", linewidth=.8, zorder=5, label="Off-frontier return")
                off_frontier_points.set_sketch_params(None)
            color_ax = frontier_ax.inset_axes([.10, .20, .39, .05] if args.font_scale > 1 else [.065, .12, .40, .045])
            colorbar = fig.colorbar(found_points, cax=color_ax, orientation="horizontal", ticks=[0, 10, 20])
            colorbar.ax.set_xticklabels(["0/20", "10/20", "20/20"])
            colorbar.ax.tick_params(labelsize=fs(10 if args.font_scale > 1 else 9), length=2, pad=3)
            colorbar.outline.set_linewidth(.6)
            colorbar.solids.set_sketch_params(None)
            colorbar.solids.set_path_effects([])
            colorbar.outline.set_sketch_params(None)
            colorbar.outline.set_path_effects([])
            colorbar.ax.set_title("Returned seeds" if args.font_scale > 1 else "Returned (out of 20)",
                                  pad=7, fontproperties=FontProperties(fname=font.get_file(), size=fs(9) if args.font_scale > 1 else 10))
        else:
            found_points = frontier_ax.scatter(selected_raw[:, 1], 100*selected_raw[:, 0],
                                                s=65, color=accent, edgecolor="#333333",
                                                linewidth=.8, zorder=4, label="CC-Gittins")
        # Keep actual point positions and connecting segments exact.
        # Hand-drawn styling is confined to axes, bars, and typography.
        for artist in (background, line, true_points, found_points):
            artist.set_sketch_params(None)
        frontier_title = "Pareto frontier (20 seeds)" if args.aggregate_seeds else f"Pareto frontier (seed {run['seed']})"
        if budget is not None and not args.budget_as_savings_label:
            frontier_title = f"Pareto frontier ({100*args.budget_fraction:g}% budget, 20 seeds)" if args.aggregate_seeds else f"Pareto frontier ({100*args.budget_fraction:g}% budget, seed {run['seed']})"
        frontier_ax.set_title(frontier_title, pad=17,
                              fontproperties=FontProperties(fname=font.get_file(), size=fs(13) if args.font_scale > 1 else 10))
        frontier_ax.set_xlabel("Cost / question (USD, log scale)" if args.font_scale > 1 else "Cost per question (USD, log scale)", fontsize=fs(11), labelpad=8, fontproperties=font)
        frontier_ax.set_ylabel("Accuracy (%)", fontsize=fs(13), labelpad=8, fontproperties=font)
        frontier_ax.set_xscale("log")
        frontier_ax.set_xlim(raw[:, 1].min()/1.3, raw[:, 1].max()*1.3)
        frontier_ax.set_ylim(100*raw[:, 0].min()-3, 100*raw[:, 0].max()+4)
        frontier_ax.xaxis.set_major_locator(FixedLocator([.0005, .005, .05] if args.font_scale > 1 else [.0005, .001, .005, .01, .05]))
        frontier_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        frontier_ax.xaxis.set_minor_locator(NullLocator())
        frontier_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        frontier_ax.tick_params(axis="both", length=3, labelsize=fs(10))
        for side in ("top", "right"):
            frontier_ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            frontier_ax.spines[side].set_linewidth(1)
        frontier_ax.legend(loc="upper left" if args.font_scale > 1 else "lower left", frameon=False,
                           bbox_to_anchor=(.015, 1.01) if args.font_scale > 1 else (.015, .24) if args.aggregate_seeds else (0, 0),
                           prop=FontProperties(fname=font.get_file(), size=fs(10)),
                           handletextpad=.4, borderaxespad=.25, labelspacing=.45)
        if args.show_quality:
            quality_recall = float(np.mean(recalls)) if args.aggregate_seeds else len(selected_ids & front_ids)/len(front_ids)
            quality_precision = float(np.mean(precisions)) if args.aggregate_seeds else len(selected_ids & front_ids)/len(selected_ids)
            quality_text = f"Recall: {100*quality_recall:.2f}%\nPrecision: {100*quality_precision:g}%"
            frontier_ax.text(.98, .25, quality_text,
                             transform=frontier_ax.transAxes, ha="right", va="center",
                             fontsize=fs(11), fontproperties=font, linespacing=1.4)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{args.stem}.{extension}",
                    dpi=250, facecolor="white")
    plt.close(fig)
