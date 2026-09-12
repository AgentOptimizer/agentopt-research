"""Plot saved LCB recommendations and every membership checkpoint.

All accuracy/cost positions are offline full-data evaluations of the returned
configurations. These values are not available to the selector. Hollow versus
filled circles indicate whether the selected configuration was completed at
that checkpoint; gray reference dots do not encode observation status.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

LABELS = {"completed_only": "Completed-only recommendation", "lcb": "LCB recommendation",
          "hybrid_lcb": "Hybrid LCB recommendation"}
COLORS = {"completed_only": "#175d8d", "lcb": "#cb6334", "hybrid_lcb": "#36806b"}
GRAY = "#b8c1c9"


def config_id(name):
    return name.split("config_id=", 1)[1].split("|", 1)[0] if "config_id=" in name else name


def checkpoint_sequence(run):
    """Warm start, C1..Cn for every set change, and a separate Final panel."""
    points = run["points"]
    if not points:
        return []
    sequence = [{**points[0], "checkpoint": "Warm start", "added_arm_indices": [], "removed_arm_indices": []}]
    previous = set(points[0]["selected_arm_indices"])
    number = 0
    for point in points[1:]:
        current = set(point["selected_arm_indices"])
        if current != previous:
            number += 1
            sequence.append({**point, "checkpoint": f"C{number}",
                             "added_arm_indices": sorted(current - previous),
                             "removed_arm_indices": sorted(previous - current)})
        previous = current
    # Final is deliberately separate even when it shares the last change's
    # cost: it communicates terminal sample counts/completion status.
    sequence.append({**points[-1], "checkpoint": "Final", "added_arm_indices": [], "removed_arm_indices": []})
    for point in sequence:
        for prefix in ("selected", "added", "removed"):
            names = [run["model_names"][arm] for arm in point[f"{prefix}_arm_indices"]]
            point[f"{prefix}_model_names"] = names
            point[f"{prefix}_config_ids"] = [config_id(name) for name in names]
    return sequence


def _draw_panel(ax, point, run, method):
    truth = np.asarray(run["raw_truth_vectors"])
    reference = truth[run["full_data_pareto_arm_indices"]]
    reference = reference[np.argsort(reference[:, 1])]
    color = COLORS[method]
    ax.scatter(truth[:, 1], truth[:, 0], s=14, color=GRAY, alpha=0.65, zorder=1)
    ax.plot(reference[:, 1], reference[:, 0], ":", color="#717b84", linewidth=1.2, zorder=2)
    selected = point["selected_arm_indices"]
    if selected:
        vectors = truth[selected]
        # Do not connect selected points into a purported raw Pareto frontier:
        # nondomination is in recommendation space and raw domination can occur.
        flags = np.asarray(point["selected_completed_flags"], dtype=bool)
        for complete in (False, True):
            subset = vectors[flags == complete]
            if len(subset):
                ax.scatter(subset[:, 1], subset[:, 0], s=49, marker="o",
                           facecolors=color if complete else "white", edgecolors=color,
                           linewidths=1.55, zorder=4)
        for position, (vector, label, count) in enumerate(zip(vectors, point["selected_config_ids"], point["selected_sample_counts"])):
            # Alternate nearby labels; expensive configurations are annotated
            # inward so the rightmost endpoint remains inside its panel.
            on_right = vector[1] > np.quantile(truth[:, 1], 0.8)
            ax.annotate(f"{label} (n={count})", (vector[1], vector[0]),
                        xytext=(-4 if on_right else 4, 5 + 12 * (position % 2)),
                        textcoords="offset points", ha="right" if on_right else "left",
                        fontsize=6.8, color="#27323b", zorder=6,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.65})
    else:
        ax.text(0.5, 0.48, "No recommendation", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#52606b")
    if np.all(truth[:, 1] > 0):
        ax.set_xscale("log")
    ax.margins(x=0.14)
    ax.set_ylim(max(-0.025, float(truth[:, 0].min()) - 0.025), min(1.05, float(truth[:, 0].max()) + 0.105))
    added = ", ".join(point["added_config_ids"]) or "none"
    removed = ", ".join(point["removed_config_ids"]) or "none"
    subtitle = ("Initial recommendation after 4 samples per configuration" if point["checkpoint"] == "Warm start"
                else "Terminal recommendation and completion status" if point["checkpoint"] == "Final"
                else f"Added: {added}; removed: {removed}")
    ax.set_title(f"{point['checkpoint']} · {point['cost_fraction']:.6%} search cost\n"
                 + textwrap.fill(subtitle, width=65), fontsize=9, linespacing=1.3)
    ax.text(0.03, 0.97, f"{point['evaluations']:,} evaluations · {point['partial_recommended_count']} partial recommended",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#52606b")
    ax.set_xlabel("Full-data mean cost (USD / question)", fontsize=8.5)
    ax.set_ylabel("Full-data accuracy", fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _render_checkpoints(method, run, points, outdir):
    label = LABELS[method]
    if method in ("lcb", "hybrid_lcb"):
        label += f" (β={run['recommendation_beta']:g})"
    color = COLORS[method]
    prefix = f"{method}_pareto_key_checkpoints"
    png_paths = []
    page_count = math.ceil(len(points) / 9)
    all_pages = outdir / f"{prefix}_all_pages.pdf"
    with PdfPages(all_pages) as pdf:
        for start in range(0, len(points), 9):
            page = points[start:start + 9]
            rows = math.ceil(len(page) / 3)
            fig, axes = plt.subplots(rows, 3, figsize=(14.4, 3.9 * rows + 1.35), squeeze=False)
            for ax, point in zip(axes.flat, page):
                _draw_panel(ax, point, run, method)
            for ax in list(axes.flat)[len(page):]:
                ax.set_visible(False)
            handles = [
                Line2D([], [], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor=color,
                       markersize=6, label="Recommended · completed"),
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=color,
                       markeredgewidth=1.5, markersize=6, label="Recommended · partial"),
                Line2D([], [], marker="o", linestyle="none", color=GRAY, markersize=4,
                       label="All full-data configurations (reference)"),
                Line2D([], [], linestyle=":", color="#717b84", label="Full-data Pareto front (reference)"),
            ]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.043),
                       ncol=2, fontsize=9, frameon=False)
            number = sum(point["checkpoint"].startswith("C") for point in points)
            fig.suptitle(f"Radial-Gittins-decay · BIRD dev · {label}\n"
                         f"Seed 42 · asynchronous η · all {number} membership changes · page {start // 9 + 1}/{page_count}",
                         fontsize=13, y=0.985)
            fig.text(0.5, 0.010,
                     "Offline evaluation: all plotted accuracy/cost values use the full dataset and are unavailable to the selector.\n"
                     "Gray does not mean unseen. Labels: config ID (sample count). HV uses normalized objectives. Acquisitions are identical between rules.",
                     ha="center", va="bottom", fontsize=8.2)
            fig.tight_layout(rect=(0, 0.14 if rows == 1 else 0.12 if rows == 2 else 0.09, 1,
                                   0.90 if rows == 1 else 0.93 if rows == 2 else 0.945))
            suffix = "" if start == 0 else f"_page{start // 9 + 1:03d}"
            path = outdir / f"{prefix}{suffix}.png"
            fig.savefig(path, dpi=170, facecolor="white")
            fig.savefig(outdir / f"{prefix}{suffix}.pdf", facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            png_paths.append(path.name)
            if (start // 9 + 1) % 10 == 0 or start + 9 >= len(points):
                print(f"Rendered {method}: page {start // 9 + 1}/{page_count}", flush=True)
    return {"membership_change_count": len(points) - 2, "panel_count": len(points),
            "page_count": page_count, "png_files": png_paths, "all_pages_pdf": all_pages.name}


def _comparison_curves(saved, outdir):
    runs = saved["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for method, run in runs.items():
        points = run["points"]
        x = np.asarray([point["cost_fraction"] for point in points] + [1.0])
        metrics = (
            (axes[0, 0], "best_recommended_accuracy"), (axes[0, 1], "relative_hv_regret"),
            (axes[1, 0], "relative_hv_regret"), (axes[1, 1], "offline_dominated_selected_count"),
        )
        for ax, key in metrics:
            values = [np.nan if point[key] is None else point[key] for point in points]
            y = np.asarray(values + [values[-1]])
            ax.step(x, y, where="post", color=COLORS[method], label=LABELS[method], linewidth=1.8,
                    linestyle="-" if method == "completed_only" else "--")
        first = next((point for point in points if point["contains_true_accuracy_best"]), None)
        if first:
            axes[0, 0].scatter(first["cost_fraction"], first["best_recommended_accuracy"], s=35,
                               color=COLORS[method], zorder=5)
            axes[0, 0].annotate(f"First: {first['cost_fraction']:.4%}",
                                (first["cost_fraction"], first["best_recommended_accuracy"]),
                                xytext=(6, -24 if method == "completed_only" else -43),
                                textcoords="offset points", fontsize=8.5, color=COLORS[method],
                                arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": 0.8})
    axes[0, 0].axhline(next(iter(runs.values()))["oracle_best_accuracy"], color="#777777", linestyle=":", linewidth=1)
    axes[0, 0].set(title="Highest accuracy in returned recommendation", ylabel="Full-data accuracy (offline)")
    axes[0, 1].set(title="Whole recommendation quality", ylabel="Relative HV regret (offline)")
    axes[1, 0].set(title="Early recommendation quality · first 5% spend", ylabel="Relative HV regret (offline)")
    axes[1, 1].set(title="Early mistakes · first 5% spend", ylabel="Full-data-dominated configurations selected")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_yscale("symlog", linthresh=0.001)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.set_xlim(0, 0.05 if ax in axes[1] else 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Cumulative search cost / full matrix cost")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    method = saved["config"]["recommendation_rule"]
    beta = runs[method]["recommendation_beta"]
    fig.suptitle(f"Radial-Gittins-decay · BIRD dev · {LABELS[method]} β={beta:g} vs completed-only\n"
                 "Same asynchronous η acquisition · seed 42", fontsize=14)
    fig.text(0.5, 0.012,
             "All quality metrics are offline full-data evaluation, unavailable to the selector. LCB may recommend partial configurations.\n"
             "Physical samples, eta events and total acquisition spend are identical. Earlier recommendation does not reduce acquisition spend.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    for extension in ("png", "pdf"):
        fig.savefig(outdir / f"comparison.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def export_plots(saved, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _comparison_curves(saved, outdir)
    manifest, rows = {}, []
    for method, run in saved["runs"].items():
        points = checkpoint_sequence(run)
        manifest[method] = _render_checkpoints(method, run, points, outdir)
        for point in points:
            row = {"method": method, "checkpoint": point["checkpoint"],
                   "cost_fraction": point["cost_fraction"], "cost_usd": point["cost_usd"],
                   "evaluations": point["evaluations"], "relative_hv_regret": point["relative_hv_regret"],
                   "contains_true_accuracy_best": point["contains_true_accuracy_best"],
                   "offline_dominated_selected_count": point["offline_dominated_selected_count"]}
            for key in ("selected_arm_indices", "selected_model_names", "selected_config_ids", "selected_sample_counts",
                        "selected_completed_flags", "added_arm_indices", "added_config_ids", "removed_arm_indices",
                        "removed_config_ids", "true_best_accuracy_sample_counts"):
                row[key] = json.dumps(point[key])
            rows.append(row)
    with (outdir / "checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(outdir / "checkpoints.csv", outdir / "pareto_snapshots.csv")
    # The overview alias is explicitly page one; the manifest links every
    # page and the full multipage PDF, without dropping any checkpoint.
    method = saved["config"]["recommendation_rule"]
    shutil.copyfile(outdir / f"{method}_pareto_key_checkpoints.png", outdir / "pareto_snapshots.png")
    shutil.copyfile(outdir / f"{method}_pareto_key_checkpoints_all_pages.pdf", outdir / "pareto_snapshots.pdf")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    saved = json.loads(args.comparison.read_text())
    print(json.dumps(export_plots(saved, args.comparison.parent), indent=2))


if __name__ == "__main__":
    main()
