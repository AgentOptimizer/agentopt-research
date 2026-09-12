#!/usr/bin/env python3
"""Plot the completed-mean / unfinished-LCB diagnostic on a saved BIRD trace.

No acquisition replay occurs. All membership changes appear in numbered PNG
pages, one multipage PDF, and CSV. An optional excerpt retains original C labels. Coordinate
positions are offline full-data evaluations, never inputs to recommendations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.plot.plot_lcb_recommendations import (
    COLORS, GRAY, _draw_panel, checkpoint_sequence,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

RESULTS = ROOT / "experiments/combined_objective/results"
OUTDIR = RESULTS / "hybrid_lcb_recommendations_bird_dev_seed42_beta1"


def make_figure(points, run, subtitle):
    rows = (len(points) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14.4, 3.9 * rows + 1.45), squeeze=False)
    for ax, point in zip(axes.flat, points):
        _draw_panel(ax, point, run, "lcb")
    for ax in list(axes.flat)[len(points):]:
        ax.set_visible(False)
    color = COLORS["lcb"]
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor=color, label="Recommended · completed"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=color, label="Recommended · partial"),
        Line2D([], [], marker="o", linestyle="none", color=GRAY,
               label="All full-data configurations (reference)"),
        Line2D([], [], linestyle=":", color="#717b84", label="Full-data Pareto front (reference)"),
    ]
    fig.suptitle(
        "BIRD dev · hybrid recommendation · completed: μ; partial: μ − σ\n"
        f"Seed 42 · asynchronous η · {subtitle}", fontsize=13, y=0.986,
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.046),
               ncol=2, fontsize=9, frameon=False)
    fig.text(
        0.5, 0.013,
        "Recommendations reconstructed from the same saved acquisition trace; no sampling or stopping changes.\n"
        "Plotted accuracy/cost uses full-data offline evaluation. Hollow circles are recommended partial configurations. Labels: config ID (samples).",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.18 if rows == 1 else 0.13 if rows == 2 else 0.10,
                           1, 0.87 if rows == 1 else 0.925 if rows == 2 else 0.945))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=OUTDIR / "diagnostic.json")
    parser.add_argument("--reference", type=Path,
                        default=RESULTS / "lcb_recommendations_bird_dev_seed42_beta1/comparison.json")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--pdf-only", action="store_true",
                        help="Write the complete PDF and CSV without rendering per-page PNGs")
    parser.add_argument("--include-selected", action="store_true",
                        help="Also export an explicitly labelled excerpt from first highest-accuracy recommendation onward")
    args = parser.parse_args()
    diagnostic = json.loads(args.diagnostic.read_text())
    original = json.loads(args.reference.read_text())["runs"]["lcb"]
    summary = diagnostic["methods"]["hybrid"]
    snapshots = [summary["warm"], *diagnostic["membership_changes"]["hybrid"], summary["final"]]
    full_count = diagnostic["full_sample_count"]
    points = []
    for snapshot in snapshots:
        flags = [count == full_count for count in snapshot["selected_samples"]]
        points.append({
            **snapshot,
            "selected_arm_indices": snapshot["selected_arms"],
            "selected_sample_counts": snapshot["selected_samples"],
            "selected_completed_flags": flags,
            "partial_recommended_count": sum(not flag for flag in flags),
        })
    run = {**original, "points": points}
    sequence = checkpoint_sequence(run)
    expected = summary["membership_changes_excluding_warm"]
    assert len(sequence) == expected + 2
    first = summary["frontier_target"]["first"]
    sustained = summary["frontier_target"]["sustained"]
    focus = [point for point in sequence if point["evaluations"] >= first["evaluations"]]
    args.outdir.mkdir(parents=True, exist_ok=True)
    focus_files = []
    if args.include_selected:
        for start in range(0, len(focus), 9):
            selected = focus[start:start + 9]
            fig = make_figure(selected, run,
                              f"selected checkpoints only · {selected[0]['checkpoint']}–{selected[-1]['checkpoint']}")
            suffix = "" if start == 0 else f"_page{start // 9 + 1:02d}"
            path = args.outdir / f"pareto_selected_checkpoints{suffix}.png"
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            focus_files.append(path.name)
    pdf_path = args.outdir / "pareto_key_checkpoints_all_pages.pdf"
    pages = (len(sequence) + 8) // 9
    page_files, page_checkpoints = [], []
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(sequence), 9):
            page = sequence[start:start + 9]
            page_number = start // 9 + 1
            labels = [point["checkpoint"] for point in page]
            page_checkpoints.append(labels)
            fig = make_figure(page, run,
                              f"all {expected} membership changes · page {page_number}/{pages} · {labels[0]}–{labels[-1]}")
            if not args.pdf_only:
                page_path = args.outdir / f"pareto_key_checkpoints_page{page_number:02d}.png"
                fig.savefig(page_path, dpi=180, facecolor="white")
                page_files.append(page_path.name)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    rows = []
    for point in sequence:
        row = {key: point[key] for key in ("checkpoint", "cost_fraction", "cost_usd", "evaluations")}
        for key in ("selected_config_ids", "selected_arm_indices", "selected_sample_counts",
                    "selected_completed_flags", "added_config_ids", "removed_config_ids"):
            row[key] = json.dumps(point[key])
        rows.append(row)
    with (args.outdir / "checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "rule": diagnostic["rule"], "source_trace": diagnostic["input_trace"],
        "diagnostic_sha256": hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "membership_changes": expected, "panels": len(sequence), "pdf_pages": pages,
        "page_files": page_files, "page_checkpoints": page_checkpoints,
        "focus_checkpoints": [point["checkpoint"] for point in focus],
        "focus_files": focus_files, "full_pdf": pdf_path.name,
        "first_best_accuracy_cost_fraction": first["cost_fraction"],
        "sustained_best_accuracy_cost_fraction": sustained["cost_fraction"],
        "selector_replayed": False,
    }
    (args.outdir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
