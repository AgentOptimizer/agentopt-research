"""Render every recommendation change in compact eta-decay comparisons.

This module only reads saved results. Numbered checkpoints follow the original
``key_checkpoints`` membership-change rules; a later terminal state is shown as
an unnumbered Final panel. No selector replay or fixed-budget sampling occurs.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from experiments.combined_objective.offline_radial_gittins import raw_nondominated_indices
from experiments.combined_objective.plot.plot_anytime_radial_gittins import key_checkpoints

LABELS = {"global_stop": "Global eta decay", "direction_stop": "Independent eta per direction"}
COLORS = {"global_stop": "#175d8d", "direction_stop": "#d26b35"}
GRAY = "#b8c1c9"
GOLD = "#e8b62c"


def _config_id(model_name: str) -> str:
    return model_name.split("config_id=", 1)[1].split("|", 1)[0] if "config_id=" in model_name else model_name


def _checkpoint_sequence(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Adapt compact fields to the existing membership-checkpoint function."""
    points = run["points"]
    if not points:
        return []
    adapted = []
    for index, point in enumerate(points):
        selected = list(point["selected_arm_indices"])
        vectors = point["estimated_raw_archive_vectors"]
        if len(selected) != len(vectors):
            raise ValueError("selected arms and observed archive vectors must align")
        adapted.append({
            **point,
            "cumulative_evaluations": point["evaluations"],
            "cumulative_search_cost_usd": point["cost_usd"],
            "budget_fraction": point["cost_fraction"],
            "archive_scope": "deployable",
            "event": "after_warm_start" if index == 0 else "final" if index == len(points) - 1 else "adaptive_pull",
            "selected_arm_indices": selected,
            "direction_winner_arm_indices": selected,
            "estimated_raw_winner_vectors": vectors,
        })
    adapter = {
        "params": {"recommendation_changes_only": False},
        "recommendation_initial_snapshot": adapted[0],
        "recommendation_trajectory": adapted,
    }
    numbered = key_checkpoints(adapter)
    final = adapted[-1]
    previous = numbered[-1] if numbered else adapted[0]
    later_end = (
        final["cumulative_evaluations"] > previous["cumulative_evaluations"]
        or final["cumulative_search_cost_usd"] > previous["cumulative_search_cost_usd"]
    )
    if later_end:
        numbered.append({**final, "checkpoint": "Final", "added_arm_indices": [], "removed_arm_indices": []})
    for point in numbered:
        for prefix, key in (("selected", "selected_arm_indices"), ("added", "added_arm_indices"), ("removed", "removed_arm_indices")):
            names = [run["model_names"][arm] for arm in point[key]]
            point[f"{prefix}_model_names"] = names
            point[f"{prefix}_config_ids"] = [_config_id(name) for name in names]
    return numbered


def _draw_panel(ax: Any, point: Mapping[str, Any], truth: np.ndarray,
                reference: np.ndarray, method: str) -> None:
    color = COLORS.get(method, "#175d8d")
    ax.scatter(truth[:, 1], truth[:, 0], s=15, color=GRAY, alpha=0.65, zorder=1)
    ax.plot(reference[:, 1], reference[:, 0], ":", color="#717b84", linewidth=1.2, zorder=2)
    selected = point["selected_arm_indices"]
    if selected:
        vectors = np.asarray(point["estimated_raw_archive_vectors"], dtype=float)
        order = np.argsort(vectors[:, 1])
        ax.plot(vectors[order, 1], vectors[order, 0], "o-", color=color,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7,
                markersize=6, linewidth=1.5, zorder=3)
        by_arm = dict(zip(selected, vectors))
        added = [arm for arm in point["added_arm_indices"] if arm in by_arm]
        if added:
            new_vectors = np.asarray([by_arm[arm] for arm in added])
            ax.scatter(new_vectors[:, 1], new_vectors[:, 0], marker="*", s=150,
                       color=GOLD, edgecolor="#835f00", linewidth=0.55, zorder=5)
        for position, (arm, vector, label) in enumerate(zip(selected, vectors, point["selected_config_ids"])):
            ax.annotate(label, (vector[1], vector[0]), xytext=(4, 5 + 11 * (position % 2)),
                        textcoords="offset points", fontsize=7.5, color="#27323b",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.65},
                        zorder=6)
    else:
        ax.text(0.5, 0.5, "No completed recommendation", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    if np.all(truth[:, 1] > 0):
        ax.set_xscale("log")
    ax.margins(x=0.12)
    ax.set_ylim(max(-0.025, float(np.min(truth[:, 0])) - 0.025),
                min(1.05, float(np.max(truth[:, 0])) + 0.095))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    added_text = ", ".join(point["added_config_ids"]) or "none"
    removed_text = ", ".join(point["removed_config_ids"]) or "none"
    subtitle = ("No change after last key checkpoint" if point["checkpoint"] == "Final"
                else f"Added: {added_text}   Removed: {removed_text}")
    ax.set_title(f"{point['checkpoint']} · {point['cost_fraction']:.6%} search cost\n"
                 f"{subtitle}", fontsize=9.5, linespacing=1.4)
    ax.text(0.03, 0.97, f"{point['evaluations']:,} evaluations · {len(selected)} recommended",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#52606b")
    ax.set_xlabel("Mean deployment cost (USD / question)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _render_method(method: str, run: Mapping[str, Any], points: list[dict[str, Any]],
                   outdir: Path) -> list[Path]:
    if not points:
        return []
    truth = np.asarray(run["raw_truth_vectors"], dtype=float)
    front_indices = sorted(raw_nondominated_indices(truth), key=lambda arm: truth[arm, 1])
    reference = truth[front_indices]
    color = COLORS.get(method, "#175d8d")
    label = LABELS.get(method, method)
    paths = []
    for start in range(0, len(points), 9):
        page = points[start:start + 9]
        rows = math.ceil(len(page) / 3)
        fig, axes = plt.subplots(rows, 3, figsize=(13.5, 3.5 * rows + 1.25), squeeze=False)
        for ax, point in zip(axes.flat, page):
            _draw_panel(ax, point, truth, reference, method)
        for ax in list(axes.flat)[len(page):]:
            ax.set_visible(False)
        handles = [
            Line2D([], [], marker="o", color=color, markerfacecolor=color, markersize=6,
                   label="Currently recommended (completed)"),
            Line2D([], [], marker="*", linestyle="none", color=GOLD, markeredgecolor="#835f00",
                   markersize=11, label="Newly recommended at this checkpoint"),
            Line2D([], [], marker="o", linestyle="none", color=GRAY, markersize=4,
                   label="All full-data configurations (reference)"),
            Line2D([], [], linestyle=":", color="#717b84", label="Full-data Pareto front (reference)"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.039),
                   ncol=2, fontsize=9, frameon=False)
        numbered_count = sum(point["checkpoint"] != "Final" for point in points)
        fig.suptitle(f"BIRD dev · {label} · key recommendation checkpoints\n"
                     f"Seed 42 · completed only · C1–C{numbered_count} mark every membership change",
                     fontsize=13, y=0.985)
        fig.text(0.5, 0.012,
                 "Gray dots include evaluated configurations; gray does not mean unseen. Labels are config IDs.\n"
                 "Warm start is unnumbered and omitted; Final is the later terminal state. Raw accuracy–cost is shown; HV uses normalized objectives.",
                 ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.12 if rows <= 2 else 0.09, 1, 0.93 if rows <= 2 else 0.945))
        suffix = "" if start == 0 else f"_page{start // 9 + 1:02d}"
        base = outdir / f"{method}_pareto_key_checkpoints{suffix}"
        for extension in ("png", "pdf"):
            path = base.with_suffix(f".{extension}")
            fig.savefig(path, dpi=180, facecolor="white")
            paths.append(path)
        plt.close(fig)
    return paths


def export_key_checkpoint_frontiers(
    runs: Mapping[str, Mapping[str, Any]], outdir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Write per-method contact sheets and return C1..Cn plus a later Final.

    ``pareto_snapshots`` aliases the independent-direction contact sheet and
    the complete checkpoint CSV, replacing the former fixed-budget artifacts.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sequences = {method: _checkpoint_sequence(run) for method, run in runs.items()}
    csv_rows = []
    for method, run in runs.items():
        paths = _render_method(method, run, sequences[method], outdir)
        if method == "direction_stop":
            for path in paths[:2]:
                shutil.copyfile(path, outdir / f"pareto_snapshots{path.suffix}")
        for point in sequences[method]:
            row = {
                "method": method, "checkpoint": point["checkpoint"],
                "cost_fraction": point["cost_fraction"], "cost_usd": point["cost_usd"],
                "evaluations": point["evaluations"],
                "relative_hv_regret": point["relative_hv_regret"],
                "highest_accuracy_completed_and_recommended": point["contains_true_accuracy_best"],
            }
            for prefix in ("selected", "added", "removed"):
                for suffix in ("arm_indices", "config_ids", "model_names"):
                    row[f"{prefix}_{suffix}"] = json.dumps(point[f"{prefix}_{suffix}"])
            csv_rows.append(row)
    if csv_rows:
        with (outdir / "checkpoints.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        shutil.copyfile(outdir / "checkpoints.csv", outdir / "pareto_snapshots.csv")
    return sequences
