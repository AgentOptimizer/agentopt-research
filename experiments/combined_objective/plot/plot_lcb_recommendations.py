"""Plot saved LCB recommendations and every membership checkpoint.

All accuracy/cost positions use full-data evaluations. Completed recommended
points equal their observed means; partial-arm truth and the full reference
frontier are offline diagnostics. Hollow versus filled circles indicate
completion at that checkpoint; gray dots do not encode observation status.
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

LABELS = {"completed_only": "Completed-only recommendation", "finite_lcb": "Full-test mean LCB",
          "finite_mean": "Full-test mean Pareto"}
COLORS = {"completed_only": "#175d8d", "finite_lcb": "#8056a2", "finite_mean": "#b87718"}
PENALIZED_RULES = ("finite_lcb",)
PARTIAL_RULES = (*PENALIZED_RULES, "finite_mean")
GRAY = "#b8c1c9"


def _method_label(method, run, *, show_question_order=False):
    if method == "completed_only" and run.get("recommendation_filter") == "all_completed_empirical_raw_pareto":
        label = "Completed empirical Pareto frontier"
    else:
        label = LABELS[method]
    if method in ("finite_lcb", "finite_mean") and run.get("recommendation_min_samples", 0):
        label += f" (n≥{run['recommendation_min_samples']})"
    if show_question_order:
        label += f" [{run.get('question_order', 'independent')}]"
    return label


def config_id(name, arm_index=None):
    if "config_id=" in name:
        return name.split("config_id=", 1)[1].split("|", 1)[0]
    return str(arm_index) if arm_index is not None else name


def _limited_prefix(config, run):
    scope = str(config.get("audit_scope", "")).lower()
    reason = str(run.get("stop_reason", "")).lower()
    scope_is_limited = any(word in scope for word in ("first ", "prefix", "limited", "truncat"))
    limit = config.get("max_total_question_evaluations")
    evaluations = run.get("evaluations", 0)
    full_evaluations = run.get("model_count", 0) * run.get("common_question_count", 0)
    full_cells_observed = full_evaluations > 0 and evaluations >= full_evaluations
    full_cost_observed = run.get("cost_fraction", 0.0) >= 1.0 - 1e-9
    reached_restrictive_limit = (limit is not None and evaluations >= limit
                                 and (not full_evaluations or limit < full_evaluations))
    incomplete_budget_stop = "budget" in reason and not (full_cells_observed or full_cost_observed)
    return scope_is_limited or reached_restrictive_limit or incomplete_budget_stop


def _plot_context(config, run):
    benchmark = str(config.get("benchmark", "bird_dev"))
    benchmark = {"bird_dev": "BIRD dev", "stackoverflow": "Stack Overflow",
                 "hotpotqa": "HotpotQA", "mathqa": "MathQA"}.get(benchmark, benchmark)
    limited = _limited_prefix(config, run)
    schedule = run.get("eta_decay_schedule", config.get("eta_decay_schedule", "direction_stop"))
    return {
        "benchmark": benchmark,
        "seed": run.get("seed", config.get("seed", 42)),
        # A primary run's config must not relabel a historical reference.
        "question_order": run.get("question_order", "independent"),
        "eta_label": "asynchronous η" if schedule == "direction_stop" else "global η decay",
        "cost_label": "raw mean cost" if run.get("cost_model", config.get("cost_model")) == "raw_mean"
        else "reciprocal cost",
        "limited": limited,
        "terminal_label": ("Budget end" if "budget" in str(run.get("stop_reason", "")) else "Prefix end")
        if limited else "Final",
    }


def checkpoint_sequence(run, *, terminal_label=None):
    """Warm start, every membership change, and the actual terminal snapshot."""
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
    if terminal_label is None:
        terminal_label = _plot_context({}, run)["terminal_label"]
    sequence.append({**points[-1], "checkpoint": terminal_label, "added_arm_indices": [], "removed_arm_indices": []})
    for point in sequence:
        for prefix in ("selected", "added", "removed"):
            names = [run["model_names"][arm] for arm in point[f"{prefix}_arm_indices"]]
            point[f"{prefix}_model_names"] = names
            point[f"{prefix}_config_ids"] = [
                config_id(name, arm) for name, arm in zip(names, point[f"{prefix}_arm_indices"])
            ]
    return sequence


def _annotate_selected(ax, vectors, labels, counts):
    """Spread compact labels inside the panel and connect them to their points."""
    anchors = ax.transAxes.inverted().transform(ax.transData.transform(vectors[:, [1, 0]]))
    placed = []
    offsets = (0, 0.06, -0.06, 0.12, -0.12, 0.18, -0.18, 0.24, -0.24,
               0.30, -0.30, 0.36, -0.36, 0.42, -0.42, 0.48, -0.48)
    for index in np.lexsort((anchors[:, 1], anchors[:, 0])):
        vector, anchor = vectors[index], anchors[index]
        label = f"{labels[index]} (n={counts[index]})"
        width, height = min(0.70, max(0.17, len(label) * 0.017)), 0.052
        candidates = []
        for preferred_x in (anchor[0] + 0.018, anchor[0] - width - 0.018):
            x = float(np.clip(preferred_x, 0.012, 0.988 - width))
            for offset in offsets:
                y = float(np.clip(anchor[1] + 0.035 + offset, 0.035, 0.885))
                rectangle = (x, y - height / 2, x + width, y + height / 2)
                overlap = sum(
                    max(0, min(rectangle[2], other[2]) - max(rectangle[0], other[0]))
                    * max(0, min(rectangle[3], other[3]) - max(rectangle[1], other[1]))
                    for other in placed
                )
                covered = sum(rectangle[0] <= point[0] <= rectangle[2]
                              and rectangle[1] <= point[1] <= rectangle[3] for point in anchors)
                distance = (y - anchor[1]) ** 2 + 0.4 * (x + width / 2 - anchor[0]) ** 2
                candidates.append((1000 * overlap + 0.15 * covered + distance, rectangle))
        _, rectangle = min(candidates, key=lambda item: item[0])
        placed.append(rectangle)
        ax.annotate(label, (vector[1], vector[0]),
                    xytext=(rectangle[0], (rectangle[1] + rectangle[3]) / 2),
                    textcoords="axes fraction", ha="left", va="center",
                    fontsize=6.8, color="#27323b", zorder=6,
                    arrowprops={"arrowstyle": "-", "color": "#77838b", "linewidth": 0.55,
                                "shrinkA": 2, "shrinkB": 4},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.82})


def _draw_panel(ax, point, run, method):
    truth = np.asarray(run["raw_truth_vectors"])
    reference = truth[run["full_data_pareto_arm_indices"]]
    reference = reference[np.argsort(reference[:, 1])]
    color = COLORS[method]
    ax.scatter(truth[:, 1], truth[:, 0], s=14, color=GRAY, alpha=0.65, zorder=1)
    ax.plot(reference[:, 1], reference[:, 0], ":", color="#717b84", linewidth=1.2, zorder=2)
    if np.all(truth[:, 1] > 0):
        ax.set_xscale("log")
    ax.margins(x=0.14)
    ax.set_ylim(max(-0.025, float(truth[:, 0].min()) - 0.025), min(1.05, float(truth[:, 0].max()) + 0.105))
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
        _annotate_selected(ax, vectors, point["selected_config_ids"], point["selected_sample_counts"])
    else:
        ax.text(0.5, 0.48, "No recommendation", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#52606b")
    added = ", ".join(point["added_config_ids"]) or "none"
    removed = ", ".join(point["removed_config_ids"]) or "none"
    subtitle = ("Initial recommendation after the warm start" if point["checkpoint"] == "Warm start"
                else "Terminal recommendation and completion status" if point["checkpoint"] == "Final"
                else "Last recorded recommendation in this limited prefix"
                if point["checkpoint"] in ("Budget end", "Prefix end")
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


def _render_checkpoints(method, run, points, outdir, *, config=None):
    context = _plot_context(config or {}, run)
    label = _method_label(method, run)
    if method in PENALIZED_RULES:
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
                Line2D([], [], marker="o", linestyle="none", color=GRAY, markersize=4,
                       label="All full-data configurations (reference)"),
                Line2D([], [], linestyle=":", color="#717b84", label="Full-data Pareto front (reference)"),
            ]
            if method in PARTIAL_RULES:
                handles.insert(1, Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                                        markeredgecolor=color, markeredgewidth=1.5, markersize=6,
                                        label="Recommended · partial"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.043),
                       ncol=2, fontsize=9, frameon=False)
            number = sum(point["checkpoint"].startswith("C") for point in points)
            scope = " · limited prefix" if context["limited"] else ""
            fig.suptitle(f"Radial-Gittins-decay · {context['benchmark']} · {label}\n"
                         f"Seed {context['seed']} · {context['cost_label']} · {context['eta_label']}{scope} · all {number} membership changes · "
                         f"page {start // 9 + 1}/{page_count} ({page[0]['checkpoint']}–{page[-1]['checkpoint']})",
                         fontsize=13, y=0.985)
            coordinate_note = (
                "Recommended points use completed observed means; gray configurations and the full-data frontier are offline references."
                if method == "completed_only" else
                "Offline evaluation: plotted accuracy/cost values use the full dataset; partial-arm truth is unavailable to the selector."
            )
            fig.text(0.5, 0.010,
                     coordinate_note + "\n"
                     "Gray does not mean unseen. Labels: config ID / 0-based arm index (sample count). Dominated selections are also shown.",
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
            "question_order": context["question_order"],
            "page_count": page_count, "png_files": png_paths, "all_pages_pdf": all_pages.name,
            "limited_prefix": context["limited"], "terminal_checkpoint": context["terminal_label"],
            "page_checkpoints": [[point["checkpoint"] for point in points[start:start + 9]]
                                 for start in range(0, len(points), 9)]}


def _comparison_curves(saved, outdir):
    runs = saved["runs"]
    config = saved.get("config", {})
    contexts = [_plot_context(config, run) for run in runs.values()]
    mixed_question_orders = len({context["question_order"] for context in contexts}) > 1
    observed_end = max(run["points"][-1]["cost_fraction"] for run in runs.values())
    limited = any(context["limited"] for context in contexts)
    axis_end = max(observed_end, 1e-9) if limited else max(1.0, observed_end)
    early_end = min(0.05, axis_end)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for method, run in runs.items():
        points = run["points"]
        # The last saved snapshot is the observed endpoint. Do not invent a
        # continuation to full budget, especially for an interrupted prefix.
        x = np.asarray([point["cost_fraction"] for point in points])
        metrics = (
            (axes[0, 0], "best_recommended_accuracy"), (axes[0, 1], "relative_hv_regret"),
            (axes[1, 0], "relative_hv_regret"), (axes[1, 1], "offline_dominated_selected_count"),
        )
        for ax, key in metrics:
            values = [np.nan if point[key] is None else point[key] for point in points]
            y = np.asarray(values)
            ax.step(x, y, where="post", color=COLORS[method],
                    label=_method_label(method, run, show_question_order=mixed_question_orders), linewidth=1.8,
                    linestyle="-" if method == "completed_only" else "--")
        first = next((point for point in points if point["contains_true_accuracy_best"]), None)
        if first:
            axes[0, 0].scatter(first["cost_fraction"], first["best_recommended_accuracy"], s=35,
                               color=COLORS[method], zorder=5)
            axes[0, 0].annotate(f"First: {first['cost_fraction']:.4%}",
                                (first["cost_fraction"], first["best_recommended_accuracy"]),
                                xytext=(6, -24 - 19 * list(runs).index(method)),
                                textcoords="offset points", fontsize=8.5, color=COLORS[method],
                                arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": 0.8})
    axes[0, 0].axhline(next(iter(runs.values()))["oracle_best_accuracy"], color="#777777", linestyle=":", linewidth=1)
    axes[0, 0].set(title="Highest accuracy in returned recommendation", ylabel="Full-data accuracy (offline)")
    axes[0, 1].set(title="Whole recommendation quality", ylabel="Relative HV regret (offline)")
    early_label = "first 5% spend" if early_end == 0.05 else f"through {early_end:.4%} spend"
    axes[1, 0].set(title=f"Early recommendation quality · {early_label}", ylabel="Relative HV regret (offline)")
    axes[1, 1].set(title=f"Early mistakes · {early_label}", ylabel="Full-data-dominated configurations selected")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_yscale("symlog", linthresh=0.001)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in (axes[0, 1], axes[1, 0]):
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    for ax in axes.flat:
        ax.set_xlim(0, early_end if ax in axes[1] else axis_end)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Cumulative search cost / full matrix cost")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [_method_label(method, run) + (f" β={run['recommendation_beta']:g}" if method in PENALIZED_RULES and run.get("recommendation_beta") is not None else "")
              for method, run in runs.items()]
    seeds = ", ".join(str(seed) for seed in dict.fromkeys(context["seed"] for context in contexts))
    scope = " · limited prefix" if limited else ""
    method_separator = "\n" if len(runs) > 2 else " · "
    fig.suptitle(f"Radial-Gittins-decay · {contexts[0]['benchmark']}{method_separator}{' vs '.join(labels)}\n"
                 f"Seed {seeds} · {contexts[0]['cost_label']} · {contexts[0]['eta_label']}{scope}",
                 fontsize=12.5 if len(runs) > 2 else 14)
    validation = config.get("acquisition_validation", {})
    parity_verified = (len(runs) > 1 and validation.get("exact_physical_sample_sequence")
                       and validation.get("exact_eta_event_sequence"))
    partial_allowed = any(method in PARTIAL_RULES for method in runs)
    acquisition_note = ("Physical samples and η events match across rules; earlier recommendations do not imply lower search spend."
                        if parity_verified else "Curves stop at the last recorded search cost.")
    if mixed_question_orders:
        acquisition_note += " Legend brackets identify shared or independent question order."
    recommendation_note = ("Finite-test rules may recommend partial configurations." if partial_allowed
                           else "Only completed configurations are recommended.")
    fig.text(0.5, 0.012,
             "All quality metrics are offline full-data evaluation, unavailable to the selector. " + recommendation_note + "\n"
             + acquisition_note,
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.90 if len(runs) > 2 else 0.93))
    for extension in ("png", "pdf"):
        fig.savefig(outdir / f"comparison.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def export_plots(saved, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _comparison_curves(saved, outdir)
    manifest, rows = {}, []
    for method, run in saved["runs"].items():
        context = _plot_context(saved.get("config", {}), run)
        points = checkpoint_sequence(run, terminal_label=context["terminal_label"])
        manifest[method] = _render_checkpoints(method, run, points, outdir, config=saved.get("config", {}))
        for point in points:
            row = {"method": method, "checkpoint": point["checkpoint"],
                   "question_order": run.get("question_order", "independent"),
                   "recommendation_min_samples": run.get("recommendation_min_samples", 0),
                   "cost_fraction": point["cost_fraction"], "cost_usd": point["cost_usd"],
                   "evaluations": point["evaluations"], "relative_hv_regret": point["relative_hv_regret"],
                   "contains_true_accuracy_best": point["contains_true_accuracy_best"],
                   "offline_dominated_selected_count": point["offline_dominated_selected_count"]}
            for key in ("selected_arm_indices", "selected_model_names", "selected_config_ids", "selected_sample_counts",
                        "selected_completed_flags", "added_arm_indices", "added_config_ids", "removed_arm_indices",
                        "removed_config_ids", "true_best_accuracy_sample_counts"):
                row[key] = json.dumps(point[key])
            for key in ("finite_target_mean_vectors", "finite_target_std_vectors", "recommendation_raw_vectors"):
                row[key] = json.dumps(point.get(key, []))
            rows.append(row)
    with (outdir / "checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(outdir / "checkpoints.csv", outdir / "pareto_snapshots.csv")
    # The overview alias is explicitly page one; the manifest links every
    # page and the full multipage PDF, without dropping any checkpoint.
    method = saved.get("config", {}).get("recommendation_rule", next(iter(saved["runs"])))
    if method not in saved["runs"]:
        method = next(iter(saved["runs"]))
    shutil.copyfile(outdir / f"{method}_pareto_key_checkpoints.png", outdir / "pareto_snapshots.png")
    shutil.copyfile(outdir / f"{method}_pareto_key_checkpoints_all_pages.pdf", outdir / "pareto_snapshots.pdf")
    (outdir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="Overlay a compatible saved completed-only or finite-test comparison; repeat for multiple references; also render their checkpoints")
    parser.add_argument("--outdir", type=Path,
                        help="Plot output directory; defaults to the primary comparison's directory")
    args = parser.parse_args()
    saved = json.loads(args.comparison.read_text())
    for reference_path in args.reference:
        saved = with_reference(saved, json.loads(reference_path.read_text()))
    print(json.dumps(export_plots(saved, args.outdir or args.comparison.parent), indent=2))


def with_completed_reference(saved, reference):
    """Compatibility wrapper for the original completed-only overlay API."""
    return with_reference(saved, reference, method="completed_only")


def with_reference(saved, reference, *, method=None):
    """Combine compatible saved recommendations without implying trace parity."""
    reference_runs = reference.get("runs", {})
    method = method or reference.get("config", {}).get("recommendation_rule")
    if method is None and len(reference_runs) == 1:
        method = next(iter(reference_runs))
    run_reference = reference_runs.get(method)
    valid_filters = {
        "completed_only": ("all_completed_empirical_raw_pareto",),
        "finite_lcb": ("all_arms_finite_test_lcb_raw_pareto", "eligible_arms_finite_test_lcb_raw_pareto"),
        "finite_mean": ("eligible_arms_finite_test_mean_raw_pareto",),
    }
    if run_reference is None or run_reference.get("recommendation_filter") not in valid_filters.get(method, ()):
        raise ValueError("The plot reference must identify an empirical completed Pareto or finite-test Pareto rule")
    if method in saved["runs"]:
        raise ValueError(f"The primary comparison already contains a {method} reference")
    actual_hashes = {Path(name).name: value for name, value in saved.get("config", {}).get("lookup_sha256", {}).items()}
    expected_hashes = {Path(name).name: value for name, value in reference.get("config", {}).get("lookup_sha256", {}).items()}
    if not actual_hashes or actual_hashes != expected_hashes:
        raise ValueError("Cannot overlay a reference with missing or mismatched lookup hashes")
    for run in saved["runs"].values():
        for key in ("seed", "model_names", "raw_truth_vectors", "common_question_count",
                    "bruteforce_search_cost_usd", "cost_model", "metric_space",
                    "metric_cost_reference_usd", "ground_truth_hypervolume"):
            if run.get(key) != run_reference.get(key):
                raise ValueError(f"Cannot overlay {method} reference with mismatched {key}")
    config = dict(saved.get("config", {}))
    config.pop("acquisition_validation", None)
    # Missing historical fields mean independent, regardless of primary config.
    run_reference = {**run_reference, "question_order": run_reference.get("question_order", "independent")}
    combined = {method: run_reference, **{
        name: {**run, "question_order": run.get("question_order", "independent")}
        for name, run in saved["runs"].items()
    }}
    # Keep comparable panels and annotations in the same order regardless of
    # the CLI order, while the primary config still controls the PDF alias.
    combined = {name: combined[name] for name in LABELS if name in combined}
    return {**saved, "config": config, "runs": combined}


if __name__ == "__main__":
    main()
