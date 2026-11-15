"""Plot held-out Pareto metrics from a replay curve summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
import numpy as np


BENCHMARKS = {
    "hotpotqa": "HotpotQA",
    "restaurant_valid": "Restaurant Valid",
    "bird_mini_dev": "BIRD Mini Dev",
    "bing_querylogs": "Bing Query Logs",
    "mathqa": "MathQA",
    "restaurant_test": "Restaurant Test",
    "bird_dev": "BIRD Dev",
    "stackoverflow": "Stack Overflow",
}
STYLES = {
    "cc_gittins": ("CC-Gittins", "#ff7f0e", "-", "o"),
    "random_questions": ("Random Questions", "#17becf", (0, (5, 2)), "s"),
    "cohv_ucb": ("CoHV-UCB", "#bcbd22", (0, (4, 1.5, 1, 1.5)), "^"),
    "cost_aware_scalarization_ucb": ("Scalarization UCB", "#9467bd", (0, (1.5, 1.3)), "D"),
}
METRIC_TITLES = {
    "hv_regret": "Held-out hypervolume regret",
    "gd": "Held-out generational distance (GD)",
    "igd": "Held-out inverted generational distance (IGD)",
}
REFERENCE_COLOR = "#777777"
HV_LINEAR_THRESHOLD = 0.10
HV_TICKS = np.array([0.0, 0.025, 0.05, 0.10, 0.30, 1.0])


def draw_panel(ax, caps, data, methods, metric, title):
    reference = data["exhaustive_search_heldout_metrics"][metric]
    if not np.isfinite(reference) or reference < 0:
        raise ValueError(f"{title}: invalid {metric} reference")
    ax.axhline(reference, color=REFERENCE_COLOR, linestyle=":", linewidth=1.7, zorder=2)
    maximum = reference

    available = np.ones(caps.shape, dtype=bool)
    if metric != "hv_regret":
        warm_cap = data["all_warm_cap"]
        available = np.zeros(caps.shape, dtype=bool) if warm_cap is None else caps >= warm_cap - 1e-12
        first = caps[available][0] if available.any() else caps[-1]
        if first > 0:
            ax.axvspan(0, first, facecolor="#f0f0f0", edgecolor="none", zorder=0)

    for method in methods:
        _, color, linestyle, marker = STYLES[method]
        values = np.asarray(data["methods"][method][metric], dtype=float)
        if (values.shape != caps.shape or not np.isfinite(values[available]).all()
                or (values[available] < 0).any() or not np.isnan(values[~available]).all()):
            raise ValueError(f"{title}, {method}: invalid {metric} curve")
        if not available.any():
            continue
        x, y = caps[available], values[available]
        maximum = max(maximum, float(y.max()))
        ax.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=1.9, zorder=3)
        checkpoints = np.isin(x, (0.10, 0.30))
        ax.plot(x[checkpoints], y[checkpoints], color=color, marker=marker, linestyle="none",
                markersize=4.8, markeredgecolor="white", markeredgewidth=0.65,
                clip_on=False, zorder=4)

    if not available.any():
        ax.text(0.5, 0.49, "Some runs do not complete warm-up by 30%",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#585858")

    top = max(maximum, 1e-4) * 1.10
    if metric == "hv_regret" and top > HV_LINEAR_THRESHOLD:
        ax.set_yscale("symlog", linthresh=HV_LINEAR_THRESHOLD)
        ax.set_yticks(HV_TICKS[HV_TICKS <= top])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    ax.set_xlim(caps[0], caps[-1])
    ax.set_xticks([0.0, 0.10, 0.20, 0.30])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, top)
    ax.tick_params(axis="both", labelsize=11, color="#777777", length=3.5)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["left", "bottom"]].set_color("#9a9a9a")


def plot_metric(caps, data, fold_count, seed_count, metric, directory):
    order = [key for key in BENCHMARKS if key in data]
    methods = [key for key in STYLES if key in data[order[0]]["methods"]]
    if not methods or any(set(data[key]["methods"]) != set(methods) for key in order):
        raise ValueError("Expected the same supported methods in every benchmark")

    legend = [
        Line2D([], [], color=color, linestyle=linestyle, marker=marker, linewidth=2.0,
               markersize=5.0, label=name)
        for name, color, linestyle, marker in (STYLES[method] for method in methods)
    ]
    legend.append(Line2D([], [], color=REFERENCE_COLOR, linestyle=":", linewidth=1.7,
                         label="Exhaustive search-pool frontier (held out)"))
    columns = min(len(order), 4)
    rows = math.ceil(len(order) / columns)
    legend_columns = min(len(legend), {1: 2, 2: 3, 3: 4, 4: 5}[columns])
    legend_rows = math.ceil(len(legend) / legend_columns)
    width = max(6.8, 3.7 * columns)
    height = 2.5 * rows + 2.15 + 0.22 * (legend_rows - 1)
    fig, axes = plt.subplots(rows, columns, sharex=True, squeeze=False, figsize=(width, height))
    fig.subplots_adjust(left=0.82 / width, right=1 - 0.24 / width,
                        bottom=0.95 / height, top=1 - (1.18 + 0.22 * (legend_rows - 1)) / height,
                        wspace=0.37, hspace=0.42)
    title = METRIC_TITLES[metric]
    fig.suptitle(title, y=1 - 0.15 / height, fontsize=20, fontweight="bold")
    fig.supylabel("Metric value (lower is better)", x=0.18 / width, y=0.49, fontsize=13)
    fig.supxlabel("Search-cost cap (% of exhaustive search-pool cost)", y=0.47 / height, fontsize=13)
    fig.legend(handles=legend, ncol=legend_columns, loc="upper center",
               bbox_to_anchor=(0.52, 1 - 0.50 / height), frameon=False,
               fontsize=10 if columns < 3 else 12, handlelength=2.3, handletextpad=0.42,
               columnspacing=1.35)

    folds_label = "fold" if fold_count == 1 else "folds"
    seeds_label = "seed" if seed_count == 1 else "seeds"
    cohort = f"{fold_count} {folds_label}, {seed_count} search {seeds_label} per fold."
    if len(order) > 1:
        cohort += " Vertical ranges vary by dataset."
    fig.text(0.5, 0.31 / height, cohort, ha="center", fontsize=10, color="#484848")
    detail = (
        "HV regret uses an empty set before warm-up. Vertical scales are logarithmic above 0.10."
        if metric == "hv_regret" else "Shading marks the budgets before every run has a recommendation."
    )
    fig.text(0.5, 0.14 / height, detail, ha="center", fontsize=10, color="#484848")

    for index, ax in enumerate(axes.flat):
        if index >= len(order):
            ax.set_visible(False)
            continue
        key = order[index]
        draw_panel(ax, caps, data[key], methods, metric, BENCHMARKS[key])
        ax.tick_params(axis="x", labelbottom=index + columns >= len(order))

    for extension in ("png", "pdf"):
        destination = directory / f"heldout_{metric}_curve.{extension}"
        fig.savefig(destination, dpi=185, metadata={"Title": title})
        print(f"Saved {destination}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="Directory containing curve_summary.json")
    args = parser.parse_args()
    try:
        summary = json.loads((args.outdir / "curve_summary.json").read_text(encoding="utf-8"))
        caps = np.asarray(summary["caps"], dtype=float)
        if (caps.ndim != 1 or caps.size < 2 or not np.isfinite(caps).all()
                or not np.all(np.diff(caps) > 0) or caps[0] != 0 or caps[-1] != 0.30):
            raise ValueError("Expected increasing search-cost caps from 0% to 30%")
        data = summary["benchmarks"]
        if not data or not set(data) <= set(BENCHMARKS):
            raise ValueError("Summary has no benchmarks or contains unrecognized benchmark IDs")
        fold_count = len(next(iter(data.values()))["folds"])
        seed_count = len(summary["search_seeds"])
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "text.color": "#202020",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        for metric in METRIC_TITLES:
            plot_metric(caps, data, fold_count, seed_count, metric, args.outdir)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Held-out plotting failed: {error}\n")


if __name__ == "__main__":
    main()
