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

registry = json.loads((DATA / "three_color_map.json").read_text())
rows = list(csv.DictReader((DATA/'three_direction_runtime.csv').open(newline="", encoding="utf-8")))
variants = [
    ("four_direction_real_cost", "G0 Four directions", 4),
    ("q_only_real_cost", "G1 One direction (Q)", 1),
    ("l_only_real_cost", "G2 One direction (L)", 1),
    ("d_only_real_cost", "G3 One direction (D)", 1),
    ("pure_axes_real_cost", "G4 Three directions", 3),
    ("six_ql_dense_real_cost", "G5 Six directions (Q-L dense)", 6),
    ("six_three_edges_real_cost", "G6 Six directions (three edges)", 6),
]
assert len(rows) == 14
assert all(int(r["n_runs"]) == 20 and float(r["mean_total_evaluations"]) == 6000
           and float(r["mean_realized_entry_fraction"]) == 0.3 for r in rows)
index = {(r["configuration"], r["dataset"]): r for r in rows}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8, "svg.fonttype": "none",
})
fig, axes = plt.subplots(1, 2, figsize=(18.5, 8.5), squeeze=False)
x = np.arange(len(variants), dtype=float)
colors = [registry["colors"][registry["configuration_ids"][name]] for name, _, _ in variants]
for ax, dataset, label in zip(axes.flat, ["hotpotqa", "mathqa"], ["HotpotQA", "MathQA"]):
    means = np.array([float(index[name, dataset]["mean_seconds"]) for name, _, _ in variants])
    errors = np.array([float(index[name, dataset]["error_2se_seconds"]) for name, _, _ in variants])
    ax.bar(x, means, yerr=errors, width=0.72, color=colors, edgecolor="black", linewidth=0.7,
           error_kw={"ecolor": "black", "elinewidth": 1.1, "capsize": 4.0, "capthick": 1.1}, zorder=3)
    ax.set_title(label, fontsize=25, pad=9)
    ax.set_xlim(-0.65, len(variants) - 0.35)
    ax.set_xticks([])
    ax.set_ylim(0.0, float(np.max(means + errors)) * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", labelsize=23, width=1.0, length=5)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
handles = [Patch(facecolor=color, edgecolor="black", linewidth=0.7, label=label)
           for (_, label, _), color in zip(variants, colors)]
handles.append(axes[0, 0].errorbar([np.nan], [np.nan], yerr=[1.0], fmt="none", ecolor="black",
                                elinewidth=1.2, capsize=4.0, capthick=1.2, label=r"$\pm 2$ SE"))
legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06),
                    ncol=4, frameon=False, fontsize=23, columnspacing=0.8,
                    handlelength=1.7, handletextpad=0.5)
fig.supylabel("Mean policy wall-clock runtime (s)", fontsize=27, x=0.010, y=0.55)
fig.suptitle("Direction Runtime Comparison", fontsize=29, y=0.99)
fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.25, wspace=0.34)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_box = legend.get_window_extent(renderer)
if legend_box.width > fig.bbox.width - 30:
    fig.set_figwidth((legend_box.width + 60) / fig.dpi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
assert legend_box.x0 >= 0 and legend_box.x1 <= fig.bbox.width
assert all("\n" not in label for _, label, _ in variants)
stem = OUT / "three_objective_direction_runtime_seven_settings"
for ext in ["png", "pdf", "svg"]:
    fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
