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

BETAS = (0.0, 1.0)

COLORS = {0.0: '#1f77b4', 1.0: '#ff7f0e'}

METRICS = (('precision', 'Precision $\\uparrow$'), ('recall', 'Recall $\\uparrow$'), ('gd', 'GD $\\downarrow$'), ('igd', 'IGD $\\downarrow$'), ('hv_regret', 'HV Regret $\\downarrow$'))

DATASETS = ('hotpotqa', 'mathqa')
def main():
    args=SimpleNamespace(input=DATA/'recommendation',dimensions=[2,3])
    with (args.input / "curve_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 24, "axes.titlesize": 28,
        "axes.labelsize": 28, "xtick.labelsize": 23, "ytick.labelsize": 23,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .8,
    })
    for dims in args.dimensions:
        fig, axes = plt.subplots(2, len(METRICS), figsize=(21.2, 7.8), sharex=True)
        fig.subplots_adjust(left=.065, right=.975, top=.91, bottom=.25,
                            wspace=.42, hspace=.46)
        for row, dataset in enumerate(DATASETS):
            for col, (metric, title) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_box_aspect(1 / 1.58)
                for beta in BETAS:
                    records = sorted((r for r in rows if int(r["dimensions"]) == dims
                                      and r["dataset"] == dataset and r["metric"] == metric
                                      and float(r["beta"]) == beta),
                                     key=lambda r: float(r["cost_fraction"]))
                    if not records:
                        raise ValueError(f"Missing {dims}/{dataset}/{metric}/{beta}")
                    # Start once all seeds have completed warm start. Do not
                    # change the seed population over the plotted curve.
                    n = max(int(r["n_runs"]) for r in records)
                    records = [r for r in records if int(r["n_runs"]) == n]
                    x, y, band = np.array([[float(r[k]) for k in ("cost_fraction", "mean", "two_se")]
                                           for r in records]).T
                    lower, upper = np.maximum(0, y-band), y+band
                    if metric in ("precision", "recall"):
                        upper = np.minimum(1, upper)
                    ax.fill_between(x, lower, upper, color=COLORS[beta], alpha=.11,
                                    linewidth=0, zorder=1)
                    ax.plot(x, y, color=COLORS[beta], lw=2.6 if beta == 1 else 1.9,
                            zorder=5 if beta == 1 else 3)
                ax.set_xlim(-.003, .303)
                ax.set_xticks([0., .1, .2, .3])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.set_ylim(bottom=0)
                if metric in ("precision", "recall"):
                    ax.set_ylim(0, 1.03)
                # Match the x-axis's 1% origin padding so the zero level is
                # slightly above the bottom spine instead of touching it.
                ax.set_ylim(bottom=-.01 * ax.get_ylim()[1])
                ax.grid(True, color="#b2becb", alpha=.24, lw=.6)
                ax.set_axisbelow(True)
                if row == 0:
                    ax.set_title(title, pad=10)
                if col == 0:
                    ax.set_ylabel("HotpotQA" if dataset == "hotpotqa" else "MathQA", labelpad=13)
        fig.supxlabel("Percentage of Exhaustive Evaluation Cost", fontsize=28, y=.14)
        labels = {1.: r"G0: LCB/UCB ($\beta=1$)",
                  0.: rf"G{13 if dims == 2 else 18}: Mean only ($\beta=0$)"}
        handles = [Line2D([], [], color=COLORS[b], lw=2.6, label=labels[b]) for b in (1., 0.)]
        handles.append(Patch(facecolor="#aaaaaa", alpha=.25, label=r"$\pm 2$ SE"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .023),
                   frameon=False, ncol=3, fontsize=28, handlelength=2.3, columnspacing=1.7)
        stem = OUT / f"recommendation_confidence_{dims}d"
        fig.savefig(stem.with_suffix(".png"), dpi=200)
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
        print(stem.with_suffix(".png"))


main()
