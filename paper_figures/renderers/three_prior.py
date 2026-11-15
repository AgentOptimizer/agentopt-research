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

import three_directions as common
VARIANTS = (('fixed', 'warm', 'warm'), ('warm', 'warm', 'fixed'), ('fixed', 'warm', 'fixed'), ('warm', 'fixed', 'warm'), ('fixed', 'fixed', 'warm'), ('warm', 'fixed', 'fixed'), ('fixed', 'fixed', 'fixed'), ('warm', 'warm', 'warm'))

def variant_name(modes: tuple[str, str, str]) -> str:
    return 'q_{}_l_{}_d_{}'.format(*modes)
COLORS = ('tab:orange', '#5254a3', '#637939', '#8c6d31', 'tab:blue', 'tab:cyan', '#888888', 'tab:green')

def main() -> None:
    args=SimpleNamespace(main_quality_prior_mode='warm',output_dir=OUT)
    main_modes = (args.main_quality_prior_mode, 'warm', 'warm')
    alternate_modes = ('warm' if args.main_quality_prior_mode == 'fixed' else 'fixed', 'warm', 'warm')
    all_modes = (main_modes, alternate_modes) + tuple((modes for modes in VARIANTS if modes not in (main_modes, alternate_modes)))
    labels = {modes: 'G{} Q/L/D: {}'.format(index if index == 0 else index + 10, '/'.join(('warm' if value == 'warm' else '0.04' for value in modes))) for index, modes in enumerate(all_modes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.configure_matplotlib()
    summary = DATA / 'three_prior.csv'
    rows = []
    curves = {}
    with summary.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for dataset in common.DATASETS:
        for modes in all_modes:
            for metric, _ in common.METRICS:
                selected = [row for row in rows if row['dataset'] == dataset and row['configuration'] == variant_name(modes) and (row['metric'] == metric)]
                selected.sort(key=lambda row: float(row['cost_fraction']))
                if len(selected) != len(common.GRID) or not np.allclose([float(row['cost_fraction']) for row in selected], common.GRID):
                    raise ValueError(f'Incomplete summary: {dataset}, {modes}, {metric}')
                curves[dataset, modes, metric] = (np.asarray([float(row['mean'] or 'nan') for row in selected]), np.asarray([float(row['two_se'] or 'nan') for row in selected]))
    figure, axes = plt.subplots(2, 3, figsize=(20, 11.2), sharex=True)
    figure.subplots_adjust(left=0.092, right=0.987, top=0.93, bottom=0.295, wspace=0.34, hspace=0.32)
    for row, dataset in enumerate(common.DATASETS):
        for column, (metric, title) in enumerate(common.METRICS):
            axis = axes[row, column]
            upper = 0.0
            for modes, color in zip(all_modes[1:] + all_modes[:1], COLORS[1:] + COLORS[:1]):
                mean, band = curves[dataset, modes, metric]
                main = modes == main_modes
                axis.fill_between(common.GRID, np.maximum(0, mean - band), mean + band, color=color, alpha=0.14 if main else 0.05, linewidth=0, zorder=2 if main else 1)
                axis.plot(common.GRID, mean, color=color, linewidth=3.24 if main else 2.4, zorder=10 if main else 3)
                upper = max(upper, float(np.nanmax(mean + band)))
            axis.set_xlim(-0.005, 0.305)
            axis.set_ylim(-0.025 * upper, 1.06 * upper)
            axis.set_xticks([0, 0.1, 0.2, 0.3])
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            axis.tick_params(axis='both', labelsize=28, width=1, length=5)
            axis.grid(True, alpha=0.22, color='#9da9bb', linewidth=0.7)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(title, fontsize=32, pad=13)
            if column == 0:
                axis.set_ylabel({'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA'}[dataset], fontsize=32, labelpad=11)
    xlabel = figure.supxlabel('Percentage of Exhaustive Evaluation Cost', fontsize=30, y=0.194)
    figure.canvas.draw()
    top = (xlabel.get_window_extent(figure.canvas.get_renderer()).y0 - figure.dpi * 0.15) / figure.bbox.height
    handles = [Line2D([], [], color=color, linewidth=3.24 if modes == main_modes else 2.4, label=labels[modes]) for modes, color in zip(all_modes, COLORS)]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, top), ncol=2, frameon=False, fontsize=25, handlelength=2.3, columnspacing=0.8, handletextpad=0.5, labelspacing=0.55)
    stem = args.output_dir / 'three_objective_prior_variance'
    for extension in ('png', 'pdf', 'svg'):
        figure.savefig(stem.with_suffix('.' + extension), dpi=300, bbox_inches='tight', pad_inches=0.12, facecolor='white')
    plt.close(figure)
    print(stem.with_suffix('.png'))

main()
