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

IDS = {'four_direction_real_cost': 'G0', 'q_only_real_cost': 'G1', 'l_only_real_cost': 'G2', 'd_only_real_cost': 'G3', 'pure_axes_real_cost': 'G4', 'six_ql_dense_real_cost': 'G5', 'six_three_edges_real_cost': 'G6', 'q_only_unit_cost': 'G7', 'pure_axes_unit_cost': 'G8', 'four_direction_unit_cost': 'G9', 'four_direction_fixed_eta': 'G10'}
SEEDS = tuple(range(42, 62))

DATASETS = ('hotpotqa', 'mathqa')

DATASET_LABELS = {'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA'}

GRID = np.arange(1, 61, dtype=np.float64) / 200.0

CURVE_LINEWIDTH = 2.4

HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

CONFIGURATIONS = ('q_only_real_cost', 'l_only_real_cost', 'd_only_real_cost', 'pure_axes_real_cost', 'four_direction_real_cost', 'six_ql_dense_real_cost', 'six_three_edges_real_cost')

STYLES: dict[str, dict[str, str | float]] = {'q_only_real_cost': {'color': 'tab:gray', 'label': 'Q-only', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'l_only_real_cost': {'color': 'tab:brown', 'label': 'L-only', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'd_only_real_cost': {'color': 'tab:olive', 'label': 'D-only', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'pure_axes_real_cost': {'color': 'tab:blue', 'label': 'Pure axes', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'four_direction_real_cost': {'color': 'tab:orange', 'label': 'Four directions (+ Q-L midpoint)', 'linewidth': HIGHLIGHT_LINEWIDTH, 'linestyle': '-'}, 'six_ql_dense_real_cost': {'color': 'tab:green', 'label': 'Six directions (Q-L dense)', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'six_three_edges_real_cost': {'color': 'tab:purple', 'label': 'Six directions (three edges)', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}}

METRICS = (('hv_regret', 'HV Regret'), ('generational_distance', 'GD'), ('inverted_generational_distance', 'IGD'))

NATURAL_STOPS = {'direction_eta_numerical_floor', 'all_arms_completed', 'all_cells_observed', 'question_budget', 'all_arms_exhausted'}

def load_curve_summary(path: Path) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if not row['mean'] or not row['two_se']:
                continue
            key = (row['dataset'], row['configuration'], row['metric'])
            grouped.setdefault(key, []).append((float(row['cost_fraction']), float(row['mean']), float(row['two_se']), int(row['n_runs'])))
    curves = {}
    for key, rows in grouped.items():
        array = np.asarray(rows, dtype=np.float64)
        curves[key] = (array[:, 0], array[:, 1], array[:, 2], array[:, 3])
    return curves

def configure_matplotlib() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})

def plot(summary_path: Path, output_stem: Path) -> None:
    curves = load_curve_summary(summary_path)
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 9.0), squeeze=False)
    percent = FuncFormatter(lambda value, _: f'{value:.0%}')
    compact = FuncFormatter(lambda value, _: f'{value:g}')
    for row, dataset in enumerate(DATASETS):
        for column, (field, title) in enumerate(METRICS):
            axis = axes[row, column]
            for configuration in CONFIGURATIONS:
                xs, means, two_se, _ = curves[dataset, configuration, field]
                style = STYLES[configuration]
                axis.plot(xs, means, color=str(style['color']), linewidth=float(style['linewidth']), linestyle=str(style['linestyle']), zorder=5 if configuration == 'four_direction_real_cost' else 3)
                axis.fill_between(xs, np.maximum(0.0, means - two_se), means + two_se, color=str(style['color']), alpha=0.1, linewidth=0, zorder=1)
            if row == 0:
                axis.set_title(title, fontsize=28, pad=13)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis='both', labelsize=27, width=1.0, length=5)
            axis.grid(color='#D5D9DE', linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)
    subplot_bottom = 0.37
    subplot_top = 0.925
    figure.text(0.535, 0.265, 'Percentage of Exhaustive Evaluation Cost', ha='center', va='center', fontsize=30)
    figure.subplots_adjust(left=0.085, right=0.985, top=subplot_top, bottom=subplot_bottom, wspace=0.3, hspace=0.34)
    row_centers = [(axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0 for row in range(2)]
    for row, dataset in enumerate(DATASETS):
        figure.text(0.02, row_centers[row], DATASET_LABELS[dataset], ha='center', va='center', rotation=90, fontsize=31)
    handles = [Line2D([], [], color=str(STYLES[configuration]['color']), linewidth=float(STYLES[configuration]['linewidth']), linestyle=str(STYLES[configuration]['linestyle']), label=str(STYLES[configuration]['label'])) for configuration in CONFIGURATIONS]
    handles.append(Patch(facecolor='#777777', alpha=0.14, edgecolor='none', label='$\\pm2$ SE'))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=30, handlelength=2.8, columnspacing=1.0, handletextpad=0.55, labelspacing=0.45)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {'facecolor': 'white', 'bbox_inches': 'tight', 'pad_inches': 0.12}
    figure.savefig(output_stem.with_suffix('.png'), dpi=300, **save_options)
    figure.savefig(output_stem.with_suffix('.pdf'), **save_options)
    figure.savefig(output_stem.with_suffix('.svg'), **save_options)
    plt.close(figure)

for cfg,st in STYLES.items():st['label']=IDS[cfg]+' '+st['label']
if __name__=='__main__':
    configure_matplotlib()
    plot(DATA/'three_directions.csv',OUT/'three_objective_directions_affine_legend')
