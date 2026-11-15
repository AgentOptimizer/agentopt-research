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

CURVE_LINEWIDTH = 2.4
HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

DATASETS = ('hotpotqa', 'mathqa')

DATASET_LABELS = {'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA'}

METHODS = ('cc_gittins', 'ege_sh', 'ape_k', 'qnehvi', 'random_configurations', 'random_questions')

STYLES: dict[str, dict[str, str | float]] = {'cc_gittins': {'color': 'tab:orange', 'label': 'CC-Gittins', 'linewidth': HIGHLIGHT_LINEWIDTH, 'linestyle': '-'}, 'ege_sh': {'color': 'tab:green', 'label': 'EGE-SH', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'ape_k': {'color': 'tab:purple', 'label': 'APE-k', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'qnehvi': {'color': 'tab:pink', 'label': 'qNEHVI', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'random_configurations': {'color': 'tab:brown', 'label': 'Random configurations', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}, 'random_questions': {'color': 'tab:cyan', 'label': 'Random questions', 'linewidth': CURVE_LINEWIDTH, 'linestyle': '-'}}

METRICS = (('hv_regret', 'HV Regret', 1.0), ('generational_distance', 'GD', 1.0), ('inverted_generational_distance', 'IGD', 1.0))

NATURAL_STOPS = {'direction_eta_numerical_floor', 'all_arms_completed', 'all_cells_observed', 'question_budget', 'all_arms_exhausted'}

def configure_matplotlib() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})

def load_summary(path: Path) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if not row['mean'] or not row['two_se']:
                continue
            key = (row['dataset'], row['method'], row['metric'])
            grouped.setdefault(key, []).append((float(row['cost_fraction']), float(row['mean']), float(row['two_se']), int(row['n_runs'])))
    output = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        output[key] = (array[:, 0], array[:, 1], array[:, 2], array[:, 3])
    return output

def plot(summary_path: Path, output_stem: Path) -> None:
    curves = load_summary(summary_path)
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 11.7), squeeze=False)
    percent = FuncFormatter(lambda value, _: f'{value:.0%}')
    compact = FuncFormatter(lambda value, _: f'{value:g}')
    for row, dataset in enumerate(DATASETS):
        for column, (field, title, _) in enumerate(METRICS):
            axis = axes[row, column]
            for method in METHODS:
                xs, means, two_se, _ = curves[dataset, method, field]
                style = STYLES[method]
                is_random = method in {'random_configurations', 'random_questions'}
                axis.plot(xs, means, color=str(style['color']), linewidth=float(style['linewidth']), linestyle=str(style['linestyle']), zorder=5 if method == 'cc_gittins' else 3, drawstyle='steps-post' if is_random else 'default')
                axis.fill_between(xs, np.maximum(0.0, means - two_se), means + two_se, color=str(style['color']), alpha=0.1, linewidth=0, zorder=1, step='post' if is_random else None)
            if row == 0:
                axis.set_title(title, fontsize=33, pad=16)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis='both', labelsize=30, width=1.0, length=5)
            axis.grid(color='#D5D9DE', linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)
            y_top = axis.get_ylim()[1]
            axis.set_ylim(-0.02 * y_top, y_top)
    subplot_bottom = 0.26
    subplot_top = 0.925
    figure.text(0.535, 0.165, 'Percentage of Exhaustive Evaluation Cost', ha='center', va='center', fontsize=33)
    figure.subplots_adjust(left=0.085, right=0.985, top=subplot_top, bottom=subplot_bottom, wspace=0.3, hspace=0.34)
    row_centers = [(axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0 for row in range(2)]
    figure.text(0.02, row_centers[0], DATASET_LABELS[DATASETS[0]], ha='center', va='center', rotation=90, fontsize=33)
    figure.text(0.02, row_centers[1], DATASET_LABELS[DATASETS[1]], ha='center', va='center', rotation=90, fontsize=33)
    handles = [Line2D([], [], color=str(STYLES[method]['color']), linewidth=float(STYLES[method]['linewidth']), linestyle=str(STYLES[method]['linestyle']), label=str(STYLES[method]['label'])) for method in METHODS]
    handles = [handles[i] for i in [0,4,1,5,2,6,3,7]]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, fontsize=28, handlelength=1.5, columnspacing=0.75, handletextpad=0.5, labelspacing=0.65)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {'facecolor': 'white', 'bbox_inches': 'tight', 'pad_inches': 0.12}
    figure.savefig(output_stem.with_suffix('.png'), dpi=300, **save_options)
    figure.savefig(output_stem.with_suffix('.pdf'), **save_options)
    figure.savefig(output_stem.with_suffix('.svg'), **save_options)
    plt.close(figure)

METHODS=['cc_gittins','ege_sh','ape_k','cost_aware_qnehvi','random_configurations','random_questions','copsi30','cohv_ucb']
STYLES['cost_aware_qnehvi']=dict(STYLES['qnehvi'],label='qNEHVI-PC')
STYLES['copsi30']=dict(color='tab:blue',label='CoPSI (30% budget)',linewidth=2.4,linestyle='-')
STYLES['cohv_ucb']=dict(color='tab:olive',label='CoHV-UCB',linewidth=2.4,linestyle='-')
configure_matplotlib()
plot(DATA/'three_main.csv', OUT/'three_objective_eight_methods')
