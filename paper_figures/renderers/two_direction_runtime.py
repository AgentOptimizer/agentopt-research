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

G0_CONFIGURATION='g0_current_gittins'
PANEL_ORDER = ('hotpotqa', 'restaurant_valid', 'bird_mini_dev', 'bing_querylogs', 'mathqa', 'restaurant_test', 'bird_dev', 'stackoverflow')

DATASET_LABELS = {'restaurant_test': 'Restaurant Test', 'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA', 'stackoverflow': 'Stack Overflow', 'bird_dev': 'BIRD Dev', 'restaurant_valid': 'Restaurant Valid', 'bing_querylogs': 'Bing Query Logs', 'bird_mini_dev': 'BIRD Mini Dev'}

CONFIG_COLORS = {G0_CONFIGURATION: '#e67e22', 'g1_q_only_real_cost': '#1f77b4', 'g2_q_only_unit_cost': '#2ca02c', 'g3_two_axis_unit_cost': '#e377c2', 'g4_d_only_real_cost': '#9467bd', 'g5_axes_midpoint_real_cost': '#8c564b', 'g6_five_directions_real_cost': '#17becf', 'g7_fixed_eta_no_stop': '#7f7f7f', 'g8_q_then_d': '#bcbd22', 'g9_d_then_q': '#003f5c'}

def configure_matplotlib() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})
CONFIGURATIONS = (G0_CONFIGURATION, 'g5_axes_midpoint_real_cost', 'g6_five_directions_real_cost')

CONFIGURATION_LABELS = {G0_CONFIGURATION: 'G0 Two directions', 'g5_axes_midpoint_real_cost': 'G5 Three directions', 'g6_five_directions_real_cost': 'G6 Five directions'}

def plot_runtime(summary: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    values = {(str(row['configuration']), str(row['dataset'])): row for row in summary}
    configure_matplotlib()
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), squeeze=False)
    x = np.arange(len(CONFIGURATIONS), dtype=float)
    for axis, dataset in zip(axes.flat, PANEL_ORDER):
        means = np.asarray([values[name, dataset]['mean_seconds'] for name in CONFIGURATIONS], dtype=float)
        errors = np.asarray([values[name, dataset]['error_2se_seconds'] for name in CONFIGURATIONS], dtype=float)
        axis.bar(x, means, yerr=errors, width=0.72, color=[CONFIG_COLORS[name] for name in CONFIGURATIONS], edgecolor='black', linewidth=0.7, error_kw={'ecolor': 'black', 'elinewidth': 1.1, 'capsize': 4.0, 'capthick': 1.1}, zorder=3)
        axis.set_title(DATASET_LABELS[dataset], fontsize=25, pad=9)
        axis.set_xlim(-0.65, len(CONFIGURATIONS) - 0.35)
        axis.set_xticks([])
        axis.set_ylim(0.0, float(np.max(means + errors)) * 1.16)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.tick_params(axis='y', labelsize=23, width=1.0, length=5)
        axis.grid(axis='y', color='#d9dde3', linewidth=0.6, alpha=0.55)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
        axis.set_axisbelow(True)
    legend_handles = [Patch(facecolor=CONFIG_COLORS[name], edgecolor='black', linewidth=0.7, label=CONFIGURATION_LABELS[name]) for name in CONFIGURATIONS]
    legend_handles.append(axes[0, 0].errorbar([np.nan], [np.nan], yerr=[1.0], fmt='none', ecolor='black', elinewidth=1.2, capsize=4.0, capthick=1.2, label='$\\pm 2$ SE'))
    figure.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.115), ncol=len(legend_handles), frameon=False, fontsize=23, columnspacing=0.8, handlelength=1.7, handletextpad=0.5)
    figure.supylabel('Mean policy wall-clock runtime (s)', fontsize=27, x=0.01, y=0.585)
    figure.suptitle('Direction Runtime Comparison', fontsize=29, y=1.055)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.25, wspace=0.34, hspace=0.42)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'direction_runtime_comparison_20seed'
    png_path = stem.with_suffix('.png')
    pdf_path = stem.with_suffix('.pdf')
    figure.savefig(png_path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    figure.savefig(pdf_path, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    plt.close(figure)
    return (png_path, pdf_path)

plot_runtime(list(csv.DictReader((DATA/'two_direction_runtime.csv').open())),OUT)
