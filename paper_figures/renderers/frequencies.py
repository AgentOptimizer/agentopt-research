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

DATASETS = ('hotpotqa', 'restaurant_valid', 'bird_mini_dev', 'bing_querylogs', 'mathqa', 'restaurant_test', 'bird_dev', 'stackoverflow')

DATASET_LABELS = {'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA', 'restaurant_test': 'Restaurant Test', 'stackoverflow': 'Stack Overflow', 'bird_dev': 'BIRD Dev', 'restaurant_valid': 'Restaurant Valid', 'bing_querylogs': 'Bing Query Logs', 'bird_mini_dev': 'BIRD Mini Dev'}

FULL_METHOD_NAME = 'Cost-Coupled Gittins'

@dataclass(frozen=True)
class FrontierSummary:
    counts: Counter[int]
    available: int
    mean_start_usd: float
    mean_end_usd: float
    mean_start_fraction: float
    mean_end_fraction: float

def nondominated_indices(points: np.ndarray) -> np.ndarray:
    """Indices not dominated when both coordinates are maximized."""
    array = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)

def raw_front_indices(truth: np.ndarray) -> np.ndarray:
    metric = np.column_stack((truth[:, 0], -truth[:, 1]))
    return nondominated_indices(metric)

def _mix(color: tuple[float, float, float], other: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(((1.0 - amount) * value + amount * target for value, target in zip(color, other)))

def _frequency_colormap(color: str='tab:orange', name: str='recommendation_frequency') -> LinearSegmentedColormap:
    base = to_rgb(color)
    return LinearSegmentedColormap.from_list(name, (_mix(base, (1, 1, 1), 0.82), _mix(base, (1, 1, 1), 0.45), base, _mix(base, (0, 0, 0), 0.38)), N=256)

def _usd(value: float) -> str:
    if value < 1.0:
        return f'${value:.2f}'
    if value < 100.0:
        return f'${value:.1f}'
    return f'${value:,.0f}'

def draw_frontier_panel(axis: plt.Axes, truth: np.ndarray, summary: FrontierSummary, title: str, cmap: mpl.colors.Colormap, *, title_fontsize: float=37, subtitle_fontsize: float=33, tick_labelsize: float=34, show_seed_count: bool=True, show_usd_interval: bool=True) -> None:
    frontier = raw_front_indices(truth)
    frontier = frontier[np.argsort(truth[frontier, 1])]
    axis.scatter(truth[:, 1], truth[:, 0], s=48, color='#c0c5cc', alpha=0.64, edgecolors='none', zorder=1)
    axis.plot(truth[frontier, 1], truth[frontier, 0], color='#25282c', linewidth=1.7, zorder=2)
    axis.scatter(truth[frontier, 1], truth[frontier, 0], s=72, facecolors='white', edgecolors='#25282c', linewidths=1.25, zorder=3)
    shown = np.asarray(sorted(summary.counts), dtype=int)
    if len(shown):
        frequencies = np.asarray([summary.counts[int(index)] for index in shown])
        axis.scatter(truth[shown, 1], truth[shown, 0], c=frequencies, cmap=cmap, norm=mpl.colors.Normalize(vmin=1, vmax=20), s=175, edgecolors='#6f4a22', linewidths=1.1, zorder=4)
    shown_title = title if summary.available == 20 or not show_seed_count else f'{title}  [n={summary.available}/20]'
    axis.text(0.5, 1.24, shown_title, transform=axis.transAxes, ha='center', va='bottom', fontsize=title_fontsize)
    if summary.available:
        if show_usd_interval:
            start_usd = _usd(summary.mean_start_usd).replace('$', '\\$')
            end_usd = _usd(summary.mean_end_usd).replace('$', '\\$')
            subtitle = f'{start_usd}–{end_usd} ({summary.mean_start_fraction:.1%}–{summary.mean_end_fraction:.1%})'
        else:
            subtitle = f'{summary.mean_start_fraction:.1%}–{summary.mean_end_fraction:.1%}'
    else:
        subtitle = 'pending'
    axis.text(0.5, 1.19, subtitle, transform=axis.transAxes, ha='center', va='top', fontsize=subtitle_fontsize)
    if np.all(truth[:, 1] > 0.0):
        axis.set_xscale('log')
    axis.grid(color='#d9dde3', linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
    axis.tick_params(axis='both', labelsize=tick_labelsize)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axis.xaxis.set_minor_formatter(NullFormatter())

def finish_frontier_figure(figure: plt.Figure, *, checkpoint: float, cmap: mpl.colors.Colormap, method_label: str, stem: Path) -> None:
    figure.supxlabel('Mean deployment cost (USD per query, log scale)', fontsize=40, y=0.08)
    figure.supylabel('Mean quality', fontsize=40, x=0.015)
    figure.suptitle(f'{method_label} recommendations at {checkpoint:.0%} of brute-force search cost', fontsize=46, y=0.918)
    figure.legend(handles=(Line2D([], [], linestyle='none', marker='o', markersize=13, markerfacecolor='#c0c5cc', markeredgecolor='none', label='All configurations'), Line2D([], [], color='#25282c', linewidth=2, marker='o', markersize=10, markerfacecolor='white', markeredgewidth=1.25, label='Full-data Pareto frontier'), Line2D([], [], linestyle='none', marker='o', markersize=15, markerfacecolor=cmap(0.6), markeredgecolor='#6f4a22', label='Recommendations (color = frequency)')), loc='lower center', bbox_to_anchor=(0.48, 0.008), ncol=3, frameon=False, fontsize=38, columnspacing=2.2, handletextpad=0.65)
    figure.subplots_adjust(left=0.08, right=0.865, bottom=0.16, top=0.78, wspace=0.26, hspace=0.56)
    colorbar_axis = figure.add_axes([0.9, 0.2, 0.021, 0.6])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, 20), cmap=cmap), cax=colorbar_axis)
    colorbar.set_label('Recommendation frequency (out of 20 seeds)', fontsize=38, labelpad=19)
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=34)
    colorbar.outline.set_linewidth(0.9)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix('.png'), dpi=200, facecolor='white')
    figure.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(figure)

def configure_matplotlib() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})

configure_matplotlib()
truth=json.loads((DATA/'frequency_truth.json').read_text())
for budget in (.1,.3):
    rows=list(csv.DictReader((DATA/f'frequency_{round(100*budget)}pct.csv').open()))
    cmap=_frequency_colormap();fig,axes=plt.subplots(2,4,figsize=(28,19))
    for ax,ds in zip(axes.flat,DATASETS):
        rr=[r for r in rows if r['dataset']==ds]
        assert len(rr)==20
        counts=Counter(i for r in rr for i in json.loads(r['selected_arm_indices']))
        summary=FrontierSummary(counts,20,*[float(np.mean([float(r[k]) for r in rr])) for k in
            ('snapshot_cost_usd','membership_end_cost_usd','snapshot_cost_fraction','membership_end_cost_fraction')])
        draw_frontier_panel(ax,np.array(truth[ds]),summary,DATASET_LABELS[ds],cmap)
        ax.set_box_aspect(1)
    finish_frontier_figure(fig,checkpoint=budget,cmap=cmap,method_label=FULL_METHOD_NAME,
        stem=OUT/f'gittins_20seed_frequency_frontiers_{round(100*budget)}pct')
