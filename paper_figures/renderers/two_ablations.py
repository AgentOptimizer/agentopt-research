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


def widen(fig):
    width,height=fig.get_size_inches();extra=3.5
    boxes=[ax.get_position().frozen() for ax in fig.axes]
    textpos=[t.get_position() for t in fig.texts]
    legends=[l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).frozen() for l in fig.legends]
    fig.set_size_inches(width+extra,height)
    for k,(ax,b) in enumerate(zip(fig.axes,boxes)):
        ax.set_position([(b.x0*width+(k%3)*extra/2)/(width+extra),b.y0,b.width*width/(width+extra),b.height])
    for t,(x,y) in zip(fig.texts,textpos):
        t.set_position(((x*width+(extra/2 if x>.2 else 0))/(width+extra),y))
    for l,b in zip(fig.legends,legends):
        l.set_bbox_to_anchor(((b.x0*width+extra/2)/(width+extra),b.y0,b.width*width/(width+extra),b.height),transform=fig.transFigure)
G0_CONFIGURATION='g0_current_gittins'
CONFIGURATIONS = {G0_CONFIGURATION: {'pair_name': 'exact_axes', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G0 Current (axes, real cost)', 'source_configuration': 'g2_exact_axes'}, 'g1_q_only_real_cost': {'pair_name': 'quality_only', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G1 Q-only, real cost'}, 'g2_q_only_unit_cost': {'pair_name': 'quality_only', 'acquisition_cost_mode': 'unit', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G2 Q-only, unit cost'}, 'g3_two_axis_unit_cost': {'pair_name': 'exact_axes', 'acquisition_cost_mode': 'unit', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G3 Axes, unit cost'}, 'g4_d_only_real_cost': {'pair_name': 'deployment_only', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G4 D-only, real cost'}, 'g5_axes_midpoint_real_cost': {'pair_name': 'axes_midpoint', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G5 Axes + midpoint'}, 'g6_five_directions_real_cost': {'pair_name': 'five_directions', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'round_robin', 'label': 'G6 Five directions'}, 'g7_fixed_eta_no_stop': {'pair_name': 'exact_axes', 'acquisition_cost_mode': 'real', 'continuation_mode': 'fixed_eta_no_stop', 'eta_decay_schedule': 'global_stop', 'direction_scheduler': 'round_robin', 'label': 'G7 Fixed eta (no stop)'}, 'g8_q_then_d': {'pair_name': 'exact_axes', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'quality_then_deployment', 'label': 'G8 Q then D'}, 'g9_d_then_q': {'pair_name': 'exact_axes', 'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay', 'eta_decay_schedule': 'direction_stop', 'direction_scheduler': 'deployment_then_quality', 'label': 'G9 D then Q'}}
GRID_METRICS = (('hv_regret', 'HV Regret'), ('generational_distance', 'GD'), ('inverted_generational_distance', 'IGD'))

GRID_DATASETS = ('hotpotqa', 'mathqa', 'restaurant_valid', 'restaurant_test', 'bird_mini_dev', 'bird_dev', 'bing_querylogs', 'stackoverflow')

GRID_LEGEND_LABEL_OVERRIDES = {'directions': {G0_CONFIGURATION: 'G0 Axes', 'g1_q_only_real_cost': 'G1 Q-only', 'g4_d_only_real_cost': 'G4 D-only'}, 'continuation': {G0_CONFIGURATION: 'G0 $\\eta$-decay'}, 'scheduler': {G0_CONFIGURATION: 'G0 Round-robin'}}

def plot_ablation_metric_grid(output: Path, family: str, configurations: tuple[str, ...], curves: dict[tuple[str, str, str], ablation_plot.MeanCurve]) -> None:
    """Render the paper's eight-dataset by three-metric ablation grid."""
    rows = len(GRID_DATASETS)
    figure, axes = plt.subplots(rows, len(GRID_METRICS), figsize=(20.5, 3.82 * rows + 1.5), squeeze=False, sharex=True)
    percent = FuncFormatter(lambda value, _: f'{value:.0%}')
    compact = FuncFormatter(lambda value, _: f'{value:g}')
    for row, dataset in enumerate(GRID_DATASETS):
        for column, (metric, title) in enumerate(GRID_METRICS):
            axis = axes[row, column]
            for configuration in configurations:
                curve = curves[configuration, dataset, metric]
                color = ablation_plot.CONFIG_COLORS[configuration]
                is_g0 = configuration == G0_CONFIGURATION
                axis.plot(curve.x, curve.mean, color=color, linewidth=ablation_plot.HIGHLIGHT_LINEWIDTH if is_g0 else ablation_plot.CURVE_LINEWIDTH, zorder=5 if is_g0 else 3)
                axis.fill_between(curve.x, np.maximum(0.0, curve.mean - curve.two_se), curve.mean + curve.two_se, color=color, alpha=0.09, linewidth=0, zorder=1)
            if row == 0:
                axis.set_title(title, fontsize=40, pad=12)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(compact)
            axis.tick_params(axis='both', labelsize=30, width=1.0, length=6)
            axis.grid(color='#D5D9DE', linewidth=0.65, alpha=0.65)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)
            y_bottom, y_top = axis.get_ylim()
            if y_bottom >= 0.0:
                axis.set_ylim(-0.02 * y_top, y_top)
    figure.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.115, wspace=0.28, hspace=0.29)
    for row, dataset in enumerate(GRID_DATASETS):
        position = axes[row, 0].get_position()
        figure.text(0.037, (position.y0 + position.y1) / 2.0, ablation_plot.DATASET_LABELS[dataset], ha='center', va='center', rotation=90, fontsize=32)
    figure.text(0.55, 0.078, 'Percentage of Exhaustive Evaluation Cost', ha='center', va='center', fontsize=36)
    legend_configurations = (G0_CONFIGURATION, 'g1_q_only_real_cost', 'g4_d_only_real_cost', 'g5_axes_midpoint_real_cost', 'g6_five_directions_real_cost') if family == 'directions' else configurations
    handles = [Line2D([], [], color=ablation_plot.CONFIG_COLORS[configuration], linewidth=ablation_plot.HIGHLIGHT_LINEWIDTH if configuration == G0_CONFIGURATION else ablation_plot.CURVE_LINEWIDTH, label=GRID_LEGEND_LABEL_OVERRIDES.get(family, {}).get(configuration, str(CONFIGURATIONS[configuration]['label']))) for configuration in legend_configurations]
    handles.append(Patch(facecolor='#777777', alpha=0.14, edgecolor='none', label='$\\pm2$ SE'))
    legend_columns = 2 if family == 'scheduler' else 3
    legend_y = 0.03 if len(configurations) == 2 else 0.002
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, legend_y), ncol=legend_columns, frameon=False, fontsize=34, handlelength=2.8, columnspacing=1.0, handletextpad=0.6, labelspacing=0.55)
    stem = output / 'curves' / family / f'two_objective_{family}_ablation_hv_gd_igd_8x3'
    stem.parent.mkdir(parents=True, exist_ok=True)
    options = {'facecolor': 'white', 'bbox_inches': 'tight', 'pad_inches': 0.12}
    widen(figure)
    figure.savefig(stem.with_suffix('.png'), dpi=300, **options)
    figure.savefig(stem.with_suffix('.pdf'), **options)
    figure.savefig(stem.with_suffix('.svg'), **options)
    plt.close(figure)

FAMILIES={'cost_mechanism': ['g0_current_gittins', 'g1_q_only_real_cost', 'g2_q_only_unit_cost', 'g3_two_axis_unit_cost'], 'directions': ['g0_current_gittins', 'g1_q_only_real_cost', 'g4_d_only_real_cost', 'g5_axes_midpoint_real_cost', 'g6_five_directions_real_cost'], 'continuation': ['g0_current_gittins', 'g7_fixed_eta_no_stop'], 'scheduler': ['g0_current_gittins', 'g8_q_then_d', 'g9_d_then_q'], 'prior_variance': ['g0_current_gittins', 'q_fixed_d_warm', 'q_warm_d_fixed', 'q_fixed_d_fixed']}
COLORS={'g0_current_gittins': 'tab:orange', 'g1_q_only_real_cost': 'tab:blue', 'g2_q_only_unit_cost': 'tab:green', 'g3_two_axis_unit_cost': 'tab:pink', 'g4_d_only_real_cost': 'tab:purple', 'g5_axes_midpoint_real_cost': 'tab:brown', 'g6_five_directions_real_cost': 'tab:cyan', 'g7_fixed_eta_no_stop': 'tab:gray', 'g8_q_then_d': 'tab:olive', 'g9_d_then_q': '#003f5c', 'q_fixed_d_warm': '#5254a3', 'q_warm_d_fixed': '#637939', 'q_fixed_d_fixed': '#8c6d31'}

ablation_plot=SimpleNamespace(CONFIG_COLORS=COLORS,HIGHLIGHT_LINEWIDTH=3.24,CURVE_LINEWIDTH=2.4,
    DATASET_LABELS=dict(zip(GRID_DATASETS,('HotpotQA','MathQA','Restaurant Valid','Restaurant Test','BIRD Mini Dev','BIRD Dev','Bing Query Logs','Stack Overflow'))))
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8})
groups=defaultdict(list)
for r in csv.DictReader((DATA/'two_ablations.csv').open()):
    groups[r['configuration'],r['dataset'],r['metric']].append(r)
curves={}
for k,rs in groups.items():
    rs.sort(key=lambda r:float(r['cost_fraction']))
    curves[k]=SimpleNamespace(x=np.array([float(r['cost_fraction']) for r in rs]),
        mean=np.array([float(r['mean']) if r['mean'] else np.nan for r in rs]),
        two_se=np.array([float(r['two_se']) if r['two_se'] else np.nan for r in rs]),
        count=np.array([int(r['n_runs']) for r in rs]))
for family in ('cost_mechanism','directions','continuation','scheduler'):
    plot_ablation_metric_grid(OUT,family,tuple(FAMILIES[family]),curves)
