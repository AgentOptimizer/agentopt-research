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
CURVE_LINEWIDTH = 2.4

HIGHLIGHT_LINEWIDTH = CURVE_LINEWIDTH * 1.35

DATASETS = ('hotpotqa', 'mathqa')

DATASET_LABELS = {'hotpotqa': 'HotpotQA', 'mathqa': 'MathQA'}

METRICS = (('hv_regret', 'HV Regret'), ('generational_distance', 'GD'), ('inverted_generational_distance', 'IGD'))

def style(color: str, label: str, *, highlight: bool=False) -> dict[str, Any]:
    return {'color': color, 'label': label, 'linewidth': HIGHLIGHT_LINEWIDTH if highlight else CURVE_LINEWIDTH, 'linestyle': '-'}

FAMILIES: dict[str, dict[str, Any]] = {'cost_mechanism': {'directory': 'cost_mechanism_ablation', 'summary': 'curve_summary.csv', 'stem': 'three_objective_cost_mechanism_ablation_hv_gd_igd', 'legend_columns': 3, 'styles': {'four_direction_real_cost': style('tab:orange', 'Four directions · real cost', highlight=True), 'four_direction_unit_cost': style('tab:red', 'Four directions · unit cost'), 'pure_axes_real_cost': style('tab:blue', 'Pure axes · real cost'), 'pure_axes_unit_cost': style('tab:purple', 'Pure axes · unit cost'), 'q_only_real_cost': style('tab:green', 'Q-only · real cost'), 'q_only_unit_cost': style('tab:cyan', 'Q-only · unit cost')}}, 'direction': {'directory': 'direction_ablation', 'summary': 'direction_curve_summary.csv', 'stem': 'three_objective_direction_ablation_hv_gd_igd', 'legend_columns': 3, 'styles': {'four_direction_real_cost': style('tab:orange', 'Four directions (+ Q-L midpoint)', highlight=True), 'pure_axes_real_cost': style('tab:blue', 'Pure axes'), 'six_ql_dense_real_cost': style('tab:green', 'Six directions (Q-L dense)'), 'six_three_edges_real_cost': style('tab:purple', 'Six directions (three edges)')}}, 'continuation': {'directory': 'continuation_ablation', 'summary': 'curve_summary.csv', 'stem': 'three_objective_continuation_ablation_hv_gd_igd', 'legend_columns': 3, 'styles': {'four_direction_real_cost': style('tab:orange', 'Direction-local $\\eta$ decay', highlight=True), 'four_direction_fixed_eta': style('tab:blue', 'Fixed $\\eta=1$ (no stop)')}}}

def configure_matplotlib() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})

def load_curves(path: Path) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if not row['mean'] or not row['two_se']:
                continue
            key = (row['dataset'], row['configuration'], row['metric'])
            grouped.setdefault(key, []).append((float(row['cost_fraction']), float(row['mean']), float(row['two_se'])))
    curves = {}
    for key, values in grouped.items():
        array = np.asarray(sorted(values), dtype=np.float64)
        curves[key] = (array[:, 0], array[:, 1], array[:, 2])
    return curves

def update_metadata(path: Path, styles: dict[str, dict[str, Any]]) -> None:
    payload: dict[str, Any] = {}
    if path.is_file():
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
    payload['styles'] = styles
    payload['style_policy'] = 'one color per variant; solid curves; selected method in orange'
    payload['plotter'] = 'experiments/combined_objective/plot_three_objective_ablations.py'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def plot_family(root: Path, family: str, specification: dict[str, Any]) -> None:
    directory = root / str(specification['directory'])
    curves = load_curves(DATA / ('three_'+family+'.csv'))
    styles: dict[str, dict[str, Any]] = specification['styles']
    expected = {(dataset, configuration, metric) for dataset in DATASETS for configuration in styles for metric, _ in METRICS}
    missing = sorted(expected - curves.keys())
    if missing:
        raise ValueError(f'{family}: missing {len(missing)} curve series: {missing[:3]}')
    figure, axes = plt.subplots(2, 3, figsize=(18.5, 10.5), squeeze=False)
    percent = FuncFormatter(lambda value, _: f'{value:.0%}')
    compact = FuncFormatter(lambda value, _: f'{value:g}')
    draw_order = [key for key in styles if key != 'four_direction_real_cost']
    draw_order.append('four_direction_real_cost')
    for row, dataset in enumerate(DATASETS):
        for column, (metric, title) in enumerate(METRICS):
            axis = axes[row, column]
            for configuration in draw_order:
                xs, means, two_se = curves[dataset, configuration, metric]
                variant = styles[configuration]
                highlighted = configuration == 'four_direction_real_cost'
                axis.plot(xs, means, color=variant['color'], linewidth=variant['linewidth'], linestyle='-', zorder=6 if highlighted else 3)
                axis.fill_between(xs, np.maximum(0.0, means - two_se), means + two_se, color=variant['color'], alpha=0.1, linewidth=0, zorder=2 if highlighted else 1)
            if row == 0:
                axis.set_title(title, fontsize=28, pad=13)
            axis.set_xlim(-0.0045, 0.306)
            axis.set_xticks((0.0, 0.1, 0.2, 0.3))
            axis.xaxis.set_major_formatter(percent)
            if row == 0:
                axis.tick_params(axis='x', labelbottom=False)
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
    compact_legend = len(styles) <= 2
    figure.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.22 if compact_legend else 0.31, wspace=0.3, hspace=0.34)
    row_centers = [(axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0 for row in range(2)]
    for row, dataset in enumerate(DATASETS):
        figure.text(0.02, row_centers[row], DATASET_LABELS[dataset], ha='center', va='center', rotation=90, fontsize=31)
    figure.text(0.535, 0.13 if compact_legend else 0.215, 'Percentage of Exhaustive Evaluation Cost', ha='center', va='center', fontsize=29)
    handles = [Line2D([], [], color=variant['color'], linewidth=variant['linewidth'], linestyle='-', label=variant['label']) for variant in styles.values()]
    handles.append(Patch(facecolor='#777777', alpha=0.14, edgecolor='none', label='$\\pm2$ SE'))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.015 if compact_legend else -0.015), ncol=int(specification['legend_columns']), frameon=False, fontsize=30, handlelength=2.6, columnspacing=1.2, handletextpad=0.65, labelspacing=0.65)
    output = directory / str(specification['stem'])
    save_options = {'facecolor': 'white', 'bbox_inches': 'tight', 'pad_inches': 0.12}
    figure.savefig(output.with_suffix('.png'), dpi=300, **save_options)
    figure.savefig(output.with_suffix('.pdf'), **save_options)
    figure.savefig(output.with_suffix('.svg'), **save_options)
    plt.close(figure)
    metadata_name = 'direction_ablation_metadata.json' if family == 'direction' else 'metadata.json'
    update_metadata(directory / metadata_name, styles)
    print(output.with_suffix('.png'))

for spec in FAMILIES.values():
    for cfg,st in spec['styles'].items():st['label']=IDS[cfg]+' '+st['label']
configure_matplotlib()
for family in ('cost_mechanism','continuation'):
    (OUT/FAMILIES[family]['directory']).mkdir(parents=True,exist_ok=True)
    plot_family(OUT,family,FAMILIES[family])
