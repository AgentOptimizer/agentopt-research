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

CAP = 0.3

METHODS = ['four_direction_real_cost', 'ege_sh', 'ape_k', 'cost_aware_qnehvi', 'random_configurations', 'random_questions', 'copsi', 'cohv_ucb']

NAMES = ['CC-Gittins', 'EGE-SH', 'APE-k', 'qNEHVI-PC', 'Random configurations', 'Random questions', 'CoPSI', 'CoHV-UCB']

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
truth=json.loads((DATA/'three_truth.json').read_text())
countrows=list(csv.DictReader((DATA/'three_frontier_counts.csv').open()))
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
def cmap(color):
    base = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list(color, [base * .18 + .82, base * .55 + .45, base, base * .62])


orange, blue = cmap('tab:orange'), cmap('tab:blue')
norm = Normalize(1, 20)
summary, points, outputs = [], [], []
sizes = dict(panel_title=27, axis_label=25, xy_label=19, ticks=19,
             legend=25, colorbar_label=21, colorbar_ticks=18)
with PdfPages(OUT / 'three_objective_frontiers_30pct.pdf') as combined:
    for ds in ['hotpotqa', 'mathqa']:
        raw = np.array(truth[ds]['raw_truth_vectors'])
        q, latency, cost = raw.T
        rewards = raw * [1, -1, -1]
        front = np.array([not np.any(np.all(rewards >= p, axis=1) & np.any(rewards > p, axis=1)) for p in rewards])
        x, y = np.log10(cost), np.log10(latency)
        fig = plt.figure(figsize=(20, 13.5))
        axis_labels = []
        for index, (method, title) in enumerate(zip(METHODS, NAMES)):
            row, col = divmod(index, 4)
            left, bottom, width, height = .065 + .232 * col, .62 if row == 0 else .23, .214, .32
            ax = fig.add_axes([left, bottom, width, height], projection='3d', computed_zorder=False)
            ax.view_init(elev=24, azim=-53)
            ax.set_box_aspect((1.25, 1.05, 1.05), zoom=.94)
            counts = np.array([int(r['frequency']) for r in countrows if r['dataset']==ds and r['method']==method])
            ax.scatter(x, y, q, s=15, c='#bdbdbd', alpha=.5, edgecolors='none', depthshade=False, zorder=1)
            ax.scatter(x[front], y[front], q[front], s=48, facecolors='white', edgecolors='#555555',
                       linewidths=.8, depthshade=False, zorder=2)
            for mask, cm, edge in [((counts > 0) & front, orange, '#805a38'),
                                   ((counts > 0) & ~front, blue, '#184b6b')]:
                ax.scatter(x[mask], y[mask], q[mask], s=40, c=counts[mask], cmap=cm, norm=norm,
                           edgecolors=edge, linewidths=.65, depthshade=False, zorder=3)
            costticks = np.array([.0001, .001, .01, .1])
            costticks = costticks[(costticks > cost.min() * .75) & (costticks < cost.max() * 1.35)]
            latticks = np.array([2, 10, 50] if ds == 'hotpotqa' else [5, 10, 20])
            latticks = latticks[(latticks > latency.min() * .8) & (latticks < latency.max() * 1.3)]
            ax.set_xticks(np.log10(costticks), [f'{v:g}' for v in costticks])
            ax.set_yticks(np.log10(latticks), [f'{v:g}' for v in latticks])
            ax.set_xlim(x.min() - .1, x.max() + .1)
            ax.set_ylim(y.min() - .06, y.max() + .06)
            ax.set_zlim((.25, .78) if ds == 'hotpotqa' else (.72, .99))
            ax.set_zticks([.3, .5, .7] if ds == 'hotpotqa' else [.75, .85, .95])
            ax.zaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.zaxis.set_ticks_position('lower')
            ax.zaxis.set_label_position('lower')
            ax.set_xlabel('Mean deployment cost', fontsize=sizes['xy_label'], labelpad=12)
            ax.set_ylabel('Mean latency', fontsize=sizes['xy_label'], labelpad=12)
            if col == 0:
                ax.zaxis.set_rotate_label(False)
                ax.set_zlabel('Mean quality', fontsize=sizes['axis_label'], labelpad=36, rotation=90)
            else:
                ax.set_zticklabels([])
            axis_labels.extend([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label])
            ax.tick_params(labelsize=sizes['ticks'], pad=1)
            ax.zaxis.set_tick_params(pad=6)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor('#e0e0e0')
                axis.line.set_color('#555555')
                axis._axinfo['grid'].update(color=(.45, .45, .45, .14), linewidth=.55)
            fig.text(left + width / 2, bottom + height - .025, title, ha='center', fontsize=sizes['panel_title'])
            summary.append(dict(dataset=ds, method=method, budget_fraction=CAP, n_seeds=20,
                                recommended_pareto=int(((counts > 0) & front).sum()),
                                recommended_dominated=int(((counts > 0) & ~front).sum()),
                                dominated_occurrences=int(counts[~front].sum())))
            for arm in range(len(raw)):
                points.append(dict(dataset=ds, method=method, budget_fraction=CAP, n_seeds=20, arm=arm,
                                   model=truth[ds]['models'][arm], quality=q[arm], latency=latency[arm],
                                   cost=cost[arm], true_pareto=bool(front[arm]), frequency=int(counts[arm])))
        handles = [Line2D([], [], ls='none', marker='o', mfc='#bdbdbd', mec='none', ms=7, label='All configurations'),
                   Line2D([], [], ls='none', marker='o', mfc='white', mec='#555555', ms=8, label='Full-data Pareto frontier'),
                   Line2D([], [], ls='none', marker='o', mfc='tab:orange', mec='none', ms=8, label='Pareto recommendations'),
                   Line2D([], [], ls='none', marker='o', mfc='tab:blue', mec='none', ms=8, label='Dominated recommendations')]
        fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(.51, .041),
                   frameon=False, fontsize=sizes['legend'], columnspacing=.9, handletextpad=.4)
        for left, cm, label in [(.10, orange, 'Pareto recommendation frequency'),
                                (.57, blue, 'Dominated recommendation frequency')]:
            cax = fig.add_axes([left, .145, .34, .018])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation='horizontal')
            cb.set_ticks([1, 5, 10, 15, 20])
            cb.ax.tick_params(labelsize=sizes['colorbar_ticks'], length=4)
            cb.set_label(label, fontsize=sizes['colorbar_label'], labelpad=5)
        stem = f'{ds}_all_methods_2x4_30pct'
        extra_artists = axis_labels + list(fig.texts) + list(fig.legends)
        fig.canvas.draw()
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / f'{stem}.{extension}', dpi=230, facecolor='white', bbox_inches='tight',
                        bbox_extra_artists=extra_artists, pad_inches=.16)
        combined.savefig(fig, facecolor='white', bbox_inches='tight', bbox_extra_artists=extra_artists, pad_inches=.16)
        plt.close(fig)
        outputs.append(stem)
