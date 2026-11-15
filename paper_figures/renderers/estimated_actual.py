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

METHODS = [('radial_gittins', 'CC-Gittins', 'tab:orange'), ('ege_sh', 'EGE-SH', 'tab:green'), ('ape_k', 'APE-k', 'tab:purple'), ('qnehvi_pc', 'qNEHVI-PC', 'tab:pink'), ('random_configurations', 'Random configurations', 'tab:brown'), ('random_questions', 'Random questions', 'tab:cyan'), ('copsi', 'CoPSI', 'tab:blue'), ('cohv', 'CoHV-UCB', 'tab:olive')]

DATASETS = [('hotpotqa', 'HotpotQA'), ('restaurant_valid', 'Restaurant Valid'), ('bird_mini_dev', 'BIRD Mini Dev'), ('bing_querylogs', 'Bing Query Logs'), ('mathqa', 'MathQA'), ('restaurant_test', 'Restaurant Test'), ('bird_dev', 'BIRD Dev'), ('stackoverflow', 'Stack Overflow')]

cache=json.loads((DATA/'estimated_actual.json').read_text())
plotrows=[];bounds=[]
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.9})
def save(fig,stem,qa_width=1300):
    for ext in ('png','pdf'):fig.savefig(OUT/(stem+'.'+ext),dpi=230,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
def frontier(target, figure_number):
    fig, axes = plt.subplots(8, 4, figsize=(18, 25.4))
    for di, (ds, label) in enumerate((DATASETS[0], DATASETS[-1])):
        data = cache['datasets'][ds]
        truth = np.asarray(data['truth'])
        front = truth[np.asarray(data['frontier_indices'], int)]
        front = front[np.argsort(front[:, 1])]
        values = [truth]
        for method, _, _ in METHODS:
            cp = data['methods'][method][str(target)]
            values.extend([np.asarray(cp['actual_vectors']), np.asarray(cp['estimated_vectors'])])
        v = np.vstack(values)
        xmin, xmax = (v[:, 1].min(), v[:, 1].max())
        ymin, ymax = (v[:, 0].min(), v[:, 0].max())
        yp = max(0.08 * (ymax - ymin), 0.02)
        lmin, lmax = np.log10([xmin, xmax])
        xp = max(0.05 * (lmax - lmin), 0.03)
        xlim = (10 ** (lmin - xp), 10 ** (lmax + xp))
        ylim = (ymin - yp, ymax + yp)
        for row, (method, methodlabel, _) in enumerate(METHODS):
            cp = data['methods'][method][str(target)]
            ids = cp['selected_arm_indices']
            np.testing.assert_allclose(cp['actual_vectors'], truth[np.asarray(ids, int)], rtol=0, atol=1e-12)
            plotrows.append(dict(target_cost_fraction=target, dataset=ds, method=method, seed=42, snapshot_cost_fraction=cp['snapshot_cost_fraction'], cumulative_evaluations=cp['cumulative_evaluations'], selected_arm_indices=json.dumps(list(ids)), actual_vectors=json.dumps(np.asarray(cp['actual_vectors']).tolist()), estimated_vectors=json.dumps(np.asarray(cp['estimated_vectors']).tolist())))
            for offset, kind in enumerate(('actual_vectors', 'estimated_vectors')):
                ax = axes[row, di * 2 + offset]
                v = np.asarray(cp[kind])
                ax.scatter(truth[:, 1], truth[:, 0], s=27, c='#d8dce2', alpha=0.8, edgecolors='none', zorder=1)
                ax.plot(front[:, 1], front[:, 0], c='#3f4854', lw=1.6, marker='o', ms=6, mfc='white', mew=1.1, zorder=2)
                ax.scatter(v[:, 1], v[:, 0], s=105, c='tab:orange', edgecolors='#725b46', linewidths=0.8, zorder=3)
                ax.set(xscale='log', xlim=xlim, ylim=ylim)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(color='.45', alpha=0.14, lw=0.55)
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=21, labelleft=True, labelbottom=row == 7)
                bounds.append(dict(figure=figure_number, dataset=ds, method=method, kind=kind, ymin=ylim[0], ymax=ylim[1], y_tick_labels=True))
                if di == 0 and offset == 0:
                    ax.set_ylabel(methodlabel, fontsize=22, labelpad=12)
                if row == 0:
                    ax.set_title(label + '\n' + ('Actual' if offset == 0 else 'Estimated'), fontsize=25, pad=12)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.067, wspace=0.29, hspace=0.18)
    fig.supxlabel('Mean deployment cost (USD per query, log scale)', fontsize=27, y=0.032)
    fig.supylabel('Mean quality', fontsize=27, x=0.009)
    handles = [Line2D([], [], ls='', marker='o', ms=8, color='#d8dce2', label='All configurations'), Line2D([], [], color='#3f4854', lw=1.6, marker='o', mfc='white', ms=7, label='Full-data Pareto frontier'), Line2D([], [], ls='', marker='o', ms=10, mfc='tab:orange', mec='#725b46', label='Recommended configurations')]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.55, 0.002), frameon=False, fontsize=21, columnspacing=1.2)
    fig.suptitle(f'{target:.0%} search-cost checkpoint (seed 42)', fontsize=28, y=0.985)
    save(fig, f'figure{figure_number}_eight_methods_{round(100 * target)}pct', 1500)

frontier(.1,11)
frontier(.3,12)
