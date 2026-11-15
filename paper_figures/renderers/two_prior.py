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
VARIANTS = ('q_warm_d_warm', 'q_fixed_d_warm', 'q_warm_d_fixed', 'q_fixed_d_fixed')

IDS = {'two': dict(zip(VARIANTS, ('G0', 'G10', 'G11', 'G12'))), 'three': dict(zip(VARIANTS, ('G0', 'G11', 'G12', 'G13')))}

COLORS = dict(zip(VARIANTS, ('tab:orange', '#5254a3', '#637939', '#8c6d31')))

LABELS = {'q_warm_d_warm': 'Q warm-start, D warm-start', 'q_fixed_d_warm': 'Q 0.04, D warm-start', 'q_warm_d_fixed': 'Q warm-start, D 0.04', 'q_fixed_d_fixed': 'Q 0.04, D 0.04'}

DATASETS = ('hotpotqa', 'mathqa', 'restaurant_valid', 'restaurant_test', 'bird_mini_dev', 'bird_dev', 'bing_querylogs', 'stackoverflow')

DS_LABELS = dict(zip(DATASETS, ('HotpotQA', 'MathQA', 'Restaurant Valid', 'Restaurant Test', 'BIRD Mini Dev', 'BIRD Dev', 'Bing Query Logs', 'Stack Overflow')))

METRICS = ('hv_regret', 'generational_distance', 'inverted_generational_distance')

TITLES = ('HV Regret', 'GD', 'IGD')

def draw(ax, mode, ds, metric):
    upper = 0
    for v in (*VARIANTS[1:], VARIANTS[0]):
        rr = sorted(index[mode, ds, v, metric], key=lambda r: float(r['cost_fraction']))
        assert len(rr) == 60
        x = np.asarray([float(r['cost_fraction']) for r in rr])
        y = np.asarray([float(r['mean']) if r['mean'] else np.nan for r in rr])
        se = np.asarray([float(r['two_se']) if r['two_se'] else np.nan for r in rr])
        assert rr[-1]['n_runs'] == '20'
        upper = max(upper, float(np.nanmax(y + se)))
        main = v == VARIANTS[0]
        ax.fill_between(x, np.maximum(0, y - se), y + se, color=COLORS[v], alpha=0.09 if mode == 'two' else 0.14 if main else 0.05, linewidth=0, zorder=2 if main else 1)
        ax.plot(x, y, color=COLORS[v], linewidth=3.24 if main else 2.4, linestyle='-', zorder=10 if main else 3)
    ax.set_xlim(-0.0045, 0.306) if mode == 'two' else ax.set_xlim(-0.005, 0.305)
    if mode == 'two':
        bottom, top = ax.get_ylim()
        if bottom >= 0:
            ax.set_ylim(-0.02 * top, top)
    else:
        ax.set_ylim(-0.025 * upper, 1.06 * upper)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.tick_params(axis='both', labelsize=30 if mode == 'two' else 28, width=1, length=6 if mode == 'two' else 5)
    ax.grid(color='#D5D9DE', linewidth=0.65, alpha=0.65) if mode == 'two' else ax.grid(True, alpha=0.22, color='#9da9bb', linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('black')
        s.set_linewidth(0.8 if mode == 'two' else 1)

def legend(fig, mode, anchor):
    handles = [Line2D([], [], color=COLORS[v], linewidth=3.24 if v == VARIANTS[0] else 2.4, linestyle='-', label=IDS[mode][v] + ' ' + LABELS[v]) for v in VARIANTS] + [Patch(facecolor='#777777', alpha=0.14, label='$\\pm2$ SE')]
    leg = fig.legend(handles=handles, loc='lower center' if mode == 'two' else 'upper center', bbox_to_anchor=anchor, ncol=3, frameon=False, fontsize=34 if mode == 'two' else 27, handlelength=2.8 if mode == 'two' else 2.3, columnspacing=1.0, handletextpad=0.6, labelspacing=0.55, borderaxespad=0)
    fig.canvas.draw()
    box = leg.get_window_extent(fig.canvas.get_renderer())
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width, (mode, box.bounds, fig.bbox.width)
    return leg

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8})
remap={'g0_current_gittins':'q_warm_d_warm','q_fixed_d_warm':'q_fixed_d_warm','q_warm_d_fixed':'q_warm_d_fixed','q_fixed_d_fixed':'q_fixed_d_fixed'}
index=defaultdict(list)
for r in csv.DictReader((DATA/'two_ablations.csv').open()):
    if r['configuration'] in remap:index['two',r['dataset'],remap[r['configuration']],r['metric']].append(r)
for key,rs in index.items():
    existing={round(float(r['cost_fraction']),8) for r in rs}
    rs.extend(dict(cost_fraction=i/200,mean='',two_se='',n_runs='0') for i in range(1,61) if round(i/200,8) not in existing)
two,axes=plt.subplots(8,3,figsize=(20.5,3.82*8+1.5),squeeze=False,sharex=True)
for i,ds in enumerate(DATASETS):
 for j,metric in enumerate(METRICS):
  draw(axes[i,j],'two',ds,metric)
  if i==0:axes[i,j].set_title(TITLES[j],fontsize=40,pad=12)
two.subplots_adjust(left=.11,right=.985,top=.965,bottom=.115,wspace=.28,hspace=.29)
for i,ds in enumerate(DATASETS):
 pos=axes[i,0].get_position();two.text(.037,(pos.y0+pos.y1)/2,DS_LABELS[ds],ha='center',va='center',rotation=90,fontsize=32)
two.text(.55,.078,'Percentage of Exhaustive Evaluation Cost',ha='center',va='center',fontsize=36);legend(two,'two',(.5,.002))

widen(two)
for ext in ('png','pdf'):two.savefig(OUT/('two_objective_prior_variance.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12,facecolor='white')
plt.close(two)
