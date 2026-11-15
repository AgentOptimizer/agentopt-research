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

DATASETS = [('hotpotqa', 'HotpotQA'), ('restaurant_valid', 'Restaurant Valid'), ('bird_mini_dev', 'BIRD Mini Dev'), ('bing_querylogs', 'Bing Query Logs'), ('mathqa', 'MathQA'), ('restaurant_test', 'Restaurant Test'), ('bird_dev', 'BIRD Dev'), ('stackoverflow', 'Stack Overflow')]

METHODS = [('radial_gittins', 'CC-Gittins', 'tab:orange'), ('ege_sh', 'EGE-SH', 'tab:green'), ('ape_k', 'APE-k', 'tab:purple'), ('qnehvi_pc', 'qNEHVI-PC', 'tab:pink'), ('random_configurations', 'Random configurations', 'tab:brown'), ('random_questions', 'Random questions', 'tab:cyan'), ('copsi', 'CoPSI (30%)', 'tab:blue'), ('cohv', 'CoHV-UCB', 'tab:olive')]
summary=list(csv.DictReader((DATA/'job_runtime.csv').open()))
for r in summary:
 r['n']=int(r['n'])
 for k in ('mean_seconds','two_se_seconds'):r[k]=float(r[k]) if r[k] else None
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.9})
fig,axes=plt.subplots(2,4,figsize=(18.5,7.7))
for ax,(ds,label) in zip(axes.flat,DATASETS):
 for i,(method,_,color) in enumerate(METHODS):
  r=next(r for r in summary if r['dataset']==ds and r['method']==method)
  if r['n']:
   ax.bar(i,r['mean_seconds'],color=color,width=.75,edgecolor='black',lw=.5,yerr=r['two_se_seconds'],error_kw={'elinewidth':.9,'capsize':2.2},zorder=3)
  else:ax.text(i,5,'—',ha='center',va='bottom',fontsize=17)
 ax.set_title(label,fontsize=23,pad=7)
 ax.set(yscale='log',ylim=(3,300000),xlim=(-.65,7.65),xticks=[])
 ax.tick_params(axis='y',labelsize=20,labelleft=(ax in axes[:,0]))
 ax.grid(axis='y',color='.45',alpha=.14,lw=.55);ax.set_axisbelow(True)
fig.supylabel('Mean job wall-clock time (s)',fontsize=25,x=.006,y=.59)
handles=[Patch(facecolor=c,edgecolor='black',lw=.5,label=label) for _,label,c in METHODS]
order=[0,4,1,5,2,6,3,7]
fig.legend(handles=[handles[i] for i in order],loc='lower center',bbox_to_anchor=(.53,.01),ncol=4,fontsize=21,frameon=False,columnspacing=1.35,handlelength=1.4)
fig.subplots_adjust(left=.075,right=.99,bottom=.21,top=.96,wspace=.32,hspace=.3)
for ext in ('png','pdf','svg'):
 p=OUT/('figure8_job_elapsed_eight_methods.'+ext)
 fig.savefig(p,dpi=230,bbox_inches='tight',pad_inches=.15)
plt.close(fig)
