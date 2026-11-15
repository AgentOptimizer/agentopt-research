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

DS = [('hotpotqa', 'HotpotQA'), ('restaurant_valid', 'Restaurant Valid'), ('bird_mini_dev', 'BIRD Mini Dev'), ('bing_querylogs', 'Bing Query Logs'), ('mathqa', 'MathQA'), ('restaurant_test', 'Restaurant Test'), ('bird_dev', 'BIRD Dev'), ('stackoverflow', 'Stack Overflow')]

METHODS = [('g0_current_gittins', 'CC-Gittins'), ('g1_q_only_real_cost', 'Quality-only (G1)')]

O=OUT
data=json.loads((DATA/'quality_truth.json').read_text())
summary=list(csv.DictReader((DATA/'quality_summary.csv').open()))
counts_rows=list(csv.DictReader((DATA/'quality_frequencies.csv').open()))
for r in summary+counts_rows:
    for k,v in list(r.items()):
        if k not in ('dataset','configuration','true_pareto'):r[k]=float(v)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.9})
cmap=LinearSegmentedColormap.from_list('paper_orange',['#fff0e2','#ff7f0e','#a64b00']);norm=Normalize(1,20)
for b in (.1,):
 fig,axes=plt.subplots(4,4,figsize=(18.5,17.3))
 for k,(ds,label) in enumerate(DS):
  raw=np.array(data[ds]['truth_raw_vectors']);front=raw[data[ds]['frontier']];front=front[np.argsort(front[:,1])]
  low,high=raw[:,0].min(),raw[:,0].max();yp=max(.08*(high-low),.015);xmin,xmax=raw[:,1].min(),raw[:,1].max()
  assert xmin>0
  a,z=np.log10([xmin,xmax]);xp=max(.06*(z-a),.025);xlim=(10**(a-xp),10**(z+xp))
  for j,(m,name) in enumerate(METHODS):
   ax=axes[k//2,(k%2)*2+j];rr=[r for r in counts_rows if (r['cost_fraction'],r['dataset'],r['configuration'])==(b,ds,m)]
   ax.scatter(raw[:,1],raw[:,0],s=20,c='#c8cfd6',alpha=.5,edgecolors='none',zorder=1)
   ax.plot(front[:,1],front[:,0],color='#4e555b',lw=1.2,marker='o',ms=5.2,mfc='white',mew=.8,zorder=2)
   ax.scatter([r['deployment_cost_usd'] for r in rr],[r['quality'] for r in rr],c=[r['recommendation_frequency'] for r in rr],cmap=cmap,norm=norm,s=60,edgecolors='#916c4c',linewidths=.5,zorder=3)
   ax.set(xscale='log',xlim=xlim,ylim=(low-yp,high+yp));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
   ax.tick_params(labelsize=24);ax.grid(color='.45',alpha=.14,lw=.55);ax.set_axisbelow(True);ax.set_title(label+'\n'+name,fontsize=30,pad=7)
   st=next(r for r in summary if (r['cost_fraction'],r['dataset'],r['configuration'])==(b,ds,m))
   ax.text(.98,.04,f"P={st['mean_precision']:.2f}   R={st['mean_recall']:.2f}",transform=ax.transAxes,ha='right',va='bottom',fontsize=22,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.5),zorder=5)
 fig.subplots_adjust(left=.075,right=.985,top=.95,bottom=.155,wspace=.26,hspace=1.0)
 fig.supylabel('Mean quality',fontsize=31,x=.008,y=.56)
 fig.supxlabel('Mean deployment cost (USD per query, log scale)',fontsize=31,y=.10)
 handles=[Line2D([],[],ls='',marker='o',ms=9,color='#c8cfd6',label='All configurations'),Line2D([],[],color='#4e555b',lw=1.2,marker='o',mfc='white',ms=6,label='Full-data Pareto frontier'),Line2D([],[],ls='',marker='o',ms=9,color='tab:orange',label='Recommendations')]
 fig.legend(handles=handles,ncol=3,frameon=False,fontsize=26,loc='center',bbox_to_anchor=(.36,.057),columnspacing=1.1,handletextpad=.5)
 cax=fig.add_axes([.715,.053,.245,.014]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[1,5,10,15,20]);cb.ax.tick_params(labelsize=22,pad=2);cb.set_label('Recommendation frequency (20 seeds)',fontsize=25,labelpad=4)
 stem=f'cc_vs_quality_only_frontiers_{round(b*100)}pct'
 for ext in ('png','pdf'):fig.savefig(O/(stem+'.'+ext),dpi=230,bbox_inches='tight',pad_inches=.15)
 plt.close(fig)
