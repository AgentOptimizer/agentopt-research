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

plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.9,'xtick.major.width':.9,'ytick.major.width':.9})
def cmap_for(color):
    b=np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list('frequency_'+color.replace(':','_'),
        [b*.18+.82,b*.55+.45,b,b*.62],N=256)
orange=cmap_for('tab:orange');blue=cmap_for('tab:blue');norm=Normalize(1,20)
pdfs=[]

# Figure 6: eight methods, 20 matched seeds, original raw quality/USD geometry.
counts_path=DATA/'stackoverflow_frontier.csv'
allrows=list(csv.DictReader(counts_path.open()))
methods=['radial_gittins','ege_sh','ape_k','cost_aware_qnehvi','random_configurations','random_questions','copsi10','cohv_ucb']
titles=['CC-Gittins','EGE-SH','APE-k','qNEHVI-PC','Random configurations','Random questions','CoPSI','CoHV-UCB']
fig,axes=plt.subplots(2,4,figsize=(13.4,7.15),sharex=True,sharey=True)
outrows=[]
for ax,m,title in zip(axes.flat,methods,titles):
    rows=[r for r in allrows if r['dataset']=='stackoverflow' and r['method']==m and float(r['budget_fraction'])==.1]
    rows.sort(key=lambda r:int(r['arm_index']))
    assert rows and all(int(r['n_seeds'])==20 for r in rows)
    q=np.array([float(r['mean_quality']) for r in rows]);d=np.array([float(r['mean_cost_usd']) for r in rows])
    count=np.array([int(r['recommendation_count']) for r in rows]);front=np.array([int(r['true_pareto'])==1 for r in rows])
    assert np.all(count<=20)
    fs=np.flatnonzero(front);fs=fs[np.argsort(d[fs])]
    ax.scatter(d,q,s=11,c='#c6ccd2',alpha=.65,linewidths=0,zorder=1)
    ax.plot(d[fs],q[fs],c='#4e5357',lw=1.0,marker='o',ms=4,mfc='white',mew=.65,zorder=2)
    shown=count>0
    ax.scatter(d[shown],q[shown],c=count[shown],cmap=orange,norm=norm,s=37,edgecolor='#805a38',linewidth=.5,zorder=3)
    ax.set_title(title,fontsize=20,pad=8)
    ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,numticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=16,length=4.8)
    ax.grid(color='.45',alpha=.14,linewidth=.55);ax.set_axisbelow(True)
    for r in rows:outrows.append(r)
fig.subplots_adjust(left=.075,right=.985,top=.89,bottom=.30,wspace=.15,hspace=.45)
fig.supylabel('Mean quality',fontsize=23,x=.012,y=.60)
fig.supxlabel('Mean deployment cost (USD per query, log scale)',fontsize=22,y=.205)
handles=[Line2D([],[],marker='o',ls='none',color='#c6ccd2',ms=5,label='All configurations'),
         Line2D([],[],marker='o',color='#4e5357',mfc='white',ms=4,lw=1,label='Full-data Pareto frontier'),
         Line2D([],[],marker='o',ls='none',color='tab:orange',ms=6,label='Recommendations')]
legend=fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.065,.158),ncol=3,frameon=False,fontsize=17.5,handlelength=1.1,handletextpad=.45,columnspacing=.7,borderaxespad=0)
cax=fig.add_axes([.67,.147,.31,.022]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=orange),cax=cax,orientation='horizontal')
cb.set_ticks([1,5,10,15,20]);cb.ax.tick_params(labelsize=15,length=3)
cb.set_label('Recommendation frequency (out of 20 seeds)',fontsize=17.5,labelpad=3)
stem='figure6_larger_fonts'
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=230,bbox_inches='tight',pad_inches=.16)
plt.close(fig);pdfs.append(OUT/f'{stem}.pdf')
