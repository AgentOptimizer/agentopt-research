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

rows=list(csv.DictReader((DATA/'main_curves.csv').open()))
styles=json.loads((DATA/'method_styles.json').read_text())
styles.update(cost_aware_qnehvi=dict(color='tab:pink',label='qNEHVI-PC',linestyle='-',linewidth=2.4),
              copsi10=dict(color='tab:blue',label='CoPSI (10% budget)',linestyle='-',linewidth=2.4),
              copsi30=dict(color='tab:blue',label='CoPSI (30% budget)',linestyle='-',linewidth=2.4),
              cohv_ucb=dict(color='tab:olive',label='CoHV-UCB',linestyle='-',linewidth=2.4))
datasets=['hotpotqa','restaurant_valid','bird_mini_dev','bing_querylogs','mathqa','restaurant_test','bird_dev','stackoverflow']
labels=['HotpotQA','Restaurant Valid','BIRD Mini Dev','Bing Query Logs','MathQA','Restaurant Test','BIRD Dev','Stack Overflow']
metrics={'hv_regret':'Hypervolume Regret','generational_distance':'Generational Distance (GD)','inverted_generational_distance':'Inverted Generational Distance (IGD)'}
index=defaultdict(list)
for r in rows:index[r['dataset'],r['method'],r['metric']].append(r)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.8})
pdfs=[];counts=[];used=[]

for cap,copsi in [(.3,'copsi30'),(.1,'copsi10')]:
    methods=['radial_gittins','ege_sh','ape_k','cost_aware_qnehvi','random_configurations','random_questions',copsi,'cohv_ucb']
    assert len(methods)==8 and 'qnehvi' not in methods
    for metric,title in metrics.items():
        fig,axes=plt.subplots(2,4,figsize=(18.5,9.0))
        for ax,ds,label in zip(axes.flat,datasets,labels):
            for method in methods:
                rr=sorted((r for r in index[ds,method,metric] if float(r['cost_fraction'])<=cap+1e-12),key=lambda r:float(r['cost_fraction']))
                assert rr
                x=np.array([float(r['cost_fraction']) for r in rr]);n=np.array([int(r['n_runs']) for r in rr])
                y=np.array([float(r['mean']) if r['mean'] else np.nan for r in rr])
                se=np.array([float(r['two_se']) if r['two_se'] else np.nan for r in rr])
                assert len(np.unique(x))==len(x) and np.all((n>=0)&(n<=20))
                y[n==0]=np.nan;se[n==0]=np.nan
                st=styles[method];random=method.startswith('random_')
                ax.plot(x,y,color=st['color'],linestyle=st['linestyle'],linewidth=st['linewidth'],drawstyle='steps-post' if random else 'default',zorder=4 if method=='radial_gittins' else 3)
                ax.fill_between(x,np.maximum(0,y-se),y+se,color=st['color'],alpha=.10,linewidth=0,zorder=1,step='post' if random else None)
                end=next(r for r in rr if abs(float(r['cost_fraction'])-cap)<1e-10)
                if metric=='hv_regret':
                    counts.append(dict(dataset=ds,method=method,budget=cap,n_at_endpoint=end['n_runs'],min_positive_n=min(n[n>0]) if any(n>0) else 0,last_available_cost=max(x[n>0]) if any(n>0) else None))
                used.extend(dict(display_budget=cap,**r) for r in rr)
            ax.set_title(label,fontsize=29,pad=10)
            ax.set_xlim(-.015*cap,cap*1.02)
            top=ax.get_ylim()[1];ax.set_ylim(-.02*top,top)
            ax.set_xticks([0,.1,.2,.3] if cap==.3 else [0,.05,.1]);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.0%}'))
            ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            ax.tick_params(labelsize=27,width=1.1,length=6);ax.grid(color='#D5D9DE',linewidth=.65,alpha=.65);ax.set_axisbelow(True)
        fig.text(.52,.158,'Percentage of Exhaustive Evaluation Cost',ha='center',va='center',fontsize=30)
        fig.text(.022,.605,title,ha='center',va='center',rotation=90,fontsize=30)
        handles=[Line2D([],[],color=styles[m]['color'],linestyle=styles[m]['linestyle'],linewidth=styles[m]['linewidth'],label=styles[m]['label']) for m in methods]
        handles=[handles[i] for i in [0,4,1,5,2,6,3,7]]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,-.015),ncol=4,frameon=False,fontsize=24,handlelength=1.9,columnspacing=1.0,handletextpad=.65,labelspacing=.7)
        fig.subplots_adjust(left=.085,right=.985,top=.945,bottom=.265,wspace=.34,hspace=.44)
        stem=OUT/f'eight_methods_{int(cap*100)}pct_{metric}'
        for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=230,bbox_inches='tight',pad_inches=.12)

        pdfs.append(stem.with_suffix('.pdf'));plt.close(fig)
        print(stem.name,flush=True)
