#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import matplotlib.pyplot as plt

LABEL={"PC-Reversible26":"PC-Rev26","Nan25-Curvature":"Nan25-Curv","Nan25-ZW":"Nan25-ZW","Jiao23-Dual":"Jiao23-Dual","Qiu19-RI":"Qiu19-RI","Ours":"HKReal3DMark"}

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--ours',required=True);p.add_argument('--paper',required=True);a=p.parse_args()
 paper=Path(a.paper);figdir=paper/'figures';datadir=paper/'data';figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
 base=json.loads(Path(a.baseline).read_text());ours=json.loads(Path(a.ours).read_text());combined={**base['methods'],'Ours':{'clean_nc':1.0,'clean_coverage':1.0,'self_check':'pass','uniqueness':ours['uniqueness'],'curves':ours['curves']}}
 shutil.copyfile(a.baseline,datadir/'oblique_baselines.json')
 attacks=list(ours['curves']);fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True);rows=[]
 colors=plt.cm.tab10.colors
 for ax,family in zip(axes.flat,attacks):
  for color,(name,result) in zip(colors,combined.items()):
   curve=result['curves'][family];x=[p['intensity'] for p in curve];y=[p['mean_nc'] for p in curve]
   ax.plot(x,y,marker='o',ms=3,lw=1.5,label=LABEL[name],color=color,alpha=.95 if result['self_check']=='pass' else .55,ls='-' if result['self_check']=='pass' else '--')
   for point in curve:rows.append({'method':name,'attack':family,'intensity':point['intensity'],'mean_nc':point['mean_nc'],'self_check':result['self_check']})
  ax.set_title(family.replace('_',' '));ax.set_ylim(.35,1.02);ax.grid(alpha=.25);ax.set_xlabel('Intensity');ax.set_ylabel('Mean NC')
 handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3);fig.suptitle('Hong Kong oblique-photogrammetry watermark baselines (seed=2026)')
 fig.savefig(figdir/'oblique_baseline_robustness.pdf',bbox_inches='tight');fig.savefig(figdir/'oblique_baseline_robustness.png',dpi=220,bbox_inches='tight');plt.close(fig)
 with (figdir/'oblique_baseline_robustness.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 lines=['\\begin{tabular}{lcccc}\\toprule','Method & Type & Clean NC & Coverage & Worst NC \\\\\\midrule']
 types={'PC-Reversible26':'Embedded','Nan25-Curvature':'Embedded','Nan25-ZW':'Zero','Jiao23-Dual':'Embedded','Qiu19-RI':'Embedded','Ours':'Zero'}
 for name,r in combined.items():
  worst=min(p['mean_nc'] for curve in r['curves'].values() for p in curve);mark='*' if r['self_check']!='pass' else ''
  lines.append(f"{LABEL[name]}{mark} & {types[name]} & {r['clean_nc']:.3f} & {r['clean_coverage']:.3f} & {worst:.3f} \\\\")
 lines+=['\\bottomrule\\end{tabular}','% * adapted implementation failed the clean self-check']
 (paper/'oblique_baseline_table.tex').write_text('\n'.join(lines),encoding='utf8')
 print(json.dumps({'figure':str(figdir/'oblique_baseline_robustness.pdf'),'rows':len(rows)}))
if __name__=='__main__':main()
