#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LABELS={'rotation':'Rotation','scale':'Scaling','noise':'Gaussian noise','quantization':'Quantization','point_delete':'Point deletion','crop':'Spatial cropping','simplify':'Simplification','outliers':'Outliers','sequential':'Sequential attack'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--paper',default='paper');a=p.parse_args();d=json.loads(Path(a.result).read_text());out=Path(a.paper);fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
 with (fig/'robustness.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['attack','intensity','mean_nc','q05_nc','auc','eer'])
  for k,v in d['curves'].items():
   for x in v:w.writerow([k,x['intensity'],x['mean_nc'],x['q05_nc'],x['auc'],x['eer']])
 plt.style.use('seaborn-v0_8-whitegrid');F,A=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
 for ax,(k,v) in zip(A.flat,d['curves'].items()):
  x=np.arange(len(v));ax.plot(x,[q['mean_nc'] for q in v],'-o',label='Mean NC');ax.plot(x,[q['q05_nc'] for q in v],'--s',label='5th percentile NC');ax.set_xticks(x,[str(q['intensity']) for q in v],rotation=25);ax.set_ylim(0,1.03);ax.set_title(LABELS[k]);ax.set_xlabel('Attack intensity');ax.set_ylabel('NC')
 A.flat[0].legend(fontsize=8);F.savefig(fig/'robustness_grid.pdf');F.savefig(fig/'robustness_grid.png',dpi=300);plt.close(F)
 u=d['uniqueness'];worst=min(q['auc'] for v in d['curves'].values() for q in v)
 (out/'results_macros.tex').write_text(f"\\newcommand{{\\ModelParams}}{{{d['parameters']:,}}}\n\\newcommand{{\\UniqueMean}}{{{u['mean']:.3f}}}\n\\newcommand{{\\UniqueQ}}{{{u['q95']:.3f}}}\n\\newcommand{{\\UniqueMax}}{{{u['max']:.3f}}}\n\\newcommand{{\\WorstAUC}}{{{worst:.3f}}}\n",encoding='utf8')
 baseline_path=Path(a.result).with_name('baselines.json')
 if baseline_path.exists():
  baselines=json.loads(baseline_path.read_text());rows=[]
  for name,v in baselines.items():rows.append((name,v['uniqueness']['q95'],min(x['auc'] for z in v['curves'].values() for x in z)))
  rows.append(('HKReal3DMark',u['q95'],worst))
  text=['\\begin{tabular}{lcc}\\toprule 方法 & 异源NC q95 & 最差AUC \\\\ \\midrule']+[f'{n} & {q:.3f} & {w:.3f} \\\\' for n,q,w in rows]+['\\bottomrule\\end{tabular}']
  (out/'baseline_table.tex').write_text('\n'.join(text),encoding='utf8')
 print(json.dumps({'figure':str(fig/'robustness_grid.pdf'),'worst_auc':worst,'uniqueness':u}))
if __name__=='__main__':main()
