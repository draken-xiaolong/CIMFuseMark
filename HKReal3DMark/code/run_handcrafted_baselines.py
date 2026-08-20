#!/usr/bin/env python3
"""Same-data, same-attack handcrafted zero-watermark baselines."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from run_experiments import ATTACKS,attack,canonical

def feature(x,name):
 r=np.linalg.norm(x,axis=1); eps=1e-8
 if name=='RadialHist': return np.histogram(r,bins=128,range=(0,3),density=True)[0]
 if name=='SphericalHist':
  az=np.arctan2(x[:,1],x[:,0]);el=np.arctan2(x[:,2],np.linalg.norm(x[:,:2],axis=1)+eps)
  return np.r_[np.histogram(r,64,(0,3),density=True)[0],np.histogram(az,64,(-np.pi,np.pi),density=True)[0],np.histogram(el,64,(-np.pi/2,np.pi/2),density=True)[0]]
 q=np.linspace(.01,.99,64);cov=np.linalg.eigvalsh(x.T@x/len(x));return np.r_[np.quantile(r,q),np.quantile(np.abs(x[:,0]),q),np.quantile(np.abs(x[:,1]),q),np.quantile(np.abs(x[:,2]),q),cov]
def bits(x,name,key=2026,B=1024):
 f=feature(x,name).astype(np.float64);f=(f-f.mean())/(f.std()+1e-8);R=np.random.default_rng(key+len(f)).normal(size=(B,len(f)));return R@f>=0
def score(a,b):return float(np.mean(a==b))
def auth(pos,neg):
 y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];s=np.r_[pos,neg];fpr,tpr,_=roc_curve(y,s);i=np.argmin(abs(fpr-(1-tpr)));return float(roc_auc_score(y,s)),float((fpr[i]+1-tpr[i])/2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed');p.add_argument('--out',default='paper/data/baselines.json');p.add_argument('--seed',type=int,default=2026);a=p.parse_args();root=Path(a.data);rows=json.loads((root/'manifest.json').read_text())['models'];raw=np.load(root/'points.npz')['points'];data=np.stack([canonical(x) for x in raw]);ids=[i for i,x in enumerate(rows) if x['split']=='test'];out={}
 for name in ['RadialHist','SphericalHist','MomentFusion']:
  clean=[bits(data[i],name,a.seed) for i in ids];neg=[score(clean[i],clean[j]) for i,j in itertools.combinations(range(len(ids)),2)];curves={}
  for fam,levels in ATTACKS.items():
   vv=[]
   for level in levels:
    pos=[score(clean[k],bits(attack(data[idx],fam,float(level),np.random.default_rng(a.seed+idx+int(float(level)*1e6))),name,a.seed)) for k,idx in enumerate(ids)];auc,eer=auth(pos,neg);vv.append({'intensity':level,'mean_nc':float(np.mean(pos)),'auc':auc,'eer':eer})
   curves[fam]=vv
  out[name]={'uniqueness':{'mean':float(np.mean(neg)),'q95':float(np.quantile(neg,.95)),'max':float(np.max(neg))},'curves':curves}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps({k:{'q95':v['uniqueness']['q95'],'worst_auc':min(x['auc'] for z in v['curves'].values() for x in z)}for k,v in out.items()}))
if __name__=='__main__':main()
