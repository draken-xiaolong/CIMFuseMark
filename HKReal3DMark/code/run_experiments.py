#!/usr/bin/env python3
"""Train, personalize and evaluate HKReal3DMark with reproducible attack sweeps."""
from __future__ import annotations

import argparse, itertools, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


ATTACKS = {
 "rotation":[30,60,90,120,180], "scale":[.5,.7,1.5,2.0],
 "noise":[.001,.003,.005,.01,.02], "quantization":[.001,.003,.005,.01,.02],
 "point_delete":[.1,.2,.4,.6,.8], "crop":[.1,.2,.4,.6,.8],
 "simplify":[.1,.2,.4,.6,.8], "outliers":[.01,.03,.05,.1,.2],
 "sequential":[.1,.2,.4,.6,.8],
}


def canonical(points):
    x=points.astype(np.float64); x-=np.median(x,0,keepdims=True)
    radii=np.linalg.norm(x,axis=1); robust=x[radii<=np.quantile(radii,.80)]
    x/=max(float(np.quantile(np.linalg.norm(robust,axis=1),.95)),1e-8)
    robust=x[np.linalg.norm(x,axis=1)<=np.quantile(np.linalg.norm(x,axis=1),.80)]
    _,v=np.linalg.eigh(robust.T@robust/max(len(robust),1)); x=x@v[:,::-1]
    x=np.clip(x,-3.0,3.0)
    signs=np.sign(np.mean(x**3,0)); signs[signs==0]=1; return (x*signs).astype(np.float32)


def attack(points, family, level, rng):
    x=points.copy(); n=len(x)
    if family=="rotation":
        a=np.deg2rad(level); axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
        K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R=np.eye(3)*np.cos(a)+(1-np.cos(a))*np.outer(axis,axis)+np.sin(a)*K; x=x@R.T
    elif family=="scale": x*=level
    elif family=="noise": x+=rng.normal(0,level,x.shape)
    elif family=="quantization": x=np.round(x/level)*level
    elif family in {"point_delete","simplify"}:
        keep=max(16,int(n*(1-level))); x=x[rng.choice(n,keep,False)]
    elif family=="crop":
        axis=rng.integers(0,3); order=np.argsort(x[:,axis]); keep=max(16,int(n*(1-level))); x=x[order[:keep]]
    elif family=="outliers":
        k=max(1,int(n*level)); idx=rng.choice(n,k,False); x[idx]=rng.uniform(-2,2,(k,3))
    elif family=="sequential":
        x=attack(x,"noise",.02*level,rng); x=attack(x,"quantization",.02*level,rng)
        x=attack(x,"point_delete",level,rng); x=attack(x,"rotation",180*level,rng)
    x=canonical(x)
    if len(x)<n: x=x[rng.choice(len(x),n,True)]
    elif len(x)>n: x=x[rng.choice(len(x),n,False)]
    return x


class Encoder(nn.Module):
    def __init__(self, width=64, embedding=256):
        super().__init__(); self.point=nn.Sequential(nn.Linear(7,width),nn.GELU(),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width*2))
        self.att=nn.Linear(width*2,1); self.out=nn.Sequential(nn.Linear(width*6+24,embedding),nn.LayerNorm(embedding),nn.GELU(),nn.Linear(embedding,embedding))
    def forward(self,x):
        r=torch.linalg.norm(x,dim=-1,keepdim=True); f=torch.cat([x,x.abs(),r],-1); h=self.point(f)
        mean=h.mean(1); maximum=h.max(1).values; weighted=(h*torch.softmax(self.att(h),1)).sum(1)
        # Stable global distribution statistics complement learned local features.
        radii=torch.linalg.norm(x,dim=-1); qs=torch.quantile(radii,torch.linspace(.04,.96,24,device=x.device),dim=1).T
        return F.normalize(self.out(torch.cat([mean,maximum,weighted,qs],1)),dim=1)


def nc(a,b): return (a==b).float().mean(-1)
def metrics(pos,neg):
    pos=np.asarray(pos); neg=np.asarray(neg); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; s=np.r_[pos,neg]
    auc=float(roc_auc_score(y,s)); fpr,tpr,thr=roc_curve(y,s); fnr=1-tpr; i=np.argmin(abs(fpr-fnr))
    return {"mean_nc":float(pos.mean()),"q05_nc":float(np.quantile(pos,.05)),"auc":auc,"eer":float((fpr[i]+fnr[i])/2),"negative_q95":float(np.quantile(neg,.95))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="/Volumes/SANDISK-ELE/HKReal3DMarkData/converted/hk_points"); ap.add_argument("--out",default="/Volumes/SANDISK-ELE/HKReal3DMarkData/results/final"); ap.add_argument("--epochs",type=int,default=500); ap.add_argument("--seed",type=int,default=2026); ap.add_argument("--width",type=int,default=64); ap.add_argument("--bits",type=int,default=1024); ap.add_argument("--device",default="mps" if torch.backends.mps.is_available() else "cpu"); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device(args.device)
    root=Path(args.data); rows=json.loads((root/"manifest.json").read_text())["models"]
    # Clean and attacked branches must enter the encoder through the identical
    # canonicalization path; otherwise even a pure scale attack is evaluated
    # against a differently preprocessed clean reference.
    data=np.stack([canonical(points) for points in np.load(root/"points.npz")["points"]])
    train=np.array([i for i,r in enumerate(rows) if r["split"]=="train"]); test=np.array([i for i,r in enumerate(rows) if r["split"]=="test"])
    model=Encoder(args.width).to(device); opt=torch.optim.AdamW(model.parameters(),2e-4,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(args.seed); projection=F.normalize(torch.randn(args.bits,256,generator=generator),dim=1).to(device)
    history=[]; started=time.time()
    families=["rotation","noise","quantization","point_delete","crop","outliers","sequential"]
    for epoch in range(args.epochs):
        model.train(); batch=np.random.choice(train,min(24,len(train)),False); views=[]
        progress=(epoch+1)/args.epochs; max_strength=.25+.60*progress
        for idx in batch:
            rng=np.random.default_rng(args.seed+epoch*100003+int(idx)); p=data[idx]
            # Three independently attacked views per identity.
            scheduled=[families[epoch%len(families)],"crop","outliers"]
            vs=[p]+[attack(p,f,rng.uniform(.05,max_strength),rng) for f in scheduled]
            views.append(np.stack(vs))
        x=torch.from_numpy(np.stack(views)).to(device); B,V,N,D=x.shape; z=model(x.reshape(B*V,N,D)).reshape(B,V,-1)
        sim=torch.einsum('bvd,cmd->bvcm',z,z)/.12; labels=torch.arange(B,device=device)
        # Clean view must identify its attacked counterpart among the batch.
        contrast=sum(F.cross_entropy(sim[:,0,:,v],labels) for v in range(1,V))/(V-1)
        logits=torch.einsum('bvd,kd->bvk',z,projection); soft=torch.tanh(logits/.15)
        per_view=((soft[:,1:]-soft[:,:1])**2).mean((0,2)); stable=per_view.mean()+per_view.max(); quant=(1-soft.abs()).mean(); balance=soft[:,0].mean(0).pow(2).mean()
        clean=soft[:,0]; corr=(clean@clean.T)/args.bits; mask=~torch.eye(B,dtype=torch.bool,device=device); separation=F.relu(corr[mask]-.05).mean()
        loss=contrast+2.5*stable+.15*quant+.1*balance+1.0*separation
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if epoch==0 or (epoch+1)%50==0: history.append({"epoch":epoch+1,"loss":float(loss),"contrast":float(contrast),"stable":float(stable),"separation":float(separation)})
    model.eval(); clean=torch.from_numpy(data[test]).to(device)
    with torch.no_grad(): z=model(clean)
    # Clean-only enrollment: learn a balanced random code for each registered test tile.
    K=len(test); code=torch.empty(K,args.bits,device=device); rng=np.random.default_rng(args.seed)
    for bit in range(args.bits):
        col=np.r_[np.ones((K+1)//2),-np.ones(K//2)].astype(np.float32); rng.shuffle(col); code[:,bit]=torch.tensor(col,device=device)
    # Attack-aware enrollment still personalizes *only* the projection. Synthetic
    # views are generated online from each clean registration tile; encoder weights
    # stay frozen and no evaluation attack file is observed.
    enrollment=[]
    for view in range(4):
        augmented=np.stack([data[idx] if view==0 else attack(
            data[idx], ("crop","outliers","sequential")[view-1],
            (.45,.10,.35)[view-1], np.random.default_rng(args.seed+700001+view*1009+idx))
            for idx in test])
        with torch.no_grad(): enrollment.append(model(torch.from_numpy(augmented).to(device)))
    enrollment=torch.stack(enrollment,1)
    personalized=nn.Parameter(projection.clone()); po=torch.optim.Adam([personalized],lr=.015)
    for _ in range(1200):
        logits=torch.einsum('kvd,bd->kvb',enrollment,personalized)
        targets=code[:,None,:].expand_as(logits)
        loss=F.mse_loss(torch.tanh(logits/.12),targets)+.1*F.relu(.25-targets*logits).mean()+1e-4*(personalized-projection).pow(2).mean()
        po.zero_grad(); loss.backward(); po.step()
    with torch.no_grad(): clean_bits=(z@personalized.T>=0); neg=[float(nc(clean_bits[i],clean_bits[j])) for i,j in itertools.combinations(range(K),2)]
    curves={}
    for family,levels in ATTACKS.items():
        points=[]
        for level in levels:
            attacked=np.stack([attack(data[idx],family,float(level),np.random.default_rng(args.seed+idx+int(float(level)*1e6))) for idx in test])
            with torch.no_grad(): bits=(model(torch.from_numpy(attacked).to(device))@personalized.T>=0); pos=nc(clean_bits,bits).cpu().numpy()
            points.append({"intensity":level,**metrics(pos,neg),"scores":pos.tolist()})
        curves[family]=points
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    result={"dataset":{"total":len(rows),"train":len(train),"test":len(test)},"config":vars(args),"parameters":sum(p.numel() for p in model.parameters()),"history":history,"uniqueness":{"mean":float(np.mean(neg)),"q95":float(np.quantile(neg,.95)),"max":float(np.max(neg)),"pairs":len(neg)},"curves":curves,"elapsed_seconds":time.time()-started}
    (out/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); torch.save({"encoder":model.state_dict(),"projection":personalized.detach().cpu(),"config":vars(args)},out/"model.pt")
    print(json.dumps({"out":str(out),"uniqueness":result["uniqueness"],"elapsed":result["elapsed_seconds"],"worst_auc":min(p["auc"] for x in curves.values() for p in x)}))

if __name__=="__main__": main()
