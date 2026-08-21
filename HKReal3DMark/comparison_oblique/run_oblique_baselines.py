#!/usr/bin/env python3
"""Unified same-data adaptations of five oblique-photogrammetry baselines.

Four embedded methods retain their paper-specific carrier construction but use
one common deterministic sector/repetition wrapper, needed because the papers'
original example-specific grouping does not cover arbitrary B3DM tiles.  The
zero-watermark baseline never modifies geometry.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from run_experiments import ATTACKS, attack, canonical

METHODS = ("PC-Reversible26", "Nan25-Curvature", "Nan25-ZW", "Jiao23-Dual", "Qiu19-RI")


def nc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def sectors(x: np.ndarray, bits: int) -> np.ndarray:
    angle = (np.arctan2(x[:, 1], x[:, 0]) + np.pi) / (2 * np.pi)
    return np.minimum((angle * bits).astype(np.int64), bits - 1)


def local_geometry(x: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    ids = NearestNeighbors(n_neighbors=min(k + 1, len(x))).fit(x).kneighbors(return_distance=False)
    curvature = np.empty(len(x)); residual = np.empty_like(x, dtype=np.float64)
    for i, row in enumerate(ids):
        local = x[row[1:]]; center = local.mean(0); residual[i] = x[i] - center
        values = np.linalg.eigvalsh((local-center).T @ (local-center) / max(1, len(local)))
        curvature[i] = values[0] / max(values.sum(), 1e-12)
    return curvature, residual


def carrier(x: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar carrier, movement direction and reliable-point mask."""
    eps = 1e-9; radius = np.linalg.norm(x, axis=1); radial = x / np.maximum(radius[:, None], eps)
    if method == "Qiu19-RI":
        return radius, radial, np.ones(len(x), bool)
    if method == "Jiao23-Dual":
        # Geometry branch of the paper's cluster-centroid distance carrier.
        octant = (x[:, 0] >= 0).astype(int) * 4 + (x[:, 1] >= 0).astype(int) * 2 + (x[:, 2] >= 0)
        centers = np.stack([np.median(x[octant == j], 0) if np.any(octant == j) else np.zeros(3) for j in range(8)])
        delta = x - centers[octant]; distance = np.linalg.norm(delta, axis=1)
        return distance, delta / np.maximum(distance[:, None], eps), np.ones(len(x), bool)
    curvature, residual = local_geometry(x)
    if method == "Nan25-Curvature":
        # Multi-level curvature selects stable vertices; local height residual is QIM carrier.
        mask = curvature >= np.quantile(curvature, .55); direction = np.zeros_like(x); direction[:, 2] = np.sign(residual[:, 2]); direction[direction[:, 2] == 0, 2] = 1
        return np.abs(residual[:, 2]), direction, mask
    # PC-Reversible26: normal-angle/curvature feature selection and local radius.
    mask = curvature >= np.quantile(curvature, .60); distance = np.linalg.norm(residual, axis=1)
    return distance, residual / np.maximum(distance[:, None], eps), mask


def qim_embed(points: np.ndarray, watermark: np.ndarray, method: str, step: float) -> np.ndarray:
    x = canonical(points).astype(np.float64)
    # Feature selection/cluster centres can move after embedding. Re-embedding
    # converges the blind decoder without retaining original vertex indices.
    for _ in range(6):
        index = sectors(x, len(watermark)); values, direction, mask = carrier(x, method)
        for bit in range(len(watermark)):
            chosen = np.flatnonzero((index == bit) & mask)
            if not len(chosen): continue
            mean = float(values[chosen].mean()); q = int(np.round(mean / step))
            if q % 2 != int(watermark[bit]): q += 1 if (q + 1) % 2 == int(watermark[bit]) else -1
            shift = q * step - mean
            # Repetition: all carriers in the sector vote for the same bit.
            x[chosen] += direction[chosen] * shift
    return x.astype(np.float32)


def qim_extract(points: np.ndarray, bits: int, method: str, step: float) -> tuple[np.ndarray, np.ndarray]:
    # Inputs are already canonical: embedding returns that frame and the shared
    # attack function canonicalizes its output. A second PCA pass can flip axes.
    x = np.asarray(points, dtype=np.float64); index = sectors(x, bits); values, _, mask = carrier(x, method)
    recovered = np.zeros(bits, dtype=np.uint8); observed = np.zeros(bits, bool)
    for bit in range(bits):
        chosen = np.flatnonzero((index == bit) & mask)
        if len(chosen):
            recovered[bit] = int(np.round(float(values[chosen].mean()) / step)) & 1; observed[bit] = True
    return recovered, observed


def nan25_feature(points: np.ndarray, bits: int, seed: int) -> np.ndarray:
    """Vertical-skewness/SVD-style content code used by Nan25-ZW."""
    x = canonical(points).astype(np.float64); z = x[:, 2]
    # Spatial cells preserve the paper's vertical-distribution premise.
    gx = np.clip(((x[:, 0] + 3) / 6 * 8).astype(int), 0, 7); gy = np.clip(((x[:, 1] + 3) / 6 * 8).astype(int), 0, 7)
    values=[]
    for cell in range(64):
        v=z[gx*8+gy==cell]
        if len(v)<3: values.extend((0.,0.,0.))
        else:
            c=v-v.mean(); s=c.std()+1e-9; values.extend((v.mean(),s,float(np.mean((c/s)**3))))
    matrix=np.asarray(values).reshape(16,12); u,s,v=np.linalg.svd(matrix,full_matrices=False); feature=np.r_[u.ravel(),s,v.ravel()]
    projection=np.random.default_rng(seed).standard_normal((bits,len(feature)))
    return (projection@feature>=0).astype(np.uint8)


def evaluate_method(data: np.ndarray, method: str, watermark: np.ndarray, seed: int, step: float) -> dict:
    if method == "Nan25-ZW":
        clean=[nan25_feature(x,len(watermark),seed) for x in data]
        clean_nc=[1.0]*len(data); coverage=[1.0]*len(data); marked=data
        negative=[nc(clean[i],clean[j]) for i,j in itertools.combinations(range(len(clean)),2)]
        uniqueness={"mean":float(np.mean(negative)),"q95":float(np.quantile(negative,.95)),"max":float(np.max(negative)),"pairs":len(negative)}
    else:
        marked=np.stack([qim_embed(x,watermark,method,step) for x in data])
        decoded=[qim_extract(x,len(watermark),method,step) for x in marked]
        clean_nc=[nc(bits[obs],watermark[obs]) if obs.any() else 0.0 for bits,obs in decoded]
        coverage=[float(obs.mean()) for _,obs in decoded]; clean=None; uniqueness=None
    curves={}
    for family,levels in ATTACKS.items():
        rows=[]
        for level in levels:
            scores=[]; cover=[]
            for i,x in enumerate(marked):
                attacked=attack(x,family,float(level),np.random.default_rng(seed+i+int(float(level)*1e6)))
                if method == "Nan25-ZW": scores.append(nc(clean[i],nan25_feature(attacked,len(watermark),seed))); cover.append(1.0)
                else:
                    bits,obs=qim_extract(attacked,len(watermark),method,step); scores.append(nc(bits[obs],watermark[obs]) if obs.any() else 0.0); cover.append(float(obs.mean()))
            rows.append({"intensity":level,"mean_nc":float(np.mean(scores)),"q05_nc":float(np.quantile(scores,.05)),"mean_coverage":float(np.mean(cover)),"scores":scores})
        curves[family]=rows
    clean_mean=float(np.mean(clean_nc)); coverage_mean=float(np.mean(coverage))
    return {"clean_nc":clean_mean,"clean_coverage":coverage_mean,
            "self_check":"pass" if clean_mean>=.95 and coverage_mean>=.60 else "fail",
            "uniqueness":uniqueness,"curves":curves}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data",default="/Volumes/SANDISK-ELE/HKReal3DMarkData/converted/hk_points"); p.add_argument("--out",default="/Volumes/SANDISK-ELE/HKReal3DMarkData/results/oblique_baselines/adapted_results.json"); p.add_argument("--seed",type=int,default=2026); p.add_argument("--bits",type=int,default=256); p.add_argument("--step",type=float,default=.02); p.add_argument("--limit",type=int,default=0); a=p.parse_args()
    root=Path(a.data); manifest=json.loads((root/"manifest.json").read_text())["models"]; raw=np.load(root/"points.npz")["points"]
    ids=[i for i,r in enumerate(manifest) if r["split"]=="test"]; ids=ids[:a.limit or None]; data=np.stack([canonical(raw[i]) for i in ids])
    watermark=np.random.default_rng(a.seed).integers(0,2,a.bits,dtype=np.uint8); results={}
    for method in METHODS:
        print(f"running {method}",flush=True); results[method]=evaluate_method(data,method,watermark,a.seed,a.step)
    payload={"protocol":{"seed":a.seed,"bits":a.bits,"step":a.step,"models":len(data),"reproduction":"same-data paper-core adaptation"},"methods":results}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({m:{"clean_nc":v["clean_nc"],"coverage":v["clean_coverage"],"worst_nc":min(p["mean_nc"] for rows in v["curves"].values() for p in rows)} for m,v in results.items()}))


if __name__=="__main__": main()
