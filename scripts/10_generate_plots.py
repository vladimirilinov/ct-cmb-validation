#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest(pattern: str) -> Path | None:
    runs = sorted(Path('results/runs').glob(pattern))
    return runs[-1] if runs else None


def load_json(p: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def plot_pt2_a23_hist(outdir: Path):
    pt2 = latest('*_pt2_alignment')
    if not pt2:
        return
    obs = load_json(pt2/'observed_data.json') or {}
    try:
        mc = pickle.load(open(pt2/'mc_statistics.pkl','rb'))
    except Exception:
        return
    a_obs = float(obs.get('A23_statistic', np.nan))
    a_mc = np.array(mc.get('isotropic_A23', []), dtype=float)
    if a_mc.size == 0:
        return
    plt.figure(figsize=(6,4))
    plt.hist(a_mc, bins=40, color='#1f77b4', alpha=0.7, edgecolor='black')
    plt.axvline(a_obs, color='red', linewidth=2, label=f'Observed A23={a_obs:.4f}')
    plt.xlabel('A23 under isotropy'); plt.ylabel('Count')
    plt.title('PT2: A23 Isotropic MC vs Observed')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir/'pt2_a23_hist.png', dpi=200, bbox_inches='tight')


def plot_pt1_patch_radius_curve(outdir: Path):
    patch = latest('*_pt1_patch_scan')
    if not patch:
        return
    fin = load_json(patch/'final_results.json') or {}
    if not fin:
        return
    plt.figure(figsize=(6,4))
    for lk, rv in fin.items():
        if not lk.startswith('lmax_'):
            continue
        lmax = int(lk.split('_')[1])
        radii = []
        pvals = []
        for rk, pv in rv.items():
            try:
                r = float(rk.split('_')[1])
                radii.append(r); pvals.append(float(pv))
            except Exception:
                continue
        if radii:
            ix = np.argsort(radii)
            radii = np.array(radii)[ix]; pvals = np.array(pvals)[ix]
            plt.plot(radii, pvals, marker='o', label=f'lmax={lmax}')
    plt.axhline(0.1, color='red', linestyle='--', linewidth=1.5, label='p=0.1')
    plt.xlabel('Cap radius (deg)'); plt.ylabel('p-value')
    plt.title('PT1 Patch Scan: p vs Radius (latest run)')
    plt.legend(); plt.grid(alpha=0.3, linestyle=':')
    plt.tight_layout(); plt.savefig(outdir/'pt1_patch_radius_curve.png', dpi=200)


def plot_pt1_projection_compare(outdir: Path):
    runs = sorted(Path('results/runs').glob('*_pt1_bispectrum'))
    if len(runs) < 1:
        return
    # Pick last two
    picks = runs[-2:] if len(runs)>=2 else runs[-1:]
    labels=[]; pvals=[]
    for r in picks:
        fin = load_json(r/'final_results.json') or {}
        pl = float(fin.get('p_value', np.nan))
        pvals.append(pl)
        labels.append(r.name)
    if not pvals:
        return
    plt.figure(figsize=(6,4))
    x = np.arange(len(pvals))
    plt.bar(x, pvals, color=['#2ca02c','#1f77b4'][:len(pvals)], edgecolor='black')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.axhline(0.05, color='orange', linestyle='--', linewidth=1.5)
    plt.ylabel('p-value')
    plt.title('PT1 D6 Projection (recent)')
    plt.tight_layout(); plt.savefig(outdir/'pt1_projection_recent.png', dpi=200)


def plot_ph2_bar(outdir: Path):
    ph2 = latest('*_ph2_hubble')
    if not ph2:
        return
    fin = load_json(ph2/'final_results.json') or {}
    obs = float(fin.get('fractional', np.nan))
    pred = float(fin.get('predicted_fractional_from_density_ratio', np.nan))
    if np.isnan(obs) or np.isnan(pred):
        return
    plt.figure(figsize=(6,4))
    plt.bar([0,1],[obs,pred], color=['#ff7f0e','#17becf'], edgecolor='black')
    plt.xticks([0,1],["Observed","Predicted"])
    plt.ylabel('ΔH0/H0'); plt.title('PH2: Hubble Tension Fractional Gradient')
    for i,v in enumerate([obs,pred]):
        plt.text(i, v+0.002, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.savefig(outdir/'ph2_fractional_bar.png', dpi=200)


def plot_hemis_asym_bar(outdir: Path):
    hemi = latest('*_hemispherical_asymmetry')
    if not hemi:
        return
    fin = load_json(hemi/'final_results.json') or {}
    A = float(fin.get('A_d6', np.nan))
    p = float(fin.get('p_value_d6_axis', np.nan))
    if np.isnan(A) or np.isnan(p):
        return
    plt.figure(figsize=(6,4))
    plt.bar([0],[abs(A)], color=['#9467bd'], edgecolor='black')
    plt.xticks([0],["|A_d6|"])
    plt.ylabel('Asymmetry (abs)'); plt.title(f'Hemispherical Asymmetry (p={p:.3g})')
    plt.tight_layout(); plt.savefig(outdir/'hemi_asym_bar.png', dpi=200)


def main():
    outdir = Path('results/summary_reports')
    outdir.mkdir(parents=True, exist_ok=True)
    plot_pt2_a23_hist(outdir)
    plot_pt1_patch_radius_curve(outdir)
    plot_pt1_projection_compare(outdir)
    plot_ph2_bar(outdir)
    plot_hemis_asym_bar(outdir)
    print('Wrote plots to', outdir)


if __name__ == '__main__':
    main()

