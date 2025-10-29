#!/usr/bin/env python3
"""
Cap-restricted GAUNT projection at lmax=128 around the D6 primary axis.

Steps:
- Build a circular cap mask at radius=25 deg around D6 primary axis
- Build a GAUNT D6 template with this mask (small N_mc for speed)
- Estimate cap-restricted equilateral bispectrum (bins of width 16 up to lmax)
- Save cap bispectrum to data/processed/estimated_bispectrum_cap.fits
- Run the PT1 projection script, which now prefers the cap bispectrum
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.utils.config_loader import ensure_dir
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.theory.d6_group import D6Group


def latest_pt2_euler() -> list[float]:
    runs = sorted(Path("results/runs").glob("*_pt2_alignment"))
    if not runs:
        return [0.0, 0.0, 0.0]
    fin = json.load(open(runs[-1]/"final_results.json"))
    return fin.get("best_fit_d6_orientation_euler_zyz", [0.0, 0.0, 0.0])


def build_cap_mask(nside: int, radius_deg: float) -> np.ndarray:
    import healpy as hp  # type: ignore
    from scipy.spatial.transform import Rotation as R
    eul = latest_pt2_euler()
    rot = R.from_euler("zyz", eul)
    d6 = D6Group()
    primary = rot.apply(d6.get_symmetry_axes()["primary"])  # (3,)
    npix = hp.nside2npix(nside)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    center = primary / (np.linalg.norm(primary) or 1.0)
    cosang = vecs @ center
    mask = (cosang >= np.cos(np.radians(radius_deg))).astype(float)
    return mask


def estimate_cap_bispectrum(m: np.ndarray, mask: np.ndarray, lmax: int, binw: int, N_mc: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import healpy as hp  # type: ignore
    # Mask map
    mm = m * mask
    # Bins
    edges = np.arange(2, lmax + 1, binw)
    if edges[-1] != lmax:
        edges = np.append(edges, lmax)
    bins = list(zip(edges[:-1], edges[1:]))
    Lc = np.array([(a + b) / 2.0 for a, b in bins], dtype=float)

    def top_hat_filter(Lmax, a, b):
        f = np.zeros(Lmax + 1)
        f[a:b+1] = 1.0
        return f

    # Observed
    alm = hp.map2alm(mm, lmax=lmax)
    B = np.zeros(len(bins), dtype=float)
    for i,(a,b) in enumerate(bins):
        fl = top_hat_filter(lmax, a, b)
        mf = hp.alm2map(hp.almxfl(alm, fl), nside=hp.get_nside(m), lmax=lmax, verbose=False)
        fcap = float(mask.mean())
        B[i] = float(np.mean((mf*mask) ** 3) * 4.0 * np.pi / max(fcap, 1e-6))

    # Variance via MC
    cl = hp.anafast(m, lmax=lmax)
    V = np.zeros_like(B)
    for s in range(N_mc):
        mg = hp.synfast(cl, nside=hp.get_nside(m), new=True, verbose=False)
        ag = hp.map2alm(mg*mask, lmax=lmax)
        for i,(a,b) in enumerate(bins):
            fl = top_hat_filter(lmax, a, b)
            mf = hp.alm2map(hp.almxfl(ag, fl), nside=hp.get_nside(m), lmax=lmax, verbose=False)
            val = float(np.mean((mf*mask) ** 3) * 4.0 * np.pi / max(mask.mean(), 1e-6))
            V[i] += val*val
    VAR = V/N_mc if N_mc>0 else np.zeros_like(B)
    return Lc, B, VAR


def write_cap_bispectrum_fits(out_path: Path, L: np.ndarray, B: np.ndarray, VAR: np.ndarray):
    from astropy.io import fits  # type: ignore
    cols = [
        fits.Column(name="L", format="D", array=L.astype(float)),
        fits.Column(name="B_equil", format="D", array=B.astype(float)),
        fits.Column(name="VAR", format="D", array=VAR.astype(float)),
    ]
    tbl = fits.BinTableHDU.from_columns(cols, name="BISP_EQUIL")
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(out_path, overwrite=True)


def main():
    logger = setup_logging()
    lmax = 128
    binw = 16
    nside = 512
    radius = 25.0
    # Build cap mask
    import healpy as hp  # type: ignore
    mask = build_cap_mask(nside, radius)
    cap_mask_path = Path("results/runs")/f"cap_mask_n{nside}_r{int(radius)}.fits"
    hp.write_map(str(cap_mask_path), mask, overwrite=True)
    # Build GAUNT template with this mask (small N_mc)
    env = os.environ.copy()
    env.update({
        "CT_WORKERS":"10",
        "CT_PT1_LMAX":str(lmax),
        "CT_PT1_BIN_WIDTH":str(binw),
        "CT_PT1_NSIDE":"256",
        "CT_PT1_NMC_TEMPLATE":"12",
        "CT_PT1_CHUNK_SIZE":"1",
        "CT_PT1_LOG_EVERY":"1",
        "CT_PT1_GAUNT":"1",
        "CT_PT1_LZMAX":"16",
        "CT_PT1_MASK":str(cap_mask_path),
    })
    subprocess.run([sys.executable, "scripts/02b_build_d6_bispectrum_template.py"], env=env, check=True)
    # Estimate cap bispectrum
    m = hp.read_map("data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits", field=0, dtype=float, verbose=False)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)
    Lc, Bcap, VAR = estimate_cap_bispectrum(m, mask, lmax=lmax, binw=binw, N_mc=300)
    cap_fits = Path("data/processed/estimated_bispectrum_cap.fits")
    ensure_dir(str(cap_fits.parent))
    write_cap_bispectrum_fits(cap_fits, Lc, Bcap, VAR)
    logger.info("Wrote cap bispectrum to %s", cap_fits)
    # Run projection
    subprocess.run([sys.executable, "scripts/02_run_pt1_bispectrum_test.py"], check=True)


if __name__ == "__main__":
    main()
