#!/usr/bin/env python3
"""
PT1 — D6 Anisotropic Bispectrum Test (m-selection per D6 symmetry).

Procedure (theory_validation.md §4 + §3 selection rules):
- Rotate observed alm by the inverse of the PT2 best-fit orientation so that D6 axes
  align with the canonical z-axis.
- For each ℓ-bin, zero out alm outside the bin AND zero out harmonics that violate
  D6 selection rules (ℓ even, m ≡ 0 mod 6 or m = 0). Reconstruct filtered map and
  compute S_bin = ∫ [T_bin(n)]^3 dΩ ≈ mean(T^3)·4π.
- Generate isotropic Gaussian alm from observed C_ℓ and compute the same statistic
  (no rotation needed due to isotropy) to form the null distribution.
- Aggregate S across bins (e.g., inverse-variance weighting) to form S_obs and p-value.

This test targets azimuthal structure (m) associated with D6, improving sensitivity
over rotationally invariant equilateral estimators.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.spatial.transform import Rotation as R

from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging


def load_best_orientation_euler() -> np.ndarray | None:
    runs = sorted(Path("results/runs").glob("*_pt2_alignment"))
    for r in reversed(runs):
        fr = r / "final_results.json"
        if fr.exists():
            try:
                data = json.load(open(fr))
                eul = data.get("best_fit_d6_orientation_euler_zyz")
                if eul and len(eul) == 3:
                    return np.array(eul, dtype=float)
            except Exception:
                continue
    return None


def build_masks(lmax: int, bin_width: int):
    import healpy as hp  # type: ignore

    nalm = hp.Alm.getsize(lmax)
    ell, emm = hp.Alm.getlm(lmax, np.arange(nalm))
    # D6 selection mask
    d6_mask = (ell % 2 == 0) & ((emm == 0) | (np.mod(emm, 6) == 0))
    # Bins
    edges = np.arange(2, lmax + 1, bin_width)
    if edges[-1] != lmax:
        edges = np.append(edges, lmax)
    bins = list(zip(edges[:-1], edges[1:]))
    bin_masks = []
    for a, b in bins:
        bin_mask = d6_mask & (ell >= a) & (ell <= b)
        bin_masks.append(bin_mask)
    return bins, bin_masks


def alm_apply_mask(alm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(alm)
    out[mask] = alm[mask]
    return out


def stat_from_alm(alm: np.ndarray, lmax: int, nside: int, *, mask: np.ndarray | None = None, f_sky: float = 1.0) -> float:
    import healpy as hp  # type: ignore

    m = hp.alm2map(alm, nside=nside, lmax=lmax, verbose=False)
    if mask is not None:
        m = m * mask
    return float(np.mean(m ** 3) * 4.0 * np.pi / max(f_sky, 1e-6))


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get(
        "planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    )
    pt1 = cfg.get("pt1_bispectrum", {})
    lmax = int(os.getenv("CT_PT1_LMAX", pt1.get("lmax_aniso", 512)))
    bin_width = int(os.getenv("CT_PT1_BIN_WIDTH", pt1.get("bin_width_aniso", pt1.get("bin_width", 32))))
    nside = int(os.getenv("CT_PT1_NSIDE", pt1.get("nside_aniso", 256)))
    N_mc = int(os.getenv("CT_PT1_NMC_ANISO", pt1.get("N_mc_aniso", 300)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt1_d6_anisotropic")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    import healpy as hp  # type: ignore

    # Load map and alm
    m_obs = hp.read_map(map_path, field=0, dtype=float, verbose=False)
    if hp.get_nside(m_obs) != nside:
        m_obs = hp.ud_grade(m_obs, nside_out=nside)
    # Optional mask
    mask = None
    f_sky = 1.0
    mask_path = os.getenv("CT_PT1_MASK", "")
    if mask_path and os.path.exists(mask_path):
        mask = hp.read_map(mask_path, dtype=float, verbose=False)
        if hp.get_nside(mask) != nside:
            mask = hp.ud_grade(mask, nside_out=nside)
        mask = (mask > 0).astype(float)
        f_sky = float(mask.sum() / mask.size)
        m_obs = m_obs * mask
    alm_obs = hp.map2alm(m_obs, lmax=lmax)

    # Rotate observed alm to align D6 axes with z
    eul = load_best_orientation_euler()
    if eul is not None:
        # Aligning axes: rotate map by inverse orientation
        alpha, beta, gamma = (-eul[0], -eul[1], -eul[2])
        hp.rotate_alm(alm_obs, alpha, beta, gamma, lmax=lmax)
        logger.info("Rotated observed alm by inverse best-fit Euler zyz=%s", eul)
    else:
        logger.warning("No best-fit orientation found; using identity")

    # Cl for MC
    cl = hp.anafast(m_obs, lmax=lmax)

    # Precompute masks
    bins, bin_masks = build_masks(lmax, bin_width)

    # Observed stats per bin
    S_bins = []
    for i, bmask in enumerate(bin_masks):
        alm_b = alm_apply_mask(alm_obs, bmask)
        s = stat_from_alm(alm_b, lmax, nside, mask=mask, f_sky=f_sky)
        S_bins.append(s)
    S_bins = np.array(S_bins, dtype=float)
    # Aggregate (inverse variance later, for now simple L2 norm)
    S_obs = float(np.sum(S_bins))

    # Monte Carlo (parallel threads; isotropy => no rotation needed)
    workers = int(os.getenv("CT_WORKERS", "0") or 0)

    def _one(seed: int) -> float:
        rr = np.random.default_rng(seed)
        alm_g = hp.synalm(cl, lmax=lmax, new=True, verbose=False)
        ssum = 0.0
        for bmask in bin_masks:
            alm_b = alm_apply_mask(alm_g, bmask)
            ssum += stat_from_alm(alm_b, lmax, nside, mask=mask, f_sky=f_sky)
        return float(ssum)

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        vals = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for v in ex.map(_one, range(N_mc)):
                vals.append(v)
        S_mc = np.asarray(vals, dtype=float)
    else:
        S_mc = np.array([_one(i) for i in range(N_mc)], dtype=float)

    # Two-sided p-value wrt null centered at zero
    p_val = float(np.mean(np.abs(S_mc) >= abs(S_obs)))

    # Save
    json.dump(
        {
            "S_bins": S_bins.tolist(),
            "S_obs": S_obs,
            "p_value": p_val,
            "lmax": lmax,
            "bin_width": bin_width,
            "N_mc": N_mc,
        },
        open(run_dir / "final_results.json", "w"),
    )

    # Also save MC stats
    import pickle

    pickle.dump({"S_mc": S_mc}, open(run_dir / "mc_statistics.pkl", "wb"))


if __name__ == "__main__":
    main()
