#!/usr/bin/env python3
"""
PT1 patch-based bispectrum scan using D6-aware cubic statistics.

Stages over increasing lmax and patch radii to search for localized signal
before committing heavy full-sky resources. Optionally ingests CMB-BEST
reconstruction FITS for metadata (k-grid, norms), though the estimator is
map-based and independent of basis.

Outputs a run directory with per-stage JSON results and logs.
"""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.theory.d6_group import D6Group


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def cap_mask(nside: int, center_vec: np.ndarray, radius_deg: float) -> np.ndarray:
    import healpy as hp  # type: ignore

    npix = hp.nside2npix(nside)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    center = unit(center_vec)
    cosang = vecs @ center
    return (cosang >= math.cos(math.radians(radius_deg))).astype(float)


def d6_m_selection_mask(lmax: int) -> np.ndarray:
    import healpy as hp  # type: ignore

    nalm = hp.Alm.getsize(lmax)
    ell, emm = hp.Alm.getlm(lmax, np.arange(nalm))
    mask = (ell % 2 == 0) & ((emm == 0) | (np.mod(emm, 6) == 0))
    return mask


def cubic_stat_from_masked_map(m: np.ndarray, lmax: int, sel_mask: np.ndarray, patch_mask: np.ndarray, f_patch: float) -> float:
    import healpy as hp  # type: ignore

    alm = hp.map2alm(m, lmax=lmax)
    alm_sel = np.zeros_like(alm)
    alm_sel[sel_mask] = alm[sel_mask]
    mp = hp.alm2map(alm_sel, nside=hp.get_nside(m), lmax=lmax, verbose=False)
    mp = mp * patch_mask
    return float(np.mean(mp ** 3) * 4.0 * np.pi / max(f_patch, 1e-6))


def latest_pt2_euler() -> List[float] | None:
    for r in sorted(Path("results/runs").glob("*_pt2_alignment"), reverse=True):
        fr = r / "final_results.json"
        if fr.exists():
            try:
                eul = json.load(open(fr))["best_fit_d6_orientation_euler_zyz"]
                return eul
            except Exception:
                pass
    return None


def maybe_ingest_cmbbest(fits_candidates: List[Path]) -> Dict[str, str]:
    from astropy.io import fits  # type: ignore

    meta = {}
    for p in fits_candidates:
        if p and p.exists():
            try:
                with fits.open(p) as hdul:
                    hdr = hdul[0].header
                    meta = {"basis": hdr.get("BASIS", ""), "f_sky": hdr.get("F_SKY", "")}
                break
            except Exception:
                continue
    return meta


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get("planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits")

    # Stages: increasing lmax and patch radii
    stages_env = os.getenv("CT_PT1_PATCH_LSTAGE", "64,96,128").split(",")
    l_stages = [int(s.strip()) for s in stages_env if s.strip()]
    radii_env = os.getenv("CT_PT1_PATCH_RADII", "20,30").split(",")
    radii = [float(s.strip()) for s in radii_env if s.strip()]
    nside = int(os.getenv("CT_PT1_PATCH_NSIDE", "256"))
    N_mc = int(os.getenv("CT_PT1_PATCH_NMC", "200"))
    workers = int(os.getenv("CT_WORKERS", "0") or 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt1_patch_scan")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    import healpy as hp  # type: ignore
    from scipy.spatial.transform import Rotation as R

    logger.info("Reading map: %s", map_path)
    m = hp.read_map(map_path, field=0, dtype=float, verbose=False)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)

    # Optional external mask to remove contaminated pixels globally
    mask_path = os.getenv("CT_PT1_MASK", "")
    gmask = None
    if mask_path and os.path.exists(mask_path):
        logger.info("Applying global mask: %s", mask_path)
        gmask = hp.read_map(mask_path, dtype=float, verbose=False)
        if hp.get_nside(gmask) != nside:
            gmask = hp.ud_grade(gmask, nside_out=nside)
        gmask = (gmask > 0).astype(float)
        m = m * gmask

    cl = hp.anafast(m, lmax=max(l_stages))

    # D6 axes from PT2
    eul = latest_pt2_euler() or [0.0, 0.0, 0.0]
    rot = R.from_euler("zyz", eul)
    d6 = D6Group()
    axes = d6.get_symmetry_axes()
    primary = rot.apply(axes["primary"])  # (3,)
    secondaries = rot.apply(axes["secondary"])  # (6,3)
    target_dirs = [primary] + [s for s in secondaries]

    # Ingest CMB-BEST FITS metadata if available
    cmbbest_meta = maybe_ingest_cmbbest([
        Path("data/processed/cmbbest_bispectrum.fits"),
        # Auto-discover latest reconstruct run
        next((p / "cmbbest_bispectrum.fits" for p in sorted(Path("results/runs").glob("*_cmbbest_reconstruct"), reverse=True) if (p / "cmbbest_bispectrum.fits").exists()),
             None) or Path("nonexistent"),
    ])
    if cmbbest_meta:
        json.dump(cmbbest_meta, open(run_dir / "cmbbest_meta.json", "w"), indent=2)

    # Iterate stages
    results: Dict[str, Dict[str, float]] = {}
    for lmax in l_stages:
        sel_mask = d6_m_selection_mask(lmax)
        stage_key = f"lmax_{lmax}"
        results[stage_key] = {}
        for rad in radii:
            # Build a reference patch (primary axis) for MC under isotropy — reuse for all axes
            ref_patch = cap_mask(nside, primary, rad)
            if gmask is not None:
                ref_patch = ref_patch * gmask
            f_ref = float(ref_patch.sum() / ref_patch.size)

            def mc_one_ref(seed: int) -> float:
                rr = np.random.default_rng(seed)
                mg = hp.synfast(cl, nside=nside, new=True, verbose=False)
                return cubic_stat_from_masked_map(mg * ref_patch, lmax, sel_mask, ref_patch, f_ref)

            if workers and workers > 1:
                vals = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for vstat in ex.map(mc_one_ref, range(N_mc)):
                        vals.append(vstat)
                S_mc_ref = np.asarray(vals, dtype=float)
            else:
                S_mc_ref = np.array([mc_one_ref(i) for i in range(N_mc)], dtype=float)

            best_p = 1.0
            best_dir = None
            per_dir = []
            for v in target_dirs:
                patch = cap_mask(nside, v, rad)
                if gmask is not None:
                    patch = patch * gmask
                f_patch = float(patch.sum() / patch.size)
                # Observed stat
                S_obs = cubic_stat_from_masked_map(m * patch, lmax, sel_mask, patch, f_patch)
                # Use reference MC (isotropy ⇒ rotation-invariant distribution)
                p_val = float(np.mean(np.abs(S_mc_ref) >= abs(S_obs)))
                per_dir.append({"dir": v.tolist(), "S_obs": S_obs, "p": p_val})
                if p_val < best_p:
                    best_p = p_val
                    best_dir = v

            logger.info("Stage lmax=%d, rad=%.1f deg: best p=%.4f", lmax, rad, best_p)
            results[stage_key][f"radius_{rad}"] = best_p
            json.dump({"per_dir": per_dir, "best": {"p": best_p, "dir": best_dir.tolist() if best_dir is not None else None}},
                      open(run_dir / f"stage_{stage_key}_rad_{int(rad)}.json", "w"), indent=2)

    json.dump(results, open(run_dir / "final_results.json", "w"), indent=2)
    logger.info("Saved patch-scan results to %s", run_dir / "final_results.json")


if __name__ == "__main__":
    main()
