#!/usr/bin/env python3
"""
P-T1 Bispectrum Test (D6-invariant template projection).

Reference: theory_validation.md §4 (D6 non-Gaussian signatures).

This script expects observed bispectrum and covariance from Planck FITS or preprocessed
arrays. Due to computational intensity of exact Wigner-D actions, this implementation
uses a documented proxy (group-averaging in array space); see
adhoc_derivations/d6_bispectrum_template_notes.md for limitations and required upgrades.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.theory.d6_group import D6Group
from src.ct_cmb_validation.theory.bispectrum import D6BispectrumTemplates
from src.ct_cmb_validation.analysis.statistics import monte_carlo_p_value
from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    N_mc = int(cfg.get("pt1_bispectrum", {}).get("N_mc", 200))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt1_bispectrum")
    ensure_dir(str(run_dir))
    ensure_dir(str(run_dir / "plots"))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    # Attempt to load observed bispectrum; fallback to synthetic arrays
    from src.ct_cmb_validation.data_handling.data_loader import load_planck_bispectrum
    rng = np.random.default_rng(0)
    B_obs = None
    C = None
    # Try our processed estimator output first, then official Planck FITS
    bispec_candidates = [
        "data/processed/estimated_bispectrum_cap.fits",
        "data/processed/estimated_bispectrum_equilateral.fits",
        cfg.get("paths", {}).get(
            "planck_bispectrum_fits", "data/raw/planck/COM_Bispectrum_I_2048_R3.00.fits"
        ),
    ]
    for bispec_path in bispec_candidates:
        try:
            logger.info("Checking bispectrum candidate: %s (exists=%s)", bispec_path, os.path.exists(bispec_path) if bispec_path else None)
            if bispec_path and os.path.exists(bispec_path):
                B_obs, C = load_planck_bispectrum(bispec_path)
                logger.info("Loaded bispectrum and covariance from %s [shape=%s, Cdim=%s]", bispec_path, getattr(B_obs, 'shape', None), getattr(C, 'shape', None))
                break
        except Exception as e:
            logger.warning("Failed to load candidate %s: %s", bispec_path, e)
            continue
    if B_obs is None:
        logger.warning(
            "Bispectrum FITS not available or failed to parse; using synthetic arrays."
        )
        B_obs = rng.normal(size=(16, 16, 16))
        C = np.eye(B_obs.size)

    # Load D6 template if available; else fall back to proxy averaging of a basis
    T_d6 = None
    try:
        t_candidates = [
            "data/processed/d6_bispectrum_template.fits",
        ]
        for tpath in t_candidates:
            if tpath and os.path.exists(tpath):
                from src.ct_cmb_validation.data_handling.data_loader import load_planck_bispectrum
                T_d6, _Ctmp = load_planck_bispectrum(tpath)
                logger.info("Loaded D6 template from %s [shape=%s]", tpath, getattr(T_d6, 'shape', None))
                break
    except Exception as e:
        logger.warning("Failed to load D6 template FITS: %s", e)

    if T_d6 is None:
        d6 = D6Group()
        templ = D6BispectrumTemplates(d6)
        basis = np.ones_like(B_obs)
        T_d6 = templ.generate_d6_template(basis)
        np.save(run_dir / "T_d6_template.npy", T_d6)

    # Align template/bispectrum lengths if needed
    def _align_len(x: np.ndarray, target_len: int) -> np.ndarray:
        if x.size == target_len:
            return x
        # Interpolate over normalized index space
        xs = np.linspace(0.0, 1.0, x.size)
        xt = np.linspace(0.0, 1.0, target_len)
        return np.interp(xt, xs, x)

    if isinstance(B_obs, np.ndarray) and B_obs.ndim == 1 and isinstance(T_d6, np.ndarray) and T_d6.ndim == 1:
        if B_obs.size != T_d6.size:
            T_d6 = _align_len(T_d6.astype(float), B_obs.size)

    # Compute S_obs using the available covariance
    logger.info("Shapes before projection: B=%s, T=%s, C=%s", getattr(B_obs, 'shape', None), getattr(T_d6, 'shape', None), getattr(C, 'shape', None))
    from src.ct_cmb_validation.theory.bispectrum import D6BispectrumTemplates as _Tmp
    _dummy = _Tmp(D6Group())
    S_obs, sigma_obs = _dummy.project_onto_template(B_obs, T_d6, C)

    # Monte Carlo under isotropy (Gaussian null)
    # Use vectorized generation for speed; optionally split across workers if N_mc is large.
    workers = int(os.getenv("CT_WORKERS", "0") or 0)
    if N_mc <= 1000 or workers <= 1:
        # Vectorized batch
        B_iso = rng.normal(size=(N_mc,) + B_obs.shape)
        Tflat = T_d6.ravel().astype(float)
        Cmat = C
        S_mc = np.empty(N_mc, dtype=float)
        # Compute (B·T)/sqrt(T·C·T) for each sample
        denom = float(np.sqrt(np.maximum(Tflat @ (Cmat @ Tflat), 1e-12)))
        B_flat = B_iso.reshape(N_mc, -1)
        S_mc[:] = (B_flat @ Tflat) / denom
    else:
        from concurrent.futures import ProcessPoolExecutor

        def _one(seed: int, shape, Tflat, Cmat):
            rng = np.random.default_rng(seed)
            B = rng.normal(size=shape)
            denom = float(np.sqrt(np.maximum(Tflat @ (Cmat @ Tflat), 1e-12)))
            return float(B.ravel().astype(float) @ Tflat) / denom

        with ProcessPoolExecutor(max_workers=workers) as ex:
            S_mc = list(ex.map(lambda s: _one(s, B_obs.shape, Tflat, C), range(N_mc)))
        S_mc = np.asarray(S_mc, dtype=float)

    p_value = monte_carlo_p_value(S_obs, S_mc)

    json.dump(
        {"S_obs": float(S_obs), "sigma_obs": float(sigma_obs), "p_value": float(p_value)},
        open(run_dir / "final_results.json", "w"),
    )
    logger.info("Saved bispectrum results to %s", run_dir)


if __name__ == "__main__":
    main()
