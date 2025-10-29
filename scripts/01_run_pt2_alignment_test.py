#!/usr/bin/env python3
"""
P-T2 Alignment Test (Quadrupole-Octupole alignment vs. isotropy and D6 fit).

Reference: theory_validation.md §3 (Impedance and low-ℓ alignment).

Notes: Uses healpy if available for robust preferred-direction extraction. In its
absence, falls back to an algebraic quadrupole tensor approximation for ℓ=2.
See adhoc_derivations/quadrupole_tensor_from_a2m.md for mapping notes.
"""
from __future__ import annotations

import json
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.analysis.multipole_analysis import (
    compute_alignment_statistic,
    get_preferred_direction,
)
from src.ct_cmb_validation.analysis.statistics import (
    find_optimal_orientation,
    monte_carlo_p_value,
)
from src.ct_cmb_validation.data_handling.data_loader import load_planck_alm
from src.ct_cmb_validation.data_handling.simulation import random_unit_vector
from src.ct_cmb_validation.theory.d6_group import D6Group
from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging


def run() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    lmax = int(cfg.get("pt2_alignment", {}).get("lmax", 64))
    nside_preview = int(cfg.get("pt2_alignment", {}).get("nside_preview", 64))
    N_mc_iso = int(cfg.get("pt2_alignment", {}).get("N_mc_iso", 1000))
    N_mc_d6 = int(cfg.get("pt2_alignment", {}).get("N_mc_d6", 200))

    # Results directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt2_alignment")
    ensure_dir(str(run_dir))
    ensure_dir(str(run_dir / "plots"))

    # Copy config for audit
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    # Load alm (requires healpy)
    alm_path = cfg.get("paths", {}).get(
        "planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    )
    try:
        alm, lmax_in = load_planck_alm(alm_path, lmax)
    except Exception as e:
        # Graceful failure with audit trail
        logger.error("PT2 requires healpy to read/convert a_lm: %s", e)
        with open(run_dir / "final_results.json", "w") as f:
            json.dump(
                {
                    "status": "FAILED",
                    "reason": "healpy not installed or alm load failed",
                    "detail": str(e),
                },
                f,
            )
        logger.info("Wrote failure report to %s", run_dir / "final_results.json")
        return
    logger.info("Loaded alm up to lmax_in=%d from %s", lmax_in, alm_path)

    # Observed preferred axes
    v2 = get_preferred_direction(alm, lmax_in, 2, nside_preview=nside_preview)
    v3 = get_preferred_direction(alm, lmax_in, 3, nside_preview=nside_preview)
    A23_obs = compute_alignment_statistic(v2, v3)
    json.dump(
        {"v2_axis": v2.tolist(), "v3_axis": v3.tolist(), "A23_statistic": float(A23_obs)},
        open(run_dir / "observed_data.json", "w"),
    )
    logger.info("Observed A23=%.4f", A23_obs)

    # Isotropic MC (vectorized)
    rng = np.random.default_rng()
    V2 = rng.normal(size=(N_mc_iso, 3))
    V3 = rng.normal(size=(N_mc_iso, 3))
    V2 /= np.linalg.norm(V2, axis=1, keepdims=True)
    V3 /= np.linalg.norm(V3, axis=1, keepdims=True)
    iso_stats = np.abs(np.sum(V2 * V3, axis=1))
    p_iso = monte_carlo_p_value(A23_obs, iso_stats)
    logger.info("p_iso=%.6f", p_iso)

    # D6 fit
    d6 = D6Group()
    R_best, score_best = find_optimal_orientation(np.vstack([v2, v3]), d6)
    logger.info("Best D6 score=%.4f (Euler zyz=%s)", score_best, R_best.as_euler("zyz").tolist())

    # D6 MC
    # D6 MC (optionally parallel due to optimizer cost)
    import os
    workers = int(os.getenv("CT_WORKERS", "0") or 0)
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        def _score_one(seed: int):
            rr = np.random.default_rng(seed)
            v2r = rr.normal(size=3); v2r /= np.linalg.norm(v2r)
            v3r = rr.normal(size=3); v3r /= np.linalg.norm(v3r)
            _, sc = find_optimal_orientation(np.vstack([v2r, v3r]), d6)
            return float(sc)

        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                d6_stats = np.fromiter(ex.map(_score_one, range(N_mc_d6)), dtype=float, count=N_mc_d6)
        except Exception as e:
            logger.warning("Process pool unavailable (%s); falling back to threads", e)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                d6_stats = np.fromiter(ex.map(_score_one, range(N_mc_d6)), dtype=float, count=N_mc_d6)
    else:
        d6_stats = np.zeros(N_mc_d6, dtype=float)
        for j in range(N_mc_d6):
            v2r = random_unit_vector(rng)
            v3r = random_unit_vector(rng)
            _, sc = find_optimal_orientation(np.vstack([v2r, v3r]), d6)
            d6_stats[j] = sc
    p_d6 = monte_carlo_p_value(score_best, d6_stats)
    logger.info("p_d6=%.6f", p_d6)

    # Final results
    falsification = "SUPPORTED" if p_iso < 0.05 else "INCONCLUSIVE"
    json.dump(
        {
            "p_value_isotropic": float(p_iso),
            "best_fit_d6_orientation_euler_zyz": R_best.as_euler("zyz").tolist(),
            "best_fit_d6_score": float(score_best),
            "p_value_d6_fit": float(p_d6),
            "falsification_status": falsification,
        },
        open(run_dir / "final_results.json", "w"),
    )
    pickle.dump(
        {"isotropic_A23": iso_stats, "d6_fit_scores": d6_stats},
        open(run_dir / "mc_statistics.pkl", "wb"),
    )
    logger.info("Results saved under %s", run_dir)


if __name__ == "__main__":
    run()
