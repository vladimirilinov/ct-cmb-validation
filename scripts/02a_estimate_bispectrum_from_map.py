#!/usr/bin/env python3
"""
Estimate a simple, binned equilateral bispectrum from the SMICA temperature map
and save it as a FITS file usable by the PT1 bispectrum script.

This is a practical workaround when the official Planck bispectrum FITS is not
locally available. See theory_validation.md §4 and derivation notes in
adhoc_derivations/d6_bispectrum_template_notes.md for limitations.
"""
from __future__ import annotations

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

from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.analysis.bispectrum_estimator import (
    estimate_equilateral_from_map,
    write_equilateral_bispectrum_fits,
)


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get(
        "planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    )

    params = cfg.get("pt1_bispectrum", {})
    lmax = int(params.get("lmax_estimate", 512))
    binw = int(params.get("bin_width", 32))
    nside_map = int(params.get("nside_estimate", 256))
    N_mc = int(params.get("N_mc_estimate", 50))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt1_bispectrum_from_map")
    ensure_dir(str(run_dir))

    logger.info(
        "Estimating equilateral bispectrum from %s (lmax=%d, bin=%d, nside=%d, N_mc=%d)",
        map_path,
        lmax,
        binw,
        nside_map,
        N_mc,
    )
    res = estimate_equilateral_from_map(
        map_path, lmax=lmax, bin_width=binw, nside_map=nside_map, N_mc=N_mc
    )
    out_fits = run_dir / "estimated_bispectrum_equilateral.fits"
    write_equilateral_bispectrum_fits(str(out_fits), res)
    logger.info("Wrote bispectrum FITS to %s", out_fits)

    # also copy to data/processed for PT1 script to find it easily
    target = Path("data/processed/estimated_bispectrum_equilateral.fits")
    ensure_dir(str(target.parent))
    try:
        import shutil

        shutil.copy2(out_fits, target)
        logger.info("Copied bispectrum FITS to %s", target)
    except Exception as e:
        logger.warning("Could not copy to data/processed: %s", e)


if __name__ == "__main__":
    main()

