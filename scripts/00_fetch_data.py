#!/usr/bin/env python3
"""
Fetch/prepare raw data layout for ct-cmb-validation.

Mathematical context: theory_validation.md defines how these data are used
in P-T2 (alm), P-T1 (bispectrum), and P-H2 (H0). This script only prepares
paths and optional placeholders; it does not validate any physics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from shutil import copyfile

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")

    # Prepare raw data directories
    base = Path("data/raw")
    planck = base / "planck"
    local = base / "local_universe"
    ctparams = base / "ct_params"
    for p in [planck, local, ctparams]:
        ensure_dir(str(p))

    # Create placeholder JSON files if missing
    ph0 = planck / "planck_h0_cosmic.json"
    if not ph0.exists():
        json.dump({"value": 67.4, "stderr": 0.5}, open(ph0, "w"))
        logger.info("Wrote placeholder %s", ph0)

    lh0 = local / "h0_local.json"
    if not lh0.exists():
        json.dump({"value": 73.0, "stderr": 1.0}, open(lh0, "w"))
        logger.info("Wrote placeholder %s", lh0)

    tparams = ctparams / "t_d6_parameters.json"
    if not tparams.exists():
        json.dump(
            {
                "eta_star": 2.0,
                "density_ratio_rhoD_rhoR": 1.7e6,
                "theta_23_pmns": 0.759,
                "observed_As": 2.1e-9,
            },
            open(tparams, "w"),
        )
        logger.info("Wrote placeholder %s", tparams)

    logger.info("Data layout prepared under data/raw/ (FITS files must be added manually)")


if __name__ == "__main__":
    main()
