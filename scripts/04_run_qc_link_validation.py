#!/usr/bin/env python3
"""
QC-Link validation: Predict A_s from T_D6 parameters and compare to Planck.

Reference: theory_validation.md §6 (Quantum-Cosmic Link).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.theory.qc_link import predict_As, validate_As
from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.data_handling.data_loader import load_json


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    tol_frac = float(cfg.get("qc_link", {}).get("tolerance_fraction", 0.1))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_qc_link")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    ct_params_path = paths.get("ct_params_json", "data/raw/ct_params/t_d6_parameters.json")
    params = load_json(ct_params_path)  # contains eta_star, density_ratio_rhoD_rhoR, theta_23_pmns, observed_As
    As_pred = predict_As(params["eta_star"], params["density_ratio_rhoD_rhoR"], params["theta_23_pmns"])
    is_valid = validate_As(As_pred, params["observed_As"], tol_frac)
    json.dump(
        {
            "predicted_As": As_pred,
            "observed_As": params["observed_As"],
            "is_valid": bool(is_valid),
        },
        open(run_dir / "final_results.json", "w"),
    )
    logger.info("Saved QC-Link validation to %s", run_dir)


if __name__ == "__main__":
    main()
