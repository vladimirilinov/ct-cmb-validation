#!/usr/bin/env python3
"""
P-H2 Hubble tension interpretation as calibration gradient.

Reference: theory_validation.md §5 (Hierarchical Gradients, P-H2).
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

from src.ct_cmb_validation.theory.gradients import interpret_hubble_tension, predict_hubble_fractional_from_density_ratio
from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.data_handling.data_loader import load_json


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_ph2_hubble")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    h0_local_path = paths.get("h0_local_json", "data/raw/local_universe/h0_local.json")
    h0_cosmic_path = paths.get("planck_h0_cosmic_json", "data/raw/planck/planck_h0_cosmic.json")
    h0_local = load_json(h0_local_path)  # {value, stderr}
    h0_cosmic = load_json(h0_cosmic_path)  # {value, stderr}
    result = interpret_hubble_tension(h0_local["value"], h0_cosmic["value"])
    # Optional quantitative prediction from energetic separation ρ_D/ρ_R
    ct_params_path = cfg.get("paths", {}).get("ct_params_json", "data/raw/ct_params/t_d6_parameters.json")
    try:
        params = load_json(ct_params_path)
        frac_pred = predict_hubble_fractional_from_density_ratio(params.get("density_ratio_rhoD_rhoR", 1.0))
        result.update({
            "predicted_fractional_from_density_ratio": frac_pred,
            "matches_within_tol": abs(frac_pred - result["fractional"]) <= 0.05  # 5% absolute tolerance
        })
    except Exception:
        pass
    json.dump(result, open(run_dir / "final_results.json", "w"))
    logger.info("Saved H0 interpretation to %s", run_dir)


if __name__ == "__main__":
    main()
