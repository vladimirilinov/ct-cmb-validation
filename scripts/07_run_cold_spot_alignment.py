#!/usr/bin/env python3
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

from src.ct_cmb_validation.utils.config_loader import ensure_dir, load_config
from src.ct_cmb_validation.utils.logging_setup import setup_logging
from src.ct_cmb_validation.theory.d6_group import D6Group


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def vangle_deg(a, b):
    a = unit(a); b = unit(b)
    ca = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(ca)))


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get("planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits")
    pt = cfg.get("pt1_bispectrum", {})
    nside = int(pt.get("nside_coldspot", 512))
    fwhm_deg = float(pt.get("coldspot_smoothing_deg", 5.0))
    N_mc = int(pt.get("N_mc_coldspot", 1000))
    workers = int(os.getenv("CT_WORKERS", "0") or 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_cold_spot_alignment")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    import healpy as hp  # type: ignore
    from scipy.spatial.transform import Rotation as R

    # Load and smooth map
    logger.info("Reading map: %s", map_path)
    m = hp.read_map(map_path, field=0, dtype=float, verbose=False)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)
    m_s = hp.smoothing(m, fwhm=np.radians(fwhm_deg), verbose=False)

    # Coldest pixel direction
    ipix = int(np.argmin(m_s))
    vec_cold = np.array(hp.pix2vec(nside, ipix))

    # D6 axes at best-fit orientation
    eul = None
    for r in sorted(Path("results/runs").glob("*_pt2_alignment"), reverse=True):
        fr = r / "final_results.json"
        if fr.exists():
            try:
                eul = json.load(open(fr))["best_fit_d6_orientation_euler_zyz"]
                break
            except Exception:
                pass
    rot = R.from_euler("zyz", eul) if eul is not None else R.from_euler("zyz", [0.0, 0.0, 0.0])
    d6 = D6Group()
    axes = d6.get_symmetry_axes()
    p = rot.apply(axes["primary"])  # (3,)
    S = rot.apply(axes["secondary"])  # (6,3)

    # Minimal angle between cold spot and any D6 axis
    angles = [vangle_deg(vec_cold, p)] + [vangle_deg(vec_cold, s) for s in S]
    min_angle = float(min(angles))
    logger.info("Min angle between cold spot and D6 axes: %.2f deg", min_angle)

    # MC: random orientations of D6 axes
    def mc_one(seed: int) -> float:
        rr = np.random.default_rng(seed)
        # random orientation via random rotation
        u = rr.random(3)
        # using zyz Euler generation
        Rr = R.from_euler("zyz", [2*np.pi*u[0], np.arccos(2*u[1]-1), 2*np.pi*u[2]])
        pp = Rr.apply(axes["primary"]) 
        SS = Rr.apply(axes["secondary"]) 
        angs = [vangle_deg(vec_cold, pp)] + [vangle_deg(vec_cold, s) for s in SS]
        return float(min(angs))

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        vals = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for v in ex.map(mc_one, range(N_mc)):
                vals.append(v)
        angles_mc = np.asarray(vals, dtype=float)
    else:
        angles_mc = np.array([mc_one(i) for i in range(N_mc)], dtype=float)

    p_val = float(np.mean(angles_mc <= min_angle))
    json.dump(
        {
            "cold_spot_vec": vec_cold.tolist(),
            "min_angle_deg": min_angle,
            "p_value_alignment": p_val,
            "angles_to_axes_deg": angles,
        },
        open(run_dir / "final_results.json", "w"),
    )
    logger.info("Saved cold spot/D6 alignment results to %s", run_dir)


if __name__ == "__main__":
    main()

