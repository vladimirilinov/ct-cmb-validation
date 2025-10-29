#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def hemisphere_asymmetry(m: np.ndarray, axis: np.ndarray) -> float:
    import healpy as hp  # type: ignore

    nside = hp.get_nside(m)
    npix = hp.nside2npix(nside)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    a = unit(axis)
    mask_n = (vecs @ a) >= 0
    mask_s = ~mask_n
    # Use mean square as power proxy
    Pn = float(np.mean(np.square(m[mask_n])))
    Ps = float(np.mean(np.square(m[mask_s])))
    if Pn + Ps == 0:
        return 0.0
    return (Pn - Ps) / (Pn + Ps)


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get("planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits")
    pt = cfg.get("pt1_bispectrum", {})
    nside = int(pt.get("nside_asym", 256))
    N_mc = int(pt.get("N_mc_asym", 500))
    workers = int(os.getenv("CT_WORKERS", "0") or 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_hemispherical_asymmetry")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())

    import healpy as hp  # type: ignore
    from scipy.spatial.transform import Rotation as R

    # Load map
    logger.info("Reading map: %s", map_path)
    m = hp.read_map(map_path, field=0, dtype=float, verbose=False)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)
    cl = hp.anafast(m, lmax=3 * nside - 1)

    # Load D6 primary axis from PT2 best-fit orientation
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
    axis_d6 = rot.apply([0.0, 0.0, 1.0])

    # Observed asymmetry along D6 axis
    A_d6 = hemisphere_asymmetry(m, axis_d6)
    logger.info("Observed hemispherical asymmetry along D6 axis: %.5f", A_d6)

    # MC under isotropy for fixed axis
    def mc_one(seed: int) -> float:
        rr = np.random.default_rng(seed)
        mg = hp.synfast(cl, nside=nside, new=True, verbose=False)
        return hemisphere_asymmetry(mg, axis_d6)

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        vals = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for v in ex.map(mc_one, range(N_mc)):
                vals.append(v)
        A_mc = np.asarray(vals, dtype=float)
    else:
        A_mc = np.array([mc_one(i) for i in range(N_mc)], dtype=float)

    p_val = float(np.mean(np.abs(A_mc) >= abs(A_d6)))

    # Also search for a best axis from random samples
    K = int(pt.get("N_axes_search", 200))
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(K, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    As = np.array([hemisphere_asymmetry(m, a) for a in axes])
    ibest = int(np.argmax(np.abs(As)))
    axis_best = axes[ibest]
    A_best = float(As[ibest])
    # Angle between best axis and D6 primary
    cosang = float(np.dot(unit(axis_best), unit(axis_d6)))
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(cosang), -1.0, 1.0))))

    json.dump(
        {
            "A_d6": float(A_d6),
            "p_value_d6_axis": p_val,
            "axis_d6": axis_d6.tolist(),
            "A_best": A_best,
            "axis_best": axis_best.tolist(),
            "angle_best_to_d6_deg": angle_deg,
        },
        open(run_dir / "final_results.json", "w"),
    )
    logger.info("Saved hemispherical asymmetry results to %s", run_dir)


if __name__ == "__main__":
    main()

