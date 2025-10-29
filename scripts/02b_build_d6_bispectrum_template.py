#!/usr/bin/env python3
"""
Build a D6 bispectrum template via impedance-modulated Gaussian simulations.

Outline (theory_validation.md §§3–4):
- Construct Z_D6(n) using D6 axes and best-fit orientation (from PT2).
- Draw Gaussian CMB maps from C_ℓ inferred from SMICA map.
- Modulate each map by Z_D6 and estimate the binned equilateral bispectrum.
- Average across MC to obtain a D6-shaped template T_D6(L); record variance.
- Save to FITS at data/processed/d6_bispectrum_template.fits for PT1 projection.

This is an approximation that captures anisotropy induced by multiplicative impedance;
it avoids explicit Wigner-D and Gaunt expansions while remaining faithful to the CT
mechanism described in §3.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
import json
import signal
import time
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
from src.ct_cmb_validation.theory.d6_group import D6Group
from src.ct_cmb_validation.theory.impedance import D6ImpedanceMap


def load_best_orientation_from_pt2() -> R | None:
    runs = sorted(Path("results/runs").glob("*_pt2_alignment"))
    for r in reversed(runs):
        fr = r / "final_results.json"
        if fr.exists():
            try:
                data = json.load(open(fr))
                eul = data.get("best_fit_d6_orientation_euler_zyz")
                if eul and len(eul) == 3:
                    return R.from_euler("zyz", eul)
            except Exception:
                continue
    return None


def bin_edges_and_filters(lmax: int, bin_width: int) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    import healpy as hp  # type: ignore

    edges = np.arange(2, lmax + 1, bin_width)
    if edges[-1] != lmax:
        edges = np.append(edges, lmax)
    bins = list(zip(edges[:-1], edges[1:]))
    fls = []
    for a, b in bins:
        f = np.zeros(lmax + 1)
        f[a : b + 1] = 1.0
        fls.append(f)
    return bins, fls


def estimate_equilateral_binned_from_map(m: np.ndarray, lmax: int, bins, fls, nside_map: int) -> np.ndarray:
    import healpy as hp  # type: ignore

    alm = hp.map2alm(m, lmax=lmax)
    out = np.zeros(len(bins), dtype=float)
    for i, (a, b) in enumerate(bins):
        alm_f = hp.almxfl(alm, fls[i])
        mf = hp.alm2map(alm_f, nside=nside_map, lmax=lmax, verbose=False)
        out[i] = float(np.mean(mf ** 3) * 4.0 * np.pi)
    return out


def main() -> None:
    logger = setup_logging()
    cfg = load_config("config/main_config.yaml")
    paths = cfg.get("paths", {})
    map_path = paths.get(
        "planck_alm_fits", "data/raw/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    )
    pt1 = cfg.get("pt1_bispectrum", {})
    # Allow environment overrides for heavy runs
    lmax = int(os.getenv("CT_PT1_LMAX", pt1.get("lmax_template", pt1.get("lmax_estimate", 512))))
    bin_width = int(os.getenv("CT_PT1_BIN_WIDTH", pt1.get("bin_width", 32)))
    nside_map = int(os.getenv("CT_PT1_NSIDE", pt1.get("nside_estimate", 256)))
    N_mc = int(os.getenv("CT_PT1_NMC_TEMPLATE", pt1.get("N_mc_template", 100)))
    epsilon = float(os.getenv("CT_PT1_EPSILON", pt1.get("epsilon_template", 0.02)))
    wp = float(os.getenv("CT_PT1_WEIGHT_PRIMARY", pt1.get("weight_primary", 1.0)))
    ws = float(os.getenv("CT_PT1_WEIGHT_SECONDARY", pt1.get("weight_secondary", 1.0)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{ts}_pt1_d6_template")
    ensure_dir(str(run_dir))
    with open("config/main_config.yaml", "rb") as src, open(run_dir / "run_config.yaml", "wb") as dst:
        dst.write(src.read())
    chk_dir = run_dir / "checkpoints"
    ensure_dir(str(chk_dir))

    import healpy as hp  # type: ignore

    logger.info("Loading map from %s", map_path)
    m = hp.read_map(map_path, field=0, dtype=float, verbose=False)
    if hp.get_nside(m) != nside_map:
        m = hp.ud_grade(m, nside_out=nside_map, order_in="NESTED", order_out="RING") if hp.isnsideok(hp.get_nside(m), nest=True) else hp.ud_grade(m, nside_out=nside_map)
    logger.info("Computing Cl up to lmax=%d", lmax)
    cl = hp.anafast(m, lmax=lmax)

    # Optional mask (applied in pixel space consistently for MC and obs)
    mask_path = os.getenv("CT_PT1_MASK", "")
    mask = None
    f_sky = 1.0
    if mask_path and os.path.exists(mask_path):
        logger.info("Loading mask from %s", mask_path)
        mask = hp.read_map(mask_path, dtype=float, verbose=False)
        if hp.get_nside(mask) != nside_map:
            mask = hp.ud_grade(mask, nside_out=nside_map)
        mask = (mask > 0).astype(float)
        f_sky = float(mask.sum() / mask.size)

    # Orientation
    rot = load_best_orientation_from_pt2() or R.from_euler("zyz", [0.0, 0.0, 0.0])
    d6 = D6Group()
    imp = D6ImpedanceMap(d6)
    logger.info("Computing impedance map (epsilon=%.4g, wp=%.3g, ws=%.3g)", epsilon, wp, ws)
    Z = imp.compute_impedance_map(nside_map, rot, epsilon, {"primary": wp, "secondary": ws})

    bins, fls = bin_edges_and_filters(lmax, bin_width)
    Lc = np.array([(a + b) / 2.0 for a, b in bins], dtype=float)
    logger.info("Binning configured: %d bins (width=%d)", len(bins), bin_width)

    # MC accumulation
    rng = np.random.default_rng(0)
    # Online mean/variance accumulators
    n_done = 0
    mean = np.zeros(len(bins), dtype=float)
    M2 = np.zeros(len(bins), dtype=float)

    workers = int(os.getenv("CT_WORKERS", "0") or 0)
    # Chunking and logging cadence
    chunk_size = int(os.getenv("CT_PT1_CHUNK_SIZE", 10))
    log_every = int(os.getenv("CT_PT1_LOG_EVERY", 1))
    use_gaunt = bool(int(os.getenv("CT_PT1_GAUNT", "0") or 0))
    # Optional cap on lz_max for GAUNT to control cost
    try:
        lzmax_override = int(os.getenv("CT_PT1_LZMAX", "0") or 0)
    except Exception:
        lzmax_override = 0

    # Metadata
    meta = {
        "map_path": map_path,
        "mask_path": mask_path,
        "f_sky": f_sky,
        "lmax": lmax,
        "bin_width": bin_width,
        "nside": nside_map,
        "N_mc": N_mc,
        "epsilon": epsilon,
        "wp": wp,
        "ws": ws,
        "CT_WORKERS": workers,
        "CT_PT1_GAUNT": int(use_gaunt),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    stop_flag = {"stop": False}

    def _sig_handler(signum, frame):
        logger.warning("Received signal %s; writing checkpoint and partial FITS, then exiting.", signum)
        write_partial()
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    def _one(seed: int) -> np.ndarray:
        rr = np.random.default_rng(seed)
        alm_g = hp.synalm(cl, lmax=lmax, new=True, verbose=False)
        # Rigorous modulation via Gaunt convolution when requested
        if use_gaunt:
            from src.ct_cmb_validation.theory.impedance import D6ImpedanceMap
            imp = D6ImpedanceMap(d6)
            z_alm = hp.map2alm(Z, lmax=lmax)
            lz_eff = min(24, lmax)
            if lzmax_override and lzmax_override > 0:
                lz_eff = min(lzmax_override, lz_eff)
            alm_mod = imp.apply_modulation_gaunt(alm_g, z_alm, lmax=lmax, lz_max=lz_eff)
            mZ = hp.alm2map(alm_mod, nside=nside_map, lmax=lmax, verbose=False)
        else:
            mg = hp.alm2map(alm_g, nside=nside_map, lmax=lmax, verbose=False)
            if mask is not None:
                mg = mg * mask
            mZ = mg * Z
        # Adjust cubic integral for sky fraction
        v = estimate_equilateral_binned_from_map(mZ, lmax, bins, fls, nside_map)
        return v / max(f_sky, 1e-6)

    def write_partial():
        # Write partial FITS (mean/M2)
        from astropy.io import fits  # type: ignore
        n = max(n_done, 1)
        var = (M2 / (n - 1)) if n > 1 else np.zeros_like(M2)
        cols = [
            fits.Column(name="L", format="D", array=Lc.astype(float)),
            fits.Column(name="T_D6", format="D", array=mean.astype(float)),
            fits.Column(name="VAR", format="D", array=var.astype(float)),
            fits.Column(name="N_eff", format="K", array=np.array([n], dtype=np.int64)),
        ]
        tbl = fits.BinTableHDU.from_columns(cols, name="BISP_D6")
        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary, tbl])
        out_partial = run_dir / "d6_bispectrum_template.partial.fits"
        hdul.writeto(out_partial, overwrite=True)

    def save_checkpoint():
        ck = {
            "n_done": n_done,
            "mean": mean,
            "M2": M2,
            "Lc": Lc,
            "bins": np.array(bins, dtype=int),
            "meta": meta,
            "time": time.time(),
        }
        np.savez(chk_dir / f"checkpoint_{n_done:06d}.npz", **ck)
        write_partial()

    # Resume support
    resume = bool(int(os.getenv("CT_PT1_RESUME", "0") or 0))
    resume_dir = os.getenv("CT_PT1_RESUME_DIR", "")
    if resume:
        base = Path(resume_dir) if resume_dir else chk_dir
        cks = sorted(base.glob("checkpoint_*.npz"))
        if cks:
            latest = cks[-1]
            logger.info("Resuming from checkpoint %s", latest)
            data = np.load(latest, allow_pickle=True)
            n_done = int(data["n_done"])
            mean = data["mean"].astype(float)
            M2 = data["M2"].astype(float)
        else:
            logger.info("Resume requested but no checkpoints found; starting fresh.")

    start = time.time()
    total = N_mc
    seeds_iter = iter(range(n_done, N_mc))
    chunk_idx = 0
    while n_done < N_mc and not stop_flag["stop"]:
        seeds = []
        try:
            for _ in range(chunk_size):
                seeds.append(next(seeds_iter))
        except StopIteration:
            pass
        if not seeds:
            break
        t0 = time.time()
        # pywigxjpf (GAUNT) can be sensitive to multi-threading; force serial for GAUNT
        workers_eff = workers if (workers and workers > 1 and not use_gaunt) else 0
        if workers_eff:
            from concurrent.futures import ThreadPoolExecutor
            outs = []
            with ThreadPoolExecutor(max_workers=workers_eff) as ex:
                for val in ex.map(_one, seeds):
                    outs.append(val)
            outs = np.asarray(outs, dtype=float)
        else:
            outs = np.vstack([_one(s) for s in seeds])
        # Update online mean/variance
        for v in outs:
            n_done += 1
            delta = v - mean
            mean += delta / n_done
            M2 += delta * (v - mean)
        chunk_idx += 1
        elapsed = time.time() - start
        rate = n_done / max(elapsed, 1e-6)
        eta = (total - n_done) / max(rate, 1e-6)
        if (chunk_idx % log_every) == 0:
            logger.info(
                "MC progress: %d/%d (%.1f%%), chunk=%d, elapsed=%.1fs, ETA=%.1fs",
                n_done,
                total,
                100.0 * n_done / max(total, 1),
                len(outs),
                elapsed,
                eta,
            )
        save_checkpoint()
        logger.info("Checkpoint saved at n=%d (chunk %.2fs)", n_done, time.time() - t0)
        if stop_flag["stop"]:
            break

    n_eff = max(n_done, 1)
    T = mean
    VAR = (M2 / (n_eff - 1)) if n_eff > 1 else np.zeros_like(M2)

    # Write FITS
    from astropy.io import fits  # type: ignore

    cols = [
        fits.Column(name="L", format="D", array=Lc.astype(float)),
        fits.Column(name="T_D6", format="D", array=T.astype(float)),
        fits.Column(name="VAR", format="D", array=VAR.astype(float)),
        fits.Column(name="N_eff", format="K", array=np.array([n_eff], dtype=np.int64)),
    ]
    tbl = fits.BinTableHDU.from_columns(cols, name="BISP_D6")
    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary, tbl])
    out_path = run_dir / "d6_bispectrum_template.fits"
    hdul.writeto(out_path, overwrite=True)
    logger.info("Wrote D6 template to %s", out_path)

    # Copy to processed path for PT1 consumption
    target = Path("data/processed/d6_bispectrum_template.fits")
    ensure_dir(str(target.parent))
    try:
        import shutil
        shutil.copy2(out_path, target)
        logger.info("Copied D6 template FITS to %s", target)
    except Exception as e:
        logger.warning("Could not copy D6 template: %s", e)


if __name__ == "__main__":
    main()
