#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def list_runs(pattern: str) -> list[Path]:
    return sorted(Path("results/runs").glob(pattern))


def latest_run(pattern: str) -> Path | None:
    runs = list_runs(pattern)
    return runs[-1] if runs else None


def run_patch_scan(workers: int = 10, nmc: int = 400, lstages: str = "64,96,128", radii: str = "20,30", nside: int = 256) -> Path:
    env = os.environ.copy()
    env["CT_WORKERS"] = str(workers)
    env["CT_PT1_PATCH_LSTAGE"] = lstages
    env["CT_PT1_PATCH_RADII"] = radii
    env["CT_PT1_PATCH_NSIDE"] = str(nside)
    env["CT_PT1_PATCH_NMC"] = str(nmc)

    before = set(list_runs("*_pt1_patch_scan"))
    cmd = [sys.executable, "scripts/02d_run_pt1_patch_scan.py"]
    subprocess.run(cmd, env=env, check=True)
    after = set(list_runs("*_pt1_patch_scan"))
    new = sorted(list(after - before))
    if not new:
        # fallback to latest
        r = latest_run("*_pt1_patch_scan")
        if not r:
            raise RuntimeError("Patch scan produced no run directory")
        return r
    return new[-1]


def pick_compelling_candidate(patch_run: Path, threshold: float = 0.1) -> tuple[int, float] | None:
    res = json.load(open(patch_run / "final_results.json"))
    best = (None, None, 1.0)
    for k, v in res.items():
        if not k.startswith("lmax_"):
            continue
        lmax = int(k.split("_")[1])
        for rad_key, p in v.items():
            rad = float(rad_key.split("_")[1])
            if p < best[2]:
                best = (lmax, rad, p)
    if best[0] is not None and best[2] < threshold:
        return best[0], best[1]
    return None


def run_d6_template_gaunt(lmax: int, workers: int = 10, nmc: int = 12, nside: int = 256, binw: int = 16, mask: str | None = None) -> Path:
    env = os.environ.copy()
    env["CT_WORKERS"] = str(workers)
    env["CT_PT1_LMAX"] = str(lmax)
    env["CT_PT1_BIN_WIDTH"] = str(binw)
    env["CT_PT1_NSIDE"] = str(nside)
    env["CT_PT1_NMC_TEMPLATE"] = str(nmc)
    env["CT_PT1_CHUNK_SIZE"] = "2"
    env["CT_PT1_LOG_EVERY"] = "1"
    env["CT_PT1_GAUNT"] = "1"
    if mask:
        env["CT_PT1_MASK"] = mask
    before = set(list_runs("*_pt1_d6_template"))
    cmd = [sys.executable, "scripts/02b_build_d6_bispectrum_template.py"]
    subprocess.run(cmd, env=env, check=True)
    after = set(list_runs("*_pt1_d6_template"))
    new = sorted(list(after - before))
    if not new:
        r = latest_run("*_pt1_d6_template")
        if not r:
            raise RuntimeError("D6 template (GAUNT) produced no run directory")
        return r
    return new[-1]


def run_projection() -> Path:
    before = set(list_runs("*_pt1_bispectrum"))
    cmd = [sys.executable, "scripts/02_run_pt1_bispectrum_test.py"]
    subprocess.run(cmd, check=True)
    after = set(list_runs("*_pt1_bispectrum"))
    new = sorted(list(after - before))
    if not new:
        r = latest_run("*_pt1_bispectrum")
        if not r:
            raise RuntimeError("PT1 projection produced no run directory")
        return r
    return new[-1]


def main():
    # 1) Patch scan with moderate cost
    patch_run = run_patch_scan(workers=10, nmc=400, lstages="64,96,128", radii="20,30", nside=256)
    cand = pick_compelling_candidate(patch_run, threshold=0.1)
    summary = {"patch_run": str(patch_run), "candidate": cand}
    print("Patch scan complete:", summary, flush=True)

    # 2) If compelling candidate, do GAUNT=1 template at that lmax and project
    if cand:
        lmax, rad = cand
        # Conservative GAUNT run to stay within a couple of hours
        mask = "data/raw/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00 (1).fits"
        d6_run = run_d6_template_gaunt(lmax=lmax, workers=10, nmc=12, nside=256, binw=16, mask=mask)
        proj_run = run_projection()
        summary.update({"d6_gaunt_run": str(d6_run), "projection_run": str(proj_run)})
    else:
        summary.update({"note": "No compelling candidate (p<0.1) found; GAUNT step skipped."})

    out = Path("results/summary_reports")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"pt1_orchestrator_summary_{ts}.json").write_text(json.dumps(summary, indent=2))
    print("Summary written:", out / f"pt1_orchestrator_summary_{ts}.json")


if __name__ == "__main__":
    main()

