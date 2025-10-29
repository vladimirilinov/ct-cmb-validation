from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EquilateralBispectrumResult:
    L: np.ndarray  # bin centers
    B: np.ndarray  # estimated equilateral bispectrum per bin
    var: np.ndarray  # MC variance per bin (diagonal covariance)


def _top_hat_filter(lmax: int, lmin: int, lmax_bin: int) -> np.ndarray:
    f = np.zeros(lmax + 1, dtype=float)
    f[lmin : lmax_bin + 1] = 1.0
    return f


def estimate_equilateral_from_map(
    map_fits_path: str,
    *,
    lmax: int = 512,
    bin_width: int = 32,
    nside_map: int | None = None,
    N_mc: int = 50,
    rng: np.random.Generator | None = None,
) -> EquilateralBispectrumResult:
    """
    Estimate a binned equilateral bispectrum from a full-sky map by
    filtering in ℓ and integrating M_L^3 over the sphere.

    This is a documented, approximate estimator to provide a usable bispectrum FITS
    when official Planck bispectrum products are not available locally.
    """
    import healpy as hp  # type: ignore

    rng = rng or np.random.default_rng(0)
    # Read temperature map (I), get NSIDE
    m = hp.read_map(map_fits_path, field=0, dtype=float, verbose=False)
    if nside_map is None:
        nside_map = hp.get_nside(m)
    # Compute alm up to lmax
    alm = hp.map2alm(m, lmax=lmax)
    # Get Cl for Gaussian MC
    cl = hp.anafast(m, lmax=lmax)

    # Binning
    edges = np.arange(2, lmax + 1, bin_width)
    if edges[-1] != lmax:
        edges = np.append(edges, lmax)
    bins = list(zip(edges[:-1], edges[1:]))
    Lc = np.array([(a + b) / 2.0 for a, b in bins], dtype=float)
    B = np.zeros(len(bins), dtype=float)

    # Prepare denominator normalization (optional)
    # For now, use raw cubic integral; normalization constant cancels in template projection scaling.

    for i, (a, b) in enumerate(bins):
        fl = _top_hat_filter(lmax, a, b)
        # apply filter in ℓ-space
        # healpy packs alm; use almxfl which accepts f(l)
        alm_f = hp.almxfl(alm, fl)
        m_f = hp.alm2map(alm_f, nside=nside_map, lmax=lmax, verbose=False)
        # integral of cube over sphere ≈ mean(m^3) * 4π
        B[i] = float(np.mean(m_f ** 3) * 4.0 * np.pi)

    # MC for variance under Gaussian null
    V = np.zeros_like(B)
    for k in range(N_mc):
        alm_g = hp.synalm(cl, lmax=lmax, new=True, verbose=False)
        for i, (a, b) in enumerate(bins):
            fl = _top_hat_filter(lmax, a, b)
            mg = hp.alm2map(hp.almxfl(alm_g, fl), nside=nside_map, lmax=lmax, verbose=False)
            val = float(np.mean(mg ** 3) * 4.0 * np.pi)
            V[i] += val * val
    if N_mc > 1:
        V /= N_mc
    # Use sample variance ≈ E[X^2] - (E[X])^2; Gaussian null gives E[X]=0 theoretically
    var = V
    return EquilateralBispectrumResult(L=Lc, B=B, var=var)


def write_equilateral_bispectrum_fits(output_path: str, res: EquilateralBispectrumResult) -> None:
    from astropy.io import fits  # type: ignore

    cols = [
        fits.Column(name="L", format="D", array=res.L.astype(float)),
        fits.Column(name="B_equil", format="D", array=res.B.astype(float)),
        fits.Column(name="VAR", format="D", array=res.var.astype(float)),
    ]
    tbl = fits.BinTableHDU.from_columns(cols, name="BISP_EQUIL")
    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary, tbl])
    hdul.writeto(output_path, overwrite=True)

