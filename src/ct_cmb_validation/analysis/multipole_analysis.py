from __future__ import annotations

import numpy as np
from typing import Optional


def _alm_index(lmax: int, ell: int, m: int) -> int:
    """Return healpy-compatible alm index for given (ell, m).

    Assumes RING indexing: index = ell(ell+1)/2 + m for m>=0 in the packed alm format.
    For libraries using complex alm with m in [0..ell] packed, this matches.
    """
    if m < 0 or m > ell:
        raise ValueError("m must be in [0, ell] for packed alm")
    return ell * (ell + 1) // 2 + m


def extract_multipole_alm(alm: np.ndarray, lmax_in: int, ell: int) -> np.ndarray:
    """
    Safely extract all a_{ell m} for a single ell from a packed alm array.

    Parameters
    ----------
    alm : np.ndarray
        Packed alm array with length (lmax_in+1)(lmax_in+2)/2 (healpy convention).
    lmax_in : int
        Max multipole contained in `alm`.
    ell : int
        Multipole to extract.

    Returns
    -------
    np.ndarray
        Complex array of length ell+1 with m=0..ell.
    """
    if ell > lmax_in:
        raise ValueError("ell exceeds input lmax")
    out = np.zeros(ell + 1, dtype=complex)
    for m in range(ell + 1):
        out[m] = alm[_alm_index(lmax_in, ell, m)]
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector")
    return v / n


def get_preferred_direction(
    alm: np.ndarray, lmax_in: int, ell: int, *, nside_preview: Optional[int] = None
) -> np.ndarray:
    """
    Preferred direction for a given multipole ell.

    Strategy:
    - If healpy is available and `nside_preview` is provided, reconstruct the ell-only map
      and take the direction of maximum absolute temperature as the preferred axis.
    - Else, fall back to a robust algebraic approximation using the real-symmetric
      quadrupole tensor for ell=2. For ell=3 without healpy, raise a clear error.

    This function aligns with theory_validation.md §3 (Low-ℓ alignment).
    """
    try:
        import healpy as hp  # type: ignore

        if nside_preview is None:
            nside_preview = 32
        # Build zero alm up to lmax_in with only the requested ell retained
        alm_sel = np.zeros_like(alm)
        for m in range(ell + 1):
            idx = _alm_index(lmax_in, ell, m)
            alm_sel[idx] = alm[idx]
        m = hp.alm2map(alm_sel, nside=nside_preview, lmax=lmax_in, verbose=False)
        ipix = np.argmax(np.abs(m))
        vec = np.array(hp.pix2vec(nside_preview, ipix))
        return _unit(vec)
    except Exception:
        # healpy not available or failed — use spherical harmonics sampling
        try:
            from scipy.special import sph_harm
        except Exception as e:
            if ell == 2:
                # last-resort algebraic fallback for ell=2
                a2m = extract_multipole_alm(alm, lmax_in, 2)
                a20 = a2m[0]
                a21 = a2m[1]
                a22 = a2m[2]
                Q = np.zeros((3, 3), dtype=float)
                Q[2, 2] = float(a20.real)
                Q[0, 0] = float(-0.5 * a20.real + a22.real)
                Q[1, 1] = float(-0.5 * a20.real - a22.real)
                xy = float(a22.imag)
                xz = float(a21.real)
                yz = float(a21.imag)
                Q[0, 1] = Q[1, 0] = xy
                Q[0, 2] = Q[2, 0] = xz
                Q[1, 2] = Q[2, 1] = yz
                vals, vecs = np.linalg.eigh(Q)
                v = vecs[:, np.argmax(np.abs(vals))]
                return _unit(v)
            raise RuntimeError("scipy.special.sph_harm not available for fallback") from e

        # Sample the single-ell field on a spherical grid and find |T| maximum
        a_lm = extract_multipole_alm(alm, lmax_in, ell)
        n_theta = 90
        n_phi = 180
        thetas = np.linspace(0.0, np.pi, n_theta, endpoint=True)
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        max_abs = -1.0
        best_dir = None
        for th in thetas:
            # Precompute Y_{l m}(th, phi) decomposition efficiently across phi
            Y_th = []
            for m in range(ell + 1):
                Y_th.append(sph_harm(m, ell, phis, th))  # complex array length n_phi
            # Build T(th,phi)
            T = a_lm[0] * sph_harm(0, ell, phis, th)
            for m in range(1, ell + 1):
                T += 2.0 * np.real(a_lm[m] * Y_th[m])
            idx = int(np.argmax(np.abs(T)))
            val = float(np.abs(T[idx]))
            if val > max_abs:
                max_abs = val
                phi_best = phis[idx]
                best_dir = np.array(
                    [np.sin(th) * np.cos(phi_best), np.sin(th) * np.sin(phi_best), np.cos(th)]
                )
        if best_dir is None:
            raise RuntimeError("Failed to determine preferred direction via sampling")
        return _unit(best_dir)


def compute_alignment_statistic(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Alignment statistic A_23 = |v1 · v2| in [0,1].
    """
    v1u = _unit(v1)
    v2u = _unit(v2)
    return float(abs(np.dot(v1u, v2u)))
