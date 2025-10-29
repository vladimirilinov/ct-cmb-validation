from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .d6_group import D6Group


class D6ImpedanceMap:
    """
    Geometric impedance Z_D6(\hat{n}) as in theory_validation.md §3 (Def 3.2):

        Z(n) = Z0 + ε sum_{A in Axes} w_A (n · A)^2

    We support fast evaluation on a HEALPix grid (if healpy is available).
    """

    def __init__(self, d6_group: D6Group):
        self.d6 = d6_group

    def compute_impedance_map(
        self,
        nside: int,
        orientation,
        epsilon: float,
        weights: Dict[str, float] | None = None,
        Z0: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Z_D6 on the sphere using HEALPix.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE
        orientation : scipy.spatial.transform.Rotation
            Orientation of the canonical D6 axes.
        epsilon : float
            Coupling strength.
        weights : Dict[str, float]
            Weights for axis families: {'primary': wp, 'secondary': ws}
        Z0 : float
            Baseline.
        """
        try:
            import healpy as hp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "healpy is required for compute_impedance_map; install per requirements.txt"
            ) from e

        weights = weights or {"primary": 1.0, "secondary": 1.0}
        axes = self.d6.get_symmetry_axes()
        p = orientation.apply(axes["primary"])  # (3,)
        S = orientation.apply(axes["secondary"])  # (6,3)

        npix = hp.nside2npix(nside)
        vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T  # (npix,3)
        # Compute (n·A)^2 sum
        val = Z0 + epsilon * (
            weights.get("primary", 1.0) * (vecs @ p) ** 2
            + weights.get("secondary", 1.0) * np.sum((vecs @ S.T) ** 2, axis=1)
        )
        return val.astype(float)

    def compute_impedance_alm(
        self, nside: int, orientation, epsilon: float, weights: Dict[str, float] | None = None
    ) -> np.ndarray:
        try:
            import healpy as hp  # type: ignore
        except Exception as e:
            raise RuntimeError("healpy required for impedance_alm") from e

        zmap = self.compute_impedance_map(nside, orientation, epsilon, weights)
        lmax = 3 * nside - 1
        return hp.map2alm(zmap, lmax=lmax)

    def apply_modulation(self, cmb_alm: np.ndarray, z_alm: np.ndarray) -> np.ndarray:
        """
        Harmonic-space convolution using Gaunt coefficients:
            a_obs = sum a_cmb * z * ∫ Y Y Y dΩ.

        This convenience method provides a light-weight placeholder that scales
        a_cmb by the monopole of Z: a_obs ≈ (1 + z_00/√{4π}) a_cmb. For the
        rigorous evaluation, use `apply_modulation_gaunt` below which computes
        the exact Gaunt convolution via Wigner 3j symbols.
        """
        # z_00 corresponds to monopole term
        z00 = z_alm[0] if z_alm.size > 0 else 0.0
        # Monopole normalization for Y00 = 1/sqrt(4π)
        scale = 1.0 + float(np.real(z00)) / np.sqrt(4.0 * np.pi)
        return (scale * cmb_alm).astype(complex)

    def apply_modulation_gaunt(
        self,
        cmb_alm: np.ndarray,
        z_alm: np.ndarray,
        *,
        lmax: int,
        lz_max: int = 24,
    ) -> np.ndarray:
        """
        Exact harmonic-space convolution using Gaunt coefficients computed via
        Wigner 3j from pywigxjpf.

        a_{lm}^{obs} = Σ_{l1 m1 l2 m2} a_{l1 m1}^{cmb} z_{l2 m2} G^{l m}_{l1 m1, l2 m2},
        where G = sqrt((2l+1)(2l1+1)(2l2+1)/4π) (-1)^m (l l1 l2; 0 0 0) (l l1 l2; -m m1 m2).
        """
        import healpy as hp  # type: ignore
        import pywigxjpf

        pywigxjpf.wig_table_init(2 * max(lmax, lz_max) + 10, 9)
        pywigxjpf.wig_temp_init(2 * max(lmax, lz_max) + 10)

        def a_get(alm_arr: np.ndarray, L: int, ell: int, m: int) -> complex:
            if m >= 0:
                return alm_arr[hp.Alm.getidx(L, ell, m)]
            return ((-1) ** m) * np.conjugate(alm_arr[hp.Alm.getidx(L, ell, -m)])

        def z_get(z_arr: np.ndarray, Lz: int, ell: int, m: int) -> complex:
            if ell > Lz:
                return 0.0
            if m >= 0:
                return z_arr[hp.Alm.getidx(Lz, ell, m)] if ell <= Lz else 0.0
            return ((-1) ** m) * np.conjugate(z_arr[hp.Alm.getidx(Lz, ell, -m)])

        Lcmb = lmax
        Lz = min(lz_max, hp.Alm.getlmax(z_alm.size))
        out = np.zeros_like(cmb_alm, dtype=complex)

        two_pi = 4.0 * np.pi
        for ell in range(0, lmax + 1):
            norm_l = np.sqrt((2 * ell + 1) / two_pi)
            for m in range(0, ell + 1):  # packed alm only stores m>=0
                acc = 0.0 + 0.0j
                for l2 in range(0, min(Lz, lz_max) + 1):
                    if l2 % 2 == 1:
                        continue
                    for l1 in range(abs(ell - l2), min(lmax, ell + l2) + 1):
                        w000 = pywigxjpf.wig3jj(2 * ell, 2 * l1, 2 * l2, 0, 0, 0)
                        if w000 == 0.0:
                            continue
                        pref = ((-1) ** m) * np.sqrt((2 * l1 + 1) * (2 * l2 + 1)) * w000 * norm_l
                        for m1 in range(-l1, l1 + 1):
                            m2 = m - m1
                            if m2 < -l2 or m2 > l2:
                                continue
                            wmm = pywigxjpf.wig3jj(2 * ell, 2 * l1, 2 * l2, -2 * m, 2 * m1, 2 * m2)
                            if wmm == 0.0:
                                continue
                            acc += a_get(cmb_alm, Lcmb, l1, m1) * z_get(z_alm, Lz, l2, m2) * pref * wmm
                out[hp.Alm.getidx(lmax, ell, m)] = acc

        pywigxjpf.wig_temp_free()
        pywigxjpf.wig_table_free()
        return out
