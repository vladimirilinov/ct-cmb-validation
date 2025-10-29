from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np


def load_planck_alm(fits_path: str, lmax: int | None = None) -> Tuple[np.ndarray, int]:
    """
    Load Planck a_{lm} from a FITS file using healpy. If the file is a map
    (e.g., COM_CMB_IQU-smica_2048_R3.00_full.fits), compute a_lm via map2alm.

    Returns alm (packed, complex128) and lmax actually used.
    """
    try:
        import healpy as hp  # type: ignore

        try:
            # Try direct alm read
            alm = hp.read_alm(fits_path)
            if lmax is None:
                nalm = alm.size
                lm = 0
                while (lm + 1) * (lm + 2) // 2 < nalm:
                    lm += 1
                lmax_eff = lm
            else:
                lmax_eff = lmax
            return alm.astype(np.complex128), int(lmax_eff)
        except Exception:
            # Try reading a map (I component) and convert to alm
            m = hp.read_map(fits_path, field=0, dtype=float, verbose=False)
            if lmax is None:
                # Default lmax choice following HEALPix: ~3*nside-1
                nside = hp.get_nside(m)
                lmax_eff = 3 * nside - 1
            else:
                lmax_eff = lmax
            alm = hp.map2alm(m, lmax=lmax_eff)
            return alm.astype(np.complex128), int(lmax_eff)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get Planck alm from {fits_path}. Ensure healpy is installed. Error: {e}"
        )


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_planck_bispectrum(fits_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to load observed bispectrum and its covariance from a FITS file.
    Returns (B, C). If the format is unrecognized, raises an error.

    This loader is best-effort and may need adjustment for specific Planck
    product formats.
    """
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        raise RuntimeError("astropy is required to read bispectrum FITS files") from e

    with fits.open(fits_path) as hdul:
        # Case 1a: Our estimator FITS with BinTable HDU named BISP_EQUIL
        for h in hdul[1:]:
            name = getattr(h, "name", "")
            if name.upper().startswith("BISP_EQUIL") and hasattr(h, "data") and h.data is not None:
                L = np.asarray(h.data["L"]).astype(float)
                B = np.asarray(h.data["B_equil"]).astype(float)
                VAR = np.asarray(h.data["VAR"]).astype(float)
                return B, np.diag(VAR)
        # Case 1b: D6 template FITS with BinTable HDU named BISP_D6
        for h in hdul[1:]:
            name = getattr(h, "name", "")
            if name.upper().startswith("BISP_D6") and hasattr(h, "data") and h.data is not None:
                L = np.asarray(h.data["L"]).astype(float)
                T = np.asarray(h.data["T_D6"]).astype(float)
                VAR = np.asarray(h.data["VAR"]).astype(float)
                return T, np.diag(VAR)

        # Case 2: Generic arrays in the first two HDUs
        arrays = []
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            try:
                arr = np.asarray(data)
                if arr.size > 0:
                    arrays.append(arr)
            except Exception:
                continue
        if len(arrays) >= 2:
            B = np.asarray(arrays[0])
            C = np.asarray(arrays[1])
            if C.ndim == 1:
                C = np.diag(C)
            return B, C
    raise RuntimeError(
        "Unrecognized bispectrum FITS structure; supply Planck bispectrum FITS or use our estimator script to create one."
    )
