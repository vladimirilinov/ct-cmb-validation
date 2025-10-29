from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import curve_fit


def _sech2(x: np.ndarray, x0: float, xi: float, amp: float) -> np.ndarray:
    u = (x - x0) / xi
    return amp * (1.0 / np.cosh(u)) ** 2


def fit_sech_squared_profile(x_data: np.ndarray, y_data: np.ndarray) -> Dict:
    """
    Fit y ≈ amp * sech^2((x-x0)/xi) using non-linear least squares.
    theory_validation.md §5 (P-H1)
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    # initial guesses
    x0_guess = x[np.argmax(y)] if x.size else 0.0
    xi_guess = (x.max() - x.min()) / 10.0 if x.size > 1 else 1.0
    amp_guess = float(np.max(y)) if y.size else 1.0
    popt, pcov = curve_fit(_sech2, x, y, p0=[x0_guess, xi_guess, amp_guess])
    x0, xi, amp = map(float, popt)
    return {"x0": x0, "xi": xi, "amp": amp, "cov": pcov}


def interpret_hubble_tension(h0_local: float, h0_cosmic: float) -> Dict:
    """
    Interpret ΔH0 as a calibration gradient (theory_validation.md §5, P-H2).
    """
    delta = float(h0_local - h0_cosmic)
    frac = float(delta / h0_cosmic) if h0_cosmic != 0 else float("nan")
    return {"H0_local": float(h0_local), "H0_cosmic": float(h0_cosmic), "delta": delta, "fractional": frac}


def predict_hubble_fractional_from_density_ratio(density_ratio: float) -> float:
    """
    Predict the fractional H0 gradient from the energetic separation density_ratio ≈ ρ_D/ρ_R.

    Heuristic calibration: fractional ≈ k * log10(ρ_D/ρ_R), with k chosen so that
    ρ_D/ρ_R ≈ 1.7e6 produces ≈ 8.3% (Planck vs local). This yields k ≈ 0.0133.

    See adhoc_derivations/ph2_gradient_scaling.md for derivation and assumptions.
    """
    import numpy as np

    k = 0.0133
    return float(k * np.log10(max(density_ratio, 1.0)))
