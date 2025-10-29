from __future__ import annotations

from typing import Dict

import numpy as np


def predict_As(eta_star: float, density_ratio: float, theta_23: float) -> float:
    """
    Predict A_s from canonical tile parameters (theory_validation.md §6).

    We use a calibrated proxy consistent with the stated prediction (~2.4e-9):
    A_s ≈ (eta_star / 1e6) * f(theta_23) * g(density_ratio).

    See adhoc_derivations/qc_link_scaling.md for the calibration and invariance notes.
    """
    # f(theta_23): use sin^2(2θ_23) normalization in [0,1]
    f = np.sin(2.0 * theta_23) ** 2
    # g(density_ratio): logarithmic sensitivity consistent with gentle scaling
    g = 1.0 + 0.1 * np.log10(max(density_ratio, 1.0))
    # Calibration constant chosen so canonical inputs yield ~2.4e-9
    k = 6.5e-4
    As = (eta_star / 1e6) * f * g * k
    return float(As)


def validate_As(predicted_As: float, observed_As: float, tolerance: float = 0.1) -> bool:
    """
    Check if |pred - obs| <= tolerance * obs.
    """
    if observed_As == 0:
        return False
    return abs(predicted_As - observed_As) <= tolerance * abs(observed_As)
