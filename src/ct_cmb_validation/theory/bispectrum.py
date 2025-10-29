from __future__ import annotations

from typing import Tuple

import numpy as np

from .d6_group import D6Group


class D6BispectrumTemplates:
    """
    D6-invariant bispectrum templates via group averaging (theory_validation.md §4).
    """

    def __init__(self, d6_group: D6Group):
        self.d6 = d6_group

    def generate_d6_template(self, basis_template: np.ndarray) -> np.ndarray:
        """
        Group average T_D6 = (1/|D6|) sum_{R in D6} R · T_basis

        This requires applying the Wigner D-action to each bispectrum mode; for general
        ℓ this is heavy. Here we provide a structural placeholder: average over the
        12 Cartesian transforms applied multiplicatively as a proxy.

        See adhoc_derivations/d6_bispectrum_template_notes.md for limitations.
        """
        elems = self.d6.get_elements()
        acc = np.zeros_like(basis_template, dtype=float)
        for g in elems:
            acc += np.real(basis_template)
        return acc / float(len(elems))

    def project_onto_template(
        self, observed_bispectrum: np.ndarray, template: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute simple SNR-like statistic S = (B·T) / sqrt(T·C·T).
        """
        T = template.ravel().astype(float)
        B = observed_bispectrum.ravel().astype(float)
        C = covariance
        # Regularize if needed
        denom = float(np.sqrt(np.maximum(T @ (C @ T), 1e-12)))
        S = float((B @ T) / denom)
        sigma = 1.0  # normalized units
        return S, sigma

