from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


class D6Group:
    """
    Dihedral group D6 in its (3,3) real representation acting on R^3.

    Reference: theory_validation.md §2 (Def 2.1–2.3)
    """

    def __init__(self) -> None:
        # Rotation by pi/3 about z-axis
        c = 0.5
        s = math.sqrt(3) / 2.0
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        # Reflection in the x-z plane (flip y)
        Sy = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

        elems: List[np.ndarray] = []
        I = np.eye(3)
        M = I.copy()
        # r^k
        for _ in range(6):
            elems.append(M.copy())
            M = M @ Rz
        # s r^k
        M = Sy.copy()
        for _ in range(6):
            elems.append(M.copy())
            M = M @ Rz

        # Store unique elements within numerical tolerance
        unique: List[np.ndarray] = []
        for E in elems:
            if not any(np.allclose(E, U) for U in unique):
                unique.append(E)
        self._elems = unique

        # symmetry axes per theory_validation.md §2.3
        self._primary = np.array([0.0, 0.0, 1.0], dtype=float)
        self._secondary = np.array(
            [
                [math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0), 0.0]
                for k in range(6)
            ],
            dtype=float,
        )

    def get_elements(self) -> List[np.ndarray]:
        return [E.copy() for E in self._elems]

    def get_symmetry_axes(self) -> Dict[str, np.ndarray]:
        return {"primary": self._primary.copy(), "secondary": self._secondary.copy()}

    def apply_to_direction(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (g @ v).astype(float)

    def wigner_d_matrix(self, ell: int, g: np.ndarray) -> np.ndarray:
        """
        Wigner D-matrix D^ell(g) of shape (2ell+1, 2ell+1).

        For general ell this requires robust special functions for numerical stability.
        Here we implement exact cases for ell=0,1 using the 3x3 rotation inferred from g.
        For ell>=2, raise NotImplementedError and recommend wigxjpf/ssht.

        Justification and references: theory_validation.md §4; see
        adhoc_derivations/wigner_d_notes.md for derivation notes.
        """
        from scipy.spatial.transform import Rotation as R

        if ell < 0:
            raise ValueError("ell must be non-negative")
        if ell == 0:
            return np.ones((1, 1), dtype=complex)
        if ell == 1:
            # D^1 is the standard 3D rotation in the spherical basis. We approximate by
            # using the Cartesian basis representation; for our group averaging it suffices
            # up to basis change. See derivation notes for basis conventions.
            return g.astype(complex)
        raise NotImplementedError(
            "Wigner D for ell>=2 not implemented; use external library for production."
        )

    def d6_selection_rules(self, ell: int, m: int) -> bool:
        """
        D6 selection rules for z_{ell m}: ell even and m multiple of 6 or 0.
        theory_validation.md §3.1.
        """
        if ell % 2 != 0:
            return False
        if m == 0:
            return True
        return (m % 6) == 0

