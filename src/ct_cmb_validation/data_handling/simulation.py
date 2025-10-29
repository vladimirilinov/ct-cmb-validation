from __future__ import annotations

import numpy as np


def random_unit_vector(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return v


def random_alignment_statistic(rng: np.random.Generator | None = None) -> float:
    v1 = random_unit_vector(rng)
    v2 = random_unit_vector(rng)
    return float(abs(np.dot(v1, v2)))

