from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from ..theory.d6_group import D6Group


def monte_carlo_p_value(observed_stat: float, mc_stats: np.ndarray) -> float:
    mc = np.asarray(mc_stats, dtype=float)
    if mc.size == 0:
        return float("nan")
    return float(np.sum(mc >= observed_stat) / mc.size)


def find_optimal_orientation(
    observed_axes: np.ndarray, d6_group: D6Group
) -> Tuple[R, float]:
    """
    Maximize alignment between observed axes and rotated D6 symmetry axes.

    Objective aligns with theory_validation.md §3 (P-T2 alignment test).
    Returns the best Rotation and the alignment score in [0,1].
    """
    observed_axes = np.asarray(observed_axes, dtype=float)
    observed_axes = np.vstack([a / np.linalg.norm(a) for a in observed_axes])

    base_axes = d6_group.get_symmetry_axes()
    primary = base_axes["primary"].astype(float)
    secondary = base_axes["secondary"].astype(float)  # (6,3)

    def score_from_euler(euler_zyz):
        rot = R.from_euler("zyz", euler_zyz)
        p = rot.apply(primary)
        s = rot.apply(secondary)
        # Max alignment across observed vs. any D6 axis
        cand_axes = np.vstack([p[None, :], s])
        # compute pairwise dot magnitudes and take the maximum per observed axis, then average
        dots = np.abs(observed_axes @ cand_axes.T)
        best_per_obs = np.max(dots, axis=1)
        # negative for minimizer
        return -np.mean(best_per_obs)

    # Coarse initial point
    x0 = np.array([0.0, 0.0, 0.0])
    res = minimize(score_from_euler, x0, method="Powell")
    best_rot = R.from_euler("zyz", res.x)
    best_score = -float(res.fun)
    return best_rot, best_score

