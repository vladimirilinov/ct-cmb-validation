#!/usr/bin/env python3
"""
Axis-of-Evil figure — clarity/impact v3 (stable)
------------------------------------------------
Fixes vs previous attempt:
- Restore 3-panel layout (3rd panel is a dedicated summary so it never overlaps the sky map)
- Large, non-overlapping sky markers with pole-aware jitter and inline labels
- No inset or anchored boxes on the sky panel (prevents clipping/overdraw)
- Great-circle arc shown in BOTH 3D and sky panels
- Clean spacing (no constrained_layout surprises)

Output: results/summary_reports/axis_of_evil_CORRECTED.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------- IO ----------------
def _latest_pt2():
    runs = sorted(Path("results/runs").glob("*_pt2_alignment"))
    return runs[-1] if runs else None


def _read_json_safe(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _load_axes_and_meta():
    r = _latest_pt2()
    if not r:
        v2 = np.array([0.0, 0.0, 1.0])
        v3 = np.array([0.0, 0.0, -1.0])
        a23 = 1.0
        euler = [0.0, 0.0, 0.0]
        meta = {"p_iso": None, "p_d6": None, "run": "N/A",
                "ph2_obs": None, "ph2_pred": None, "A_hemi": None, "p_hemi": None}
        return v2, v3, a23, euler, meta

    obs = _read_json_safe(r / "observed_data.json")
    fin = _read_json_safe(r / "final_results.json")

    v2 = np.array(obs.get("v2_axis", obs.get("axis_l2", [0, 0, 1.0])), dtype=float)
    v3 = np.array(obs.get("v3_axis", obs.get("axis_l3", [0, 0, -1.0])), dtype=float)

    a23 = obs.get("A23_statistic", obs.get("A_23", obs.get("A23")))
    if a23 is None:
        v2n = v2 / (np.linalg.norm(v2) or 1.0)
        v3n = v3 / (np.linalg.norm(v3) or 1.0)
        a23 = float(abs(np.dot(v2n, v3n)))
    else:
        a23 = float(a23)

    euler = fin.get("best_fit_d6_orientation_euler_zyz",
                    fin.get("best_fit_euler_zyz", [0.0, 0.0, 0.0]))
    euler = [float(e) for e in euler]

    meta = {
        "p_iso": fin.get("p_iso", fin.get("p_isotropic")),
        "p_d6": fin.get("p_d6", fin.get("p_align_d6")),
        "run": r.name,
        "ph2_obs": fin.get("ph2_fractional_h0_obs") or fin.get("fractional_h0_obs"),
        "ph2_pred": fin.get("ph2_fractional_h0_pred") or fin.get("fractional_h0_pred"),
        "A_hemi": fin.get("A_d6") or fin.get("A_hemi_d6"),
        "p_hemi": fin.get("p_hemi_d6") or fin.get("p_d6_hemi"),
    }
    return v2, v3, a23, euler, meta


# ---------------- Math ----------------
def _euler_zyz_to_R(a, b, g):
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    Rz_a = np.array([[ca, -sa, 0.0],[sa, ca, 0.0],[0.0, 0.0, 1.0]])
    Ry_b = np.array([[cb, 0.0, sb],[0.0, 1.0, 0.0],[-sb, 0.0, cb]])
    Rz_g = np.array([[cg, -sg, 0.0],[sg, cg, 0.0],[0.0, 0.0, 1.0]])
    return Rz_a @ Ry_b @ Rz_g


def _vec_to_lonlat(v):
    v = v / (np.linalg.norm(v) or 1.0)
    x, y, z = v
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    return float(lon), float(lat)


def _slerp_arc(u, v, npts=120):
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    om = np.arccos(dot)
    if om < 1e-6:
        return np.tile(u, (npts, 1))
    t = np.linspace(0, 1, npts)
    s = np.sin(om)
    return (np.sin((1 - t) * om)[:, None] * u + np.sin(t * om)[:, None] * v) / s


# ---------------- Plot ----------------
def create_corrected_axis_figure():
    v2, v3, A23, euler, meta = _load_axes_and_meta()

    # normalize
    v2 = v2 / (np.linalg.norm(v2) or 1.0)
    v3 = v3 / (np.linalg.norm(v3) or 1.0)

    # D6 axis from Euler(zyz)
    R = _euler_zyz_to_R(*euler)
    d6 = R @ np.array([0.0, 0.0, 1.0])
    d6 = d6 / (np.linalg.norm(d6) or 1.0)

    ang_deg = np.degrees(np.arccos(np.clip(A23, -1.0, 1.0)))

    # figure (no constrained_layout to avoid auto-shifts)
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="mollweide")
    ax3 = fig.add_subplot(133)

    # ---------- (A) 3D ----------
    u = np.linspace(0, 2*np.pi, 42)
    th = np.linspace(0, np.pi, 21)
    xs = 0.98 * np.outer(np.cos(u), np.sin(th))
    ys = 0.98 * np.outer(np.sin(u), np.sin(th))
    zs = 0.98 * np.outer(np.ones_like(u), np.cos(th))
    ax1.plot_wireframe(xs, ys, zs, alpha=0.08, color="gray", linewidth=0.6)

    off = 0.14
    ax1.plot([0, -off], [0, 0], [0, 0], linestyle=":", color="gray", linewidth=1)
    ax1.plot([0,  +off], [0, 0], [0, 0], linestyle=":", color="gray", linewidth=1)
    ax1.quiver(-off, 0, 0, *v2, color="red",  linewidth=5.5, arrow_length_ratio=0.12, alpha=0.95, label="v2 (ℓ=2, offset)")
    ax1.quiver(+off, 0, 0, *v3, color="blue", linewidth=5.5, arrow_length_ratio=0.12, alpha=0.95, label="v3 (ℓ=3, offset)")
    t = np.linspace(-1.1, 1.1, 2)
    ax1.plot(t*d6[0], t*d6[1], t*d6[2], "g--", linewidth=3.0, alpha=0.8, label="D6 axis")

    arc = _slerp_arc(v2, -v3, npts=120)
    ax1.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="orange", linewidth=2.2, alpha=0.95)
    mid = arc[len(arc)//2]
    ax1.text(mid[0], mid[1], mid[2], f"{ang_deg:.3f}°", color="orange", fontsize=11, weight="bold")

    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.view_init(elev=12, azim=-70)
    ax1.set_title(f"(A) 3D View\nA23={A23:.6f}, angle={ang_deg:.3f}°", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # ---------- (B) Mollweide ----------
    lon2, lat2 = _vec_to_lonlat(v2)
    lon3, lat3 = _vec_to_lonlat(v3)
    lonp, latp = _vec_to_lonlat(d6)
    lonn, latn = _vec_to_lonlat(-d6)

    # pole-aware jitter to avoid overlap (particularly at the south pole)
    def jitter(lon, lat, order_idx, pole="south"):
        if pole == "south" and lat < -np.radians(80):
            lon += np.radians((-1, 0, +1, +2)[order_idx % 4] * 4.0)  # spread by 4°
            lat = -np.radians(80)  # keep inside frame
        if pole == "north" and lat > np.radians(80):
            lon += np.radians((-1, 0, +1, +2)[order_idx % 4] * 4.0)
            lat = np.radians(80)
        return lon, lat

    lon2, lat2 = jitter(lon2, lat2, 0, "north" if lat2 > 0 else "south")
    lon3, lat3 = jitter(lon3, lat3, 1, "north" if lat3 > 0 else "south")
    lonp, latp = jitter(lonp, latp, 2, "north" if latp > 0 else "south")
    lonn, latn = jitter(lonn, latn, 3, "north" if latn > 0 else "south")

    ms = 160
    ax2.scatter([lon2], [lat2], c="red",  s=ms, marker="^", edgecolor="black", linewidth=0.8, zorder=4)
    ax2.scatter([lon3], [lat3], c="blue", s=ms, marker="v", edgecolor="black", linewidth=0.8, zorder=4)
    ax2.scatter([lonp], [latp], c="green", s=ms*1.05, marker="*", edgecolor="black", linewidth=0.8, zorder=5)
    ax2.scatter([lonn], [latn], c="green", s=ms*0.9, marker="x", linewidths=2.0, zorder=5)

    # great-circle arc (v2 to -v3) on sky
    arc_sky = _slerp_arc(v2, -v3, npts=180)
    lons = np.arctan2(arc_sky[:, 1], arc_sky[:, 0])
    lats = np.arcsin(np.clip(arc_sky[:, 2], -1, 1))
    ax2.plot(lons, lats, color="orange", lw=1.8, alpha=0.9, zorder=3)

    # inline labels (white background to avoid grid bleed)
    def label(ax, lon, lat, txt, dy=18):
        ax.annotate(txt, (lon, lat), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, zorder=6, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.9))
    label(ax2, lon2, lat2, "v2")
    label(ax2, lon3, lat3, "v3", dy=-26)
    label(ax2, lonp, latp, "D6 +z", dy=20)
    label(ax2, lonn, latn, "D6 −z", dy=-30)

    ax2.grid(True, alpha=0.35, linewidth=0.9)
    ax2.set_title("(B) Sky Positions (Mollweide)", fontsize=11, fontweight="bold")

    # ---------- (C) Summary ----------
    ax3.axis("off")
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1])

    p_iso = meta["p_iso"]; p_d6 = meta["p_d6"]
    p_iso_str = f"{p_iso:.3g}" if isinstance(p_iso, (float, int)) else "n/a"
    p_d6_str  = f"{p_d6:.3g}" if isinstance(p_d6,  (float, int)) else "n/a"
    ph2_obs = meta.get("ph2_obs"); ph2_pred = meta.get("ph2_pred")
    hemi_A = meta.get("A_hemi"); hemi_p = meta.get("p_hemi")
    ph2_line = f"Fractional H0: {ph2_obs:.3g}% / {ph2_pred:.3g}%" if (ph2_obs is not None and ph2_pred is not None) else "Fractional H0: n/a"
    hemi_line = f"Hemisph. Asym. (D6): A≈{hemi_A:.4g}; p≈{hemi_p:.3g}" if (hemi_A is not None and hemi_p is not None) else "Hemisph. Asym.: n/a"

    txt = (
        f"Run: {meta['run']}\n"
        f"A23 = {A23:.6f}   angle = {ang_deg:.3f}°\n"
        f"Euler(zyz) = [{euler[0]:.3f}, {euler[1]:.3e}, {euler[2]:.3f}]\n"
        f"p_iso = {p_iso_str}   p_d6 = {p_d6_str}\n"
        f"{ph2_line}\n{hemi_line}\n\n"
        "Geometric Impedance (CT):  Z(n^) = Z0 + ε Σ_A w_A (n^·A)^2\n"
        "Low-ℓ multipoles integrate lens structure → D6 imprint."
    )
    ax3.text(0.02, 0.98, txt, fontsize=10, family="monospace", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.28,
                       edgecolor="black", linewidth=0.8))

    # layout + save
    fig.suptitle("Axis of Evil — Corrected Visualization (anti-parallel v2/v3; true D6)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.0)
    out = Path("results/summary_reports/axis_of_evil_CORRECTED.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    create_corrected_axis_figure()
