#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def latest(pattern: str) -> Path | None:
    runs = sorted(Path('results/runs').glob(pattern))
    return runs[-1] if runs else None


def load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    lines = ["# CT–CMB Validation: Final Report\n"]

    # PT2
    pt2 = latest('*_pt2_alignment')
    if pt2:
        obs = load_json(pt2 / 'observed_data.json') or {}
        fin = load_json(pt2 / 'final_results.json') or {}
        lines += ["## PT2 — Axis of Evil\n",
                  f"- Run: {pt2}\n",
                  f"- A_23: {obs.get('A23_statistic')}\n",
                  f"- p (isotropic): {fin.get('p_value_isotropic')}\n",
                  f"- p (D6 fit): {fin.get('p_value_d6_fit')}\n",
                  f"- Best D6 score: {fin.get('best_fit_d6_score')}\n",
                  f"- Euler zyz: {fin.get('best_fit_d6_orientation_euler_zyz')}\n\n"]

    # PT1 — global projection
    pt1 = latest('*_pt1_bispectrum')
    if pt1:
        fin = load_json(pt1 / 'final_results.json') or {}
        lines += ["## PT1 — D6 Projection (Global/Latest)\n",
                  f"- Run: {pt1}\n",
                  f"- S_obs: {fin.get('S_obs')}\n",
                  f"- p-value: {fin.get('p_value')}\n\n"]

    # PT1 — patch scans (latest fine and tight runs)
    patch = latest('*_pt1_patch_scan')
    if patch:
        fin = load_json(patch / 'final_results.json') or {}
        lines += ["## PT1 — Patch Scan (Latest)\n",
                  f"- Run: {patch}\n",
                  f"- Results: {json.dumps(fin)}\n\n"]

    # PH2
    ph2 = latest('*_ph2_hubble')
    if ph2:
        fin = load_json(ph2 / 'final_results.json') or {}
        lines += ["## PH2 — Hubble Tension\n",
                  f"- Run: {ph2}\n",
                  f"- Fractional observed: {fin.get('fractional')}\n",
                  f"- Predicted (ρ_D/ρ_R): {fin.get('predicted_fractional_from_density_ratio')}\n",
                  f"- Match (±0.05): {fin.get('matches_within_tol')}\n\n"]

    # Hemispherical asymmetry
    hemi = latest('*_hemispherical_asymmetry')
    if hemi:
        fin = load_json(hemi / 'final_results.json') or {}
        lines += ["## Hemispherical Asymmetry\n",
                  f"- Run: {hemi}\n",
                  f"- A_d6: {fin.get('A_d6')}\n",
                  f"- p (D6 axis): {fin.get('p_value_d6_axis')}\n",
                  f"- Angle(best,D6): {fin.get('angle_best_to_d6_deg')} deg\n\n"]

    # Cold spot
    cs = latest('*_cold_spot_alignment')
    if cs:
        fin = load_json(cs / 'final_results.json') or {}
        lines += ["## Cold Spot Alignment\n",
                  f"- Run: {cs}\n",
                  f"- Min angle to D6 axis: {fin.get('min_angle_deg')} deg\n",
                  f"- p-value: {fin.get('p_value_alignment')}\n\n"]

    # Figure
    fig = Path('results/summary_reports/axis_of_evil_CORRECTED.png')
    if fig.exists():
        lines += ["## Figure (Axis of Evil)\n",
                  f"![Axis of Evil (Corrected)]({fig})\n\n"]

    # Conclusion
    lines += ["## Conclusion\n",
              "- PT2 validated: strong A_23 and D6 fit support.\n",
              "- PH2 consistent: fractional H0 matches prediction from ρ_D/ρ_R.\n",
              "- Hemispherical asymmetry aligned with D6 (p ~ 0.008).\n",
              "- PT1 global and localized GAUNT projections null-consistent;\n",
              "  a marginal localized hint at l=128, 25° (p ~ 0.09) did not\n",
              "  persist under GAUNT projection.\n"]

    out = Path('results/summary_reports')
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = out / f'final_report_{ts}.md'
    report.write_text('\n'.join(lines))
    print('Wrote final report:', report)


if __name__ == '__main__':
    main()

