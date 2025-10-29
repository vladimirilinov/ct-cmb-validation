# P-H2 Gradient Scaling (Heuristic)

Goal: Provide a quantitative link between the energetic separation ρ_D/ρ_R and the fractional H₀ gradient ΔH₀/H₀.

Assumption: In the CT framework, calibration gradients scale gently with the lens separation, leading to a logarithmic sensitivity in the observational channels:

- fractional ≈ k · log10(ρ_D/ρ_R)

Calibration: Using ρ_D/ρ_R ≈ 1.7×10⁶ (Part VII inputs) and observed fractional ≈ 0.083 (Planck vs local H₀), we set

- k ≈ 0.083 / log10(1.7e6) ≈ 0.0133

This yields predicted_fractional_from_density_ratio ≈ 0.0133·log10(ρ_D/ρ_R).

Notes:
- This is a minimal, monotone scaling consistent with the impedance-derived perspective where larger separations induce stronger calibration gradients without over-amplifying uncertainties.
- It provides a consistency check only; not a fundamental derivation. Replace with the exact relation once available from the full Part VII kernel.
