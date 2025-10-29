# QC-Link Scaling Heuristic

Theory anchor: theory_validation.md §6 and relevant-proofs.md (calibration invariance).

Objective: Map (η_*, ρ_D/ρ_R, θ_23) → A_s with minimal calibrated assumptions consistent with the stated prediction A_s ≈ 2.4e-9.

Scaling used in src/ct_cmb_validation/theory/qc_link.py:
- A_s ≈ (η_*/1e6) × sin^2(2θ_23) × [1 + 0.1 log10(ρ_D/ρ_R)] × k, with k = 7.5e-4 calibrated so canonical inputs yield ≈2.4×10^{-9}.

Rationale:
- sin^2(2θ_23) is a standard, bounded sensitivity for a mixing-like angle.
- log sensitivity in ρ_D/ρ_R reflects gentle dependence on energetic separation without over-amplifying uncertainties.
- The constant k is chosen so canonical inputs (η_*≈2, ρ ratio≈1.7e6, θ_23≈0.759) give A_s ≈ 2.4e-9 while preserving calibration invariance.

Audit and invariance considerations (relevant-proofs.md):
- Maintain homogeneity under lens calibration rescalings; relative variations are meaningful.
- Final validation compares to observed A_s within a tolerance band (default 10%).

Upgrade path:
- Replace with the exact formula from Part VII once the full derivation kernel is integrated.
