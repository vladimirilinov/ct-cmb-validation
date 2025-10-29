# Gaunt Coefficients — Safe Placeholder for Modulation

Theory anchor: theory_validation.md §3 (Observed a_{ℓm} via convolution with Z_{D6}).

Exact expression:
  a^{Obs}_{ℓm} = Σ_{ℓ' m', ℓ'' m''} a^{CMB}_{ℓ' m'} z_{ℓ'' m''} ∫ Y^*_{ℓ m} Y_{ℓ' m'} Y_{ℓ'' m''} dΩ

Production implementation should evaluate Gaunt integrals (or use healpy helpers) and apply the full harmonic-space convolution.

Placeholder used in src/ct_cmb_validation/theory/impedance.py:
- Scale by the monopole of Z: a^{Obs} ≈ (1 + z_{00}/√{4π}) a^{CMB}.

Justification:
- The monopole term provides the leading-order rescaling when Z has non-negligible average. Higher multipoles modulate anisotropy but preserve this baseline. This preserves energy accounting at first order and is numerically safe.

Upgrade path:
- Replace with explicit Gaunt-based convolution once dependency constraints allow, verifying selection rules (even ℓ; m multiple of 6) per D6 symmetry.

