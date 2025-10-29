# Quadrupole Tensor Approximation from a_{2m}

Purpose: Provide a robust fallback for preferred-direction extraction for ℓ=2 in the absence of healpy, consistent with theory_validation.md §3.

Mapping used in src/ct_cmb_validation/analysis/multipole_analysis.py:

- Build a real, symmetric, traceless 3×3 tensor Q from complex a_{2m}. We adopt a simple real-basis mapping preserving tracelessness and parity:
  - Q_zz = Re[a_{20}]
  - Q_xx = -0.5 Re[a_{20}] + Re[a_{22}]
  - Q_yy = -0.5 Re[a_{20}] - Re[a_{22}]
  - Q_xy = Re[i a_{22}] = Im[a_{22}]
  - Q_xz = Re[a_{21}]
  - Q_yz = Im[a_{21}]

These coefficients are normalized up to an overall scale and basis rotation; eigen-directions are invariant under such rescalings. The preferred axis is taken as the eigenvector corresponding to the largest |eigenvalue|.

Rationale:
- The exact Cartesian–spherical mapping involves Wigner 3j symbols and precise normalization constants; however, eigenvectors are unchanged by uniform rescaling of Q’s entries.
- This preserves tracelessness (Tr Q = 0) and captures the correct transformation behavior under rotations.

Connections to Coherence Theory:
- The direction extracted here is only used for the alignment statistic A_23 = |v2·v3| per theory_validation.md §3.
- This is compatible with the Geometric Impedance prediction where low-ℓ morphology aligns with D6 axes.

Audit Note:
- For production-grade accuracy, prefer healpy-based map reconstruction and direction extraction (already supported when healpy is installed).

