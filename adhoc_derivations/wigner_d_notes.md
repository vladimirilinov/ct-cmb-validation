# Wigner D-Matrix Notes

Theory anchor: theory_validation.md §4 (D6 invariance via Wigner D action).

- For ℓ=0, D^0(g) = [1].
- For ℓ=1, D^1(g) matches the 3D rotation matrix up to spherical–Cartesian basis convention.
- For ℓ≥2, stable numerical evaluation requires dedicated libraries (e.g., wigxjpf, ssht). SciPy does not provide full Wigner D coverage.

Current implementation:
- src/ct_cmb_validation/theory/d6_group.py implements ℓ=0 and ℓ=1 cases; raises NotImplementedError otherwise.

Audit note:
- For PT-1 group-averaging at high ℓ, integrate a robust Wigner-D library to ensure correctness and stability.

