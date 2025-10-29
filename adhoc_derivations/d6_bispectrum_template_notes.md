# D6 Bispectrum Template — Averaging Proxy

Theory anchor: theory_validation.md §4.

Ideal construction:
  T_{D6} = (1/|D6|) Σ_{R∈D6} (D^{ℓ1}(R) ⊗ D^{ℓ2}(R) ⊗ D^{ℓ3}(R)) · T_{basis}

Proxy used here:
- Average the basis array itself across group elements without explicit Wigner action (equivalent to identity action). This preserves auditability and provides a deterministic baseline but underestimates symmetry-induced mode-mixing.

Why acceptable as a placeholder:
- The projection statistic S = (B·T)/√(T·C·T) remains well-defined, enabling end-to-end run audit while flagging the need for upgraded math kernels.

Upgrade path:
- Replace with explicit Wigner-D action per multipole triplets using a robust library.

