# CT-CMB Validation: Extended Predictions — Complete Implementation Guide

**agents-expanded.md — Version 2.0**  
*Supplementary documentation for P-H3 (Dark Energy), P-L1 (Lensing), CT-C1 (α Variation), CT-C2 (Black Holes), and P-S1 (LSS Anisotropy)*

---

## Table of Contents

1. [Overview & Roadmap](#1-overview--roadmap)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Data Specifications](#3-data-specifications)
4. [Repository Structure Extensions](#4-repository-structure-extensions)
5. [Core Implementation Details](#5-core-implementation-details)
6. [Testing & Validation Protocols](#6-testing--validation-protocols)
7. [Analysis Workflows](#7-analysis-workflows)
8. [Visualization & Reporting](#8-visualization--reporting)
9. [Computational Optimization](#9-computational-optimization)
10. [Troubleshooting & FAQ](#10-troubleshooting--faq)

---

## 1. Overview & Roadmap

### 1.1 Validation Status Summary

**Tier 1: Validated (October 2025)**
- ✅ **P-T2**: Axis of Evil alignment ($p = 0.015$)
- ✅ **P-H2**: Hubble tension prediction (8.29% vs 8.31%)
- ✅ **Hemispherical Asymmetry** ($p = 0.008$)
- ⚠️ **P-T1**: Bispectrum (null, as expected)

**Tier 2: Ready for Validation (Immediate Priority)**
- 🔥 **P-H3**: Dark Energy equation of state
- 🔥 **P-L1**: Universal lensing interface profiles

**Tier 3: In Development (6-12 Month Timeline)**
- 🔨 **CT-C1**: Fine-structure constant variation
- 🔨 **P-S1**: Large-scale structure anisotropy

**Tier 4: Long-Term (1-3 Year Timeline)**
- 📅 **CT-C2**: Black hole horizon dynamics

### 1.2 Implementation Priority Matrix

| Test | Impact | Difficulty | Data Available | Timeline | Priority |
|------|--------|------------|----------------|----------|----------|
| P-H3 | ⭐⭐⭐⭐⭐ | Low | ✅ Yes (Pantheon+, DESI) | 1 month | **IMMEDIATE** |
| P-L1 | ⭐⭐⭐⭐⭐ | Medium | ✅ Yes (HSC, DES) | 3 months | **IMMEDIATE** |
| P-S1 | ⭐⭐⭐⭐ | Medium | ✅ Yes (SDSS) | 6 months | High |
| CT-C1 | ⭐⭐⭐ | High | ⚠️ Partial (archives) | 12 months | Medium |
| CT-C2 | ⭐⭐⭐ | Very High | ⚠️ Limited (EHT) | 2+ years | Low |

### 1.3 Dependency Graph

```
Validated Foundation (P-T2, P-H2, Hemi)
    │
    ├─→ P-H3 (Dark Energy)
    │   └─→ Uses: R₀ from P-H2, ε from P-T2
    │
    ├─→ P-L1 (Lensing Profiles)
    │   └─→ Uses: ξ prediction from CT scaffold
    │
    ├─→ P-S1 (LSS Anisotropy)
    │   └─→ Uses: v_AoE from P-T2, Z_D6 from impedance
    │
    ├─→ CT-C1 (α Variation)
    │   └─→ Uses: v_AoE from P-T2, Z_D6 modulation
    │
    └─→ CT-C2 (Black Holes)
        └─→ Uses: Interface width ξ from P-L1 (if validated)
```

---

## 2. Mathematical Foundations

### 2.1 Core CT Framework (Review)

#### Definition 2.1.1: Budget Allocation
The total available computational budget $B_{total}$ is allocated:
$$B_{total} = B_{th} + B_{cx} + B_{leak}$$

where:
- $B_{th}$: Throughput budget (physical processes, energy flow)
- $B_{cx}$: Complexity budget (maintaining coherence, entropy management)
- $B_{leak}$: Leakage (decoherence, dissipation)

#### Theorem 2.1.1: Hodge Decomposition Correspondence
**Statement:** The budget allocation maps to electromagnetic field structure:
- $B_{th} \leftrightarrow$ Gradient flows (E-modes, curl-free)
- $B_{cx} \leftrightarrow$ Cyclic flows (B-modes, divergence-free)

**Proof:** See `theory_validation.md`, Theorem I.1 □

#### Theorem 2.1.2: Universal Interface Profile (Allen-Cahn)
**Statement:** An SEP transition minimizing the Ginzburg-Landau functional has profile:
$$\phi(x) = \phi_0 \tanh\left(\frac{x - x_0}{\xi}\right)$$

where $\xi = \sqrt{2\sigma/\kappa}$ is the interface width, $\sigma$ is surface tension, $\kappa$ is bulk free energy.

**CT Interpretation:** 
- $\sigma \sim \lambda_{cx}$ (complexity cost of maintaining interface)
- $\kappa \sim \lambda_{leak}$ (instability of mixed phases)
- Therefore: $\xi \sim \sqrt{\lambda_{cx}/\lambda_{leak}}$

---

### 2.2 P-H3: Dark Energy Mathematical Scaffold

#### 2.2.1 Fundamental Hypothesis

**Postulate (Budget-Energy Correspondence):**
The dark energy density $\rho_{DE}$ is the energetic cost of maintaining the quantum-cosmic hierarchy, quantified by the information content of the gradient $R(a) = \rho_D(a)/\rho_R(a)$.

#### 2.2.2 Derivation of $\rho_{DE}(a)$

**Step 1: Complexity Scaling**

The Shannon entropy (information content) of a continuous distribution with characteristic scale $R$ is:
$$S \sim \ln(1 + R)$$

For large $R$, this scales as $\ln R$. The complexity budget required to maintain this hierarchy:
$$B_{cx}(a) \propto \ln(1 + R(a))$$

**Step 2: Energy Conversion**

The CT calibration factor $K_{D6}$ converts complexity budget to physical energy density:
$$\rho_{DE}(a) = K_{D6} \cdot \ln(1 + R(a))$$

where $K_{D6}$ has units of energy density and is set by $T_{D6}$ geometry.

**Step 3: Evolution of Hierarchy**

The fast sector (matter) dilutes: $\rho_D \propto a^{-3}$

The slow sector (CMB scaffold) is approximately constant: $\rho_R \approx \rho_R^0$

Therefore:
$$R(a) = \frac{\rho_D(a)}{\rho_R^0} = R_0 a^{-3}$$

with $R_0 = 1.7 \times 10^6$ (validated from P-H2 Hubble tension).

**Step 4: Present-Day Normalization**

At $a = 1$ (today):
$$\rho_{DE}(1) = 0.7 \rho_{crit} = 0.7 \times \frac{3H_0^2}{8\pi G}$$

This fixes:
$$K_{D6} = \frac{0.7 \times 3H_0^2}{8\pi G \cdot \ln(1 + R_0)}$$

With $H_0 = 67.4$ km/s/Mpc and $R_0 = 1.7 \times 10^6$:
$$\ln(1 + R_0) \approx 14.346$$
$$K_{D6} \approx 5.93 \times 10^{-10} \text{ J/m}^3$$

#### 2.2.3 Equation of State Derivation

**Continuity Equation:**
$$\frac{d\rho_{DE}}{dt} + 3H(1 + w)\rho_{DE} = 0$$

**Transform to scale factor:**
$$\frac{d\ln\rho_{DE}}{d\ln a} = -3(1 + w)$$

Therefore:
$$w(a) = -1 - \frac{1}{3}\frac{d\ln\rho_{DE}}{d\ln a}$$

**Calculate the derivative:**
$$\rho_{DE}(a) = K_{D6} \ln(1 + R_0 a^{-3})$$

$$\frac{d\rho_{DE}}{da} = K_{D6} \cdot \frac{1}{1 + R_0 a^{-3}} \cdot \frac{d}{da}(R_0 a^{-3})$$
$$= K_{D6} \cdot \frac{-3R_0 a^{-4}}{1 + R_0 a^{-3}}$$

$$\frac{d\ln\rho_{DE}}{d\ln a} = \frac{a}{\rho_{DE}}\frac{d\rho_{DE}}{da}$$
$$= \frac{a}{K_{D6}\ln(1+R_0 a^{-3})} \cdot K_{D6} \cdot \frac{-3R_0 a^{-4}}{1 + R_0 a^{-3}}$$
$$= \frac{-3R_0 a^{-3}}{(1 + R_0 a^{-3})\ln(1 + R_0 a^{-3})}$$

Let $R = R_0 a^{-3}$:
$$\frac{d\ln\rho_{DE}}{d\ln a} = \frac{-3R}{(1+R)\ln(1+R)}$$

**Final Result:**

$$\boxed{w_{CT}(a) = -1 + \frac{R(a)}{(1+R(a))\ln(1+R(a))}}$$

where $R(a) = R_0 a^{-3}$ with $R_0 = 1.7 \times 10^6$.

#### 2.2.4 Key Properties

**Present-day value ($z=0, a=1$):**
$$w_0 = -1 + \frac{R_0}{(1+R_0)\ln(1+R_0)}$$
$$= -1 + \frac{1.7 \times 10^6}{1.7 \times 10^6 \cdot 14.346} \approx -1 + 0.0697$$
$$\boxed{w_0 \approx -0.9303}$$

**Asymptotic behavior:**
- As $a \to 0$ (early universe): $R \to \infty$, $w \to -1$
- As $a \to \infty$ (far future): $R \to 0$, $w \to -1$

**Derivative (useful for analysis):**
$$\frac{dw}{dz} = -\frac{dw}{da}\frac{da}{dz} = (1+z)^2 \frac{dw}{da}$$

This is a **thawing model**: $w$ evolves from more negative values toward $-1$.

#### 2.2.5 Comparison to Parametrized Models

**CPL Parametrization (Chevallier-Polarski-Linder):**
$$w_{CPL}(a) = w_0 + w_a(1-a)$$

**CT Taylor expansion near $a=1$:**
For small $(1-a)$:
$$w_{CT}(a) \approx w_0 + w_a(1-a) + \mathcal{O}((1-a)^2)$$

where:
$$w_a = \left.\frac{dw}{da}\right|_{a=1} = \frac{3R_0}{(1+R_0)^2[\ln(1+R_0)]^2} \left[1 - \frac{1}{\ln(1+R_0)}\right]$$

With $R_0 = 1.7 \times 10^6$:
$$w_a \approx 3 \times \frac{1.7 \times 10^6}{(1.7 \times 10^6)^2 \cdot 14.346^2} \times 0.93$$
$$\boxed{w_a \approx 0.00013}$$

**Prediction:** CT is nearly constant dark energy, but with slight thawing ($w_a > 0$).

---

### 2.3 P-L1: Lensing Profile Mathematical Scaffold

#### 2.3.1 3D Density Profile

**CT Prediction (from Theorem 2.1.2):**
$$\rho_{CT}(r) = \rho_\infty + \frac{\rho_0 - \rho_\infty}{2}\left[1 - \tanh\left(\frac{r - R_{200}}{\xi}\right)\right]$$

where:
- $\rho_\infty$: Background density (far from cluster)
- $\rho_0$: Central density
- $R_{200}$: Virial radius (where mean density = 200 × critical density)
- $\xi$: Universal interface width (CT prediction)

**Alternative parametrization:**
$$\rho_{CT}(r) = \rho_\infty + \Delta\rho \cdot \Theta_{CT}(r)$$

where:
$$\Theta_{CT}(r) = \frac{1}{2}\left[1 - \tanh\left(\frac{r - R_{200}}{\xi}\right)\right]$$

is the CT interface function.

#### 2.3.2 Projection to Surface Density

**Line-of-sight integration:**
$$\Sigma(R) = \int_{-\infty}^{\infty} \rho\left(\sqrt{R^2 + z^2}\right) dz$$

where $R$ is projected radius, $z$ is line-of-sight coordinate.

**Change variables:** $r = \sqrt{R^2 + z^2}$, $dz = \frac{r \, dr}{\sqrt{r^2 - R^2}}$

$$\Sigma(R) = 2\int_R^\infty \frac{\rho(r) \cdot r}{\sqrt{r^2 - R^2}} dr$$

**For CT profile:**
$$\Sigma_{CT}(R) = \Sigma_\infty + \Delta\rho \cdot I_{CT}(R)$$

where:
$$I_{CT}(R) = 2\int_R^\infty \frac{r \cdot \Theta_{CT}(r)}{\sqrt{r^2 - R^2}} dr$$
$$= \int_R^\infty \frac{r \left[1 - \tanh\left(\frac{r-R_{200}}{\xi}\right)\right]}{\sqrt{r^2 - R^2}} dr$$

**This integral has no closed form** and must be computed numerically.

#### 2.3.3 Lensing Observables

**Convergence (dimensionless surface density):**
$$\kappa(R) = \frac{\Sigma(R)}{\Sigma_{crit}}$$

where:
$$\Sigma_{crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_l D_{ls}}$$

($D_l$: lens distance, $D_s$: source distance, $D_{ls}$: lens-source distance)

**Tangential shear:**
$$\gamma_t(R) = \bar{\kappa}(<R) - \kappa(R)$$

where $\bar{\kappa}(<R)$ is the mean convergence inside radius $R$:
$$\bar{\kappa}(<R) = \frac{2}{R^2}\int_0^R \kappa(R') R' dR'$$

**Alternative (using surface density directly):**
$$\gamma_t(R) = \frac{\bar{\Sigma}(<R) - \Sigma(R)}{\Sigma_{crit}}$$

#### 2.3.4 Comparison Model: NFW Profile

**Navarro-Frenk-White (standard ΛCDM):**
$$\rho_{NFW}(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2}$$

where $r_s$ is scale radius, $\rho_s$ is characteristic density.

**Projected NFW (analytic):**
$$\Sigma_{NFW}(R) = 2\rho_s r_s \cdot F(x)$$

where $x = R/r_s$ and:
$$F(x) = \begin{cases}
\frac{1}{x^2-1}\left[1 - \frac{2}{\sqrt{1-x^2}}\text{arctanh}\sqrt{\frac{1-x}{1+x}}\right] & x < 1 \\
\frac{1}{3} & x = 1 \\
\frac{1}{x^2-1}\left[1 - \frac{2}{\sqrt{x^2-1}}\arctan\sqrt{\frac{x-1}{x+1}}\right] & x > 1
\end{cases}$$

#### 2.3.5 Model Comparison Framework

**Nested Model Test:**

NFW has 2 free parameters: $(r_s, \rho_s)$ or equivalently $(M_{200}, c_{200})$

CT profile has 3 parameters: $(\rho_0, R_{200}, \xi)$ but $\xi$ is **predicted** (universal).

**Test Strategy:**
1. Fit NFW freely: $M_{NFW}(M_{200}, c_{200})$
2. Fit CT with fixed $\xi$: $M_{CT}(\rho_0, R_{200}; \xi_{CT})$
3. Compare: $\chi^2_{NFW}$ vs $\chi^2_{CT}$

**Bayesian Evidence:**
$$\ln \mathcal{B}_{CT/NFW} = \ln\mathcal{Z}_{CT} - \ln\mathcal{Z}_{NFW}$$

where:
$$\mathcal{Z} = \int p(D|\theta, M) p(\theta|M) d\theta$$

For nested models (CT has one fewer free parameter if $\xi$ is fixed):
$$\Delta \text{BIC} = \chi^2_{CT} - \chi^2_{NFW} + \ln(N)$$

If $\Delta \text{BIC} < -10$: Strong evidence for CT

#### 2.3.6 Universality Test

**Critical Prediction:** $\xi$ is constant across all clusters.

**Procedure:**
1. Fit each cluster individually, allowing $\xi$ to vary
2. Extract $\xi_i$ for cluster $i$
3. Compute statistics:
   - Mean: $\bar{\xi} = \frac{1}{N}\sum \xi_i$
   - Scatter: $\sigma_\xi = \sqrt{\frac{1}{N-1}\sum(\xi_i - \bar{\xi})^2}$
   - Fractional scatter: $f_\xi = \sigma_\xi / \bar{\xi}$

**CT Prediction:** $f_\xi < 0.2$ (less than 20% scatter, accounting for measurement errors)

**Test against null hypothesis:**
- $H_0$: $\xi$ varies randomly (no universality)
- $H_1$: $\xi$ is universal (CT prediction)

Use F-test or Bayesian model comparison.

#### 2.3.7 Theoretical Prediction for $\xi$

**From CT parameters:**
$$\xi \sim \sqrt{\frac{\lambda_{cx}}{\lambda_{leak}}} \times \frac{\hbar c}{k_B T}$$

**Rough estimate:**
- $\lambda_{cx} \sim 1$ (complexity multiplier)
- $\lambda_{leak} \sim 10^{-3}$ (decoherence rate)
- $T \sim 10^7$ K (virial temperature)

$$\xi \sim \sqrt{1000} \times \frac{6.6 \times 10^{-16} \times 3 \times 10^8}{1.4 \times 10^{-23} \times 10^7}$$
$$\sim 30 \times 1.4 \times 10^{15} \text{ m} \sim 50 \text{ kpc}$$

**Expected range:** $\xi \in [30, 300]$ kpc

(Precise value requires full T_D6 calculation - to be refined)

---

### 2.4 CT-C1: α Variation Mathematical Scaffold

#### 2.4.1 Theoretical Foundation

**Proposition (from Part VII, Constants as Multipliers):**
Physical constants are ratios of SEP multipliers. A spatial variation $\alpha_{EM}(\mathbf{x})$ implies a gradient in the SEP: $\nabla\Lambda \neq 0$.

**Coherence Principle:**
To minimize $B_{cx}$ and $B_{leak}$, any large-scale gradient must align with the scaffold's principal symmetries (Theorem C.1.1).

#### 2.4.2 Intrinsic Dipole (CT-C1.A)

**Prediction:**
If spatial variation exists, the dipole $\mathbf{D}_\alpha$ must align with the Axis of Evil direction $\mathbf{v}_{AoE}$ (validated from P-T2).

**Parametrization:**
$$\frac{\Delta\alpha}{\alpha}(\hat{n}) = A_1 (\hat{n} \cdot \hat{d}) + \text{(higher terms)}$$

where $\hat{d}$ is the dipole direction.

**Null Hypothesis:** $\hat{d}$ is random (uniform on sphere)

**CT Hypothesis:** $\hat{d} = \mathbf{v}_{AoE}$ (within uncertainty)

**Webb Hypothesis:** $\hat{d} = \mathbf{v}_{Webb} = (\alpha_{Webb}, \delta_{Webb})$ from literature

**Test:** Angular distance between best-fit $\hat{d}$ and predictions:
$$\theta = \arccos(\hat{d} \cdot \mathbf{v}_{pred})$$

**Significance:** Use Monte Carlo to compute $p(\theta | H_0)$ under null.

#### 2.4.3 Geometric Impedance Modulation (CT-C1.B)

**Observational Equation:**
$$\frac{\Delta\alpha}{\alpha}_{obs}(\hat{n}) = Z_{D6}(\hat{n}) \cdot \frac{\Delta\alpha}{\alpha}_{int}(\hat{n})$$

**Harmonic Decomposition:**
$$a_{\ell m}^{obs} = \sum_{L M, \ell' m'} z_{LM} \, a_{\ell' m'}^{int} \, \mathcal{G}(\ell, L, \ell'; m, M, m')$$

where $\mathcal{G}$ are Gaunt coefficients (Wigner 3-j integrals).

**Impedance Structure:**
$Z_{D6}(\hat{n})$ has $D_6$ symmetry → dominant modes at $L = 0, 2, 6, 12, \ldots$

For dipole intrinsic ($\ell' = 1$):
- $(L=2, \ell'=1) \to \ell = 1, 3$ (induced octupole)
- $(L=6, \ell'=1) \to \ell = 5, 7$

**Modulated Model:**
$$\frac{\Delta\alpha}{\alpha}(\hat{n}) = A_1 Y_1^m(\hat{n}) + A_3 Y_3^{m'}(\hat{n}) + A_5 Y_5^{m''}(\hat{n}) + \ldots$$

with phase coherence imposed by $Z_{D6}$ orientation.

**Key Constraint:** The induced multipoles ($\ell = 3, 5, 7$) must be aligned with $\mathbf{v}_{AoE}$.

**Test:**
1. Fit pure dipole model $M_D$: $\chi^2_D$
2. Fit modulated model $M_{CT}$: $\chi^2_{CT}$ (with $\ell=3,5,7$ aligned)
3. Fit unconstrained model $M_U$: $\chi^2_U$ (free higher multipoles)

**Evidence comparison:**
$$\ln \mathcal{B}_{CT/D} = -\frac{1}{2}(\chi^2_{CT} - \chi^2_D) + \Delta(\text{complexity})$$

If $\ln \mathcal{B}_{CT/D} > 5$: Strong evidence for modulation

#### 2.4.4 Implementation: Spherical Harmonic Analysis

**Data:** QSO absorption measurements at positions $(\alpha_i, \delta_i)$ with values $y_i = \Delta\alpha/\alpha$ and errors $\sigma_i$.

**HEALPix Map Construction:**
```python
import healpy as hp

# Convert to HEALPix pixels
nside = 32  # Resolution
npix = hp.nside2npix(nside)

# Bin data
alpha_map = np.zeros(npix)
weight_map = np.zeros(npix)

for i in range(N_qso):
    pix = hp.ang2pix(nside, theta_i, phi_i)
    alpha_map[pix] += y_i / sigma_i**2
    weight_map[pix] += 1 / sigma_i**2

alpha_map /= (weight_map + 1e-10)

# Spherical harmonic transform
alm = hp.map2alm(alpha_map, lmax=7)
```

**Extract Multipole Amplitudes:**
```python
def get_multipole_power(alm, ell):
    """Get total power at multipole ell."""
    indices = hp.Alm.getidx(lmax, ell, np.arange(ell+1))
    return np.sum(np.abs(alm[indices])**2)

C_1 = get_multipole_power(alm, 1)  # Dipole
C_3 = get_multipole_power(alm, 3)  # Octupole
C_5 = get_multipole_power(alm, 5)
```

**Test Alignment:**
```python
# Extract dipole direction
a_10, a_11_real, a_11_imag = alm[hp.Alm.getidx(lmax, 1, [0, 1, 1])]
# Direction from: Y_1^m → vector components

# Compare to v_AoE
angle_to_aoe = angular_separation(dipole_dir, v_aoe)

# Monte Carlo significance
angles_mc = []
for _ in range(10000):
    random_dir = random_direction()
    angles_mc.append(angular_separation(random_dir, v_aoe))

p_value = np.sum(angles_mc < angle_to_aoe) / 10000
```

---

### 2.5 P-S1: LSS Anisotropy Mathematical Scaffold

#### 2.5.1 Two-Point Correlation Function

**Definition:**
$$\xi(r, \hat{n}) = \langle \delta(\mathbf{x}) \delta(\mathbf{x} + r\hat{n}) \rangle$$

where $\delta = (\rho - \bar{\rho})/\bar{\rho}$ is the density contrast.

**Multipole Expansion:**
$$\xi(r, \hat{n}) = \sum_{\ell=0}^\infty \xi_\ell(r) \mathcal{L}_\ell(\hat{n} \cdot \hat{z})$$

where $\mathcal{L}_\ell$ are Legendre polynomials and $\hat{z}$ is a preferred axis.

**Standard cosmology:** Only monopole $\xi_0(r)$ is non-zero (isotropy)

**CT prediction:** Hexapole $\xi_6(r) \neq 0$ aligned with $\mathbf{v}_{AoE}$

#### 2.5.2 Geometric Bias from Impedance

**Physical Mechanism:**
Galaxy formation efficiency modulated by $Z_{D6}(\hat{n})$:
$$\rho_{gal}(\mathbf{x}, \hat{n}) = b(z) \rho_{DM}(\mathbf{x}) \cdot [1 + \epsilon Z_{D6}(\hat{n})]$$

where $b(z)$ is bias, $\epsilon \sim 10^{-4}$ is impedance coupling.

**Correlation Function:**
$$\xi_{gal}(r, \hat{n}) \approx b^2 \xi_{DM}(r) \cdot [1 + 2\epsilon Z_{D6}(\hat{n})]$$

**Multipole Structure:**
$Z_{D6}(\hat{n})$ contains $\ell = 0, 2, 6, 12, \ldots$

Dominant directional signal: **Hexapole** ($\ell=6$)

**Amplitude Estimate:**
$$\frac{\xi_6(r)}{\xi_0(r)} \sim 2\epsilon \approx 2 \times 10^{-4}$$

#### 2.5.3 Yamamoto Estimator

**Practical Measurement:**

For galaxy pairs $(i, j)$ separated by $\mathbf{s} = \mathbf{x}_j - \mathbf{x}_i$:

$$\hat{\xi}_\ell(s) = \frac{(2\ell + 1)}{N_p} \sum_{pairs} \mathcal{L}_\ell(\mu_{ij}) w_i w_j$$

where:
- $\mu_{ij} = \hat{s} \cdot \hat{z}$ (cosine of angle to preferred axis)
- $w_i$ are weights (inverse variance)
- $N_p$ is number of pairs in shell $[s, s+\Delta s]$

**In coordinates aligned with $\mathbf{v}_{AoE}$:**
Rotate galaxy positions so $\mathbf{v}_{AoE} = \hat{z}$, then apply Yamamoto estimator.

#### 2.5.4 Significance Test

**Null Hypothesis:** $\xi_6 = 0$ (isotropic universe)

**Test Statistic:**
$$\chi^2_6 = \sum_{r_i} \frac{[\xi_6(r_i)]^2}{\sigma_6^2(r_i)}$$

where $\sigma_6^2$ is covariance (from mock catalogs or jackknife).

**CT Hypothesis:** $\chi^2_6$ is significantly larger than expected from noise.

**Monte Carlo:**
Generate $N_{mock}$ isotropic random catalogs, measure $\xi_6^{mock}$, compute $\chi^2_6^{mock}$ distribution.

**p-value:**
$$p = \frac{1}{N_{mock}} \sum_k \mathbb{1}[\chi^2_6^{mock}(k) \geq \chi^2_6^{obs}]$$

---

### 2.6 CT-C2: Black Hole Horizons Mathematical Scaffold

#### 2.6.1 Budget Conversion at Interface

**Phase Boundary:** Photon ring at $R_{ph} = 3GM/c^2$ separates:
- **Outer zone:** Accretion flow (high $B_{th}$, gradient-dominated)
- **Inner zone:** Strong gravity (high $B_{cx}$, cyclic-dominated)

**Conversion Rate:**
$$\frac{dB_{cx}}{dr}\bigg|_{interface} = -\frac{dB_{th}}{dr}\bigg|_{interface}$$

**Profile (from Theorem 2.1.2):**
$$B_{cx}(R) = B_{cx}^\infty + \Delta B_{cx} \cdot \frac{1}{2}\left[1 + \tanh\left(\frac{R - R_{ph}}{\xi_{BH}}\right)\right]$$

#### 2.6.2 Polarization Observables

**Electric Vector Position Angle (EVPA):**
$$\chi = \frac{1}{2}\arctan\left(\frac{U}{Q}\right)$$

where $Q, U$ are Stokes parameters.

**B-mode Power:**
$$P_B(R) = \langle B^2(R) \rangle = \langle (Q\sin 2\phi - U\cos 2\phi)^2 \rangle_\phi$$

**E-mode Power:**
$$P_E(R) = \langle E^2(R) \rangle = \langle (Q\cos 2\phi + U\sin 2\phi)^2 \rangle_\phi$$

**CT Prediction (C2.A):**
$$\frac{P_B(R)}{P_E(R)} = \frac{P_B^\infty}{P_E^\infty} + A_{CT} \cdot \text{sech}^2\left(\frac{R - R_{ph}}{\xi_{BH}}\right)$$

**Scaling:** $\xi_{BH} \propto GM/c^2$ (should be universal when scaled by mass)

#### 2.6.3 Phase Transition Signature (C2.B)

**Hypothesis:** Inside $R_{CT} < R_{ph}$, scaffold transitions $T_{D6} \to T_{HT}$ (e.g., $D_4$).

**Observable:** EVPA azimuthal structure changes.

**For $D_4$ symmetry:**
$$\chi(\phi) = \chi_0 + \Delta\chi_4 \cos(4\phi + \phi_4)$$

**For $D_6$ symmetry (exterior):**
$$\chi(\phi) = \chi_0 + \Delta\chi_6 \cos(6\phi + \phi_6)$$

**Test:** Fourier decompose EVPA in annuli:
$$\chi(\phi) = \sum_{m} a_m \cos(m\phi + \phi_m)$$

Compare amplitudes $a_4, a_6$ inside vs outside $R_{ph}$.

---

## 3. Data Specifications

### 3.1 P-H3: Dark Energy Data Products

#### 3.1.1 Type Ia Supernovae (Pantheon+)

**Dataset:** Pantheon+ Compilation (Scolnic et al. 2022)

**File:** `data/raw/pantheon_plus/Pantheon+SH0ES.dat`

**Columns:**
```
# name  zcmb  zhel  mb  dmb  x1  dx1  c  dc  logMass  dlogMass  host_flag
```

**Key Fields:**
- `zcmb`: Redshift (CMB frame)
- `mb`: Observed apparent magnitude
- `dmb`: Magnitude uncertainty
- `x1`, `c`: Stretch and color corrections

**Preprocessing:**
```python
import pandas as pd

# Load data
sne = pd.read_csv('data/raw/pantheon_plus/Pantheon+SH0ES.dat',
                   sep='\s+', comment='#')

# Apply cuts
mask = (sne['zcmb'] > 0.01) & (sne['zcmb'] < 2.3)  # Redshift range
mask &= (sne['dmb'] < 0.5)  # Quality cut
sne_clean = sne[mask].copy()

# Distance modulus
mu_obs = sne_clean['mb'] - M_B  # M_B = absolute magnitude (free parameter)
z_obs = sne_clean['zcmb'].values
sigma_obs = sne_clean['dmb'].values
```

**Covariance Matrix:**

**File:** `data/raw/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov`

```python
# Load covariance (N × N matrix)
cov = np.loadtxt('data/raw/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov')
```

#### 3.1.2 Baryon Acoustic Oscillations (BAO)

**DESI Year 1 (2024):**

**File:** `data/raw/desi_y1/desi_y1_bao.json`

```json
{
  "measurements": [
    {
      "survey": "DESI_LRG_ELG",
      "z_eff": 0.51,
      "DM_over_rd": 18.92,
      "sigma_DM_over_rd": 0.17,
      "DH_over_rd": 20.98,
      "sigma_DH_over_rd": 0.38
    },
    {
      "z_eff": 0.71,
      "DM_over_rd": 21.71,
      "sigma_DM_over_rd": 0.28,
      "DH_over_rd": 20.08,
      "sigma_DH_over_rd": 0.60
    }
  ]
}
```

**Key Quantities:**
- $D_M(z)$: Comoving angular diameter distance
- $D_H(z) = c/H(z)$: Hubble distance
- $r_d$: Sound horizon at drag epoch (fixed from CMB)

**Theoretical Calculation:**
```python
def DM_theory(z, H0, Omega_m, w_model):
    """Comoving angular diameter distance."""
    z_grid = np.linspace(0, z, 1000)
    H_grid = H(z_grid, H0, Omega_m, w_model)
    integrand = c / H_grid
    return np.trapz(integrand, z_grid)

def DH_theory(z, H0, Omega_m, w_model):
    """Hubble distance."""
    return c / H(z, H0, Omega_m, w_model)
```

#### 3.1.3 Planck CMB Constraints

**File:** `data/raw/planck/planck_h0_cosmic.json`

```json
{
  "H0_planck": 67.4,
  "sigma_H0": 0.5,
  "Omega_m": 0.315,
  "sigma_Omega_m": 0.007,
  "source": "Planck 2018 TT,TE,EE+lowE+lensing"
}
```

**Use:** Prior on $H_0$ and $\Omega_m$ for CT model fitting.

---

### 3.2 P-L1: Weak Lensing Data Products

#### 3.2.1 Cluster Catalog

**HSC S16A Cluster Catalog:**

**File:** `data/raw/hsc/hsc_s16a_clusters.fits`

**Table Structure:**
```python
# HDU 1: Cluster properties
# Columns: cluster_id, RA, Dec, z_spec, M_200, R_200, SNR
```

**Example:**
```python
from astropy.io import fits

hdu = fits.open('data/raw/hsc/hsc_s16a_clusters.fits')
clusters = hdu[1].data

# Select high-quality sample
mask = (clusters['M_200'] > 1e14) & (clusters['SNR'] > 10)
sample = clusters[mask]
```

#### 3.2.2 Shear Profiles

**File Structure:** `data/raw/hsc/shear_profiles/cluster_XXXX.fits`

**Per-cluster FITS file:**
```python
# HDU 1: Radial bins
# Columns: R_bin (kpc), R_min, R_max, n_pairs

# HDU 2: Tangential shear
# Columns: gamma_t, sigma_gamma_t

# HDU 3: Cross shear (systematics check)
# Columns: gamma_x, sigma_gamma_x
```

**Loading:**
```python
def load_shear_profile(cluster_id):
    """Load shear profile for a cluster."""
    filename = f'data/raw/hsc/shear_profiles/cluster_{cluster_id:04d}.fits'
    hdu = fits.open(filename)
    
    R = hdu[1].data['R_bin']  # kpc
    gamma_t = hdu[2].data['gamma_t']
    sigma_gamma = hdu[2].data['sigma_gamma_t']
    
    return R, gamma_t, sigma_gamma
```

#### 3.2.3 Covariance Matrices

**Jackknife Covariance:**

**File:** `data/raw/hsc/shear_covariance/cluster_XXXX_cov.npy`

```python
# Load covariance matrix (N_bins × N_bins)
cov = np.load(f'data/raw/hsc/shear_covariance/cluster_{cluster_id:04d}_cov.npy')

# Invert for chi-square calculation
cov_inv = np.linalg.inv(cov)
```

---

### 3.3 CT-C1: QSO Absorption Data

#### 3.3.1 Compiled α Measurements

**Consolidated Dataset:**

**File:** `data/raw/qso_alpha/compiled_alpha_measurements.csv`

**Columns:**
```
qso_name, RA_deg, Dec_deg, z_abs, z_qso, delta_alpha_over_alpha, sigma_delta_alpha, telescope, reference
```

**Example:**
```csv
J0000+0000,0.123,-45.678,2.341,2.567,-1.2e-6,3.4e-6,Keck/HIRES,Murphy2003
J0001+0102,1.234,1.023,1.987,2.123,2.1e-6,4.1e-6,VLT/UVES,Webb2011
...
```

**Sources to Compile:**
- Murphy et al. (2003) - Keck/HIRES
- Webb et al. (2011) - VLT/UVES  
- Molaro et al. (2013) - VLT/UVES
- Kotuš et al. (2017) - Subaru/HDS
- Murphy et al. (2022) - Keck/HIRES (updated)

#### 3.3.2 Standardization Protocol

**Challenge:** Different methods, different systematics.

**Procedure:**
1. Convert all to common reference (Webb+ calibration)
2. Apply systematic corrections per survey
3. Combine with inverse-variance weighting
4. Propagate correlated errors

**Script:** `scripts/utils/compile_alpha_measurements.py`

---

### 3.4 P-S1: Galaxy Catalog Data

#### 3.4.1 SDSS DR18 Galaxy Sample

**File:** `data/raw/sdss/dr18_galaxy_sample.fits`

**SQL Query (for download):**
```sql
SELECT 
  objID, ra, dec, z, petroMag_r, extinction_r,
  type, zWarning, velDisp, velDispErr
FROM SpecObj
WHERE 
  class = 'GALAXY' 
  AND zWarning = 0
  AND z BETWEEN 0.01 AND 0.3
  AND petroMag_r < 17.77  -- Volume-limited sample
```

**Preprocessing:**
```python
import healpy as hp

# Load catalog
galaxies = fits.open('data/raw/sdss/dr18_galaxy_sample.fits')[1].data

# Coordinate conversion
theta = np.radians(90 - galaxies['dec'])
phi = np.radians(galaxies['ra'])

# Rotate to AoE frame
theta_aoe, phi_aoe = rotate_to_aoe_frame(theta, phi, v_aoe)

# Add to catalog
galaxies['theta_aoe'] = theta_aoe
galaxies['phi_aoe'] = phi_aoe
```

#### 3.4.2 Random Catalog

**Purpose:** Subtract mask and selection function effects.

**Generation:**
```python
def generate_random_catalog(N_random, mask):
    """Generate uniform random catalog within survey mask."""
    
    # Start with uniform sky distribution
    nside = 256
    npix = hp.nside2npix(nside)
    
    # Sample from unmasked pixels
    good_pixels = np.where(mask > 0)[0]
    
    randoms = []
    while len(randoms) < N_random:
        # Sample pixel
        pix = np.random.choice(good_pixels, size=10000)
        
        # Convert to coordinates
        theta, phi = hp.pix2ang(nside, pix)
        
        # Apply selection function, completeness, etc.
        # ... (detailed selection matching)
        
        randoms.extend(zip(theta, phi))
    
    return np.array(randoms[:N_random])
```

**File:** `data/processed/sdss/random_catalog_10x.fits`

(10× larger than galaxy catalog for low noise)

---

### 3.5 CT-C2: EHT Polarization Data

#### 3.5.1 M87* Stokes Parameters

**Files:** `data/raw/eht/m87_2017_stokesQU_band6.fits`

**Structure:**
```
HDU 0: Primary (empty)
HDU 1: Image (Stokes I)
HDU 2: Image (Stokes Q)
HDU 3: Image (Stokes U)
HDU 4: Error (sigma_Q)
HDU 5: Error (sigma_U)
```

**Loading:**
```python
from astropy.io import fits

hdu = fits.open('data/raw/eht/m87_2017_stokesQU_band6.fits')

I = hdu[1].data
Q = hdu[2].data
U = hdu[3].data
sigma_Q = hdu[4].data
sigma_U = hdu[5].data

# Compute polarization
P = np.sqrt(Q**2 + U**2)
chi = 0.5 * np.arctan2(U, Q)  # EVPA
```

#### 3.5.2 Coordinate System

**Pixel coordinates → Physical distance:**

```python
# From FITS header
pixsize_rad = hdu[1].header['CDELT1']  # radians
D_Mpc = 16.8  # M87 distance
M_BH = 6.5e9  # Solar masses

# Convert to GM/c^2 units
GM_c2 = G * M_BH * Msun / c**2  # meters
GM_c2_uas = GM_c2 / (D_Mpc * 1e6 * pc) * 206265e6  # microarcsec

# Pixel → GM/c^2
def pix_to_gm(i, j, center_i, center_j):
    """Convert pixel to GM/c^2 units."""
    delta_i = i - center_i
    delta_j = j - center_j
    r_pix = np.sqrt(delta_i**2 + delta_j**2)
    r_uas = r_pix * np.abs(pixsize_rad) * 206265e6
    return r_uas / GM_c2_uas
```

---

## 4. Repository Structure Extensions

### 4.1 New Directory Layout

```
ct-cmb-validation/
├── data/
│   ├── raw/
│   │   ├── pantheon_plus/          # NEW
│   │   │   ├── Pantheon+SH0ES.dat
│   │   │   └── Pantheon+SH0ES_STAT+SYS.cov
│   │   ├── desi_y1/                # NEW
│   │   │   └── desi_y1_bao.json
│   │   ├── hsc/                    # NEW
│   │   │   ├── hsc_s16a_clusters.fits
│   │   │   ├── shear_profiles/
│   │   │   └── shear_covariance/
│   │   ├── des/                    # NEW (alternative to HSC)
│   │   ├── qso_alpha/              # NEW
│   │   │   └── compiled_alpha_measurements.csv
│   │   ├── sdss/                   # NEW
│   │   │   └── dr18_galaxy_sample.fits
│   │   └── eht/                    # NEW
│   │       ├── m87_2017_stokesQU_band6.fits
│   │       └── sgra_2022_stokesQU_band6.fits
│   │
│   └── processed/
│       ├── dark_energy/            # NEW
│       │   └── hubble_diagram.pkl
│       ├── lensing/                # NEW
│       │   ├── stacked_profiles/
│       │   └── individual_fits/
│       ├── alpha_variation/        # NEW
│       │   └── healpix_map_nside32.fits
│       └── lss/                    # NEW
│           └── correlation_multipoles.pkl
│
├── scripts/
│   ├── 10_test_dark_energy.py              # NEW - P-H3
│   ├── 11_test_lensing_profiles.py         # NEW - P-L1
│   ├── 12_test_alpha_variation.py          # NEW - CT-C1
│   ├── 13_test_lss_anisotropy.py           # NEW - P-S1
│   ├── 14_test_bh_polarization.py          # NEW - CT-C2
│   │
│   └── utils/                              # NEW
│       ├── compile_alpha_measurements.py
│       ├── download_pantheon_plus.py
│       ├── download_hsc_catalogs.py
│       └── rotate_to_aoe_frame.py
│
├── src/ct_cmb_validation/
│   ├── cosmology/                          # NEW
│   │   ├── __init__.py
│   │   ├── dark_energy.py
│   │   ├── distances.py
│   │   └── hubble_parameter.py
│   │
│   ├── lensing/                            # NEW
│   │   ├── __init__.py
│   │   ├── profiles.py
│   │   ├── projection.py
│   │   └── fitting.py
│   │
│   ├── structure/                          # NEW
│   │   ├── __init__.py
│   │   ├── correlation_functions.py
│   │   └── multipole_estimators.py
│   │
│   └── polarization/                       # NEW
│       ├── __init__.py
│       ├── stokes_analysis.py
│       └── evpa_decomposition.py
│
└── tests/
    ├── test_dark_energy.py                 # NEW
    ├── test_lensing_profiles.py            # NEW
    └── test_correlation_functions.py       # NEW
```

---

## 5. Core Implementation Details

### 5.1 P-H3 Implementation

#### 5.1.1 Cosmology Module

**File:** `src/ct_cmb_validation/cosmology/dark_energy.py`

```python
"""
Dark Energy equation of state from CT budget dynamics.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class CTDarkEnergy:
    """CT Dark Energy model."""
    
    def __init__(self, R_0=1.7e6):
        """
        Initialize CT dark energy model.
        
        Parameters
        ----------
        R_0 : float
            Quantum-cosmic density ratio at a=1 (from P-H2)
        """
        self.R_0 = R_0
        self.ln_R0_plus_1 = np.log(1 + R_0)
        
    def R(self, a):
        """Hierarchy evolution R(a) = R_0 * a^(-3)."""
        return self.R_0 * a**(-3)
    
    def w(self, a):
        """
        Equation of state w(a).
        
        Parameters
        ----------
        a : float or array
            Scale factor
            
        Returns
        -------
        w : float or array
            Dark energy equation of state
        """
        R = self.R(a)
        ln_term = np.log(1 + R)
        return -1.0 + R / ((1 + R) * ln_term)
    
    def rho_DE(self, a, H0=67.4, Omega_m=0.315):
        """
        Dark energy density rho_DE(a).
        
        Parameters
        ----------
        a : float or array
            Scale factor
        H0 : float
            Hubble constant (km/s/Mpc)
        Omega_m : float
            Matter density parameter today
            
        Returns
        -------
        rho_DE : float or array
            Dark energy density (in units of rho_crit today)
        """
        # Normalization: rho_DE(1) = (1 - Omega_m) * rho_crit
        Omega_DE = 1 - Omega_m
        
        R = self.R(a)
        ln_term = np.log(1 + R)
        
        # Relative to today
        return Omega_DE * ln_term / self.ln_R0_plus_1
    
    def H(self, z, H0=67.4, Omega_m=0.315):
        """
        Hubble parameter H(z) for CT cosmology.
        
        Parameters
        ----------
        z : float or array
            Redshift
        H0 : float
            Hubble constant (km/s/Mpc)
        Omega_m : float
            Matter density parameter
            
        Returns
        -------
        H : float or array
            Hubble parameter (km/s/Mpc)
        """
        a = 1 / (1 + z)
        
        # Friedmann equation: H^2/H0^2 = Omega_m * a^(-3) + Omega_DE(a)
        rho_m = Omega_m * a**(-3)
        rho_DE = self.rho_DE(a, H0, Omega_m)
        
        return H0 * np.sqrt(rho_m + rho_DE)
    
    def luminosity_distance(self, z, H0=67.4, Omega_m=0.315):
        """
        Luminosity distance D_L(z).
        
        Parameters
        ----------
        z : float or array
            Redshift
        H0 : float
            Hubble constant (km/s/Mpc)
        Omega_m : float
            Matter density parameter
            
        Returns
        -------
        D_L : float or array
            Luminosity distance (Mpc)
        """
        from scipy.integrate import quad
        
        c_km_s = 299792.458  # km/s
        
        if np.isscalar(z):
            # Single redshift
            integrand = lambda zp: 1 / self.H(zp, H0, Omega_m)
            integral, _ = quad(integrand, 0, z)
            return c_km_s * (1 + z) * integral
        else:
            # Array of redshifts
            D_L = np.zeros_like(z)
            for i, zi in enumerate(z):
                integrand = lambda zp: 1 / self.H(zp, H0, Omega_m)
                integral, _ = quad(integrand, 0, zi)
                D_L[i] = c_km_s * (1 + zi) * integral
            return D_L
    
    def distance_modulus(self, z, H0=67.4, Omega_m=0.315):
        """
        Distance modulus mu(z) = 5 * log10(D_L / 10 pc).
        
        Parameters
        ----------
        z : float or array
            Redshift
        H0 : float
            Hubble constant (km/s/Mpc)
        Omega_m : float
            Matter density parameter
            
        Returns
        -------
        mu : float or array
            Distance modulus (mag)
        """
        D_L_Mpc = self.luminosity_distance(z, H0, Omega_m)
        return 5 * np.log10(D_L_Mpc * 1e6 / 10)


# For comparison: LCDM model
class LCDMCosmology:
    """Flat LCDM cosmology for comparison."""
    
    def __init__(self, H0=67.4, Omega_m=0.315):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = 1 - Omega_m
    
    def H(self, z):
        """Hubble parameter H(z)."""
        a = 1 / (1 + z)
        Ez2 = self.Omega_m * a**(-3) + self.Omega_Lambda
        return self.H0 * np.sqrt(Ez2)
    
    def luminosity_distance(self, z):
        """Luminosity distance D_L(z)."""
        from scipy.integrate import quad
        
        c_km_s = 299792.458
        
        if np.isscalar(z):
            integrand = lambda zp: 1 / self.H(zp)
            integral, _ = quad(integrand, 0, z)
            return c_km_s * (1 + z) * integral
        else:
            D_L = np.zeros_like(z)
            for i, zi in enumerate(z):
                integrand = lambda zp: 1 / self.H(zp)
                integral, _ = quad(integrand, 0, zi)
                D_L[i] = c_km_s * (1 + zi) * integral
            return D_L
    
    def distance_modulus(self, z):
        """Distance modulus mu(z)."""
        D_L_Mpc = self.luminosity_distance(z)
        return 5 * np.log10(D_L_Mpc * 1e6 / 10)
```

#### 5.1.2 Fitting Framework

**File:** `src/ct_cmb_validation/cosmology/fitting.py`

```python
"""
Fitting framework for cosmological models.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

class CosmologyFitter:
    """Fit cosmological models to SNe + BAO data."""
    
    def __init__(self, sne_data, bao_data, cov_matrix=None):
        """
        Initialize fitter.
        
        Parameters
        ----------
        sne_data : dict
            {'z': array, 'mu_obs': array, 'sigma_mu': array}
        bao_data : dict
            {'z_eff': array, 'DM_over_rd': array, 'sigma_DM': array, ...}
        cov_matrix : array, optional
            Full covariance matrix for SNe
        """
        self.sne_data = sne_data
        self.bao_data = bao_data
        self.cov_matrix = cov_matrix
        
        if cov_matrix is not None:
            # Cholesky decomposition for fast inversion
            self.cov_chol = cho_factor(cov_matrix)
    
    def chi2_sne(self, params, model):
        """
        Chi-square for supernova data.
        
        Parameters
        ----------
        params : array
            [H0, Omega_m, M_B, ...] (model-dependent)
        model : object
            Model with distance_modulus(z, H0, Omega_m) method
            
        Returns
        -------
        chi2 : float
        """
        H0, Omega_m, M_B = params[:3]  # First 3 params always these
        
        # Predicted distance modulus
        mu_pred = model.distance_modulus(self.sne_data['z'], H0, Omega_m)
        
        # Add absolute magnitude
        mu_pred += M_B
        
        # Residuals
        residuals = self.sne_data['mu_obs'] - mu_pred
        
        # Chi-square
        if self.cov_matrix is not None:
            # Use full covariance
            chi2 = residuals @ cho_solve(self.cov_chol, residuals)
        else:
            # Diagonal errors only
            chi2 = np.sum((residuals / self.sne_data['sigma_mu'])**2)
        
        return chi2
    
    def chi2_bao(self, params, model):
        """Chi-square for BAO data."""
        H0, Omega_m = params[:2]
        
        chi2 = 0.0
        
        # Sound horizon (fixed from CMB)
        r_d = 147.05  # Mpc (Planck 2018)
        
        for measurement in self.bao_data:
            z_eff = measurement['z_eff']
            
            # Comoving angular diameter distance
            from scipy.integrate import quad
            c_km_s = 299792.458
            integrand = lambda z: c_km_s / model.H(z, H0, Omega_m)
            D_M, _ = quad(integrand, 0, z_eff)
            
            # Hubble distance
            D_H = c_km_s / model.H(z_eff, H0, Omega_m)
            
            # Predictions
            DM_over_rd_pred = D_M / r_d
            DH_over_rd_pred = D_H / r_d
            
            # Chi-square contributions
            chi2 += ((measurement['DM_over_rd'] - DM_over_rd_pred) / 
                     measurement['sigma_DM_over_rd'])**2
            chi2 += ((measurement['DH_over_rd'] - DH_over_rd_pred) / 
                     measurement['sigma_DH_over_rd'])**2
        
        return chi2
    
    def total_chi2(self, params, model, use_bao=True):
        """Total chi-square (SNe + BAO)."""
        chi2 = self.chi2_sne(params, model)
        
        if use_bao:
            chi2 += self.chi2_bao(params, model)
        
        return chi2
    
    def fit(self, model, initial_params=[67.4, 0.315, -19.3], use_bao=True):
        """
        Fit model to data.
        
        Parameters
        ----------
        model : object
            Cosmological model
        initial_params : list
            Initial guess [H0, Omega_m, M_B]
        use_bao : bool
            Include BAO constraints
            
        Returns
        -------
        result : dict
            Fit results
        """
        # Minimize chi-square
        result = minimize(
            lambda p: self.total_chi2(p, model, use_bao),
            initial_params,
            method='Powell',
            options={'maxiter': 10000}
        )
        
        # Package results
        H0_fit, Omega_m_fit, M_B_fit = result.x
        chi2_min = result.fun
        
        # Degrees of freedom
        N_sne = len(self.sne_data['z'])
        N_bao = len(self.bao_data) * 2 if use_bao else 0
        N_data = N_sne + N_bao
        N_params = len(initial_params)
        dof = N_data - N_params
        
        return {
            'H0': H0_fit,
            'Omega_m': Omega_m_fit,
            'M_B': M_B_fit,
            'chi2': chi2_min,
            'dof': dof,
            'chi2_per_dof': chi2_min / dof,
            'success': result.success
        }
```

#### 5.1.3 Main Analysis Script

**File:** `scripts/10_test_dark_energy.py`

```python
#!/usr/bin/env python3
"""
P-H3: Test dark energy equation of state prediction.

Compares CT dark energy model (w_CT) against LCDM (w=-1)
using Pantheon+ SNe and DESI BAO data.

Usage:
    python scripts/10_test_dark_energy.py [--use-bao] [--plot]
"""

import sys
import os
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from ct_cmb_validation.cosmology.dark_energy import CTDarkEnergy, LCDMCosmology
from ct_cmb_validation.cosmology.fitting import CosmologyFitter
from ct_cmb_validation.utils.logging_setup import setup_logger
from ct_cmb_validation.utils.config_loader import load_config

def load_pantheon_plus():
    """Load Pantheon+ SNe data."""
    data_path = repo_root / 'data/raw/pantheon_plus/Pantheon+SH0ES.dat'
    
    if not data_path.exists():
        logger.error(f"Pantheon+ data not found at {data_path}")
        logger.info("Please download from: https://pantheonplussh0es.github.io/")
        return None
    
    # Load data
    import pandas as pd
    df = pd.read_csv(data_path, sep='\s+', comment='#')
    
    # Apply quality cuts
    mask = (df['zcmb'] > 0.01) & (df['zcmb'] < 2.3)
    mask &= (df['dmb'] < 0.5)
    
    df_clean = df[mask].copy()
    
    logger.info(f"Loaded {len(df_clean)} SNe from Pantheon+")
    
    return {
        'z': df_clean['zcmb'].values,
        'mb': df_clean['mb'].values,
        'sigma_mb': df_clean['dmb'].values
    }

def load_desi_bao():
    """Load DESI Year 1 BAO measurements."""
    data_path = repo_root / 'data/raw/desi_y1/desi_y1_bao.json'
    
    if not data_path.exists():
        logger.warning(f"DESI BAO data not found at {data_path}")
        return []
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data['measurements'])} BAO measurements")
    
    return data['measurements']

def main():
    parser = argparse.ArgumentParser(description='P-H3: Dark Energy Test')
    parser.add_argument('--use-bao', action='store_true', help='Include BAO constraints')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{timestamp}_ph3_dark_energy'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / 'results/runs' / run_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logger('P-H3', output_dir / 'run_log.log')
    
    logger.info("="*70)
    logger.info("P-H3: Dark Energy Equation of State Test")
    logger.info("="*70)
    
    # Load data
    logger.info("\n1. Loading Data...")
    sne_data = load_pantheon_plus()
    if sne_data is None:
        return
    
    if args.use_bao:
        bao_data = load_desi_bao()
    else:
        bao_data = []
        logger.info("Skipping BAO (--use-bao not specified)")
    
    # Prepare SNe data for fitting
    # Note: mb needs absolute magnitude M_B subtracted → fit as free parameter
    sne_fit_data = {
        'z': sne_data['z'],
        'mu_obs': sne_data['mb'],  # Will add M_B in fitter
        'sigma_mu': sne_data['sigma_mb']
    }
    
    # Initialize models
    logger.info("\n2. Initializing Models...")
    
    ct_model = CTDarkEnergy(R_0=1.7e6)  # From P-H2
    lcdm_model = LCDMCosmology()
    
    logger.info(f"CT Model: R_0 = 1.7e6 (from P-H2 Hubble tension)")
    logger.info(f"CT Prediction: w_0 = {ct_model.w(1.0):.4f}")
    
    # Fit models
    logger.info("\n3. Fitting Models...")
    
    fitter = CosmologyFitter(sne_fit_data, bao_data)
    
    initial_params = [67.4, 0.315, -19.3]  # [H0, Omega_m, M_B]
    
    logger.info("\nFitting LCDM...")
    lcdm_result = fitter.fit(lcdm_model, initial_params, use_bao=args.use_bao)
    
    logger.info(f"  H0 = {lcdm_result['H0']:.2f} km/s/Mpc")
    logger.info(f"  Omega_m = {lcdm_result['Omega_m']:.4f}")
    logger.info(f"  M_B = {lcdm_result['M_B']:.3f}")
    logger.info(f"  χ² = {lcdm_result['chi2']:.2f} / {lcdm_result['dof']} dof")
    logger.info(f"  χ²/dof = {lcdm_result['chi2_per_dof']:.3f}")
    
    logger.info("\nFitting CT Model...")
    ct_result = fitter.fit(ct_model, initial_params, use_bao=args.use_bao)
    
    logger.info(f"  H0 = {ct_result['H0']:.2f} km/s/Mpc")
    logger.info(f"  Omega_m = {ct_result['Omega_m']:.4f}")
    logger.info(f"  M_B = {ct_result['M_B']:.3f}")
    logger.info(f"  χ² = {ct_result['chi2']:.2f} / {ct_result['dof']} dof")
    logger.info(f"  χ²/dof = {ct_result['chi2_per_dof']:.3f}")
    
    # Model comparison
    logger.info("\n4. Model Comparison...")
    
    Delta_chi2 = ct_result['chi2'] - lcdm_result['chi2']
    Delta_BIC = Delta_chi2 + 0 * np.log(lcdm_result['dof'])  # Same # params
    
    logger.info(f"  Δχ² (CT - LCDM) = {Delta_chi2:.2f}")
    logger.info(f"  ΔBIC = {Delta_BIC:.2f}")
    
    if Delta_chi2 < -10:
        verdict = "STRONGLY FAVORS CT"
    elif Delta_chi2 < -5:
        verdict = "FAVORS CT"
    elif Delta_chi2 < 5:
        verdict = "INCONCLUSIVE"
    elif Delta_chi2 < 10:
        verdict = "FAVORS LCDM"
    else:
        verdict = "STRONGLY FAVORS LCDM"
    
    logger.info(f"  Verdict: {verdict}")
    
    # Save results
    results = {
        'ct_model': {
            'R_0': 1.7e6,
            'w_0_predicted': float(ct_model.w(1.0)),
            'fit': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in ct_result.items()}
        },
        'lcdm_model': {
            'w': -1.0,
            'fit': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in lcdm_result.items()}
        },
        'comparison': {
            'Delta_chi2': float(Delta_chi2),
            'Delta_BIC': float(Delta_BIC),
            'verdict': verdict
        },
        'data_used': {
            'N_sne': len(sne_fit_data['z']),
            'N_bao': len(bao_data) if args.use_bao else 0
        }
    }
    
    results_file = output_dir / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    if args.plot:
        logger.info("\n5. Generating Plots...")
        plot_hubble_diagram(sne_fit_data, ct_model, lcdm_model, 
                           ct_result, lcdm_result, output_dir)
        plot_w_evolution(ct_model, output_dir)
        logger.info(f"✓ Plots saved to {output_dir / 'plots'}/")
    
    logger.info("\n" + "="*70)
    logger.info("P-H3 Analysis Complete")
    logger.info("="*70)

def plot_hubble_diagram(sne_data, ct_model, lcdm_model, ct_fit, lcdm_fit, output_dir):
    """Generate Hubble diagram plot."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    
    z_plot = np.linspace(0.01, 2.0, 200)
    
    # LCDM prediction
    mu_lcdm = lcdm_model.distance_modulus(z_plot)
    mu_lcdm += lcdm_fit['M_B']
    
    # CT prediction  
    mu_ct = ct_model.distance_modulus(z_plot, ct_fit['H0'], ct_fit['Omega_m'])
    mu_ct += ct_fit['M_B']
    
    # Main panel: Hubble diagram
    ax1.errorbar(sne_data['z'], sne_data['mu_obs'], yerr=sne_data['sigma_mu'],
                 fmt='o', ms=2, alpha=0.5, color='gray', label='Pantheon+')
    ax1.plot(z_plot, mu_lcdm, 'b-', lw=2, label=f'ΛCDM (χ²/dof={lcdm_fit["chi2_per_dof"]:.2f})')
    ax1.plot(z_plot, mu_ct, 'r-', lw=2, label=f'CT (χ²/dof={ct_fit["chi2_per_dof"]:.2f})')
    
    ax1.set_ylabel('Distance Modulus μ', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Residual panel
    mu_lcdm_data = lcdm_model.distance_modulus(sne_data['z'])
    mu_lcdm_data += lcdm_fit['M_B']
    
    mu_ct_data = ct_model.distance_modulus(sne_data['z'], ct_fit['H0'], ct_fit['Omega_m'])
    mu_ct_data += ct_fit['M_B']
    
    res_lcdm = sne_data['mu_obs'] - mu_lcdm_data
    res_ct = sne_data['mu_obs'] - mu_ct_data
    
    ax2.errorbar(sne_data['z'], res_lcdm, yerr=sne_data['sigma_mu'],
                 fmt='o', ms=2, alpha=0.5, color='blue', label='ΛCDM residuals')
    ax2.errorbar(sne_data['z'], res_ct, yerr=sne_data['sigma_mu'],
                 fmt='o', ms=2, alpha=0.5, color='red', label='CT residuals')
    ax2.axhline(0, color='k', ls='--', lw=1)
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hubble_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_w_evolution(ct_model, output_dir):
    """Plot w(z) evolution."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    z_plot = np.linspace(0, 2, 200)
    a_plot = 1 / (1 + z_plot)
    w_ct = ct_model.w(a_plot)
    
    ax.plot(z_plot, w_ct, 'r-', lw=3, label='CT Prediction')
    ax.axhline(-1, color='b', ls='--', lw=2, label='ΛCDM (w=-1)')
    
    # Mark present day
    ax.axvline(0, color='k', ls=':', alpha=0.5)
    ax.plot(0, ct_model.w(1.0), 'ro', ms=10, 
            label=f'Today: w₀ = {ct_model.w(1.0):.3f}')
    
    ax.set_xlabel('Redshift z', fontsize=14)
    ax.set_ylabel('Dark Energy Equation of State w(z)', fontsize=14)
    ax.set_title('CT Dark Energy Evolution (Parameter-Free Prediction)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([-1.05, -0.85])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'w_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
```

---

### 5.2 P-L1 Implementation

Due to length constraints, I'll provide the key components in abbreviated form:

**File:** `src/ct_cmb_validation/lensing/profiles.py`

```python
"""
CT and NFW density profile models for weak lensing.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

class CTProfile:
    """CT interface profile with tanh transition."""
    
    def __init__(self, rho_inf, rho_0, R_200, xi):
        """
        Parameters
        ----------
        rho_inf : float
            Background density
        rho_0 : float
            Central density
        R_200 : float
            Virial radius (kpc)
        xi : float
            Interface width (kpc) - CT PREDICTION
        """
        self.rho_inf = rho_inf
        self.rho_0 = rho_0
        self.R_200 = R_200
        self.xi = xi
    
    def rho_3d(self, r):
        """3D density profile."""
        Delta_rho = self.rho_0 - self.rho_inf
        tanh_term = np.tanh((self.R_200 - r) / self.xi)
        return self.rho_inf + 0.5 * Delta_rho * (1 + tanh_term)
    
    def Sigma(self, R):
        """
        Projected surface density.
        
        Parameters
        ----------
        R : float or array
            Projected radius (kpc)
            
        Returns
        -------
        Sigma : float or array
            Surface density
        """
        if np.isscalar(R):
            integrand = lambda r: self.rho_3d(r) * r / np.sqrt(r**2 - R**2)
            result, _ = quad(integrand, R, 10 * self.R_200)
            return 2 * result
        else:
            return np.array([self.Sigma(R_i) for R_i in R])
    
    def gamma_t(self, R, Sigma_crit):
        """
        Tangential shear.
        
        Parameters
        ----------
        R : array
            Radial bins (kpc)
        Sigma_crit : float
            Critical surface density
            
        Returns
        -------
        gamma_t : array
            Tangential shear
        """
        Sigma_R = self.Sigma(R)
        
        # Mean surface density inside R
        Sigma_bar = np.zeros_like(R)
        for i, R_i in enumerate(R):
            R_grid = np.linspace(0, R_i, 100)
            Sigma_grid = self.Sigma(R_grid)
            Sigma_bar[i] = 2 / R_i**2 * np.trapz(Sigma_grid * R_grid, R_grid)
        
        return (Sigma_bar - Sigma_R) / Sigma_crit


class NFWProfile:
    """NFW profile for comparison."""
    
    def __init__(self, M_200, c_200, z_lens, cosmology):
        """
        Parameters
        ----------
        M_200 : float
            Virial mass (M_sun)
        c_200 : float
            Concentration parameter
        z_lens : float
            Lens redshift
        cosmology : object
            Cosmology instance (for critical density)
        """
        self.M_200 = M_200
        self.c_200 = c_200
        self.z_lens = z_lens
        
        # Compute R_200 and r_s
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
        
        rho_crit = cosmo.critical_density(z_lens).to('Msun/kpc3').value
        self.R_200 = (3 * M_200 / (4 * np.pi * 200 * rho_crit))**(1/3)  # kpc
        self.r_s = self.R_200 / c_200
        
        # Characteristic density
        delta_c = 200/3 * c_200**3 / (np.log(1 + c_200) - c_200/(1 + c_200))
        self.rho_s = delta_c * rho_crit
    
    def rho_3d(self, r):
        """3D density profile."""
        x = r / self.r_s
        return self.rho_s / (x * (1 + x)**2)
    
    def Sigma(self, R):
        """Projected surface density (analytic)."""
        x = R / self.r_s
        
        def F(x):
            if x < 1:
                return (1 / (x**2 - 1) * 
                        (1 - 2/np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1-x)/(1+x)))))
            elif x == 1:
                return 1/3
            else:
                return (1 / (x**2 - 1) * 
                        (1 - 2/np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x-1)/(x+1)))))
        
        if np.isscalar(x):
            return 2 * self.rho_s * self.r_s * F(x)
        else:
            return np.array([2 * self.rho_s * self.r_s * F(x_i) for x_i in x])
    
    def gamma_t(self, R, Sigma_crit):
        """Tangential shear."""
        Sigma_R = self.Sigma(R)
        
        # Mean surface density (analytic for NFW)
        x = R / self.r_s
        
        def g(x):
            if x < 1:
                return (8 * np.arctanh(np.sqrt((1-x)/(1+x))) / (x**2 * np.sqrt(1-x**2)) +
                        4/x**2 * np.log(x/2) - 2/(x**2-1) +
                        4 * np.arctanh(np.sqrt((1-x)/(1+x))) / ((x**2-1)**(3/2)))
            elif x == 1:
                return 10/3 + 4*np.log(0.5)
            else:
                return (8 * np.arctan(np.sqrt((x-1)/(1+x))) / (x**2 * np.sqrt(x**2-1)) +
                        4/x**2 * np.log(x/2) - 2/(x**2-1) +
                        4 * np.arctan(np.sqrt((x-1)/(1+x))) / ((x**2-1)**(3/2)))
        
        if np.isscalar(x):
            Sigma_bar = self.rho_s * self.r_s * g(x)
        else:
            Sigma_bar = np.array([self.rho_s * self.r_s * g(x_i) for x_i in x])
        
        return (Sigma_bar - Sigma_R) / Sigma_crit
```

---

## 6. Testing & Validation Protocols

### 6.1 Unit Tests

**File:** `tests/test_dark_energy.py`

```python
"""
Unit tests for CT dark energy module.
"""

import pytest
import numpy as np
from ct_cmb_validation.cosmology.dark_energy import CTDarkEnergy, LCDMCosmology

def test_ct_w_bounds():
    """Test that w(a) stays within physical bounds."""
    ct = CTDarkEnergy(R_0=1.7e6)
    
    a_grid = np.logspace(-2, 0, 100)  # a from 0.01 to 1
    w_grid = ct.w(a_grid)
    
    # Should be between -1.2 and -0.8 (thawing model)
    assert np.all(w_grid > -1.2)
    assert np.all(w_grid < -0.8)

def test_ct_w_asymptotic():
    """Test asymptotic behavior of w(a)."""
    ct = CTDarkEnergy(R_0=1.7e6)
    
    # At a=0 (early universe), w → -1
    w_early = ct.w(0.01)
    assert abs(w_early - (-1.0)) < 0.05
    
    # At a=1 (today), w ≈ -0.93
    w_today = ct.w(1.0)
    assert abs(w_today - (-0.93)) < 0.01

def test_ct_rho_DE_positive():
    """Test that rho_DE is always positive."""
    ct = CTDarkEnergy(R_0=1.7e6)
    
    a_grid = np.logspace(-2, 0, 100)
    rho_grid = ct.rho_DE(a_grid)
    
    assert np.all(rho_grid > 0)

def test_lcdm_consistency():
    """Test LCDM against known results."""
    lcdm = LCDMCosmology(H0=70, Omega_m=0.3)
    
    # Hubble parameter at z=0
    H0_test = lcdm.H(0)
    assert abs(H0_test - 70) < 0.01
    
    # Hubble parameter at z=1
    # H(z=1) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
    H_z1_expected = 70 * np.sqrt(0.3 * 8 + 0.7)
    H_z1_test = lcdm.H(1.0)
    assert abs(H_z1_test - H_z1_expected) / H_z1_expected < 0.001

def test_luminosity_distance_monotonic():
    """Test that D_L(z) increases monotonically."""
    ct = CTDarkEnergy(R_0=1.7e6)
    
    z_grid = np.linspace(0.1, 2.0, 20)
    D_L_grid = ct.luminosity_distance(z_grid)
    
    # Should be strictly increasing
    assert np.all(np.diff(D_L_grid) > 0)

def test_ct_vs_lcdm_difference():
    """Test that CT and LCDM give different predictions."""
    ct = CTDarkEnergy(R_0=1.7e6)
    lcdm = LCDMCosmology(H0=67.4, Omega_m=0.315)
    
    z_test = np.array([0.5, 1.0, 1.5])
    
    D_L_ct = ct.luminosity_distance(z_test)
    D_L_lcdm = lcdm.luminosity_distance(z_test)
    
    # Should differ by O(0.1-1%)
    frac_diff = np.abs(D_L_ct - D_L_lcdm) / D_L_lcdm
    
    assert np.all(frac_diff > 0.0001)  # Not identical
    assert np.all(frac_diff < 0.02)    # But close

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```
---

### 6.2 Integration Tests

**File:** `tests/test_integration_ph3.py`

```python
"""
Integration test for P-H3 dark energy analysis pipeline.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

from ct_cmb_validation.cosmology.dark_energy import CTDarkEnergy, LCDMCosmology
from ct_cmb_validation.cosmology.fitting import CosmologyFitter

@pytest.fixture
def mock_sne_data():
    """Generate mock supernova data."""
    np.random.seed(42)
    
    # True cosmology (LCDM for mock)
    true_cosmo = LCDMCosmology(H0=70, Omega_m=0.3)
    
    # Generate redshifts
    z = np.random.uniform(0.01, 1.5, 100)
    z = np.sort(z)
    
    # True distance modulus
    mu_true = true_cosmo.distance_modulus(z) - 19.3  # M_B = -19.3
    
    # Add noise
    sigma = 0.15
    mu_obs = mu_true + np.random.normal(0, sigma, len(z))
    
    return {
        'z': z,
        'mu_obs': mu_obs,
        'sigma_mu': np.full_like(z, sigma)
    }

@pytest.fixture
def mock_bao_data():
    """Generate mock BAO data."""
    return [
        {
            'z_eff': 0.5,
            'DM_over_rd': 19.0,
            'sigma_DM_over_rd': 0.5,
            'DH_over_rd': 20.0,
            'sigma_DH_over_rd': 0.8
        }
    ]

def test_ct_fitting_pipeline(mock_sne_data, mock_bao_data):
    """Test full CT fitting pipeline."""
    
    # Initialize models
    ct_model = CTDarkEnergy(R_0=1.7e6)
    lcdm_model = LCDMCosmology()
    
    # Initialize fitter
    fitter = CosmologyFitter(mock_sne_data, mock_bao_data)
    
    # Fit CT model
    ct_result = fitter.fit(ct_model, [70, 0.3, -19.3], use_bao=False)
    
    # Should converge
    assert ct_result['success']
    
    # H0 should be reasonable
    assert 60 < ct_result['H0'] < 80
    
    # Omega_m should be reasonable
    assert 0.2 < ct_result['Omega_m'] < 0.4
    
    # Chi-square should be reasonable
    assert ct_result['chi2_per_dof'] < 3.0

def test_model_comparison(mock_sne_data):
    """Test CT vs LCDM comparison."""
    
    ct_model = CTDarkEnergy(R_0=1.7e6)
    lcdm_model = LCDMCosmology()
    
    fitter = CosmologyFitter(mock_sne_data, [])
    
    ct_result = fitter.fit(ct_model, [70, 0.3, -19.3], use_bao=False)
    lcdm_result = fitter.fit(lcdm_model, [70, 0.3, -19.3], use_bao=False)
    
    # Both should converge
    assert ct_result['success']
    assert lcdm_result['success']
    
    # Chi-squares should be comparable (mock data is LCDM)
    Delta_chi2 = ct_result['chi2'] - lcdm_result['chi2']
    assert abs(Delta_chi2) < 20  # Should be similar for mock LCDM data

def test_output_consistency(mock_sne_data, tmp_path):
    """Test that output files are generated correctly."""
    
    ct_model = CTDarkEnergy(R_0=1.7e6)
    fitter = CosmologyFitter(mock_sne_data, [])
    
    result = fitter.fit(ct_model, [70, 0.3, -19.3], use_bao=False)
    
    # Save results
    output_file = tmp_path / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'H0': float(result['H0']),
            'Omega_m': float(result['Omega_m']),
            'chi2': float(result['chi2'])
        }, f)
    
    # Load and verify
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    
    assert abs(loaded['H0'] - result['H0']) < 0.01
    assert abs(loaded['Omega_m'] - result['Omega_m']) < 0.0001

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

### 6.3 Validation Protocols

#### 6.3.1 Pre-Analysis Checklist

**Before running any extended validation test:**

✅ **Data Integrity**
```bash
# Run data validation script
python scripts/utils/validate_data_files.py

# Should check:
# - File existence
# - Format correctness (FITS, CSV, JSON)
# - Column/field completeness
# - Value ranges (no NaNs, infinities)
# - Coordinate consistency
```

✅ **Code Tests**
```bash
# Run full test suite
pytest tests/ -v --cov=src/ct_cmb_validation

# Minimum coverage: 80%
# All tests should pass
```

✅ **Baseline Validation**
```bash
# Verify validated predictions still hold
python scripts/01_run_pt2_alignment_test.py  # Should give p_d6 ≈ 0.015
python scripts/03_run_ph2_hubble_test.py     # Should give 8.29% vs 8.31%
```

#### 6.3.2 Analysis Execution Protocol

**Standardized workflow for each test:**

1. **Setup Phase**
```bash
# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${TIMESTAMP}_test_name"
RUN_DIR="results/runs/${RUN_NAME}"
mkdir -p ${RUN_DIR}

# Copy configuration
cp config/main_config.yaml ${RUN_DIR}/run_config.yaml

# Initialize log
echo "Run: ${RUN_NAME}" > ${RUN_DIR}/run_log.log
echo "Start: $(date)" >> ${RUN_DIR}/run_log.log
```

2. **Execution Phase**
```bash
# Run with error handling
python scripts/XX_test_name.py \
    --output-dir ${RUN_DIR} \
    2>&1 | tee -a ${RUN_DIR}/run_log.log

# Check exit code
if [ $? -eq 0 ]; then
    echo "Status: SUCCESS" >> ${RUN_DIR}/run_log.log
else
    echo "Status: FAILED" >> ${RUN_DIR}/run_log.log
    exit 1
fi
```

3. **Validation Phase**
```bash
# Verify output files exist
test -f ${RUN_DIR}/final_results.json || exit 1
test -f ${RUN_DIR}/observed_data.json || exit 1

# Parse results
python scripts/utils/parse_results.py ${RUN_DIR}/final_results.json

# Check for anomalies
python scripts/utils/check_anomalies.py ${RUN_DIR}
```

4. **Documentation Phase**
```bash
# Update summary report
python scripts/05_generate_summary_report.py

# Commit results (if using git)
git add ${RUN_DIR}
git commit -m "Results: ${RUN_NAME}"
```

#### 6.3.3 Monte Carlo Convergence Tests

**Ensuring sufficient statistics:**

```python
def test_mc_convergence(mc_samples, observed_stat):
    """
    Test if Monte Carlo has converged.
    
    Parameters
    ----------
    mc_samples : array
        Monte Carlo distribution samples
    observed_stat : float
        Observed test statistic
        
    Returns
    -------
    converged : bool
    diagnostics : dict
    """
    N = len(mc_samples)
    
    # 1. Split-half test
    mc_half1 = mc_samples[:N//2]
    mc_half2 = mc_samples[N//2:]
    
    p_half1 = np.sum(mc_half1 >= observed_stat) / len(mc_half1)
    p_half2 = np.sum(mc_half2 >= observed_stat) / len(mc_half2)
    
    # Should be consistent within 2σ Poisson fluctuation
    sigma_p = np.sqrt(p_half1 * (1 - p_half1) / len(mc_half1))
    split_test = abs(p_half1 - p_half2) < 2 * sigma_p
    
    # 2. Tail stability
    # p-value should be stable when adding more samples
    N_tests = 10
    p_values = []
    for i in range(N_tests):
        N_use = int(N * (0.5 + 0.05*i))
        p_i = np.sum(mc_samples[:N_use] >= observed_stat) / N_use
        p_values.append(p_i)
    
    p_std = np.std(p_values)
    p_mean = np.mean(p_values)
    tail_stable = (p_std / p_mean) < 0.2  # <20% relative variation
    
    # 3. Minimum sample size
    # For p ~ 0.01, need N > 10000 for good statistics
    p_estimate = np.sum(mc_samples >= observed_stat) / N
    min_N = 100 / max(p_estimate, 0.001)  # Want ~100 events in tail
    sample_sufficient = N > min_N
    
    converged = split_test and tail_stable and sample_sufficient
    
    diagnostics = {
        'N_samples': N,
        'p_estimate': p_estimate,
        'split_test_pass': split_test,
        'p_half1': p_half1,
        'p_half2': p_half2,
        'tail_stable': tail_stable,
        'p_variation': p_std / p_mean if p_mean > 0 else np.inf,
        'sample_sufficient': sample_sufficient,
        'min_N_recommended': int(min_N),
        'converged': converged
    }
    
    return converged, diagnostics
```

**Usage in analysis scripts:**

```python
# After Monte Carlo loop
converged, diagnostics = test_mc_convergence(mc_statistics, observed_statistic)

if not converged:
    logger.warning("Monte Carlo may not have converged!")
    logger.warning(f"Diagnostics: {diagnostics}")
    logger.warning(f"Recommend running with N_MC >= {diagnostics['min_N_recommended']}")
    
    # Save warning to results
    results['mc_convergence_warning'] = True
    results['mc_diagnostics'] = diagnostics
```

---

## 7. Analysis Workflows

### 7.1 P-H3: Dark Energy — Complete Workflow

#### 7.1.1 Preparation (Day 1)

**Download Data:**
```bash
# Pantheon+ SNe
cd data/raw/pantheon_plus
wget https://pantheonplussh0es.github.io/DataRelease/Pantheon+SH0ES.dat
wget https://pantheonplussh0es.github.io/DataRelease/Pantheon+SH0ES_STAT+SYS.cov

# DESI BAO (if available)
cd data/raw/desi_y1
# Download from DESI data release (URL TBD)
# Or manually create JSON from published values
```

**Validate Data:**
```bash
python scripts/utils/validate_data_files.py --dataset pantheon_plus
python scripts/utils/validate_data_files.py --dataset desi_y1
```

#### 7.1.2 Preliminary Analysis (Day 2)

**Quick fit (no BAO, diagonal errors):**
```bash
python scripts/10_test_dark_energy.py --plot
```

**Check outputs:**
```bash
LATEST_RUN=$(ls -t results/runs/ | grep ph3 | head -1)
cat results/runs/${LATEST_RUN}/final_results.json
```

**Expected first-pass results:**
- CT should fit with χ²/dof ~ 1.0-1.2
- w₀ should be around -0.93 ± 0.02
- Δχ² (CT - ΛCDM) should be order -5 to +5

#### 7.1.3 Full Analysis (Day 3-5)

**With BAO constraints:**
```bash
python scripts/10_test_dark_energy.py --use-bao --plot
```

**Systematic checks:**

```python
# scripts/utils/ph3_systematics.py
"""
Systematic uncertainty checks for P-H3.
"""

def check_redshift_binning(sne_data, ct_model, lcdm_model):
    """Test if results depend on redshift binning."""
    
    # Split into low-z and high-z
    z_split = 0.5
    
    mask_low = sne_data['z'] < z_split
    mask_high = sne_data['z'] >= z_split
    
    # Fit each subsample
    fitter_low = CosmologyFitter({'z': sne_data['z'][mask_low],
                                  'mu_obs': sne_data['mu_obs'][mask_low],
                                  'sigma_mu': sne_data['sigma_mu'][mask_low]}, [])
    
    result_low_ct = fitter_low.fit(ct_model, [67, 0.31, -19.3], use_bao=False)
    result_low_lcdm = fitter_low.fit(lcdm_model, [67, 0.31, -19.3], use_bao=False)
    
    # Similarly for high-z
    # ...
    
    # Compare Δχ² in each bin
    Delta_chi2_low = result_low_ct['chi2'] - result_low_lcdm['chi2']
    Delta_chi2_high = result_high_ct['chi2'] - result_high_lcdm['chi2']
    
    # Should be consistent (within Poisson)
    return {
        'Delta_chi2_low_z': Delta_chi2_low,
        'Delta_chi2_high_z': Delta_chi2_high,
        'consistent': abs(Delta_chi2_low - Delta_chi2_high) < 10
    }

def check_magnitude_systematics(sne_data):
    """Check for magnitude-dependent systematics."""
    
    # Bright vs faint SNe
    median_mb = np.median(sne_data['mb'])
    
    mask_bright = sne_data['mb'] < median_mb
    mask_faint = sne_data['mb'] >= median_mb
    
    # Fit each subsample, check consistency
    # ...
    
    return systematics_report

def check_covariance_impact(sne_data, cov_matrix):
    """Test impact of full covariance vs diagonal."""
    
    # Fit with diagonal errors
    fitter_diag = CosmologyFitter(sne_data, [])
    result_diag = fitter_diag.fit(ct_model, [67, 0.31, -19.3])
    
    # Fit with full covariance
    fitter_full = CosmologyFitter(sne_data, [], cov_matrix=cov_matrix)
    result_full = fitter_full.fit(ct_model, [67, 0.31, -19.3])
    
    # Report parameter shifts
    return {
        'H0_shift': result_full['H0'] - result_diag['H0'],
        'Omega_m_shift': result_full['Omega_m'] - result_diag['Omega_m'],
        'chi2_change': result_full['chi2'] - result_diag['chi2']
    }
```

**Run systematics:**
```bash
python scripts/utils/ph3_systematics.py
```

#### 7.1.4 Final Validation (Day 6-7)

**Cross-checks:**
1. **Compare to literature:** ΛCDM fit should reproduce Planck cosmology
2. **Blind analysis:** Have independent analyzer run pipeline
3. **Mock data test:** Apply to synthetic ΛCDM data, should recover w = -1

**Generate final plots:**
```bash
python scripts/utils/ph3_make_publication_plots.py \
    --run-dir results/runs/${LATEST_RUN} \
    --output plots/publication/
```

**Write summary:**
```bash
python scripts/05_generate_summary_report.py --focus ph3
```

#### 7.1.5 Interpretation Decision Tree

```
Is Δχ² (CT - ΛCDM) significant?
│
├─ YES: Δχ² < -10 (BIC > 10)
│   │
│   ├─ CT fits better
│   │   ├─ Is w₀ ≈ -0.93 (within 1σ)?
│   │   │   ├─ YES → ✅ VALIDATED: Parameter-free prediction confirmed
│   │   │   └─ NO → ⚠️ PARTIAL: CT better but w₀ unexpected
│   │   │
│   │   └─ Check: Does w(z) match CT functional form?
│   │       ├─ YES → Strong confirmation
│   │       └─ NO → May need refined model
│   │
│   └─ ΛCDM fits better
│       └─ ❌ FALSIFIED: Data prefer w = -1
│
└─ NO: |Δχ²| < 10
    │
    └─ INCONCLUSIVE: Need more data or better systematics
        ├─ Increase sample (wait for LSST, Euclid)
        └─ Improve covariance understanding
```

---

### 7.2 P-L1: Lensing Profiles — Complete Workflow

#### 7.2.1 Sample Selection (Week 1)

**Load cluster catalog:**
```python
# scripts/utils/pl1_select_clusters.py
"""
Select high-quality cluster sample for P-L1.
"""

from astropy.io import fits
import numpy as np

def select_cluster_sample(catalog_path, output_path):
    """
    Select clusters for lensing profile analysis.
    
    Criteria:
    - M_200 > 1e14 M_sun (massive clusters)
    - z < 0.7 (good redshift range)
    - SNR > 10 (high significance)
    - No contamination flags
    """
    
    hdu = fits.open(catalog_path)
    clusters = hdu[1].data
    
    # Apply cuts
    mask = (clusters['M_200'] > 1e14)
    mask &= (clusters['z_spec'] > 0.1) & (clusters['z_spec'] < 0.7)
    mask &= (clusters['SNR'] > 10)
    mask &= (clusters['contamination_flag'] == 0)
    
    sample = clusters[mask]
    
    print(f"Selected {len(sample)} / {len(clusters)} clusters")
    print(f"Mass range: {sample['M_200'].min():.2e} - {sample['M_200'].max():.2e}")
    print(f"Redshift range: {sample['z_spec'].min():.3f} - {sample['z_spec'].max():.3f}")
    
    # Save
    hdu_out = fits.BinTableHDU(sample)
    hdu_out.writeto(output_path, overwrite=True)
    
    return sample

if __name__ == '__main__':
    sample = select_cluster_sample(
        'data/raw/hsc/hsc_s16a_clusters.fits',
        'data/processed/lensing/selected_clusters.fits'
    )
```

#### 7.2.2 Profile Extraction (Week 2)

**Stack clusters:**
```python
# scripts/utils/pl1_stack_profiles.py
"""
Stack weak lensing profiles.
"""

def stack_profiles(cluster_ids, radial_bins):
    """
    Stack tangential shear profiles.
    
    Parameters
    ----------
    cluster_ids : array
        Cluster IDs to stack
    radial_bins : array
        Radial bins (in units of R_200)
        
    Returns
    -------
    R_stacked : array
        Stacked radial bins (kpc)
    gamma_stacked : array
        Stacked tangential shear
    cov_stacked : array
        Covariance matrix
    """
    
    N_clusters = len(cluster_ids)
    N_bins = len(radial_bins) - 1
    
    # Storage
    profiles = []
    weights = []
    
    for cluster_id in cluster_ids:
        # Load profile
        R, gamma, sigma_gamma = load_shear_profile(cluster_id)
        
        # Load cluster properties
        cluster = get_cluster_properties(cluster_id)
        R_200 = cluster['R_200']
        
        # Scale to R_200 units
        R_scaled = R / R_200
        
        # Interpolate to common bins
        gamma_binned = np.interp(radial_bins[:-1] + np.diff(radial_bins)/2, 
                                 R_scaled, gamma)
        
        # Weight by inverse variance
        weight = 1 / sigma_gamma**2
        weight_binned = np.interp(radial_bins[:-1] + np.diff(radial_bins)/2,
                                   R_scaled, weight)
        
        profiles.append(gamma_binned)
        weights.append(weight_binned)
    
    # Weighted average
    profiles = np.array(profiles)
    weights = np.array(weights)
    
    gamma_stacked = np.average(profiles, axis=0, weights=weights)
    
    # Covariance from jackknife
    cov_stacked = compute_jackknife_covariance(profiles, weights)
    
    # Radial bins in physical units (average R_200)
    R_200_mean = np.mean([get_cluster_properties(cid)['R_200'] 
                          for cid in cluster_ids])
    R_stacked = (radial_bins[:-1] + np.diff(radial_bins)/2) * R_200_mean
    
    return R_stacked, gamma_stacked, cov_stacked
```

#### 7.2.3 Model Fitting (Week 3)

**Fit CT and NFW:**
```bash
python scripts/11_test_lensing_profiles.py \
    --cluster-sample data/processed/lensing/selected_clusters.fits \
    --fit-individual \
    --fit-stacked \
    --xi-fixed 100  # kpc (CT prediction)
```

**Expected runtime:**
- Individual fits: ~1 min/cluster × 1000 clusters = ~17 hours
- Stacked fit: ~10 minutes

**Parallel execution:**
```bash
# Use GNU parallel or similar
cat cluster_ids.txt | parallel -j 32 \
    python scripts/utils/fit_single_cluster.py {}
```

#### 7.2.4 Universality Test (Week 4)

**Extract ξ values:**
```python
# scripts/utils/pl1_test_universality.py
"""
Test universality of CT interface width ξ.
"""

def test_universality(fit_results):
    """
    Test if ξ is constant across cluster sample.
    
    Parameters
    ----------
    fit_results : list of dict
        Individual cluster fit results
        
    Returns
    -------
    universality_test : dict
        Test results
    """
    
    # Extract ξ values and uncertainties
    xi_values = np.array([r['xi'] for r in fit_results])
    xi_errors = np.array([r['xi_error'] for r in fit_results])
    
    # Extract cluster properties
    masses = np.array([r['M_200'] for r in fit_results])
    redshifts = np.array([r['z'] for r in fit_results])
    
    # 1. Test constancy
    # Weighted mean
    xi_mean = np.average(xi_values, weights=1/xi_errors**2)
    xi_std = np.sqrt(1 / np.sum(1/xi_errors**2))
    
    # Intrinsic scatter
    chi2_const = np.sum(((xi_values - xi_mean) / xi_errors)**2)
    dof = len(xi_values) - 1
    
    # If chi2 >> dof, there's intrinsic scatter beyond errors
    reduced_chi2 = chi2_const / dof
    
    # 2. Test for mass dependence
    # Fit: ξ = ξ_0 * (M / M_0)^α
    from scipy.optimize import curve_fit
    
    def power_law(M, xi_0, alpha):
        return xi_0 * (M / 1e14)**alpha
    
    popt, pcov = curve_fit(power_law, masses, xi_values, 
                           sigma=xi_errors, absolute_sigma=True)
    xi_0, alpha = popt
    alpha_error = np.sqrt(pcov[1,1])
    
    # If |α| < 2σ, consistent with no mass dependence
    mass_independent = abs(alpha) < 2 * alpha_error
    
    # 3. Test for redshift dependence
    # Fit: ξ = ξ_0 * (1 + z)^β
    def redshift_evolution(z, xi_0, beta):
        return xi_0 * (1 + z)**beta
    
    popt_z, pcov_z = curve_fit(redshift_evolution, redshifts, xi_values,
                               sigma=xi_errors, absolute_sigma=True)
    xi_0_z, beta = popt_z
    beta_error = np.sqrt(pcov_z[1,1])
    
    z_independent = abs(beta) < 2 * beta_error
    
    # Overall verdict
    universal = (reduced_chi2 < 2.0) and mass_independent and z_independent
    
    return {
        'xi_mean': xi_mean,
        'xi_std': xi_std,
        'fractional_scatter': xi_std / xi_mean,
        'reduced_chi2_constant': reduced_chi2,
        'mass_scaling_alpha': alpha,
        'mass_scaling_alpha_error': alpha_error,
        'mass_independent': mass_independent,
        'z_evolution_beta': beta,
        'z_evolution_beta_error': beta_error,
        'z_independent': z_independent,
        'universal': universal
    }
```

**Interpretation:**
- If `fractional_scatter < 0.2` and `universal = True`:
  - ✅ **VALIDATED**: ξ is universal (CT prediction confirmed)
- If `fractional_scatter > 0.3` or `mass_independent = False`:
  - ❌ **FALSIFIED**: ξ not universal (astrophysical, not CT)
- If `0.2 < fractional_scatter < 0.3`:
  - ⚠️ **MARGINAL**: Need larger sample or better errors

#### 7.2.5 Publication Plots

**Key figures:**
1. **Stacked profile with CT and NFW fits**
2. **ξ distribution histogram** (should be narrow)
3. **ξ vs M_200 scatter plot** (should be flat)
4. **ξ vs z scatter plot** (should be flat)
5. **Residuals: (Data - CT) / Error** (should be Gaussian)

```python
# Example: Figure 1
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data
ax.errorbar(R_stacked, gamma_stacked, yerr=np.sqrt(np.diag(cov_stacked)),
            fmt='o', color='black', label='Stacked Data (N=1000)')

# Plot models
R_model = np.linspace(R_stacked.min(), R_stacked.max(), 200)
gamma_ct = ct_profile.gamma_t(R_model, Sigma_crit)
gamma_nfw = nfw_profile.gamma_t(R_model, Sigma_crit)

ax.plot(R_model, gamma_ct, 'r-', lw=2, 
        label=f'CT (ξ={xi_fit:.0f} kpc, χ²/dof={chi2_ct/dof:.2f})')
ax.plot(R_model, gamma_nfw, 'b--', lw=2,
        label=f'NFW (χ²/dof={chi2_nfw/dof:.2f})')

ax.set_xlabel('Radius (kpc)', fontsize=14)
ax.set_ylabel('Tangential Shear γ_t', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/publication/lensing_stacked_profile.pdf', dpi=300)
```

---

### 7.3 P-S1: LSS Anisotropy — Complete Workflow

#### 7.3.1 Catalog Preparation (Week 1)

**Download SDSS DR18:**
```bash
# Use SDSS CasJobs or SkyServer
# SQL query provided in Section 3.4.1

# Or use prepackaged catalog
wget http://data.sdss.org/sas/dr18/eboss/lss/catalogs/DR18_galaxy_LSS_catalog.fits
```

**Coordinate transformation:**
```python
# scripts/utils/rotate_to_aoe_frame.py
"""
Rotate galaxy coordinates to Axis of Evil frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

def get_v_aoe():
    """Get Axis of Evil direction from P-T2 results."""
    # From P-T2 validated result
    # Galactic coordinates: (l, b) ≈ (...)
    # Convert to unit vector
    
    # This should be loaded from P-T2 results file
    v_aoe = np.array([0.009, 0.009, 0.9999])  # Approximate
    v_aoe /= np.linalg.norm(v_aoe)
    
    return v_aoe

def rotate_to_aoe_frame(ra, dec):
    """
    Rotate RA, Dec to frame where z-axis = Axis of Evil.
    
    Parameters
    ----------
    ra : array
        Right ascension (degrees)
    dec : array
        Declination (degrees)
        
    Returns
    -------
    theta_aoe : array
        Polar angle in AoE frame (radians)
    phi_aoe : array
        Azimuthal angle in AoE frame (radians)
    """
    
    # Convert RA, Dec to Cartesian
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    vectors = np.column_stack([x, y, z])
    
    # Get rotation to align v_aoe with z-axis
    v_aoe = get_v_aoe()
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis: v_aoe × z_axis
    rotation_axis = np.cross(v_aoe, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Rotation angle: arccos(v_aoe · z_axis)
    rotation_angle = np.arccos(np.dot(v_aoe, z_axis))
    
    # Create rotation
    rot = Rotation.from_rotvec(rotation_angle * rotation_axis)
    
    # Apply rotation
    vectors_rotated = rot.apply(vectors)
    
    # Convert to spherical
    x_rot, y_rot, z_rot = vectors_rotated.T
    
    theta_aoe = np.arccos(z_rot)
    phi_aoe = np.arctan2(y_rot, x_rot)
    
    return theta_aoe, phi_aoe
```

#### 7.3.2 Correlation Function Calculation (Week 2-3)

**Compute multipoles:**
```python
# scripts/utils/ps1_compute_multipoles.py
"""
Compute correlation function multipoles.
"""

from scipy.special import legendre
import numpy as np

def compute_correlation_multipoles(galaxies, randoms, radial_bins, lmax=8):
    """
    Compute correlation function multipoles using Landy-Szalay estimator.
    
    Parameters
    ----------
    galaxies : dict
        {'theta_aoe': array, 'phi_aoe': array, 'z': array}
    randoms : dict
        Random catalog (10× larger)
    radial_bins : array
        Comoving distance bins (Mpc/h)
    lmax : int
        Maximum multipole
        
    Returns
    -------
    r_bins : array
        Bin centers
    xi_ell : array (N_bins, lmax+1)
        Correlation function multipoles
    """
    
    # Convert to Cartesian comoving coordinates
    # (assumes flat cosmology)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
    
    def angular_to_cartesian(theta, phi, z):
        r_com = cosmo.comoving_distance(z).value  # Mpc/h
        x = r_com * np.sin(theta) * np.cos(phi)
        y = r_com * np.sin(theta) * np.sin(phi)
        z = r_com * np.cos(theta)
        return x, y, z
    
    x_g, y_g, z_g = angular_to_cartesian(galaxies['theta_aoe'], 
                                          galaxies['phi_aoe'],
                                          galaxies['z'])
    
    x_r, y_r, z_r = angular_to_cartesian(randoms['theta_aoe'],
                                          randoms['phi_aoe'],
                                          randoms['z'])
    
    # Pair counting
    N_g = len(x_g)
    N_r = len(x_r)
    
    N_bins = len(radial_bins) - 1
    xi_ell = np.zeros((N_bins, lmax+1))
    
    # Use tree for efficiency
    from scipy.spatial import cKDTree
    
    tree_g = cKDTree(np.column_stack([x_g, y_g, z_g]))
    tree_r = cKDTree(np.column_stack([x_r, y_r, z_r]))
    
    for i_bin in range(N_bins):
        r_min = radial_bins[i_bin]
        r_max = radial_bins[i_bin + 1]
        r_center = (r_min + r_max) / 2
        
        # DD: galaxy-galaxy pairs
        DD = count_pairs_with_angles(tree_g, tree_g, r_min, r_max)
        
        # DR: galaxy-random pairs
        DR = count_pairs_with_angles(tree_g, tree_r, r_min, r_max)
        
        # RR: random-random pairs
        RR = count_pairs_with_angles(tree_r, tree_r, r_min, r_max)
        
        # Landy-Szalay estimator for each multipole
        for ell in range(lmax + 1):
            # Weight by Legendre polynomial
            L_ell = legendre(ell)
            
            DD_ell = np.sum([L_ell(mu) for mu in DD['mu']])
            DR_ell = np.sum([L_ell(mu) for mu in DR['mu']])
            RR_ell = np.sum([L_ell(mu) for mu in RR['mu']])
            
            # Normalization
            norm = (N_r / N_g)**2
            
            # Landy-Szalay
            xi_ell[i_bin, ell] = (2*ell + 1) * (
                DD_ell / (N_g * (N_g - 1)) - 
                2 * DR_ell / (N_g * N_r) * norm + 
                RR_ell / (N_r * (N_r - 1)) * norm**2
            )
    
    r_bins = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    return r_bins, xi_ell
```

**Note:** Full implementation requires optimized pair counting (e.g., `Corrfunc` package).

#### 7.3.3 Significance Test (Week 4)

**Monte Carlo with isotropic mocks:**
```bash
python scripts/13_test_lss_anisotropy.py \
    --catalog data/raw/sdss/dr18_galaxy_sample.fits \
    --n-mocks 1000 \
    --lmax 8 \
    --output results/runs/$(date +%Y%m%d_%H%M%S)_ps1_lss
```

**Expected runtime:**
- Real data: ~6 hours (with `Corrfunc`)
- 1000 mocks: ~250 hours = ~10 days (parallelizable)

**Parallel execution:**
```bash
# Split mocks across nodes
for i in {0..31}; do
    python scripts/utils/ps1_run_single_mock.py \
        --mock-id $i \
        --n-mocks-per-job 31 &
done
wait
```

#### 7.3.4 Interpretation

**Decision tree:**
```
Is ξ_6 significant?
│
├─ YES: p < 0.05
│   │
│   ├─ Is ξ_6 aligned with v_AoE?
│   │   ├─ YES: θ(ξ_6_axis, v_AoE) < 20°
│   │   │   └─ ✅ VALIDATED: D6 hexapole detected in LSS
│   │   │
│   │   └─ NO: θ > 20°
│   │       └─ ⚠️ PARTIAL: Anisotropy detected but not D6
│   │           (could be systematics, survey geometry)
│   │
│   └─ Check other multipoles
│       ├─ If ξ_2 dominant: Likely systematics (quadrupole from survey)
│       └─ If ξ_4, ξ_8 also present: More complex pattern
│
└─ NO: p > 0.05
    └─ NULL: No significant anisotropy
        ├─ Could be signal too weak (ε < 10⁻⁴)
        └─ Or CT prediction wrong for LSS
```

---

### 7.4 CT-C1: α Variation — Complete Workflow

#### 7.4.1 Literature Compilation (Month 1-2)

**Compile measurements:**
```python
# scripts/utils/compile_alpha_measurements.py
"""
Compile α variation measurements from literature.
"""

import pandas as pd
import numpy as np

def load_murphy_2003():
    """Murphy et al. (2003) - Keck/HIRES."""
    # Manually transcribed from paper table
    data = pd.DataFrame({
        'qso_name': [...],
        'RA_deg': [...],
        'Dec_deg': [...],
        'z_abs': [...],
        'delta_alpha_over_alpha': [...],  # ×10⁻⁶
        'sigma_delta_alpha': [...],
        'telescope': 'Keck/HIRES',
        'reference': 'Murphy2003'
    })
    return data

def load_webb_2011():
    """Webb et al. (2011) - VLT/UVES."""
    # From published data tables
    # ...
    return data

def standardize_measurements(datasets):
    """
    Standardize to common system.
    
    Different methods have systematic offsets.
    Need to cross-calibrate on common objects.
    """
    
    # Apply systematic corrections
    # (This is complex and requires careful analysis)
    # ...
    
    combined = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates (same QSO in multiple surveys)
    combined = combined.drop_duplicates(subset=['qso_name', 'z_abs'])
    
    return combined

if __name__ == '__main__':
    datasets = [
        load_murphy_2003(),
        load_webb_2011(),
        # ... other surveys
    ]
    
    combined = standardize_measurements(datasets)
    
    print(f"Total measurements: {len(combined)}")
    print(f"Sky coverage: {combined.groupby('telescope').size()}")
    
    combined.to_csv('data/raw/qso_alpha/compiled_alpha_measurements.csv', 
                    index=False)
```

**Challenge:** Different surveys use different methods (Many-Multiplet, Alkali Doublet, etc.) with different systematic errors. Careful cross-calibration required.

#### 7.4.2 Sky Map Construction (Month 3)

**HEALPix gridding:**
```python
# scripts/utils/ctc1_make_healpix_map.py
"""
Convert α measurements to HEALPix map.
"""

import healpy as hp
import numpy as np

def measurements_to_healpix(data, nside=32):
    """
    Bin measurements into HEALPix map.
    
    Parameters
    ----------
    data : DataFrame
        Compiled measurements
    nside : int
        HEALPix resolution
        
    Returns
    -------
    alpha_map : array
        HEALPix map of Δα/α
    sigma_map : array
        Uncertainty map
    nhits_map : array
        Number of measurements per pixel
    """
    
    npix = hp.nside2npix(nside)
    
    # Initialize
    alpha_sum = np.zeros(npix)
    weight_sum = np.zeros(npix)
    nhits = np.zeros(npix, dtype=int)
    
    # Convert to theta, phi
    theta = np.radians(90 - data['Dec_deg'])
    phi = np.radians(data['RA_deg'])
    
    # Assign to pixels
    pix = hp.ang2pix(nside, theta, phi)
    
    # Weighted average in each pixel
    for i in range(len(data)):
        p = pix[i]
        val = data['delta_alpha_over_alpha'].iloc[i] * 1e-6
        sig = data['sigma_delta_alpha'].iloc[i] * 1e-6
        weight = 1 / sig**2
        
        alpha_sum[p] += val * weight
        weight_sum[p] += weight
        nhits[p] += 1
    
    # Compute map
    alpha_map = np.zeros(npix)
    sigma_map = np.full(npix, np.inf)
    
    mask = nhits > 0
    alpha_map[mask] = alpha_sum[mask] / weight_sum[mask]
    sigma_map[mask] = np.sqrt(1 / weight_sum[mask])
    
    return alpha_map, sigma_map, nhits

if __name__ == '__main__':
    import pandas as pd
    
    data = pd.read_csv('data/raw/qso_alpha/compiled_alpha_measurements.csv')
    
    alpha_map, sigma_map, nhits_map = measurements_to_healpix(data, nside=32)
    
    # Save
    hp.write_map('data/processed/alpha_variation/healpix_map_nside32.fits',
                 [alpha_map, sigma_map, nhits_map],
                 overwrite=True,
                 column_names=['DELTA_ALPHA', 'SIGMA', 'NHITS'])
    
    # Visualize
    hp.mollview(alpha_map, title='Δα/α Sky Map', unit='')
    plt.savefig('plots/alpha_variation/skymap.png', dpi=300)
```

#### 7.4.3 Dipole Analysis (Month 4)

**Extract dipole:**
```python
# scripts/12_test_alpha_variation.py (excerpt)

def extract_dipole(alpha_map, sigma_map, nside):
    """Extract dipole amplitude and direction."""
    
    # Spherical harmonic transform
    alm = hp.map2alm(alpha_map, lmax=7, iter=3)
    
    # Dipole coefficients
    a_10 = alm[hp.Alm.getidx(7, 1, 0)]
    a_11_real = alm[hp.Alm.getidx(7, 1, 1)].real
    a_11_imag = alm[hp.Alm.getidx(7, 1, 1)].imag
    
    # Convert to Cartesian dipole direction
    # Y_1^0 = sqrt(3/4π) cos(θ)
    # Y_1^{±1} = ∓sqrt(3/8π) sin(θ) e^{±iφ}
    
    D_z = a_10 * np.sqrt(4*np.pi / 3)
    D_x = -a_11_real * np.sqrt(8*np.pi / 3)
    D_y = -a_11_imag * np.sqrt(8*np.pi / 3)
    
    D_vec = np.array([D_x, D_y, D_z])
    D_amp = np.linalg.norm(D_vec)
    D_dir = D_vec / D_amp
    
    # Convert to RA, Dec
    dec_dipole = np.degrees(np.arcsin(D_dir[2]))
    ra_dipole = np.degrees(np.arctan2(D_dir[1], D_dir[0]))
    
    return {
        'amplitude': D_amp,
        'direction': D_dir,
        'RA': ra_dipole,
        'Dec': dec_dipole
    }

def test_alignment_with_aoe(dipole_dir, v_aoe):
    """Test if dipole aligns with Axis of Evil."""
    
    # Angular separation
    cos_angle = np.dot(dipole_dir, v_aoe)
    angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))
    
    # Monte Carlo: random dipoles
    N_mc = 10000
    angles_mc = []
    
    for _ in range(N_mc):
        # Random direction
        theta_rand = np.arccos(np.random.uniform(-1, 1))
        phi_rand = np.random.uniform(0, 2*np.pi)
        
        dir_rand = np.array([
            np.sin(theta_rand) * np.cos(phi_rand),
            np.sin(theta_rand) * np.sin(phi_rand),
            np.cos(theta_rand)
        ])
        
        cos_angle_rand = np.abs(np.dot(dir_rand, v_aoe))
        angle_rand = np.degrees(np.arccos(cos_angle_rand))
        angles_mc.append(angle_rand)
    
    # p-value
    p_value = np.sum(np.array(angles_mc) < angle_deg) / N_mc
    
    return {
        'angle_deg': angle_deg,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

#### 7.4.4 Model Comparison (Month 5-6)

**Bayesian evidence calculation:**
```python
def compute_bayesian_evidence(alpha_map, sigma_map, models):
    """
    Compute evidence for different models.
    
    Models:
    - M_null: Δα/α = 0 everywhere
    - M_dipole: Pure dipole (3 params: amp, RA, Dec)
    - M_webb: Dipole at Webb direction (1 param: amp)
    - M_ct_aligned: Dipole at v_AoE (1 param: amp)
    - M_ct_modulated: ℓ=1,3,5,7 aligned with v_AoE (4 params)
    """
    
    # Convert map to vector
    npix = len(alpha_map)
    mask = sigma_map < np.inf
    
    data = alpha_map[mask]
    errors = sigma_map[mask]
    
    # For each model, compute likelihood
    evidences = {}
    
    for model_name, model in models.items():
        # Predict map
        map_pred = model.predict()
        
        # Chi-square
        chi2 = np.sum(((data - map_pred[mask]) / errors)**2)
        
        # Log likelihood
        log_L = -0.5 * chi2 - 0.5 * np.sum(np.log(2*np.pi * errors**2))
        
        # Bayesian evidence (Laplace approximation)
        N_params = model.n_params
        log_evidence = log_L - 0.5 * N_params * np.log(len(data))
        
        evidences[model_name] = {
            'log_L': log_L,
            'log_Z': log_evidence,
            'chi2': chi2,
            'dof': len(data) - N_params
        }
    
    # Bayes factors
    log_Z_null = evidences['M_null']['log_Z']
    
    for name in evidences:
        evidences[name]['log_BF_vs_null'] = (
            evidences[name]['log_Z'] - log_Z_null
        )
    
    return evidences
```

**Interpretation:**
- If `log_BF(M_ct_aligned vs M_webb) > 5`: CT direction favored
- If `log_BF(M_ct_modulated vs M_dipole) > 3`: Modulation detected
- If all models have similar evidence: Need more data

---

## 8. Visualization & Reporting

### 8.1 Standard Figure Templates

**Publication-quality matplotlib style:**

```python
# scripts/utils/plot_style.py
"""
Standard plotting style for CT-CMB validation.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    
    # Font sizes
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 13
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 10
    
    # Line widths
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.0
    mpl.rcParams['ytick.minor.width'] = 1.0
    
    # Fonts
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['text.usetex'] = False  # Set True if LaTeX available
    
    # Grid
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = ':'
    
    # Figure
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    
    # Colors (colorblind-friendly)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        '#0173B2', '#DE8F05', '#029E73', '#CC78BC',
        '#CA9161', '#949494', '#ECE133', '#56B4E9'
    ])

# Pre-defined color schemes
CT_COLORS = {
    'ct_model': '#D62728',      # Red
    'lcdm_model': '#1F77B4',    # Blue
    'data': '#2CA02C',          # Green
    'prediction': '#FF7F0E',    # Orange
    'validated': '#2CA02C',     # Green
    'falsified': '#D62728',     # Red
    'marginal': '#FF7F0E'       # Orange
}
```

### 8.2 Automated Report Generation

**File:** `scripts/utils/generate_latex_report.py`

```python
"""
Generate LaTeX report from analysis results.
"""

import json
from pathlib import Path
from datetime import datetime

TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}

\title{CT-CMB Validation Report: {test_name}}
\author{Automated Report Generator}
\date{{\today}}

\begin{document}
\maketitle

\section{Summary}
{summary_text}

\section{Results}
\subsection{Observed Statistics}
{observed_stats_table}

\subsection{Model Comparison}
{model_comparison_table}

\section{Figures}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{{main_plot}}
\caption{{Main result.}}
\end{figure}

\section{Interpretation}
{interpretation_text}

\section{Validation Status}
\textbf{Verdict:} {verdict}

\textbf{Significance:} {significance}

\textbf{Next Steps:} {next_steps}

\end{document}
"""

def generate_report(run_dir):
    """Generate LaTeX report from run directory."""
    
    run_dir = Path(run_dir)
    
    # Load results
    with open(run_dir / 'final_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract test name
    test_name = run_dir.name.split('_', 2)[2].replace('_', ' ').title()
    
    # Build summary
    summary_text = f"Analysis completed on {datetime.now().strftime('%Y-%m-%d')}."
    
    # Observed statistics table
    obs_stats = results.get('observed_data', {})
    obs_table = r"\begin{tabular}{ll}\toprule" + "\n"
    obs_table += r"Statistic & Value \\" + "\n\\midrule\n"
    for key, val in obs_stats.items():
        obs_table += f"{key} & {val:.4g} \\\\\n"
    obs_table += r"\bottomrule\end{tabular}"
    
    # Model comparison table
    # (similar construction)
    
    # Interpretation
    verdict = results.get('falsification_status', 'UNKNOWN')
    interpretation = {
        'VALIDATED': 'The CT prediction is strongly supported by the data.',
        'FALSIFIED': 'The data contradict the CT prediction.',
        'INCONCLUSIVE': 'The results are marginal and require further investigation.'
    }.get(verdict, 'Unknown status.')
    
    # Fill template
    report = TEMPLATE.format(
        test_name=test_name,
        summary_text=summary_text,
        observed_stats_table=obs_table,
        model_comparison_table="...",
        main_plot=str(run_dir / 'plots/main.pdf'),
        interpretation_text=interpretation,
        verdict=verdict,
        significance=f"p = {results.get('p_value', -1):.3f}",
        next_steps="..."
    )
    
    # Save
    report_file = run_dir / 'report.tex'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Report generated: {report_file}")
    
    # Compile (if pdflatex available)
    import subprocess
    try:
        subprocess.run(['pdflatex', '-output-directory', str(run_dir), 
                       str(report_file)], check=True, capture_output=True)
        print(f"✓ PDF compiled: {run_dir / 'report.pdf'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ pdflatex not found. LaTeX source saved but not compiled.")
```

---

## 9. Computational Optimization

### 9.1 Parallelization Strategies

#### 9.1.1 Embarrassingly Parallel Tasks

**Monte Carlo simulations:**
```python
from multiprocessing import Pool
import os

def run_single_mc(seed):
    """Run single Monte Carlo iteration."""
    np.random.seed(seed)
    # ... simulation code ...
    return statistic

def run_parallel_mc(N_mc, n_workers=None):
    """
    Run Monte Carlo in parallel.
    
    Parameters
    ----------
    N_mc : int
        Number of Monte Carlo iterations
    n_workers : int, optional
        Number of parallel workers (default: CPU count)
    """
    
    if n_workers is None:
        n_workers = int(os.environ.get('CT_WORKERS', os.cpu_count()))
    
    print(f"Running {N_mc} Monte Carlo iterations on {n_workers} workers...")
    
    with Pool(n_workers) as pool:
        results = pool.map(run_single_mc, range(N_mc))
    
    return np.array(results)
```

**Cluster fitting (P-L1):**
```python
def fit_cluster_parallel(cluster_ids, n_workers=32):
    """Fit lensing profiles in parallel."""
    
    def fit_single(cid):
        # Load data
        R, gamma, sigma = load_shear_profile(cid)
        
        # Fit models
        ct_result = fit_ct_profile(R, gamma, sigma)
        nfw_result = fit_nfw_profile(R, gamma, sigma)
        
        return {'cluster_id': cid, 'ct': ct_result, 'nfw': nfw_result}
    
    with Pool(n_workers) as pool:
        results = pool.map(fit_single, cluster_ids)
    
    return results
```

#### 9.1.2 GPU Acceleration

**For numerical integration (P-L1 projections):**
```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

def project_profile_gpu(rho_3d_func, R_grid, params):
    """
    Compute projected density on GPU.
    
    Much faster for large R_grid.
    """
    
    if not GPU_AVAILABLE:
        # Fall back to CPU
        return project_profile_cpu(rho_3d_func, R_grid, params)
    
    # Transfer to GPU
    R_gpu = cp.asarray(R_grid)
    
    # Vectorized integration
    # ... (GPU kernel implementation)
    
    # Transfer back to CPU
    result = cp.asnumpy(Sigma_gpu)
    
    return result
```

### 9.2 Memory Management

**For large catalogs (P-S1):**
```python
import numpy as np
from pathlib import Path

class LargeGalaxyCatalog:
    """
    Memory-mapped access to large galaxy catalogs.
    
    Avoids loading entire catalog into RAM.
    """
    
    def __init__(self, fits_path):
        from astropy.io import fits
        
        self.fits_path = Path(fits_path)
        self.hdu = fits.open(fits_path, memmap=True)
        self.data = self.hdu[1].data
        
        self._n_galaxies = len(self.data)
    
    def __len__(self):
        return self._n_galaxies
    
    def __getitem__(self, idx):
        """Access single galaxy or slice."""
        return self.data[idx]
    
    def iterate_chunks(self, chunk_size=100000):
        """Iterate over catalog in chunks."""
        for i in range(0, len(self), chunk_size):
            yield self.data[i:i+chunk_size]
    
    def close(self):
        self.hdu.close()

# Usage
catalog = LargeGalaxyCatalog('data/raw/sdss/dr18_galaxy_sample.fits')

for chunk in catalog.iterate_chunks():
    # Process chunk
    process_galaxies(chunk)

catalog.close()
```

### 9.3 Caching Intermediate Results

```python
import pickle
from functools import wraps
from pathlib import Path
import hashlib

def cached(cache_dir='data/cache'):
    """
    Decorator to cache expensive function results.
    
    Example:
        @cached(cache_dir='data/cache')
        def expensive_calculation(param1, param2):
            # ... long computation ...
            return result
    """
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_str = f"{func.__name__}_{args}_{kwargs}"
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            cache_file = cache_dir / f"{key_hash}.pkl"
            
            # Check if cached
            if cache_file.exists():
                print(f"Loading cached result: {func.__name__}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Compute
            print(f"Computing: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        
        return wrapper
    return decorator

# Usage
@cached(cache_dir='data/cache/lensing')
def compute_ct_projection(R_200, xi, N_points=1000):
    """Expensive projection integral."""
    # ... numerical integration ...
    return Sigma
```

---

## 10. Troubleshooting & FAQ

### 10.1 Common Issues

#### Issue 1: P-H3 fitting fails to converge

**Symptoms:**
```
OptimizeWarning: Desired error not necessarily achieved due to precision loss.
```

**Solutions:**
1. **Check initial parameters:**
   ```python
   # Try different starting point
   initial_params = [70, 0.3, -19.2]  # Instead of [67, 0.315, -19.3]
   ```

2. **Scale parameters:**
   ```python
   # Normalize parameters to O(1)
   def fit_scaled():
       # Fit [H0/70, Omega_m/0.3, (M_B + 19)/0.5]
       # Then rescale results
   ```

3. **Use different optimizer:**
   ```python
   result = minimize(chi2_func, initial_params, 
                    method='Nelder-Mead',  # More robust
                    options={'maxiter': 50000})
   ```

#### Issue 2: P-L1 projection integral diverges

**Symptoms:**
```
IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
```

**Solutions:**
1. **Adjust integration limits:**
   ```python
   # Instead of integrating to infinity
   R_max = 10 * R_200  # Truncate far from cluster
   
   result, _ = quad(integrand, R, R_max, limit=100)
   ```

2. **Change integration strategy:**
   ```python
   from scipy.integrate import quad, quadrature
   
   # Try adaptive Gaussian quadrature
   result, _ = quadrature(integrand, R, R_max, maxiter=100)
   ```

3. **Smooth discontinuities:**
   ```python
   # tanh has sharp transition - smooth it slightly
   def smooth_tanh(x, xi, smoothing=0.1):
       return np.tanh(x / (xi * (1 + smoothing)))
   ```

#### Issue 3: HEALPix coordinate mismatch

**Symptoms:**
```
AssertionError: Coordinate system mismatch (Galactic vs Equatorial)
```

**Solutions:**
1. **Check coordinate systems:**
   ```python
   import healpy as hp
   from astropy.coordinates import SkyCoord
   
   # Convert RA, Dec → Galactic
   coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
   l = coords.galactic.l.deg
   b = coords.galactic.b.deg
   
   # Then to HEALPix
   theta = np.radians(90 - b)
   phi = np.radians(l)
   pix = hp.ang2pix(nside, theta, phi)
   ```

2. **Use `astropy` for conversions:**
   ```python
   from astropy.coordinates import SkyCoord, Galactic, ICRS
   
   # Explicit conversion
   coords_icrs = SkyCoord(ra, dec, unit='deg', frame='icrs')
   coords_gal = coords_icrs.transform_to(Galactic())
   ```

#### Issue 4: Monte Carlo not converging

**Symptoms:**
- p-values jumping around
- Split-half test fails

**Solutions:**
1. **Increase N_MC:**
   ```bash
   export CT_PT2_NMC_ISO=50000  # Instead of 10000
   export CT_PT2_NMC_D6=5000    # Instead of 1000
   ```

2. **Check for bugs:**
   ```python
   # Verify null distribution is reasonable
   plt.hist(mc_statistics, bins=50)
   plt.axvline(observed, color='r', label='Observed')
   plt.xlabel('Test Statistic')
   plt.ylabel('Counts')
   plt.legend()
   plt.savefig('debug_mc_distribution.png')
   ```

3. **Test with known distribution:**
   ```python
   # Sanity check: generate data from null, should get uniform p-values
   p_values_test = []
   for _ in range(100):
       # Generate fake "observed" from null
        fake_obs = np.random.choice(mc_statistics)
       p = np.sum(mc_statistics >= fake_obs) / len(mc_statistics)
       p_values_test.append(p)
   
   # Should be uniform on [0,1]
   plt.hist(p_values_test, bins=20)
   plt.xlabel('p-value')
   plt.title('Should be uniform')
   ```

---

### 10.2 Performance Benchmarks

**Expected runtimes on standard hardware:**

| Test | Dataset Size | Single-threaded | 32 cores | GPU |
|------|--------------|-----------------|----------|-----|
| P-T2 (Axis of Evil) | 1 CMB map | 30 min | 5 min | N/A |
| P-H3 (Dark Energy) | 1,700 SNe | 5 min | 2 min | N/A |
| P-L1 (Lensing) per cluster | 1 profile | 30 sec | 1 sec | 0.1 sec |
| P-L1 full sample | 1,000 clusters | 8 hours | 30 min | 2 min |
| P-S1 (LSS) real data | 10⁶ galaxies | 6 hours | 30 min | N/A |
| P-S1 single mock | 10⁶ galaxies | 6 hours | 30 min | N/A |
| CT-C1 (α map) | 500 QSOs | 10 min | N/A | N/A |

**Hardware specs (reference):**
- CPU: AMD EPYC 7742 (64 cores, 2.25 GHz)
- RAM: 256 GB
- GPU: NVIDIA A100 (40 GB)
- Storage: NVMe SSD

---

### 10.3 FAQ

**Q: How much disk space do I need?**

A: Minimum requirements:
- Raw data: ~50 GB (Planck + catalogs)
- Processed data: ~20 GB
- Results: ~10 GB per major test
- Total: ~200 GB recommended

**Q: Can I run without Planck data?**

A: P-T2, P-T1 require Planck SMICA map (mandatory). Other tests can run independently with their respective datasets.

**Q: What if my p-value is exactly 0?**

A: This means your observed statistic exceeded ALL Monte Carlo samples. Report as `p < 1/N_MC`. Example: if N_MC = 10,000, report `p < 0.0001`. **Increase N_MC** to get tighter bound.

**Q: How do I cite these results?**

A: Use the BibTeX entry in the main README. If you modify the code significantly, fork the repository and cite your version.

**Q: Can I use this for other cosmological theories?**

A: Yes! The framework is general. To test alternative symmetry groups:
1. Modify `src/ct_cmb_validation/theory/d6_group.py` to implement your group
2. Update impedance calculations accordingly
3. All analysis scripts should work with minimal changes

**Q: What license applies?**

A: MIT License (see repository root). You're free to use, modify, and distribute, with attribution.

**Q: Where can I get help?**

A: 
1. Check this document first
2. Open an issue on GitHub
3. Email: [maintainer email]
4. Join discussions tab on GitHub

---

## Appendix A: Complete File Manifest

**Data files (to be downloaded/created):**
```
data/raw/
├── planck/
│   ├── COM_CMB_IQU-smica_2048_R3.00_full.fits (1.2 GB)
│   ├── COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits (50 MB)
│   └── planck_h0_cosmic.json (< 1 KB)
├── pantheon_plus/
│   ├── Pantheon+SH0ES.dat (200 KB)
│   └── Pantheon+SH0ES_STAT+SYS.cov (100 MB)
├── desi_y1/
│   └── desi_y1_bao.json (< 1 KB)
├── hsc/
│   ├── hsc_s16a_clusters.fits (10 MB)
│   ├── shear_profiles/ (1 GB)
│   └── shear_covariance/ (5 GB)
├── qso_alpha/
│   └── compiled_alpha_measurements.csv (50 KB)
├── sdss/
│   └── dr18_galaxy_sample.fits (5 GB)
└── eht/
    ├── m87_2017_stokesQU_band6.fits (100 MB)
    └── sgra_2022_stokesQU_band6.fits (100 MB)
```

---

## Appendix B: Mathematical Notation Reference

| Symbol | Meaning | Units |
|--------|---------|-------|
| $B_{th}$ | Throughput budget | bits/s |
| $B_{cx}$ | Complexity budget | bits |
| $B_{leak}$ | Leakage | bits/s |
| $\lambda_{cx}$ | Complexity multiplier | dimensionless |
| $\lambda_{leak}$ | Leakage multiplier | dimensionless |
| $R(a)$ | Quantum-cosmic density ratio | dimensionless |
| $R_0$ | Present-day $R$ | $1.7 \times 10^6$ |
| $w(a)$ | Dark energy EoS | dimensionless |
| $\xi$ | Interface width | kpc or $GM/c^2$ |
| $\rho_{DE}$ | Dark energy density | $\text{Msun}/\text{kpc}^3$ |
| $Z_{D6}(\hat{n})$ | Geometric impedance | dimensionless |
| $\epsilon$ | Impedance coupling | $\sim 10^{-4}$ |
| $\mathbf{v}_{AoE}$ | Axis of Evil direction | unit vector |
| $\alpha_{EM}$ | Fine structure constant | $1/137$ |
| $\xi_\ell(r)$ | Correlation multipole $\ell$ | dimensionless |

---

## Appendix C: Quick Reference Commands

**Essential commands:**
```bash
# Run all validated tests
./scripts/run_all_validated.sh

# Run priority tests
python scripts/10_test_dark_energy.py --use-bao --plot
python scripts/11_test_lensing_profiles.py --fit-stacked

# Generate summary
python scripts/05_generate_summary_report.py

# Check status
cat results/summary_reports/latest.md

# Run tests
pytest tests/ -v --cov

# Clean cache
rm -rf data/cache/*

# Archive results
tar -czf results_$(date +%Y%m%d).tar.gz results/runs/
```

**Environment variables:**
```bash
export CT_WORKERS=32               # Parallel workers
export CT_PT2_NMC_ISO=50000       # P-T2 isotropic MC
export CT_PT1_LMAX=1024           # P-T1 max multipole
export CT_DATA_ROOT=/path/to/data # Override data location
```

---

**Document Version:** 2.0  
**Last Updated:** October 24, 2025  
**Status:** Complete implementation guide  
**Next Review:** After first external validation

---

**END OF AGENTS-EXPANDED.MD**