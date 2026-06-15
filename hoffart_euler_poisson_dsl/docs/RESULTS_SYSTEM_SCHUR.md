# Hoffart diocotron: first quantitative measurement on the `system-schur` path

First growth-rate table produced on the path that is faithful to the paper
(arXiv:2510.11808, section 5.3): uniform finite volumes, Strang(SSPRK3 + CondensedSchur,
theta=0.5), electrostatic/Lorentz source condensed by Schur, paper's initial drift velocity.
Status (T3, June 2026): after correcting the measurement (paper windows mapped into sim time
`t_sim = 2pi/rhobar * t_paper` plus reporting `gamma_paper = gamma_raw_sim * 2pi/rhobar`), **the full
cartesian system-schur reproduces the paper to -9.1% (l=3), -1.9% (l=4), +0.04% (l=5)**, section 9. The
"structural deficit -95%" was a metrology artifact (paper windows applied to the raw simulation time,
the transient). Cartesian FV reproduction is established with a documented residual gap; partial metrology
(the 2pi is exact/mode-independent, the residual ~0-9% is grid/resolution/window; l=5 is window-sensitive,
its +0.04% is partly fortuitous). n=96; verified by a 4-lens adversarial workflow.

## Setup
- Engine: `system-schur` (uniform System, single-rank), `square` geometry (full cartesian;
  the adc_cpp verdict records that the cut-cell has no effect on the rate).
- Time scheme: `adc.Strang(hyperbolic=adc.Explicit(method="ssprk3"),
  source=adc.CondensedSchur(theta=0.5, alpha=alpha))`, the paper's symmetric second-order splitting
  plus low-dissipation RK3 (the production path accepts ssprk3 via adc_cpp PR #230).
- Spatial: WENO5-Z + Rusanov, conservative variables. `dt = 1e-3`. Cold limit (`theta_p=0`).
- Paper parameters: `R=16, r0=6, r1=8, rho_max=1, rho_min=1e-6, beta=1e6, delta=0.1`,
  `alpha = omega = beta^2 = 1e12`.
- Observable: `|c_l(t)|` = amplitude of the azimuthal Fourier coefficient `l` of `phi` on the
  circle `r=r0=6`; rate = linear regression of `log|c_l|` over the paper's VERBATIM window.
- SHA: `adc_cpp` `06e3b90` (branch `feat/ssprk3-production-path`, PR #230);
  `adc_cases` `a50b539` (branch `feat/hoffart-strang-fidelity`, PR #21). Raw normalization
  (no `2pi/rhobar` factor: full model, not the reduced ExB path).

## 1. Resolution scan (l=3, window [0.40, 0.70])

| n | 64 | 96 | 128 | 192 |
|---|---|---|---|---|
| measured gamma_3 | 0.0198 | 0.0270 | 0.0321 | 0.0351 |

`gamma_3` converges toward ~0.035 (paper 0.772): it rises slightly with resolution
(numerical diffusion decreasing) but **plateaus ~22x below the paper**. This is not
an under-resolution problem.

## 2. Full table (n=192, paper windows)

| l | fit window | measured gamma | paper gamma | error |
|---|---|---|---|---|
| 3 | [0.40, 0.70] | 0.0351 | 0.772 | -95.5% |
| 4 | [0.60, 0.75] | 0.0329 | 0.911 | -96.4% |
| 5 | [1.15, 1.35] | 0.1153 | 0.683 | -83.1% |

All modes show a deficit of ~85 to 96%: the measured rates are an order of
magnitude too small.

## 3. Trajectory diagnostic (UNsaturated rate, delayed onset)

Growth factor `|c_3(t)| / |c_3(0)|` and local rate in a sliding window (l=3, n=128,
up to t=3.0):

| t | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| `|c_3|/|c_3(0)|` | 1.01 | 1.03 | 1.06 | 1.11 | 1.17 | 1.24 |
| local rate `d log|c_3|/dt` | 0.029 | 0.058 | 0.081 | 0.097 | 0.108 | -- |

The local rate increases steadily (0.03 -> 0.11) but stays **~7x below 0.772 at t=2.5**,
whereas the paper already expects 0.77 in [0.40, 0.70]. The growth is not a
simple window shift: the onset of the instability is strongly delayed and damped in the
early phase, then accelerates gradually without reaching the paper rate.

## Verdict

The deficit is structural, not an artifact of:
- **resolution**: gamma_3 converges toward ~0.035 (does not tend toward 0.77 as n grows); agrees
  with the repo's GH200 verdict (`adc_cpp docs/HOFFART_GEOMETRY_VERDICT.md`, #236: -95% at n=256 and
  n=384, plateau ~0.037 independent of resolution). This laptop measurement therefore reproduces the GH200 result.
- **geometry**: #236 records that square == staircase == cutcell give the same rate (cut-cell has no effect);
- **window / timing**: even in a late window [1.0, 1.4] or sliding up to t=2.5, the rate
  stays ~10x below the paper.

Causes RULED OUT by this work (new data vs #236, which tested neither Gauss nor temperature):
1. ~~**Gauss policy (R0)**~~: RULED OUT (section 4: `evolve ~= restart`).
2. ~~**Cold limit**~~ (`theta_p`): RULED OUT: temperature scan (l=3, n=128) `theta_p=0 -> 0.0321`,
   `0.25 -> 0.0290`, `1.0 -> 0.0280`: adding pressure makes it slightly worse, recovers nothing.

> **SUPERSEDED** (sections 6-8): the candidate "spatial over-damping / Rusanov dissipation" below
> is RULED OUT (section 6: HLL == Rusanov); the real cause is the fit WINDOW plus the `2pi = T_d`
> (section 8). Block kept for history.

Remaining cause: **over-damping of the spatial operator** on the cartesian path (not a
non-positivity, an important distinction vs the polar path). Full diagnostic:
`adc_cpp docs/HOFFART_SPATIAL_DIAGNOSTICS.md` (#238, workflow + adversarial review).

Correction (vs the previous version of this doc that spoke of a "non-positive reconstruction"): on the
cartesian path, min(rho) stays positive throughout the run (measured: 6.7e-7 at the 1e-6 floor, never
negative, never NaN). The cartesian does not diverge, it is over-damped. The non-positivity and blow-up
belong to the polar path (`IsothermalFluxPolar`, `1/r` metric, `1/rho` source), not here.
Consequence (proof in #238 section 3): a positivity fix (floor / Zhang-Shu) is
inert on the cartesian rate (the cells are already positive; Zhang-Shu only bites on the background
with no signal). The real over-damping candidate = Rusanov dissipation `~ alpha*(U_R-U_L)`
proportional to the 1e6 jump at the ring contact (the numerical flux, not the WENO5-Z reconstruction, which
does not collapse). Testing a less dissipative flux at 3-var = C++ work (`hll` not exposed by the
DSL; hllc/roe require a pressure that is absent in the cold isothermal case). The reduced scalar ExB model
reproduces the target (+0.2% l=4) because it has neither a moment reconstruction nor a Rusanov on the moment.
Metrology caveat (closed in section 8): a factor 2pi (= T_d) is omitted on the
cartesian-Schur path (`NORMALIZATION.md`), and the paper window is applied in simulation time
(transient). These two factors (partial metrology, not pure) explain most of the deficit;
a residual ~20% of cart vs polar grid remains, see `T2_NORMALIZATION_AUDIT.md`.

## 4. R0 experiment: GaussPolicy `restart` vs `evolve` -> **R0 RULED OUT**

The design's R0 finding (`adc_cpp docs/AMR_CONDENSED_SCHUR_DESIGN.md`, called a "decisive fact" and a
gate for all of Phase C) postulates that the head-of-step `solve_fields` re-solves Gauss
(`-Delta phi = alpha rho`) and overwrites the `phi` evolved by the Schur stage, killing the paper's
restart-free `-Delta phi` dynamics. We implemented the `System.set_gauss_policy` mechanism:
- `restart` (default): re-solves Gauss at each step (historical, bit-identical);
- `evolve`: after `phi^0`, `solve_fields` no longer re-solves the Poisson problem, the Schur stage evolves
  `phi` in-place in `ell_phi()`, reproducing the paper's `-Delta phi` evolution without restart.

Measurement (n=128, paper windows; cf. `adc_cpp` GaussPolicy PR):

| l | restart gamma | evolve gamma | paper | evolve/restart |
|---|---|---|---|---|
| 3 | 0.0321 | 0.0357 | 0.772 | 1.11x |
| 4 | ~0 (-0.005) | ~0 (-0.008) | 0.911 | -- |
| 5 | 0.1070 | 0.1091 | 0.683 | 1.02x |

**R0 verdict: RULED OUT.** `evolve` only raises the rate by 1.0 to 1.6x across all modes, staying
~10-20x below the paper. The discrete Gauss constraint is approximately conserved by the
transport, so re-imposing it (`restart`) is nearly a no-op vs the `-Delta phi` evolution. The
structural deficit is not the Gauss policy. Consequence: gating Phase C
(Schur-on-AMR) on R0 was misdirected, AMR-Schur will not correct the rate. Note: `restart`
is bit-identical to the baseline without GaussPolicy (gamma_3=0.0321 identically) -> NO-DEFAULT-CHANGE.

R0 verdict: RULED OUT (does not correct the rate).

## 5. Robustness scans: contrast and beta -> two more causes RULED OUT

Two additional scans (l=3, n=128, Strang+Schur) further isolate the lock. Read the trend,
not the absolute value (the 0.772 target only holds for the paper parameters).

**Contrast** (rho_min varies, rho_max=1):

| rho_min | contrast | gamma_3 | min(rho) over the run |
|---|---|---|---|
| 1e-6 | 1e6 | 0.0321 | 6.7e-7 (POSITIVE) |
| 1e-4 | 1e4 | 0.0321 | 6.7e-5 |
| 1e-2 | 1e2 | 0.0300 | 6.7e-3 |
| 1e-1 | 1e1 | 0.0214 | 6.3e-2 |

gamma_3 does not rise when the contrast drops (flat, even lower), and min(rho) stays positive.
Caveat (review #238): this scan is confounded (raising rho_min also changes the background charge
`alpha*rho_min`), but the flat + positive result confirms: **no cartesian non-positivity**.

**Beta** (omega=alpha=beta^2; `w=theta*dt*omega`, the Lorentz determinant is `1+w^2`):

| beta | omega | w^2 | gamma_3 |
|---|---|---|---|
| 1e2 | 1e4 | 25 | 0.0321 |
| 1e3 | 1e6 | 2.5e5 | 0.0320 |
| 1e4 | 1e8 | 2.5e9 | 0.0321 |
| 1e6 | 1e12 | 2.5e17 | 0.0321 |

gamma_3 is **exactly flat** over 4 orders of magnitude of omega (w^2 from 25 to 2.5e17, the latter at the
edge of float64 precision). -> the stiffness / omega / precision of the Lorentz eliminator is RULED OUT
as a cause. The deficit is invariant to the two extreme parameters of the problem (contrast and omega).

## Summary of RULED OUT causes (this session + #236)

| cause | verdict | proof |
|---|---|---|
| resolution | RULED OUT | converges n=64->192->256/384 (#236) |
| boundary geometry (cut-cell) | RULED OUT | square==staircase==cutcell (#236) |
| time scheme / dt | RULED OUT | dt-sweep GH200 (#236); Strang/ssprk3 delivered |
| Gauss policy (R0) | RULED OUT | evolve~=restart (section 4) |
| cold limit (temperature) | RULED OUT | theta_p scan worsens (section 3) |
| **density contrast** | **RULED OUT** | gamma_3 flat, min(rho)>0 (section 5) |
| **stiffness / omega / precision** | **RULED OUT** | gamma_3 flat from w^2=25 to 2.5e17 (section 5) |
| non-positivity (cartesian) | RULED OUT | min(rho)>0, no NaN (section 5) |
| **flux dissipation (Rusanov)** | **RULED OUT** | HLL ~= Rusanov (section 6, adc_cpp #239) |

## 6. HLL vs Rusanov test -> flux dissipation RULED OUT

HLL (Harten-Lax-van Leer, 2 waves, less diffusive than Rusanov) was exposed for the 3-var
isothermal model (adc_cpp #239: `riemann="hll"`, without requiring a pressure, gated on `model.wave_speeds`).
Cartesian system-schur measurement (n=128, Strang+Schur, paper windows):

| | rusanov | hll |
|---|---|---|
| l=3 cold (theta_p=0) | 0.0321 | 0.0316 |
| l=3 hot (theta_p=0.5) | 0.0285 | 0.0290 |

**HLL ~= Rusanov** (within ~2%, in both directions). The "Rusanov dissipation at the contact"
candidate (hypothesis #1 of playbook #238) is therefore RULED OUT: reducing the flux dissipation does not
recover the rate. The ~0.032 plateau is invariant to the flux as well as to contrast, beta, Gauss policy,
and temperature, an invariance that points to a cause not local to the flux/reconstruction.

## Remaining lock (after 9 causes ruled out)

> **RESOLVED** since: suspect 1 (Schur coupling) and suspect 3 (full vs reduced structure) are RULED OUT
> by section 7 (full == reduced); suspect 2 (2pi normalization) is closed by 7ter plus section 8
> (the residual "3-4x below 0.772" is the fit WINDOW, factor 3.20 at l=3, NOT a missing factor).
> Block kept for history.

The cartesian deficit is neither temporal, nor geometric, nor Gauss/R0, nor temperature, nor contrast, nor
stiffness/omega, nor non-positivity, nor flux dissipation. Remaining suspects (the "HLL changes
nothing" branch of the user plan):
1. **Schur coupling**: the way the condensed source stage reconstructs/applies the ExB drift
   (vs the reduced ExB that advects rho directly); the full transports a compressible moment re-derived
   at each step, the reduced does not.
2. **Observable / 2pi normalization**: metrology caveat (`NORMALIZATION.md`). Partial at best:
   even x2pi (~6.28), 0.035 -> 0.22, still 3-4x below 0.772. Does not close the factor on its own.
3. **Full vs reduced structure**: the scalar ExB reproduces the target (+0.2%), the full Euler-Poisson
   +moment+Schur gives ~0.032 no matter what -> the difference is in the moment/Schur/drift chain.

## 7. FULL vs REDUCED ExB on the SAME cartesian setup -> model structure RULED OUT (10th cause)

Decisive test: on the same cartesian (same ring IC, same observable |c_l(phi)| on r0, same n/dt/
window), compare the full (rho, m_x, m_y + Strang + CondensedSchur) to the scalar reduced ExB (n advected
by the drift v=(-d_y phi/omega, d_x phi/omega), phi=Gauss(alpha n); no moment, no Schur):

| l | full (rho,m,Schur) | scalar reduced ExB | reduced/full |
|---|---|---|---|
| 3 | 0.0321 | 0.0309 | 1.0x |
| 4 | -0.0048 | -0.0036 | 0.8x |

**The cartesian reduced ExB gives the same ~0.032 as the full.** The moment/Schur/drift chain of the full
is therefore not the cause (model structure RULED OUT, 10th cause). The deficit is common to the simplest
scalar ExB on a square grid.

### Final decomposition of the cartesian deficit (CORRECTED, see 7ter and section 8)
> **OBSOLETE**: the reading "cartesian vs polar geometry / FUNDAMENTAL limitation of cartesian FV"
> below is **SUPERSEDED** by 7ter (reversal) and section 8 (T2 audit). Kept for
> history. The deficit is NOT a fundamental geometric limitation; it decomposes into
> DIMENSIONAL factors (fit window + `2pi = T_d`), cf. `T2_NORMALIZATION_AUDIT.md`.
- **2pi normalization** (`diag/diag_polar_omega.py:35`: rhobar=rho_max=1 -> factor = 2pi ~= 6.28
  exactly, no more). Raw cartesian 0.032 x 2pi = 0.20 -> still ~3.8x below 0.772.
- ~~**Cartesian vs polar geometry**: ... the square grid does not capture the azimuthal dynamics ...
  fundamental limitation of cartesian FV.~~ RETRACTED (7ter: the same ExB on a cartesian grid in
  normalized units x2pi gives 0.64-0.72, the same order as the polar). The "~3.8x residual" above
  is the fit window, not the geometry, quantified in section 8 (window ratio 3.20 for l=3).

### 7bis. Im/Re ratio local to r0: NOT DECISIVE (warning)
Temptation: the scale-invariant ratio `gamma_raw/Omega_raw` (cart 0.06/-0.01/0.15). **This ratio local to
r0 is not reliable** (documented: `NORMALIZATION.md`, `diag_polar_omega.py`): at r0 there is no charge
enclosed in [r0,r1] -> no rigid-body rotation -> Omega_raw(r0)~0 and the ratio blows up. Proof:
the validated polar has a measured ratio 7.40/7.94/2.14 while the analytic is 0.37/0.33/0.20, and yet it
reproduces l=4. So a measured ratio far from the analytic does not prove a mode distortion.
Do not draw a verdict from it (a previous verdict "Case C / 6x modal distortion" based on this ratio was
an overclaim, retracted). The right test is below (7ter).

### 7ter. SAME reduced ExB cart vs polar (NORMALIZED units) -> REVERSAL: the cartesian REPRODUCES
Decisive test (user plan T1): the same reduced ExB `Scalar + ExB(B0=1) + ChargeDensity(charge=1)` (normalized
units, alpha=1, as in `diag_polar_omega.py`) on cartesian vs polar, same ring IC, same
observable c_l(phi) at r0, same window, same global normalization 2pi/rhobar (rhobar=1 -> x2pi). n=128.

| l | polar g_raw (x2pi) | cart g_raw (x2pi) | paper |
|---|---|---|---|
| 3 | 0.155 (0.97) | **0.101 (0.64)** | 0.772 |
| 4 | 0.143 (0.90) | **0.114 (0.72)** | 0.911 |
| 5 | 0.072 (0.45) | **0.114 (0.72)** | 0.683 |

**The reduced cartesian x2pi gives 0.64-0.72, within ~20% of the paper, the same order as the polar.** Ring-smoothing
(eps=0 top-hat 0.719, eps=1 0.762, eps=2 0.716, eps=4 0.455): the ring edge has a minor effect.

**Reversal of the geometry verdict.** The square grid does not fundamentally damp the diocotron mode
(it reproduces it to ~20% in normalized units, like the polar). The -95% deficit of the
hoffart `system-schur` path (gamma_raw=0.032, paper windows, alpha=1e12) is not the cartesian geometry.
It is the normalization and units of the hoffart run: alpha=1e12 (vs alpha=1 normalized) + the
global 2pi factor not applied (results.py forbids it for the full model) + the time-unit mapping.
-> partial metrology (not pure): the `2pi` (= T_d) and the fit window are recoverable, but
a residual ~20% remains (cart 0.72 vs polar 0.90 x2pi) which is a real grid difference,
not a fundamental geometric limitation, and not a mere cosmetic factor either.

### CONCLUSION (corrected): cartesian reproduction is reachable, the lock is the run normalization
The hoffart `system-schur` deficit (alpha=1e12) is not: Schur/moment (7), Rusanov/HLL (6), Gauss (4),
Strang/time, contrast, beta/omega, temperature, non-positivity, nor the cartesian geometry (7ter: the
same reduced ExB cartesian in normalized units reproduces to ~20%). The remaining lock = the
**normalization/units of the hoffart run**. This is partial metrology (not pure): normalization
factors recoverable to ~80%, plus a residual ~20% of grid (cart vs polar).

The four numbers to keep in mind (l=3 unless noted):

| number | value | what it is |
|---|---|---|
| raw full system-schur (paper window) | **0.032** | full hoffart run, alpha=1e12 (sections 1-2) |
| raw reduced cart (ESTABLISHED window) | **~0.10** | same velocity field (alpha/omega=1), window [3,12] (7ter, section 8) |
| reduced cart **x2pi** | **0.64-0.72** | + global factor T_d=2pi (7ter) |
| reduced polar **x2pi** | **~0.90** | validated path, ~exact at l=4 (NORMALIZATION.md) |

The transition raw full `0.032` -> raw reduced cart `0.10` is the fit window (factor 3.20 at l=3,
section 8); the transition `0.10` -> `0.64` is the `2pi = T_d`; the rest `0.64`->`0.90` (~20%) is the
cart vs polar grid. T2 (normalization audit): done, see `T2_NORMALIZATION_AUDIT.md` and
section 8 below.

Established reproduction path: reduced ExB (polar or cartesian) + 2pi/rhobar reproduces to ~20%. The
full cartesian model should follow once its window/normalization are aligned. (The full polar still
diverges, non-positivity, PATH 1 #236, separate work.)

## 8. T2 normalization audit -> the "3x residual" IS the fit window

Full audit: `T2_NORMALIZATION_AUDIT.md` + `diag/diag_normalization_audit.py`. Key results:

**Dimensional key.** `alpha = beta^2/rho_max` and `omega = beta^2` simplify in the drift
velocity: `v = (alpha/omega) grad(phi~) = grad(phi~)` with `alpha/omega = 1/rho_max = 1`. The full run
therefore advects rho with exactly the field of the normalized reduced ExB. (Cross-check: normalized reduced
ExB in paper window l=3 -> `gamma_raw=0.0312` == full system-schur `0.0321`, section 1.)

**Scales** (rho_max=1): `omega_c=|Omega|=beta^2`; `omega_d=rho_max*alpha/|Omega|=1` (diocotron,
O(1)); `T_d=2pi/omega_d=2pi` (== the repo's 2pi factor).

**Scaling candidates (derived ahead of time, applied to the established g_raw l=4=0.1135):**

| candidate | formula | justification | value |
|---|---|---|---|
| c1 | `g_raw * 2pi` | sim time -> paper via T_d | **0.7132** |
| c2 | `g_raw * 2pi * (alpha/omega)` | alpha/omega=1 -> == c1 | 0.7132 |
| c3 | `g_raw / omega_d` | omega_d=1 -> no-op | 0.1135 |
| c4 | `g_raw * T_d` | T_d=2pi -> == c1 | 0.7132 |

They all collapse onto `g_raw*2pi`. No dimensional ~3 factor exists: the "3x residual" is the
window. Same run, two windows (n=128):

| l | paper window (sim) | g_raw | established window [3,12] | g_raw | **ratio** |
|---|---|---|---|---|---|
| 3 | [0.40,0.70] | 0.0312 | [3.0,12.0] | 0.0998 | **3.20** |
| 4 | [0.60,0.75] | 0.0943 | [3.0,12.0] | 0.1135 | 1.20 |
| 5 | [1.15,1.35] | 0.1056 | [3.0,12.0] | 0.1137 | 1.08 |

`run.py:fit_growth` masks the paper window in sim time (transient, ramping rate cf. section 3),
whereas `paper_time = T_d * sim_time`. The l=3 window is the earliest -> largest window
factor (3.20) -> most severe l=3 deficit. l=3 decomposition: `0.0312 (window 3.20x) ->
0.0998 (T_d=2pi 6.28x) -> 0.627 (cart/polar grid 1.23x) -> 0.772`. Product `3.20*6.28*1.23=24.7`
== observed deficit `0.772/0.0312`. **The decomposition closes exactly.**

## 9. T3: paper-faithful measurement (code): the full system-schur reproduces to <10%

Section 8 establishes the diagnostic; T3 wires it into the code (`run.py` + `results.py`). Two helpers
(`paper_to_sim_time_window`, `gamma_to_paper_units`), `fit_growth` now fits the paper window
mapped into sim time (`t_sim = 2pi/rhobar * t_paper`), and each record carries both
`gamma_raw_sim` and `gamma_paper_units = gamma_raw_sim * 2pi/rhobar`. The premise "the full's raw
slope is directly comparable, no 2pi factor" (old `results.py` / adc_cpp `HOFFART_FIDELITY.md`) is
retracted: the 2pi applies to the full too (`alpha/|Omega| = 1/rho_max = 1`).

Table measured by the real `run.py` (FULL system-schur, Strang ssprk3 + CondensedSchur, n=96, dt=2e-3,
t_end=10; rhobar=rho_max=1 -> 2pi factor):

| l | paper window (T_d) | MAPPED sim window | `gamma_raw_sim` | `gamma_paper_units` (x2pi) | paper | error | recall: x2pi established window [3,9] |
|---|---|---|---|---|---|---|---|
| 3 | [0.40,0.70] | [2.513,4.398] | 0.1117 | **0.702** | 0.772 | **-9.1%** | 0.661 |
| 4 | [0.60,0.75] | [3.770,4.712] | 0.1423 | **0.894** | 0.911 | **-1.9%** | 0.835 |
| 5 | [1.15,1.35] | [7.226,8.482] | 0.1087 | **0.683** | 0.683 | **+0.04%** | 0.773 |

**The full cartesian system-schur reproduces the paper to -9.1% / -1.9% / +0.04%.** The mapped window (which
targets the same phase as the paper) beats the established window [3,9]: l=5 goes from +13% to +0.04% because its
late paper window [1.15,1.35] -> sim [7.23,8.48] captures the deceleration phase that the paper measures.

**CAVEATS (4-lens adversarial review, anti-overclaim):**
- **The 2pi is exact and mode-independent**: cyclic->angular conversion of the drift clock
  (`omega_d` cyclic, one turn = 2pi rad), verified to <0.5% against the analytic Petri eigenvalue
  (`diag/petri_eigenvalue.py` reproduces the targets with `Wd = 2pi omega_d`, and gives targets/6.2832 with
  `Wd = omega_d = 1`). This is not a fudge fitted after the fact.
- **Partial metrology, not pure**: after the 2pi, the residual (~9% l=3) is from the cartesian ring edge
  + finite resolution (n=96) + window roll-off (the local slope has no scale-free plateau: it
  peaks at t~3.2-4.2 then declines ~13%, WENO5 smoothing, not a nonlinear saturation; amplitude
  ~3x only at t=15.5, still growing).
- **l=5 is window-sensitive** (+/-27-29% depending on the window, cf. `NORMALIZATION.md`): its +0.04% is
  partly fortuitous, lead with l=3/l=4, do not cite l=5 as standalone support.
- **full == reduced to ~2%** in the established window across the 3 modes (the moment/Schur chain adds no
  deficit of its own); but the full seeds the paper's initial drift, so its transient (and thus the
  value in a very early window) differs from the reduced, hence the gap in the raw paper window.

## Reproduce

```bash
# adc_cpp: build ssprk3 (PR #230); adc_cases run.py = Strang + paper-faithful measurement (T3, maps the
# windows + reports gamma_raw_sim AND gamma_paper_units). t_end >= 8.5 required (mapped l=5 window ~[7.2,8.5]).
python hoffart_euler_poisson_dsl/run.py --engine system-schur \
  --n 96 --t-end 10 --modes 3 4 5 --dt 2e-3 --no-gif
# growth_rates.csv -> mode, gamma_raw_sim, gamma_paper_units, gamma_paper, relative_error_percent
# audit diagnostic (scales + candidates + window decomposition):
python hoffart_euler_poisson_dsl/diag/diag_normalization_audit.py 128
```

## 10. Paper-style figures (Fig 5.1-5.4)

`diag/make_paper_figures.py` reproduces the paper figures in its palette (white disk, slate
exterior `#3C4358`, schlieren in `Blues` colormap). Outputs in `figures/`:

- `snapshots_l3.png` / `snapshots_l4.png` / `snapshots_l5.png`: 3x3 grids of schlieren snapshots of
  the density at fractions `0.01, 1/8, ..., 7/8, t_f` (style Fig 5.1 / 5.2 / 5.3). The l-fold rollup is
  reproduced qualitatively: l=3 triangle -> 3 arms, l=4 square -> 4 vortices, l=5 pentagon -> 5 vortices.
- `diocotron_l3.gif` / `diocotron_l4.gif` / `diocotron_l5.gif`: rollup animations (same colors).
- `growth_rate.png`: style Fig 5.4, (a,b,c) `|c_l(t)|/|c_l(0)|` semilog + mapped fit window +
  paper theoretical slope; (d) `gamma_l` vs target (paper, full system-schur T3, ExB-drift).

l=3 (Fig 5.1): ring -> triangle -> 3 vortices.
![snapshots l=3](figures/snapshots_l3.png)

l=4 (Fig 5.2): ring -> square -> 4 vortices.
![snapshots l=4](figures/snapshots_l4.png)

l=5 (Fig 5.3): ring -> pentagon -> 5 vortices.
![snapshots l=5](figures/snapshots_l5.png)

Growth rate (Fig 5.4):
![growth rate](figures/growth_rate.png)

Resolution convergence (error -> 0):
![convergence](figures/convergence.png)

Full reproduction tutorial (installation, physics, math, code, run, figures): `README.md`.

NB: the snapshots/GIF are the density advected by the normalized ExB drift (the field the full
system-schur advects, `alpha/|Omega|=1/rho_max=1`), a representation of the paper's magnetic-drift limit,
up to `t_f = 10` diocotron periods. Panel (d) of `growth_rate.png` carries the rate of the
full system-schur (T3, section 9). The `gap_to_paper.png` figure (deficit -95%) is SUPERSEDED (a pre-T3
metrology artifact; cf. `figures/provenance.json`).

```bash
PYTHONPATH=<adc_cpp>/build-master/python \
  python hoffart_euler_poisson_dsl/diag/make_paper_figures.py 3 4 5 --out hoffart_euler_poisson_dsl/figures
```

## 11. Resolution convergence -> the error tends to 0 (the reproduction is confirmed)

`diag/convergence_reduced.py` (reduced ExB, paper-faithful measurement: mapped paper window + `gamma_paper =
gamma_raw * 2pi/rhobar`) shows that **the gap to the paper converges to 0 with resolution**: the residual
documented in sections 8-10 was indeed the cartesian discretization (ring edge), not a lock:

| n | l=3 | l=4 | l=5 |
|---|---|---|---|
| 64 | 0.666 (-13.7%) | 0.786 (-13.8%) | 0.682 (-0.1%) |
| 96 | 0.716 (-7.2%) | 0.891 (-2.2%) | 0.706 (+3.4%) |
| 128 | 0.743 (-3.8%) | 0.868 (-4.7%) | 0.687 (+0.6%) |
| 192 | 0.757 (-1.9%) | 0.910 (-0.1%) | 0.680 (-0.5%) |
| **256** | **0.767 (-0.6%)** | **0.913 (+0.2%)** | **0.678 (-0.7%)** |
| target | 0.772 | 0.911 | 0.683 |

At n=256 the three modes reproduce the paper to < 1%. l=3 decreases monotonically (13.7% -> 0.6%);
l=4 -> +0.2%; l=5 stays sub-percent (already close, slightly noisy due to its window sensitivity).
Figure: `figures/convergence.png`. The full system-schur also converges, measured directly by `run.py`
(mapped windows, gamma_paper_units): n=96 gives l=3 0.702 (-9.1%), l=4 0.894 (-1.9%), l=5 0.683
(+0.04%); n=128 gives l=3 0.729 (-5.6%), l=4 0.903 (-0.9%), l=5 0.681 (-0.3%). l=3 and l=4 tighten
as resolution rises, like the reduced. **Conclusion: the cartesian FV reproduction converges toward the
paper; the low-resolution residual was the cartesian ring edge.**

```bash
PYTHONPATH=<adc_cpp>/build-master/python python hoffart_euler_poisson_dsl/diag/convergence_reduced.py
```
