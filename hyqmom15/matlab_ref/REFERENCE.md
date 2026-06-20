# HyQMOM15 Matlab reference lock (RieMOM2D_Electrostatic_periodic)

This note pins the canonical Matlab reference for the `hyqmom15` port and the
fidelity plan for the goldens and drivers that follow. It is the spec for
[ADC-349](https://linear.app/romain7522/issue/ADC-349) (shared `matlab_ref`
layer), [ADC-350](https://linear.app/romain7522/issue/ADC-350) (goldens), and
[ADC-351](https://linear.app/romain7522/issue/ADC-351) (diocotron driver). It
makes no numerical change and touches no driver or golden (ADC-348 scope).

## 1. Reference position

- Canonical reference: the refactored Matlab `RieMOM2D_Electrostatic_periodic`.
  Future goldens and the diocotron driver align on it (its parameters, both
  sources, and `init_diocotron_field.m`), with the IC velocity-convention caveat
  spelled out in Section 4, D2.
- Legacy reference: the original `RIEMOM2D` (monolithic
  `main_electrostatic_wave.m`). The current goldens (`golden_*_gen.m`,
  `golden/*.csv`) were generated from it and stay valid for the closure, moment
  algebra, wave speeds, Lorentz sources, and `relaxation15` (those cores are
  byte-identical between the two Matlab trees). They are tagged "legacy" only
  where the new reference changes behavior (diocotron source and IC, below).
- The historical `ic_matlab_bug` path (the meshgrid-transposed diocotron drift
  of `initialize_dicotron.m`) is retired as the physics default. The new
  reference does NOT fix that drift: `init_diocotron_field.m` reproduces it
  verbatim (Section 4, D2), so the bit-parity path for a new diocotron golden is
  the named `--ic-matlab-bug` orientation, not the corrected default.

### Source location (not vendored)

The Matlab is not committed to `adc_cases` (608 vs 101 files, figures and movies
included). It lives maintainer-side, the same convention the existing goldens
already use (`octave --no-gui --path /chemin/vers/RIEMOM2D golden_gen.m`). On the
reference machine:

- new: `/Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic`
- old: `/Users/romaindespoulain/Documents/RIEMOM2D`
- maintainer diff: `RIEMOM2D_vs_RieMOM2D_Electrostatic_periodic_comparaison_detaillee.md`
  (plus `_unified.diff`, `_worddiff.diff`), with clarifications from Sacha
  (2026-06-19).

ADC-350 must pin the exact Matlab path and, ideally, its git commit when it
regenerates goldens.

## 2. Architecture of the new reference

`main.m` builds `params` via `init_case.m`, builds the grid via
`init_domain.m`, initializes the field via `params.init`, then loops:

```
apply_bc -> (if electrostatic) solve_poisson -> compute_speeds -> compute_dt -> time_step
```

- `init_case.m` dispatches on `case_name`: `dicotron`, `fluid_wave`,
  `electrostatic_wave`, `magnetic_wave`, `constant`.
- `apply_bc.m` -> `apply_periodic.m` fills the Np+2 ghost ring (periodic).
- `solve_poisson.m` -> `poisson_fft.m` (periodic, byte-identical to the legacy
  `poisson_fft.m`); skipped when `electrostatic == 0` (phi = 0).
- `compute_speeds.m` is `max |eigenvalues15_2D(M, flagsym)|` over the ghosted
  grid.
- `compute_dt.m` is `dt = CFL*dx/vmax`, then a source cap (Section 4, D6).
- `spatial_operator.m`: flux (`HLL` / `HLLC` / `ROE` / `WENO`) -> `compute_div`
  -> `+ source_term` if `params.source == 1`.
- `time_step.m`: `Euler` / `RK2` / `RK3`. Every committed case uses `Euler`.
- `source_term.m` sums the electric source (if `electrostatic`) and the magnetic
  source (if `magnetostatic`); both when both flags are set.

The closure, moment algebra (`InitializeM4_15`, `M2CS4_15`, `Flux_closure15_2D`,
`eigenvalues15_2D`), Lorentz sources (`electric_source_term`,
`magnetic_source_term`), `relaxation15`, and `collision15` are unchanged from the
legacy tree, so the corresponding ADC fidelity (closure to 1e-12, sources to
1e-14, speeds per block, relaxation to 4e-14) carries over unchanged.

## 3. Cases and parameters (new Matlab, exact values)

State order shared with ADC `MOMENT_NAMES`:
`[M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]`.
All cases: domain `[-0.5, 0.5]^2`, `Nmom=15`, `flagsym=1`, equilibrium
Maxwellian `rho0=1, U0=V0=0, T=1, r=0` (`C20=C02=1, C11=0`), `bc="periodic"`,
`time_scheme="Euler"`.

| Case | source files | Np | scheme (recon/limiter) | sources (es/ms) | omega_p / omega_c | k (kx, ky) | mode/eps | CFL / tmax |
|---|---|---|---|---|---|---|---|---|
| `dicotron` | `init_diocotron.m`, `init_diocotron_field.m` | 128 | HLL (first/minmod) | es=1, ms=1 | 20 / -20 | kmin=sqrt(2)*pi | mode=4, eps=0.1 | 0.5 / 1.0 |
| `electrostatic_wave` | `init_electrostatic_wave.m`, `init_electrostatic_wave_field.m` | 128 | HLL | es=1, ms=0 | 30 / -90 (inert) | (0, 4*pi) | mode=15, eps=0.01 | 0.5 / 1.0 |
| `magnetic_wave` | `init_magnetic_wave.m`, `init_magnetic_wave_field.m` (*) | 256 | HLL (muscl/minmod) | es=1, ms=1 | 20 / -40 | (2*pi, 4*pi) | mode=15, eps=0.01 | 0.5 / 1.0 |
| `fluid_wave` | `init_fluid_wave.m`, `init_fluid_wave_field.m` | 32 | ROE (first/none) | es=0, ms=0 | 30 / -90 (inert) | (4*pi, 0) | mode=15, eps=0.01 | 0.4 / 0.05 |
| `constant` | `init_constant.m`, `init_constant_field.m` | n/a | n/a | es=0, ms=0 | n/a | n/a | uniform | sanity only |

(*) `init_magnetic_wave.m` wires `init_electrostatic_wave_field` by oversight;
the intended field is `init_magnetic_wave_field` (Section 4, D4).

### Diocotron details (the milestone driver, `init_diocotron.m`)

- Ring: `rho_min=1e-4`, `rho_max=1`, `r0=0.35`, `r1=0.4`, azimuthal
  perturbation `1 - eps + eps*sin(mode*theta)`, `eps=0.1`, `mode=4`.
- Poisson: `adim_debye_length = 1/omega_p = 0.05`; `phi = poisson_fft(rho, ...)`.
- Initial drift (`init_diocotron_field.m`): E x B from the periodically ghosted,
  centered gradient of phi, `vdr(:,:,1) = -grad_phi(:,:,2)/omega_c`,
  `vdr(:,:,2) = +grad_phi(:,:,1)/omega_c`, where `grad_phi(:,:,1)` and `(:,:,2)`
  are centered differences along the first and second array index under
  `[X,Y] = meshgrid(xm, ym)`. This is structurally identical to the legacy
  `initialize_dicotron.m`, i.e. the transposed (divergent, non-azimuthal)
  meshgrid drift that ADC-198 identified, NOT the standard incompressible E x B
  (Section 4, D2).
- Moments are the unit-temperature Maxwellian raw moments of `(rho, vx, vy)`
  (e.g. `M20 = rho*(vx^2+1)`), equal to ADC `gaussian_state(rho, vx, vy, 1, 0, 1)`.
- CFL max speed: `linearized_Jacobian_magnetostatic(kmin, kmin, adim_debye,
  omega_c)`, then `max(real(eig))`. Here `lambda = diag(D)` is correct (unlike
  the electrostatic wave case, D3).

## 4. Known divergences and ADC decisions

D1-D5 are the issue's explicit points; D6-D9 were found in the source and are
documented so the next tickets do not rediscover them. "Intent, not bug" means
reproduce the physical intent of the reference, not a transcription slip.

| # | Divergence (new Matlab) | Current ADC | ADC decision | Owner |
|---|---|---|---|---|
| D1 | Diocotron runs BOTH sources: `electrostatic=1` and `magnetostatic=1`, `omega_c=-20`, so `source_term.m` adds electric + magnetic | `run_diocotron.py` builds `build_moment_model(omega_c=0.0)`: electric source only, B enters only via the IC drift | Activate the magnetic source in the port: `build_moment_model(omega_c=-20, ...)`. The core already supports it (`adc.moments.lorentz_sources`), no adc_cpp change | ADC-351 |
| D2 | `init_diocotron_field.m` keeps the legacy meshgrid E x B, structurally identical to `initialize_dicotron.m`: `vx=-grad_phi(:,:,2)/omega_c`, `vy=+grad_phi(:,:,1)/omega_c` under `meshgrid(xm,ym)`, the transposed divergent drift (ADC-198), not the standard incompressible one | ADC default is the corrected standard incompressible E x B; `--ic-matlab-bug` reproduces the transposed drift | Intent over bug: keep the corrected drift as the physics default. For strict diocotron golden parity reproduce the reference's literal IC under the named `--ic-matlab-bug` path. ADC-351 must verify the meshgrid axis convention numerically before pinning the golden | ADC-351 |
| D3 | `init_electrostatic_wave_field.m:28` `lambda_max = diag(D)` for the CFL speed, where `D` is the mode Jacobian at `(kx,ky)`; intent is `diag(Dmax)` from `Jmax` at `(kmin,kmin)` | not ported | Reproduce intent: use `diag(Dmax)` (max possible speed) for the CFL. Bug-for-bug only under a named legacy test | ADC-349/350 |
| D4 | `init_magnetic_wave.m:83` sets `params.init = init_electrostatic_wave_field`; `init_magnetic_wave_field` (magnetostatic Jacobian, `omega_c`) exists and is the intended init | not ported | Reproduce intent: port the `init_magnetic_wave_field` path. Flagged probable oversight (Sacha unconfirmed); revisit if Sacha confirms it is deliberate | ADC-349/350 |
| D5 | `HLLC`, `WENO`, `neumann`, `outflow` appear in the code | ADC has HLL (exact speeds), ROE pending audit, WENO5 in core | Do not port this milestone. Per Sacha these are refactor placeholders; HLLC is not adapted to HyQMOM15. Document, do not transcribe | ADC-348 (done here) |
| D6 | `compute_dt.m` source cap: `dt_electrostatic = CFL*dx*vmax/omega_p^2` (diocotron, `electrostatic` branch wins over `elseif magnetostatic`) | ADC caps via `source_frequency(omega_p)` -> `dt <= cfl/omega_p`, ~500x laxer, never bites (ADC-197) | Keep the source-dt fidelity out of `source_frequency`. Put the `compute_dt.m` policy in `matlab_ref/dt_policy.py`, audited in ADC-356; do not loosen tolerances to hide the gap | ADC-356 |
| D7 | `apply_periodic.m` leaves the 4 ghost corners zero; `compute_speeds.m` reads `1:Np+2` including corners -> zero-state `eigenvalues15_2D` | ADC fills halos consistently | Do not reproduce the corner artifact. It only perturbs `vmax` through a spurious corner (max over abs, usually inert). ADC-350 must confirm it does not move the golden `dt` sequence | ADC-350 |
| D8 | `RK2`/`RK3` freeze `phi` across substeps (not fully Poisson-coupled) | ADC uses `Explicit()` forward Euler to match | No action: every committed case is `Euler`. If RK is ported later, recompute `phi` per stage for full coupling | future |
| D9 | `main.m` always calls `compute_L2_error`, which only handles the wave cases, not `dicotron` | ADC has no L2 oracle yet | Refactor leftover. ADC `compute_L2_error` (ADC-349) targets the wave cases; diocotron fidelity stays invariant + golden based | ADC-349 |

### Bug-for-bug policy

Default: reproduce the physical intent of the new reference, not its
transcription bugs (D3, D4). Bug-for-bug parity is allowed only under an
explicitly named legacy test (a `*_matlab_bug` style smoke), never as the
default path. This mirrors the existing `--ic-matlab-bug` precedent in
`run_diocotron.py`.

## 5. Fidelity plan (goldens, tolerances, reproducibility)

ADC-350 generates goldens by running the new Matlab under Octave; they are never
re-transcribed by hand. Mirror the existing `golden_*_gen.m` shape (states ->
`%.17g` CSV) with new generators pointed at the new tree.

Artefacts to add (owned by ADC-350, specified here):

- diocotron golden with the magnetic source ON (D1) and the standard E x B IC
  (D2): re-generate, do not reuse the legacy diocotron field.
- per-case wave goldens (electrostatic, magnetic, fluid) with the intent fixes
  (D3, D4) baked into the matlab_ref init layer (ADC-349), so the Python init
  and the Matlab init agree.
- frozen `dt` sequences (`*_dts.csv`) so trajectory replays isolate scheme
  fidelity from the dt policy (D6).

Tolerance ladder (reuse the established values; do not loosen without a numerical
justification recorded next to the assert):

| Component | rtol / atol | source |
|---|---|---|
| flux / closure | rtol 1e-14 (1e-12 for the quasi-degenerate state) | `run.py` |
| wave speeds (well conditioned) | rtol ~1e-8, observed ~1e-15 | `run_waves.py` |
| wave speeds (near degenerate) | conditioning-scaled (measured sensitivity x100) | `run_waves.py` |
| Lorentz sources | rtol/atol 1e-14 | `run_crossing.py` |
| Euler trajectory replay | L2 ~1e-9 (HLL golden replay); ~4.5e-16 unsplit | `run_crossing.py` (~1e-9), `run.py` (~4.5e-16) |
| ssprk2 trajectory | ~4 to 5 percent (second order vs the Euler reference) | `run_crossing.py` |
| relaxation15 (5 branches) | rtol 1e-12, atol 1e-13 x scale | `run_relaxation.py` |
| compiled projector smoke (cross platform) | atol 1e-4 | `run_golden_transport_relax.py` |

Reproducibility: deterministic seed states (`gen_states.py`), frozen `dt`
sequences, `%.17g` precision, a pinned Octave `--path` and Matlab commit.

## 6. Next steps (handoff)

- ADC-349: build the shared `matlab_ref/` Python layer (params, linearized
  Jacobians, `init_*_field`, `compute_dt` policy, `compute_L2_error`) mirroring
  the new Matlab. This note is its contract.
- ADC-350: generate the goldens from `RieMOM2D_Electrostatic_periodic`
  (diocotron with B-source, wave cases) under Octave.
- ADC-351: align `run_diocotron.py` to the new periodic Matlab: `omega_p=20`,
  `omega_c=-20` with the magnetic source ON (D1), `Np=128`, `CFL=0.5`, standard
  E x B IC (D2).
- ADC-356: audit the adc_cpp gaps before any core change (ROE on a compiled
  HyQMOM15 DSL model, the Euler backend vs SSPRK2, MUSCL/minmod parity, the
  `compute_dt` source policy).

### Port priority

1. `matlab_ref` common layer (ADC-349): unblocks the rest cleanly.
2. Diocotron alignment (ADC-351): already mostly ported, needs the B-source and
   the parameter refresh.
3. Goldens (ADC-350).
4. Wave cases (electrostatic, magnetic, fluid): after the ADC-356 audit
   (ROE, backend).
5. `constant`: sanity only, lowest.

### adc_cpp gaps

This reference lock surfaces no new confirmed adc_cpp gap: the magnetic Lorentz
source (`lorentz_sources`), periodic Poisson (`System.set_poisson`), and exact
HLL speeds (`exact_speeds=True`) are already in the core. The open audit items
(ROE for a HyQMOM15 DSL model, Euler vs SSPRK2 backend, MUSCL/minmod parity, the
`compute_dt` source policy) are enumerated in ADC-356. ADC-348 changes no
adc_cpp code and proposes no new core change.

### adc_cpp audit results (ADC-356)

ADC-356 closes the open items above against the built core (CI `_adc`), with no
adc_cpp change inside the adc_cases PRs:

- Euler backend: `production` + periodic Poisson + Euler run together (with `HLL`,
  `exact_speeds`, and both the electric and magnetic sources), verified green in CI
  by ADC-351. The `aot` backend freezes SSPRK2, so the faithful Euler path is
  `backend="production"` with `adc.Explicit(method="euler")`.
- MUSCL / minmod: representable with no new core feature. `adc.FiniteVolume(limiter=...)`
  IS the reconstruction: `limiter="none"` is first order (Matlab `reconstruction="first"`),
  `limiter="minmod"` is MUSCL with minmod (Matlab `reconstruction="muscl"`, `limiter="minmod"`).
- `compute_dt` source policy: kept explicit in `matlab_ref/dt_policy.py` (D6), not
  folded into `source_frequency`. No core change.
- ROE for a HyQMOM15 DSL model: CONFIRMED GAP. `adc.FiniteVolume(riemann="roe")` is
  rejected unless the compiled model declares a `"p"` primitive or calls
  `m.enable_roe()` (`HasRoeDissipation`); `build_moment_model` and `adc.moments`
  provide neither. Tracked as adc_cpp ADC-368 (generic Roe-dissipation hook for
  moment DSL models). Until then `fluid_wave` (ADC-352) uses `riemann="hll"` +
  `exact_speeds` as a named non-strict path.

Wave drivers (ADC-352/353/354) must build the model with `backend="production"`
and `adc.Explicit(method="euler")` for a faithful Euler step.

`matlab_ref/check_reference.py` is a build-free guard that this note stays
present, ASCII, em-dash free, and complete (it asserts the canonical parameters
and the divergence decisions above are documented). It runs in CI via the
manifest.
