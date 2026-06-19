# hyqmom15: 2D Vlasov-Poisson with 15 moments (HyQMOM closure)

A 2D kinetic model: you transport the velocity moments `M_pq = integral f v_x^p v_y^q dv`
of order p+q <= 4 (15 components) of the Vlasov equation, coupled to the system's Poisson
equation. For each moment,

```text
∂t M_pq + ∂x M_{p+1,q} + ∂y M_{p,q+1} = q/m (p Ex M_{p-1,q} + q Ey M_{p,q-1})
                                        + Ωc (p M_{p-1,q+1} − q M_{p+1,q-1})
```

The flux at the highest order brings in order-5 moments that are absent from the state
vector: this is the closure problem. The HyQMOM closure (Bryngelson, Fox & Laurent 2025,
hal-05398171) expresses the six standardized order-5 moments as functions of the lower
orders and makes the system hyperbolic.

State (component order shared with the MATLAB reference RIEMOM2D):

```text
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
```

## Quick start

```bash
# from the adc_cases root, with the adc module built (PYTHONPATH pointing at build-*/python)
python hyqmom15/run.py            # flux vs MATLAB goldens + Gaussian oracle
python hyqmom15/run_waves.py      # exact wave speeds vs goldens
python hyqmom15/run_crossing.py   # E/B sources, Larmor rotation, crossing jets
python hyqmom15/run_diocotron.py  # full Vlasov-Poisson: diocotron ring
python hyqmom15/run_relaxation.py # realizability projection + crossing Ma=20
mpirun -np 2 python hyqmom15/run_mpi.py  # multi-rank MPI smoke (Poisson geometric_mg); see section
```

## Composing the model

Everything goes through `build_moment_model` ([model.py](model.py)), which delegates the
moment algebra to the generic generator `adc.moments` in adc_cpp. The only physics written
here is the closure, a callable that receives the standardized moments and returns the
order-5 ones:

```python
from model import build_moment_model, hyqmom_closure

m = build_moment_model(
    closure=hyqmom_closure,   # la physique : S (ordres 2-4) -> S50..S05
    exact_speeds=True,        # vitesses d'onde par valeurs propres du jacobien (HLL fidèle)
    with_sources=True,        # sources électriques (lit grad phi) + magnétique (omega_c)
    debye=0.04,               # couplage Poisson : Delta phi = (M00 - rho_background)/debye^2
    rho_background=rho_bg,    # fond neutralisant = moyenne de M00 (obligatoire en périodique)
    omega_p=25.0,             # borne le pas de temps de la source
)
compiled = m.compile(so_path, include_dir, backend="aot")
sim = adc.System(n=128, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="fft")
```

Writing another moment system = supplying another closure callable (same contract) and, if
you want exact per-block wave speeds, a partition of the Jacobian (`HYQMOM_BLOCKS` for this
one).

## MATLAB fidelity: proven, partial, missing

`hyqmom15` is a `validation` case (manifest `cases_manifest.toml`): the drivers assert
invariants and bit-level agreement against the MATLAB reference RIEMOM2D, they do not
reproduce a published physical curve. What that means component by component:

> Reference pivot (ADC-348): future strict goldens move from the legacy `RIEMOM2D` to
> the refactored `RieMOM2D_Electrostatic_periodic`. The canonical parameters, known
> divergences, ADC decisions, and the fidelity plan are locked in
> [matlab_ref/REFERENCE.md](matlab_ref/REFERENCE.md). The numbers in this README
> are the current port (e.g. `omega_p=25`); the canonical reference values
> (e.g. `omega_p=20`) live in that note, and ADC-351 aligns the driver to them.

| Status | Component | Evidence (number, source) |
|---|---|---|
| Proven | closure (`closureS5.m`, 6 standardized order-5 formulas) | `hyqmom_closure` ([model.py](model.py)) === `Flux_closure15_2D.m` to 1e-12 on 10 states (`run.py`), exact on Gaussians via the independent Isserlis oracle (`gaussian_raw_moment`) |
| Proven | moment algebra M->C->S->C5->M5 (binomials) | delegated to `adc.moments`; the 15-component order and the `(p,q)` map are asserted equal to `gmom.moment_names(4)` / `gmom.moment_indices(4)` at import ([model.py](model.py)) |
| Proven | flux assembly order, 15 components Fx/Fy | `MOMENT_NAMES` / `MOMENT_PQ` shared with RIEMOM2D ([model.py](model.py)) |
| Proven | per-block wave speeds | `HYQMOM_BLOCKS` === `eigenvalues15_2D.m` flagsym=1 to ~1e-11 (`run_waves.py`), near-degenerate state judged against its measured conditioning |
| Proven | Lorentz sources (E lowers order, B conserves) | `moment_sources` === reference eqs 1.3-1.7 to 1e-14, Larmor rotation === analytic (`run_crossing.py`) |
| Proven | Euler trajectory replay | replaying the HLL golden steps with `time='euler'`, L2 gap to MATLAB ~4.5e-16 after 20 steps (`run.py`); the MATLAB additive split + Euler is algebraically the unsplit Euler |
| Proven | Poisson | phi === analytic on a sinusoid to 1e-14 in `fft_spectral`, source E === -grad phi centered to 1e-16, checkpoint/restart bit-identical (`run_diocotron.py`) |
| Proven | `relaxation15` isolated projection (5 branches) | `relax15` === Octave `golden_relax_gen.m` to 4e-14 on 12 states, branch coverage asserted (`run_relaxation.py`) |
| Proven | realizability under transport | the native compiled projector (`build_projection`, emitted via `m.projection` / ADC-177) reproduces `relax15` branch by branch to ~1e-15 on the 12 goldens and `relax_field` on a field; it runs as the System post-step hook with no per-cell Python callback ([validate_native_projector.py](validate_native_projector.py), [notes/native_projector.md](notes/native_projector.md)). `relax_field` stays the oracle |
| Partial | correlated crossing IC (`r != 0`) | blocked by `NotImplementedError` in `crossing_state` ([model.py](model.py)); Octave shows `gaussian_state` === `InitializeM4_15` for `r != 0`, so the gate is removable, not a divergence (ADC-274) |
| Partial | golden coverage | a code-anchored golden runs the NATIVE projector THROUGH transport (Ma=20 crossing, [run_golden_transport_relax.py](run_golden_transport_relax.py), ADC-203); it pins our own transport x relaxation trajectory against drift, not MATLAB fidelity (the HLL golden runs `flagrelax=0` Ma=2, the relax golden runs the projection isolated, so a MATLAB-anchored transport x relaxation cross-check stays separate) |
| Missing | source dt bound === `compute_dt.m` | our `omega_p` bound is ~500x laxer than the MATLAB source CFL and never bites (ADC-197, section below) |
| Partial | BGK collision | a generic BGK relaxation toward the local Maxwellian is wired (`moments.bgk_source`, `build_moment_model(collision=True)`, ADC-277); the emitted source is verified `== nu*(M_eq - M)` to machine precision with the collisional invariants M00/M10/M01 zero (`run_relaxation.py` (5)). The full anisotropic `collision15.m` (Kn-dependent branches) is not yet matched |
| Missing | diocotron growth rate vs a long MATLAB golden | dedicated campaign, out of CI |

Use the drivers above for validation. Use the native compiled path (`backend="aot"`, exact
speeds, Poisson `geometric_mg`) for production runs, with the realizability projection emitted
natively (`projection=True`, section below) so it runs in the step without a per-cell Python loop.

### Useful options

| Option | Effect |
|---|---|
| `robust=True` | smooth floors on M00, C20, C02 (protected divisions and roots). The default `False` reproduces the MATLAB, which protects nothing. |
| `exact_speeds=False` | speed bound `u +- 3*sqrt(C)` instead of the exact eigenvalues. Good enough to start in Rusanov; realizable states exceed it (checked by run.py), so never for HLL. |
| `solver=` (drivers) | `fft` (direct periodic), `fft_spectral` (continuous symbol, exact on sinusoids), `geometric_mg` (general, required under MPI). |

## Realizability: oracle vs native projector

A moment vector must stay that of a positive distribution (realizability, tested by the
smallest eigenvalue of the `p2p2` matrix, `p2p2_2D.m`). The scheme does not preserve it.
Measured on a local crossing Ma=20 run, without projection: dt decays as ~exp(-18.5 t), a
ratio ~212 from start to end, because the state leaves the realizable set and the Jacobian
eigenvalues blow up. With the projection applied at every step, dt stays ~1.2e-3 over a full
`tend=4` run (ROMEO CPU job 654809, 3566 steps, mass exact).

The projection is `relaxation15` ([relaxation.py](relaxation.py)): clamp the standardized
moments, then relax toward a realizable target. The port follows the MATLAB branch by
branch; the "complex eigenvalues" test evaluates the order-3 Jacobian sub-blocks at the
standardized state.

Two paths, two purposes:

- Oracle: `relax_field` ([relaxation.py](relaxation.py)) is a per-cell Python loop with a host
  copy and a `numpy.linalg.eigvals` per cell. It is the reference for validation (=== Octave to
  4e-14, `run_relaxation.py`) and the source of truth for the native projector. It is not
  scalable: do not use it inside a GPU/MPI time step. Per-field usage:

  ```python
  from relaxation import make_corner_eigs, relax_field
  fn = make_corner_eigs()
  U = relax_field(U, lamin=1e-12, Ma=4.0, corner_eigs=fn)   # (15, ny, nx) -> projeté
  ```

- Native projector (production): `build_projection` ([model.py](model.py)) emits `relaxation15`
  through the generic post-step hook `m.projection` (ADC-177). Enable it at build time:

  ```python
  m = build_moment_model("hq", exact_speeds=True, projection=True, Ma=20.0, lamin=1e-12)
  ```

  It is a branch-by-branch transcription of `relax15` into the DSL `Expr` algebra with no
  dynamic branch (each MATLAB branch is a branchless mask blend in `max`/`min`/`abs_`/`sign`);
  the "complex eigenvalues" witness uses `dsl.eig_max_im` and the `p2p2` realizability gate uses
  `dsl.eig_lmin` (ADC-289, over `adc::real_eig_minmax`). The System applies the compiled
  `project(U, aux)` at the end of each whole step (post-step, MATLAB `flagrelax=1`; an
  `after_stage` variant would trade fidelity for robustness) on the valid cells, with no per-cell
  Python callback. `M00`, `M10`, `M01` pass through unchanged. See
  [notes/native_projector.md](notes/native_projector.md).

Application policy: `projection=True` requires `exact_speeds=True` (the complex-eigenvalue test
reads the order-3 flux-Jacobian sub-blocks); `Ma` and `lamin` are baked into the projector. The
hook is rejected by the `prototype` backend and the `amr_system` target (ADC-177 contract).

Validation (issue criteria, [validate_native_projector.py](validate_native_projector.py)):
compiled `project` == `relax15` on the 12 goldens (branches 0-4) to ~1e-15, well inside the 1e-12
to 1e-10 per-branch tolerance; compiled projection over a `(15, ny, nx)` field == `relax_field`;
at Ma=20 the native non-realizable rate drops as with `relax_field`; and `projection=False` emits
no hook (bit-identical transport).

Idempotence caveat: `relaxation15` is a relaxation toward a target, not a strict projection
(`P(P(U)) != P(U)`; re-relaxing relaxes again). The ADC-177 `m.projection` contract documents
idempotence as the intended property, but the System applies the hook once per macro-step, so a
single pass reproduces `relax_field` exactly (what the acceptance criteria require). Do not enable
a repeated / `after_stage` application expecting bit-for-bit `relax_field`: it would keep relaxing,
faithful to MATLAB but not idempotent. The drivers assert no idempotence, faithful to MATLAB.

## Correlated crossing initial condition (`r != 0`)

`crossing_state(..., r=...)` ([model.py](model.py)) currently raises `NotImplementedError`
for `r != 0`. The historical comment said MATLAB freezes `S22=1, S31=S13=0`, distinct from an
exact correlated Gaussian. Direct Octave check (audit ADC-274): for correlated states,
`RIEMOM2D/InitializeM4_15(...)` and our `gaussian_state(...)` produce the same 15 moments to
~1e-15, because MATLAB encodes the correlation through `C11 = r*sqrt(C20*C02)` then
`S4toC4(...)` reconstructs the equivalent raw/central moments. So `r != 0` is removable (set
`C11 = r*T` since here `C20 = C02 = T`), not a real divergence; the gate stays until ADC-274
lands the parity test. Until then, `r = 0` is the only validated crossing IC.

## Time-step policy: `omega_p` vs `compute_dt.m`

The case bounds the source coupling with a constant `omega_p` (default 25.0):
`m.source_frequency(omega_p)` ([model.py](model.py)) yields `dt <= cfl/omega_p`. MATLAB
(`compute_dt.m`) instead imposes `dt_source = CFL*dx*lambda_flux*k_min^2/max_speed^2`, and
that is the bound that bites. On the diocotron n=64 IC (Octave): `dt_source = 2.99e-5`,
`dt_MATLAB = min(dt_flux=2.69e-3, dt_source) = 2.99e-5` (source-limited). The ADC source
bound at the same point is ~1.6e-2, ~535x laxer; it never triggers, the transport bound
(`last_dt_bound()='transport:mom'`) governs, and ADC advances ~60x the MATLAB dt.

Implication (ADC-197): the runs stayed stable (ssprk2 + projection), so this is not an
observed instability, but it is an undocumented fidelity gap: a measured growth rate is taken
at a dt very different from the reference, and the explicit-source coupling is protected only
by the transport bound. Resolution is open (recompute an effective `source_frequency` per
step, or cap dt explicitly, or document it as an approved gap with a control run at the
MATLAB dt). Until then, do not claim MATLAB dt fidelity.

## Validation: regenerating the goldens

The proven rows of the fidelity table are checked in CI. The references are generated by
running the real MATLAB code (RIEMOM2D) under Octave; they are never re-transcribed:

```bash
python3 gen_states.py
octave --no-gui --path /chemin/vers/RIEMOM2D golden_gen.m        # flux + valeurs propres
octave --no-gui --path /chemin/vers/RIEMOM2D golden_hll_gen.m    # trajectoire HLL (crossing)
octave --no-gui --path /chemin/vers/RIEMOM2D golden_relax_gen.m  # relaxation15 (5 branches)
```

The ssprk2 trajectory gap to MATLAB is 4% (the second order, vs ~4.5e-16 for the `time='euler'`
replay in the table). The Ma=20 realizability contrast (`run_relaxation.py`): projected ~13%
of cells violated vs ~52% raw, the executable witness that the projection works on a field.

### Code-anchored golden: transport + native relaxation15 (ADC-203)

`run_golden_transport_relax.py` freezes a small deterministic trajectory of the CURRENT adc_cpp code in
the regime where the projector actually fires: the Ma=20 crossing flow (n=32, HLL + exact speeds,
`projection=True` so the NATIVE `relaxation15` projector is applied post-step by System, no Poisson,
3 steps at a frozen `dt=2e-4`) as `golden/golden_transport_relax_state.csv` (+ `..._meta.csv`). It is
the only fixture that runs the NATIVE projector THROUGH a transport trajectory: `run_relaxation.py`
(3)/(4) apply the Python `relax_field` ORACLE manually each step (and (4) is MATLAB-anchored, tol 5e-8),
and `validate_native_projector.py` checks the native projector in ISOLATION (no transport). It is a
non-regression freeze of OUR own trajectory, deliberately distinct from the MATLAB/RIEMOM2D-anchored
goldens: it catches silent drift of the transport x projector path, not a divergence from MATLAB. The
Ma=20 crossing is a stiff flow that Lyapunov-amplifies FP differences, and the NATIVE compiled (Kokkos)
projector + HLL diverge macOS->Linux far more than the numpy oracle: the golden is SAME-platform
bit-exact (max|dU|=0) but only CROSS-platform coarse (a macOS golden vs a Linux CI run drifts ~7.6e-6
over 3 steps). So the CI gate is `atol=1e-4` (~13x the measured drift), a non-regression SMOKE that
catches the projector firing and gross scheme/projector regressions, not subtle bit drift. The check
asserts the projector is materially active (a `projection=False` replay differs by ~8e2 here: the
unprotected Ma=20 run blows up), so the golden cannot silently degrade into a transport-only freeze. It
is serial only (not bitwise across MPI ranks) and dt is hardcoded, so an eigenvalue change cannot
silently re-pick dt and pass as "no drift".

```bash
python3 hyqmom15/run_golden_transport_relax.py            # CHECK against the committed golden (CI)
python3 hyqmom15/run_golden_transport_relax.py --regen    # rewrite after an INTENTIONAL change
git add -f hyqmom15/golden/golden_transport_relax_state.csv hyqmom15/golden/golden_transport_relax_meta.csv
```

Regenerate (and force-add, since `*.csv` is gitignored repo-wide) only after a deliberate change, and
say so in the PR.

## Initial ExB drift: a deliberate fix to a MATLAB meshgrid trap

The diocotron IC builds the velocity from the drift `v = (E x B) / B^2`, i.e.
`vx = -d_y(phi) / omega_c`, `vy = +d_x(phi) / omega_c`. For a ring potential this drift is
azimuthal and incompressible (`div v = 0`). `run_diocotron.py:108-110` (`diocotron_state`,
line 109 `vx = -gphi_y / OMEGA_C`) computes exactly that.

The reference `initialize_dicotron.m:34-48` does not. With `[X,Y] = meshgrid(xm,ym)` the first
matrix index runs along y and the second along x; the loop differences `phi_ghosted` on the
first index but stores it into the component labelled x (and divides by `dx`), without
transposing. The result is `vx = -d_x(phi) / omega_c`, `vy = +d_y(phi) / omega_c`: the two
components are swapped relative to the drift, the field is no longer azimuthal, and it is
divergent (`div v = -d2x(phi) + d2y(phi) != 0`). This is the classic MATLAB meshgrid trap.

This is a deliberate discrepancy where the port is right against the reference. Measured at
n=64 (Octave running `initialize_dicotron.m` vs the numpy `diocotron_state`):

- `rho` and `phi` are faithful: relative error 5.0e-10 and 3.7e-9 (same scalar equations, same
  `poisson_fft` to FFT round-off). Only the velocity orientation differs.
- the swap is exact: `PY vx == -MAT vy` and `PY vy == -MAT vx` to 1.1e-9; equivalently
  `MAT vx == -d_x(phi)/omega_c` to 1.1e-9, the wrong axis.
- the physical signature: the normalized divergence `max|div v| / (Vmax/h)` is 1.2e-16 for the
  standard drift (incompressible) and 0.55 for the MATLAB field (divergent).

Note: this is the IC only. The electric source term is faithful (the M10 flux uses the same
index convention as its source, so the two are consistent); do not "fix" it by analogy.

Impact on the dynamics (issue point b), measured with the same `phi` so only the orientation
changes, n=64, Rusanov, no `relaxation15`: the mode-4 azimuthal amplitude `A4` of the density
starts identical (`A4(0) = 4.93e-2`, both ICs share `phi` and `rho`) and the two runs then
diverge. The relative gap on `A4` is 11% at 20 steps, 18% at 40, 16% at 80; the kinetic-energy
proxy `0.5 (M10^2 + M01^2)` differs by 5 to 20%; the full state by `max|U_std - U_bug|` ~
0.2 to 0.36. The effect is therefore not negligible: the ring density is quasi-axisymmetric,
but the velocity field is fully reoriented (azimuthal vs divergent), so the trajectories part
quickly. These are short smoke windows (the perturbation is damped by Rusanov, `A4` decreases
in both), not the saturated growth regime; the quantitative growth rate is a dedicated campaign
(see Limitations).

To reproduce the MATLAB field on purpose (strict trajectory comparison against the Octave
reference), pass `--ic-matlab-bug`; it rebuilds the swapped, divergent drift behind
`diocotron_state(n, ic_matlab_bug=True)` and prints the impact table above. The default is off
and the standard, correct drift is always used in CI.

```bash
python hyqmom15/run_diocotron.py                  # standard ExB drift (default, CI)
python hyqmom15/run_diocotron.py --ic-matlab-bug  # reproduce the meshgrid trap + impact table
```

## Multi-rank MPI smoke (geometric_mg)

[run_mpi.py](run_mpi.py) replays the Vlasov-Poisson diocotron under `mpirun` (np=2 and 4)
with the geometric multigrid Poisson solver (`solver="geometric_mg"`). The direct FFT
solver is single-rank by design (it unrolls a single box round-robin) and the core rejects
it explicitly when `n_ranks>1`; the MG is the only distributed elliptic path.

```bash
# mpi4py built against the SAME libmpi as _adc is required: its import calls MPI_Init, after
# which the core reads the rank via _adc.my_rank()/n_ranks(). Run 1 on-node thread for parity.
OMP_NUM_THREADS=1 mpirun -np 1 python hyqmom15/run_mpi.py   # serial reference (writes the np=1 state)
OMP_NUM_THREADS=1 mpirun -np 2 python hyqmom15/run_mpi.py   # np=2: checks + parity + FFT rejection
OMP_NUM_THREADS=1 mpirun -np 4 python hyqmom15/run_mpi.py   # np=4: same
```

Topology (measured, not assumed): the Cartesian `adc.System` is SINGLE-BOX (one box covers
the whole domain); under MPI the `DistributionMapping(1, n_ranks)` assigns it to rank 0
alone. The smoke therefore exercises the collective safety of the MG Poisson and of the
reductions (max CFL, mass) under MPI, plus the FFT guard, but not the halo exchange between
disjoint boxes (there is only one; distributed multi-box Cartesian belongs to AMR).

Measured (n=64, 12 steps, 1 on-node thread, Mac M1, MPI+Kokkos Serial build):

- np=1/2/4: 12 steps finished, mass conserved to 2.6e-16, phi finite; `solver="fft"`
  rejected by the core message "solveur fft non supporte en MPI (n_ranks>1)",
  `solver="fft_spectral"` rejected as an unknown solver (not implemented in this revision of
  the core), both as a clean RuntimeError with no deadlock or segfault;
- np=2 and np=4 parity vs np=1: BIT-IDENTICAL (dU_max = dphi_max = 0), deterministic halos;
- indicative cost/scaling: the 12-step loop ~0.11 s (np=1), ~0.21 s (np=2), ~0.19 s (np=4);
  no speedup, which is expected (the grid fits in one box on rank 0, the other ranks only do
  the collectives): this smoke validates MPI correctness, not load balancing.

## AMR

The same model runs on the adaptive `adc.AmrSystem` hierarchy (refinement on M00, composite
`geometric_mg` Poisson, conservative reflux) by compiling the `.so` for the AMR facade:

```python
compiled = m.compile(so_path, include_dir, backend="production", target="amr_system")
sim = adc.AmrSystem(n=48, L=1.0, periodic=True, regrid_every=4)
sim.add_equation("mom", compiled,
                 spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                 time=adc.Explicit())        # on AMR: forward Euler + reflux (default)
sim.set_refinement(threshold=0.5)            # tags the ring on M00
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_conservative_state("mom", U0)        # the 15 moments, prolonged to the fine patches
```

Note: on AMR the default `adc.Explicit()` is forward Euler + reflux, the scheme closest to the
reference MATLAB (cf. the `time='euler'` replay in run_crossing); `ssprk3` exists only for
native `add_block` blocks, the `.so` loader rejects it explicitly. `relaxation15` is not
available on this path: short horizon (smoke), realizable long runs stay on the uniform
`System`. The `run_amr.py` driver validates construction, per-level mass conservation,
coarse/fine consistency against a uniform `System` at the fine resolution, and the clean
rejections:

```bash
python hyqmom15/run_amr.py
```

## Scale: validated CPU/GPU/MPI, and what is not

What the compiled path has been measured to do:

- CPU serial: full crossing Ma=20 run, `tend=4`, 3566 steps, mass exact, dt stable ~1.2e-3
  with per-step projection (ROMEO job 654809).
- GPU GH200: 24706 steps device-clean (dense eigenvalues / HLL / sources / multigrid Poisson)
  on the diocotron Vlasov-Poisson path (ROMEO job 654562).
- MPI: bit-identical np=1/2/4 on the geometric_mg Poisson and the reductions (section above),
  single-box topology only.

What is not validated at scale:

- The realizability projection inside a device/MPI step. The `relaxation15` path that keeps
  dt stable is still a Python per-cell loop with a host copy; it does not run on device and
  is not part of the GPU 24706-step run, which therefore advances without per-step projection.
  The compiled projector (ADC-275, hook ADC-177) is the prerequisite for a high-Ma production
  run that is both device-resident and realizability-stable.
- Distributed multi-box Cartesian halo exchange: the MPI smoke is single-box (rank 0 only),
  so it validates collective correctness, not halo exchange or load balancing. Distributed
  multi-box belongs to AMR.
- The diocotron growth rate vs a long MATLAB golden: dedicated campaign, out of CI.

## Limitations

- The closure is exact on Gaussians, but the scheme does not preserve realizability: long
  runs => project (realizability section).
- The realizability projection is a Python per-cell oracle; a compiled device-safe projector
  is blocked on two ADC-177/DSL gaps (ADC-275, see the realizability section and
  [notes/native_projector_feasibility.md](notes/native_projector_feasibility.md)).
- `crossing_state(r != 0)` raises `NotImplementedError`; the MATLAB parity exists but the gate
  is not yet removed (ADC-274).
- The source dt bound is ~500x laxer than `compute_dt.m` and never bites: no MATLAB dt
  fidelity (ADC-197).
- A generic BGK relaxation toward the local Maxwellian is wired (`build_moment_model(collision=True)`,
  ADC-277); the full anisotropic `collision15.m` (Kn-dependent branches) is not yet fidelity-matched.
- Transport with the native relaxation15 projector active is frozen by a code-anchored non-regression
  golden ([run_golden_transport_relax.py](run_golden_transport_relax.py), ADC-203), but a
  MATLAB-anchored transport x relaxation cross-check is still missing (the golden pins our own
  trajectory, not MATLAB fidelity).
- `riemann="hllc"`/`"roe"` unavailable: no contact wave nor closed eigenstructure for this
  system.
- Diocotron growth rate vs a long MATLAB golden: dedicated campaign, out of CI.

## Structuring Linear issues

- [ADC-274](https://linear.app/romain7522/issue/ADC-274) crossing `r != 0` parity with `InitializeM4_15`
- [ADC-275](https://linear.app/romain7522/issue/ADC-275) native compiled `relaxation15` projector (replaces the Python per-cell loop)
- [ADC-177](https://linear.app/romain7522/issue/ADC-177) generic post-step projection hook in the core (adc_cpp)
- [ADC-197](https://linear.app/romain7522/issue/ADC-197) source dt bound vs `compute_dt.m`
- [ADC-203](https://linear.app/romain7522/issue/ADC-203) spatial golden with relaxation active, independent Isserlis oracle, per-state rtol
- [ADC-277](https://linear.app/romain7522/issue/ADC-277) scope and port `collision15.m` / BGK

M8 (`RieMOM2D_Electrostatic_periodic` reference pivot, see [matlab_ref/REFERENCE.md](matlab_ref/REFERENCE.md)):

- [ADC-348](https://linear.app/romain7522/issue/ADC-348) lock the canonical Matlab reference and the fidelity plan
- [ADC-349](https://linear.app/romain7522/issue/ADC-349) shared `matlab_ref` layer (params, Jacobians, init, dt, L2)
- [ADC-350](https://linear.app/romain7522/issue/ADC-350) generate goldens from `RieMOM2D_Electrostatic_periodic`
- [ADC-351](https://linear.app/romain7522/issue/ADC-351) align the diocotron driver on the new periodic Matlab
- [ADC-356](https://linear.app/romain7522/issue/ADC-356) audit the adc_cpp gaps before any core change
