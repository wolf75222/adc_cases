# hyqmom15: 2D Vlasov-Poisson with 15 moments (HyQMOM closure)

A 2D kinetic model: you transport the velocity moments `M_pq = integral f v_x^p v_y^q dv`
of order p+q <= 4 (15 components) of the Vlasov equation, coupled to the system's Poisson
equation. For each moment,

```
∂t M_pq + ∂x M_{p+1,q} + ∂y M_{p,q+1} = q/m (p Ex M_{p-1,q} + q Ey M_{p,q-1})
                                        + Ωc (p M_{p-1,q+1} − q M_{p+1,q-1})
```

The flux at the highest order brings in order-5 moments that are absent from the state
vector: this is the closure problem. The HyQMOM closure (Bryngelson, Fox & Laurent 2025,
hal-05398171) expresses the six standardized order-5 moments as functions of the lower
orders and makes the system hyperbolic.

State (component order shared with the MATLAB reference RIEMOM2D):

```
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
| Partial | realizability under transport | the per-cell Python `relax_field` is an oracle, not a scalable path (ADC-275); the native compiled projector is not yet wired (depends on ADC-177) |
| Partial | correlated crossing IC (`r != 0`) | blocked by `NotImplementedError` in `crossing_state` ([model.py](model.py)); Octave shows `gaussian_state` === `InitializeM4_15` for `r != 0`, so the gate is removable, not a divergence (ADC-274) |
| Partial | golden coverage | no spatial golden with relaxation active (transport x relaxation); the HLL golden runs `flagrelax=0` Ma=2, the relax golden runs the projection isolated (ADC-203) |
| Missing | source dt bound === `compute_dt.m` | our `omega_p` bound is ~500x laxer than the MATLAB source CFL and never bites (ADC-197, section below) |
| Missing | BGK collision (`collision15.m`) | not ported; MATLAB branches with `Kn <= 10` are not yet fidelity-covered (ADC-277) |
| Missing | diocotron growth rate vs a long MATLAB golden | dedicated campaign, out of CI |

Use the drivers above for validation. Use the native compiled path (`backend="aot"`, exact
speeds, Poisson `geometric_mg`) for production runs, with the caveat that the realizability
projection is still a Python per-cell loop (section below).

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
branch; the "complex eigenvalues" test evaluates the model's autodiff Jacobian. Per-field
usage:

```python
from relaxation import make_corner_eigs, relax_field
fn = make_corner_eigs()
U = relax_field(U, lamin=1e-12, Ma=4.0, corner_eigs=fn)   # (15, ny, nx) -> projeté
```

Two paths, two purposes:

- Oracle (current): `relax_field` is a per-cell Python loop with a host copy and a
  `numpy.linalg.eigvals` per cell. It is the reference for validation (=== Octave to 4e-14,
  `run_relaxation.py`) and the source of truth for the native port. It is not scalable: not
  usable inside a GPU/MPI time step (ADC-275).
- Native projector (expected, not yet wired): a compiled device-safe `U -> U_projected`
  applied after each whole step (the MATLAB `flagrelax=1` semantics are per-step, not
  per-RK-stage). It depends on the generic post-step projection hook ADC-177 (adc_cpp PR #318);
  the HyQMOM15 formulas stay on the case side, only the hook is core (ADC-275). Until it
  lands, production GPU runs apply no projection in the step and must keep Ma moderate or
  accept the dt collapse above.

`relaxation15` is a relaxation toward a target, not an idempotent projector: re-relaxing
relaxes again (verified under Octave); the drivers assert no idempotence, faithful to MATLAB.

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
  is not yet wired (ADC-275, depends on ADC-177).
- `crossing_state(r != 0)` raises `NotImplementedError`; the MATLAB parity exists but the gate
  is not yet removed (ADC-274).
- The source dt bound is ~500x laxer than `compute_dt.m` and never bites: no MATLAB dt
  fidelity (ADC-197).
- The MATLAB BGK collision (`collision15.m`) is not ported: branches with `Kn <= 10` are not
  fidelity-covered (ADC-277).
- No spatial golden with relaxation active (transport x relaxation interaction) (ADC-203).
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
