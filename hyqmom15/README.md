# hyqmom15: 2D Vlasov-Poisson with the 15-moment HyQMOM closure

A 2D kinetic solver built on the `adc` engine. It transports the velocity moments
`M_pq = integral f v_x^p v_y^q dv` of order `p+q <= 4` (15 components) of the Vlasov
equation, coupled to the system's Poisson equation:

```text
d_t M_pq + d_x M_{p+1,q} + d_y M_{p,q+1}
   = (q/m) (p Ex M_{p-1,q} + q Ey M_{p,q-1}) + Omega_c (p M_{p-1,q+1} - q M_{p+1,q-1})
```

The flux at the highest order needs order-5 moments absent from the state: the closure
problem. The HyQMOM closure (Bryngelson, Fox and Laurent 2025, hal-05398171) expresses the
six standardized order-5 moments from the lower orders and makes the system hyperbolic. The
state vector order matches the MATLAB reference RIEMOM2D:

```text
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
```

This package is the Python case suite (drivers, validation, analysis, the ROMEO campaign).
The numerics live in adc_cpp; the physics written here is only the closure.

## Quick start

Build the `adc` module first (point `PYTHONPATH` at an adc_cpp `build-*/python`), then:

```bash
# validation drivers (each asserts against the MATLAB reference)
python hyqmom15/runs/run.py                       # flux/closure vs goldens + Gaussian oracle
python hyqmom15/runs/run_waves.py                 # exact wave speeds vs goldens
python hyqmom15/runs/run_relaxation.py            # realizability projection, crossing Ma=20
python hyqmom15/runs/run_diocotron_periodic.py    # full Vlasov-Poisson: the diocotron ring

# build-free checks (no adc build needed)
python hyqmom15/matlab_ref/check_reference.py     # the REFERENCE.md contract
python hyqmom15/matlab_ref/check_goldens.py       # the layer vs the Octave goldens
python hyqmom15/diagnostics/check_diagnostics.py  # realizability + symmetry diagnostics
python hyqmom15/campaigns/check_campaign.py        # campaign infrastructure
python check_cases.py                              # manifest + README lint (from the repo root)
```

## The model

Everything goes through `build_moment_model` ([model.py](model.py)), which delegates the
moment algebra to the generic `adc.moments` generator. You supply the closure (a callable
that maps standardized moments to the order-5 ones); writing another moment system means
supplying another closure with the same contract.

```python
from model import build_moment_model, hyqmom_closure

m = build_moment_model(
    closure=hyqmom_closure,   # the physics: standardized orders 2-4 -> S50..S05
    exact_speeds=True,        # wave speeds from the Jacobian eigenvalues (faithful HLL)
    with_sources=True,        # electric (reads grad phi) + magnetic (omega_c) sources
    debye=0.04,               # Poisson coupling: laplacian(phi) = (M00 - rho_background)/debye^2
    rho_background=rho_bg,     # neutralizing background (mean of M00; required when periodic)
    projection=False,         # True emits the native realizability projector (see below)
)
compiled = m.compile(so_path, include_dir, backend="production")
sim = adc.System(n=128, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                 time=adc.Explicit(method="euler"))
sim.set_poisson(rhs="charge_density", solver="fft")
```

## Cases

Five reference cases on the periodic domain `[-0.5, 0.5]^2`, ported from the MATLAB
RieMOM2D_Electrostatic_periodic suite. `Np` is the per-case mesh resolution.

| case | Np | riemann | sources | notes |
|---|---|---|---|---|
| `constant` | 64 | HLL / MUSCL | none | uniform-state non-regression |
| `fluid_wave` | 32 | ROE | none | fluid eigenmode, no Poisson |
| `electrostatic_wave` | 128 | HLL | E + Poisson | electrostatic eigenmode |
| `magnetic_wave` | 256 | HLL | E + B + Poisson | magnetic eigenmode |
| `diocotron` | 128 | HLL | E + B + Poisson | the diocotron rollup |

Drivers are under [runs/](runs); each is a single case wired to its MATLAB initial condition.

## Validation

The closure, flux assembly, per-block wave speeds, Lorentz sources, the Euler trajectory,
and the Poisson coupling are each proven against the MATLAB reference to machine precision
(roughly 1e-16 to 1e-11 depending on conditioning). The proven / partial / missing map, the
exact tolerances, and the golden-regeneration recipe live in
[matlab_ref/REFERENCE.md](matlab_ref/REFERENCE.md) -- this README does not duplicate them.
The `matlab_ref/` layer is the Matlab-faithful reference (initializers, the dt policy, the
linearized eigenmodes); the build-free `check_*.py` scripts guard it without an adc build.

## Realizability

A moment vector must stay that of a positive distribution. The scheme does not preserve this,
so long runs project. Two implementations, validated branch by branch against each other:

- **Oracle**: `relax_field` ([relaxation.py](relaxation.py)) is the per-cell Python reference
  (a port of `relaxation15.m`), used for validation.
- **Native projector**: `build_moment_model(projection=True)` emits `build_projection`
  ([model.py](model.py)), the compiled device-safe projector. It runs as the System post-step
  hook with no per-cell Python callback and reproduces `relax_field`. See
  [notes/native_projector.md](notes/native_projector.md).

The diocotron rollup is the case that needs it: without the projector it leaves the realizable
set (see Campaign results).

## Analysis, figures, and ParaView

The campaign ([campaigns/](campaigns)) runs the five cases, writes
`adc.System.write` snapshots plus the native ParaView `.vti` per snapshot and a
`run_meta.json` provenance sidecar, and monitors realizability and symmetry at the snapshot
interval (non-fatal). Post-processing turns a run into exploitable artefacts:

- `diagnostics/` -- realizability margin and symmetry residuals, decoupled from the solver.
- `plots/` -- Matlab-like figures (density, phi, per-moment, velocities, realizability maps
  and series, symmetry) plus GIF animations. See [plots/README.md](plots/README.md).
- `campaigns/export_h5.py` -- one consolidated HDF5 per case; `to_paraview.py` -- an enriched
  ParaView time series (density, velocities, the realizability margin, the 15 moments) plus a
  `.pvd` collection; `make_rapport.py` -- a per-case report. See
  [campaigns/README.md](campaigns/README.md).

## Campaign results

A full ROMEO run (x64cpu, Kokkos OpenMP) of the five cases:

- **Four of five** (`constant`, `fluid_wave`, `electrostatic_wave`, `magnetic_wave`) stay
  fully realizable and conserve mass to ~1e-16. The wave cases are source-CFL-limited (dt
  ~5e-6), so they take tens of thousands of steps at full resolution.
- **`diocotron` loses realizability without the projector** (`projection=False`): the
  smallest `p2p2` eigenvalue goes strongly negative, every cell becomes non-realizable, and
  M00 hits the positivity floor. Re-running with the native projector
  (`romeo_rie_mom2d.py --projection`) keeps it realizable, which is exactly what the
  realizability monitoring is for.

## Limitations

- `crossing_state(r != 0)` raises `NotImplementedError`; the MATLAB parity exists but the gate
  is not yet removed.
- The source dt bound is laxer than `compute_dt.m` and never bites; no MATLAB dt fidelity.
- The diocotron growth rate versus a long MATLAB golden is a dedicated campaign, out of CI.
- GPU/MPI scale is validated separately; the Cartesian System is mono-box.

## Layout

```text
matlab_ref/   Matlab-faithful reference layer + REFERENCE.md (the fidelity contract)
runs/         per-case validation drivers
diagnostics/  realizability + symmetry diagnostics
plots/        figures and animations
campaigns/    the ROMEO campaign, HDF5/ParaView export, the per-case report
notes/        design notes (native projector, BGK scoping)
```
