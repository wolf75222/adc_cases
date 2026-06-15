# adc_cases

**The named Python use cases that drive the `adc` solver.**

`adc_cases` holds the named scenarios: diocotron, Euler-Poisson, two-fluid, and more. The generic
physics and the cell-by-cell compute live in the engine,
[adc_cpp](https://github.com/wolf75222/adc_cpp), and are exposed by its `adc` Python module
(pybind11). Here, Python composes and drives: one folder per case, each importing `adc`, writing
its initial conditions in numpy, and running the simulation. The core names no scenario, so the
names live here.

Most cases are pure Python. A bespoke case (for example [`two_fluid_ap/`](two_fluid_ap/), an
asymptotic-preserving solver) can carry its own C++: it is compiled on the fly against the
`adc_cpp` headers and loaded through `ctypes`. The shared machinery (header location, compilation,
ABI check) lives in `adc_cases.common.native`; build artifacts go under `out/<case>/build/`.

## Quickstart

Build the `adc` module from `adc_cpp` (Kokkos-only; CMake fetches and builds Kokkos if none is
installed), put it on the path, and run a case:

```bash
# 1. Build the module (from adc_cpp)
cd ../adc_cpp
cmake -B build-py -DADC_BUILD_PYTHON=ON          # Kokkos fetched and built automatically (Serial)
cmake --build build-py --target _adc -j
export PYTHONPATH=$PWD/build-py/python           # the directory that holds the adc/ package

# 2. Run a case (from adc_cases)
cd ../adc_cases
python3 euler_poisson/run.py
```

Run a case with the same Python interpreter that built the module (the extension carries a
`cpython-3XY` ABI suffix). With several Pythons installed, pin the build with
`-DPython_EXECUTABLE=$(which python3.12)`. To reuse an existing Kokkos install (faster, and
required at runtime for the `production` and native DSL cases, see `ADC_KOKKOS_ROOT`), pass
`-DKokkos_ROOT=...` to the configure step.

To import the shared package without touching `sys.path`, install it once, editable:

```bash
cd adc_cases
pip install -e .               # pulls numpy
pip install -e '.[figures]'    # plus matplotlib (the diocotron reproduction)
```

Without the install, each case still runs directly: a short preamble puts the repository root on
the import path only when the package is not installed. Dependencies: `numpy`; `matplotlib` for
the diocotron reproduction; a C++20 compiler for `two_fluid_ap` and the DSL cases.

## The cases

One folder per case, each with its own `README.md` (objective, equations, model, method, file
map, exact command, initial conditions, invariants, outputs, cost, limits, CI). The table below
covers the 16 cases; the *Category* column comes from the [manifest](cases_manifest.toml). The
descriptions are honest: a `reproduction-candidate` is not an established reproduction, and an
`experimental` case is an unfinished prototype.

| Folder | Category | Case | What it shows |
|---|---|---|---|
| [`diocotron/`](diocotron/README.md) | reproduction | Diocotron instability (ExB drift) | Reproduces [arXiv:2510.11808](https://arxiv.org/abs/2510.11808): analytic growth rate (Petri, numpy) vs measured, composed generically via `adc.System` (`ExB` + `BackgroundDensity` bricks + a conducting wall), figures + GIF; long, out of CI. The `band_instability.py` sub-script is a minimal periodic variant (instability growth, no figures), classified `validation` (in CI). |
| [`tutorial/`](tutorial/README.md) | tutoriel | The diocotron written three ways | The same diocotron physics built via the `models.diocotron` helper, via hand-rebuilt native bricks, and via `adc.dsl.Model` formulas: the three final states are bit-identical (`np.array_equal`). The adc_cases mirror of the adc_cpp Sphinx tutorial; figures (growth + GIF + bricks vs DSL). The CI variant `tutorial/equivalence.py` (`validation`, `cxx`) locks the equivalence without figures. |
| [`composition/`](composition/README.md) | tutoriel | Multi-block composition | Electrons (Euler, VanLeer + HLLC, IMEX, 10 substeps) + ions (isothermal, Minmod + Rusanov, explicit); per-block implicit/explicit choice, reversible; guardrails; a time integrator written in Python (`adc.integrate.ssprk2_step`). |
| [`euler_poisson/`](euler_poisson/README.md) | validation | Euler + self-consistent field | Self-gravity (attractive) vs plasma / Langmuir (repulsive); a single coupling sign separates them; mass and momentum conserved. |
| [`multispecies/`](multispecies/README.md) | validation | Two heterogeneous fluids | Euler electrons (4 vars) + isothermal ions (3 vars) coupled by a system Poisson `f = sum_s q_s n_s`; mass conserved per species. |
| [`two_euler/`](two_euler/README.md) | validation | Two independent Euler | Electrons + ions, two uncoupled Euler gases, same bricks (`CompressibleFlux` + HLLC + primitive reconstruction); only the initial conditions differ (lighter electrons run faster); multirate `step_adaptive`. Shows "two Euler, one code". |
| [`plasma/`](plasma/README.md) | validation | Coupled plasma (e + i + n) | Three species sharing one system Poisson (`f = sum_s q_s n_s`), coupled by inter-species sources: ionization (`add_ionization`, n_g -> n_i + n_e) and ion-neutral collision (`add_collision`); electrons in HLLC + primitive reconstruction. n_i + n_g conserved to machine precision. |
| [`two_fluid_ap/`](two_fluid_ap/README.md) | validation (needs `cxx`) | Stiff AP two-fluid | A bespoke AP integrator, not block-composable (the AP stabilization couples the time step to the elliptic): an asymptotic-preserving scheme that stays stable when `dt*omega_pe >> 1` (an explicit one would blow up). A scenario, not a generic brick: its C++ physics (`two_fluid_ap.hpp` + `_two_fluid_ap.cpp`) lives here, compiled on the fly against the generic adc_cpp headers and driven from Python (`ctypes`). |
| [`diocotron_amr/`](diocotron_amr/README.md) | validation | Diocotron on AMR | Composed via `adc.AmrSystem` (the refined counterpart of `System`: `add_block` + `set_refinement`): a hierarchy of dynamically refined patches, conservative reflux. |
| [`custom_scheme/`](custom_scheme/README.md) | tutoriel | Numerical method in Python | Diocotron transport (reconstruction, upwind flux, SSPRK2) written in numpy; `adc` serves only as a Poisson oracle (`set_density` + `solve_fields` + `potential`). Mass conserved to machine precision. |
| [`diocotron_dsl/`](diocotron_dsl/README.md) | validation (needs `cxx`) | Diocotron as formulas (DSL) | The diocotron physics (ExB transport + neutralizing background) written entirely as `adc.dsl.Model` formulas instead of native bricks; `adc.dsl` generates the C++, compiles and installs it via `add_equation`. The crux: the produced state is bit-identical to the native composition (`np.array_equal`), on the same grid / IC / Poisson. `production` backend (native zero-copy), else `aot` (host-marshaled, numerically identical). |
| [`two_species_dsl/`](two_species_dsl/README.md) | validation (needs `cxx`) | Electrons + ions as formulas (DSL) | Two species as formulas (`adc.dsl.Model`): Euler electrons (4 vars) + isothermal ions (3 vars), each with an electrostatic source (reads grad phi via the aux channel) and a charge density, coupled by one Poisson. Per-species equivalence to native: ions bit-identical, electrons to machine-epsilon (float reassociation in the shared Poisson RHS accumulation, tolerance 1e-24). |
| [`magnetic_isothermal_dsl/`](magnetic_isothermal_dsl/README.md) | validation (needs `cxx`) | Magnetized isothermal fluid as formulas (DSL) | A magnetized isothermal fluid as formulas, with a Lorentz force `q rho E + v x B` driven by a constant B_z field read on the extended aux channel (index 3, filled from Python via `set_magnetic_field`). No native reference model: correctness is proven by inter-backend parity (`production` == `aot` when both link) plus a numpy Lorentz oracle and invariants (mass, positivity, momentum rotation). |
| [`schur_magnetized_cartesian/`](schur_magnetized_cartesian/README.md) | experimental (needs `cxx`) | Schur source stage vs explicit | Measures the time-step effect of the Schur-condensed source stage (`adc.Split(Explicit, CondensedSchur)`) against explicit integration of the same stiff Lorentz source, on a cartesian magnetized isothermal fluid (same DSL as `magnetic_isothermal_dsl`). Sweeps the largest stable explicit `dt` vs Schur (theta=0.5 / theta=1.0), reports `dt*omega_c` and the gain. A measurement prototype, out of CI. |
| [`dsl_euler/`](dsl_euler/README.md) | experimental | Euler as formulas (interpreted DSL) | 2D compressible Euler as formulas (the `adc.dsl` mini-DSL), an interpreted CPU prototype: the symbolic tree is evaluated in numpy and wired to the host backend `adc.PythonFlux` (Rusanov, periodic). A user-side declarative demonstrator, not the production path (which stays the compiled bricks). Checks mass conservation, non-trivial acoustic dynamics, a physical state. |
| [`hoffart_euler_poisson_dsl/`](hoffart_euler_poisson_dsl/README.md) | reproduction-candidate pending | Magnetized Euler-Poisson (Hoffart), Schur stage | Targets [arXiv:2510.11808](https://arxiv.org/abs/2510.11808) (the full magnetized Euler-Poisson system, Schur source stage) written in `adc.dsl`. The quantitative reproduction is not yet established (validation table pending): the cartesian baseline is far from the paper and the geometry is suspect (see `adc_cpp/docs/HOFFART_FIDELITY.md`). Out of CI. The `check_model.py` sub-script (`validation`, in CI) is an analytic oracle of the model: flux, Lorentz/electric source, eigenvalues and Poisson RHS checked by assert. |

Run any of them:

```bash
cd ../adc_cases
python3 diocotron/run.py          # arXiv:2510.11808 reproduction (figures + GIF)
python3 composition/run.py        # heterogeneous composition + a Python time integrator
python3 euler_poisson/run.py      # self-gravity vs plasma (Langmuir)
python3 multispecies/run.py       # Euler electrons + isothermal ions, system Poisson
python3 two_euler/run.py          # two independent Euler, same scheme (HLLC + primitive recon)
python3 plasma/run.py             # electrons + ions + neutrals: Poisson + ionization + collision
python3 two_fluid_ap/run.py       # stiff asymptotic-preserving two-fluid
python3 diocotron_amr/run.py      # diocotron on multi-patch AMR
python3 custom_scheme/run.py      # a spatial + temporal scheme written in Python, Poisson by adc
python3 dsl_euler/run.py          # Euler written as formulas (the adc.dsl mini-DSL, experimental)

# DSL cases written as formulas (codegen + C++ compile: a C++20 compiler is required)
python3 diocotron_dsl/run.py              # diocotron as formulas, proven bit-identical to native
python3 two_species_dsl/run.py            # electrons + ions as formulas, coupled Poisson
python3 magnetic_isothermal_dsl/run.py    # magnetized isothermal fluid (Lorentz via B_z) as formulas
python3 schur_magnetized_cartesian/run.py # Schur source stage vs explicit (measurement, experimental)
python3 hoffart_euler_poisson_dsl/run.py  # magnetized Euler-Poisson (Hoffart): reproduction-candidate

# Tutorial: the same diocotron via helper / bricks / formulas (equivalence proven) + figures
python3 tutorial/run.py                   # full tutorial (three equivalent paths) + figures + GIF
python3 tutorial/equivalence.py           # CI smoke: helper == bricks == formulas, bit-identical
```

## The adc_cases package

The repository is itself an importable `adc_cases` package that centralizes what the cases share
(named models, initial conditions, grids, invariants, outputs):

| Module | Contents |
|---|---|
| `adc_cases.models` | named species models = compositions of `adc` bricks (electron_euler, ion_isothermal, diocotron, euler_poisson, euler, neutral_isothermal). One model = one species (`adc.Model`). |
| `adc_cases.recipes` | system recipes = ready-made multi-species configurations (`two_fluid`, `plasma`: blocks + Poisson + couplings). One level above the species models. |
| `adc_cases.common.grid` | cell-centered grids (`meshgrid_xy`), the `field[j, i]` convention of the `adc` facade. |
| `adc_cases.common.initial_conditions` | reused initial conditions: gaussian band (`band_density`), ring (`ring_density`), Euler pressure blob (`euler_pressure_blob`). |
| `adc_cases.common.checks` | invariants used by several cases (`assert_mass_conserved`, `assert_finite`, `assert_positive`, `relative_drift`). |
| `adc_cases.common.io` | the `out/` output directory (out of source, git-ignored). |
| `adc_cases.common.native` | on-the-fly compilation + `ctypes` loading of bespoke C++ scenarios (out-of-source cache under `out/`, explicit ABI check). |

## Outputs

Ephemeral files (working figures, GIFs, `.so`) go under `out/<case>/` (git-ignored, overridable
with `ADC_CASES_OUT`). One exception: `diocotron/run.py` writes its canonical figures to
`diocotron/figures/` (tracked) with a `provenance.json`; each asset's status is in
[`ASSETS.md`](ASSETS.md). The `hoffart_euler_poisson_dsl` figures stay ephemeral (the
`reproduction-candidate` is not yet established).

## Manifest and CI

[`cases_manifest.toml`](cases_manifest.toml) classifies each case by category (`validation`,
`tutoriel`, `reproduction`, `reproduction-candidate`, `experimental`) and records whether it runs
in CI (`ci = true`). CI runs only the light cases (`ci = true`); the long ones (the figure/GIF
diocotron reproduction), the not-yet-established candidates (`hoffart_euler_poisson_dsl/run.py`)
and the experimental ones (`dsl_euler` interpreted DSL, `schur_magnetized_cartesian`) stay out of
CI and run by hand.

## Composition API

Two levels of composition: generic (`adc.System`: blocks sharing one system Poisson, ideal for
coupling ions / electrons / neutrals) and on AMR (`adc.AmrSystem`, the same API plus
`set_refinement`). A model is a composition `adc.Model(state, transport, source, elliptic)`; the
named compositions (`diocotron`, `electron_euler`, ...) are in [`models.py`](adc_cases/models.py). A bespoke
scenario (the two-fluid AP) carries its own C++. For the engine, see
[adc_cpp](https://github.com/wolf75222/adc_cpp) and its
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

See also [CONTRIBUTING.md](CONTRIBUTING.md).
