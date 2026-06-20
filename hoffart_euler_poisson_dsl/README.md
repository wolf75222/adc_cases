# hoffart_euler_poisson_dsl

A complete tutorial, from install to run, that reproduces the magnetic diocotron test case of
Hoffart, Maier, Shadid, Tomas, *Structure-preserving finite element approximations of the magnetic
Euler-Poisson equations* (arXiv:2510.11808, section 5.3), with the `adc_cpp` finite-volume core driven
in Python by `adc_cases`.

You write the full isothermal magnetized Euler-Poisson model (continuity, momentum with the Lorentz
force, Poisson) once in the symbolic DSL, compile it to C++, then advance it with a Strang splitting
(SSPRK3 + a Schur-complement source stage). Measured in the right units, the Cartesian finite-volume path
reproduces the paper's growth rates to within 10 %, and converges to them as you refine the grid.

![Animation of the l=4 diocotron rollup](figures/diocotron_l4.gif)

The electron ring perturbed at mode 4 deforms into a square, then rolls up into four vortices, like
figure 5.2 of the paper. This README explains everything, from compilation to this animation.

## Contents

1. [The result](#1-the-result)
2. [Installation](#2-installation)
3. [Quickstart](#3-quickstart)
4. [The physics](#4-the-physics)
5. [The model in the DSL](#5-the-model-in-the-dsl)
6. [The run](#6-the-run)
7. [Measurement and the lesson of the 2pi factor](#7-measurement-and-the-lesson-of-the-2pi-factor)
8. [The figures you get](#8-the-figures-you-get)
9. [Convergence](#9-convergence)
10. [Performance and scaling (local + ROMEO)](#10-performance-and-scaling-local--romeo)
11. [Folder structure](#11-folder-structure)

## 1. The result

Growth rates of the full `system-schur` model (n=96, paper windows mapped into simulation time,
conversion `gamma_paper = gamma_raw_sim * 2pi/rhobar`):

| mode l | gamma_raw_sim | gamma_paper (x2pi) | paper target | error |
|---|---|---|---|---|
| 3 | 0.1117 | 0.702 | 0.772 | -9.1 % |
| 4 | 0.1423 | 0.894 | 0.911 | -1.9 % |
| 5 | 0.1087 | 0.683 | 0.683 | +0.04 % |

The error shrinks with resolution: at n=256 all three modes drop below 1 % (section 9). The "-95 %
deficit" of earlier versions of this case was a metrology artifact, explained in section 7.

## 2. Installation

The case needs the `adc` Python module, provided by the `adc_cpp` repository. Prerequisites: a C++20
compiler, CMake, Ninja, Python 3.12 with NumPy, and an **installed Kokkos** (`adc_cpp` is Kokkos-only: a
`Serial` Kokkos is enough for a CPU workstation). Matplotlib and Pillow are optional (figures and GIF).

```bash
# 1. build the adc module (from the adc_cpp repo) -- Kokkos-only: Kokkos required (-DKokkos_ROOT)
cd adc_cpp
cmake -B build -G Ninja \
      -DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3)
ninja -C build _adc

# 2. make adc importable
export PYTHONPATH=$PWD/build/python

# 3. check the import
python -c "import adc; print('adc OK')"
```

Note: `run.py` compiles the DSL model to C++ on the fly, which needs the `adc_cpp` headers. If the
compilation does not find them, point `ADC_INCLUDE=<adc_cpp>/include`. The reduced polar path (`Scalar`,
`ExB`, `ChargeDensity` bricks) compiles nothing and runs against any build.

## 3. Quickstart

```bash
cd adc_cases/hoffart_euler_poisson_dsl

# a) analytic oracle, no simulation: the compiled model == the hand formulas,
#    and the analytic eigenvalue reproduces the paper targets
python check_model.py
python diag/petri_eigenvalue.py

# b) the rate table (full model, paper-faithful measurement).
#    t-end >= 8.5 because the mapped window of mode 5 is [7.23, 8.48]
python run.py --engine system-schur --n 96 --t-end 10 --modes 3 4 5 --dt 2e-3 --no-gif

# b') same run, plus the raw state dumped as state_<NNNNNN>.npz next to each mode's outputs
python run.py --engine system-schur --n 96 --t-end 10 --modes 3 4 5 --dt 2e-3 --no-gif --dump-npz

# c) the normalization audit and the resolution convergence
python diag/diag_normalization_audit.py 128
python diag/convergence_reduced.py

# d) the figures and GIFs of section 8
python diag/make_paper_figures.py 3 4 5 --out figures
```

Output b) writes `growth_rates.csv` with the columns `mode, gamma_raw_sim, gamma_paper_units,
gamma_paper, relative_error_percent`.

`--dump-npz` (off by default) additionally writes the raw simulation state to
`out/<case>/mode_<l>/state_<NNNNNN>.npz` at the same times as the schlieren snapshots, via
`sim.write(format="npz", step=<snapshot_index>)` (per-block conservative fields, `phi`, the clock).
The simulation trajectory and the `amplitude.csv` / `growth_rates.csv` / PNG / GIF outputs are
unchanged; `metadata.json` additionally records the `dump_npz` flag. The dump is single-rank only:
`sim.write` gathers then writes, so under MPI (np>1, the `amr-imex` path) it is disabled with a
message rather than racing on the file.

## 4. The physics

A non-neutral electron column, in a uniform axial magnetic field, rotates under its own `ExB` drift. When
the density has a ring shape (hollow at the center), the two edges carry density jumps of opposite signs.
These two interfaces couple through the perturbed electric field and amplify each other: this is the
Kelvin-Helmholtz mechanism applied to `ExB` rotation, called here the diocotron instability. The azimuthal
mode `l` grows exponentially, then the ring folds into `l` vortices (the animations in section 8 show it).

The paper works in the magnetic drift limit: the field is so strong that the cyclotron and plasma time
scales are orders of magnitude faster than the slow drift. The scheme must step over these fast scales
without resolving them, which the implicit source stage allows (section 6).

The system, with `Omega = omega e_z` so `m x Omega = (omega m_y, -omega m_x)`:

```
d_t rho + div(m)                          = 0
d_t m   + div(m m^T/rho + p I)            = -rho grad(phi) + m x Omega
-Delta phi = alpha rho,   p = theta rho
```

## 5. The model in the DSL

This is the central point of `adc`: you write the physics once, in symbols, and the DSL derives the
Riemann solver and generates the C++ kernel. Here is `model.py` (the `schur` variant), block by block.

First the paper parameters. One property deserves attention: `alpha/omega = 1/rho_max = 1`. The two
`1e12` cancel in the drift `v = grad(phi)/omega`, so the field that advects the density does not depend on
`beta`. The full model therefore advects the density with the same field as a normalized `ExB` drift.
Section 7 builds on this fact.

```python
@dataclass(frozen=True)
class PaperParameters:
    radius = 16.0; ring_inner = 6.0; ring_outer = 8.0   # R, r0, r1
    rho_min = 1.0e-6; rho_max = 1.0; beta = 1.0e6        # densités, échelle magnétique
    perturbation = 0.1; temperature = 0.0               # delta du sin(l theta), theta (limite froide)

    @property
    def alpha(self): return self.beta * self.beta / self.rho_max   # = 1e12, couplage de Poisson
    @property
    def omega(self): return self.beta * self.beta                  # = 1e12, champ B_z (= |Omega|)
```

The model itself reads like a lab-exercise statement. You declare the conservative unknowns, define the
primitives from them, write the Euler flux component by component, give the eigenvalues to the Riemann
solver, declare the auxiliary fields that Poisson fills, write the source (electric force plus Lorentz),
then Gauss's law.

```python
m = dsl.Model("hoffart_magnetic_euler_poisson_schur")

# inconnues conservatives : densité et quantité de mouvement (pas d'énergie, modèle barotrope)
rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v",
                                  roles=["Density", "MomentumX", "MomentumY"])

# primitives définies à partir des conservatives ; le DSL gère la conversion prim<->cons
u = m.primitive("u", mx / rho)
v = m.primitive("v", my / rho)
pressure = m.primitive("p", params.temperature * rho)   # p = theta rho
m.primitive_vars(rho, u, v)
m.conservative_from([rho, rho * u, rho * v])

# le flux d'Euler, comme au tableau : masse = m ; quantité de mouvement = m u + p
m.flux(x=[mx, mx * u + pressure, mx * v],
       y=[my, my * u, my * v + pressure])

# les vitesses d'onde u, u +/- c (c = sqrt(theta)) pour la dissipation du solveur de Riemann
sound_speed = dsl.sqrt(params.temperature)
m.eigenvalues(x=[u - sound_speed, u, u + sound_speed],
              y=[v - sound_speed, v, v + sound_speed])

# champs auxiliaires remplis par le solveur de Poisson, pas avancés par le flux
m.aux("phi"); grad_x = m.aux("grad_x"); grad_y = m.aux("grad_y")

# source nulle ici : l'étage CondensedSchur avance la force électrique + Lorentz (chemin de référence)
m.source([0.0 * rho, 0.0 * mx, 0.0 * my])

# loi de Gauss -Delta phi = alpha rho (le solveur résout Delta phi = rhs, d'où le signe)
alpha = m.param("alpha", params.alpha)
m.elliptic_rhs(-alpha * rho)
m.check()
```

From these calls, `model.compile(backend="production")` produces a C++ `.so`: the numerical flux, the
Riemann solver, the auxiliary derivation, all of it is generated. You write no loop. The fidelity of this
generation is checked by `check_model.py`, which compares the compiled kernel to the hand formulas on 2x2
cells and finds an exactly zero residual. This is the case's clean boundary: the model generation is
proven bit for bit; physical reproduction is then measured by the run.

The code of `model.py` carries these explanations as comments, step by step (the eight blocks above). The
initial density (equation 35, the perturbed ring `rho_max(1 - delta + delta sin(l theta))`) and the
initial `ExB` drift `v0 = -(grad phi0 x Omega)/|Omega|^2` are in `paper_initial_density` and
`drift_velocity_from_potential`.

## 6. The run

`run.py:build_uniform` assembles the reference path. Each line has a role.

```python
def build_uniform(compiled, rho, params, geometry="square"):
    sim = adc.System(n=n, L=params.length, periodic=False)            # grille carrée n×n, côté 2R
    sim.set_poisson(rhs="composite", solver="geometric_mg",           # Poisson multigrille
                    bc="dirichlet", wall="circle", wall_radius=params.radius)   # paroi disque R
    sim.set_magnetic_field(params.omega * np.ones_like(rho))          # B_z uniforme, avant Schur
    sim.add_equation("electrons", model=compiled,
        spatial=adc.FiniteVolume(limiter="weno5", riemann="rusanov", variables="conservative"),
        time=adc.Strang(hyperbolic=adc.Explicit(method="ssprk3"),     # H(dt/2) ; S(dt) ; H(dt/2)
                        source=adc.CondensedSchur(theta=0.5, alpha=params.alpha)))
    # relaxation à deux passes du papier : poser rho, résoudre phi, en déduire la dérive v0,
    # réinstaller l'état avec v0, résoudre phi à nouveau -> état initial cohérent
    sim.set_primitive_state("electrons", rho=rho, u=zeros, v=zeros); sim.solve_fields()
    u0, v0 = drift_velocity_from_potential(np.asarray(sim.potential()), params)
    sim.set_primitive_state("electrons", rho=rho, u=u0, v=v0);        sim.solve_fields()
    return sim
```

- Square grid of side `L = 2R = 32`, non-periodic edges. The paper's disk is approximated by the circular
  Poisson wall of radius `R`.
- WENO5-Z finite volumes, Rusanov flux, conservative variables, integrated in SSPRK3.
- The Strang splitting does half-transport, full source, half-transport (order 2, like the paper).

The source stage `adc.CondensedSchur(theta=0.5, alpha=...)` advances the source implicitly, which steps
over the cyclotron and plasma scales without resolving them. The Lorentz force is inverted by a 2x2
eliminator

```
B^-1 = 1/(1+w^2) [[1, w], [-w, 1]],   w = theta dt B_z,
```

and the condensed elliptic operator is `A = I + c rho B^-1` with `c = theta^2 dt^2 alpha`. You solve `A`
for `phi^{n+theta}` (multigrid-preconditioned BiCGStab), then reconstruct the momentum `v^{n+theta} = B^-1
(v^n - theta dt grad phi^{n+theta})`.

## 7. Measurement and the lesson of the 2pi factor

The solver produced the right result from the start. The comparison to the paper was wrong on two points,
both the same `2pi` factor.

Origin of the `2pi`. Davidson's linear theory (reference [13] of the paper) gives the targets `gamma_3 =
0.772`, `gamma_4 = 0.911`, `gamma_5 = 0.683` from a 2x2 eigenvalue problem on the two edges of the ring.
The paper expresses the diocotron frequency `omega_d = 1` in cyclic form (one turn per period), but the
dispersion relation works with an angular frequency (one turn is `2pi` radians). The `2pi` is this
conversion. `diag/petri_eigenvalue.py` checks it: with `Wd = 2pi omega_d` it reproduces the three targets
to within 0.5 %, and with `Wd = omega_d = 1` it returns exactly the targets divided by `2pi`.

The numerical solver runs in the natural `ExB` clock, so `gamma_paper = gamma_raw_sim * 2pi/rhobar`
(rhobar = rho_max = 1). This is what `gamma_to_paper_units` does. And since `alpha/omega = 1` (section 5),
this factor applies to the full model just as to the reduced transport.

```python
def paper_to_sim_time_window(window_paper, rhobar=1.0):
    scale = 2.0 * math.pi / rhobar          # t_sim = (2pi/rhobar) * t_paper
    return (window_paper[0] * scale, window_paper[1] * scale)

def gamma_to_paper_units(gamma_raw_sim, rhobar=1.0):
    return gamma_raw_sim * (2.0 * math.pi / rhobar)
```

The second error was the fit window. The paper's windows are in paper time, but were applied to simulation
time. Mode 3's window `[0.40, 0.70]` corresponds to `t_sim in [2.51, 4.40]`, not to `[0.40, 0.70]`;
applied as is, it measures the transient, where the rate has not yet reached its exponential value.
`fit_growth` therefore maps the window with `paper_to_sim_time_window` before the fit.

Breakdown of mode 3's deficit (`0.0312 -> 0.772`, factor 24.7): window 3.20, then `2pi = 6.28`, then a
Cartesian-versus-polar grid residual of 1.23. The product `3.20 x 6.28 x 1.23` equals 24.7, the observed
deficit. Only the grid residual is physical, and it tends to zero with resolution (section 9). Details in
[`docs/T2_NORMALIZATION_AUDIT.md`](docs/T2_NORMALIZATION_AUDIT.md) and
[`docs/RESULTS_SYSTEM_SCHUR.md`](docs/RESULTS_SYSTEM_SCHUR.md).

## 8. The figures you get

Schlieren snapshots of the density **of the full `system-schur` model** (n=96, minmod reconstruction),
paper palette (white disk, slate exterior, Blues colormap), at the time fractions `0.01, 1/8, ..., 7/8,
t_f`. The number of vortices equals the mode.

Mode l=3 (figure 5.1 of the paper): triangle, then three arms, then three vortices.

![Snapshots l=3](figures/snapshots_l3.png)

Mode l=4 (figure 5.2): square, then four vortices.

![Snapshots l=4](figures/snapshots_l4.png)

Mode l=5 (figure 5.3): pentagon, five-pointed star, then five vortices in a crown.

![Snapshots l=5](figures/snapshots_l5.png)

Matching animations: `figures/diocotron_l3.gif`, `figures/diocotron_l4.gif` (at the top of the page),
`figures/diocotron_l5.gif`. They show the ring rotating, the mode growing, then the fold into vortices and
the stretching of the filaments.

Growth rates, figure 5.4 style. Panels (a,b,c): amplitude `|c_l(t)|/|c_l(0)|` on a log scale, the curve
follows the paper slope (red dashes) within the mapped fit window, then saturates. Panel (d): `gamma_l`
against the mode, for the paper, the full model, and the reduced ExB drift.

![Growth rates](figures/growth_rate.png)

The snapshots and GIFs are the **actual** density of the full `system-schur` model, advanced in minmod
(TVD) reconstruction: WENO5 overshoots at the ring's top-hat jump, the density goes negative and the run
collapses around t~0.38 t_f (dt->0 or NaN, see ADC-62/ADC-74); minmod keeps `rho > 0` and reaches the full
rollup, at the cost of more smeared filaments. The raw state (density + phi) of each snapshot is dumped as
a reusable `.npz` via `sim.write` (`out/hoffart_paper_figures/mode_*/`); the main run reproduces this
on demand with `run.py --dump-npz` (section 3). The amplitude curves (a,b,c) of
growth_rate use the reduced `ExB` drift (same advection field, `alpha/omega = 1`); the rates of panel (d)
come from the full `system-schur` model. The generator is `diag/make_paper_figures.py`.

## 9. Convergence

The relative error to the paper tends to zero as the grid refines. The low-resolution residual was the
Cartesian discretization of the ring edge, not a lock.

![Convergence](figures/convergence.png)

| n | l=3 | l=4 | l=5 |
|---|---|---|---|
| 64 | -13.7 % | -13.8 % | -0.1 % |
| 128 | -3.8 % | -4.7 % | +0.6 % |
| 256 | -0.6 % | +0.2 % | -0.7 % |

At n=256 all three modes reproduce the paper to within 1 %.

## 10. Performance and scaling (local + ROMEO)

### Local cost (1 core)

The reference local build is sequential (Kokkos Serial), so a single thread even on an 8-core machine;
multi-threading goes through a Kokkos OpenMP install. For the full `system-schur` model with the Schur
Krylov solve, dt=2e-3, t_end=10 (5000 steps), per mode, on Apple Silicon arm64:

| n | wall (1 core) | gamma_paper (l=3) | error |
|---|---|---|---|
| 96 | 120 s | 0.702 | -9.1 % |
| 128 | 246 s | 0.729 | -5.6 % |
| 192 | 490 s | 0.746 | -3.4 % |

![Local cost and convergence](figures/perf_local.png)

### ROMEO: thread scaling

On ROMEO (URCA, x64cpu partition, AMD EPYC 9654, OpenMP, account r250127), a 3-level AMR diocotron (direct
g++ build `-fopenmp`, header-only) measured at resolution 512, 60 steps, from 1 to 96 threads on one node:

| threads | 1 | 6 | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|
| wall (s) | 10.9 | 7.4 | 7.0 | 6.6 | 6.8 | 7.4 |
| speedup | 1.0 | 1.5 | 1.6 | 1.6 | 1.6 | 1.5 |

![ROMEO thread scaling](figures/romeo_scaling.png)

The speedup saturates around 1.6x at 24 threads, then degrades. This is not an implementation flaw: the
diocotron step is dominated by fine-grained kernels (multigrid Poisson + flux on small AMR patches),
hardly parallelizable at these sizes. The same finding is in the ROMEO log (`HERO_RESULTS.md`): at these
sizes, the step runs better on moderate multi-core CPU than by saturating many cores or a GPU. To scale
hard, you need much larger problems.

## 11. Folder structure

```
hoffart_euler_poisson_dsl/
├── model.py        modèle Euler-Poisson magnétisé en DSL (commenté), paramètres, densité + dérive initiales
├── run.py          CAS cartésien system-schur : assemblage, mesure paper-faithful (fenêtres mappées,
│                   conversion 2π/rhobar), sorties (amplitude, snapshots, GIF, table des taux)
├── run_polar.py    CAS polaire (anneau résolu) : diverge encore (cf. issue ADC-62)
├── results.py      enregistrements CSV/JSON + helpers 2π (paper_to_sim_time_window, gamma_to_paper_units)
├── check_model.py  oracle analytique comparé bit-à-bit au modèle compilé (validation, CI)
├── tests/          garde-fous sur le VRAI adc : assemblage polaire [CI], signes, flag géométrie, dump npz  → tests/README.md
├── diag/           diagnostics : Petri, audit normalisation, convergence, figures           → diag/README.md
├── docs/           notes d'audit : NORMALIZATION, T2 audit, RESULTS system-schur            → docs/README.md
├── slurm/          campagnes ROMEO : geometry, polar                                        → slurm/README.md
└── figures/        assets versionnés (snapshots, GIF, growth_rate, convergence, perf) + provenance.json
```

Each subfolder carries its own `README.md` listing its files. The core Python modules (`model`, `results`,
`run`, `run_polar`, `check_model`) stay at the root: they import each other same-dir (`from model import
...`), so they do not move into a subfolder.
