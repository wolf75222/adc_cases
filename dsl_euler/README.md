# dsl_euler: 2D Euler written as formulas (mini-DSL adc.dsl, numpy interpreter backend)

Declarative prototype: you declare the entire 2D compressible Euler system as symbolic expressions
(variables, primitives, fluxes, eigenvalues), `adc.dsl` interprets the tree in numpy, and it is
wired to the host backend `adc.PythonFlux`, which assembles $-\nabla\cdot F^*$ by finite volumes
(first-order Rusanov, periodic). No named brick, no compilation: this case is the only `*_dsl` in the
manifest that emits no C++ and does not compare against the native path. A central overpressure bubble
expands; the case checks for a finite, coherent state, not a target number.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `experimental` (`cases_manifest.toml`, `dsl_euler/run.py`, `ci = false`, `needs = []`). Interpreted CPU prototype, kept out of CI as a precaution. |
| Inputs | $64^2$ grid, $L=1$, periodic; IC gas at rest $\rho=1$, $v=0$, centered Gaussian bubble $p=1+0.4\,e^{-r^2/0.01}$, $E=p/(\gamma-1)$; $\gamma=1.4$; first-order Rusanov scheme + forward Euler, CFL $0.4$, 120 steps (dt re-evaluated each step) |
| Outputs | state $(4,n,n)=[\rho,\rho u,\rho v,E]$ in numpy memory; console diagnostics `drho_max`, `|v|_max`, `drel`, `max|dp|`; 2 figures in `figures/` + `figures/provenance.json` (written by `make_figures.py`, not by `run.py`) |
| Guaranteed invariants | the 4 `assert`s in `main` (in run.py): `assert_finite(U)`; `U[0].min()>0 and pressure(U).min()>0`; `drel < 1e-9`; `moved > 1e-3` |
| Proves | the symbolic tree declared as formulas produces a finite, positive ($\rho_{\min}=0.9222>0$, $p_{\min}=0.9752>0$) and non-trivial state (the bubble expands: `moved`$=0.3939 \gg 10^{-3}$); mass is conserved exactly (`drel`$=0.0$, bit-exact, conservative flux + periodic `np.roll`) |
| Does not prove | prototype, not production: interpreted numpy backend (`PythonFlux`), no compiled path, no GPU/MPI/AMR. No equality against the native path is checked here (`np.array_equal` absent): this is the only `*_dsl` that neither compiles nor compares (see [`diocotron_dsl`](../diocotron_dsl/)). First-order dissipative scheme: energy and momentum are not asserted, fronts are smeared. No published target, no tolerance on any physical value. Out of CI |
| Provenance | adc_cpp `01873299`, adc_cases `1affec1d`, interpreted numpy backend, $64^2$, ~0.2-0.4 s on 1 CPU core; `figures/provenance.json` |

By the end you will know: what "writing a model as formulas" means concretely (the 7 lines of
`make_euler`), how the tree is interpreted in numpy (not compiled), why mass is conserved bit-exact,
what the expanding bubble proves and does not prove, and exactly what is missing to promote this
prototype to the status of the other `*_dsl` cases.

---

## 1. What this case declares (justifies Proves: finite, coherent state)

No derivation of Euler-Poisson: this is pure compressible Euler (no source, no Poisson;
`set_source`/`set_elliptic_rhs` are never called, so `source_value` returns zeros,
in dsl.py). The content of the case is the declaration itself. `make_euler` (in run.py)
writes the entire system as symbolic expressions:

```python
e = dsl.HyperbolicModel("euler")
rho, rhou, rhov, E = e.conservative_vars("rho","rho_u","rho_v","E")     # 4 noeuds Var(cons)
u = e.primitive("u", rhou / rho)                                       # primitive = noeud Expr
v = e.primitive("v", rhov / rho)
p = e.primitive("p", (GAMMA-1.0)*(E - 0.5*rho*(u*u+v*v)))              # EOS gaz parfait
H = (E + p) / rho                                                      # enthalpie totale (Expr pur Python)
c = dsl.sqrt(GAMMA * p / rho)                                          # vitesse du son (noeud Sqrt)
e.set_flux(x=[rhou, rhou*u+p, rhou*v, rho*H*u],                        # F_x : 4 composantes
           y=[rhov, rhov*u, rhov*v+p, rho*H*v])                       # F_y : 4 composantes
e.set_eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])                    # vitesses caracteristiques
e.check()                                                             # toute var referencee declaree ?
```

- Each `/`, `*`, `-`, `+` builds a tree node (`Div`, `Mul`, `Sub`, `Add`; operator overloading on
  `Expr`). `u`, `v`, `p` are registered as primitives (`primitive` in dsl.py): at evaluation
  time they are derived from `U` in dependency order (`_env` in dsl.py). `H` and `c` are
  reused subtrees, not named primitives.
- `check()` (in dsl.py) verifies that every variable used in fluxes / eigenvalues /
  primitives is declared as cons/prim/aux; otherwise `ValueError`. This is the only static
  validation: there is no compiler behind it, the tree is the specification.

The corresponding equations (conservative form, $U=(\rho,\rho u,\rho v,E)$, $p=(\gamma-1)(E-\tfrac12\rho|v|^2)$,
$H=(E+p)/\rho$, $c=\sqrt{\gamma p/\rho}$):

$$F_x=(\rho u,\ \rho u^2+p,\ \rho u v,\ \rho H u),\quad F_y=(\rho v,\ \rho u v,\ \rho v^2+p,\ \rho H v),$$
$$\lambda_x=\{u-c,\ u,\ u+c\},\quad \lambda_y=\{v-c,\ v,\ v+c\}.$$

## 2. Who computes what: the middle layer is the tree, not a brick

For a DSL case, the central layer is not a named C++ brick: it is the expressions that `adc.dsl`
interprets. Here, the third layer = numpy host (not a device kernel).

| `run.py` symbol | Layer | What happens |
|---|---|---|
| `for _ in range(120): U = U + pf.cfl_dt(U,h,0.4)*pf.residual(U,h)` (`main` in run.py) | Python composes and integrates | choice of scheme (first-order Rusanov), integrator (forward Euler), step size (CFL 0.4 re-evaluated each step) |
| `e.set_flux(...)` / `e.set_eigenvalues(...)` -> `HyperbolicModel.flux` / `.max_wave_speed` (in dsl.py) | interpreted tree | `Expr.eval(env)` evaluates $F_x,F_y,\lambda$ in numpy over the whole array; no C++ |
| `adc.PythonFlux.residual` (in __init__.py) | numpy host kernel | assembles $-\nabla\cdot F^*$ (Rusanov, periodic `np.roll`); no device, no MPI |

Contrast with the middle layer of the other `*_dsl` cases: they call `emit_cpp_brick` /
`emit_cpp_source` -> `add_compiled_model` (in dsl.py), so the middle layer becomes a generated
C++ brick wired to the `assemble_rhs` device path. Here `to_python_flux` (in run.py and dsl.py)
short-circuits all of that: the tree feeds `PythonFlux` directly.

## 3. The scheme, line by line (justifies Proves: mass conserved bit-exact)

`PythonFlux.residual` (in __init__.py) assembles the Rusanov (local Lax-Friedrichs) flux:

```python
a = float(self.max_wave_speed(U))                    # une vitesse globale a = max_k max_cell |lambda_k|
for axis, h, d in ((2, dx, 0), (1, dy, 1)):          # x = axe 2, y = axe 1 du tableau numpy
    F  = self.flux(U, d)                             # F_x ou F_y via l'arbre interprete
    UR = np.roll(U, -1, axis=axis)                   # voisin +d (periodicite par decalage circulaire)
    face = 0.5*(F + np.roll(F,-1,axis=axis)) - 0.5*a*(UR - U)   # flux a la face +d
    res -= (face - np.roll(face,1,axis=axis)) / h    # -div : (F_{i+1/2} - F_{i-1/2}) / h
```

- `a` is a single global speed ($\max$ over both directions, `to_python_flux` in dsl.py),
  recomputed on each call: maximal diffusion, the simplest scheme. No MUSCL, no limiter, first order.
- Mass is conserved exactly. The first flux component is $\rho u$ / $\rho v$ (conservative form), and
  `np.roll` is a circular permutation: the telescoping sum of `face - roll(face)` over a periodic axis
  is identically zero line by line. The total mass $\sum\rho$ therefore moves only at floating-point
  roundoff; measured: `drel`$=0.0$ (the boundary fluxes cancel by construction, no residual roundoff
  error at $64^2$). This is the reason for `TOL_MASS`$=10^{-9}$ (the `drel < 1e-9` assert in `main`, run.py): an upper bound, a
  conservative scheme must not drift beyond machine noise; measured $0.0$, well under the tolerance.
- `assert moved > 1e-3` (in `main`, run.py; `moved`$=$`max|p - p_init|`): a lower bound at
  $10^{-3}$, three orders below the expected magnitude ($p$ varies by $\approx 0.4$); it rejects a
  frozen state (nothing moves) without rejecting the real dynamics. Measured: `moved`$=0.3939$.

## 4. Initial conditions (justifies: the expanding bubble)

The IC (in `main`, run.py): $64^2$ periodic grid, gas at rest, centered Gaussian overpressure.

```python
r2 = (gx - 0.5)**2 + (gy - 0.5)**2
p0 = 1.0 + 0.4*np.exp(-r2 / 0.01)                    # bulle +40%, ecart-type ~0.07
U[0] = 1.0;  U[3] = p0 / (GAMMA - 1.0)               # rho=1, v=0, E = p/(gamma-1) (repos)
```

Uniform density, zero velocities, peak pressure $1.4$ (central cell: $p_c(0)=1.395$). The expansion of
this bubble is what sets the system in motion and generates the radial acoustic wave.

## 5. Figures (generated by `make_figures.py`, in `figures/`)

Prototype diagnostic figures, not a versioned reproduction asset (category `experimental`): they show
that the state is finite/coherent and that the bubble expands, not a paper curve. Exact command in
section 7.

### `final_state.png`: final density and pressure maps

![Final density (rarefied core) and final pressure (radial ring), 64x64 periodic at t=0.589](figures/final_state.png)

- **Proves** (asserted in `main`, run.py): the final state is finite and positive across the whole
  domain: $\rho\in[0.9222,1.0452]$, $p\in[0.9752,1.0638]$, no NaN/Inf. The core has rarefied
  ($\rho\approx 0.94$ at the center: the bubble has emptied), surrounded by an outgoing radial ring.
- **Suggested** (not asserted): the acoustic signature (radial front moving away from the center) is
  visible but no assert measures it; the cross pattern is the interference of the wave with its
  periodic images (periodic domain + bubble aligned with the grid), not a bug artifact.
- **Not shown**: no comparison against a reference solution (Sedov, analytical blast wave); the
  first-order dissipative scheme smears the fronts, the map is not quantitatively calibrated.

### `bubble_decay.png`: decay of the perturbation

![Center pressure relaxing from 1.395 toward the mean 1.013, and amplitude max|p-p0| peaking at 0.457 at t=0.148](figures/bubble_decay.png)

- **Proves / measured**: the bubble expands: the peak pressure drops from $p_c(0)=1.395$, falls below
  the mean ($\bar p=1.013$: rarefaction rebound at $t\approx 0.15$), then rises toward $1.001$ at
  $t=0.589$. The amplitude $\max|p-p_0|$ grows, peaks at $0.457$ at $t=0.148$ (the front has formed),
  then decays toward the asserted value `moved`$=0.394$. This is the "decay of a perturbation": the
  localized peak diffuses into a spread-out wave.
- **Suggested**: the monotone relaxation toward $\bar p$ after the rebound suggests numerical damping
  (Rusanov diffusion), not quantified.
- **Not shown**: neither the exact acoustic period nor a physical damping rate; the first-order scheme
  is dissipative, the case does not separate physical from numerical damping.

## 6. What is missing to promote this prototype (limits)

- **No compiled path.** The other `*_dsl` cases ([`diocotron_dsl`](../diocotron_dsl/),
  [`two_species_dsl`](../two_species_dsl/), [`magnetic_isothermal_dsl`](../magnetic_isothermal_dsl/))
  call `emit_cpp_brick`/`add_compiled_model` and assert `np.array_equal` against the native path.
  This case stops at `to_python_flux`: it generates no C++ and compares to nothing. To promote it, you
  would emit the brick (`make_euler().emit_cpp_brick(...)`), compile it, and add the equality assert
  against the native path (`adc.CompressibleFlux`, available via [`models.euler`](../adc_cases/models.py),
  the `euler` function in models.py).
- **Host numpy backend, not device.** `PythonFlux` is documented as "off the GPU/MPI hot path: pure
  host path" (the `PythonFlux` docstring in __init__.py). No GPU, no MPI, no multi-box/AMR; a single $(4,64,64)$ array.
- **First-order dissipative scheme.** Rusanov + forward Euler: energy and momentum are not conserved
  and not asserted (only mass, positivity, finiteness, and dynamics are). Suited to a qualitative
  demo, not to a quantitative acoustic study.
- **No published reference, fixed geometry.** $64^2$ periodic hard-coded, no cli argument, no paper
  target (hence `experimental`, not `reproduction`). For source/Poisson coupling in the DSL, see the
  dedicated cases above.

## 7. Reproduce (justifies: exact command + measured cost)

```bash
cd /private/tmp/adc_cases-deeptut/dsl_euler
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : 4 asserts, ~0.2-0.4 s
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 figures + provenance.json
```

Prerequisites: `numpy` (`matplotlib` for the figures, outside the case's `needs`); the `adc` module
imported with the same interpreter that compiled it (ABI suffix `cpython-312`). The first `PYTHONPATH`
entry provides `adc` (including `dsl` and `PythonFlux`); the second makes `adc_cases` importable (the
case also has a `sys.path` fallback, in run.py). No C++ compiler required (`needs = []`), which
is the key difference from the other `*_dsl` cases (`needs = ["cxx"]`).

Expected output of `run.py` (captured, macOS arm64, identical across 3 runs):

```
modele declare en formules : 4 variables ['rho', 'rho_u', 'rho_v', 'E']
apres 120 pas : drho_max=0.123  |v|_max=0.027
masse : drel=0.00e+00   dynamique : max|dp|=0.394
OK dsl_euler
```

Cost: ~0.2-0.4 s wall time (dominated by importing the `adc` package / loading the `.so`; the pure
120-step compute at $64^2$ is negligible), ~44 MB peak memory, single-threaded numpy. Platform caveat:
the exactly-zero mass (`drel`$=0.0$), the `OK` verdict, the order of magnitude of `moved`$\approx 0.39$
and of $|v|_{\max}\approx 0.03$ are stable; the last digits may vary with the numpy version and the
summation order (cf. `figures/provenance.json`).

## File map

| File | Role |
|---|---|
| `run.py` | the case: declares Euler as formulas (`make_euler`), bubble IC, 120 steps, 4 asserts |
| `make_figures.py` | replays the physics with instrumentation; writes the 2 figures + `provenance.json` |
| `figures/final_state.png`, `figures/bubble_decay.png` | prototype diagnostics (final map, relaxation) |
| `figures/provenance.json` | adc_cpp/adc_cases SHA, backend, resolution, measured numbers |
| `<build>/python/adc/dsl.py` | `HyperbolicModel` (tree, numpy interpreter, `to_python_flux`), `sqrt`; provided by the adc_cpp build |
| `<build>/python/adc/__init__.py` | `adc` facade; `PythonFlux` (Rusanov + `np.roll` periodicity + `residual`/`cfl_dt`) |
| `../adc_cases/models.py` | `euler(gamma)` = native brick `adc.CompressibleFlux` (the compiled counterpart, `euler` in models.py) |
