# two_species_dsl: electrons + ions written as formulas, per-species equivalence with the native path

Two species (electrons in 4-variable compressible Euler, ions in 3-variable isothermal Euler),
each forced by an electrostatic field + a charge density, coupled by a single system Poisson,
written entirely as symbolic formulas (`adc.dsl.Model`) instead of named C++ bricks. This case
proves that each species' state reproduces the native composition to the expected precision: the
ions are bit-identical (`np.array_equal == True`), the electrons diverge by an epsilon
below machine tolerance ($4.93\times10^{-32} < 10^{-24}$), the sole signature of the
floating-point reassociation of the shared Poisson right-hand-side accumulation.

The physics of both species (continuity + momentum forced by $\mathbf{E}=-\nabla\phi$,
the $\gamma$ and $c_s^2$ closures, the coupled Poisson, per-species mass conservation) is derived in
the native parent case [`../multispecies/`](../multispecies/). This README does not re-derive it: it
focuses on (a) the table of core conventions reproduced as formulas, anchored to
`include/adc/physics/*.hpp`, and (b) how the bit-level equality is checked and what a divergence
would reveal.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `validation` (`ci = true`, `needs = ["cxx"]`, [`cases_manifest.toml:94-99`](../cases_manifest.toml)). Not a published reproduction: you verify a path equivalence, not a paper curve. |
| Inputs | grid $48^2$, $L=1$, periodic; electrons $n_e=1+0.02\cos(2\pi x)$, ions $n_i=1$ (charge separation, so a non-trivial Poisson); charges $q_e=-1$, $q_i=+1$; $\gamma_e=5/3$, $c_{s,i}^2=1$; 15 steps, CFL = 0.4 (`step_cfl(0.4)`), SSPRK2 + minmod + Rusanov for both blocks, Poisson `geometric_mg`, RHS `charge_density` |
| Outputs | states `get_state("electrons")` $(4,n,n)$ and `get_state("ions")` $(3,n,n)$ from both paths (native and DSL); `print` of the per-species $\max\lvert\text{DSL}-\text{native}\rvert$ + the `np.array_equal` verdict; 2 figures `figures/equivalence_{electrons,ions}.png` (3 panels per species: native $\rho$, DSL $\rho$, difference) + `figures/provenance.json` |
| Guaranteed invariants | the `assert`s in `run.py`: (1) electrons $\max\lvert\text{DSL}-\text{native}\rvert<10^{-24}$ (`run.py:227`); (2) ions `np.array_equal` or $<10^{-24}$ (`run.py:229`); (3) per-species mass `relative_drift < 1e-9` (`run.py:242-243`); (4) finite states and densities $>0$ (`run.py:238-241`) |
| Proves | ions bit-identical: $\max\lvert\text{DSL}-\text{native}\rvert=0.000\times10^{0}$ exactly, across all 3 components (`np.array_equal == True`); electrons below machine tolerance: $\max\lvert\text{DSL}-\text{native}\rvert=4.930\times10^{-32}$, confined to the single $\rho v$ component ($\rho$, $\rho u$, $E$ at $0.0$ exactly); mass conserved per species (relative drift $1.20\times10^{-14}$ electrons, $1.16\times10^{-14}$ ions) |
| Does not prove | no physical result: the same toy cosine IC as `multispecies`, 15 steps, no Debye length nor plasma frequency, no rate. The electron equality is not bit-exact ($4.93\times10^{-32}\ne0$): it is an FP reassociation of the shared Poisson RHS, not a strict equality; you assert it below $10^{-24}$, not at `array_equal`. The $4.93\times10^{-32}$ is platform-dependent (BLAS, MG reduction order); only the confinement to $\rho v$ and the $\ll10^{-24}$ order of magnitude are stable. Actual backend = `aot` (the ABI guard rejects `production` on a pre-built module) |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, DSL backend `aot` (fallback; `production` rejected by the ABI), native backend serial, $48^2$, Apple clang 21.0.0, Python 3.12.2, macOS arm64; numbers in `figures/provenance.json` |

By the end you will know: which exact core conventions the DSL formulas reproduce (header-anchored
table), why the ions come out bit-identical while the electrons do not, and what a non-black
heatmap (or an electron equality beyond $10^{-24}$) would reveal.

---

## 1. The physics: see the native parent case

Both species, their equations (continuity, momentum forced by
$\mathbf{E}=-\nabla\phi$, the $\gamma=5/3$ and $c_s^2=1$ closures), the coupled system Poisson
$\nabla^2\phi=q_e n_e+q_i n_i$, the initial charge separation, and per-species mass conservation
(flux telescoping on the torus) are derived in [`../multispecies/README.md`](../multispecies/)
(sections 1, 2, 4). `two_species_dsl` solves the same physics with the same parameters; the
only difference is the model-construction path: DSL formulas instead of named native bricks.
So nothing is re-derived here.

One calibration difference from `multispecies`: here you advance with `step_cfl(0.4)` (CFL-adaptive
step) for 15 steps, where `multispecies` does `advance(dt=0.001, nsteps=20)` (fixed step). This has no
bearing on the equivalence (native and DSL take exactly the same time path, hence the same
$dt$ at each step); it only changes the absolute values of the final state, not the comparison.

---

## 2. The core conventions, reproduced as formulas

The heart of the DSL cases: each symbolic formula must reproduce identically the convention of
the corresponding native brick, otherwise the equality breaks. Table of reproduced conventions, anchored
to the `include/adc/physics/*.hpp` headers (left = native brick, right = DSL formula `run.py`):

### Electrons (`electron_dsl_model`, `run.py:76-108`) reproduce `models.electron_euler`

| Quantity | Native brick (header) | DSL formula (`run.py`) |
|---|---|---|
| Pressure / EOS | `Euler::pressure` $p=(\gamma-1)(E-\tfrac12\rho\lvert v\rvert^2)$ ([`euler.hpp:42-47`](../../adc_cpp/include/adc/physics/euler.hpp)) | `p = (g-1)*(E - 0.5*rho*(u*u+v*v))` (`run.py:87`) |
| Convective flux $x$ | `Euler::flux` $(\rho u,\ \rho u^2+p,\ \rho u v,\ (E+p)u)$ ([`euler.hpp:94-104`](../../adc_cpp/include/adc/physics/euler.hpp)) | `x=[rhou, rhou*u+p, rhou*v, (E+p)*u]` (`run.py:91`) |
| Spectrum $x$ | `Euler::eigenvalues` $(u-c,\ u,\ u,\ u+c)$, $c=\sqrt{\gamma p/\rho}$ ([`euler.hpp:108-118`](../../adc_cpp/include/adc/physics/euler.hpp)) | `x=[u-c, u, u, u+c]`, `c=sqrt(g*p/rho)` (`run.py:88,93`) |
| Electrostatic force | `PotentialForce::apply` $s[1{:}3]=q\rho\mathbf{E}$, $s[3]=q(\rho_u E_x+\rho_v E_y)$, $\mathbf{E}=-(\text{grad\_x},\text{grad\_y})$ ([`source.hpp:33-44`](../../adc_cpp/include/adc/physics/source.hpp)) | `source([0, Q_E*rho*e_x, Q_E*rho*e_y, Q_E*(rhou*e_x+rhov*e_y)])`, `e_x=-grad_x`, `e_y=-grad_y` (`run.py:101-103`) |
| Charge density | `ChargeDensity::rhs` $f=q\,u[0]=q n$ ([`elliptic.hpp:19-25`](../../adc_cpp/include/adc/physics/elliptic.hpp)) | `elliptic_rhs(Q_E * rho)` (`run.py:105`) |

### Ions (`ion_dsl_model`, `run.py:111-137`) reproduce `models.ion_isothermal`

| Quantity | Native brick (header) | DSL formula (`run.py`) |
|---|---|---|
| Pressure / closure | `IsothermalFlux` $p=c_s^2\rho$ ([`hyperbolic.hpp:132-140`](../../adc_cpp/include/adc/physics/hyperbolic.hpp)) | `p = cs2 * rho` (`run.py:122`) |
| Convective flux $x$ | `IsothermalFlux::flux` $(\rho u,\ \rho u^2+p,\ \rho u v)$ ([`hyperbolic.hpp:132-141`](../../adc_cpp/include/adc/physics/hyperbolic.hpp)) | `x=[rhou, rhou*u+p, rhou*v]` (`run.py:125`) |
| Spectrum $x$ | `IsothermalFlux::eigenvalues` $(u-c,\ u,\ u+c)$, $c=\sqrt{c_s^2}$ ([`hyperbolic.hpp:165-174`](../../adc_cpp/include/adc/physics/hyperbolic.hpp)) | `x=[u-c, u, u+c]`, `c=sqrt(cs2)` (`run.py:123,126`) |
| Electrostatic force | `PotentialForce::apply` (3 var: no energy term, the `if constexpr (size()==4)` of [`source.hpp:41`](../../adc_cpp/include/adc/physics/source.hpp) is false) | `source([0, Q_I*rho*e_x, Q_I*rho*e_y])` (3 components, `run.py:133`) |
| Charge density | `ChargeDensity::rhs` $f=q n$ ([`elliptic.hpp:19-25`](../../adc_cpp/include/adc/physics/elliptic.hpp)) | `elliptic_rhs(Q_I * rho)` (`run.py:134`) |

Two convention subtleties the formulas must honor for the equality to hold:

- **The sign is carried by the charge, not by the elliptic operator.** On the core side, `PotentialForce.qom=q` and
  `ChargeDensity.q=q` carry the sign ($q_e=-1$, $q_i=+1$); the Poisson operator solves
  $\varepsilon\nabla^2\phi=f$ with $\varepsilon=1$ (see `../multispecies/` sec. 4.3). The DSL copies
  this choice: `Q_E*rho*e_x` (force) and `Q_E*rho` (RHS), never an extra sign on $\nabla^2$.
  This is the `PotentialForce`+`ChargeDensity` family, distinct from `GravityForce`+`GravityCoupling`
  (sign carried by the elliptic operator) used by [`../euler_poisson/`](../euler_poisson/) (sec. 2 of that case).
- **The energy component exists only for the electrons.** The work term $q(\rho_u E_x+\rho_v E_y)$
  is the 4th component of the electron source; the ions, with 3 variables, have no energy
  equation (isothermal closure), hence no work term. The core's `if constexpr (State::size()==4)`
  ([`source.hpp:41`](../../adc_cpp/include/adc/physics/source.hpp)) is reproduced on the DSL side
  by the length of the list passed to `m.source(...)`: 4 terms for electrons, 3 terms for ions.

3-layer "who computes what" table (the middle layer is no longer a named brick but the
expressions that `adc.dsl` compiles):

| `run.py` line | Layer | What happens |
|---|---|---|
| `add_equation("electrons", model=ce, spatial=FiniteVolume(limiter="minmod", riemann="rusanov"), time=Explicit())` (`run.py:187-189`); same for ions (`run.py:190-192`) | Python composes and diagnoses | choice of scheme (MUSCL minmod + Rusanov, SSPRK2); reading the states to compare against native |
| `m.flux(...)`, `m.eigenvalues(...)`, `m.source(...)`, `m.elliptic_rhs(...)` (`run.py:91-105`, `125-134`) that `m.compile(..., backend)` translates into C++ | frozen DSL expressions | the exact convention (flux, spectrum, force $q\rho\mathbf{E}$, RHS $q n$) compiled into a `.so`, cse'd at codegen |
| `assemble_rhs<minmod, rusanov>` + system Poisson `geometric_mg` (RHS $\sum_b q_b n_b$), inlined by the `aot`/`production` backend | per-cell kernel (device) | the actual computation, with no Python callback in the hot path: the same path as native, which makes bit-level equality possible |

This justifies the Proves clause: it is because these expressions reproduce the bricks exactly
and the backend inlines the same numerical path that the equality is expected, not approximate.

---

## 3. How the bit-level equality is checked (`main`, `run.py:207-245`)

The case plays two runs on the same grid, same IC, same Poisson, same scheme, same number of steps:
`run_native` (native composition `models.electron_euler`/`ion_isothermal`, `run.py:150-163`) then
`run_dsl` (compiled DSL models, `run.py:175-204`). The comparison:

```python
de = float(np.max(np.abs(ed - en)))                       # electrons : max|DSL - natif| (run.py:219)
di = float(np.max(np.abs(idd - inn)))                     # ions      : idem (run.py:220)
assert de < 1e-24, "..."                                  # electrons sous tolerance machine (run.py:227)
assert np.array_equal(idd, inn) or di < 1e-24, "..."      # ions bit-identiques (run.py:229)
```

- `np.array_equal(a, b)` is `True` only if every bit matches: no tolerance.
  It is the hardest observable for the ions.
- The electron tolerance $10^{-24}$ is a clause justified by an order of magnitude, not an
  arbitrary constant. Lower bound: the measured divergence is $4.93\times10^{-32}$, about $\sim10^{8}$
  times below the tolerance. Upper bound: the state magnitude is $O(1)$ (densities $\approx1$,
  energy $O(1)$); a real physics divergence (wrong sign convention, missing
  term) would be $O(10^{-2})$ or larger, like the dynamics. $10^{-24}$ sits between the FP
  reassociation noise ($\sim10^{-32}$, $\sim10^{8}\,\varepsilon_{\text{mach}}$ relative to $O(1)$) and any
  physics divergence: it accepts reassociation, rejects a model discrepancy.

**Why the electrons diverge and the ions do not** (the Does not prove clause). At 1 step, the residual and
the flux of each species are bit-identical to native (the formulas reproduce the bricks
identically). Over several coupled steps, the only difference comes from accumulating the
right-hand side of the shared Poisson $f=q_e n_e+q_i n_i$: two blocks contribute to it, and the order in which
the code sums the DSL vs native contributions is not guaranteed identical. Since floating-point addition
is not associative, this change of order produces a $\phi$ that differs in the last bit, hence an
$\mathbf{E}=-\nabla\phi$ that differs in the last bit, hence an electron force that differs, hence an
electron state at $\sim10^{-32}$ from native. The ions, in contrast, come out bit-identical on this
trajectory: their density stays nearly uniform ($n_i=1$ everywhere initially, induced modulation
$\sim4\times10^{-6}$ as in `multispecies`), so the reassociation does not bite at the
measurable precision of their state. This is not a physics discrepancy; it is rounding noise
from the shared accumulation, and it is exactly what the tight tolerance distinguishes from a bug.

What a divergence beyond $10^{-24}$ would reveal: a DSL formula that no longer reproduces a
core brick, a wrong sign in `source` (repulsive force instead of attractive), an energy term
forgotten or wrongly added, the spectrum $c=\sqrt{\gamma p/\rho}$ replaced by $\sqrt{c_s^2}$, or an RHS
$q n$ with the wrong $q$. The assert would then be $O(10^{-2})$, not $O(10^{-32})$.

---

## 4. Figures (regenerated by `make_figures.py`, in `figures/`)

`make_figures.py` re-plays `run.py` exactly (imports its `run_native`/`run_dsl`/`initial_conditions`
functions, same parameters, same backend) and plots, per species, a 3-panel figure on the density
component $\rho$ (index 0 of the conservative state of both models):

$$[\,\rho\ \text{natif}\,]\ \mid\ [\,\rho\ \text{DSL}\,]\ \mid\ [\,\lvert\rho_{\text{DSL}}-\rho_{\text{natif}}\rvert\,]$$

The first two panels (viridis, same color scale) show the structured density field
produced by each path: you see the modulated cosine advected, and you can tell by eye
that native and DSL are the same field (not an empty black square). The third panel (inferno,
fixed scale) shows the difference $\lvert\rho_{\text{DSL}}-\rho_{\text{natif}}\rvert$: black = match. You
prove the equality by first showing two identical fields, then the black difference map (annotated max),
instead of a single black diff that would look broken. Cited numbers: `figures/provenance.json`.
Exact command in section 5.

> **Note on the observable.** The figure focuses on $\rho$, where both species are
> bit-identical ($\max\lvert d\rvert_\rho = 0$ exactly). The electron state's machine epsilon
> ($4.93\times10^{-32}$) does not live in $\rho$ but in the $\rho v$ component (see sec. 3 and
> `provenance.json`, `max_abs_diff_per_var_electrons`). The electron difference panel therefore annotates two
> numbers: `max|d|` on $\rho$ (= 0) and `full state` (= $4.93\times10^{-32}$, max over the 4
> components), so that the FP signature stays documented even though the density itself is exact.

### `equivalence_ions.png`: visible density, black difference (bit-identical)

![Three ion panels: native rho and DSL rho (structured field, ~1e-4 modulation visible in viridis), then |rho_DSL - rho_natif| black, max=0](figures/equivalence_ions.png)

- **Proves** (asserted `run.py:229`): panels 1 and 2 = the ion density (induced modulation
  $\sim10^{-4}$ around $n_i=1$, advected), identical by eye between native and DSL; panel 3 =
  uniformly black difference, $\max\lvert\rho_{\text{DSL}}-\rho_{\text{natif}}\rvert=0.000\times10^{0}$,
  and the whole ion state is `np.array_equal == True`. A single non-black pixel in the difference panel
  would signal that the DSL isothermal formula diverges from `IsothermalFlux` or that
  `PotentialForce`/`ChargeDensity` is wrongly reproduced.
- **Not shown**: the figure shows the density and its difference, not the momentum
  components ($\rho u$, $\rho v$) nor the fine dynamics (modulation $\sim4\times10^{-6}$ of the coupling,
  see `../multispecies/` sec. 6). The per-component differences stay in `provenance.json`
  (`max_abs_diff_per_var_ions`), all at $0$.

### `equivalence_electrons.png`: visible density, black $\rho$ difference, FP epsilon in $\rho v$

![Three electron panels: native rho and DSL rho (cosine +/-1.6 % structure in viridis), then |rho_DSL - rho_natif| black, max rho=0, full state 4.93e-32 annotated](figures/equivalence_electrons.png)

- **Proves** (asserted `run.py:227`): panels 1 and 2 = the electron density (cosine
  $1\pm1.6\,\%$ in $x$, $\approx0.984$ to $1.016$, advected), identical by eye between native and DSL;
  panel 3 = difference on $\rho$ uniformly black, $\max\lvert\rho_{\text{DSL}}-\rho_{\text{natif}}\rvert=0.000\times10^{0}$
  exactly. The electron density is therefore bit-identical between the two paths.
- **Suggested (not asserted)**: the electron state's machine epsilon ($4.930\times10^{-32}$, annotated
  `full state` on the difference panel) is confined to the $\rho v$ component (y-momentum); $\rho$, $\rho u$ and $E$ are at $0$ exactly. This confinement is consistent with the coupling
  mechanism (FP reassociation of the shared Poisson RHS, see sec. 3): the electron IC is modulated in
  $x$, the field $\mathbf{E}=-\nabla\phi$ responds, and it is on $\rho v$ that the reassociation
  shows up first at measurable precision. No `assert` tests this confinement, it is read
  from `provenance.json` (`max_abs_diff_per_var_electrons`), not verified.
- **Not shown**: the figure does not plot $\rho v$ (where the epsilon lives); to see it, read
  `provenance.json`. The exact value $4.93\times10^{-32}$ and the pattern of affected cells are
  platform-dependent (multigrid reduction order, BLAS); only the $\ll10^{-24}$ order, the
  bit-exact density, and the fact that $\rho$/$\rho u$/$E$ stay at $0$ exactly are stable. The map does not
  cover stability over $>15$ steps (the rounding accumulation grows slowly).

---

## 5. Reproduce

```bash
cd /private/tmp/adc_cases-deeptut/two_species_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts d'equivalence + invariants
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 heatmaps + provenance.json
```

Prerequisites: Python 3.12 + numpy (matplotlib only for `make_figures.py`), the `adc` module
compiled and imported with the same interpreter (ABI suffix `cpython-312`), and a C++20 compiler
(`needs = ["cxx"]`): the DSL compiles a `.so` on the fly under `out/two_species_dsl/`. The first
entry of the `PYTHONPATH` provides the C++ module; the second makes `adc_cases` importable (the case has a
`sys.path` fallback, `run.py:59-64`).

Output of `run.py` (macOS arm64, Apple clang 21.0.0):

```
=== two_species_dsl : electrons + ions ecrits en formules vs briques natives ===
grille 48 x 48, 15 pas, CFL = 0.4 ; q_e = -1, q_i = 1
backend 'production' indisponible (RuntimeError: add_native_block : ABI incompatible ...), essai suivant
backend DSL retenu : 'aot'
electrons : max|DSL - natif| = 4.930e-32 (bit-identique = False)
ions      : max|DSL - natif| = 0.000e+00 (bit-identique = True)
masse electrons : derive relative 1.204e-14 ; ions : 1.165e-14
OK two_species_dsl (equivalence DSL <-> natif par espece, backend 'aot')
```

**Actual backend = `aot`.** The case prefers `production` (zero-copy native loader,
`add_native_block`, strict parity with `add_block`), but the ABI guard rejects it here: the pre-built
`build-master` module has a header/compiler signature that differs from the one expected at
binding time, so `run_dsl` falls back to `aot` (host-marshaled production path, numerically
identical). With an `_adc` module rebuilt against the same headers, `production` would be selected;
in both cases the per-species equality holds (same inlined numerical path). **Platform caveat**:
the bit-identical ions verdict, the confinement of the electron difference to $\rho v$ and its
$\ll10^{-24}$ order are stable; the exact value $4.930\times10^{-32}$ varies with the BLAS and the
multigrid reduction order (see `figures/provenance.json`).

## File map

| File | Role |
|---|---|
| [`run.py`](run.py) | the case (CI): 2 DSL models (electrons 4 var, ions 3 var), compiles, binds, compares against native per species, equivalence `assert`s + invariants |
| [`make_figures.py`](make_figures.py) | re-plays `run.py`, plots `equivalence_{electrons,ions}.png` (3 panels per species: native $\rho$, DSL $\rho$, difference) + `provenance.json` (outside CI) |
| `figures/*.png`, `figures/provenance.json` | versioned diagnostics: adc_cpp/adc_cases SHA, backend, per-component $\max\lvert d\rvert$, mass drifts |
| [`../multispecies/`](../multispecies/) | the same physics in native bricks: derivation of the equations, the Poisson coupling and mass conservation |
| [`adc_cases/models.py`](../adc_cases/models.py) | `electron_euler()`, `ion_isothermal()` = native reference oracle (`l.28-45`) |
| [`include/adc/physics/`](../../adc_cpp/include/adc/physics/) | `euler.hpp`, `hyperbolic.hpp`, `source.hpp`, `elliptic.hpp`: the bricks whose conventions the DSL formulas reproduce (sec. 2) |
| [`cases_manifest.toml`](../cases_manifest.toml) | declares the case: `validation`, `ci = true`, `needs = ["cxx"]` |
