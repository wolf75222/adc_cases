# two_euler: two independent Euler gases, same code

Two compressible Euler blocks (`electrons`, `ions`) carried by the same native model and the same
scheme, with no coupling between them. Only the initial conditions differ: the electrons are 100x
less dense, so their sound speed is ~10x larger. You check by `assert` that each block conserves its
mass, stays positive (`rho > 0`, `p > 0`), and that the fast block spreads faster; you show by figure
that the multirate integrator `step_adaptive` automatically sub-cycles the fast block 10x per macro
step. This case reproduces no published result: it checks structural invariants and exercises the
multi-block API.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `validation` (`cases_manifest.toml`: `two_euler/run.py`, `ci = true`, `needs = []`) |
| Inputs | grid $64^2$, $L=1$, periodic; 2 Euler blocks `gamma=1.4`; central Gaussian overpressure bubble $p=p_0+dp\,e^{-r^2/(\sigma^2 L^2)}$ ($p_0=1$, $dp=0.5$, $\sigma^2=0.02$), gas at rest $u=v=0$; electrons $\rho_0=0.01$, ions $\rho_0=1.0$; system Poisson with $f=0$ (zero charge); 20 macro steps `step_adaptive(0.4)` |
| Outputs | final states $(4,64,64)$ of the 2 blocks; 4 printed diagnostics (mass, positivity, front); figures in `figures/` + `figures/provenance.json` (this tutorial, outside `run.py`) |
| Guaranteed invariants | relative mass conserved `< 1e-9` per block; `rho > 0` and `p > 0` (positivity); `fe > fi` (electron front more extended); finite states (no NaN, no Inf) |
| Proves | (1) each block conserves its mass at a measured relative drift $\le 1.6\times10^{-14}$ (electrons) and $\le 5.1\times10^{-15}$ (ions), machine noise; (2) positivity holds: `rho_min` e=$7.71\times10^{-3}$, i=$0.803$; `p_min` e=$0.932$, i=$1.000$; (3) the electron front covers $86\%$ of the cells against $29\%$ for the ions; (4) the 2 blocks are strictly independent (no source, $f=0$) |
| Does not prove | this is not a published reproduction nor plasma physics: the names "electrons"/"ions" are an analogy, the 2 gases are not coupled (Poisson coupling is [`multispecies`](../multispecies/) / [`plasma`](../plasma/)). No `assert` tests the multirate sub-cycling nor the factor 10 on the sound speed: they are measured and plotted, not asserted. No device / MPI / AMR validation (CPU single-rank). |
| Provenance | adc_cpp `01873299`, adc_cases `7c7a3403`, native serial backend, $64^2$, ~0.45 s on 1 CPU core; `figures/provenance.json` |

By the end you will know: why two Euler blocks run under the same code (brick composition); why each
mass is conserved to machine precision (periodic conservation law); why the electrons are 10x faster
(sound speed); and how the multirate step sub-cycles the fast block with no explicit configuration.

---

## 1. The mechanism (lightweight: no coupled physics)

Each block is a bubble of gas at rest with a Gaussian overpressure at the center. The pressure
gradient pushes the gas outward: a radial expansion leaves the center at the local sound speed.
Nothing else acts (no force, no field). The only physical difference between the two blocks is the
background density: at equal pressure, the sound speed $c=\sqrt{\gamma\,p/\rho}$ scales as
$1/\sqrt{\rho}$, so the electrons ($\rho_0=0.01$) have
$c_e/c_i=\sqrt{\rho_i/\rho_e}=\sqrt{1/0.01}=10$: their expansion is 10x faster and covers more ground
in the same time. This is the only asymmetry; everything else (model, scheme, integrator) is
identical.

The two blocks communicate through no channel: the source is `NoSource`, and the right-hand side of
the system Poisson is $f=sum_b q_b n_b=0$ because both charges are zero (section 3). The solved
potential stays zero and feeds back on no one. So "two_euler" is two independent hyperbolic
conservation laws that share the same code.

This justifies contract clause Proves (4) (strictly independent blocks) and clause Does not prove (no
plasma physics: this case couples nothing).

---

## 2. The equations and who computes them

Each block solves compressible 2D Euler in conservative form:

$$\partial_t U+\partial_x F(U)+\partial_y G(U)=0,\qquad U=(\rho,\ \rho u,\ \rho v,\ E),\qquad p=(\gamma-1)\Big(E-\tfrac12\rho|u|^2\Big).$$

The flux $F$ transports $(\rho u,\ \rho u^2+p,\ \rho uv,\ (E+p)u)$ (and $G$ by symmetry $x\leftrightarrow y$).
With $\gamma=1.4$ (`GAMMA` in run.py). The source term is identically zero; the system Poisson is
present but with a zero right-hand side. The two blocks share these equations identically.

The species model is `adc_cases.models.euler(GAMMA)` (`euler` in models.py), the composition
of native bricks:

| Model block | Equation | Brick (`models.euler`) |
|---|---|---|
| State | $(\rho,\rho u,\rho v,E)$ | `adc.FluidState(kind="compressible", gamma=gamma)` |
| Transport | Euler flux $F,G$ | `adc.CompressibleFlux()` |
| Source | none | `adc.NoSource()` |
| Elliptic | $f=q\,n$ with $q=0$ | `adc.ChargeDensity(charge=0.0)` |

`adc.ChargeDensity(charge=0.0)` declares a zero charge density: the block's contribution to the
system Poisson ($f=sum_b q_b n_b$) is zero. `set_poisson()` is called all the same because
`step_adaptive` opens each macro step with `solve_fields()` (section 4); with $f=0$ the potential
stays zero. This is the exact justification of the comment in `main`: "Poisson f=0 (zero
charge): independent blocks, just for solve_fields".

### 3-layer table: who computes what

| `run.py` symbol | Layer | What happens |
|---|---|---|
| `add_block("electrons", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())` (`main` in run.py) | Python composes | choice of the Euler model, of the scheme (van Leer + HLLC + primitive recon), of the integrator (SSPRK2); two distinct calls, one per block |
| `models.euler(GAMMA)` -> `CompressibleFlux` / `NoSource` / `ChargeDensity(0)` (`include/adc/physics/euler.hpp`, `.../source.hpp`) | the C++ brick fixes the physics | the exact convention of the flux $F(U)$, of the wave speed $|v_n|+c$, of the zero right-hand side |
| `assemble_rhs<Limiter,Flux>` (HLLC) + system Poisson (`solve_fields`, $f=0$) | per-cell kernel | the per-cell computation, with no Python callback in the hot path |

`models.euler` names no species on the core side: the word "electrons"/"ions" lives in `adc_cases`,
the physics is a composition of generic bricks. This is what makes "2 Euler, same code" literal: the
two blocks are the same composed model object, instantiated twice, distinguished only by `set_state`
(section 5).

---

## 3. Why the invariants hold (the derivation)

This is an invariant validation case: the prediction is not a physical number from a paper, it is a
structural property the scheme must respect. Three invariants, three reasons.

### 3.1 Mass conservation (justifies Proves 1)

The first Euler component is $\partial_t\rho+\nabla\cdot(\rho\mathbf{u})=0$. A finite-volume scheme
updates each cell by the flux balance at its faces:
$\rho_{ij}^{n+1}=\rho_{ij}^n-\frac{\Delta t}{h}\big(F^x_{i+1/2,j}-F^x_{i-1/2,j}+F^y_{i,j+1/2}-F^y_{i,j-1/2}\big)$.
The total mass $M=\sum_{ij}\rho_{ij}\,h^2$ sums over all cells: each interior flux appears twice with
opposite signs (face shared between neighboring cells) and cancels by telescoping. On a periodic
domain (`periodic=True` in run.py), the boundary fluxes also close up (the right face of the
domain = the left face). Nothing remains: $M^{n+1}=M^n$ exactly in real arithmetic. The only drift
observed comes from floating-point arithmetic (summation order, rounding), so at machine-noise level
$\sim 10^{-16}\,M$. This is why the tolerance `tol=1e-9` (in run.py) is honest: it is set 7
orders of magnitude above the expected machine noise ($\sim 10^{-16}$) and well below any physical
violation ($M$ changes by $O(1)$ relative if a flux leaks), so it cleanly separates "conserved to the
bit" from "conservation bug". The measured drift ($\le 1.6\times10^{-14}$, section 6) confirms it:
you are at machine noise, $10^5$ times below the tolerance.

### 3.2 Positivity (justifies Proves 2)

Euler only makes sense for $\rho>0$ and $p>0$ ($c=\sqrt{\gamma p/\rho}$ becomes imaginary otherwise,
the scheme blows up). Nothing mathematically guarantees the positivity of a generic MUSCL+HLLC
through a strong expansion; two choices favor it here, both in `main`:

- Primitive reconstruction (`recon="primitive"`): you reconstruct $(\rho,u,v,p)$ at the faces rather
  than the conservative variables $(\rho,\rho u,\rho v,E)$. Reconstructing $\rho$ and $p$ directly
  avoids subtracting the kinetic energy $\frac12\rho|u|^2$ from a reconstructed total energy (a
  classic source of negative pressure in expansions).
- HLLC flux: restores the contact wave, less dissipative than Rusanov but robust for Euler.

So positivity is not a theorem but an empirical invariant that the case checks by `assert`
(`main` in run.py). The electron expansion is the most severe (sound speed 10x, hence the strongest
expansion), and it is the one that goes lowest: `rho_min` electrons reaches $6.99\times10^{-3}$ at
the trough (section 6), still $>0$.

### 3.3 Front ordering (justifies Proves 3)

At equal pressure $p_0=1$, the Euler sound speed is $c=\sqrt{\gamma p_0/\rho_0}$. The ratio of the
two blocks is $c_e/c_i=\sqrt{\rho_{0,i}/\rho_{0,e}}=\sqrt{1.0/0.01}=10$, the number printed in
`main` (`np.sqrt(1.0/0.01)`). The expansion front advances at the sound speed: over the fixed
duration ($t_{\text{final}}=0.085$, 20 macro steps), the electron front travels 10x more distance, so
it covers a larger fraction of cells. The `assert fe > fi` (in run.py) tests exactly this ordering,
where `fe`, `fi` are the fractions of cells where $|p-p_0|>0.02$ (function `disturbed` in
run.py). Measured: `fe = 0.861`, `fi = 0.287`.

---

## 4. The code, anchored

Reading of `main` (in run.py), the load-bearing lines only.

### 4.1 Composition (in run.py)

```python
sim = adc.System(n=n, L=L, periodic=True)
spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")
sim.add_block("electrons", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
sim.add_block("ions",      model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
sim.set_poisson()  # f = 0 (charge nulle) : blocs independants, juste pour solve_fields
```

- `adc.System(n=64, L=1, periodic=True)`: Cartesian grid $64\times64$, periodic domain $[0,1]^2$
  (periodicity closes the mass balance, section 3.1).
- `adc.Spatial(vanleer=True, flux="hllc", recon="primitive")`: a single scheme object, shared by the
  two blocks. `vanleer=True` -> van Leer limiter (MUSCL order 2, 2 ghost cells,
  `Spatial` in __init__.py); `flux="hllc"` -> HLLC Riemann solver (requires compressible transport,
  checked on the facade side); `recon="primitive"` -> reconstruction of the primitive variables
  (section 3.2).
- both `add_block` calls: same `model=models.euler(GAMMA)` (two instances), same `spatial`, same
  `time=adc.Explicit()` (SSPRK2, `substeps=1`, `stride=1` by default, `Explicit` in __init__.py).
  This is the heart of the "2 Euler, same code" message.
- `set_poisson()` configures the system Poisson (default `rhs="charge_density"`). With $q=0$
  everywhere the right-hand side is zero; the call only serves so that
  `step_adaptive -> solve_fields()` has a valid solver.

### 4.2 Initial conditions (in run.py)

```python
Ue0 = blob(n, L, rho0=0.01, p0=1.0, dp=0.5)  # electrons : legers -> c ~ 10x, rapides
Ui0 = blob(n, L, rho0=1.0,  p0=1.0, dp=0.5)  # ions : lourds, lents
sim.set_state("electrons", Ue0.reshape(-1).tolist())
sim.set_state("ions",      Ui0.reshape(-1).tolist())
```

`blob` (in run.py) delegates to `euler_pressure_blob` (in initial_conditions.py):
gas at rest $u=v=0$, uniform density $\rho_0$, Gaussian overpressure
$p(x,y)=p_0+dp\,e^{-r^2/(\sigma^2 L^2)}$ with $\sigma^2=0.02$ (default, not passed by `run.py`),
$r^2=(x-L/2)^2+(y-L/2)^2$, and $E=p/(\gamma-1)$ (pure internal energy, no kinetic). The two initial
conditions are identical in shape: only $\rho_0$ changes ($0.01$ vs $1.0$), the source of all the
speed contrast. `set_state` flattens the state $(4,64,64)$ into a flat list.

### 4.3 Multirate loop (in run.py)

```python
for _ in range(20):
    sim.step_adaptive(0.4)
```

`step_adaptive(cfl)` is the multirate integrator. Its exact semantics, read in the method
`step_adaptive` (in system_stepper.hpp):

1. `solve_fields()` solves the system Poisson (here $f=0$, zero potential).
2. For each evolving block, the max wave speed $w_b=\max_{\text{grid}}(|v_n|+c)$ is computed
   (`s.max_speed(s.U)` in system_stepper.hpp); `wmin` = the smallest (slowest block).
3. The macro step is $\text{macro\_dt}=\text{cfl}\cdot h/w_{\min}$ (`step_adaptive` in system_stepper.hpp), pinned
   to the slowest block (the ions).
4. Each block is sub-cycled $n_b=\lceil\text{stride}_b\cdot w_b/w_{\min}\rceil$ times over the
   effective step (`advance_transport_n(s, eff_dt, n)` in system_stepper.hpp). With
   `stride=1` and $w_e/w_i=10$, the `electrons` block does $n_e=\lceil10\rceil=10$ sub-steps, the
   `ions` $n_i=1$.
5. `apply_couplings(macro_dt)` (no coupling here), then `t += macro_dt`.

This is what "the multirate automatically sub-cycles the electrons" means: the factor 10 follows from
the ratio of wave speeds, with no explicit configuration. The figure `multirate.png` (section 6)
plots it directly.

### 4.4 Diagnostics and asserts (in run.py)

```python
Ue = np.array(sim.get_state("electrons")).reshape(4, n, n)
dme = assert_mass_conserved(sim.mass("electrons"), me0, tol=1e-9, label="electrons")
pe, pi = pressure(Ue), pressure(Ui)
fe, fi = disturbed(Ue, Ue0, 0.02), disturbed(Ui, Ui0, 0.02)
assert Ue[0].min() > 0 and Ui[0].min() > 0, "densite negative"
assert pe.min() > 0 and pi.min() > 0, "pression negative"
assert fe > fi, "les electrons ... devraient s'etendre plus vite que les ions"
```

- `sim.mass(name)` integrates the block's $\rho$ (conservation diagnostic), compared to the initial
  mass `me0`/`mi0` (in run.py) by `assert_mass_conserved` (in checks.py),
  which returns the relative drift and raises `AssertionError` if $\ge$ `tol`.
- `pressure(U)` = `euler_pressure(U, gamma=GAMMA)` (in initial_conditions.py):
  $p=(\gamma-1)(E-\frac12\rho|u|^2)$.
- `disturbed(U, U0, thr)` (in run.py) = `mean(|p(U)-p(U0)| > thr)`, the fraction of cells where
  the pressure changed by more than `thr=0.02`: the extent of the front.
- `assert_finite(Ue, ...)` (in run.py, `assert_finite` in checks.py): no NaN, no Inf.

The multirate has no dedicated assert: it is exercised by the loop and indirectly validated by
stability (positivity holds, electron front > ion front).

---

## 5. Initial conditions

| Block | $\rho_0$ | $p_0$ | $dp$ | $c=\sqrt{\gamma p_0/\rho_0}$ at rest | initial measured $w$ |
|---|---|---|---|---|---|
| electrons | 0.01 | 1.0 | 0.5 | $\sqrt{1.4/0.01}=11.83$ | 14.48 (with the central overpressure) |
| ions | 1.0 | 1.0 | 0.5 | $\sqrt{1.4/1.0}=1.18$ | 1.45 |

The two gases start at rest ($u=v=0$) with a Gaussian overpressure bubble identical in shape. The
only difference is $\rho_0$. The sound speed at the pressure peak (center) is slightly higher than at
rest (the local overpressure raises $p$), hence the measured wave speed $w=14.48$ for the electrons
(vs $11.83$ at the background); the ratio $w_e/w_i$ stays exactly $10.0$ because the overpressure
multiplies both by the same factor (`figures/provenance.json`: `wave_speed_ratio_init = 10.0`).

`sigma2=0.02` is not passed by `run.py`: it is the default value of `euler_pressure_blob`.

---

## 6. Figures (generated by `make_figures.py`, in `figures/`)

`make_figures.py` replays the `run.py` loop identically (same initial conditions, same scheme, same
`step_adaptive(0.4)` x 20) but instruments each macro step to produce the time series that `run.py`
does not keep (it only reads the final state). Command:

```bash
cd /private/tmp/adc_cases-deeptut/two_euler && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
/opt/homebrew/anaconda3/bin/python3.12 make_figures.py
```

### `density_maps.png`: final density of the 2 gases

![Final density maps, electrons on the left, ions on the right](figures/density_maps.png)

- **Proves**: both maps are finite, coherent states (`assert_finite`), produced by the same scheme.
  The radial expansion hollowed out the center of both gases (minimum density at the overpressure
  point, pushed outward): electron scale $[0.008,\,0.011]$ around $\rho_0=0.01$, ion scale
  $[0.80,\,1.02]$ around $\rho_0=1.0$, contrast $\sim 100$ between the two blocks.
- **Suggested (not asserted)**: the electron map shows a diagonal cross and lobes that touch the
  edges: its front (10x faster) has already reached the periodic images and interferes with itself,
  while the ion map stays a clean ring, centered, far from the edges. This is the visual signature of
  "electrons more extended" (`fe=0.861` vs `fi=0.287`), but no assert tests the shape.
- **Not shown**: the map is at a fixed $t$; no transient dynamics here (see the time figures below).

### `masses.png`: mass conserved per species (invariant 1)

![Mass(t) of the two species (flat) and relative drift on a log scale under the tolerance](figures/masses.png)

- **Proves**: left panel, both masses are flat over the whole run (electrons $\sim 41$, ions
  $\sim 4096$, the 100 ratio of the densities). Right panel (relative drift, log), both blocks stay
  between machine noise ($\sim 2\times10^{-16}$, gray dotted) and $\sim 4\times10^{-14}$, that is
  $\ge 4$ orders of magnitude below the `1e-9` tolerance (black dash). Invariant 3.1 holds:
  conservation to the bit. Some ion steps hit the eps floor exactly (rigorously zero drift on that
  step).
- **Not shown**: the figure does not separate the error contribution of the electron sub-cycling (10
  sub-steps) from that of the single ion step; both drift at the same machine level.

### `positivity.png`: positivity rho_min / p_min vs t (invariant 2)

![rho_min and p_min of the two species vs t, all strictly positive](figures/positivity.png)

- **Proves**: left panel (log scale), `rho_min` electrons stays around $7\times10^{-3}$ (trough at
  $6.99\times10^{-3}$), `rho_min` ions decreases from $1.0$ toward $0.80$: neither passes through
  zero. Right panel (linear scale), `p_min` ions stays at $1.000$, `p_min` electrons dips to $0.905$
  at the lowest then rises again: both $\gg 0$. Invariant 3.2 holds; primitive reconstruction + HLLC
  preserves positivity through the most severe expansion (electrons).
- **Suggested**: the `p_min` electron trough around $t\approx0.02$ followed by a rise is the signature
  of the expansion passing then relaxing; not asserted, only the global minimum $>0$ is.
- **Not shown**: no guaranteed positivity bound (generic HLLC has no positivity limiter); it is an
  empirical invariant of this run, not a theorem.

### `multirate.png`: the fast block is sub-cycled automatically

![Number of sub-cycles n_e (around 10) and macro_dt pinned to the ions](figures/multirate.png)

- **Proves / measures**: left panel, the `electrons` block is sub-cycled $n_e=10$ at each macro step
  (touching $11$ in the early transient where the electron expansion briefly raises $w_e/w_i$ above
  10), the `ions` stay at $n_i=1$. Right panel, `macro_dt` $\approx 4.3\times10^{-3}$ is pinned to the
  slowest block (ions) and varies by $\pm 1.5\%$ as the ion sound speed evolves. This is the multirate
  `step_adaptive`: $n_b=\lceil w_b/w_{\min}\rceil$ (section 4.3), with no explicit configuration.
- **Not shown**: no assert tests $n_e=10$ nor the value of `macro_dt`; they are measured (this
  figure), not validated against a tolerance. The multirate is exercised, not asserted.

---

## 7. What the invariant does not capture

- Not plasma physics, not a reproduction. The "electrons"/"ions" blocks are not coupled: zero source,
  Poisson with zero right-hand side ($q=0$). No field links them. A coupled electrons+ions model via
  Poisson is [`multispecies`](../multispecies/) (coupled Euler + isothermal) or [`plasma`](../plasma/)
  (3 species + ionization); this case couples nothing.
- Asymmetry purely through the initial conditions. The only ingredient distinguishing the two blocks
  is $\rho_0$ ($0.01$ vs $1.0$). The factor 10 on the sound speed and the 10x sub-cycling follow from
  it; it is not a model difference.
- Finite periodic domain, short duration. 20 macro steps ($t_{\text{final}}=0.085$). The electron
  front already touches its periodic images (density map); the case stops before this interaction
  dominates, to remain a clean invariant test. Mass conservation, on the other hand, would hold
  indefinitely (periodic conservation law).
- Positivity not guaranteed. HLLC+primitive favors $\rho>0$, $p>0$ but does not bound them
  mathematically. A more violent expansion (larger $dp$, smaller $\rho_0$) could leave positivity; the
  checked invariant holds for this parameter set.
- CPU single-rank. No device / MPI / AMR validation here: native path composed on a single grid, no
  `.so` backend.

---

## 8. Reproducing

```bash
cd /private/tmp/adc_cases-deeptut/two_euler && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
/opt/homebrew/anaconda3/bin/python3.12 run.py
```

Prerequisites: `numpy` (the case's only Python dependency, `needs=[]`), the `adc` module (adc_cpp's
pybind11 bindings) built and imported with the same interpreter that compiled it (ABI suffix
`cpython-3XY`), and the `adc_cases` package on the `PYTHONPATH`. No C++ compiler is required at run
time: the model is composed of native bricks already compiled into `adc`.

Captured output (run of 2026-06-08, deterministic over consecutive runs):

```text
== two_euler : deux Euler independants (meme schema HLLC + recon primitive) ==
  c_electrons/c_ions ~ 10.0 (electrons 100x plus legers)
  masse      : electrons drel=1.02e-14  ions drel=2.22e-16
  positivite : rho_min e=7.707e-03 i=8.033e-01 ; p_min e=9.324e-01 i=1.000e+00
  front (frac cellules perturbees) : electrons=0.861 ions=0.287
OK two_euler
```

`OK two_euler` is printed only if all the asserts pass, return code 0. Measured cost ~0.45 s of wall
time (including `adc` + numpy import; the $64^2$ x 2 blocks x 20 macro steps computation is negligible
next to the import). The signs and the order of magnitude are stable; the last digits of the mass
drifts vary with the BLAS library and the summation order. The `drel` value printed by `run.py`
(single final read) and the `mass_drift_rel_*_max` of `figures/provenance.json` (max over the 20
steps) differ slightly by construction (instant vs max), not by non-determinism.

`make_figures.py` regenerates the 4 figures + `figures/provenance.json` (same initial conditions,
~0.6 s).

## File map

| File | Role |
|---|---|
| `run.py` | the case: composes 2 Euler blocks, sets the initial conditions, advances multirate, asserts |
| `make_figures.py` | replays the instrumented loop, writes `figures/*.png` + `provenance.json` |
| `figures/*.png` | density_maps, masses, positivity, multirate |
| `figures/provenance.json` | adc_cpp/adc_cases SHA, backend, resolution, measured numbers |
| `adc_cases/models.py` (`euler`) | pure Euler model (composed of native bricks) |
| `adc_cases/common/initial_conditions.py` | `euler_pressure_blob` (initial conditions), `euler_pressure` (diagnostic) |
| `adc_cases/common/checks.py` | `assert_mass_conserved`, `assert_finite` (invariants) |
