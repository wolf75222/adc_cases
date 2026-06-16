# schur_magnetized_cartesian: the Schur complement removes the explicit cyclotron bound

A timing study. A stiff magnetized Cartesian isothermal fluid is integrated two ways: the Lorentz
force $m\times\Omega$ advanced explicitly (after transport), or condensed into an implicit source
stage by a Schur complement (`CondensedSchurSourceStepper`). For each you measure the largest stable
time step and check the falsifiable prediction: the explicit path caps at $dt\,\omega_c=O(1)$ (the
cyclotron rotation bound), the Schur path removes that bound and the step is then limited only by
transport. This case does not re-derive the physics of the magnetized isothermal model: that belongs
to [`../magnetic_isothermal_dsl/`](../magnetic_isothermal_dsl/).

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `experimental` (`cases_manifest.toml`, `schur_magnetized_cartesian/run.py`, `ci = false`, `needs = ["cxx"]`). A measurement prototype: the path is not finalized (the Schur stage is wired through a private hook, AOT backend), not a published reproduction. |
| Inputs | $16^2$ grid, $L=1$, **periodic** ($h=0.0625$); IC $\rho=1+0.05\cos(2\pi x)$, oblique velocity $u=v=0.5$; $c_s^2=10^{-4}$ (slow transport, not limiting), $q=-1$, $\alpha=1$, $B_z=\omega_c$ constant; minmod + Rusanov, SSPRK2 transport (AOT); Schur stage $\theta\in\{0.5,1.0\}$ |
| Outputs | `dt_stable` per method (explicit / Schur $\theta{=}0.5$ / Schur $\theta{=}1.0$), product $dt\,\omega_c$, gain; console + `out/<case>/dt_stable.csv` (`--csv` option); 2 figures in `figures/` + `figures/provenance.json` |
| Guaranteed invariants | no `assert`: this is a measurement, not a validation. The stability criterion is `is_stable` (in run.py): at a given $dt$, density stays finite, $\lvert\rho\rvert_{\max}\le 10^3$ and $\rho_{\min}\ge -10^{-2}$ at every step up to $t_{end}$ |
| Proven (measured, not asserted) | explicit: $dt_{stable}\propto 1/\omega_c$ (measured $5.62\times10^{-2}\to5.62\times10^{-4}$ from $\omega_c{=}10^2$ to $10^3$, factor of exactly 100), bounded product $dt\,\omega_c\le O(1)$; at $\omega_c{=}10^4$ no tested $dt$ is stable ($dt_{stable}=0$). Schur: $dt_{stable}=3.16\times10^{-1}$ independent of $\omega_c$, set by transport; measured gain 562x ($\omega_c{=}10^3$, full run) up to unbounded ($\omega_c{=}10^4$) |
| Does not prove | not a published reproduction; no fidelity to any paper (the Hoffart target [arXiv:2510.11808](https://arxiv.org/abs/2510.11808) is the separate case [`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/), itself a pending `reproduction-candidate`). The Schur stage is wired through the private hook `sim._s.set_source_stage(...)`, not through `adc.Split(Explicit, CondensedSchur)` (not wired on AOT). `dt_stable` is a discrete bound (geometric sweep at a quarter-decade), not a fine threshold; the stability criterion is heuristic (finiteness + bounds + positivity), not spectral. A fabricated stiff case (tiny $c_s^2$): the gain is specific to this operating point |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, DSL backend `aot`, $16^2$, macOS arm64; figure (1) read from the documented full run `out/dt_stable.csv` ($\omega_c{=}10^3$, $t_{end}{=}1$), figure (2) from a fresh targeted measurement (3 $\omega_c$, $t_{end}{=}0.05$, ~231 s for the three sweeps); `figures/provenance.json` |

By the end you will know: why the cyclotron rotation imposes $dt\,\omega_c<O(1)$ on the explicit
path, which implicit operator the Schur stage condenses to remove that bound (anchor
`lorentz_eliminator.hpp` / `condensed_schur_source_stepper.hpp`), how `largest_stable_dt` measures
the threshold, and why this measurement goes through a private hook and an AOT backend.

---

## 1. Reference: the fluid physics is not re-derived here

The magnetized isothermal model (flux, eigenvalues, closure $p=c_s^2\rho$, Lorentz force, system
Poisson) belongs to the case [`../magnetic_isothermal_dsl/`](../magnetic_isothermal_dsl/), which
validates it (analytic oracle for the Lorentz term, cyclotron rotation with conserved magnitude).
This case reuses those equations without replaying them and measures a property of the integrator.
The equations solved (conservative variables $U=(\rho,m_x,m_y)$, $m=\rho v$, $\Omega=\omega_c\hat z$,
$\omega_c=B_z$):

$$\partial_t\rho+\nabla\cdot(\rho v)=0,\qquad
\partial_t m+\nabla\cdot\!\Big(\tfrac{m\,m^{\mathsf T}}{\rho}+c_s^2\rho\,I\Big)=q\rho(-\nabla\phi)+m\times\Omega,\qquad
\Delta\phi=q\rho.$$

Projected in 2D, the rotation $m\times\Omega$ gives $(+B_z m_y,\,-B_z m_x)$ on momentum and $0$ on
energy ($v\times B\perp v$): this is exactly the `MagneticLorentzForce` convention
(in source.hpp), $s_1=+q_{om}B_z m_y$, $s_2=-q_{om}B_z m_x$. This rotation is the stiff term: it
rotates $m$ at angular frequency $\omega_c$ without dissipating it. The larger $\omega_c$, the faster
the rotation, the finer the step the explicit advance must take to follow it. This justifies the
Proven clause (the explicit bound) and its converse (the Schur path removing it).

---

## 2. The falsifiable prediction: the $dt\,\omega_c$ bound and the Schur gain

Integrating a pure rotation $\dot m = \Omega\times m$ with an explicit scheme is conditionally
stable: the step must satisfy $dt\,\omega_c<C$ with $C=O(1)$ (the exact value depends on the RK
scheme). At large $\omega_c$, $dt$ collapses as $1/\omega_c$. The testable prediction has two parts,
both checked in section 6:

1. explicit: $dt_{stable}\,\omega_c$ is bounded (an $O(1)$ plateau); equivalently, $dt_{stable}$
   decreases as $1/\omega_c$. A bound that grows with $\omega_c$ would reveal that the source is not
   the limiting factor (transport would dominate); a bound that drops faster than $1/\omega_c$ would
   reveal a source-transport coupling not captured by this toy model.
2. Schur: $dt_{stable}$ is independent of $\omega_c$ (the cyclotron bound disappears), set by the
   transport step $\sim h/c_s$. The gain $dt_{stable}^{Schur}/dt_{stable}^{expl}$ therefore grows
   linearly with $\omega_c$: this is the announced gain factor.

---

## 3. The implicit operator the Schur stage solves (core anchor)

Why does the Schur path remove the bound? It advances the source not by an explicit increment but by
solving the implicit advance of the rotation. The Lorentz eliminator encodes the implicit rotation
matrix $B=\begin{pmatrix}1&-w\\w&1\end{pmatrix}$, $w=\theta\,dt\,B_z$, $\det B=1+w^2>0$ for any real
$w$ (`LorentzEliminator` in lorentz_eliminator.hpp): $B$ is invertible whatever $dt\,B_z$ is, so the reconstruction
$v^{n+\theta}=B^{-1}(v^n-\theta\,dt\,\nabla\phi^{n+\theta})$ (`SchurReconstructKernel`
in condensed_schur_source_stepper.hpp) stays bounded even at arbitrarily large $\omega_c$. This
is the algebraic root of the absence of a bound: $\det B=1+w^2$ never vanishes.

The stage assembles the condensed operator $A_{op}=I+c\,\rho\,B^{-1}$ with $c=\theta^2 dt^2\alpha$ and
solves $L_{schur}(\phi)=-\nabla\!\cdot(A_{op}\nabla\phi)=\text{rhs}$ by matrix-free BiCGStab
preconditioned with multigrid (`CondensedSchurSourceStepper::step` in condensed_schur_source_stepper.hpp). Two $\theta$
values are measured: $\theta=0.5$ (Crank-Nicolson, marginally stable for a pure rotation) and
$\theta=1.0$ (backward Euler, unconditionally stable). The step from the $\theta$-stage to $n+1$ is a
linear extrapolation with factor $1/\theta$ (`SchurExtrapolateVelocityKernel` in condensed_schur_source_stepper.hpp); energy
is touched only if the Energy role exists (absent here, the 3-variable isothermal model).

---

## 4. The three layers: who computes what (DSL case: the middle layer is expressions)

The model is written once as `adc.dsl.Model` (`magnetized_model` in run.py), instantiated in
two variants that share flux/eigenvalues/Poisson and differ only by their source.

| `run.py` symbol | Layer | What happens |
|---|---|---|
| `sim.add_equation("plasma", model=compiled, spatial=adc.FiniteVolume(minmod, rusanov, conservative), time=adc.Explicit())` (`build` in run.py); `sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta, alpha)` (`build` in run.py) | Python composes and measures | choice of transport scheme, wiring of the condensed source stage; the `largest_stable_dt` sweep (in run.py) reads density to judge stability |
| `m.flux(...)`, `m.eigenvalues(...)`, `m.source([0, q*rho*(-gx)+bz*my, q*rho*(-gy)-bz*mx])` (local) or `m.source([0*rho,0*mx,0*my])` (schur), `m.elliptic_rhs(q*rho)` (`magnetized_model` in run.py) | expressions that `adc.dsl` compiles and freezes | the exact convention of the isothermal flux, the eigenvalues $v_n\pm c_s$, the Lorentz term $(+B_z m_y,-B_z m_x)$, the right-hand side $q\rho$ |
| `CondensedSchurSourceStepper::step` (assembles $A_{op}=I+c\rho B^{-1}$, BiCGStab+MG, reconstructs $B^{-1}$) | per-cell kernel (device) | the real implicit solve of the source, with no Python callback; named device-clean functors |

The `schur` variant zeroes its local source (`magnetized_model` in run.py): the condensed stage carries the full
source, and leaving it locally would advance it twice. The DSL names no scenario: it re-declares the
formulas of the `IsothermalFlux` / `MagneticLorentzForce` / `ChargeDensity` bricks, a table of
conventions anchored in [`../magnetic_isothermal_dsl/`](../magnetic_isothermal_dsl/) (section 2).

---

## 5. The measurement: `largest_stable_dt` and the stability criterion (in run.py)

Initial conditions `initial_state` (in run.py): cosine density + constant oblique velocity.
The velocity $u=v=0.5$ makes both momentum components nonzero, so the Lorentz rotation
$(+B_z m_y,-B_z m_x)$ is active from the first step (otherwise $m_x=m_y=0$: nothing to rotate, stiffness
invisible).

```python
sim.set_magnetic_field(omega_c * np.ones((n, n)))     # B_z = omega_c everywhere
if schur:
    sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta, alpha)
```
- `set_magnetic_field` populates the extended aux channel (canonical index 3, read by the Lorentz
  term and by the Schur stage) with a constant field $\omega_c$: so $\omega_c=B_z$ is the only
  stiffness parameter swept.

```python
def is_stable(...):
    sim = build(...); nst = max(2, ceil(t_end / dt))
    for _ in range(nst):
        sim.step(dt)
        d = np.asarray(sim.density("plasma"))
        if not np.isfinite(d).all() or abs(d).max() > 1e3 or d.min() < -1e-2:
            return False
    return True
```
- Stability = density finite, bounded ($\le 10^3$), nearly positive ($\ge -10^{-2}$) at every step up
  to $t_{end}$. This is a heuristic proxy: a numerical instability makes the density diverge or change
  sign before anything else. The threshold $-10^{-2}$ tolerates a small limiter undershoot without
  accepting an outright negative density; $10^3$ catches exponential blow-up.

```python
def largest_stable_dt(...):
    best = 0.0
    for e in range(-16, 5):
        dt = 10.0 ** (e / 4.0)
        if dt > dt_max: continue
        if is_stable(..., dt, ...): best = dt
    return best
```
- Geometric sweep at a quarter-decade ($dt=10^{e/4}$), from smallest to largest: `best` keeps the
  largest stable $dt$. This is a discrete bound (quarter-decade resolution, factor
  $10^{1/4}\approx1.78$ between steps), not a continuous threshold: two methods capped at the same
  step return the same $dt_{stable}$ (cf. section 6, $\theta=0.5$ vs $\theta=1.0$ at short $t_{end}$).

---

## 6. Figures (generated by `make_figures.py`, in `figures/`)

No physical figure: a timing case shows $dt_{stable}$ values, not a field. The numbers are read, not
invented: panel (1) from the documented full run `out/dt_stable.csv`, panel (2) from a fresh targeted
measurement `/tmp/schur_measure.json` (fields in `figures/provenance.json`).

### `timing_dt_stable.png`: stable dt per method (reference point)

![Log bars of stable dt: explicit at 3.16e-4 below the bound, Schur theta=0.5 at 0.178 (562x), theta=1.0 at 0.316 (1000x)](figures/timing_dt_stable.png)

- Proven (measured): at $\omega_c=10^3$, $t_{end}=1$, the explicit path caps at
  $dt_{stable}=3.162\times10^{-4}$, i.e. $dt\,\omega_c=0.316$: the $O(1)$ cyclotron bound, dashed
  line. The Schur path holds at $dt_{stable}=0.178$ ($\theta{=}0.5$, $dt\,\omega_c=178$, gain 562x)
  and $0.316$ ($\theta{=}1.0$, $dt\,\omega_c=316$, gain 1000x): two to three orders of magnitude above
  the explicit bound.
- Suggested (not asserted): $\theta=1.0$ (backward Euler, unconditionally stable) gains more than
  $\theta=0.5$ (Crank-Nicolson, marginally stable), consistent with rotation theory; but no assert
  ranks the two $\theta$ values, and the gap (a single sweep step) is at the measurement resolution,
  not a fine margin.
- Not shown: these $dt_{stable}$ values are quarter-decade bounds, not continuous thresholds; the
  Schur path at $0.316$ approaches the transport step ($\sim h/c_s$), evidence that transport rather
  than the source now limits (read from the next panel).

### `timing_vs_omega.png`: explicit as 1/omega_c, Schur flat

![Two panels: left, stable dt vs omega_c (explicit follows slope -1, Schur flat on the transport step, explicit unstable at omega_c=1e4); right, dt*omega_c (explicit bounded O(1), Schur grows linearly)](figures/timing_vs_omega.png)

- Proven (fresh measurement, $t_{end}=0.05$): the explicit path follows slope $-1$ ($dt_{stable}$ goes
  from $5.62\times10^{-2}$ at $\omega_c{=}10^2$ to $5.62\times10^{-4}$ at $\omega_c{=}10^3$, factor of
  exactly 100 for a factor 10 on $\omega_c$), then at $\omega_c{=}10^4$ no tested $dt$ is stable
  ($dt_{stable}=0$, annotated). The Schur path is flat at $3.16\times10^{-1}$ for all $\omega_c$: the
  cyclotron bound has disappeared, the step is set just above transport ($4.42\times10^{-2}$, dashed).
- Suggested: the right panel shows the explicit $dt\,\omega_c$ crossing below the line $1$ at
  $\omega_c{=}10^3$ ($0.562$): this is the signature of the $O(1)$ bound. At $\omega_c{=}10^2$ the
  product is $5.62$, above $1$: there, the explicit path is limited by transport ($5.62\times10^{-2}>$
  transport $4.42\times10^{-2}$ at the neighboring step), not by the cyclotron. The bound/transport
  crossover is plausible by eye but not asserted.
- Not shown: $\theta=0.5$ and $\theta=1.0$ return the same $dt_{stable}$ here (both capped at the
  transport step, which the sweep does not distinguish at short $t_{end}$); their gap appears only in
  the full run ($t_{end}=1$, panel 1). The gain at $\omega_c{=}10^4$ is unbounded (explicit zero): a
  finite ratio is meaningless, and we report "explicit unstable at every tested dt".

---

## 7. Why this measurement goes through non-standard paths (platform caveats)

Category `experimental`: the path is a prototype, and three choices must be named.

- **Private hook instead of `adc.Split`.** The high-level Schur splitting
  `adc.Split(adc.Explicit, adc.CondensedSchur)` is wired only by the native `production` path: the AOT
  `.so` ABI does not carry the SSPRK3 substep that `adc.Split` expects. The case therefore wires the
  condensed stage directly via `sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta,
  alpha)` (`build` in run.py), which runs the same C++ (`CondensedSchurSourceStepper`) that `adc.Split`
  would produce on the production side. This is a low-level access, subject to renaming: promoting it
  requires wiring `adc.Split` on AOT.
- **AOT backend (host-marshaled).** The `production` DSL backend (native zero-copy) does not link on
  macOS arm64 (`dlopen` failure, header ABI of the prebuilt module): only `aot` is exercised. The AOT
  path exposes only SSPRK2 for transport (not SSPRK3). No effect on the conclusion: the measured
  factor comes from the source (Schur stage), not from the transport RK scheme, which both variants
  share.
- **Fabricated stiffness.** $c_s^2=10^{-4}$ is chosen deliberately tiny so that the transport step
  $\sim h/c_s$ ($\approx4.4\times10^{-2}$) stays large compared to $1/\omega_c$, isolating the source
  stiffness. The gains (562x at $\omega_c{=}10^3$, unbounded at $10^4$) are specific to this operating
  point; a case where transport already limited the step would not see such a factor.

---

## 8. Reproduce (exact command, measured cost)

Do not run in CI (long, experimental). The full sweep ($\omega_c=10^3$, $t_{end}=1$) writes the
reference CSV for panel (1):

```bash
cd /private/tmp/adc_cases-deeptut/schur_magnetized_cartesian
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py --csv     # writes out/dt_stable.csv
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py  # 2 figures + provenance.json
```

Prerequisites: the C++ `adc` module (pybind11 bindings) on the `PYTHONPATH`, `adc_cases` importable, a
C++20 compiler (`needs = ["cxx"]`: the DSL compiles both `.so` files on the fly in `aot`, ~12 s),
`numpy`, and `matplotlib` for the figures. The core headers are located by `adc_include()`.

Measured cost (macOS arm64, single core): the fresh targeted measurement for panel (2), three
`largest_stable_dt` sweeps at $\omega_c\in\{10^2,10^3,10^4\}$ and $t_{end}=0.05$, took 112 s + 81 s +
37 s = ~231 s (the small-$dt$ trials run the most steps and dominate). The default full run
($t_{end}=1$, $\omega_c=10^3$) is ~20x longer on the small-$dt$ steps: it was not re-run here, its
numbers come from `out/dt_stable.csv` (an earlier run, same configuration). Platform caveat: the
signs, the order of magnitude (gain $\sim10^2$-$10^3$x), the explicit slope $-1$ and the Schur plateau
are stable; the exact $dt_{stable}$ is a quarter-decade bound, it can jump by one step
($\times10^{1/4}$) with the platform, the compiler or $t_{end}$.

## File map

| File | Role |
|---|---|
| `run.py` | DSL model `local`/`schur`, build, `largest_stable_dt` measurement, console + CSV |
| `make_figures.py` | 2 timing figures (bars + vs omega_c) + `provenance.json`; reads the CSV and the measurement JSON, recomputes nothing |
| `figures/timing_dt_stable.png` | log bars of $dt_{stable}$ per method (documented reference point) |
| `figures/timing_vs_omega.png` | $dt_{stable}$ and $dt\,\omega_c$ vs $\omega_c$ (fresh measurement) |
| `figures/provenance.json` | SHA, backend, sources of the two panels, measured numbers |
| `out/dt_stable.csv` | table of the full run ($\omega_c{=}10^3$, $t_{end}{=}1$), source of panel (1) |
| `../magnetic_isothermal_dsl/` | validates the shared magnetized isothermal model (Lorentz oracle, rotation) |
| `../hoffart_euler_poisson_dsl/` | targets the full paper (Schur), a pending `reproduction-candidate` |
