# diocotron_amr: diocotron instability on a multi-patch AMR grid

The same diocotron dynamics as [`../diocotron/`](../diocotron/) (a charge advected by its own
ExB drift), carried no longer on a uniform grid but on an adaptive-mesh-refinement hierarchy
`adc.AmrSystem`: a coarse base level plus a fine level re-decomposed dynamically (Berger-Rigoutsos
regrid) to track the charge band, with conservative reflux at the coarse/fine interfaces. This case
does not measure a growth rate: it validates that refinement comes from the tagging criterion (and
not from the hierarchy build alone), that it changes the solution where it acts, and that reflux
conserves mass to machine roundoff.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `validation` (`cases_manifest.toml`, `ci = true`, `needs = []`). Not a published reproduction: you verify invariants of the AMR API, not a paper curve. |
| Inputs | base grid $64\times 64$, $L=1$, periodic; AMR `regrid_every=10`, 1 fine level, tag threshold `threshold = n_{i0} + 0.15`; IC `band_density` (gaussian band, `mode=4`, `width=0.05`, `disp=0.02`, `amp=1`, `floor=1`); model `diocotron(B0=1, alpha=1, n_i0=<n_e>)`; neutralizing background $n_{i0}=\langle n_e\rangle$; 40 steps at `CFL=0.4`, NoSlope + Rusanov scheme |
| Outputs | stdout trace (patches, mass, drel per step); 3 diagnostic figures in `figures/` + `figures/provenance.json` |
| Guaranteed invariants | the `assert`s in `main` (in run.py): `n_patches() >= 2` at every step; `drel < 1e-9`; finite density; `min(patches_seen) > npatch_ctrl`; `gap > 1e-3` |
| Proves | (1) the band is covered by 2 fine patches at every step, vs 1 for a control run with an unreachable threshold: refinement comes from tagging; (2) the refined solution differs from the unrefined one by `max|delta| = 6.40e-2` (`run.py`, sup difference > 1e-3); (3) the AMR mass is conserved to `drel <= 3.06e-15` (reflux); (4) finite density everywhere |
| Does not prove | this is not a reproduction of a rate $\gamma_l$ or of a paper figure (the established reproduction lives in [`../diocotron/`](../diocotron/), uniform grid; the full magnetized Euler-Poisson candidate in [`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/), pending status). The CI caps at 2 patches and 1 fine level: no deep hierarchy nor a large patch count is tested. The threshold is tuned empirically, not derived from an error estimator. No assert tests convergence or the growth rate. |
| Provenance | adc_cpp `01873299`, adc_cases `7c7a3403`, native backend (`adc.AmrSystem` + `adc.System`), base $64^2$, Python 3.12.2, macOS arm64; `figures/provenance.json` |

By the end you will know: why this AMR does not touch mass (the reflux math + periodic Poisson),
where it concentrates resolution (the band edges, where the gradient lives), and what the validation
does not cover (a single level, 2 patches, no rate).

---

## 1. Physics: the same instability, on an adaptive mesh

The mechanism is the one of the parent case: a charge density $n_e$ creates its potential $\phi$ via
Poisson, drifts at $\mathbf{v}=(\mathbf{E}\times\mathbf{B})/B_0^2$ (divergence-free velocity, so
$n_e$ is purely advected), and the shear of the differential rotation winds up the perturbation. The
full derivation (rotation $\Omega(r)$, Rayleigh criterion, eigenvalue problem, rate $\gamma_l$) is in
[`../diocotron/README.md` section 4](../diocotron/README.md); it is not repeated here.

Two differences from the parent, both serving AMR validation:

- **IC geometry.** Here the charge is a `mode=4` undulated horizontal gaussian band
  (`band_density`, section 8), not the benchmark ring. A band offers a sharp, localized transverse
  gradient (the two band edges), which is exactly what the tag criterion must catch. The domain is
  periodic (not conducting-wall), which simplifies the Poisson coupling on the hierarchy.
- **Mesh.** The parent uniform grid becomes a hierarchy: coarse $64^2$ + one fine level,
  re-decomposed every 10 steps to track the band. This is the object under test.

The physical question of the case is therefore not "what rate" but: does an adaptive mesh change the
conserved physics? The expected (and asserted) answer is no for mass (reflux protects it) and yes
for the local detail (the fine level resolves the edge better). This is an invariant validation, in
the sense of the guide: mass is the invariant, reflux is the reason.

---

## 2. Equations and who computes them

The block evolves a scalar density $n_e(x,y,t)$ advected by the ExB drift, coupled to a periodic
Poisson with a neutralizing background:

$$\partial_t n_e + \nabla\cdot(n_e\,\mathbf{v}) = 0,\qquad
\mathbf{v}=\frac{1}{B_0}(-\partial_y\phi,\ \partial_x\phi),\qquad
-\nabla^2\phi = \alpha\,(n_e - n_{i0}).$$

| Block | Equation | `adc` brick |
|---|---|---|
| State | scalar density $n_e$ | `adc.Scalar()` |
| Transport | $\partial_t n_e+\nabla\cdot(n_e\mathbf v)=0$, ExB drift | `adc.ExB(B0=1)` |
| Source | none | `adc.NoSource()` |
| Elliptic | $-\nabla^2\phi=\alpha(n_e-n_{i0})$, periodic | `adc.BackgroundDensity(alpha=1, n0=n_{i0})` |

This is exactly `models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0)` (in models.py), the
same model as the parent: only the mesh (AMR), the IC (band) and the BC (periodic) change. Who
computes what, the three layers pinned to lines of `run.py`:

| `run.py` line | Layer | What happens |
|---|---|---|
| `sim.add_block("ne", model=models.diocotron(...), spatial=adc.Spatial(none=True))` (`build_sim` in run.py) | Python composes | choice of model, of spatial scheme (NoSlope + Rusanov), carried onto the hierarchy; the explicit integrator is the default of `step_cfl` |
| `models.diocotron(...)` -> `ExB` / `BackgroundDensity` (`include/adc/physics/{hyperbolic,elliptic}.hpp`) | C++ brick freezes the physics | the exact convention of the flux $n\,v(\mathrm{dir})$, of the eigenvalue $v(\mathrm{dir})$, of the RHS $\alpha(n-n_{i0})$ |
| `assemble_rhs<NoSlope,Rusanov>` + Berger-Rigoutsos regrid + reflux + Poisson `geometric_mg` (`step_cfl` in run.py) | per-cell / per-patch kernel | the actual computation: transport on each patch, hierarchy re-decomposition, flux correction at interfaces, with no Python callback in the hot path |

`models.diocotron` names no scenario on the core side: the word "diocotron" lives in `adc_cases`, the
physics is a composition of generic bricks. `adc.AmrSystem` is the refined counterpart of
`adc.System`; it carries the same block, plus the regrid/reflux machinery.

---

## 3. The falsifiable prediction: reflux => invariant mass, AMR => modified local solution

The case contrasts two runs that differ only by the tag threshold (`build_sim` in run.py,
reused for both):

- **nominal**: `threshold = n_i0 + 0.15` (`REFINE_FRAC` in run.py), the band cells are tagged;
- **control**: `threshold = 1e30` (`NO_REFINE` in run.py), no cell exceeds it, the criterion
  never tags.

Three predictions fall out of this contrast, each justifying a Proves clause of the contract:

1. **Refinement comes from tagging**: nominal `n_patches()` $\ge 2$ at every step (in run.py),
   but the control stays at $1$ (in run.py). If the hierarchy produced patches without tagging,
   the control would have as many: it does not.
2. **AMR changes the solution**: the nominal density projected differs from the control density by
   `gap > 1e-3` (in run.py). An equality would signal an inert fine level.
3. **Reflux conserves mass**: `drel < 1e-9` at every step (in run.py). Without reflux, each
   regrid would leak mass at the coarse/fine interfaces.

The figures of section 6 confront these three predictions to the eye and to the number.

---

## 4. Math: why mass is invariant (and why $n_{i0}=\langle n_e\rangle$)

This is the heart of an invariant validation: not a rate derivation, but the structural reason why
mass does not move, and why the neutralizing background is mandatory.

### 4.1 Conservation = divergence form + reflux

Transport is in conservative form $\partial_t n_e+\nabla\cdot(\mathbf F)=0$ with
$\mathbf F=n_e\mathbf v$. On a uniform grid, a finite-volume scheme conserves exactly the sum
$\sum_{ij} n_e\,h^2$: the flux leaving a cell is the flux entering its neighbor (telescoping). This
is what you read on the uniform run: `drel <= 6.12e-15` (section 6, fig. 3).

On an AMR hierarchy this telescoping breaks at the coarse/fine interfaces: the coarse cell sees a
flux computed at its step, the neighboring fine cell a flux computed at its own; the two do not
coincide, and mass drifts at each regrid. Reflux (refluxing) corrects it: it replaces the coarse
flux of the interface by the sum of the coincident fine fluxes, restoring the telescoping. Asserted
result: `drel <= 3.06e-15` on the AMR (section 6, fig. 3), of the same order as the uniform. This is
the observable that proves reflux acts: without it, the trace would show a growing drift at each step
multiple of `regrid_every=10`, not a roundoff floor.

### 4.2 Why $n_{i0}=\langle n_e\rangle$: solvability of the periodic Poisson

On a periodic domain, $-\nabla^2\phi=f$ has a solution only if $\langle f\rangle=0$ (the integral of
the Laplacian over a torus is zero; this is the compatibility condition). Here
$f=\alpha(n_e-n_{i0})$, so you need $\langle n_e\rangle=n_{i0}$. The case guarantees it by measuring
the background: `n_i0 = float(ne.mean())` (in run.py), value $n_{i0}=1.088623$. An arbitrary
$n_{i0}$ (e.g. $0$) would violate compatibility and the multigrid would not converge to a field with
a defined mean. This $n_{i0}$ also serves as the reference point for the tag threshold:
`threshold = n_i0 + 0.15 = 1.238623`, i.e. "the density exceeds the background by $0.15$", which tags
only the band (not the floor at $1.0$).

### 4.3 The tolerance `TOL_MASS = 1e-9`, justified by an order of magnitude

`TOL_MASS = 1e-9` (in run.py) sits between the measured machine noise ($\mathrm{drel}\sim
3\times 10^{-15}$, i.e. the floating-point roundoff accumulated over $40$ steps + regrids) and the
alarm threshold you would want: a reflux leak would be at least $O(h)\sim 10^{-2}$ per regrid. The
margin between $10^{-15}$ measured and $10^{-9}$ asserted is six orders: wide enough not to trip on
BLAS variability, tight enough to catch a reflux leak from the very first regrid. Likewise,
`MIN_SOLUTION_GAP = 1e-3` (in run.py) is tuned well below the measured difference $\sim 6\times
10^{-2}$ (section 6) and well above the noise: it attests a real effect, not a scheme wobble.

---

## 5. The code, function by function (`run.py`)

The file reads in two parts: a `build_sim` factory and a `main`.

**`build_sim(ne, n_i0, threshold)` (in run.py)** builds an `AmrSystem` identical to the nominal
up to the threshold (it is also what serves the control, guaranteeing that only the threshold
changes):

```python
sim = adc.AmrSystem(n=N, L=L, regrid_every=10, periodic=True)
sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
              spatial=adc.Spatial(none=True))
sim.set_refinement(threshold=threshold)
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne)
```
- `adc.AmrSystem(n=64, regrid_every=10, periodic=True)`: hierarchy with a $64^2$ base level, regrid
  every 10 steps, periodic BCs (inherited by the Poisson).
- `adc.Spatial(none=True)`: NoSlope limiter (order-1 reconstruction, "none") + Rusanov flux
  (default of `Spatial`, cf. signature `Spatial(limiter, flux, recon, *, none, ...)`). Order 1,
  dissipative, chosen for a band that winds up on a coarse AMR grid without oscillations.
- `set_refinement(threshold)`: the discriminating parameter. Cells with $n_e>\text{threshold}$ are
  tagged, and the fine level is re-decomposed into rectangular patches (Berger-Rigoutsos) to cover
  them.
- `set_poisson(rhs="charge_density", solver="geometric_mg")`: geometric multigrid, RHS carried by
  the `BackgroundDensity` brick of the model.

**`main()` (in run.py)**:

```python
ne = band_density(N, L, amp=1.0, width=0.05, mode=MODE, disp=0.02)  # band IC
n_i0 = float(ne.mean())                                            # neutralizing background
sim = build_sim(ne, n_i0, threshold=n_i0 + REFINE_FRAC)            # nominal run
for k in range(NSTEPS):
    sim.step_cfl(0.4)                                              # 1 macro-step CFL=0.4
    npatch = sim.n_patches()                                       # current fine patches
    drel = relative_drift(mass, mass0)                             # mass drift
    assert npatch >= 2                                             # >= 2 patches
    assert drel < TOL_MASS                                         # mass conserved
    assert_finite(sim.density(), ...)                             # no NaN/Inf
```
- `step_cfl(0.4)` (in run.py): one explicit macro-step at CFL=0.4 ($dt=\text{CFL}\cdot h/w_{\max}$),
  which triggers the regrid every `regrid_every` steps and applies the reflux.
- `n_patches()` (in run.py) returns the number of current fine patches; the assert $\ge 2$
  requires a multi-patch coverage of the band, not just "a fine level exists".
- `relative_drift(mass, mass0)` (in checks.py) = $|m-m_0|/\max(|m_0|,10^{-30})$.
- `assert_finite` (in checks.py): no NaN, no Inf.

The control run (in run.py) rebuilds the same IC with `threshold = NO_REFINE = 1e30`, advances
40 steps, then:
```python
gap = float(np.abs(dens - dens_ctrl).max())            # sup difference nominal vs control
assert min(patches_seen) > npatch_ctrl                 # the threshold discriminates
assert gap > MIN_SOLUTION_GAP                          # refinement changes the solution
```
These two asserts close the logical loop: the patches come from the tag (otherwise the control would
have as many), and they act on the solution (otherwise `gap` would be zero).

API note: on `AmrSystem`, `mass()` / `density()` / `n_patches()` are read without a block name (the
hierarchy aggregates the single block); on uniform `adc.System` it is `mass("ne")` / `density("ne")`
with the name. The figures (section 6) exploit this difference to compare the two paths.

---

## 6. Figures (generated by `make_figures.py`, in `figures/`)

`make_figures.py` replays the same physics on two paths, AMR (`adc.AmrSystem`) and uniform
(`adc.System` $64^2$), with the same IC / the same model / the same scheme, and changes only the
mesh. All the numbers below are those of the run (cf. `figures/provenance.json`).

### `density_compare.png`: same dynamics, uniform vs AMR

![Final density: uniform 64x64, AMR base 64x64 + fine level, and their difference](figures/density_compare.png)

- **Proves** (by the asserts of `run.py` and the measurement here): the two runs carry the same
  dynamics (band modulated 4 times, AMR $n_e^{\max}=1.967$ vs uniform $=1.920$); the difference panel
  is nonzero, `max|delta n_e| = 8.68e-2` (of the same order as the `gap=6.40e-2` asserted in
  run.py, different measurement windows): AMR modifies the solution.
- **Suggested (not asserted)**: the difference is structured at the band edges (alternating red/blue
  lobes along $y\approx 0.45$ and $0.57$), not a diffuse noise: this is exactly where the fine level
  resolves the transverse gradient better. Visible, not tested by a spatial assert.
- **Not shown**: neither map is compared to a converged reference solution; you do not prove which
  one is "the right one", only that they differ where AMR acts.

### `patch_map.png`: where AMR concentrates resolution

![Footprint of tagged cells at 3 instants + n_patches over time](figures/patch_map.png)

- **Proves**: `n_patches()` is 2 at every step (right panel, flat line; `patches observed = [2]` in
  the provenance), which satisfies the assert $\ge 2$ (in run.py). The band is indeed covered by
  several fine patches, not a single degenerate level.
- **Suggested**: the footprint of tagged cells (proxy for the fine patch coverage, $\approx 500$
  cells, i.e. ~12 % of the domain) follows the band: the 4 lobes of the modulation are visible at
  $t=0.33$ and spread into a smooth band at $t=6.62$ as the order-1 scheme diffuses the edges. The
  tag concentrates where $n_e>1.239$, never on the floor at $1.0$.
- **Not shown**: this footprint is the tagged zone (density > threshold), not the exact geometry of
  the Berger-Rigoutsos rectangles: the binding does not expose the patch boxes, only their count
  (`n_patches()`). The footprint approximates the coverage, it does not draw it pixel-perfect. The
  count stays fixed at 2: neither patch merge/split nor a large patch count is tested.

### `mass_conservation.png`: reflux keeps mass at machine roundoff

![Relative mass drift vs t, AMR vs uniform, below the 1e-9 tolerance](figures/mass_conservation.png)

- **Proves**: both curves stay glued to the machine roundoff floor ($\sim 10^{-15}$), six orders of
  magnitude below the tolerance `TOL_MASS = 1e-9` (dashed line). Measured: AMR `drel_max = 3.06e-15`,
  uniform `drel_max = 6.12e-15`. The assert `drel < 1e-9` (in run.py) passes at every step.
- **Suggested**: the AMR is not less conservative than the uniform despite its coarse/fine interfaces
  re-decomposed every 10 steps: its curve is even slightly lower in places. This is the expected
  signature of a correct reflux; no assert compares the two floors.
- **Not shown**: the scenario without reflux is not shown (it would leave the graph by a staircase
  drift at each regrid). The figure proves that the reflux *present* conserves, not the
  counterfactual.

### `diocotron_amr_hero.gif`: the hero figure of the adc_cpp README, in a local version

The adc_cpp README displays an animation at the top, `docs/anim_romeo_diocotron_amr3.gif`: a single
panel where a horizontal charge band (mode $l=2$) winds up into a cat's-eye, tracked by AMR
refinement frames. `make_hero_gif.py` produces a version of the same type, reproducible locally (same
framing: a single panel, dark background, inferno colormap, title `diocotron AMR : densite n_e`,
mode $l=2$ band that winds up), but where the frames are the solver's fine patches, not a proxy.

![Final state of the mode l=2 diocotron tracked by AMR (GIF cover image)](figures/diocotron_amr_hero_cover.png)

![Diocotron animation on AMR: mode l=2 band wound into a cat's-eye, real AMR patches](figures/diocotron_amr_hero.gif)

- **Proves / visible (solver physics)**: the band is advected by the solver (ExB drift of
  `models.diocotron`, charge Poisson solved by geometric multigrid on `adc.AmrSystem`). The winding
  into two vortices (cat's-eye, Kelvin-Helmholtz instability of the diocotron at mode $l=2$) is the
  code's output, not a scripted animation.
- **Proves / visible (solver frames)**: each cyan rectangle is the exact geometry of a fine patch,
  read by `AmrSystem.patch_rectangles()` (binding `patch_boxes()`). No density proxy, no `scipy`:
  these are the patches the engine actually refined. You see it follow the physics: at the start the
  patches tile the sinusoidal band, then concentrate on the vortex cores and the filament as the
  instability winds up (tag criteria above the floor, `set_refinement(threshold)`; the regrid
  replaces them at each window). The number of patches varies (logged in `provenance.json`, field
  `n_patches_*`).
- **scope (1 level, not 3)**: the Python facade `adc.AmrSystem` refines on a single multi-patch fine
  level (Berger-Rigoutsos): all patches are level 1 (cyan; the code colors by level,
  1=cyan/2=green/3=red, and would be ready if a future one exposed more levels). The hero figure of
  the adc_cpp README was itself produced by the multi-level C++ engine (`advance_amr`, 3 levels) on
  ROMEO (GH200). This GIF reproduces the type of the figure (diocotron tracked by an adaptive AMR),
  with the facade's patches, not the exact 3 levels of the ROMEO run.
- Generated by `python make_hero_gif.py`; provenance in `figures/provenance.json` (fields
  `physique_reelle`, `cadres`, `difference_avec_hero`, `n_patches_*`).

---

## 7. What the invariant does not capture

Mass conserved to $10^{-15}$ and the multi-patch coverage are structural invariants: they say that
the AMR machinery (tag -> regrid -> reflux) is correct and active, not that the resolved physics is
faithful. The following remain outside the validation:

- **Rate fidelity.** The NoSlope + Rusanov scheme is order 1, deliberately dissipative: it smears the
  band edges (visible fig. `patch_map`, t=6.62) and would lower a measured growth rate, just as the
  parent's uniform version underestimates $\gamma_l$ by $-22$ to $-27\%$. Here no rate is measured or
  asserted.
- **Convergence.** A single fine level, $64^2$ base, 2 patches: no resolution sweep, no deep
  hierarchy, no demonstration that the solution converges when you refine further.
- **Multi-block and device.** The `adc_cpp` core validates the AMR brick on device backends (B_z
  regrid GH200) and MPI multi-box in the upstream project; this case only composes native bricks via
  the Python facade on the charge binding (CPU host). No GPU/MPI path is exercised here.

---

## 8. Initial conditions

IC = `band_density` (in initial_conditions.py): horizontal gaussian charge band,
undulated `mode` times along $x$.

$$n_e(x,y)=\text{floor}+\text{amp}\cdot e^{-(y-y_0)^2/\text{width}^2},\qquad
y_0=0.5\,L+\text{disp}\cdot\cos(2\pi\,\text{mode}\,x/L).$$

Case parameters (in run.py): `N=64`, `L=1`, `amp=1`, `width=0.05`, `mode=4`,
`disp=0.02`, `floor=1` (default). Band centered at $y=0.5$, undulated 4 times. Convention `ne[j,i]`
at cell centers (`common/grid.py:meshgrid_xy`). The neutralizing background `n_i0 = ne.mean() = 1.088623`
(measured) ensures the zero mean of the periodic Poisson RHS (section 4.2).

This `band_density` IC is shared with `../diocotron/` (periodic variant) and `custom_scheme`. It
differs from the `ring_density` ring of the published benchmark: `diocotron_amr` is not a
reproduction of `arXiv:2510.11808` (section 7).

---

## 9. Reproduce

```bash
cd /private/tmp/adc_cases-deeptut/diocotron_amr
# the case (asserts, ~0.4 s CPU host, without matplotlib):
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
# the diagnostic figures (replays AMR + uniform, writes figures/*.png + provenance.json):
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py
```
The first element of `PYTHONPATH` brings the `adc` module (C++ binding, ABI suffix
`cpython-312-darwin`); the second brings the `adc_cases` package. Prerequisites: `numpy` (the case),
`matplotlib` (the figures only). No C++ compiler at runtime (`needs = []` in the manifest): nothing
is compiled on the fly, the `_adc.so` binding is already built.

Expected output of the case (real captured values):
```
# n_base=64 regrid_every=10 band_mode=4  n_i0=1.0886
  0     0.1642   2        1.08862269e+00 4.079e-16
  ...
  39    6.6204   2        1.08862269e+00 8.159e-16
# patchs observes : [2]
# masse : init=1.088622692545e+00 final=1.088622692545e+00 drel=8.159e-16
# densite : min=1.000000e+00 max=1.966797e+00
# controle (seuil 1e+30) : patches=1  ecart_sup solution=6.395745e-02
OK diocotron_amr
```
The signs and the order of magnitude are stable; the last digits of `drel` and of `gap` vary with the
BLAS library and the patch summation order (cf. `figures/provenance.json`: AMR `drel_max`
$=3.06\times 10^{-15}$, figures `gap` $=8.68\times 10^{-2}$ on a measurement window different from the
`run.py` assert, $6.40\times 10^{-2}$).

## File map

| File | Role |
|---|---|
| `run.py` | the case: `AmrSystem`, 40-step loop, asserts, control run at an unreachable threshold |
| `make_figures.py` | replays AMR + uniform, writes `figures/*.png` + `figures/provenance.json` |
| `figures/density_compare.png` | final density uniform \| AMR \| difference |
| `figures/patch_map.png` | footprint of tagged cells (3 instants) + `n_patches(t)` |
| `figures/mass_conservation.png` | relative mass drift vs t, AMR vs uniform |
| `figures/provenance.json` | adc_cpp/adc_cases SHA, backend, resolution, measured numbers |
| `../adc_cases/models.py` | `diocotron(B0, alpha, n_i0)` = composition of the 4 bricks |
| `../adc_cases/common/initial_conditions.py` | `band_density(...)`: the perturbed gaussian band |
| `../adc_cases/common/checks.py` | `assert_finite`, `relative_drift` |
| `../diocotron/` | the physical reproduction (rate $\gamma_l$, figures, gif) on a uniform grid |
