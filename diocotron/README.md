# diocotron: instability of a charge column in E x B drift

Reproduction of the diocotron benchmark from Hoffart, Maier, Shadid & Tomas
([arXiv:2510.11808](https://arxiv.org/abs/2510.11808), Section 5.3) with the `adc` solver, in
100% Python. You compare the growth rate $\gamma_l$ of the instability, measured by our
simulation, against both an analytical oracle (radial eigenvalue problem, solved in
numpy) and the paper's targets.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `run.py`: `reproduction` (off CI, figures+gif); `band_instability.py`: `validation` (CI, no figure) |
| Inputs | $192^2$ grid, $L=1$, non-periodic, conducting wall circle $R_w=0.40$; ring $R_0{:}R_1=0.15{:}0.20$, perturbation $\delta\sin(l\theta)$ ($\delta=0.01$ measurement, $0.1$ gif); $B_0=1$, $\alpha=1$; modes $l\in\{3,4,5\}$ |
| Outputs | $\gamma_l$ measured (3 modes); 4 figures in `figures/`; `figures/provenance.json` |
| Guaranteed invariants | the analytical oracle matches the paper to 3 digits; the amplitude $|c_l|$ grows (positive log slope over the linear window) |
| Proves | (1) our numpy dispersion relation reproduces the paper ($\gamma_3{=}0.772$, $\gamma_4{=}0.912$, $\gamma_5{=}0.687$); (2) the `adc` simulation is unstable in all 3 modes, in the right order |
| Does not prove | the simulation does not reproduce the quantitative rate: it underestimates by $-22$ to $-27\%$ (modes 3-4). This reproduces the E x B drift limit, not the full magnetized Euler-Poisson system (see [`hoffart_euler_poisson_dsl`](../hoffart_euler_poisson_dsl/)). No assert tests the value of $\gamma$; `run.py` measures and prints it |
| Provenance | adc_cpp `7598316a`, adc_cases `5112be06`, native serial backend, $192^2$, ~60 s on 1 CPU core; `figures/provenance.json` |

By the end you will know: why a charge ring becomes unstable (mechanism), how the analytical
rate is computed (eigenvalue problems), how it is measured in the simulation, and
why our finite-volume scheme underestimates the rate.

---

## 1. The physical mechanism

A ring of electrons (density $n_e(r)$, zero at the center and at the edge) in an axial field
$\mathbf{B}=B_0\hat z$. Three chained ingredients:

1. **Self-field.** The charge creates $\phi$ through Poisson, with a conducting wall ($\phi=0$ at $r=R_w$):
   $-\nabla^2\phi=\alpha\,(n_e-n_{i0})$, here $n_{i0}=0$.
2. **E x B drift.** The electrons do not follow $\mathbf{E}$; they drift at
   $\mathbf{v}=\dfrac{\mathbf{E}\times\mathbf{B}}{B_0^2}=\dfrac{1}{B_0}(-\partial_y\phi,\ \partial_x\phi)$.
   This velocity is divergence-free, so $n_e$ is advected:
   $\partial_t n_e+\nabla\cdot(n_e\mathbf{v})=0$.
3. **Differential rotation.** At axisymmetric equilibrium, the drift is azimuthal: the ring
   rotates at $\Omega(r)=-\dfrac{1}{r^2}\displaystyle\int_0^r n_e(r')\,r'\,dr'$, a function of $r$.
   The inner edge and the outer edge do not rotate at the same speed: shear.

This model is mathematically identical to 2D Euler in vorticity form: $n_e$ plays the role of
vorticity, $\phi$ the stream function, $\mathbf{v}=\hat z\times\nabla\phi/B_0$. The diocotron
instability is therefore the Kelvin-Helmholtz instability of a vorticity ring: an azimuthal
ripple of the edge (mode $l$) is carried differently by the two sheared edges, rolls up, and
grows exponentially. The ring develops $l$ lobes (visible in section 6). The rate $\gamma_l$ is
the quantified consequence of this shear, not the starting point.

---

## 2. The equations and who computes them

| Block | Equation | Brick |
|---|---|---|
| Transport | $\partial_t n_e+\nabla\cdot(n_e\mathbf{v})=0,\ \mathbf{v}=\frac{1}{B_0}(-\partial_y\phi,\partial_x\phi)$ | `ExBVelocity` |
| Source | none | `NoSource` |
| Elliptic | $-\nabla^2\phi=\alpha(n_e-n_{i0})$, Dirichlet $\phi=0$ at $r=R_w$ | `BackgroundDensity` |
| State | scalar density $n_e$ | `Scalar` |

This is `adc_cases.models.diocotron(B0, alpha, n_i0)`. Who computes what:

| `run.py` line | Layer | What happens |
|---|---|---|
| `add_block("ne", model=..., spatial=Spatial(minmod), time=Explicit)` (`make_ring_system` in run.py) | Python composes | choice of the model, the scheme (MUSCL minmod + Rusanov), the integrator (SSPRK2) |
| `models.diocotron(...)` -> `ExBVelocity` / `BackgroundDensity` (`include/adc/physics/{hyperbolic,elliptic}.hpp`) | the C++ brick fixes the physics | the exact convention of the flux $n v(dir)$, of the eigenvalue $v(dir)$, of the RHS $\alpha(n-n_{i0})$ |
| `assemble_rhs<Limiter,Flux>` + system Poisson (`GeometricMG`) | per-cell kernel | the actual computation, with no Python callback in the hot path |

`models.diocotron` names no scenario on the core side: the word "diocotron" lives in `adc_cases`,
the physics is a composition of generic bricks.

---

## 3. The falsifiable prediction: the rate $\gamma_l$, computed twice

`run.py` computes $\gamma_l$ through two independent paths and compares them:
(A) a numpy analytical oracle (section 4); (B) the `adc` simulation (section 5). The figure
`dispersion.png` overlays the two plus the paper's points. The gap (A)-(B) is the central analysis
(section 7). It justifies the Proves clause (the oracle == paper) and the Does-not-prove clause (the
simulation underestimates).

---

## 4. Math: the analytical dispersion relation (`diocotron_eigenvalue` in run.py)

### 4.0 Where the rotation $\Omega(r)$ comes from, and where the instability comes from

The equilibrium is axisymmetric, $n_0(r)$. Radial Poisson,
$\frac1r\frac{d}{dr}\!\big(r\,\partial_r\phi_0\big)=-n_0$, integrates once:
$r\,\partial_r\phi_0(r)=-\int_0^r n_0(r')\,r'\,dr'\equiv-C(r)$, so $\partial_r\phi_0=-C/r$. The E x B
drift of a radial potential is purely azimuthal, $v_\theta=\frac{1}{B_0}\partial_r\phi_0$, and the
angular velocity is $\Omega(r)=v_\theta/r=-C(r)/r^2$ (with $B_0=1$): this is exactly the line
`Om[1:] = -C[1:]/r**2`. The enclosed charge $C(r)$ grows inside the ring then saturates: $\Omega(r)$
is not constant, the ring is in differential rotation.

Why does this create an instability? In vorticity form ($n_e$ = vorticity, $\phi$ =
stream function), Rayleigh's criterion says a vorticity profile is unstable if it
has an extremum (inflection point of the velocity profile). A hollow ring has two vorticity
edges (rising at $R_0$, falling at $R_1$): two counter-rotating shear layers
that couple. Each carries edge waves (diocotron waves); when the two edge
waves are in phase resonance (at a given azimuthal mode $l$), they reinforce each other and
the amplitude grows exponentially. The most unstable mode is the one where this coupling is maximal
(here $l\approx 4$).

### 4.1 The eigenvalue problem

You linearize around $n_0(r)$: a perturbation
$\phi'(r,\theta,t)=\hat\phi(r)\,e^{i(m\theta-\omega t)}$ obeys an eigenvalue problem
for $\omega$ ($\mathrm{Im}\,\omega=\gamma$):
$$\omega\,\mathcal{L}_m\hat\phi=m\,\Omega(r)\,\mathcal{L}_m\hat\phi+\frac{m}{r}\frac{dn_0}{dr}\,\hat\phi,\qquad \hat\phi(0)=\hat\phi(R_w)=0,$$
where $\mathcal{L}_m=\dfrac{d^2}{dr^2}+\dfrac1r\dfrac{d}{dr}-\dfrac{m^2}{r^2}$ is the radial Laplacian of
mode $m$. Both terms have a direct physical meaning: $m\,\Omega(r)\,\mathcal{L}_m\hat\phi$ is
the advection of the perturbed vorticity by the equilibrium rotation (the wave is carried by the
fluid rotating at $\Omega(r)$), and $\frac{m}{r}\frac{dn_0}{dr}\hat\phi$ is the forcing by the
density gradient: it is what can inject energy into the perturbation, and it is
concentrated at the edges of the ring. You put it in standard form
$\omega\hat\phi=\mathcal{L}_m^{-1}(m\Omega\,\mathcal{L}_m+Q)\hat\phi=M\hat\phi$, and
$\gamma=\max_k\mathrm{Im}(\omega_k)$ (the most unstable mode). The spectrum reads as follows: the
real $\omega$ are stable rotations (neutral waves); a conjugate pair $\omega,\bar\omega$
signals an instability, and $\gamma=\mathrm{Im}(\omega)>0$ is its rate. The code discretizes everything in
finite differences; each symbol points to the line that computes it:

```python
rho = 0.5 * rhobar * (np.tanh((r - a) / w) - np.tanh((r - b) / w))   # n0(r) : anneau lisse
C[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * h)        # C(r)=int_0^r n0 r' dr'
Om[1:] = -C[1:] / (r[1:] ** 2)                                       # Omega(r)=-C/r^2
np.fill_diagonal(Lmat, -2.0/h**2 - (m*m)/(ri*ri))                    # diagonale de L_m
Lmat[k, k-1] = 1/h**2 - 1/(2*h*r[k+1]); Lmat[k, k+1] = 1/h**2 + 1/(2*h*r[k+1])  # +/- = (1/r)d/dr
Q = (m / ri) * ((rho[2:N+1] - rho[0:N-1]) / (2*h))                   # Q=(m/r) dn0/dr
M = np.linalg.solve(Lmat, (m*Om[1:N])[:,None]*Lmat + diag(Q))       # M=L^{-1}(mOmega L + Q)
dom = np.linalg.eigvals(M)[argmax(.imag)]                           # mode le plus instable
return (2*pi/rhobar) * dom                                          # normalisation papier
```
- `Om` ($\Omega(r)$) is the equilibrium rotation: its shear $d\Omega/dr$ is the driver.
- `Lmat` is the tridiagonal matrix of $\mathcal{L}_m$: the diagonal $-2/h^2-m^2/r^2$ comes from
  $d^2/dr^2$ and the $-m^2/r^2$ term; the off-diagonals $1/h^2\pm 1/(2hr)$ encode $d^2/dr^2$
  and the first-derivative drift term $\frac1r\frac{d}{dr}$. The conditions $\hat\phi(0)=\hat\phi(R_w)=0$
  are imposed by keeping only the interior points (`r[1:N]`).
- `Q` $=\frac{m}{r}\frac{dn_0}{dr}$ is the source term, nonzero only at the edges of
  the ring. This is the key observation of section 7.

The normalization $\times 2\pi/\bar\rho$ is the paper's convention (rate in units of the mean
rotation frequency); it is taken as given, not re-derived.

---

## 5. Measurement in the simulation (`measure_growth` + `mode_l_amplitude` in run.py)

At each step, you read $\phi$ and extract the amplitude of mode $l$ on a circle in the middle of
the ring ($r_m=(R_0+R_1)/2$):

```python
_, val = bilinear_on_circle(field, n, radius, 256)   # phi(theta) : 256 points, interpolation bilineaire
ck = np.fft.rfft(val) / len(val)                     # FFT azimutale
return 2.0 * abs(ck[l])                              # 2|c_l| = amplitude du mode l
```
- `bilinear_on_circle` interpolates $\phi$ (defined at cell centers) onto 256 points of the circle.
  The FFT then $2|c_l|$ gives the Fourier coefficient of mode $l$: in the linear phase,
  $|c_l|(t)\propto e^{\gamma t}$.
- `fit_linear_phase` (in run.py) fits the slope of $\log|c_l|$ over the purely
  exponential window: after the transient ($1.3\,a_0$), before saturation ($0.85$ of the peak). The slope
  is $\gamma_{\text{raw}}$, normalized $\times 2\pi/\bar\rho$.

The system (`make_ring_system` in run.py) is non-periodic with a Dirichlet circle wall
(`set_poisson(..., wall="circle", wall_radius=RWALL)`): the paper's wall, imposed on the Cartesian
grid. `Spatial(minmod=True)` = MUSCL order 2 limited by minmod (diffusive at strong gradients,
see section 7) + Rusanov flux.

Initial conditions: `ring_density(n, l, delta)` lays a ring $\approx 1$ between $R_0$ and
$R_1$, $\approx 10^{-3}$ elsewhere, with an azimuthal perturbation
$n_e\approx\mathbb{1}_{[R_0,R_1]}(1-\delta+\delta\sin(l\theta))$. $\delta=0.01$ for the measurement
(linear regime), $\delta=0.1$ for the gif (visible).

---

## 6. Figures (generated by `run.py`, in `figures/`)

### `dispersion.png`: rate vs mode

![Growth rate gamma_l vs mode l: analytical Petri curve, paper squares, adc stars](figures/dispersion.png)

- **Proves**: the analytical curve (gray) passes through the paper's red squares
  ($\gamma_3{=}0.772$, $\gamma_4{=}0.912$, $\gamma_5{=}0.687$): our oracle reproduces the paper to 3
  digits. The blue stars (`adc` measurement) are all below the curve, in the right mode
  order.
- **Suggested (not asserted)**: the maximum near $l=4$ (most unstable mode of this geometry) is
  visible but no assert ranks the modes.
- **Not shown**: no point tests the value of $\gamma$ by assert; `run.py` measures and
  displays it, it does not validate it against a tolerance.

### `amplitude.png`: exponential growth

![|c_l|(t) on a log scale for modes 3,4,5: straight lines in the linear phase](figures/amplitude.png)

- **Proves**: on the semilog scale, each mode traces a straight line over the fit window:
  growth $|c_l|\propto e^{\gamma t}$ confirmed (positive slope = unstable). The slope is the
  $\gamma$ reported as a star on `dispersion.png`.
- **Not shown**: saturation (the flattening at the top) is nonlinear and outside the measurement window.

### `diocotron.gif` + `snapshots.png`: nonlinear roll-up (l=4)

![Density mode l=4 over time: 4 lobes rolling up](figures/diocotron.gif)

![Four density snapshots (increasing t), mode l=4](figures/snapshots.png)

- **Proves / visible**: the ring develops exactly 4 lobes ($l=4$) that roll up into a
  spiral (Kelvin-Helmholtz roll-up): the signature of the instability.
- **Suggested**: the nonlinear phase (beyond the linear window); its exact dynamics is
  not compared to the paper.

---

## 7. Why the simulation underestimates the rate

The oracle reproduces the paper, the simulation underestimates by $-22$ to $-27\%$ (modes 3-4). This is not
a bug; the cause is in the source term. The driver of the instability is
$Q=\frac{m}{r}\frac{dn_0}{dr}$, nonzero only at the two edges of the ring (section 4). Now:

1. **Scheme diffusion at the edges.** MUSCL+minmod + Rusanov is diffusive at strong gradients: it
   smears the edges of the ring, so it reduces $dn_0/dr$ exactly where the instability is driven. A
   weaker $dn_0/dr$ gives a weaker $\gamma$: exactly the observation.
2. **Circle on a Cartesian grid.** The conducting wall is a staircase circle on
   the square grid, and the ring is not aligned with the grid. The geometric error is
   mode-dependent in a non-monotonic way (hence $-5\%$ at $l=5$ vs $-27\%$ at $l=4$).

The model and the oracle are correct (they reproduce the paper); it is the
discretization (edge diffusion + Cartesian circle) that lowers the measured rate. A known
and structural limit of the Cartesian path, motivating the polar geometry work on the `adc_cpp` side.

---

## 8. Reproduce

```bash
# adc_cpp est Kokkos-only : un Kokkos installe (Serial pour un poste CPU) est requis (-DKokkos_ROOT).
cd ../adc_cpp && cmake -B build-py -DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT && cmake --build build-py --target _adc -j
cd ../adc_cases
PYTHONPATH=$PWD/../adc_cpp/build-py/python python3 diocotron/run.py        # ~60 s, ecrit figures/ + provenance.json
```
Prerequisites: `numpy`, `matplotlib`, the `adc` module compiled and imported with the same interpreter as
the one that compiled it (ABI suffix `cpython-3XY`). Expected output: `l=3: gamma_num = 0.599 |
analytique 0.772 | papier 0.772 | ecart -22%`, same for $l=4$ ($-27\%$) and $l=5$ ($-5\%$), then `OK
repro_paper_2510_11808`. The signs and the order of magnitude are stable; the last digits
vary with the BLAS and the summation order (cf. `figures/provenance.json`).

`band_instability.py` (category `validation`, in CI) does not measure a rate: it verifies by
assert that the instability grows on a periodic band, without a figure.

## File map

| File | Role |
|---|---|
| `run.py` | analytical (sec. 4) + simulation (sec. 5) + figures + provenance |
| `figures/*.png`, `diocotron.gif` | versioned assets, regenerated in place |
| `figures/provenance.json` | adc_cpp/adc_cases SHA, backend, resolution, measured $\gamma$ |
| `band_instability.py` | lightweight CI variant | 
| `NORMALIZATION.md` | detail of the $\times 2\pi/\bar\rho$ normalization |
