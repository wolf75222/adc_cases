# T2: normalization audit of the `system-schur` path

This document closes the question left open by `RESULTS_SYSTEM_SCHUR.md` (sections 7ter / conclusion):
the -95% deficit of the hoffart `system-schur` run (`gamma_raw ~ 0.032`, paper windows,
`alpha = omega = 1e12`) does not come from the Cartesian geometry; it decomposes entirely into
dimensional factors derived *ahead of time* (not fitted after the fact), measured by
`diag/diag_normalization_audit.py`.

One-line result: the l=3 deficit (x24.7 between 0.0312 and 0.772) is the product of three factors:

    deficit = (window 3.20x) x (T_d = 2 pi = 6.28x) x (cart vs polar grid residual 1.23x) = 24.7x

The first two are metrology/normalization (recoverable); only the third (~20%) is
a genuine physical grid difference. So this is not pure metrology (the 2 pi is real but a
~20% residual remains), nor a fundamental geometric limitation: the Cartesian reproduction
is reachable.

---

## 1. The dimensional key: `alpha/omega = 1`, the `1e12` cancels

`model.py` sets (paper params, `rho_max = 1`):

    alpha = beta^2 / rho_max = 1e12          (Poisson charge: -Delta phi = alpha rho)
    omega = beta^2           = 1e12          (= |Omega| = omega_c, the B_z field)

The drift velocity of the full run (`build_uniform` -> `drift_velocity_from_potential`) is
`v = grad(phi)/omega` with `-Delta phi = alpha rho`. Setting `phi = alpha phi~` (so
`-Delta phi~ = rho`):

    v = (alpha/omega) grad(phi~) ,   alpha/omega = 1/rho_max = 1
    => v == grad(phi~) == EXACTLY the NORMALIZED ExB drift (alpha = 1, B = 1).

The `1e12` of `alpha` and the `1e12` of `omega` cancel in the transport. The full run
(`alpha = omega = 1e12`) and the normalized reduced ExB (`B0 = 1`, `charge = 1`, the validated path of
`diag_polar_omega.py`) advect `rho` with the same velocity field, in the same simulation-time
units. So `gamma_raw` is directly comparable between the two, and the only possible
difference between `0.032` (full run) and `~0.10` (reduced cart, RESULTS section 7ter) is the
fit window, not a physical scale.

> Cross-check: the normalized reduced ExB, fitted over the l=3 paper window `[0.40,0.70]`
> *applied in simulation time*, gives `gamma_raw = 0.0312`, within 3% of the full-run measurement
> (RESULTS section 1, n=128: `0.0321`). The full == reduced equivalence (RESULTS section 7) is thus reconfirmed here by
> the raw numbers themselves.

## 2. The diocotron scales

With `rho_max = 1`:

| quantity | definition | value |
|---|---|---|
| `omega_c = \|Omega\|` | `beta^2` | `1e12` (cyclotron, FAST scale) |
| `omega_d` | `rho_max * alpha / \|Omega\|` | **`1`** (diocotron/drift, SLOW scale) |
| `T_d` | `2 pi / omega_d` | **`2 pi ~ 6.283`** (diocotron period) |
| `alpha/omega` | only dimensionless combination | **`1`** (`= 1/rho_max`) |

`omega_d = rho_max (beta^2/rho_max) / beta^2 = 1`: the `beta^2` cancels, the slow dynamics live in
O(1) units. `T_d = 2 pi` is the `2 pi` factor of the deposition (`NORMALIZATION.md`,
`diag_polar_omega.py:35`): it is the diocotron period, not a fudge.

## 3. The scaling candidates all collapse onto `x 2 pi`

The four requested candidates (T2), applied to the established `gamma_raw` (l=4, window `[3,12]`,
n=128, `gamma_raw = 0.1135`):

| candidate | formula | dimensional justification | value |
|---|---|---|---|
| c1 | `gamma_raw * 2 pi` | sim-time -> paper-time conversion via `T_d` | **0.7132** |
| c2 | `gamma_raw * 2 pi * (alpha/omega)` | `alpha/omega = 1` -> **identical to c1** | 0.7132 |
| c3 | `gamma_raw / omega_d` | `omega_d = 1` -> **no-op** | 0.1135 |
| c4 | `gamma_raw * T_d` | `T_d = 2 pi` -> **identical to c1** | 0.7132 |
| -- | l=4 paper target | -- | 0.9110 |

Conclusion section 3: every dimensionally honest candidate collapses onto `gamma_raw * 2 pi`
(because `alpha/omega = 1`, `omega_d = 1`, `T_d = 2 pi`). There is no extra ~3 factor at the
dimensional level. `c1` gives `0.713`, i.e. ~22% below the paper (0.911), exactly the cart-vs-polar
grid residual of RESULTS section 7ter (cart x2pi `0.72` vs polar `0.90`). The `x 2 pi` is therefore the
only legitimate normalization factor, and it is not the lock: applied to the established `gamma_raw`
it reproduces to ~20%.

## 4. The "~3x residual" is the fit window (measured)

`run.py:fit_growth` masks the simulation time directly with `PAPER_FIT_WINDOWS`
(`[0.40,0.70]`, ...). But `paper_time = T_d x sim_time`: the paper window applied in sim time
falls in the transient (rate still ramping, cf. RESULTS section 3: local rate 0.03->0.11 over
`t in [0.5, 2.5]`), not in the established exponential. Measurement (`diag_normalization_audit.py`, n=128,
same run, two windows):

| l | paper window (sim) | `gamma_raw` (paper) | established window `[3,12]` | `gamma_raw` (established) | **established/paper ratio** |
|---|---|---|---|---|---|
| 3 | `[0.40,0.70]` | **0.0312** | `[3.0,12.0]` | **0.0998** | **3.20** |
| 4 | `[0.60,0.75]` | 0.0943 | `[3.0,12.0]` | 0.1135 | 1.20 |
| 5 | `[1.15,1.35]` | 0.1056 | `[3.0,12.0]` | 0.1137 | 1.08 |

The 3.20 ratio (l=3) is the "~3x residual beyond the 2 pi". It is a window effect, not a
missing scale: the l=3 paper window is the earliest (`[0.40,0.70]`), so the most deeply
buried in the transient -> the largest factor. For l=4 / l=5 the paper windows are later
(`[0.60,0.75]`, `[1.15,1.35]`) -> the window factor drops to 1.20 / 1.08. This is why the deficit
was maximal at l=3 (-95.5%) and smaller at l=5 (-83%).

## 5. Full deficit decomposition (l=3): it closes exactly

| factor | from -> to | value | nature |
|---|---|---|---|
| fit window | `gamma_raw` paper `0.0312` -> established `0.0998` | **3.20x** | metrology (run.py fits the transient) |
| `T_d = 2 pi` | `0.0998` -> `0.627` | **6.28x** | metrology (diocotron period) |
| cart vs polar grid | `0.627` -> paper `0.772` | **1.23x (~20%)** | physics (only NON-metrological residual) |
| **product** | `0.0312` -> `0.772` | **24.7x** | == observed -95.5% deficit |

`3.20 x 6.28 x 1.23 = 24.7`: the decomposition closes the measured l=3 deficit exactly
(`0.772 / 0.0312 = 24.7`).

## 6. T2 verdict

- The deficit does not come from the Cartesian geometry. The `T_d = 2 pi` and the window factor are
  recoverable normalization/metrology (~20x of the ~24.7x). Cartesian reproduction is reachable.
- It is not pure metrology either: after the two `2 pi` factors (time + window),
  a ~20% residual remains (cart grid `0.72` vs polar `0.90` x2pi) which is a genuine physical
  difference of azimuthal discretization, not a cosmetic factor.
- No ~3 dimensional factor exists: `alpha/omega = 1`, `omega_d = 1`, `T_d = 2 pi`, all the
  candidates collapse onto `x 2 pi`. The "3x residual" was the fit window, now
  quantified (ratio 3.20 for l=3, section 4).

### Actionable implication for `run.py` -> **DONE (T3)**
The `system-schur` path measurement fitted the paper window in simulation time, hence in the
transient. T3 (June 2026) fixes this in the code: `run.py:fit_growth` now fits the
mapped paper window (`sim_window = 2 pi/rhobar x paper_window`) and `results.py` reports
both `gamma_raw_sim` and `gamma_paper_units = gamma_raw_sim x 2 pi/rhobar` (the raw value is kept for
reproducibility). Helpers: `paper_to_sim_time_window`, `gamma_to_paper_units`.

### 7. Direct verification on the full system-schur (not the reduced proxy)
Sections 4-5 use the reduced ExB as a proxy (justified by `alpha/omega=1`). T3 measures the actual full
system-schur (Strang ssprk3 + CondensedSchur, drift-seeded) with the mapped windows (n=96,
t_end=10):

| l | mapped sim window | `gamma_raw_sim` | `gamma_paper_units` (x2pi) | paper | error |
|---|---|---|---|---|---|
| 3 | [2.513,4.398] | 0.1117 | **0.702** | 0.772 | **-9.1%** |
| 4 | [3.770,4.712] | 0.1423 | **0.894** | 0.911 | **-1.9%** |
| 5 | [7.226,8.482] | 0.1087 | **0.683** | 0.683 | **+0.04%** |

The full reproduces the paper to -9 / -2 / +0% with the mapped windows (better than the established window
`[3,9]`: l=5 goes from +13% to +0.04%, its late window capturing the same phase as the paper). The full
tracks the reduced to ~2% in the established window (the proxy was valid). Caveats (adversarial review): the
2 pi is exact/mode-independent (Petri <0.5%); the ~0-9% residual is grid/resolution(n=96)/window roll-off
(no scale-free plateau, WENO5 smoothing != saturation); l=5 is sensitive to the window
(+/-27-29%), so its +0.04% is partly fortuitous, lead with l=3/l=4. Detail: `RESULTS_SYSTEM_SCHUR.md`
section 9.

## Reproduce

```bash
PYTHONPATH=<adc_cpp>/build-master/python \
    python hoffart_euler_poisson_dsl/diag/diag_normalization_audit.py 128
```

Output: the dimensional scales, the paper-window vs established table per mode (the ratio is the
window factor) and the collapse of the 4 candidates onto `x 2 pi`. See also `NORMALIZATION.md` (validated
polar path) and `RESULTS_SYSTEM_SCHUR.md` section 7ter (geometry reversal).
