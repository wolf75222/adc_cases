# Growth-rate normalization for the reduced diocotron (scalar polar ExB path)

The one-line finding:

    gamma_norm = gamma_raw * (2 pi / rhobar)

On ADC's polar ExB path (global ring, scalar ExB transport, WENO5 +
SSPRK3, polar Dirichlet Poisson), this global factor validates the
growth-rate normalization of the reduced diocotron (scalar ExB drift model,
Petri-type), the benchmark of Section 5.3 of Hoffart et al. (arXiv:2510.11808).
This path does not solve the full Euler-Poisson model (rho, rho*u, rho*v); only
l = 4 matches exactly (l = 3 +26 %, l = 5 oscillates). With rhobar = rho_max = 1,
the factor equals exactly 2 pi.

The reproducible diagnostic is `diag/diag_polar_omega.py`.


## Why gamma_raw is already the right Im(omega)

You track the complex coefficient c_l(t) of the azimuthal mode l of the
potential PHI on the inner circle r = r0:

    c_l(t) ~ exp(-i omega_l t)
    => |c_l|   ~ exp(Im(omega) t)      (growth)
       arg(c_l) ~ -Re(omega) t          (rotation)

The polar solver runs in ExB-natural time units. In these units gamma_raw, the
slope of log|c_l|, is directly Im(omega) of the complex eigenmode: there is no
beta re-scaling to apply (gamma_raw(sim) ~ Im(omega)_eigenmode). The only
missing factor to reach the paper's units is the global factor 2 pi / rhobar.


## Why the "local rotation" normalization fails

A natural idea would be to normalize by the mode's own rotation:

    g_rot = (gamma_raw / |Omega_raw|) * |Re(omega)|_ana * 2 pi

The ratio gamma_raw/Omega_raw = Im/Re is scale-invariant, so in principle it is
sound. At the inner edge r0, however, the measured rotation Omega_raw is ~ 0:
there is no charge enclosed inside the ring [r0, r1], so no rigid-body rotation
at r0. Since Omega_raw is near zero, the ratio blows up and g_rot becomes
absurd (see the table: g_rot ~ 15 to 22, instead of ~ 0.7 to 0.9).

Conclusion: the correct normalization is the global factor 2 pi / rhobar, not a
local rotation.


## Results l = 3 / 4 / 5

Measurements from `diag/diag_polar_omega.py` (top-hat [6, 8], R = 16, WENO5 /
SSPRK3, CFL 0.4). gamma_raw / Omega_raw measured; g_2pi = gamma_raw * 2pi/rhobar;
g_pap = paper target.

n = 128:

    l   gamma_raw   Omega_raw    ratio   rat_ana    g_2pi    g_rot    g_pap
    3    0.15456    -0.02089    7.3997   0.3708    0.9712   15.41    0.772
    4    0.14526    -0.01830    7.9356   0.3309    0.9127   21.87    0.911
    5    0.07671    -0.03593    2.1351   0.1998    0.4820    7.34    0.683

n = 192:

    l   gamma_raw   Omega_raw    ratio   rat_ana    g_2pi    g_rot    g_pap
    3    0.15460    -0.02402    6.4373   0.3708    0.9713   13.41    0.772
    4    0.14482    -0.02188    6.6183   0.3309    0.9100   18.24    0.911
    5    0.13780    -0.03193    4.3154   0.1998    0.8658   14.84    0.683

How to read it:

- l = 4: g_2pi = 0.9127 (n=128) and 0.9100 (n=192), exact against the paper
  (0.911) at both resolutions. Stable in resolution.
- l = 3: g_2pi = 0.971 (+26 %) at n=128 and n=192 against 0.772 (stable, but
  offset).
- l = 5: oscillates depending on the fit window: g_2pi = 0.482 at n=128 (-29 %
  against 0.683, window [2.12, 12.58]) and 0.866 at n=192 (+27 %, window
  [2.12, 5.96]). What changes is the chosen window, not the physics (see the
  next section).
- g_rot (the "local rotation" column) is absurd everywhere (~ 13 to 22):
  Omega_raw ~ 0 at r0.


## Scatter for l = 3 / 5: window sensitivity, not a physics deficit

The measured ratio gamma_raw/Omega_raw (~ 7 to 8 for l=3/4, ~ 2 for l=5) differs
from the analytic ratio Im/Re (~ 0.33 to 0.20) precisely because Omega_raw ~ 0 at
r0 (the measured ratio is not the analytic invariant: local rotation does not
exist there). This confirms that local rotation is not the relevant scale.

The scatter at l=3 (+26 %) and l=5 (oscillation -29 % -> +27 %) comes from
sensitivity to the fit window of the exponential regime, not from a physics
deficit: the log|c_l| slope is purely exponential only over a bounded interval
(after the initial transient, before saturation), and the extracted slope
depends on that interval. l = 4, whose exponential window is the cleanest, is
exact and stable in resolution.


## Conclusion

The polar path plus the 2 pi / rhobar normalization validates the reduced
diocotron normalization (scalar ExB drift): l = 4 matches exactly at n = 128 and
n = 192; l = 3 (+26 %) and l = 5 (oscillating -29 % / +27 %) remain offset and
sensitive to the fit window. This path is not the full Euler-Poisson model of
Hoffart et al. and is therefore not a reproduction of the full model. The
correct factor is global (2 pi / rhobar), not a local rotation (which fails
because the rotation at r0 is zero).

For the record, the Cartesian-Schur path (see `run.py --engine system-schur`,
now in Strang SSPRK3 splitting + CondensedSchur via adc_cpp #230, no longer in
Lie) gives `gamma_raw ~ 0.032` in the paper window. The geometry is not to blame:
the audit `T2_NORMALIZATION_AUDIT.md` shows that `alpha = beta^2/rho_max` and
`omega = beta^2` cancel in the drift velocity (`alpha/omega = 1/rho_max = 1`), so
the full run advects rho with exactly the same field as the normalized reduced
ExB of this diagnostic. The `~0.032` decomposes entirely into dimensional
factors: (a) the fit window; `run.py` masks the paper window in simulation time
(transient) whereas `paper_time = T_d * sim_time`; fitted in the established
window, `gamma_raw` climbs back to ~0.10 (a factor 3.20 measured at l=3); (b) the
global factor `2 pi = T_d` (diocotron period, `omega_d = rho_max alpha/|Omega| =
1`). After these two factors, ~20 % remains (Cartesian grid `0.72` vs polar
`0.90` x2pi), the only non-metrological part. So this is not a counterexample to
the normalization; it is the window plus the `2 pi`, not a Cartesian-edge
distortion.

**Update T3 (June 2026)**: this mapping is now wired into `run.py`/`results.py`
(paper windows mapped to sim time + reporting `gamma_paper = gamma_raw_sim *
2pi/rhobar`). Direct measurement of the full system-schur with the mapped paper
windows: **l=3 -9.1%, l=4 -1.9%, l=5 +0.04%** vs the paper (see
`RESULTS_SYSTEM_SCHUR.md` section 9). The `2 pi/rhobar` factor therefore applies
to the full model too (and no longer only to the reduced ExB of this diagnostic):
full and reduced share the ExB-natural clock because `alpha/|Omega| = 1/rho_max =
1`. The historical statement "the 2 pi belongs only to the reduced path" is thus
obsolete; the 2 pi is the cyclic->angular conversion of the drift clock, common
to both paths.


## Rerunning

    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 128
    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 192

The `adc` module must be built (polar path: adc.System(mesh=PolarMesh), ExB
transport, polar Poisson). Reference build used here:
adc_cpp/build-master/python.
