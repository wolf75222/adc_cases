# Petri/Davidson analytic diocotron eigenvalue (target confirmation)

This diagnostic confirms the targets of Hoffart et al. (arXiv:2510.11808, Sec 5.3)
independently, using the linear diocotron theory of the hollow column (reference [13] of
the paper). It derives, not fits, the analytic growth rate gamma_l of the azimuthal
mode l for a hollow electron column in a top-hat profile, with uniform density rho_max on
the ring [r0, r1] = [6, 8], inside a conducting wall of radius R = 16, and shows

    gamma_3 ~ 0.772 ,  gamma_4 ~ 0.911 ,  gamma_5 ~ 0.683

in units omega_d = 1, with no factor of 2 pi applied after the fact.

Reproducible diagnostic: `petri_eigenvalue.py` (numpy only, no dependency on the adc
engine).

Result (`python hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py`):

    l   gamma_l (Im)   papier   abs-err   rel%  |  Re(ExB)   RE_ANA   rel%  |  Im/Re  rat_ana
    3      0.77297    0.7720   0.00097  0.126%  |  0.33134  0.33144 0.031% | 0.3713  0.3708
    4      0.91182    0.9110   0.00082  0.090%  |  0.43838  0.43859 0.048% | 0.3310  0.3309
    5      0.68338    0.6830   0.00038  0.055%  |  0.54711  0.54747 0.067% | 0.1988  0.1998

gamma_l (column `Im`) is the raw max Im(omega) of the eigenvalue problem, in units
omega_d, with no factor applied afterward: it matches the targets to < 0.13 %, well inside
the margin the paper reports (+/- 0.024 on eq (5.1) / Fig 5.4).


## Physical model

Guiding-center limit (ExB drift), axisymmetric equilibrium. The paper solves

    -Delta phi0 = alpha rho0 ,   v0 = -(grad phi0 x Omega) / |Omega|^2 .

Radial Poisson gives `r phi0'(r) = -alpha M(r)` with `M(r) = int_0^r rho0(s) s ds` the
enclosed charge. The azimuthal drift velocity is `v0_theta = -(1/|Omega|) phi0'`, hence
the equilibrium drift angular frequency

    omega_E(r) = v0_theta / r = (alpha / |Omega|) M(r) / r^2 .

For the top-hat `rho0 = rho_max` on `[r0, r1]` (0 elsewhere):

    r < r0      : omega_E = 0
    r0 < r < r1 : omega_E(r) = (Wd / 2) (1 - r0^2 / r^2)
    r > r1      : omega_E(r) = (Wd / 2) (r1^2 - r0^2) / r^2

where `Wd` is the drift angular scale (see the 2 pi block below).

Two surface waves, edge displacements `eta_in` (at r0) and `eta_out` (at r1), of the form
`exp(i l theta - i omega t)`, couple through the perturbed potential (harmonic l, Laplace
in each ring, regular at the center, Dirichlet `phi1(R) = 0`). The edge kinematic
condition and the potential matching give a 2 x 2 eigenvalue problem whose geometric self-
and cross-coupling coefficients (Dirichlet wall at R) are the standard hollow-column form
(Davidson):

    s_in  = (1 / 2l) (1 - (r0/R)^{2l})              inner self-coupling
    s_out = (1 / 2l) (1 - (r1/R)^{2l})              outer self-coupling
    s_mut = (1 / 2l) (r0/r1)^l (1 - (r1/R)^{2l})    inner <-> outer cross-coupling

The matrix (displacements `[eta_in, eta_out]`) is

    M = [[ l omega_E(r0) + l Wd s_in ,        - l Wd s_mut          ],
         [        l Wd s_mut         ,  l omega_E(r1) - l Wd s_out   ]] .

The density-jump signs, inner edge `0 -> rho_max` (+1) and outer edge `rho_max -> 0` (-1),
make the two cross-couplings opposite in sign. This is the Kelvin-Helmholtz/Rayleigh
mechanism: it produces the complex-conjugate eigenvalue pair, hence the instability. The
growth rate is `gamma_l = max Im(omega)`.


## Cross-checks (the MODE is right, not just Im)

Two time-unit-independent quantities confirm that the matrix solves the right complex
eigenmode, not a coincidental number:

- **Re(omega) in the ExB-natural unit** recovers `RE_ANA = {3: 0.33144, 4: 0.43859,
  5: 0.54747}` (the mode's own rotation, published in `../NORMALIZATION.md` /
  `diag_polar_omega.py`) to < 0.07 %.
- **The Im/Re ratio**, scale-invariant (independent of any time unit), recovers
  `RATIO_ANA = {3: 0.3708, 4: 0.3309, 5: 0.1998}` to < 0.5 %.

Since the ratio is right and the real part is right, the imaginary part is right too: this
is the correct eigenmode, not a fit.


## Where the 2 pi is (reduced ExB path <-> full-model omega_d relation)

The paper (lines 313-317) defines the cyclic diocotron frequency

    omega_d := rho_max * alpha / |Omega| = 1 ,   period T_d := 1 / omega_d = 1

(and `tf = 10 = 10 T_d`). `omega_d` is a cyclic frequency: one full revolution (2 pi
radians) per period `T_d = 1`. The corresponding angular scale (radians per time unit),
the one the dispersion relation expects via `omega_E`, is therefore

    Wd := 2 pi * omega_d        (= 2 pi for omega_d = 1).

Setting `Wd = 2 pi omega_d` in the matrix makes raw `Im(omega)` already gamma_l in units
omega_d: nothing is multiplied after the fact. The 2 pi is the cyclic -> angular
conversion, internal to the problem.

On the scalar reduced-ExB path (`diag_polar_omega.py`), the polar solver runs on an
ExB-natural clock where the equilibrium drift counts revolutions in turns (1 turn = 2 pi
radians). The measured slope `gamma_raw = slope of log|c_l|` is the growth rate per
"turn-time". To convert to the omega_d unit (where `T_d = 1/omega_d = 1` denotes one
turn), you multiply by the number of ExB-natural units in one `T_d`, namely 2 pi (the
angle swept in one `T_d` is 2 pi). Hence the global normalization of the reduced path

    gamma_norm = gamma_raw * (2 pi / rhobar) ,   rhobar = rho_max = 1 .

The `rhobar = rho_max` appears because the drift scale `omega_E ~ (alpha/|Omega|)
M(r)/r^2 ~ rho_max * (...)` is proportional to rho_max: normalizing by rhobar brings it
back to the unit ring amplitude. The global factor `2 pi / rhobar` is therefore the exact
ExB-natural-clock (radian/turn) -> omega_d-clock (cyclic) conversion, not a fit.

**Consequence for HOFFART_FIDELITY.md (full model).** The factor `2 pi / rhobar` belongs
only to the scalar reduced-ExB path (`diag_polar_omega.py`), which measures on the
ExB-natural clock. The present computation builds directly in units omega_d
(`Wd = 2 pi omega_d`) and recovers the targets with no factor: this is exactly what the
normalization resolution of `HOFFART_FIDELITY.md` states. The full model's raw growth
slope (`run.py --engine system-schur`, which evolves in units omega_d) is directly
comparable to 0.772 / 0.911 / 0.683, with no 2 pi factor.


## Rerun

    python hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py

numpy only, lightweight (two 2 x 2 per mode). The script asserts the three checks
(gamma_l < 1 %, Re < 1 %, Im/Re < 2 %) and exits with an error if any one falls outside
its margin.


## References

- D. Hoffart, R. Maier, J. N. Shadid, I. Tomas, structure-preserving FE for magnetic
  Euler-Poisson, arXiv:2510.11808 (Sec 5.3, eq (5.1), Fig 5.4, reference [13]).
- R. C. Davidson, *Physics of Nonneutral Plasmas*, ch. 6 (diocotron instability).
- R. H. Levy, *Diocotron Instability in a Cylindrical Geometry*, Phys. Fluids 8 (1965) 1288.
