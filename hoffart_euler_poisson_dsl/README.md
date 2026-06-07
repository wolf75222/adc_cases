# Magnetic Euler-Poisson diocotron (Hoffart et al.)

> CLASSIFICATION: reproduction-candidate (NOT a reproduction).
>
> Honest status (Phase 0 / 1): the full Euler-Poisson system is written
> verbatim from the paper and the measurement harness is pre-registered, but NO
> quantitative reproduction of the paper growth rates 0.772 / 0.911 / 0.683 is
> established yet. The Cartesian-square baseline is now MEASURED (n=256 and n=384,
> raw slope, paper windows, NO 2 pi): l=3 -95%, l=4 -94/-93%, l=5 -82%, and it does
> NOT improve from n=256 to n=384 (resolution-independent). That rules out "needs
> more resolution" and confirms the prime suspect is GEOMETRY (a Cartesian square
> box plus an embedded circular Poisson wall diffuses the ring edge), NOT the time
> splitting (the analytic rate depends only on l, omega_d, r0, r1, R). The
> conservative disc cut-cell transport (T5) is in progress to test this. The
> reduced ExB-scalar normalization study
> (`diag/diag_polar_omega.py`, the `2 pi / rhobar` factor) is a DIFFERENT,
> reduced model and is not a reproduction of the full system either. Method-level
> caveats also remain (Lie, not Strang, splitting; once-per-step Gauss re-solve).
> See `adc_cpp/docs/HOFFART_FIDELITY.md` and `adc_cpp/docs/HOFFART_STEP_SEQUENCE.md`
> for the full fidelity audit. Until the system-schur cells are filled, no claim
> that the full model reproduces the paper is permitted.

This case writes the equations of
[Hoffart et al., arXiv:2510.11808](https://arxiv.org/abs/2510.11808)
directly with `adc.dsl.Model`:

```text
d_t rho + div(m) = 0
d_t m + div(m m^T / rho + p I) = -rho grad(phi) + m x Omega
-Delta(phi) = alpha rho
p = theta rho
```

The paper's diocotron test uses this barotropic closure, so the energy
equation of the full Euler system is not evolved: the conservative state is
exactly `(rho, rho*u, rho*v)`.

The generated C++ model is used by ADC finite volumes. No Python callback is
executed per cell.

## Paper parameters

```text
disk radius       R = 16
annulus           r0 = 6, r1 = 8
rho_min / rho_max = 1e-6 / 1
beta              = 1e6
alpha             = beta^2 / rho_max = 1e12
Omega             = beta^2 = 1e12
perturbation      delta = 0.1
modes             l = 3, 4, 5
final time        tf = 10
```

The paper defines the isothermal pressure `p = theta rho` but does not give a
numerical value for `theta` in the published source. This case therefore uses
the cold limit `theta=0` by default and records the chosen value in
`metadata.json`. Use `--temperature` to test another value.

## Growth-rate normalization (the 2 pi / rhobar factor)

The `2 pi / rhobar` global normalization is validated on the POLAR ExB path, which
solves the REDUCED scalar ExB-drift diocotron model (NOT the full Euler-Poisson
system above): `gamma_norm = gamma_raw * 2 pi / rhobar`. With this factor mode
l = 4 matches the paper target exactly, while l = 3 (+26 %) and l = 5 (oscillating)
remain offset. This is a normalization validation of the reduced diocotron
benchmark, NOT a reproduction of the full Hoffart Euler-Poisson model. See
[`NORMALIZATION.md`](NORMALIZATION.md) for the full story (why `gamma_raw` is
already `Im(omega)` in ExB-natural units, why a local-rotation normalization fails,
and the l = 3 / 4 / 5 numbers at n = 128 and n = 192). The reproducible diagnostic
is [`diag/diag_polar_omega.py`](diag/diag_polar_omega.py):

```bash
PYTHONPATH=/path/to/adc_cpp/build-master/python \
  python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 128
```

## Uniform reference with the Schur source stage

```bash
cd /path/to/adc_cases
PYTHONPATH=/path/to/adc_cpp/build-py/python \
  python hoffart_euler_poisson_dsl/run.py \
    --engine system-schur --n 384 --t-end 10 --dt 1e-3
```

This path uses:

- WENO5-Z finite volumes and Rusanov flux;
- SSPRK3 for the hyperbolic stage;
- `CondensedSchur(theta=0.5)` for the electrostatic/Lorentz source;
- the initial drift velocity computed from the initial Poisson solve.

ADC currently applies transport then source as a first-order Lie composition,
whereas the paper uses Strang splitting. The PDE and source solve are the same,
but this splitting difference must be removed before claiming method-level
identity.

## AMR, Kokkos and MPI

The AMR path runs the same symbolic PDE through the production C++ backend:

```bash
cd /path/to/adc_cases
PYTHONPATH=/path/to/adc_cpp/build-kokkos/python \
  mpirun -np 4 python hoffart_euler_poisson_dsl/run.py \
    --engine amr-imex \
    --acknowledge-amr-approximation \
    --n 192 --t-end 10 --dt 1e-3 \
    --distribute-coarse
```

It uses dynamic AMR, reflux, Kokkos kernels and MPI. It is not yet the paper
algorithm:

1. `CondensedSchur` is not implemented on `AmrSystem`; the AMR path uses the
   cell-local backward-Euler source step.
2. `AmrSystem` exposes density initialization but not full conservative or
   primitive state initialization, so the run starts with zero momentum instead
   of the paper drift state.
3. The circular wall is enforced by the Poisson solve; transport still uses the
   Cartesian square domain.

The AMR output is therefore labelled experimental. It is useful to validate the
PDE code generation and the AMR/Kokkos/MPI execution path, not yet to claim a
quantitative reproduction.

## Pre-registered measurement harness (`results.py`)

The full-model measurement is honest and pre-registered:

- the fit windows are the verbatim paper windows (Fig. 5.4): l=3 `[0.40,0.70]`,
  l=4 `[0.60,0.75]`, l=5 `[1.15,1.35]`. `run.py` asserts at startup that
  `PAPER_FIT_WINDOWS` equals these (`results.verify_paper_windows`); no adaptive
  window is ever introduced for the full-model comparison;
- the reported growth rate is the RAW exp-slope of the full system-schur model,
  compared DIRECTLY to 0.772 / 0.911 / 0.683 with NO `2 pi` and NO `rhobar`
  factor. That `2 pi / rhobar` factor belongs ONLY to the reduced ExB-scalar path
  (`diag/diag_polar_omega.py`); the harness labels the full model
  `engine = full-system-schur` and the reduced path `engine = reduced-ExB` so the
  `2 pi`-bearing reduced numbers are never mixed with the raw full-model numbers;
- each run emits a per-run record `measurement_record.csv` / `.json` under
  `out/hoffart_euler_poisson_dsl_<engine>/`, capturing: adc_cpp SHA, adc_cases SHA,
  backend, n, dt, splitting (Lie/Strang), schur(theta), fit window, gamma_numeric
  (RAW), gamma_paper and err_pct. This is the seed of the Phase-2 validation table.

The harness only measures and records what a run produces. When a run has not
returned, `gamma_numeric` and `err_pct` are recorded as `PENDING`; no number is
fabricated. The module is pure Python and self-tests in CI (`python results.py`).

## Validation table (mandatory before any reproduction claim)

No `gamma_numeric` cell is invented: full-Hoffart runs that have not returned are
marked PENDING. Rows with `engine = reduced-ExB` are the already-known reduced
diocotron reconciliation (`diag/diag_polar_omega.py`); they must NOT be confused
with the full Euler-Poisson system. The `amr-imex` rows stay experimental even when
they return (zero initial momentum, no paper Schur). Only the `system-schur` rows
(emitted with the explicit label `full-system-schur`) are a path toward a full-model
fidelity discussion, and they remain encumbered by Lie (not Strang) splitting.
`err_pct = 100*(gamma_numeric - gamma_paper)/gamma_paper`.

| mode | n | gamma_numeric | gamma_paper | err_pct | fit_window | engine | dt | splitting | schur |
|------|-----|---------------|-------------|---------|--------------------|--------------|--------|------------------|----------------------|
| 3 | 256 | 0.0372  | 0.772 | -95.2   | [0.40,0.70] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 4 | 256 | 0.0489  | 0.911 | -94.6   | [0.60,0.75] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 5 | 256 | 0.1211  | 0.683 | -82.3   | [1.15,1.35] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 3 | 384 | 0.0385  | 0.772 | -95.0   | [0.40,0.70] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 4 | 384 | 0.0613  | 0.911 | -93.3   | [0.60,0.75] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 5 | 384 | 0.1257  | 0.683 | -81.6   | [1.15,1.35] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 3 | 512 | PENDING | 0.772 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 4 | 512 | PENDING | 0.911 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 5 | 512 | PENDING | 0.683 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 3 | 128 | 0.9712  | 0.772 | +25.8   | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 4 | 128 | 0.9127  | 0.911 | +0.2    | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 5 | 128 | 0.4820  | 0.683 | -29.4   | [2.12, 12.58]      | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 3 | 192 | 0.9713  | 0.772 | +25.8   | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 4 | 192 | 0.9100  | 0.911 | -0.1    | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 5 | 192 | 0.8658  | 0.683 | +26.8   | [2.12, 5.96]       | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 3 | 192 | PENDING | 0.772 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |
| 4 | 192 | PENDING | 0.911 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |
| 5 | 192 | PENDING | 0.683 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |

The `reduced-ExB` numbers come from `NORMALIZATION.md` (g_2pi measured). Until the
`system-schur` cells are filled, no statement that the full model reproduces the
paper is permitted.

## Outputs

Outputs are written under:

```text
out/hoffart_euler_poisson_dsl_<engine>/
```

For every mode the runner writes:

- `amplitude.csv` and `amplitude.png`;
- a 3 x 3 density-schlieren panel at the paper snapshot times;
- an animated GIF;
- `growth_rates.csv`, `growth_rates.png` and `metadata.json`.

Use `--quick` for a compilation and short execution smoke test.
