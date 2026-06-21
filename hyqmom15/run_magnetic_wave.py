#!/usr/bin/env python3
"""Cas hyqmom15/run_magnetic_wave : onde magnetique du nouveau Matlab.

Porte `magnetic_wave` de RieMOM2D_Electrostatic_periodic : le cas wave le plus
sensible (Poisson + source ELECTRIQUE + source MAGNETIQUE + reconstruction MUSCL).
Perturbation eigenmode 2D (mode 15, kx=2*pi, ky=4*pi, eps=0.01). Voir
matlab_ref/REFERENCE.md et la couche matlab_ref (ADC-349).

Decision D4 (init) : init_magnetic_wave.m cable par oversight l'init
electrostatique ; on porte l'INTENTION -- init_magnetic_wave_field (Jacobien
magnetostatique complexe, avec omega_c). Le chemin as_written (electrostatique)
reste nomme et l'ecart est verifie (intended != as_written).

Schema : HLL + exact_speeds ; reconstruction="muscl" -> limiter="minmod" (ADC-356 :
adc.FiniteVolume(limiter=...) EST la reconstruction) ; time_scheme="Euler" ->
adc.Explicit(method="euler") sur backend="production" ; Poisson periodique ;
source electrique ET magnetique (omega_c=-40 dans build_moment_model, D1).

Orientation de grille : l'IC eigenmode matlab_ref est M[k, i, j] avec i selon x
(kx) et j selon y (ky), comme le Matlab. ADC attend (k, ny, nx) (x en dernier axe,
cf. run_crossing). Comme kx ET ky sont non nuls ici, l'orientation COMPTE : on
transpose donc l'IC en (k, ny, nx) avant set_state (swapaxes 1,2) et on retranspose
la sortie pour l'oracle L2 (convention layer). Le diocotron n'a pas ce swap (son IC
meshgrid est deja (k, ny, nx)).

Validation
----------
  (1) IC : init_magnetic_wave_field intended (magnetostatique, verrouille ADC-350)
      finie, M00 > 0, L2(IC,0)=0 ; intended (magnetostatique) != as_written
      (electrostatique) ;
  (2) smoke natif : Poisson + source E + source B (omega_c=-40) + HLL exact +
      MUSCL(minmod) + Euler (production), dt via compute_dt : fini, M00 > 0, masse
      conservee, phi fini.

Ne prouve pas : la fidelite trajectoire longue, le golden un-pas natif (suivi), la
resolution Matlab pleine Np=256 (smoke a n reduit ; run plein hors CI).
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import build_moment_model  # noqa: E402

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

import adc  # noqa: E402

from matlab_ref import (  # noqa: E402
    compute_L2_error,
    compute_dt,
    explicit_for,
    get_case,
    init_magnetic_wave_field,
)

CASE = get_case("magnetic_wave")  # Np=256, mode=15, kx=2pi, ky=4pi, es=1, ms=1, oc=-40
SMOKE_N = 64  # smoke reduit pour CI ; Np=256 du Matlab = run plein hors CI


def mag_case(n: int):
    return dataclasses.replace(CASE, Np=n)


def mag_ic(n: int) -> np.ndarray:
    """IC eigenmode magnetostatique (15, n, n) via matlab_ref, layer (k, x, y)."""
    return init_magnetic_wave_field(mag_case(n), wiring="intended").M


def _to_adc(M: np.ndarray) -> np.ndarray:
    """Layer (k, x, y) -> ADC (k, ny, nx) : x en dernier axe (cf. run_crossing)."""
    return np.swapaxes(M, 1, 2)


def build_mag_sim(n: int, rho_bg: float, name: str = "mom") -> "adc.System":
    """System periodique : source ELECTRIQUE + MAGNETIQUE + Poisson, HLL exact, Euler.

    omega_c=-40 active la source magnetique (D1). MUSCL(minmod) via limiter="minmod"
    (ADC-356). backend="production" + Explicit(method="euler") pour Euler fidele.
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_magwave",
        with_sources=True,
        q_over_m=1.0,
        omega_c=CASE.omega_c,                 # -40 : source magnetique ACTIVE (D1)
        debye=CASE.adim_debye_length,         # 1/omega_p = 1/20
        rho_background=rho_bg,
        omega_p=CASE.omega_p,
        exact_speeds=True,
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_magwave.so"),
        adc_include(),
        backend="production",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="minmod", riemann="hll"),  # reconstruction=muscl
        time=explicit_for(CASE.time_scheme),  # "Euler" -> adc.Explicit(method="euler")
    )
    sim.set_poisson(rhs="charge_density", solver="fft")
    return sim


def check_ic() -> None:
    """(1) IC : finie, M00 > 0, L2(IC,0)=0 ; intended (magnetostatique) != as_written."""
    U = mag_ic(SMOKE_N)
    assert U.shape == (15, SMOKE_N, SMOKE_N), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC magnetic_wave non finie"
    assert np.all(U[0] > 0), "IC magnetic_wave : M00 non positif"
    l2 = compute_L2_error(U, 0.0, mag_case(SMOKE_N))
    assert l2 < 1e-12, "L2(IC, t=0) = %g, attendu ~0" % l2
    # D4 : init intended (magnetostatique) differe de as_written (electrostatique).
    aw = init_magnetic_wave_field(mag_case(SMOKE_N), wiring="as_written").M
    assert not np.allclose(U, aw), "intended (magnetostatique) devrait differer de as_written"
    print(
        "(1) IC matlab_ref intended (magnetostatique, mode=%d, kx=2*pi, ky=4*pi, eps=%g) : "
        "(15,%d,%d) finie, M00 > 0, L2(IC,0)=%.2e ; intended != as_written -- OK"
        % (CASE.mode, CASE.eps, SMOKE_N, SMOKE_N, l2)
    )


def check_smoke(n: int = SMOKE_N, nsteps: int = 10) -> None:
    """(2) smoke natif : Poisson + E + B + HLL exact + MUSCL(minmod) + Euler (production)."""
    case_n = mag_case(n)
    cfl = case_n.cfl
    U0 = mag_ic(n)
    rho_bg = float(U0[0].mean())

    probe = build_mag_sim(n, rho_bg=rho_bg)
    probe.set_state("mom", _to_adc(U0))
    probe.solve_fields()
    dt_cfl = probe.step_cfl(cfl)
    vmax = cfl * case_n.dx / dt_cfl

    sim = build_mag_sim(n, rho_bg=rho_bg)
    sim.set_state("mom", _to_adc(U0))
    sim.solve_fields()
    mass0 = float(U0[0].sum())
    t = 0.0
    for _ in range(nsteps):
        dt = compute_dt(vmax, case_n, t)
        sim.step(dt)
        t += dt
    U = np.swapaxes(np.array(sim.get_state("mom")), 1, 2)  # ADC (k,ny,nx) -> layer (k,x,y)
    assert np.all(np.isfinite(U)), "smoke : etat non fini"
    assert np.all(U[0] > 0), "smoke : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke : masse non conservee (%.2e)" % drift
    assert np.all(np.isfinite(np.array(sim.potential()))), "smoke : phi non fini"
    l2 = compute_L2_error(U, t, case_n)
    assert l2 < 1e-2, "smoke : L2 vs eigenmode trop grand (%.2e) -- instabilite ?" % l2
    print(
        "(2) smoke natif Poisson + E + B(oc=%g) + HLL exact + MUSCL(minmod) + Euler(production), "
        "dt matlab_ref (vmax=%.3f, dt=%.2e) : %d pas, M00 > 0, masse %.1e, phi fini, L2=%.2e -- OK"
        % (CASE.omega_c, vmax, compute_dt(vmax, case_n, 0.0), nsteps, drift, l2)
    )


def main() -> None:
    print("=== hyqmom15/run_magnetic_wave : onde magnetique E+B (ADC-354) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
