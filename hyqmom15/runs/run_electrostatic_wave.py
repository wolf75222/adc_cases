#!/usr/bin/env python3
"""Cas hyqmom15/run_electrostatic_wave : onde electrostatique du nouveau Matlab.

Porte `electrostatic_wave` de RieMOM2D_Electrostatic_periodic (init_electrostatic_wave.m
+ init_electrostatic_wave_field.m) : perturbation eigenmode (mode 15, kx=0, ky=4*pi,
eps=0.01) couplee au Poisson periodique avec la source ELECTRIQUE seule (pas de
source magnetique). Voir matlab_ref/REFERENCE.md et la couche matlab_ref (ADC-349).

Decision D3 (Dmax) : la vitesse CFL vient de diag(Dmax) (Jacobien a (kmin, kmin)),
pas du diag(D) du Jacobien du mode -- l'init layer expose `max_speed` selon la
politique `intended` par defaut (le bug `as_written` reste nomme). Cette vitesse
est vestigiale dans la boucle (le runtime prend le vmax dynamique = compute_speeds) ;
on verifie quand meme que intended (Dmax) != as_written (D).

Schema : HLL + exact_speeds (Matlab space_scheme="HLL") ; reconstruction non fixee
cote Matlab -> limiter="none" ; time_scheme="Euler" -> adc.Explicit(method="euler")
sur backend="production" ; Poisson periodique (solveur "fft", analogue poisson_fft.m).
La source magnetique est inactive (magnetostatic=0) : omega_c=0 dans le modele.

Validation
----------
  (1) IC : matlab_ref.init_electrostatic_wave_field (verrouille vs golden Octave en
      ADC-350) finie, M00 > 0, L2(IC, t=0) == 0 ; Dmax (intended) != D (as_written) ;
  (2) smoke natif : Poisson + source electrique + HLL exact + Euler (production),
      dt via matlab_ref.compute_dt : etat fini, M00 > 0, masse conservee, phi fini.

Ne prouve pas : la fidelite trajectoire longue, le golden un-pas natif (suivi), la
resolution Matlab pleine Np=128 (smoke a n reduit ; run plein hors CI).
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # hyqmom15/ : model, relaxation, gen_states

from model import build_moment_model  # noqa: E402

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import adc  # noqa: E402

from matlab_ref import (  # noqa: E402
    compute_L2_error,
    compute_dt,
    explicit_for,
    get_case,
    init_electrostatic_wave_field,
)

CASE = get_case("electrostatic_wave")  # Np=128, mode=15, kx=0, ky=4pi, es=1, ms=0
SMOKE_N = 64  # smoke reduit pour CI ; Np=128 du Matlab = run plein hors CI


def es_case(n: int):
    return dataclasses.replace(CASE, Np=n)


def es_ic(n: int) -> np.ndarray:
    """IC eigenmode electrostatique (15, n, n) via matlab_ref (ADC-350), Dmax intended."""
    return init_electrostatic_wave_field(es_case(n), dmax_policy="intended").M


def build_es_sim(n: int, rho_bg: float, name: str = "mom") -> "adc.System":
    """System periodique : source ELECTRIQUE seule + Poisson, HLL exact, Euler.

    omega_c=0 (magnetostatic=0) : la source de Lorentz se reduit a la partie
    electrique (E = -grad phi). exact_speeds=True ; backend="production" +
    Explicit(method="euler") pour Euler fidele (aot figerait SSPRK2, ADC-356).
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_eswave",
        with_sources=True,
        q_over_m=1.0,
        omega_c=0.0,                          # magnetostatic=0 : pas de cyclotron
        debye=CASE.adim_debye_length,         # 1/omega_p = 1/30
        rho_background=rho_bg,
        omega_p=CASE.omega_p,
        exact_speeds=True,
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_eswave.so"),
        adc_include(),
        backend="production",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        time=explicit_for(CASE.time_scheme),  # "Euler" -> adc.Explicit(method="euler")
    )
    sim.set_poisson(rhs="charge_density", solver="fft")
    return sim


def check_ic() -> None:
    """(1) IC matlab_ref : finie, M00 > 0, L2(IC,0)=0 ; Dmax (intended) != D (as_written)."""
    U = es_ic(SMOKE_N)
    assert U.shape == (15, SMOKE_N, SMOKE_N), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC electrostatic_wave non finie"
    assert np.all(U[0] > 0), "IC electrostatic_wave : M00 non positif"
    l2 = compute_L2_error(U, 0.0, es_case(SMOKE_N))
    assert l2 < 1e-12, "L2(IC, t=0) = %g, attendu ~0" % l2
    # D3 : la CFL speed vient de Dmax (intended), pas de D (as_written).
    s_intended = init_electrostatic_wave_field(CASE, dmax_policy="intended").max_speed
    s_aswritten = init_electrostatic_wave_field(CASE, dmax_policy="as_written").max_speed
    assert abs(s_intended - s_aswritten) > 1e-6, (
        "Dmax (intended) devrait differer de D (as_written) : %s vs %s" % (s_intended, s_aswritten)
    )
    print(
        "(1) IC matlab_ref (mode=%d, kx=0, ky=4*pi, eps=%g) : (15,%d,%d) finie, M00 > 0, "
        "L2(IC,0)=%.2e ; Dmax=%.3f != D=%.3f -- OK"
        % (CASE.mode, CASE.eps, SMOKE_N, SMOKE_N, l2, s_intended.real, s_aswritten.real)
    )


def check_smoke(n: int = SMOKE_N, nsteps: int = 10) -> None:
    """(2) smoke natif : Poisson + source electrique + HLL exact + Euler (production)."""
    case_n = es_case(n)
    cfl = case_n.cfl
    U0 = es_ic(n)
    rho_bg = float(U0[0].mean())

    # Orientation grille : IC eigenmode layer (k, x, y) -> ADC (k, ny, nx) (x en
    # dernier axe, cf. run_crossing) via swapaxes(1,2). Pour ky=4*pi (kx=0) l'onde
    # est 1D mais on garde l'orientation correcte par coherence avec magnetic_wave.
    U0_adc = np.swapaxes(U0, 1, 2)
    probe = build_es_sim(n, rho_bg=rho_bg)
    probe.set_state("mom", U0_adc)
    probe.solve_fields()
    dt_cfl = probe.step_cfl(cfl)
    vmax = cfl * case_n.dx / dt_cfl

    sim = build_es_sim(n, rho_bg=rho_bg)
    sim.set_state("mom", U0_adc)
    sim.solve_fields()
    mass0 = float(U0[0].sum())
    t = 0.0
    for _ in range(nsteps):
        dt = compute_dt(vmax, case_n, t)   # politique compute_dt.m (cap source omega_p^2 + clamp)
        sim.step(dt)
        t += dt
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "smoke : etat non fini"
    assert np.all(U[0] > 0), "smoke : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke : masse non conservee (%.2e)" % drift
    assert np.all(np.isfinite(np.array(sim.potential()))), "smoke : phi non fini"
    print(
        "(2) smoke natif Poisson + source electrique + HLL exact + Euler(production), dt matlab_ref "
        "(vmax=%.3f, dt=%.2e) : %d pas, M00 > 0, masse %.1e, phi fini -- OK"
        % (vmax, compute_dt(vmax, case_n, 0.0), nsteps, drift)
    )


def main() -> None:
    print("=== hyqmom15/run_electrostatic_wave : onde electrostatique (ADC-353) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
