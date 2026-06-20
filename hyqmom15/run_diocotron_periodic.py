#!/usr/bin/env python3
"""Cas hyqmom15/run_diocotron_periodic : diocotron du nouveau Matlab periodique.

Aligne le diocotron sur la reference canonique RieMOM2D_Electrostatic_periodic
(init_diocotron.m), distincte du driver historique run_diocotron.py (cale sur
l'ancien RIEMOM2D, omega_p=25, omega_c=-30, source magnetique inactive). Voir
hyqmom15/matlab_ref/REFERENCE.md (ADC-348) et la couche matlab_ref (ADC-349).

Parametres du nouveau Matlab (init_diocotron.m), via matlab_ref.get_case :
omega_p=20, omega_c=-20, adim_debye=1/20, Np=128, CFL=0.5, tmax=1.0, mode=4,
HLL / first / minmod / Euler / periodique, electrostatic=1, magnetostatic=1.

Differences cle vs run_diocotron.py (acte la migration, ne la masque pas) :
  - source MAGNETIQUE active : build_moment_model(omega_c=-20) (D1), pas omega_c=0 ;
  - integration EULER fidele : adc.Explicit(method="euler") sur le backend
    "production" (le backend "aot" fige SSPRK2, cf. ADC-356) ;
  - pas de temps via matlab_ref.compute_dt (politique compute_dt.m, D6) ;
  - IC = matlab_ref.init_diocotron_field, derive ExB STANDARD corrigee par defaut
    (D2 ; le piege meshgrid transpose reste l'option nommee orientation="matlab_bug").

Validation
----------
  (1) IC : matlab_ref.init_diocotron_field (verrouille vs golden Octave en ADC-350)
      finie, M00 > 0, et l'orientation standard differe du piege matlab_bug ;
  (2) smoke natif : HLL + exact_speeds + source electrique + source magnetique
      (omega_c=-20) + Poisson periodique + Euler (production), pas de temps
      matlab_ref.compute_dt : etat fini, M00 > 0, masse conservee, phi fini.

Ne prouve pas : le taux de croissance diocotron (campagne dediee), la realisabilite
long terme sans relaxation15, la resolution Matlab pleine Np=128 (smoke a n reduit ;
le run plein est hors CI). Le golden un-pas natif est un suivi (chemin natif, cf.
ADC-356 sur Euler+Poisson+production).
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

from matlab_ref import compute_dt, get_case, init_diocotron_field  # noqa: E402

CASE = get_case("dicotron")
SMOKE_N = 64  # smoke reduit pour CI ; Np=128 du Matlab = run plein hors CI


def periodic_case(n: int):
    """Le Case diocotron a la resolution ``n`` (parametres physiques inchanges)."""
    return dataclasses.replace(CASE, Np=n)


def diocotron_ic(n: int, orientation: str = "standard") -> np.ndarray:
    """IC conservative (15, n, n) via matlab_ref (couche verrouillee ADC-350).

    Layout ADC (k, ny, nx) : matlab_ref construit M[k, i, j] avec i=ligne=y,
    j=colonne=x (meshgrid(xm, ym)), donc deja (k, ny, nx).
    """
    return init_diocotron_field(periodic_case(n), orientation=orientation).M


def build_periodic_sim(n: int, rho_bg: float, name: str = "mom") -> "adc.System":
    """System periodique fidele au nouveau Matlab : E+B sources, HLL exact, Euler.

    omega_c=-20 active la source magnetique (D1). exact_speeds=True donne les
    vitesses d'onde par valeurs propres (HLL fidele). backend="production" +
    Explicit(method="euler") pour coller a time_scheme="Euler" (le backend "aot"
    figerait SSPRK2). Poisson periodique direct (solveur "fft", l'analogue de
    poisson_fft.m).
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_vp_periodic",
        with_sources=True,
        q_over_m=1.0,
        omega_c=CASE.omega_c,                 # -20 : source magnetique ACTIVE (D1)
        debye=CASE.adim_debye_length,         # 1/omega_p = 0.05
        rho_background=rho_bg,
        omega_p=CASE.omega_p,                 # borne dt source
        exact_speeds=True,
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_vp_periodic.so"),
        adc_include(),
        backend="production",                 # requis pour Euler (cf. ADC-356)
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        time=adc.Explicit(method="euler"),
    )
    sim.set_poisson(rhs="charge_density", solver="fft")
    return sim


def check_ic() -> None:
    """(1) IC matlab_ref : finie, M00 > 0, orientation standard != matlab_bug."""
    U = diocotron_ic(SMOKE_N, orientation="standard")
    assert U.shape == (15, SMOKE_N, SMOKE_N), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC diocotron non finie"
    assert np.all(U[0] > 0), "IC diocotron : M00 non positif"
    assert CASE.rho_min - 1e-12 <= U[0].min() and U[0].max() <= CASE.rho_max + 1e-12, (
        "densite hors anneau [rho_min, rho_max]"
    )
    bug = diocotron_ic(SMOKE_N, orientation="matlab_bug")
    np.testing.assert_allclose(U[0], bug[0], rtol=0, atol=1e-15,
                               err_msg="la densite ne doit pas dependre de l'orientation")
    assert not np.allclose(U, bug), "standard et matlab_bug devraient differer (vitesses)"
    print(
        "(1) IC matlab_ref (op=%g, oc=%g, mode=%d) : (15,%d,%d) finie, M00 in [%.1e, %.3f], "
        "standard != matlab_bug -- OK"
        % (CASE.omega_p, CASE.omega_c, CASE.mode, SMOKE_N, SMOKE_N, U[0].min(), U[0].max())
    )


def check_smoke(n: int = SMOKE_N, nsteps: int = 10) -> None:
    """(2) smoke natif : E+B sources, HLL exact, Euler production, dt matlab_ref.

    Le diocotron est source-limite (vmax ~ vitesse thermique < omega_p=20), donc
    la borne source de compute_dt.m mord et impose un dt bien plus petit que le CFL
    de flux : on STEP ce dt impose (Euler fidele), on ne se contente pas de step_cfl.
    """
    case_n = periodic_case(n)
    cfl = case_n.cfl
    U0 = diocotron_ic(n, orientation="standard")
    rho_bg = float(U0[0].mean())

    # vmax exact via une etape sonde CFL sur un sim jetable (cfl*dx/dt_cfl ==
    # max |eigenvalues15_2D| sur la grille) : c'est le compute_speeds de la boucle
    # Matlab, PAS le matlab_ref.diocotron_max_speed vestigial (inutilise au runtime).
    # Puis dt = compute_dt(vmax, ...).
    probe = build_periodic_sim(n, rho_bg=rho_bg)
    probe.set_state("mom", U0)
    probe.solve_fields()
    dt_cfl = probe.step_cfl(cfl)
    vmax = cfl * case_n.dx / dt_cfl
    dt0 = compute_dt(vmax, case_n, 0.0)
    source_limited = dt0 < dt_cfl - 1e-15

    sim = build_periodic_sim(n, rho_bg=rho_bg)
    sim.set_state("mom", U0)
    sim.solve_fields()
    mass0 = float(U0[0].sum())
    t = 0.0
    for _ in range(nsteps):
        dt = compute_dt(vmax, case_n, t)   # politique compute_dt.m (cap source + clamp)
        sim.step(dt)
        t += dt
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "smoke : etat non fini"
    assert np.all(U[0] > 0), "smoke : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke : masse non conservee (%.2e)" % drift
    assert np.all(np.isfinite(np.array(sim.potential()))), "smoke : phi non fini"
    print(
        "(2) smoke natif HLL exact + E+B (oc=%g) + Poisson + Euler(production), dt matlab_ref "
        "(vmax=%.3f, dt=%.2e, source-limite=%s) : %d pas, M00 > 0, masse %.1e, phi fini -- OK"
        % (CASE.omega_c, vmax, dt0, source_limited, nsteps, drift)
    )


def main() -> None:
    print("=== hyqmom15/run_diocotron_periodic : diocotron Matlab periodique (ADC-351) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
