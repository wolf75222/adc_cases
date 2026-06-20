#!/usr/bin/env python3
"""Cas hyqmom15/run_fluid_wave : onde fluide eigenmode du nouveau Matlab.

Porte `fluid_wave` de RieMOM2D_Electrostatic_periodic (init_fluid_wave.m +
init_fluid_wave_field.m) : le cas wave le plus simple (pas de Poisson, pas de
source electrique ni magnetique). Perturbation eigenmode lineaire d'amplitude
eps=0.01 sur la Maxwellienne au repos, mode 15, kx=4*pi, ky=0. Voir
matlab_ref/REFERENCE.md et la couche matlab_ref (ADC-349).

Schema : le Matlab utilise space_scheme="ROE". ROE n'est PAS disponible pour un
modele DSL HyQMOM15 (pas de hook Roe / primitive 'p' ; audit ADC-356), donc on
utilise riemann="hll" + exact_speeds (chemin NOMME non-strict), en attendant le
hook generique adc_cpp ADC-368. reconstruction="first" -> limiter="none" ;
time_scheme="Euler" -> adc.Explicit(method="euler") sur backend="production".

Validation
----------
  (1) IC : matlab_ref.init_fluid_wave_field (verrouille vs golden Octave en
      ADC-350) finie ; L2(IC, t=0) == 0 contre la solution eigenmode analytique ;
  (2) smoke natif : HLL + exact_speeds + Euler (production), pas de Poisson ni de
      source : etat fini, M00 > 0, masse conservee (transport pur), L2 borne.

Ne prouve pas : la parite ROE stricte (gap adc_cpp ADC-368 ; HLL non-strict ici),
la fidelite trajectoire longue, le golden un-pas natif (suivi).
"""
from __future__ import annotations

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

from matlab_ref import compute_L2_error, get_case, init_fluid_wave_field  # noqa: E402

CASE = get_case("fluid_wave")  # Np=32, eps=0.01, mode=15, kx=4*pi, ky=0, no sources


def fluid_ic() -> np.ndarray:
    """IC eigenmode (15, Np, Np) via matlab_ref (couche verrouillee ADC-350)."""
    return init_fluid_wave_field(CASE).M


def build_fluid_sim(n: int, name: str = "mom") -> "adc.System":
    """System periodique transport pur : HLL exact, Euler (production), pas de source.

    ROE du Matlab -> HLL non-strict (gap ADC-368). exact_speeds=True pour les
    vitesses d'onde par valeurs propres. backend="production" pour Euler fidele
    (le backend "aot" figerait SSPRK2, cf. ADC-356). reconstruction="first" du
    Matlab -> limiter="none".
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_fluid",
        with_sources=False,       # fluid_wave : source=0, electrostatic=0, magnetostatic=0
        exact_speeds=True,
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_fluid.so"),
        adc_include(),
        backend="production",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="hll"),  # ROE -> HLL non-strict
        time=adc.Explicit(method="euler"),
    )
    return sim


def check_ic() -> None:
    """(1) IC matlab_ref : finie, M00 > 0, L2(IC, t=0) == 0 (eigenmode analytique)."""
    U = fluid_ic()
    assert U.shape == (15, CASE.Np, CASE.Np), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC fluid_wave non finie"
    assert np.all(U[0] > 0), "IC fluid_wave : M00 non positif"
    l2 = compute_L2_error(U, 0.0, CASE)
    assert l2 < 1e-12, "L2(IC, t=0) = %g, attendu ~0" % l2
    print(
        "(1) IC matlab_ref (mode=%d, kx=4*pi, ky=0, eps=%g) : (15,%d,%d) finie, M00 > 0, "
        "L2(IC,0)=%.2e -- OK" % (CASE.mode, CASE.eps, CASE.Np, CASE.Np, l2)
    )


def check_smoke(nsteps: int = 10) -> None:
    """(2) smoke natif : HLL exact + Euler production, transport pur, L2 borne."""
    n = CASE.Np
    cfl = CASE.cfl
    U0 = fluid_ic()
    sim = build_fluid_sim(n)
    sim.set_state("mom", U0)
    mass0 = float(U0[0].sum())
    t = 0.0
    for _ in range(nsteps):
        # fluid_wave n'a pas de source : compute_dt se reduit au CFL de flux, donc
        # step_cfl == dt Matlab. On cumule t pour l'oracle L2.
        t += sim.step_cfl(cfl)
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "smoke : etat non fini"
    assert np.all(U[0] > 0), "smoke : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke : masse non conservee (transport pur) (%.2e)" % drift
    l2 = compute_L2_error(U, t, CASE)
    # HLL (non-strict) diffuse legerement vs l'eigenmode analytique ROE : sur ce smoke
    # court (eps=0.01, onde lisse) L2 reste petit ; borne genereuse anti-blow-up.
    assert l2 < 1e-2, "smoke : L2 vs eigenmode trop grand (%.2e) -- instabilite ?" % l2
    print(
        "(2) smoke natif HLL exact + Euler(production), transport pur : %d pas (t=%.3e), "
        "M00 > 0, masse %.1e, L2=%.2e (HLL non-strict vs ROE, gap ADC-368) -- OK"
        % (nsteps, t, drift, l2)
    )


def main() -> None:
    print("=== hyqmom15/run_fluid_wave : onde fluide eigenmode (ADC-352) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
