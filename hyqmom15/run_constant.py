#!/usr/bin/env python3
"""Cas hyqmom15/run_constant : etat uniforme (non-regression) du nouveau Matlab.

Porte `constant` de RieMOM2D_Electrostatic_periodic (init_constant.m /
init_constant_field.m) : Maxwellienne au repos uniforme, sans onde, sans Poisson,
sans source. Cas de NON-REGRESSION : un champ uniforme en transport pur doit
rester EXACTEMENT uniforme ; toute dynamique parasite (flux, reconstruction,
source) casserait l'uniformite. Voir matlab_ref/REFERENCE.md et la couche
matlab_ref (ADC-349).

Schema (init_constant.m) : HLL + exact_speeds, reconstruction="muscl" ->
limiter="minmod" (ADC-356), time_scheme="Euler" -> adc.Explicit(method="euler")
sur backend="production", periodique, source=0 / electrostatic=0 / magnetostatic=0.

Validation
----------
  (1) IC : init_constant_field uniforme (verrouille vs golden Octave en ADC-350),
      == Mi partout, M00 > 0 ;
  (2) smoke natif : HLL exact + MUSCL(minmod) + Euler (production), transport pur :
      l'etat reste uniforme (== IC a la precision machine), masse conservee, fini.
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

from matlab_ref import get_case, init_constant_field  # noqa: E402

CASE = get_case("constant")  # Np=64, source=0, es=0, ms=0, muscl/minmod


def constant_ic() -> np.ndarray:
    """IC uniforme (15, Np, Np) via matlab_ref : Mi partout (transpose-invariant)."""
    return init_constant_field(CASE).M


def build_constant_sim(n: int, name: str = "mom") -> "adc.System":
    """System periodique transport pur : HLL exact, MUSCL(minmod), Euler (production)."""
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_constant",
        with_sources=False,       # constant : source=0, electrostatic=0, magnetostatic=0
        exact_speeds=True,
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_constant.so"),
        adc_include(),
        backend="production",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="minmod", riemann="hll"),  # reconstruction=muscl
        time=adc.Explicit(method="euler"),
    )
    return sim


def check_ic() -> None:
    """(1) IC matlab_ref : uniforme, == Mi partout, M00 > 0."""
    U = constant_ic()
    assert U.shape == (15, CASE.Np, CASE.Np), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC constant non finie"
    assert np.all(U[0] > 0), "IC constant : M00 non positif"
    Mi = CASE.equilibrium_moments()
    np.testing.assert_allclose(U, np.broadcast_to(Mi[:, None, None], U.shape), rtol=0, atol=1e-14,
                               err_msg="IC constant non uniforme / != Mi")
    print(
        "(1) IC matlab_ref : (15,%d,%d) uniforme == Mi, M00 = %.3f -- OK"
        % (CASE.Np, CASE.Np, float(U[0, 0, 0]))
    )


def check_smoke(nsteps: int = 10) -> None:
    """(2) smoke natif : transport pur d'un champ uniforme -> reste uniforme."""
    n = CASE.Np
    cfl = CASE.cfl
    U0 = constant_ic()
    sim = build_constant_sim(n)
    sim.set_state("mom", U0)   # uniforme : transpose-invariant, pas de swap
    mass0 = float(U0[0].sum())
    for _ in range(nsteps):
        sim.step_cfl(cfl)
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "smoke : etat non fini"
    assert np.all(U[0] > 0), "smoke : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke : masse non conservee (%.2e)" % drift
    # Non-regression : un champ uniforme en transport pur ne doit PAS bouger.
    dev = float(np.max(np.abs(U - U0)))
    assert dev < 1e-12, "smoke : l'etat uniforme a derive (max|U-U0|=%.2e) -- dynamique parasite" % dev
    print(
        "(2) smoke natif HLL exact + MUSCL(minmod) + Euler(production), transport pur : %d pas, "
        "uniforme preserve (max|U-U0|=%.2e), masse %.1e -- OK" % (nsteps, dev, drift)
    )


def main() -> None:
    print("=== hyqmom15/run_constant : etat uniforme (non-regression, ADC-355) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
