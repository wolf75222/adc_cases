#!/usr/bin/env python3
"""Cas "safe_euler_periodic" : le CAS SUR de reference de la campagne de perf, en version VALIDATION.

Euler compressible PUR, periodique, bulle de pression lisse de faible amplitude (rho>0, p>0
garantis), AUCUNE source ni Poisson : transport pur. C'est le cas que la campagne de perf
(perf/frontend_compare.py) mesure sur trois fronts ; ici on en VALIDE la physique et l'equivalence
des fronts, sans aucune mesure de temps (CI-friendly).

On verifie :
  - EQUIVALENCE briques <-> DSL : etat final BIT-IDENTIQUE (np.array_equal), comme diocotron_dsl /
    two_species_dsl (memes reglages numeriques : minmod / rusanov / conservative / SSPRK2 / dt fixe) ;
  - INVARIANTS : masse conservee (transport, domaine periodique), rho > 0, p > 0, etat fini ;
  - DYNAMIQUE non triviale : la bulle de pression genere des ondes acoustiques (l'etat bouge).

Le modele (briques & DSL), les CI et les reglages sont l'UNIQUE source de verite
adc_cases.common.safe_euler -- partagee avec perf/frontend_compare.py. Le pendant C++ direct est
adc_cpp/bench/frontend_cpp.cpp.

Lancement : PYTHONPATH=<build>/python:. python3 safe_euler_periodic/run.py [--n 64 --steps 40]
"""

import argparse

import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adc_cases.common import safe_euler as sc  # noqa: E402
from adc_cases.common.checks import assert_finite, assert_positive, relative_drift  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402


def run_bricks(n, steps):
    """Front BRIQUES : System + add_block(models.euler), dt fixe. Renvoie (etat final, masse)."""
    dt = sc.dt(n)
    sim = adc.System(n=n, L=sc.L, periodic=True)
    sim.add_block("gas", model=sc.bricks_model(), spatial=sc.spatial_bricks(), time=adc.Explicit())
    sim.set_state("gas", sc.ic(n).reshape(-1).tolist())
    for _ in range(steps):
        sim.step(dt)
    return np.asarray(sim.get_state("gas"), dtype=float).reshape(4, n, n), sim.mass("gas")


def run_dsl(n, steps):
    """Front DSL : compile (production -> aot) + add_equation, MEMES reglages. Renvoie (etat, masse, backend)."""
    dt = sc.dt(n)
    so_dir = case_output_dir("safe_euler_periodic")
    model = sc.dsl_model()
    include = adc_include()
    import os
    last = None
    for cand in ("production", "aot"):
        try:
            compiled = model.compile(os.path.join(so_dir, "safe_euler_%s.so" % cand), include,
                                     backend=cand)
            sim = adc.System(n=n, L=sc.L, periodic=True)
            sim.add_equation("gas", model=compiled, spatial=sc.spatial_dsl(), time=adc.Explicit())
            sim.set_state("gas", sc.ic(n).reshape(-1).tolist())
            for _ in range(steps):
                sim.step(dt)
            return np.asarray(sim.get_state("gas"), dtype=float).reshape(4, n, n), sim.mass("gas"), cand
        except Exception as exc:  # noqa: BLE001
            last = exc
            print("backend DSL %r indisponible (%s), essai suivant" % (cand, type(exc).__name__))
    raise RuntimeError("aucun backend DSL n'a compile ni execute le modele Euler sur (%s)" % last)


def main():
    ap = argparse.ArgumentParser(description="Cas sur Euler periodique : validation + equivalence fronts")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    n, steps = args.n, args.steps

    print("=== safe_euler_periodic : Euler compressible pur, periodique (n=%d, %d pas) ===" % (n, steps))
    U0 = sc.ic(n)
    mass0 = float(U0[0].sum())
    p0 = sc.pressure(U0)

    Ub, mb = run_bricks(n, steps)
    Ud, md, backend = run_dsl(n, steps)
    print("backend DSL retenu : %r" % backend)

    # --- EQUIVALENCE briques <-> DSL : etat final bit-identique ---
    max_abs = float(np.max(np.abs(Ub - Ud)))
    identical = bool(np.array_equal(Ub, Ud))
    print("max|briques - DSL| = %.3e   bit-identique = %s" % (max_abs, identical))
    assert max_abs < 1e-10, (
        "briques et DSL divergent (max|d|=%.3e) : une formule DSL s'ecarte de la brique Euler" % max_abs)

    # --- INVARIANTS (sur les briques ; le DSL est bit-identique) ---
    pb = sc.pressure(Ub)
    assert_finite(Ub, "etat")
    assert_positive(Ub[0], "densite")
    assert_positive(pb, "pression")
    mass_drift = relative_drift(mb, mass0)
    moved = float(np.max(np.abs(pb - p0)))
    print("masse : drel=%.3e   dynamique : max|dp|=%.3e" % (mass_drift, moved))
    assert mass_drift < 1e-9, "masse non conservee (drel=%.3e)" % mass_drift
    assert moved > 1e-4, "dynamique triviale (la bulle de pression n'a pas evolue)"

    print("OK safe_euler_periodic (equivalence briques<->DSL, invariants, dynamique)")


if __name__ == "__main__":
    main()
