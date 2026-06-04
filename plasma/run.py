#!/usr/bin/env python3
"""Cas "plasma" : electrons + ions + neutres couples (Poisson + ionisation + collision).

Compose le scenario plasma vise par le tuteur depuis Python via une RECETTE SYSTEME
(`models.plasma`) : trois especes (electrons Euler, ions et neutres isothermes) partageant un
Poisson de systeme (f = q_e n_e + q_i n_i), couplees par des SOURCES inter-especes : ionisation
(un neutre devient un ion + un electron) et collision ion-neutre (friction). La recette cable
add_block + set_poisson + add_ionization + add_collision ; le cas ne fait que poser les CI et
verifier. On verifie que la machinerie de couplage se compose et tourne :
  - Poisson de systeme actif (champ non nul issu de la separation de charge) ;
  - ionisation : masse n_i + n_g conservee (transfert neutre -> ion), neutres consommes ;
  - integrite : densites finies et positives sur tout le run.

Les electrons exercent le schema de la Phase 1 (HLLC + reconstruction PRIMITIVE). La collision
ion-neutre (friction) est cablee et active ; sa conservation de quantite de mouvement est verifiee
isolement dans le test des bindings (ici le champ agit aussi sur les ions). L'ionisation transfere
la densite (comp 0) ; le transfert de quantite de mouvement / energie des particules creees est
une simplification (cf. la brique add_ionization).
"""
import os
import sys

import numpy as np

import adc

# Rend le depot importable si le paquet n'est pas installe (cf. adc_cases.ensure_importable).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (recettes systeme nommees, cote application)
from adc_cases.common.checks import relative_drift  # noqa: E402

PI = np.pi


def main():
    n, L = 48, 1.0
    x = (np.arange(n) + 0.5) / n
    ne = 1.0 + 0.05 * np.cos(2 * PI * x)[None, :] * np.ones((n, n))  # faible separation de charge

    sim = adc.System(n=n, L=L, periodic=True)
    models.plasma(sim, ne=ne, ni=np.ones((n, n)), ng=np.ones((n, n)),
                  ionization_rate=0.3, collision_rate=0.5)  # recette systeme : blocs + couplages

    sim.solve_fields()
    phi0 = float(np.abs(np.array(sim.potential())).max())
    mi0, mg0 = sim.mass("ions"), sim.mass("neutrals")

    print("== plasma : electrons + ions + neutres (Poisson + ionisation + collision) ==")
    print("  |phi|_max = %.3e  (Poisson de systeme actif)" % phi0)

    for _ in range(20):
        sim.step_cfl(0.3)

    mi1, mg1 = sim.mass("ions"), sim.mass("neutrals")
    drel = relative_drift(mi1 + mg1, mi0 + mg0)
    dens = {s: np.array(sim.density(s)) for s in ("electrons", "ions", "neutrals")}
    finite_pos = all(np.isfinite(d).all() and float(d.min()) > 0.0 for d in dens.values())

    print("  ionisation : n_i %.4f -> %.4f,  n_g %.4f -> %.4f,  (n_i+n_g) drel = %.2e"
          % (mi0, mi1, mg0, mg1, drel))
    print("  densites   : min e=%.3e i=%.3e n=%.3e (toutes finies et positives : %s)"
          % (dens["electrons"].min(), dens["ions"].min(), dens["neutrals"].min(), finite_pos))

    assert phi0 > 1e-8, "Poisson inactif (pas de separation de charge ?)"
    assert mg1 < mg0 - 1e-6 and mi1 > mi0 + 1e-6, "ionisation : on attend neutres -> ions"
    assert drel < 1e-7, "ionisation : masse n_i + n_g non conservee (drel=%.2e)" % drel
    assert finite_pos, "densite non finie ou negative"
    print("OK plasma")


if __name__ == "__main__":
    main()
