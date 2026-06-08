#!/usr/bin/env python3
"""Demo "multispecies" : couplage de deux fluides heterogenes par un meme Poisson.

Capacite demontree
------------------
Le systeme generique `adc.System` integre simultanement deux especes decrites
par des modeles physiques differents, compiles en C++ :

  * les electrons : Euler compressible complet (4 variables, rho/rho*u/rho*v/E),
  * les ions      : Euler isotherme (3 variables, ferme par cs2).

Ces deux fluides heterogenes sont couples par un seul probleme elliptique
(Poisson de systeme) dont le second membre agrege les charges des deux especes :

        Poisson(phi) = f = q_e * n_e + q_i * n_i   (rhs = "charge_density")

Python ne fait ici que composer le systeme (add_block), poser l'etat initial,
piloter l'avancee en temps et diagnostiquer ; toute la physique (les deux fluides
+ le Poisson couple) est en C++.

Invariants verifies (assert)
-----------------------------
  * Conservation de la masse par espece : |mass_e - mass_e0| < 1e-9 et idem pour
    les ions. C'est le test fort du decouplage des bilans de masse : meme couplees
    par le champ, les deux especes conservent independamment leur masse.
  * Densites finies (pas de NaN/Inf) en fin d'integration.
  * Positivite des densites (un fluide physique reste positif).
  * Presence effective d'une separation de charge initiale (max|f| > 1e-6),
    qui est precisement le terme source qui alimente le Poisson couple.
"""

import numpy as np
import adc

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (compositions de briques nommees, cote application)
from adc_cases.common.checks import (  # noqa: E402
    assert_finite, assert_mass_conserved, assert_positive)


def main():
    # --- Systeme : grille 48x48, deux especes de charges opposees --------------
    sim = adc.System(
        n=48,             # grille 48x48, petite pour rester rapide
        L=1.0,            # domaine carre [0, L]^2
        periodic=True,    # conditions aux limites periodiques
    )

    # Electrons : Euler complet, charge -1, reconstruction minmod, temps explicite.
    # gamma = 5/3 (adiabatique) est porte par le modele compose, pas par le systeme.
    sim.add_block(
        "electrons",
        model=models.electron_euler(charge=-1.0, gamma=5.0 / 3.0),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    # Ions : Euler isotherme, charge +1, fermeture cs2 = 1.0 portee par le modele.
    sim.add_block(
        "ions",
        model=models.ion_isothermal(charge=+1.0, cs2=1.0),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )

    # Poisson couple : second membre = densite de charge q_e n_e + q_i n_i.
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")

    # --- Etat initial : separation de charge ------------------------------------
    # On perturbe les electrons par un petit cosinus le long de x et on laisse les
    # ions uniformes : cela cree un desequilibre local entre n_e et n_i, donc un
    # second membre f = q_e n_e + q_i n_i non nul qui pilote le Poisson couple.
    n = sim.nx()
    x = (np.arange(n) + 0.5) / n * 1.0  # x = (i+0.5)/n * L le long de l'axe 1
    ne = 1.0 + 0.02 * np.cos(2.0 * np.pi * x / 1.0)
    ne2d = np.broadcast_to(ne, (n, n)).copy()  # x varie le long de axis=1
    ni2d = np.ones((n, n))

    sim.set_density("electrons", ne2d)
    sim.set_density("ions", ni2d)

    mass_e0 = sim.mass("electrons")
    mass_i0 = sim.mass("ions")

    # Separation de charge initiale calculee en numpy a partir des deux densites :
    # max|q_e n_e + q_i n_i| = max|(-1) n_e + (+1) n_i|.
    de0 = sim.density("electrons")
    di0 = sim.density("ions")
    charge0 = (-1.0) * de0 + (+1.0) * di0
    qmax0 = float(np.max(np.abs(charge0)))

    print(f"[init] grille nx = {n} x {n}")
    print(f"[init] masse electrons mass_e0 = {mass_e0:.12e}")
    print(f"[init] masse ions      mass_i0 = {mass_i0:.12e}")
    print(f"[init] separation de charge max|f| = {qmax0:.6e}")

    # La separation de charge doit etre non nulle : sinon le Poisson couple est
    # trivial et la demo ne demontre rien.
    assert qmax0 > 1e-6, "separation de charge initiale absente"

    # --- Avancee en temps : 20 pas de dt = 0.001 --------------------------------
    dt = 0.001
    nsteps = 20
    sim.advance(dt, nsteps)

    mass_e1 = sim.mass("electrons")
    mass_i1 = sim.mass("ions")
    de = sim.density("electrons")
    di = sim.density("ions")
    charge1 = (-1.0) * de + (+1.0) * di
    qmax1 = float(np.max(np.abs(charge1)))

    print(f"[t={sim.time():.4f}] masse electrons mass_e = {mass_e1:.12e}")
    print(f"[t={sim.time():.4f}] masse ions      mass_i = {mass_i1:.12e}")
    print(f"[t={sim.time():.4f}] separation de charge max|f| = {qmax1:.6e}")

    # --- Verification des invariants physiques (utilitaires partages) -----------
    # Conservation de la masse par espece (coeur de la demo de couplage), en absolu
    # comme historiquement.
    drift_e = assert_mass_conserved(mass_e1, mass_e0, tol=1e-9, label="electrons",
                                    relative=False)
    drift_i = assert_mass_conserved(mass_i1, mass_i0, tol=1e-9, label="ions",
                                    relative=False)
    print(f"[diag] derive masse electrons |dM_e| = {drift_e:.3e}")
    print(f"[diag] derive masse ions      |dM_i| = {drift_i:.3e}")

    # Densites finies (stabilite numerique) et positives (fluide physique).
    assert_finite(de, "densite electrons")
    assert_finite(di, "densite ions")
    assert_positive(de, "densite electrons")
    assert_positive(di, "densite ions")

    print(f"[diag] n_e dans [{de.min():.6f}, {de.max():.6f}]")
    print(f"[diag] n_i dans [{di.min():.6f}, {di.max():.6f}]")

    print("OK multispecies")


if __name__ == "__main__":
    main()
