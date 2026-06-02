#!/usr/bin/env python3
"""Demo "multispecies" : couplage de deux fluides heterogenes par un meme Poisson.

Capacite demontree
------------------
Le systeme generique ``adc.System`` integre simultanement DEUX especes decrites
par des modeles physiques DIFFERENTS, compiles en C++ :

  * les electrons : Euler compressible complet (4 variables, rho/rho*u/rho*v/E),
  * les ions      : Euler isotherme (3 variables, ferme par cs2).

Ces deux fluides heterogenes sont couples par UN SEUL probleme elliptique
(Poisson de systeme) dont le second membre agrege les charges des deux especes :

        Poisson(phi) = f = q_e * n_e + q_i * n_i   (rhs = "charge_density")

Python ne fait ici que composer le systeme (add_block), poser l'etat initial,
piloter l'avancee en temps et DIAGNOSTIQUER ; toute la physique (les deux fluides
+ le Poisson couple) est en C++.

Invariants verifies (assert)
-----------------------------
  * Conservation de la masse PAR ESPECE : |mass_e - mass_e0| < 1e-9 et idem pour
    les ions. C'est le test fort du decouplage des bilans de masse : meme couplees
    par le champ, les deux especes conservent independamment leur masse.
  * Densites finies (pas de NaN/Inf) en fin d'integration.
  * Positivite des densites (un fluide physique reste positif).
  * Presence effective d'une separation de charge initiale (max|f| > 1e-6),
    qui est precisement le terme source qui alimente le Poisson couple.
"""

import numpy as np
import adc


def main():
    # --- Systeme : grille 48x48, deux especes de charges opposees --------------
    sim = adc.System(
        n=48,             # grille 48x48, petite pour rester rapide
        L=1.0,            # domaine carre [0, L]^2
        gamma=5.0 / 3.0,  # adiabatique pour les electrons (Euler complet)
        cs2=1.0,          # vitesse du son^2 des ions (fermeture isotherme)
        periodic=True,    # conditions aux limites periodiques
    )

    # Electrons : Euler complet, charge -1, reconstruction minmod, temps explicite.
    sim.add_block(
        "electrons",
        model="electron_euler",
        charge=-1.0,
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    # Ions : Euler isotherme, charge +1.
    sim.add_block(
        "ions",
        model="ion_isothermal",
        charge=+1.0,
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

    # --- Verification des invariants physiques ----------------------------------
    drift_e = abs(mass_e1 - mass_e0)
    drift_i = abs(mass_i1 - mass_i0)
    print(f"[diag] derive masse electrons |dM_e| = {drift_e:.3e}")
    print(f"[diag] derive masse ions      |dM_i| = {drift_i:.3e}")

    # Conservation de la masse PAR ESPECE (coeur de la demo de couplage).
    assert drift_e < 1e-9, f"masse electrons non conservee: {drift_e:.3e}"
    assert drift_i < 1e-9, f"masse ions non conservee: {drift_i:.3e}"

    # Densites finies (stabilite numerique).
    assert np.all(np.isfinite(de)), "densite electrons non finie (NaN/Inf)"
    assert np.all(np.isfinite(di)), "densite ions non finie (NaN/Inf)"

    # Positivite des densites (un fluide physique reste positif).
    assert de.min() > 0.0, f"densite electrons negative: min = {de.min():.3e}"
    assert di.min() > 0.0, f"densite ions negative: min = {di.min():.3e}"

    print(f"[diag] n_e dans [{de.min():.6f}, {de.max():.6f}]")
    print(f"[diag] n_i dans [{di.min():.6f}, {di.max():.6f}]")

    print("OK multispecies")


if __name__ == "__main__":
    main()
