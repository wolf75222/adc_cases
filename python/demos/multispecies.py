#!/usr/bin/env python3
"""Demo "multispecies" : couplage de deux fluides heterogenes par un meme Poisson.

Capacite demontree
------------------
Le solveur ``MultiSpeciesSolver`` integre simultanement DEUX especes decrites par
des modeles physiques DIFFERENTS, compilees en C++ :

  * les electrons : Euler compressible complet (4 variables, rho/rho*u/rho*v/E),
  * les ions      : Euler isotherme (3 variables, ferme par cs2_i).

Ces deux fluides heterogenes sont couples par UN SEUL probleme elliptique
(Poisson de systeme) dont le second membre agrege les charges des deux especes :

        -eps^2 * Laplacien(phi) = f = q_e * n_e + q_i * n_i

Python ne fait ici que composer la configuration, piloter l'avancee en temps et
DIAGNOSTIQUER ; toute la physique (les deux fluides + le Poisson couple) est en C++.

Invariants verifies (assert)
-----------------------------
  * Conservation de la masse PAR ESPECE : |mass_e - mass_e0| < 1e-9 et idem pour
    les ions. C'est le test fort du decouplage des bilans de masse : meme couplees
    par le champ, les deux especes conservent independamment leur masse.
  * Densites finies (pas de NaN/Inf) en fin d'integration.
  * Presence effective d'une separation de charge initiale (max_charge() > 0),
    qui est precisement le terme source qui alimente le Poisson couple.
"""

import numpy as np
import adc


def main():
    # --- Configuration : n=48, eps=0.02, deux especes de charges opposees -------
    cfg = adc.MultiSpeciesConfig()
    cfg.n = 48              # grille 48x48, petite pour rester rapide
    cfg.L = 1.0             # domaine carre [0, L]^2
    cfg.gamma = 5.0 / 3.0   # adiabatique pour les electrons (Euler complet)
    cfg.cs2_i = 1.0         # vitesse du son^2 des ions (fermeture isotherme)
    cfg.qom_e = -1.0        # rapport charge/masse electrons
    cfg.qom_i = 1.0         # rapport charge/masse ions
    cfg.q_e = -1.0          # charge des electrons (terme source Poisson)
    cfg.q_i = 1.0           # charge des ions       (terme source Poisson)
    cfg.eps = 0.02          # parametre elliptique (regime quasi-neutre)

    solver = adc.MultiSpeciesSolver(cfg)

    # --- Etat initial : separation de charge ------------------------------------
    # La perturbation initiale cree un desequilibre local entre n_e et n_i :
    # c'est le second membre f = q_e n_e + q_i n_i qui pilote le Poisson couple.
    n = solver.nx()
    mass_e0 = solver.mass_e()
    mass_i0 = solver.mass_i()
    qmax0 = solver.max_charge()

    print(f"[init] grille nx = {n} x {n}, eps = {cfg.eps}")
    print(f"[init] masse electrons mass_e0 = {mass_e0:.12e}")
    print(f"[init] masse ions      mass_i0 = {mass_i0:.12e}")
    print(f"[init] separation de charge max|f| = {qmax0:.6e}")

    # La separation de charge doit etre non nulle : sinon le Poisson couple est
    # trivial et la demo ne demontre rien.
    assert qmax0 > 1e-6, "separation de charge initiale absente"

    # --- Avancee en temps : 20 pas de dt = 0.001 --------------------------------
    dt = 0.001
    nsteps = 20
    solver.advance(dt, nsteps)

    mass_e1 = solver.mass_e()
    mass_i1 = solver.mass_i()
    de = solver.density_e()
    di = solver.density_i()
    qmax1 = solver.max_charge()

    print(f"[t={solver.time():.4f}] masse electrons mass_e = {mass_e1:.12e}")
    print(f"[t={solver.time():.4f}] masse ions      mass_i = {mass_i1:.12e}")
    print(f"[t={solver.time():.4f}] separation de charge max|f| = {qmax1:.6e}")

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
