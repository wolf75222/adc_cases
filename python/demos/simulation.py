#!/usr/bin/env python3
"""Demo "simulation" : composition multi-especes a l'EXECUTION via Simulation.add_species.

Capacite demontree
------------------
La facade C++ `adc.Simulation` permet au tuteur de COMPOSER son cas physique
espece par espece DEPUIS Python, a l'execution, sans recompiler ni toucher au
C++. Chaque add_species(name, model, charge) instancie un modele cinetique/fluide
compile ; Python ne fait qu'assembler la liste des especes, fixer les densites
initiales et lire les diagnostics. Toute la physique (resolution de Poisson,
advection E x B, Euler electronique, fermeture isotherme ionique) reste en C++.

Deux parties independantes dans le meme script :

(A) "Diocotron a ions mobiles" : deux especes du MEME modele "diocotron", de
    charges opposees (electrons -1, ions +1). Historiquement le fond ionique
    etait fige ; ici les ions sont une espece a part entiere, donc INTEGREE au
    meme titre que les electrons. On verifie :
      - le potentiel self-consistent est non nul (couplage de Poisson actif) ;
      - la masse de CHAQUE espece est conservee (advection conservative) ;
      - les ions sont reellement MOBILES : une perturbation ionique evolue dans
        le temps (le fond n'est plus un parametre constant du probleme).
    Remarque physique : la derive E x B du diocotron est a divergence nulle, donc
    une densite ionique parfaitement uniforme est un point fixe (elle reste
    uniforme). Pour exhiber la mobilite on ajoute donc un cas temoin ou les ions
    portent une petite perturbation et l'on mesure son evolution.

(B) "Especes heterogenes au runtime" : deux modeles de TAILLES D'ETAT
    differentes se composent dans la meme simulation :
      - "electron_euler"  : fluide d'Euler complet (4 variables : rho, q_x, q_y, E) ;
      - "ion_isothermal"  : fluide isotherme (3 variables : rho, q_x, q_y).
    On montre que des especes structurellement differentes coexistent et que
    chacune conserve sa masse.

Sorties : diagnostics numeriques imprimes (aucune dependance graphique).
Invariants verifies par assert (conservation de masse, potentiel non nul,
mobilite ionique). Le script imprime "OK simulation" en cas de succes.
"""

import numpy as np
import adc


# Tolerance sur la conservation de masse (advection conservative en C++).
MASS_TOL = 1e-10


def _meshgrid_centres(n, L):
    """Grille des centres de cellules : x_i = (i + 0.5)/n * L, row-major (n, n).

    Renvoie (X, Y) au format meshgrid 'xy' (X varie selon les colonnes, Y selon
    les lignes), coherent avec les densites 2D row-major attendues par adc.
    """
    coord = (np.arange(n) + 0.5) / n * L
    X, Y = np.meshgrid(coord, coord, indexing="xy")
    return X, Y


def partie_A():
    """Diocotron a ions mobiles : deux especes 'diocotron' de charges opposees."""
    print("== Partie (A) : diocotron a ions mobiles ==")

    cfg = adc.SimulationConfig()
    cfg.n = 48
    cfg.B0 = 1.0
    sim = adc.Simulation(cfg)

    # Composition espece par espece, a l'execution, cote Python uniquement.
    sim.add_species("electrons", "diocotron", -1.0)
    sim.add_species("ions", "diocotron", +1.0)
    assert sim.n_species() == 2, "deux especes attendues en partie A"
    print(f"  n_species              = {sim.n_species()}")

    n = sim.nx()
    L = cfg.L
    X, _ = _meshgrid_centres(n, L)

    # Densites initiales : electrons perturbes en cosinus, fond ionique uniforme.
    rho_e0 = 1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)
    rho_i0 = np.ones((n, n))
    sim.set_density("electrons", rho_e0.copy())
    sim.set_density("ions", rho_i0.copy())

    # Resolution des champs self-consistents : le potentiel doit etre non nul.
    sim.solve_fields()
    phi = sim.potential()
    phi_amp = float(np.max(np.abs(phi)))
    print(f"  |phi|_max (initial)    = {phi_amp:.6e}")
    assert phi_amp > 1e-8, "le potentiel self-consistent doit etre non nul"

    # Masses initiales par espece.
    m_e0 = sim.mass("electrons")
    m_i0 = sim.mass("ions")

    # Avancee en temps : 10 pas de dt = 0.002.
    sim.advance(0.002, 10)

    m_e1 = sim.mass("electrons")
    m_i1 = sim.mass("ions")
    de = abs(m_e1 - m_e0)
    di = abs(m_i1 - m_i0)
    print(f"  derive masse electrons = {de:.3e}")
    print(f"  derive masse ions      = {di:.3e}")
    assert de < MASS_TOL, "masse electronique non conservee"
    assert di < MASS_TOL, "masse ionique non conservee"

    # --- Temoin de MOBILITE ionique ---------------------------------------
    # Une densite ionique uniforme est un point fixe de la derive E x B (a
    # divergence nulle) : elle reste uniforme, ce qui ne prouve pas la mobilite.
    # On relance donc un cas ou les ions portent une petite perturbation et l'on
    # mesure son evolution : si elle bouge, l'espece ionique est bien INTEGREE
    # (et non un fond fige passe en parametre constant).
    sim2 = adc.Simulation(cfg)
    sim2.add_species("electrons", "diocotron", -1.0)
    sim2.add_species("ions", "diocotron", +1.0)
    _, Y2 = _meshgrid_centres(n, L)
    sim2.set_density("electrons", (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)).copy())
    sim2.set_density("ions", (1.0 + 0.05 * np.cos(2.0 * np.pi * Y2 / L)).copy())
    ion_avant = sim2.density("ions").copy()
    m_i2_0 = sim2.mass("ions")
    sim2.advance(0.002, 10)
    ion_apres = sim2.density("ions")
    bouge = float(np.max(np.abs(ion_apres - ion_avant)))
    di2 = abs(sim2.mass("ions") - m_i2_0)
    print(f"  evolution ions (temoin)= {bouge:.3e}  (fond ionique mobile)")
    print(f"  derive masse ions (T)  = {di2:.3e}")
    assert bouge > 1e-7, "les ions doivent etre mobiles (fond non fige)"
    assert di2 < MASS_TOL, "masse ionique non conservee (temoin)"


def partie_B():
    """Especes heterogenes au runtime : Euler (4 var) + isotherme (3 var)."""
    print("== Partie (B) : especes heterogenes au runtime ==")

    cfg = adc.SimulationConfig()
    cfg.n = 48
    sim = adc.Simulation(cfg)

    # Deux modeles de tailles d'etat differentes composes a l'execution.
    sim.add_species("electrons", "electron_euler", -1.0)  # 4 variables d'etat
    sim.add_species("ions", "ion_isothermal", +1.0)        # 3 variables d'etat
    assert sim.n_species() == 2, "deux especes attendues en partie B"
    print(f"  n_species              = {sim.n_species()}")

    n = sim.nx()
    L = cfg.L
    X, _ = _meshgrid_centres(n, L)

    # Densites initiales : electrons faiblement perturbes, ions uniformes.
    sim.set_density("electrons", (1.0 + 0.01 * np.cos(2.0 * np.pi * X / L)).copy())
    sim.set_density("ions", np.ones((n, n)))

    m_e0 = sim.mass("electrons")
    m_i0 = sim.mass("ions")

    # Avancee en temps : 6 pas de dt = 0.001.
    sim.advance(0.001, 6)

    m_e1 = sim.mass("electrons")
    m_i1 = sim.mass("ions")
    de = abs(m_e1 - m_e0)
    di = abs(m_i1 - m_i0)
    print(f"  derive masse electrons = {de:.3e}  (modele Euler, 4 var)")
    print(f"  derive masse ions      = {di:.3e}  (modele isotherme, 3 var)")
    assert de < MASS_TOL, "masse electronique (Euler) non conservee"
    assert di < MASS_TOL, "masse ionique (isotherme) non conservee"


def main():
    partie_A()
    partie_B()
    # Message recapitulatif : la composition s'est faite entierement cote Python.
    print("Composition multi-especes realisee depuis Python, sans toucher au C++.")
    print("OK simulation")


if __name__ == "__main__":
    main()
