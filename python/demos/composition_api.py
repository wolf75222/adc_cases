#!/usr/bin/env python3
"""Demo "composition_api" : composer un systeme multi-especes BLOC par BLOC.

Capacite demontree
------------------
C'est le niveau d'abstraction vise par le tuteur : depuis Python, l'utilisateur
COMPOSE son systeme un bloc d'equation a la fois, et choisit pour CHAQUE bloc,
independamment :

    sim.add_block(name="electrons", model="electron_euler", charge=-1.0,
                  limiter="vanleer", flux="hllc", time="imex", substeps=10)
    sim.add_block(name="ions",      model="ion_isothermal", charge=+1.0,
                  limiter="minmod", flux="rusanov", time="explicit", substeps=1)
    sim.solve_fields(); sim.advance(dt, n)

soit, par bloc :
  - le MODELE physique     (electron_euler 4 var / ion_isothermal 3 var / diocotron 1 var) ;
  - la RECONSTRUCTION       spatiale (limiter : none / minmod / vanleer) ;
  - le FLUX numerique       (rusanov robuste / hllc onde de contact, Euler complet) ;
  - le TRAITEMENT temporel  (explicit = SSPRK2 / imex = transport explicite + source implicite) ;
  - le nombre de SOUS-PAS   par macro-pas (p.ex. 10 sous-pas electrons : 1 ion).

Python dit QUOI assembler ; tout le calcul cellule par cellule (assemble_rhs<Limiter,
Flux>, Newton local de la source implicite, Poisson de systeme Sum_s q_s n_s) reste
en C++ compile et fige a l'ajout du bloc. Aucun callback Python dans le hot path :
chaque bloc embarque une fermeture d'avancee compilee, type-erased SEULEMENT au niveau
de la liste de blocs.

Ce que le script verifie
-------------------------
(A) Composition heterogene : electrons Euler (HLLC + VanLeer + IMEX + 10 sous-pas) et
    ions isothermes (Rusanov + Minmod + explicite + 1 sous-pas) coexistent ; chacun
    conserve sa masse ; le Poisson couple est actif (potentiel non nul).
(B) Equivalence des raccourcis : add_species(...) == add_block(..., minmod, rusanov,
    explicit, 1) au bit pres (memes densites apres avancee identique).
(C) Garde-fous : flux "hllc" demande a un modele non-Euler (diocotron, isotherme) leve
    une erreur claire ; un limiter / flux / time inconnu aussi.

Sorties : diagnostics numeriques imprimes (aucune dependance graphique). Invariants par
assert. Le script imprime "OK composition_api" en cas de succes.
"""

import numpy as np
import adc


MASS_TOL = 1e-10


def _meshgrid_centres(n, L):
    coord = (np.arange(n) + 0.5) / n * L
    return np.meshgrid(coord, coord, indexing="xy")


def partie_A():
    """Composition heterogene : un schema numerique DIFFERENT par bloc."""
    print("== Partie (A) : un schema (modele/spatial/temps/sous-pas) par bloc ==")

    cfg = adc.SimulationConfig()
    cfg.n = 48
    cfg.gamma = 1.4
    cfg.cs2 = 0.5
    sim = adc.Simulation(cfg)

    # Electrons : Euler complet, reconstruction VanLeer, flux HLLC (onde de contact),
    # traitement IMEX (transport explicite + force electrostatique implicite), et
    # 10 SOUS-PAS par macro-pas (les electrons, plus raides, sont sous-cycles).
    sim.add_block(name="electrons", model="electron_euler", charge=-1.0,
                  limiter="vanleer", flux="hllc", time="imex", substeps=10)
    # Ions : fluide isotherme, Minmod + Rusanov, explicite, 1 sous-pas (plus lents).
    sim.add_block(name="ions", model="ion_isothermal", charge=+1.0,
                  limiter="minmod", flux="rusanov", time="explicit", substeps=1)
    assert sim.n_species() == 2
    print(f"  n_species              = {sim.n_species()}")
    print("  electrons : electron_euler | vanleer + hllc | imex | substeps=10")
    print("  ions      : ion_isothermal | minmod  + rusanov | explicit | substeps=1")

    n, L = sim.nx(), cfg.L
    X, _ = _meshgrid_centres(n, L)
    sim.set_density("electrons", (1.0 + 0.02 * np.cos(2.0 * np.pi * X / L)).copy())
    sim.set_density("ions", np.ones((n, n)))

    sim.solve_fields()
    phi_amp = float(np.max(np.abs(sim.potential())))
    print(f"  |phi|_max (initial)    = {phi_amp:.6e}")
    assert phi_amp > 1e-8, "Poisson couple inactif"

    m_e0, m_i0 = sim.mass("electrons"), sim.mass("ions")
    rho_e0 = sim.density("electrons").copy()
    sim.advance(0.001, 8)
    de = abs(sim.mass("electrons") - m_e0)
    di = abs(sim.mass("ions") - m_i0)
    bouge_e = float(np.max(np.abs(sim.density("electrons") - rho_e0)))
    print(f"  derive masse electrons = {de:.3e}  (Euler/HLLC/IMEX, 10 sous-pas)")
    print(f"  derive masse ions      = {di:.3e}  (isotherme/Rusanov/explicite)")
    print(f"  evolution electrons    = {bouge_e:.3e}  (dynamique non triviale)")
    assert de < MASS_TOL, "masse electronique non conservee"
    assert di < MASS_TOL, "masse ionique non conservee"
    assert bouge_e > 1e-9, "les electrons devraient evoluer"


def partie_B():
    """add_species(...) doit etre EXACTEMENT add_block(minmod, rusanov, explicit, 1)."""
    print("== Partie (B) : add_species == raccourci de add_block (bit pour bit) ==")

    cfg = adc.SimulationConfig()
    cfg.n = 32
    n, L = cfg.n, cfg.L
    X, _ = _meshgrid_centres(n, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)).copy()

    # Via add_species (raccourci).
    a = adc.Simulation(cfg)
    a.add_species("e", "diocotron", -1.0)
    a.set_density("e", rho0.copy())
    a.advance(0.002, 12)
    da = a.density("e")

    # Via add_block avec les memes defauts explicites.
    b = adc.Simulation(cfg)
    b.add_block(name="e", model="diocotron", charge=-1.0,
                limiter="minmod", flux="rusanov", time="explicit", substeps=1)
    b.set_density("e", rho0.copy())
    b.advance(0.002, 12)
    db = b.density("e")

    ecart = float(np.max(np.abs(da - db)))
    print(f"  ecart max add_species vs add_block = {ecart:.3e}")
    assert ecart == 0.0, "add_species doit etre le raccourci EXACT de add_block"


def partie_C():
    """Garde-fous : combinaisons invalides => erreur claire, pas un plantage."""
    print("== Partie (C) : garde-fous des combinaisons invalides ==")

    cfg = adc.SimulationConfig()
    cfg.n = 16
    sim = adc.Simulation(cfg)

    def doit_lever(fn, why):
        try:
            fn()
        except Exception as exc:  # pybind traduit std::runtime_error en RuntimeError
            print(f"  rejete ({why}) : {str(exc)[:70]}")
            return
        raise AssertionError(f"aurait du lever : {why}")

    # HLLC exige un modele Euler complet (4 var + pression) : diocotron et isotherme non.
    doit_lever(lambda: sim.add_block(name="d", model="diocotron", charge=-1.0,
                                     flux="hllc"), "hllc sur diocotron (1 var)")
    doit_lever(lambda: sim.add_block(name="i", model="ion_isothermal", charge=1.0,
                                     flux="hllc"), "hllc sur isotherme (3 var)")
    # Tags inconnus.
    doit_lever(lambda: sim.add_block(name="x", model="electron_euler", charge=-1.0,
                                     limiter="weno"), "limiter inconnu")
    doit_lever(lambda: sim.add_block(name="y", model="electron_euler", charge=-1.0,
                                     flux="roe"), "flux inconnu")
    doit_lever(lambda: sim.add_block(name="z", model="electron_euler", charge=-1.0,
                                     time="rk4"), "time inconnu")
    doit_lever(lambda: sim.add_block(name="w", model="electron_euler", charge=-1.0,
                                     substeps=0), "substeps < 1")
    doit_lever(lambda: sim.add_block(name="u", model="inconnu", charge=0.0),
               "modele inconnu")
    assert sim.n_species() == 0, "aucun bloc invalide ne doit avoir ete ajoute"


def main():
    partie_A()
    partie_B()
    partie_C()
    print("Systeme compose bloc par bloc depuis Python ; calcul 100 % C++ compile.")
    print("OK composition_api")


if __name__ == "__main__":
    main()
