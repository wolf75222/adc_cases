#!/usr/bin/env python3
"""Demo "composition_api" : composer un systeme multi-especes BLOC par BLOC.

Capacite demontree
------------------
C'est le niveau d'abstraction vise par le tuteur : depuis Python, l'utilisateur
COMPOSE son systeme un bloc d'equation a la fois, avec une API OBJET lisible, et
choisit pour CHAQUE bloc, independamment :

    sim = adc.System(n=48, gamma=1.4, cs2=0.5)
    sim.add_block("electrons", model="electron_euler", charge=-1.0,
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.IMEX(substeps=10))
    sim.add_block("ions", model="ion_isothermal", charge=+1.0,
                  spatial=adc.Spatial(minmod=True, flux="rusanov"),
                  time=adc.Explicit())
    sim.set_poisson(); sim.solve_fields(); sim.advance(dt, n)

soit, par bloc :
  - le MODELE physique     (electron_euler 4 var / ion_isothermal 3 var / diocotron 1 var) ;
  - la RECONSTRUCTION       spatiale (adc.Spatial : none / minmod / vanleer) ;
  - le FLUX numerique       (rusanov robuste / hllc onde de contact, Euler complet) ;
  - le TRAITEMENT temporel  (adc.Explicit = SSPRK2 / adc.IMEX = transport explicite +
                             source raide implicite ; adc.Implicit alias d'IMEX) ;
  - le nombre de SOUS-PAS   par macro-pas (p.ex. 10 sous-pas electrons : 1 ion).

Python dit QUOI assembler ; tout le calcul cellule par cellule (assemble_rhs<Limiter,
Flux>, Newton local de la source implicite, Poisson de systeme Sum_s q_s n_s) reste
en C++ compile et fige a l'ajout du bloc. Aucun callback Python dans le hot path :
chaque bloc embarque une fermeture d'avancee compilee, type-erased SEULEMENT au niveau
de la liste de blocs. SAUF si l'on ecrit soi-meme son integrateur temporel en Python
(partie D) via les primitives solve_fields / eval_rhs / get_state / set_state : le
schema en temps est alors en Python (par PAS), le calcul du residu et Poisson restant
en C++ (par CELLULE).

Ce que le script verifie
-------------------------
(A) Composition heterogene : electrons Euler (HLLC + VanLeer + IMEX + 10 sous-pas) et
    ions isothermes (Rusanov + Minmod + explicite + 1 sous-pas) coexistent ; chacun
    conserve sa masse ; le Poisson couple est actif (potentiel non nul) ; les electrons
    evoluent.
(B) Equivalence des raccourcis : add_species(...) == add_block(..., minmod, rusanov,
    Explicit(substeps=1)) au bit pres (memes densites apres avancee identique).
(C) Garde-fous : flux "hllc" demande a un modele non-Euler (diocotron, isotherme) leve
    une erreur claire ; un modele inconnu aussi ; substeps < 1 aussi.
(D) Integrateur temporel ECRIT EN PYTHON : on avance un bloc diocotron avec
    adc.integrate.ssprk2_step(sim, dt) (SSPRK2 Python, Poisson re-resolu per-stage) ;
    la masse reste conservee et l'etat fini ; on ecrit son propre take_step en Python
    tandis que le calcul par cellule reste en C++.

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

    # API objet : les champs de SystemConfig passent en kwargs de adc.System.
    sim = adc.System(n=48, gamma=1.4, cs2=0.5, periodic=True)

    # Electrons : Euler complet, reconstruction VanLeer, flux HLLC (onde de contact),
    # traitement IMEX (transport explicite + force electrostatique implicite), et
    # 10 SOUS-PAS par macro-pas (les electrons, plus raides, sont sous-cycles).
    sim.add_block("electrons", model="electron_euler", charge=-1.0,
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.IMEX(substeps=10))
    # Ions : fluide isotherme, Minmod + Rusanov, explicite, 1 sous-pas (plus lents).
    sim.add_block("ions", model="ion_isothermal", charge=+1.0,
                  spatial=adc.Spatial(minmod=True, flux="rusanov"),
                  time=adc.Explicit())
    assert sim.n_species() == 2
    print(f"  n_species              = {sim.n_species()}")
    print(f"  blocs                  = {sim.block_names()}")
    print("  electrons : electron_euler | Spatial(vanleer, hllc) | IMEX(substeps=10)")
    print("  ions      : ion_isothermal | Spatial(minmod, rusanov) | Explicit()")

    n, L = sim.nx(), 1.0
    X, _ = _meshgrid_centres(n, L)
    # Electrons perturbes (cos), ions uniformes : densite de charge non triviale, donc
    # potentiel non nul, mais charge nette ~ 0 -> Poisson periodique solvable.
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")
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
    """add_species(...) doit etre EXACTEMENT add_block(minmod, rusanov, Explicit, 1)."""
    print("== Partie (B) : add_species == raccourci de add_block (bit pour bit) ==")

    n, L = 32, 1.0
    X, _ = _meshgrid_centres(n, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)).copy()
    n_i0 = float(rho0.mean())  # fond neutralisant -> Poisson periodique solvable

    # Via add_species (raccourci).
    a = adc.System(n=n, periodic=True, n_i0=n_i0)
    a.add_species("e", "diocotron", -1.0)
    a.set_poisson()
    a.set_density("e", rho0.copy())
    a.advance(0.002, 12)
    da = a.density("e")

    # Via add_block avec les memes defauts, mais ecrits explicitement avec les objets.
    b = adc.System(n=n, periodic=True, n_i0=n_i0)
    b.add_block("e", model="diocotron", charge=-1.0,
                spatial=adc.Spatial(minmod=True, flux="rusanov"),
                time=adc.Explicit(substeps=1))
    b.set_poisson()
    b.set_density("e", rho0.copy())
    b.advance(0.002, 12)
    db = b.density("e")

    ecart = float(np.max(np.abs(da - db)))
    print(f"  ecart max add_species vs add_block = {ecart:.3e}")
    assert ecart == 0.0, "add_species doit etre le raccourci EXACT de add_block"


def partie_C():
    """Garde-fous : combinaisons invalides => erreur claire, pas un plantage."""
    print("== Partie (C) : garde-fous des combinaisons invalides ==")

    sim = adc.System(n=16)

    def doit_lever(fn, why):
        try:
            fn()
        except Exception as exc:  # pybind traduit std::runtime_error en RuntimeError
            print(f"  rejete ({why}) : {str(exc)[:70]}")
            return
        raise AssertionError(f"aurait du lever : {why}")

    # HLLC exige un modele Euler complet (4 var + pression) : diocotron et isotherme non.
    doit_lever(lambda: sim.add_block("d", model="diocotron", charge=-1.0,
                                     spatial=adc.Spatial(flux="hllc")),
               "hllc sur diocotron (1 var)")
    doit_lever(lambda: sim.add_block("i", model="ion_isothermal", charge=1.0,
                                     spatial=adc.Spatial(flux="hllc")),
               "hllc sur isotherme (3 var)")
    # Modele inconnu.
    doit_lever(lambda: sim.add_block("u", model="inconnu", charge=0.0),
               "modele inconnu")
    # Sous-pas invalide.
    doit_lever(lambda: sim.add_block("w", model="electron_euler", charge=-1.0,
                                     time=adc.Explicit(substeps=0)),
               "substeps < 1")
    assert sim.n_species() == 0, "aucun bloc invalide ne doit avoir ete ajoute"


def partie_D():
    """Integrateur temporel ECRIT EN PYTHON : take_step custom, calcul par cellule en C++."""
    print("== Partie (D) : integrateur temporel custom en Python (SSPRK2) ==")

    n, L = 32, 1.0
    X, Y = _meshgrid_centres(n, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L) * np.sin(2.0 * np.pi * Y / L)).copy()
    n_i0 = float(rho0.mean())  # fond neutralisant -> Poisson periodique solvable

    sim = adc.System(n=n, periodic=True, n_i0=n_i0)
    sim.add_species("e", "diocotron", -1.0)
    sim.set_poisson()
    sim.set_density("e", rho0.copy())

    # On n'appelle PAS sim.advance(...) : on ecrit la boucle en temps NOUS-MEMES.
    # adc.integrate.ssprk2_step assemble les etages RK en Python (par PAS) en
    # s'appuyant sur solve_fields / eval_rhs / get_state / set_state ; le residu
    # -div F + S et le Poisson de couplage restent calcules en C++ (par CELLULE).
    m0 = sim.mass("e")
    dt = 0.001
    nsteps = 20
    for _ in range(nsteps):
        adc.integrate.ssprk2_step(sim, dt)

    dm = abs(sim.mass("e") - m0)
    rho = sim.density("e")
    fini = bool(np.isfinite(rho).all())
    print(f"  pas Python (SSPRK2)    = {nsteps}  (Poisson re-resolu per-stage)")
    print(f"  derive masse           = {dm:.3e}  (integrateur ecrit en Python)")
    print(f"  etat fini              = {fini}")
    assert dm < 1e-9, "masse non conservee par l'integrateur Python"
    assert fini, "l'etat doit rester fini"


def main():
    partie_A()
    partie_B()
    partie_C()
    partie_D()
    print("Systeme compose bloc par bloc depuis Python ; calcul 100 % C++ compile.")
    print("Le schema en temps lui-meme peut etre ecrit en Python (partie D),")
    print("le calcul par cellule restant en C++.")
    print("OK composition_api")


if __name__ == "__main__":
    main()
