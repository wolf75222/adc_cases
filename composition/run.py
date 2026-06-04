#!/usr/bin/env python3
"""Demo "composition_api" : composer un systeme multi-especes BLOC par BLOC.

Capacite demontree
------------------
C'est le niveau d'abstraction vise par le tuteur : depuis Python, l'utilisateur
COMPOSE son systeme un bloc d'equation a la fois, avec une API OBJET lisible, et
choisit pour CHAQUE bloc, independamment :

    sim = adc.System(n=48, L=1.0, periodic=True)   # config = MAILLAGE seul
    sim.add_block("electrons", model=models.electron_euler(),
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.IMEX(substeps=10))
    sim.add_block("ions", model=models.ion_isothermal(),
                  spatial=adc.Spatial(minmod=True, flux="rusanov"),
                  time=adc.Explicit())
    sim.set_poisson(); sim.solve_fields(); sim.advance(dt, n)

Le MODELE physique n'est plus une chaine ("electron_euler") mais une COMPOSITION de
briques generiques de `adc` (etat / transport / source / elliptique), nommee cote
application dans `models.py` :

  - models.electron_euler() = Euler compressible + force du potentiel + densite de charge ;
  - models.ion_isothermal() = Euler isotherme + force du potentiel + densite de charge ;
  - models.diocotron()      = derive E x B d'un scalaire + fond neutralisant (Poisson).

La parametrisation physique (gamma, cs2, B0, charge, fond n_i0) vit DANS les briques du
modele, plus dans la config du systeme : `adc.System(...)` ne porte que le MAILLAGE
(n, L, periodic). On choisit ensuite, par bloc :
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
(B) Determinisme de la composition : un MEME modele diocotron compose DEUX fois,
    independamment (deux appels a models.diocotron), avec la meme CI numpy et la meme
    avancee, donne des densites identiques au bit pres (ecart == 0). La composition de
    briques est reproductible : memes briques -> meme calcul C++ fige.
(C) Garde-fous : flux "hllc" demande a un modele non-Euler (diocotron, transport
    scalaire) leve une erreur claire ; une source fluide (PotentialForce) posee sur un
    transport scalaire (Scalar) aussi ; un modele incoherent (Scalar + CompressibleFlux)
    est rejete des sa composition.
(D) Integrateur temporel ECRIT EN PYTHON : on avance un bloc diocotron avec
    adc.integrate.ssprk2_step(sim, dt) (SSPRK2 Python, Poisson re-resolu per-stage) ;
    la masse reste conservee et l'etat fini ; on ecrit son propre take_step en Python
    tandis que le calcul par cellule reste en C++.

Sorties : diagnostics numeriques imprimes (aucune dependance graphique). Invariants par
assert. Le script imprime "OK composition_api" en cas de succes.
"""

import os
import sys

import numpy as np
import adc

# Rend le depot importable si le paquet n'est pas installe (cf. adc_cases.ensure_importable).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (compositions de briques, cote application)
from adc_cases.common.grid import meshgrid_xy  # noqa: E402


MASS_TOL = 1e-10


def _meshgrid_centres(n, L):
    return meshgrid_xy(n, L)


def partie_A():
    """Composition heterogene : un schema numerique DIFFERENT par bloc."""
    print("== Partie (A) : un schema (modele/spatial/temps/sous-pas) par bloc ==")

    # Config = MAILLAGE seul : adc.System ne porte plus la physique (gamma, cs2, B0...).
    sim = adc.System(n=48, L=1.0, periodic=True)

    # Electrons : Euler complet, reconstruction VanLeer, flux HLLC (onde de contact),
    # traitement IMEX (transport explicite + force electrostatique implicite), et
    # 10 SOUS-PAS par macro-pas (les electrons, plus raides, sont sous-cycles).
    # La physique (gamma, charge) est portee par la composition models.electron_euler().
    sim.add_block("electrons", model=models.electron_euler(),
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.IMEX(substeps=10))
    # Ions : fluide isotherme, Minmod + Rusanov, explicite, 1 sous-pas (plus lents).
    sim.add_block("ions", model=models.ion_isothermal(),
                  spatial=adc.Spatial(minmod=True, flux="rusanov"),
                  time=adc.Explicit())
    assert sim.n_species() == 2
    print(f"  n_species              = {sim.n_species()}")
    print(f"  blocs                  = {sim.block_names()}")
    print("  electrons : electron_euler() | Spatial(vanleer, hllc) | IMEX(substeps=10)")
    print("  ions      : ion_isothermal() | Spatial(minmod, rusanov) | Explicit()")

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
    """Determinisme : un MEME modele compose deux fois donne le meme calcul (bit pour bit)."""
    print("== Partie (B) : determinisme de la composition de briques (bit pour bit) ==")

    n, L = 32, 1.0
    X, _ = _meshgrid_centres(n, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)).copy()
    n_i0 = float(rho0.mean())  # fond neutralisant -> Poisson periodique solvable

    def construire_et_avancer():
        # Le modele diocotron est RECOMPOSE a partir des briques generiques a chaque
        # appel : memes parametres (B0, alpha, fond n_i0) -> meme spec figee cote C++.
        s = adc.System(n=n, L=L, periodic=True)
        s.add_block("e", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
                    spatial=adc.Spatial(minmod=True, flux="rusanov"),
                    time=adc.Explicit(substeps=1))
        s.set_poisson()
        s.set_density("e", rho0.copy())
        s.advance(0.002, 12)
        return s.density("e")

    da = construire_et_avancer()  # premiere composition independante
    db = construire_et_avancer()  # seconde composition, strictement identique

    ecart = float(np.max(np.abs(da - db)))
    print(f"  ecart max (deux compositions independantes) = {ecart:.3e}")
    assert ecart == 0.0, "la composition de briques doit etre deterministe (bit pour bit)"


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

    # HLLC exige un transport compressible (4 var + pression) : diocotron (transport
    # scalaire ExB) ne peut pas l'utiliser -> rejet a l'ajout du bloc.
    doit_lever(lambda: sim.add_block(
        "d", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=0.0),
        spatial=adc.Spatial(flux="hllc")),
        "hllc sur diocotron (transport scalaire)")

    # Source fluide (PotentialForce) posee sur un transport scalaire (Scalar + ExB) :
    # incoherent (la force du potentiel agit sur une quantite de mouvement fluide) ->
    # rejet a l'ajout du bloc. Le modele se COMPOSE (state<->transport coherents) mais
    # la source est invalide pour ce transport.
    source_fluide_sur_scalaire = adc.Model(
        state=adc.Scalar(), transport=adc.ExB(B0=1.0),
        source=adc.PotentialForce(charge=-1.0),
        elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
    doit_lever(lambda: sim.add_block(
        "s", model=source_fluide_sur_scalaire, spatial=adc.Spatial(minmod=True)),
        "source PotentialForce sur transport scalaire")

    # Modele incoherent des la COMPOSITION : un etat scalaire exige un transport ExB,
    # pas un flux compressible -> adc.Model(...) leve directement.
    doit_lever(lambda: adc.Model(
        state=adc.Scalar(), transport=adc.CompressibleFlux(),
        source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
        "modele incoherent (Scalar + CompressibleFlux)")

    assert sim.n_species() == 0, "aucun bloc invalide ne doit avoir ete ajoute"


def partie_D():
    """Integrateur temporel ECRIT EN PYTHON : take_step custom, calcul par cellule en C++."""
    print("== Partie (D) : integrateur temporel custom en Python (SSPRK2) ==")

    n, L = 32, 1.0
    X, Y = _meshgrid_centres(n, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L) * np.sin(2.0 * np.pi * Y / L)).copy()
    n_i0 = float(rho0.mean())  # fond neutralisant -> Poisson periodique solvable

    sim = adc.System(n=n, L=L, periodic=True)
    sim.add_block("e", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
                  spatial=adc.Spatial(minmod=True), time=adc.Explicit())
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
    print("Systeme compose bloc par bloc depuis Python (modeles = compositions de briques) ;")
    print("calcul 100 % C++ compile. Le schema en temps lui-meme peut etre ecrit en Python")
    print("(partie D), le calcul par cellule restant en C++.")
    print("OK composition_api")


if __name__ == "__main__":
    main()
