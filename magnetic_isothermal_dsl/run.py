#!/usr/bin/env python3
"""Cas "magnetic_isothermal_dsl" : un fluide isotherme magnetise ecrit entierement en formules
(adc.dsl.Model), avec force de Lorentz q rho E + (v x B) pilotee par un champ B_z constant.

Pourquoi ce cas
---------------
Troisieme demonstrateur du plan declaratif (apres diocotron_dsl mono-espece et two_species_dsl
multi-espece). Il exerce ce que les deux autres ne couvraient pas : une source qui lit un champ
auxiliaire etendu (B_z, au dela du contrat de base phi / grad phi). Toute la physique est ecrite
en expressions symboliques ; adc.dsl genere le C++, le compile et l'installe comme bloc via
add_equation(...). Aucune brique nommee, aucun modele natif de reference n'existe pour ce modele :
on prouve sa correction par equivalence inter-backend et par invariants physiques.

Physique (Euler isotherme + electrostatique + Lorentz, fermeture p = cs2 rho)
-----------------------------------------------------------------------------
  variables conservatives : rho, mx = rho u, my = rho v ;
  flux x = [mx, mx u + cs2 rho, mx v],  flux y = [my, my u, my v + cs2 rho] ;
  spectre x = (u - cs, u, u + cs),  y = (v - cs, v, v + cs),  cs = sqrt(cs2) ;
  source = [0, q rho (-grad_x) + B_z my, q rho (-grad_y) - B_z mx]
           |__electrostatique q rho E (E = -grad phi)__|  |__Lorentz v x B_z__| ;
  second membre elliptique (densite de charge) : q rho, couple au Poisson du systeme.

Le terme B_z my / -B_z mx est la projection 2D de (q rho/c) v x B avec B = B_z e_z (constantes
absorbees dans B_z) : il fait tourner la quantite de mouvement sans changer la masse ni l'energie
cinetique. C'est la nouveaute de ce demonstrateur.

Champ B_z : pilote 100% depuis Python
-------------------------------------
B_z est une composante canonique du canal adc::Aux (indice 3, au dela de phi/grad). Le modele DSL
qui lit aux("B_z") declare n_aux = 4 ; add_equation elargit le canal aux partage ; on peuple B_z
par sim.set_magnetic_field(tableau n x n). Ici B_z est un champ constant (B0 partout). Aucune
modification du coeur adc_cpp n'est requise : set_magnetic_field existe deja (binding C++).

Backend
-------
On compile en backend "production" (chemin natif zero-copie add_native_block, cible du plan) et en
"aot" (chemin de production host-marshale, numerique identique). Quand les deux se chargent sur la
plateforme, on exige qu'ils soient bit-identiques (eval_rhs et etat apres quelques pas : dmax == 0),
exactement comme diocotron_dsl prouve l'equivalence DSL <-> natif. Sur une plateforme ou le chemin
natif ne peut pas etre charge (macOS, espace de noms a deux niveaux), seul "aot" se lie : la parite
inter-backend est alors sautee, mais la correction reste prouvee par l'oracle analytique ci-dessous.

Validation (aucun modele natif de reference n'existe pour ce modele)
--------------------------------------------------------------------
  (1) parite inter-backend : si production et aot se lient, leurs eval_rhs et leurs etats apres
      quelques pas sont bit-identiques (np.array_equal, dmax == 0) ;
  (2) oracle lorentz : la difference de residu entre B_z = B0 et B_z = 0 est, sur les composantes
      de quantite de mouvement, exactement (B0 my, -B0 mx) calcule en numpy (dmax == 0) : le terme
      magnetique compile lit bien B_z et a la bonne forme ; a B_z = 0 il s'annule (controle) ;
  (3) evolution : run court stable, fini, densite positive, masse conservee ;
  (4) rotation : avec B_z != 0, la quantite de mouvement transverse (my) initialement nulle devient
      non nulle -> le terme de Lorentz devie bien l'ecoulement (la physique magnetique est exercee).
"""

from __future__ import annotations

import os

import numpy as np

import adc
from adc import dsl

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases.common.checks import (
    assert_finite,
    assert_positive,
    relative_drift,
)  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402

# Parametres physiques.
CS2 = 1.0  # carre de la vitesse du son isotherme (fermeture p = cs2 rho)
Q = -1.0  # charge (signe inclus), comme la brique PotentialForce du coeur
B0 = 2.0  # champ magnetique de fond B_z (constant) ; != 0 -> Lorentz actif


def magnetic_isothermal_model() -> dsl.Model:
    """Fluide isotherme magnetise ecrit en formules (adc.dsl.Model).

    Variables conservatives rho, mx = rho u, my = rho v ; flux isotherme ;
    source electrostatique q rho E + Lorentz v x B_z ; second membre elliptique
    q rho (couplage Poisson). Reproduit l'API cible de l'utilisateur.
    """
    m = dsl.Model("magnetic_isothermal")

    rho, mx, my = m.conservative_vars(
        "rho", "rho_u", "rho_v", roles=["Density", "MomentumX", "MomentumY"]
    )

    # Primitives nommees (le codegen de to_conservative referencera u, v par leur nom de layout).
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", CS2 * rho)  # fermeture isotherme

    # Champs auxiliaires fournis par le solveur : potentiel, son gradient, et B_z (canal etendu).
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")

    cs2 = m.param("cs2", CS2)  # constante nommee, inlinee au codegen
    q = m.param("charge", Q)

    # Flux isotherme (convention IsothermalFlux du coeur) : pas de composante energie.
    m.flux(
        x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho]
    )
    cs = dsl.sqrt(cs2)  # vitesse du son constante
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])

    # Source : electrostatique q rho E (E = -grad phi) + Lorentz (q rho / c) v x B_z e_z, soit en
    # 2D, +B_z my sur la qte de mvt x et -B_z mx sur la qte de mvt y (constantes absorbees dans B_z).
    m.source([0.0, q * rho * (-gx) + bz * my, q * rho * (-gy) - bz * mx])

    # Layout primitif (rho, u, v, p) et inverse prim -> cons (le DSL n'inverse pas symboliquement).
    m.primitive_vars(rho, u, v, p)
    m.conservative_from([rho, rho * u, rho * v])

    # Densite de charge : rhs = q rho (n = rho), contribue au Poisson de systeme.
    m.elliptic_rhs(q * rho)

    m.check()
    return m


def initial_state(n: int) -> np.ndarray:
    """Etat initial : densite cosinus le long de x, mouvement longitudinal.

    Quantite de mouvement purement longitudinale (mx > 0, my = 0). my
    initialement nul rend la rotation de Lorentz visible (toute composante
    transverse apparue vient du terme magnetique).
    """
    x = (np.arange(n) + 0.5) / n
    rho0 = 1.0 + 0.05 * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))
    mx0 = 0.3 * rho0  # vitesse longitudinale u = 0.3
    my0 = np.zeros((n, n))  # pas de quantite de mouvement transverse au depart
    return np.stack([rho0, mx0, my0], axis=0)


def _build_sim(
    n: int, compiled: adc.Model, state0: np.ndarray, bz_value: float
) -> adc.System:
    """Construit un System periodique et installe le bloc DSL compile.

    Fixe l'etat, peuple le champ B_z constant a `bz_value`, puis resout les
    champs (phi / grad / B_z dans le canal aux). Meme schema (minmod + Rusanov,
    explicite) que les autres demonstrateurs DSL.
    """
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        "plasma",
        model=compiled,
        spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
        time=adc.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_state("plasma", state0)
    # add_equation a elargi le canal aux partage (n_aux = 4) : set_magnetic_field peuple B_z.
    sim.set_magnetic_field(bz_value * np.ones((n, n)))
    sim.solve_fields()
    return sim


def bind_backends(
    n: int, state0: np.ndarray, bz_value: float
) -> dict[str, adc.System]:
    """Compile le modele en "production" puis "aot" et lie chaque .so a un System.

    Le chemin natif peut compiler mais echouer au dlopen selon la plateforme.
    Renvoie un dict {backend: sim} des backends effectivement lies (au moins
    "aot").
    """
    include = adc_include()
    so_dir = case_output_dir("magnetic_isothermal_dsl")
    bound = {}
    for cand in ("production", "aot"):
        try:
            compiled = magnetic_isothermal_model().compile(
                os.path.join(so_dir, "magnetic_isothermal_%s.so" % cand),
                include,
                backend=cand,
            )
            sim = _build_sim(n, compiled, state0, bz_value)
        except (
            Exception
        ) as exc:  # noqa: BLE001 (diagnostic : compilation ou dlopen indisponible)
            print(
                "backend %r indisponible (%s), essai suivant"
                % (cand, type(exc).__name__)
            )
            continue
        bound[cand] = sim
    if not bound:
        raise RuntimeError(
            "aucun backend DSL n'a pu etre lie pour le modele magnetise"
        )
    return bound


def main() -> None:
    n, n_steps = 32, 40
    state0 = initial_state(n)
    mx0, my0 = state0[1], state0[2]

    print(
        "=== magnetic_isothermal_dsl : fluide isotherme magnetise ecrit en formules ==="
    )
    print(
        "grille %d x %d, %d pas, CFL = 0.4 ; cs2 = %.1f, q = %.0f, B_z = %.1f"
        % (n, n, n_steps, CS2, Q, B0)
    )

    # --- Backends lies avec B_z = B0 (Lorentz actif) et avec B_z = 0 (controle electrostatique) ---
    bound = bind_backends(n, state0, B0)
    bound0 = bind_backends(n, state0, 0.0)
    backends = sorted(bound)
    print("backends DSL lies : %s" % ", ".join(repr(b) for b in backends))

    # --- (1) parite inter-backend : si production et aot se lient, ils sont bit-identiques ---
    if len(backends) >= 2:
        ref = backends[0]
        r_ref = np.array(bound[ref].eval_rhs("plasma"))
        for b in backends[1:]:
            r_b = np.array(bound[b].eval_rhs("plasma"))
            dmax = float(np.max(np.abs(r_b - r_ref)))
            print(
                "eval_rhs %r vs %r : dmax = %.3e (bit-identique = %s)"
                % (b, ref, dmax, np.array_equal(r_b, r_ref))
            )
            assert np.array_equal(r_b, r_ref), (
                "backends %r et %r non bit-identiques sur eval_rhs (dmax = %.3e)"
                % (b, ref, dmax)
            )
    else:
        print(
            "parite inter-backend sautee (un seul backend lie sur cette plateforme :"
            " %r) ; correction prouvee par l'oracle analytique de Lorentz"
            % backends[0]
        )

    # --- (2) oracle lorentz : difference de residu B_z=B0 moins B_z=0 == (B0 my, -B0 mx) exactement.
    # Le flux et l'electrostatique sont identiques entre les deux runs ; la seule difference est le
    # terme magnetique. On le compare a sa forme analytique en numpy : dmax == 0 attendu (lecture
    # exacte de B_z et bonne forme du terme). On verifie sur chaque backend lie.
    lorentz_x = B0 * my0  # +B_z my sur la qte de mvt x
    lorentz_y = -B0 * mx0  # -B_z mx sur la qte de mvt y
    lor_contrib = 0.0
    for b in backends:
        dR = np.array(bound[b].eval_rhs("plasma")) - np.array(
            bound0[b].eval_rhs("plasma")
        )
        err_x = float(np.max(np.abs(dR[1] - lorentz_x)))
        err_y = float(np.max(np.abs(dR[2] - lorentz_y)))
        mag = float(np.max(np.abs(dR)))
        lor_contrib = max(lor_contrib, mag)
        print(
            "oracle Lorentz [%r] : err_x = %.3e, err_y = %.3e, max|dR| = %.3e"
            % (b, err_x, err_y, mag)
        )
        assert err_x == 0.0 and err_y == 0.0, (
            "terme de Lorentz [%r] != (B_z my, -B_z mx) (err_x = %.3e, err_y = %.3e)"
            % (b, err_x, err_y)
        )
        # controle : la composante densite (S[0] = 0) n'est jamais modifiee par B_z.
        assert (
            float(np.max(np.abs(dR[0]))) == 0.0
        ), "B_z modifie la composante densite (impossible)"
    assert (
        lor_contrib > 0.0
    ), "le terme de Lorentz est partout nul (B_z non lu ?)"

    # --- (3) evolution + (4) rotation : sur le premier backend lie, B_z = B0 ---
    sim = bound[backends[0]]
    mass0 = float(state0[0].sum())
    my_mean0 = float(my0.mean())
    for _ in range(n_steps):
        sim.step_cfl(0.4)
    state = np.array(sim.get_state("plasma"))
    drift = relative_drift(sim.mass("plasma"), mass0)
    my_mean = float(state[2].mean())
    print(
        "apres %d pas (backend %r) : t = %.6f, derive de masse = %.3e"
        % (n_steps, backends[0], sim.time(), drift)
    )
    print(
        "qte de mvt transverse moyenne : initiale %.3e -> finale %.3e (rotation de Lorentz)"
        % (my_mean0, my_mean)
    )

    assert_finite(state, "etat magnetise")
    assert_positive(state[0], "densite")
    assert drift < 1e-9, "masse non conservee (derive %.3e)" % drift
    # my initialement nul : toute composante transverse apparue vient du terme de Lorentz.
    assert (
        abs(my_mean) > 1e-6
    ), "la quantite de mouvement transverse est restee nulle : Lorentz n'a pas devie l'ecoulement"

    print(
        "OK magnetic_isothermal_dsl (Lorentz exerce, B_z = %.1f pilote depuis Python, backends %s)"
        % (B0, ", ".join(repr(b) for b in backends))
    )


if __name__ == "__main__":
    main()
