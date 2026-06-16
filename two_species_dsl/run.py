#!/usr/bin/env python3
"""Cas "two_species_dsl" : deux especes DSL couplees par un meme Poisson.

Deux especes ecrites entierement en formules (adc.dsl.Model), couplees par un
meme Poisson, prouvees equivalentes a la composition native de briques.

Pourquoi ce cas
---------------
Le cas "multispecies" couple deux fluides heterogenes en composant des briques natives :
electrons (Euler compressible, 4 var) + ions (Euler isotherme, 3 var), chacun avec une force
electrostatique (PotentialForce) et une densite de charge (ChargeDensity), relies par un seul
Poisson de systeme (rhs = q_e n_e + q_i n_i). Ici, on ecrit la meme physique des deux especes
sans aucune brique nommee : variables, primitives, flux, valeurs propres, source (Lorentz/
electrostatique q rho E avec E = -grad phi) et second membre elliptique (q n) sont des expressions
symboliques. Chaque espece est compilee par adc.dsl et installee via add_equation(...).

Capacite demontree (au dela du diocotron mono-espece)
-----------------------------------------------------
  - un modele DSL a source (la force electrostatique lit grad phi via le canal aux) ;
  - deux especes DSL de tailles d'etat differentes (4 et 3 variables) dans le meme System ;
  - un Poisson couple dont le RHS agrege les charges des deux blocs DSL (rhs = "charge_density").

Le traitement temporel est explicite (SSPRK2) pour les deux blocs, A L'identique du natif, pour
que l'equivalence soit testable. add_equation accepte par ailleurs un time/substeps par bloc
(adc.Explicit(substeps=k) / adc.IMEX), mais un sous-cyclage different romprait la comparaison
bit-a-bit avec le natif (qui avance les deux blocs au meme pas).

Conventions reproduites (ancrees dans le coeur adc_cpp)
-------------------------------------------------------
  - Euler compressible (include/adc/physics/euler.hpp) : p = (g-1)(E - 1/2 rho|v|^2),
        flux x = [rho u, rho u^2 + p, rho u v, (E+p) u], idem y ; spectre (u-c, u, u, u+c).
  - Euler isotherme (include/adc/physics/hyperbolic.hpp, IsothermalFlux) : p = cs2 rho,
        flux x = [rho u, rho u^2 + p, rho u v], idem y ; spectre (u-c, u, u+c), c = sqrt(cs2).
  - Force electrostatique (include/adc/physics/source.hpp, PotentialForce) : E = (-grad_x, -grad_y),
        s = [0, q rho E_x, q rho E_y, q (rho u E_x + rho v E_y)] (la composante energie si 4 var).
  - Densite de charge (include/adc/physics/elliptic.hpp, ChargeDensity) : rhs = q n (n = U[0]).

Equivalence
-----------
Le residu et le flux de chaque espece DSL sont bit-identiques au natif (verifie a 1 pas :
np.array_equal). Sur plusieurs pas couples, l'etat des electrons derive d'un epsilon machine
(~1e-30) du natif : la seule difference est une reassociation flottante dans l'accumulation du
second membre de Poisson partage (deux blocs y contribuent). Ce n'est pas un ecart de physique ;
on l'asserte donc avec une tolerance serree (1e-24), tandis que les ions restent bit-identiques.

Backend : "production" (natif zero-copie) si disponible, sinon "aot" (numerique identique).

Invariants verifies (assert)
----------------------------
  - equivalence : etat DSL ~ etat natif (electrons a 1e-24, ions bit-identique) par espece ;
  - conservation de la masse par espece (transport advectif) ;
  - finitude et positivite des densites.
"""

from __future__ import annotations

import numpy as np

import adc
from adc import dsl

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import (
    models,
)  # noqa: E402  (composition native de briques, oracle de reference)
from adc_cases.common.checks import (
    assert_finite,
    assert_positive,
    relative_drift,
)  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402

# Parametres physiques partages (DSL et natif) : ils doivent coincider pour que l'equivalence tienne.
GAMMA_E = 5.0 / 3.0  # electrons : Euler compressible adiabatique
CS2_I = 1.0  # ions : fermeture isotherme p = cs2 rho
Q_E, Q_I = (
    -1.0,
    1.0,
)  # charges (signe inclus) ; q/m = charge dans le coeur (PotentialForce.qom)


def electron_dsl_model() -> dsl.Model:
    """Electrons : Euler 4 var + force electrostatique + densite de charge.

    Ecrit entierement en formules, reproduit a l'identique les briques Euler /
    PotentialForce / ChargeDensity du coeur.
    """
    m = dsl.Model("electron_euler")
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")
    g = m.param(
        "gamma", GAMMA_E
    )  # nomme : inline au codegen + set_gamma (metadonnee ABI coherente)

    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    c = dsl.sqrt(g * p / rho)  # vitesse du son

    # Flux convectif d'Euler (convention euler.hpp) : energie (E + p) v_n.
    m.flux(
        x=[rhou, rhou * u + p, rhou * v, (E + p) * u],
        y=[rhov, rhov * u, rhov * v + p, (E + p) * v],
    )
    m.eigenvalues(x=[u - c, u, u, u + c], y=[v - c, v, v, v + c])

    # Layout primitif (rho, u, v, p) par la forme positionnelle (les primitives sont deja definies
    # ci-dessus ; on ne fait que fixer l'ordre de Prim, sans les redefinir).
    m.primitive_vars(rho, u, v, p)
    m.conservative_from(
        [rho, rho * u, rho * v, p / (g - 1.0) + 0.5 * rho * (u * u + v * v)]
    )

    # Force electrostatique (q/m) rho E, E = -grad phi (convention PotentialForce, 4 var = + travail).
    e_x = -grad_x
    e_y = -grad_y
    m.source(
        [0.0, Q_E * rho * e_x, Q_E * rho * e_y, Q_E * (rhou * e_x + rhov * e_y)]
    )
    # Densite de charge : rhs = q n (n = rho), contribue au Poisson de systeme.
    m.elliptic_rhs(Q_E * rho)

    m.check()
    return m


def ion_dsl_model() -> dsl.Model:
    """Ions : Euler isotherme 3 var + force electrostatique + densite de charge.

    Ecrit entierement en formules, reproduit a l'identique les briques
    IsothermalFlux / PotentialForce / ChargeDensity.
    """
    m = dsl.Model("ion_isothermal")
    rho, rhou, rhov = m.conservative_vars("rho", "rho_u", "rho_v")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")
    cs2 = m.param("cs2", CS2_I)

    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", cs2 * rho)  # fermeture isotherme
    c = dsl.sqrt(cs2)  # vitesse du son constante

    m.flux(x=[rhou, rhou * u + p, rhou * v], y=[rhov, rhov * u, rhov * v + p])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])

    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])

    e_x = -grad_x
    e_y = -grad_y
    m.source(
        [0.0, Q_I * rho * e_x, Q_I * rho * e_y]
    )  # 3 var : pas de composante energie
    m.elliptic_rhs(Q_I * rho)

    m.check()
    return m


def initial_conditions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Separation de charge : electrons en cosinus le long de x, ions uniformes.

    Les deux densites different localement -> RHS de Poisson
    f = q_e n_e + q_i n_i non nul.
    """
    x = (
        np.arange(n) + 0.5
    ) / n  # x = (i + 0.5)/n le long de l'axe des colonnes
    ne = 1.0 + 0.02 * np.cos(2.0 * np.pi * x)
    ne2d = np.broadcast_to(ne, (n, n)).copy()
    ni2d = np.ones((n, n))
    return ne2d, ni2d


def run_native(
    n: int, ne2d: np.ndarray, ni2d: np.ndarray, n_steps: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Reference : composition native (models.electron_euler / ion_isothermal)."""
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block(
        "electrons",
        model=models.electron_euler(charge=Q_E, gamma=GAMMA_E),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    sim.add_block(
        "ions",
        model=models.ion_isothermal(charge=Q_I, cs2=CS2_I),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("electrons", ne2d)
    sim.set_density("ions", ni2d)
    for _ in range(n_steps):
        sim.step_cfl(0.4)
    return (
        np.asarray(sim.get_state("electrons")),
        np.asarray(sim.get_state("ions")),
        sim.mass("electrons"),
        sim.mass("ions"),
    )


def _compile(model: dsl.Model, tag: str, backend: str):
    """Compile @p model avec le backend demande, sans fallback interne.

    Le fallback est gere au niveau de run_dsl pour que les deux especes
    partagent le meme backend. Renvoie le CompiledModel.
    """
    import os

    include = adc_include()
    so_dir = case_output_dir("two_species_dsl")
    return model.compile(
        os.path.join(so_dir, "%s_%s.so" % (tag, backend)),
        include,
        backend=backend,
    )


def run_dsl(
    n: int, ne2d: np.ndarray, ni2d: np.ndarray, n_steps: int
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Le meme systeme, mais les deux blocs sont des modeles DSL compiles.

    Meme schema et meme temps que le natif. On prefere "production" (natif) ; si
    le chemin natif echoue (compilation ou branchement add_native_block, la garde
    d'ABI se declenche au branchement, pas a la compilation), on retombe sur "aot"
    pour les deux especes a la fois (un backend coherent par run). Renvoie
    (etat_e, etat_i, masse_e, masse_i, backend).
    """
    last = None
    for backend in ("production", "aot"):
        try:
            ce = _compile(electron_dsl_model(), "electron", backend)
            ci = _compile(ion_dsl_model(), "ion", backend)
            sim = adc.System(n=n, L=1.0, periodic=True)
            sim.add_equation(
                "electrons",
                model=ce,
                spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                time=adc.Explicit(),
            )
            sim.add_equation(
                "ions",
                model=ci,
                spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                time=adc.Explicit(),
            )
            sim.set_poisson(rhs="charge_density", solver="geometric_mg")
            sim.set_density("electrons", ne2d)
            sim.set_density("ions", ni2d)
            for _ in range(n_steps):
                sim.step_cfl(0.4)
            return (
                np.asarray(sim.get_state("electrons")),
                np.asarray(sim.get_state("ions")),
                sim.mass("electrons"),
                sim.mass("ions"),
                backend,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 (diagnostic : on essaie le backend suivant)
            last = exc
            print(
                "backend %r indisponible (%s: %s), essai suivant"
                % (backend, type(exc).__name__, str(exc).splitlines()[0][:80])
            )
    raise RuntimeError(
        "aucun backend DSL n'a pu etre branche au System"
    ) from last


def main() -> None:
    n, n_steps = 48, 15
    ne2d, ni2d = initial_conditions(n)

    print(
        "=== two_species_dsl : electrons + ions ecrits en formules vs briques natives ==="
    )
    print(
        "grille %d x %d, %d pas, CFL = 0.4 ; q_e = %.0f, q_i = %.0f"
        % (n, n, n_steps, Q_E, Q_I)
    )

    en, inn, me_n, mi_n = run_native(n, ne2d, ni2d, n_steps)
    ed, idd, me_d, mi_d, backend = run_dsl(n, ne2d, ni2d, n_steps)
    print("backend DSL retenu : %r" % backend)

    # --- equivalence par espece ---
    de = float(np.max(np.abs(ed - en)))
    di = float(np.max(np.abs(idd - inn)))
    print(
        "electrons : max|DSL - natif| = %.3e (bit-identique = %s)"
        % (de, np.array_equal(ed, en))
    )
    print(
        "ions      : max|DSL - natif| = %.3e (bit-identique = %s)"
        % (di, np.array_equal(idd, inn))
    )

    # Electrons : residu/flux bit-identiques (verifie a 1 pas) ; sur plusieurs pas couples, la seule
    # divergence est une reassociation flottante dans l'accumulation du RHS de Poisson partage (epsilon
    # machine). Tolerance serree : on prouve l'equivalence numerique, pas un simple "proche".
    assert de < 1e-24, (
        "electrons : ecart DSL/natif %.3e trop grand (pas une reassociation FP)"
        % de
    )
    # Ions : bit-identiques (un seul bloc isotherme contribue moins a l'accumulation du RHS).
    assert np.array_equal(idd, inn) or di < 1e-24, (
        "ions : ecart DSL/natif %.3e trop grand" % di
    )

    # --- Invariants physiques (conservation de la masse par espece, finitude, positivite) ---
    me0 = float(ne2d.sum())
    mi0 = float(ni2d.sum())
    drift_e = relative_drift(me_d, me0)
    drift_i = relative_drift(mi_d, mi0)
    print(
        "masse electrons : derive relative %.3e ; ions : %.3e"
        % (drift_e, drift_i)
    )

    assert_finite(ed, "electrons")
    assert_finite(idd, "ions")
    assert_positive(ed[0], "densite electronique")
    assert_positive(idd[0], "densite ionique")
    assert drift_e < 1e-9, (
        "masse electronique non conservee (derive %.3e)" % drift_e
    )
    assert drift_i < 1e-9, "masse ionique non conservee (derive %.3e)" % drift_i

    print(
        "OK two_species_dsl (equivalence DSL <-> natif par espece, backend %r)"
        % backend
    )


if __name__ == "__main__":
    main()
