#!/usr/bin/env python3
"""Cas "diocotron_dsl" : le modele diocotron ECRIT ENTIEREMENT EN FORMULES (adc.dsl.Model),
PUIS PROUVE bit-identique a la composition native de briques (adc_cases.models.diocotron).

Pourquoi ce cas
---------------
Le cas "diocotron" (et sa variante CI "band_instability") compose la physique a partir de
briques NATIVES nommees : adc.Scalar + adc.ExB + adc.BackgroundDensity. Ici, on ecrit la MEME
physique sans aucune brique nommee : on declare la variable conservative, les champs auxiliaires
(phi / grad phi), le flux d'advection E x B, les valeurs propres (vitesses de derive) et le second
membre elliptique (densite de charge neutralisee) comme des EXPRESSIONS symboliques. adc.dsl
genere le C++, le compile et l'installe comme bloc du System via add_equation(...).

Le point de la demonstration est l'EQUIVALENCE : sur la meme grille, la meme condition initiale,
le meme Poisson periodique et le meme nombre de pas, l'etat produit par le modele DSL est
BIT-IDENTIQUE a celui de la composition native. Les formules DSL reproduisent EXACTEMENT les
conventions des briques du coeur (cf. ci-dessous), donc il n'y a aucune tolerance : np.array_equal.

Conventions reproduites (ancrees dans le coeur adc_cpp)
-------------------------------------------------------
  - Transport E x B (include/adc/physics/hyperbolic.hpp, struct ExBVelocity) :
        v = (-grad_y / B0, grad_x / B0)  (a divergence nulle)
        flux  f = n * v(dir)
        valeur propre (1 onde) = v(dir)
        variable conservative unique "n" (role Density), primitif = conservatif.
  - Second membre elliptique (include/adc/physics/elliptic.hpp, struct BackgroundDensity) :
        rhs = alpha * (n - n_i0)   (fond neutralisant, RHS a moyenne nulle sur domaine periodique).
  - Pas de source (adc.NoSource).

Backend
-------
On compile en backend "production" (chemin NATIF zero-copie add_native_block : meme moteur que
add_block, parite stricte). C'est la cible du plan. Si le module n'expose pas le chemin natif sur
cette plateforme, on retombe sur "aot" (chemin de production host-marshale, numerique identique) :
les deux donnent un etat bit-identique au natif (verifie). Le backend retenu est affiche.

Invariants verifies (assert)
----------------------------
  - EQUIVALENCE (coeur du cas) : etat DSL == etat natif, BIT-IDENTIQUE (np.array_equal) ;
  - conservation de la masse (transport advectif, domaine periodique) ;
  - croissance de l'instabilite (amplitude finale > amplitude initiale), sur les deux modeles.
"""

import numpy as np

import adc
from adc import dsl

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (composition native de briques, oracle de reference)
from adc_cases.common.checks import relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402

# Parametres physiques partages par les deux modeles (DSL et natif) : ils DOIVENT coincider pour
# que l'equivalence soit testable (memes conventions de briques).
B0 = 1.0       # champ magnetique de fond (derive E x B)
ALPHA = 1.0    # facteur du second membre elliptique alpha (n - n_i0)


def diocotron_dsl_model(n_i0):
    """Modele diocotron ECRIT EN FORMULES (adc.dsl.Model), reproduisant a l'identique les briques
    natives ExBVelocity (transport) et BackgroundDensity (elliptique). @p n_i0 : fond ionique
    neutralisant (moyenne de la densite initiale, pour la solubilite de Poisson periodique)."""
    m = dsl.Model("diocotron_dsl")

    # Variable conservative unique : la densite "n" (role canonique Density, comme la brique native).
    (n,) = m.conservative_vars("n")

    # Champs auxiliaires fournis par le solveur (canal adc::Aux) : le potentiel et son gradient.
    # grad_x / grad_y sont lus par le flux d'advection ; phi est declare pour completer le contrat.
    m.aux("phi")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    # Vitesse de derive E x B : v = (-grad_y / B0, grad_x / B0). Convention EXACTE de ExBVelocity.
    vx = (-grad_y) / B0
    vy = grad_x / B0

    # Flux physique d'advection f = n * v(dir), une composante (1 variable conservative).
    m.flux(x=[n * vx], y=[n * vy])
    # Spectre : une onde, la vitesse de derive dans la direction consideree.
    m.eigenvalues(x=[vx], y=[vy])

    # Scalaire transporte : primitif = conservatif (layout Prim = [n], inversion triviale).
    m.primitive_vars(n=n)
    m.conservative_from([n])

    # Second membre elliptique (densite de charge neutralisee) : alpha (n - n_i0), convention
    # EXACTE de BackgroundDensity. Couple le bloc au Poisson de systeme (rhs = "charge_density").
    m.elliptic_rhs(ALPHA * (n - n_i0))

    m.check()  # verifie que toute variable referencee (flux / valeurs propres / elliptique) est declaree
    return m


def perturbation_amplitude(density):
    """Amplitude L2 de la perturbation = deviation par rapport a la moyenne en x (cf. cas diocotron).
    La bande non perturbee est uniforme le long de x (axis=1) ; ce qui reste porte l'instabilite."""
    base = density.mean(axis=1, keepdims=True)
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


def make_system(ne0):
    """Construit un System diocotron periodique vide (sans bloc). Le bloc (natif ou DSL) est ajoute
    par l'appelant ; tout le reste (grille / Poisson / densite) est IDENTIQUE entre les deux."""
    return adc.System(n=ne0.shape[0], L=1.0, periodic=True)


def run_native(ne0, n_i0, n_steps):
    """Reference : la composition NATIVE de briques (adc_cases.models.diocotron), minmod + Rusanov,
    explicite. Renvoie (densite finale, temps, masse)."""
    sim = make_system(ne0)
    sim.add_block("ne", model=models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0),
                  spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne0)
    for _ in range(n_steps):
        sim.step_cfl(0.4)
    return np.asarray(sim.density("ne")), sim.time(), sim.mass("ne")


def run_dsl(ne0, n_i0, n_steps):
    """Le MEME systeme, mais le bloc "ne" est le modele DSL compile (backend "production" natif si
    disponible, sinon "aot"). Memes schema (minmod + Rusanov) et integrateur que le natif.
    Renvoie (densite finale, temps, masse, backend_retenu)."""
    include = adc_include()
    so_dir = case_output_dir("diocotron_dsl")
    model = diocotron_dsl_model(n_i0)

    # Backend : on PREFERE "production" (chemin natif zero-copie, cible du plan) ; si la compilation
    # native echoue OU si add_equation refuse le bloc natif sur cette plateforme, on retombe sur
    # "aot" (numerique identique, host-marshale). Le chemin natif (add_native_block) verifie une cle
    # ABI incluant la signature des en-tetes : quand le module compile (_adc) a ete bati contre des
    # en-tetes differents de include/, la compilation REUSSIT mais add_native_block leve un
    # RuntimeError "ABI incompatible". On enveloppe donc TOUTE la construction (compile + aiguillage
    # add_equation + run) dans le try : un echec sur "production" rejoue le tout en "aot" (qui passe
    # par add_compiled_block, sans cle ABI). Les deux donnent un etat bit-identique au natif ; le
    # choix n'affecte pas le resultat verifie.
    import os
    for cand in ("production", "aot"):
        try:
            compiled = model.compile(os.path.join(so_dir, "diocotron_dsl_%s.so" % cand),
                                     include, backend=cand)
            sim = make_system(ne0)
            # add_equation aiguille sur le backend du CompiledModel (add_native_block / add_compiled_block).
            sim.add_equation("ne", model=compiled,
                             spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                             time=adc.Explicit())
            sim.set_poisson(rhs="charge_density", solver="geometric_mg")
            sim.set_density("ne", ne0)
            for _ in range(n_steps):
                sim.step_cfl(0.4)
            return np.asarray(sim.density("ne")), sim.time(), sim.mass("ne"), cand
        except Exception as exc:  # noqa: BLE001 (diagnostic : on essaie le backend suivant)
            print("backend %r indisponible (%s), essai suivant" % (cand, type(exc).__name__))
    raise RuntimeError("aucun backend DSL n'a compile ni execute le modele diocotron")


def main():
    # --- Condition initiale en bande : MEME grille / IC que la variante CI native (mode 2) ---
    n, L = 96, 1.0
    ne0 = band_density(n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    # Fond ionique neutralisant : moyenne de la densite initiale (solubilite de Poisson periodique).
    n_i0 = float(ne0.mean())
    n_steps = 60

    amp0 = perturbation_amplitude(ne0)

    print("=== diocotron_dsl : modele ecrit en formules (adc.dsl.Model) vs briques natives ===")
    print("grille n = %d x %d, %d pas, CFL = 0.4" % (n, n, n_steps))
    print("fond ionique n_i0 = %.6e (moyenne de ne)" % n_i0)

    # --- Reference native, puis modele DSL, sur la MEME configuration ---
    dn, tn, mn = run_native(ne0, n_i0, n_steps)
    dd, td, md, backend = run_dsl(ne0, n_i0, n_steps)

    print("backend DSL retenu : %r" % backend)
    print("natif : t = %.6f, masse = %.10e" % (tn, mn))
    print("DSL   : t = %.6f, masse = %.10e" % (td, md))

    # --- EQUIVALENCE (coeur du cas) : etat DSL == etat natif, BIT-IDENTIQUE ---
    max_abs = float(np.max(np.abs(dd - dn)))
    identical = bool(np.array_equal(dd, dn))
    print("max|DSL - natif| = %.3e   bit-identique = %s" % (max_abs, identical))
    assert identical, (
        "le modele DSL n'est PAS bit-identique au natif (max|d| = %.3e) : une formule DSL "
        "diverge d'une brique du coeur (ExBVelocity / BackgroundDensity)" % max_abs)

    # --- Invariants physiques (sur le modele DSL ; le natif est l'oracle deja valide ailleurs) ---
    amp_dsl = perturbation_amplitude(dd)
    mass0 = float(ne0.sum())               # masse initiale (sim.mass = somme de la densite, cf. coeur)
    mass_drift = relative_drift(md, mass0)  # transport advectif -> masse conservee
    print("amplitude : initiale %.6e -> finale %.6e (facteur %.4f)"
          % (amp0, amp_dsl, amp_dsl / amp0))
    print("derive de masse relative (DSL) = %.3e" % mass_drift)

    assert td == tn, "les deux runs n'ont pas avance du meme temps (t_dsl=%.6f, t_natif=%.6f)" % (td, tn)
    assert md == mn, "masses non identiques entre DSL et natif (m_dsl=%.10e, m_natif=%.10e)" % (md, mn)
    assert mass_drift < 1e-6, "masse non conservee par le modele DSL (derive %.3e)" % mass_drift
    assert amp_dsl > amp0, "l'instabilite diocotron n'a pas cru (amp_finale <= amp_initiale)"

    print("OK diocotron_dsl (equivalence DSL <-> natif bit-identique, backend %r)" % backend)


if __name__ == "__main__":
    main()
