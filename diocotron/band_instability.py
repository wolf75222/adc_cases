"""Demo "diocotron" : instabilite diocotron (derive E x B).

Capacite demontree : transport hyperbolique d'une densite de charge couple a la
resolution de Poisson sur une grille uniforme, le tout calcule en C++ compile.
Python ne fait que composer la configuration et diagnostiquer le resultat.

Physique : un plasma non neutre confine par un champ magnetique B0 derive a la
vitesse E x B. Une bande de densite perturbee (condition initiale "Band")
developpe l'instabilite diocotron : une petite perturbation periodique le long
de x croit exponentiellement, signature de l'instabilite de cisaillement de la
derive.

Architecture (nouvelle API "composition par blocs")
----------------------------------------------------
Il n'existe plus de DiocotronSolver dedie : on reconstruit la MEME physique a
partir du systeme generique adc.System, en lui ajoutant un seul bloc de modele
"diocotron" (1 variable, la densite ne). Le systeme :

  - avance la densite par transport hyperbolique (limiteur minmod, flux Rusanov,
    integration explicite SSPRK2) ;
  - recalcule le potentiel via Poisson a chaque sous-pas, avec un second membre
    de charge Sum_s elliptic_rhs_s ; pour le bloc diocotron ce terme vaut
    alpha * (ne - n_i0).

Comme le transport est purement advectif (champ de vitesse E x B a divergence
nulle), la masse totale doit etre conservee exactement.

Condition initiale "Band" (reconstruite en numpy)
--------------------------------------------------
Une bande horizontale de charge, perturbee sinusoidalement le long de x :

    ne(x, y) = 1 + band_amp * exp(-(y - y0)^2 / band_width^2)
    y0       = 0.5*L + band_disp * cos(2*pi*band_mode*x/L)

Le domaine est PERIODIQUE : Poisson exige un second membre a moyenne nulle, donc
on neutralise la bande par un fond ionique n_i0 = moyenne(ne). Sans ce fond, le
probleme elliptique periodique n'est pas soluble (compatibilite de Fredholm).

Diagnostics imprimes : a chaque k pas, le temps t, l'amplitude de la perturbation
(norme L2 de la deviation de la densite par rapport a sa moyenne en x, c.-a-d. la
partie non axisymetrique du mode) et la masse totale.

Invariants verifies par assert :
  - conservation de la masse : |mass - mass0| / mass0 < 1e-6 a chaque pas ;
  - croissance de l'instabilite : amplitude finale > amplitude initiale.
"""

import os
import sys

import numpy as np

import adc

# Rend le depot importable si le paquet n'est pas installe (cf. adc_cases.ensure_importable).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (compositions de briques nommees, cote application)
from adc_cases.common.checks import relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402


def perturbation_amplitude(density):
    """Amplitude L2 de la perturbation = deviation par rapport a la moyenne en x.

    La bande non perturbee est uniforme le long de x (axis=1) et structuree le
    long de y (axis=0). La moyenne par ligne (sur x) reconstruit donc le profil
    de base ; ce qui reste est la perturbation portant l'instabilite.
    """
    base = density.mean(axis=1, keepdims=True)  # profil moyen en x, par ligne
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


def main():
    # --- Condition initiale en bande, grille 96 x 96 (CI partagee, mode 2) ---
    n = 96
    L = 1.0
    ne0 = band_density(n, L, amp=1.0, width=0.05, mode=2, disp=0.02)

    # Fond ionique neutralisant : moyenne de la densite initiale. Indispensable
    # pour la solubilite de Poisson sur un domaine periodique (RHS a moyenne nulle).
    n_i0 = float(ne0.mean())

    # --- Composition du systeme : un seul bloc "diocotron" (1 variable) ---
    sim = adc.System(n=n, L=L, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
                  spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    # Poisson periodique sans paroi : RHS = densite de charge Sum_s elliptic_rhs_s,
    # solveur multigrille geometrique.
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne0)

    # Masse de reference et amplitude initiale de la perturbation.
    mass0 = sim.mass("ne")
    amp0 = perturbation_amplitude(sim.density("ne"))
    dx = L / sim.nx()

    print("=== Demo diocotron : instabilite de derive E x B ===")
    print(f"grille n = {sim.nx()} x {sim.nx()}, dx = {dx:.6e}")
    print(f"fond ionique n_i0       = {n_i0:.6e}  (moyenne de ne, periodique)")
    print(f"masse initiale          = {mass0:.6e}")
    print(f"amplitude initiale      = {amp0:.6e}")
    print()
    print(f"{'pas':>4} {'t':>12} {'amplitude':>14} {'mass':>16}")

    # --- Boucle d'integration : ~120 pas, CFL = 0.4 ---
    n_steps = 120
    k = 10            # frequence d'impression des diagnostics
    rel_mass_tol = 1e-6

    amp_last = amp0
    for step in range(1, n_steps + 1):
        # step_cfl avance le temps d'un pas dt = CFL * dx / vitesse_max et le renvoie.
        sim.step_cfl(0.4)

        # Invariant physique : la masse totale est conservee (transport advectif).
        mass = sim.mass("ne")
        rel_mass = relative_drift(mass, mass0)
        assert rel_mass < rel_mass_tol, (
            f"masse non conservee au pas {step} : ecart relatif {rel_mass:.3e}"
        )

        amp_last = perturbation_amplitude(sim.density("ne"))

        if step % k == 0 or step == n_steps:
            print(f"{step:4d} {sim.time():12.6f} {amp_last:14.6e} {mass:16.8e}")

    print()
    print(f"amplitude finale = {amp_last:.6e}  (initiale {amp0:.6e})")
    print(f"facteur de croissance = {amp_last / amp0:.4f}")

    # --- Invariant : l'instabilite a fait croitre la perturbation ---
    assert amp_last > amp0, (
        "l'amplitude n'a pas cru : instabilite diocotron non observee"
    )

    print("OK diocotron")


if __name__ == "__main__":
    main()
