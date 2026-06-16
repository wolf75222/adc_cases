"""Cas "custom_scheme" : un schema numerique ecrit entierement en Python.

Le tuteur voulait pouvoir implementer sa propre methode numerique (spatiale et
temporelle) en Python, la lib ne fournissant que les briques couteuses. Ici on ne
demande a `adc` que la resolution de Poisson (set_density + solve_fields + potential) ;
tout le transport diocotron (reconstruction, flux upwind, pas de temps SSPRK2) est ecrit
en numpy.

C'est le pendant "spatial" de l'integrateur temporel Python de adc.integrate : la
densite vit cote Python, la lib joue le role de solveur elliptique. On verifie :
  - conservation de la masse (flux upwind conservatif, domaine periodique) ;
  - couplage actif (le potentiel resolu par adc n'est pas nul) ;
  - dynamique non triviale (la bande de charge evolue).

Le diocotron : d_t n + div(n v) = 0, v = (-d_y phi, d_x phi)/B0, lap phi = alpha (n - n_i0).
"""

from __future__ import annotations

import numpy as np

import adc

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import models  # noqa: E402
from adc_cases.common.checks import assert_finite, relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402


def drift(
    phi: np.ndarray, dx: float, B0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vitesse E x B par differences centrees periodiques.

    v = (-d_y phi, d_x phi) / B0, les derivees etant evaluees par differences
    centrees a deux points avec enroulement (domaine periodique).

    Args:
        phi: Potentiel sur la grille carree.
        dx: Pas d'espace.
        B0: Champ magnetique de fond.

    Returns:
        Le couple (vx, vy) des composantes de la vitesse de derive.
    """
    dphidx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    dphidy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    return -dphidy / B0, dphidx / B0  # (vx, vy)


def divergence_upwind(
    n: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float
) -> np.ndarray:
    """-div(n v) par flux upwind conservatif sur domaine periodique.

    La forme flux (somme telescopique des flux d'interface) garantit la
    conservation exacte de la masse. L'etat amont est choisi selon le signe
    de la vitesse d'interface.

    Args:
        n: Densite sur la grille.
        vx: Composante x de la vitesse.
        vy: Composante y de la vitesse.
        dx: Pas d'espace.

    Returns:
        Le residu -div(n v) sur la grille.
    """
    # Interface i+1/2 selon x : vitesse moyenne, etat amont.
    vxr = 0.5 * (vx + np.roll(vx, -1, axis=1))
    fxr = np.where(vxr > 0, n, np.roll(n, -1, axis=1)) * vxr  # flux en i+1/2
    fxl = np.roll(fxr, 1, axis=1)  # flux en i-1/2
    vyr = 0.5 * (vy + np.roll(vy, -1, axis=0))
    fyr = np.where(vyr > 0, n, np.roll(n, -1, axis=0)) * vyr
    fyl = np.roll(fyr, 1, axis=0)
    return -((fxr - fxl) + (fyr - fyl)) / dx


def poisson_oracle(sim: adc.System, n: np.ndarray) -> np.ndarray:
    """Demande a adc le potentiel self-consistent de la densite n.

    C'est l'unique appel a la lib : on lui confie le solveur elliptique
    (set_density + solve_fields + potential), tout le reste vit cote Python.

    Args:
        sim: Systeme adc servant d'oracle de Poisson.
        n: Densite courante.

    Returns:
        Le potentiel self-consistent resolu par adc.
    """
    sim.set_density("ne", n)
    sim.solve_fields()
    return sim.potential()


def rhs(
    sim: adc.System, n: np.ndarray, dx: float, B0: float
) -> tuple[np.ndarray, float]:
    """Residu -div(n v) en Python, le potentiel venant du solveur de adc.

    Enchaine l'oracle de Poisson, le calcul de la vitesse de derive et la
    divergence upwind ; renvoie aussi la vitesse maximale pour la condition CFL.

    Args:
        sim: Systeme adc servant d'oracle de Poisson.
        n: Densite courante.
        dx: Pas d'espace.
        B0: Champ magnetique de fond.

    Returns:
        Le couple (residu -div(n v), vitesse de derive maximale).
    """
    phi = poisson_oracle(sim, n)
    vx, vy = drift(phi, dx, B0)
    speed = float(np.hypot(vx, vy).max())
    return divergence_upwind(n, vx, vy, dx), speed


def main() -> None:
    """Avance le transport diocotron en Python et verifie masse + dynamique."""
    nx, L, B0 = 96, 1.0, 1.0
    dx = L / nx
    # CI en bande gaussienne perturbee (mode 4) : meme profil que le cas diocotron.
    n = band_density(nx, L, amp=1.0, width=0.05, mode=4, disp=0.02)
    n_i0 = float(
        n.mean()
    )  # Fond neutralisant : Poisson periodique a moyenne nulle.

    # adc.System sert uniquement d'oracle de Poisson (un bloc diocotron, alpha (n - n_i0)).
    sim = adc.System(n=nx, L=L, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=B0, alpha=1.0, n_i0=n_i0))
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")

    print(
        "== custom_scheme : transport diocotron 100 % Python, Poisson par adc =="
    )
    phi0 = poisson_oracle(sim, n)
    print(
        "  |phi|_max initial = %.6e  (Poisson de adc actif)"
        % float(np.abs(phi0).max())
    )
    assert (
        float(np.abs(phi0).max()) > 1e-8
    ), "le potentiel self-consistent doit etre non nul"

    mass0 = float(n.sum()) * dx * dx
    n0 = n.copy()
    cfl, nsteps = 0.4, 200
    for step in range(nsteps):
        r1, speed = rhs(sim, n, dx, B0)
        dt = cfl * dx / max(speed, 1e-12)
        n1 = n + dt * r1  # Etage 1 d'Euler explicite.
        r2, _ = rhs(sim, n1, dx, B0)
        n = 0.5 * n + 0.5 * (n1 + dt * r2)  # SSPRK2, ecrit en Python.
        assert_finite(n, "densite au pas %d" % step)

    mass1 = float(n.sum()) * dx * dx
    drel = relative_drift(mass1, mass0)
    moved = float(np.abs(n - n0).max())
    print("  derive de masse relative = %.3e  (flux upwind conservatif)" % drel)
    print("  evolution max|dn|        = %.3e  (dynamique non triviale)" % moved)
    assert drel < 1e-12, "masse non conservee : %.3e" % drel
    assert moved > 1e-3, "la bande n'a pas evolue"
    print(
        "Schema spatial + temporel ecrit en Python ; adc ne fait que Poisson."
    )
    print("OK custom_scheme")


if __name__ == "__main__":
    main()
