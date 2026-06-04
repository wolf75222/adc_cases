"""Cas "custom_scheme" : un schema numerique ecrit ENTIEREMENT en Python.

Le tuteur voulait pouvoir implementer sa propre methode numerique (spatiale ET
temporelle) en Python, la lib ne fournissant que les briques couteuses. Ici on ne
demande a `adc` QUE la resolution de Poisson (set_density + solve_fields + potential) ;
tout le transport diocotron (reconstruction, flux upwind, pas de temps SSPRK2) est ecrit
en numpy.

C'est le pendant "spatial" de l'integrateur temporel Python de adc.integrate : la
densite vit cote Python, la lib joue le role de solveur elliptique. On verifie :
  - conservation de la masse (flux upwind conservatif, domaine periodique) ;
  - couplage actif (le potentiel resolu par adc n'est pas nul) ;
  - dynamique non triviale (la bande de charge evolue).

Le diocotron : d_t n + div(n v) = 0, v = (-d_y phi, d_x phi)/B0, lap phi = alpha (n - n_i0).
"""

import numpy as np

import adc

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402
from adc_cases.common.checks import assert_finite, relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402


def drift(phi, dx, B0):
    """Vitesse E x B v = (-d_y phi, d_x phi)/B0 par differences centrees periodiques."""
    dphidx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    dphidy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    return -dphidy / B0, dphidx / B0  # (vx, vy)


def divergence_upwind(n, vx, vy, dx):
    """-div(n v) par flux upwind conservatif (forme flux, periodique) -> conserve la masse."""
    # interface i+1/2 selon x : vitesse moyenne, etat amont
    vxr = 0.5 * (vx + np.roll(vx, -1, axis=1))
    fxr = np.where(vxr > 0, n, np.roll(n, -1, axis=1)) * vxr   # flux en i+1/2
    fxl = np.roll(fxr, 1, axis=1)                              # flux en i-1/2
    vyr = 0.5 * (vy + np.roll(vy, -1, axis=0))
    fyr = np.where(vyr > 0, n, np.roll(n, -1, axis=0)) * vyr
    fyl = np.roll(fyr, 1, axis=0)
    return -((fxr - fxl) + (fyr - fyl)) / dx


def poisson_oracle(sim, n):
    """Demande a adc le potentiel self-consistent de la densite n (l'unique appel a la lib)."""
    sim.set_density("ne", n)
    sim.solve_fields()
    return sim.potential()


def rhs(sim, n, dx, B0):
    """Residu -div(n v) calcule EN PYTHON, le potentiel venant du solveur Poisson de adc."""
    phi = poisson_oracle(sim, n)
    vx, vy = drift(phi, dx, B0)
    speed = float(np.hypot(vx, vy).max())
    return divergence_upwind(n, vx, vy, dx), speed


def main():
    nx, L, B0 = 96, 1.0, 1.0
    dx = L / nx
    # CI en bande gaussienne perturbee (mode 4) : meme profil que le cas diocotron.
    n = band_density(nx, L, amp=1.0, width=0.05, mode=4, disp=0.02)
    n_i0 = float(n.mean())  # fond neutralisant : Poisson periodique a moyenne nulle

    # adc.System sert UNIQUEMENT d'oracle de Poisson (un bloc diocotron, alpha (n - n_i0)).
    sim = adc.System(n=nx, L=L, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=B0, alpha=1.0, n_i0=n_i0))
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")

    print("== custom_scheme : transport diocotron 100 % Python, Poisson par adc ==")
    phi0 = poisson_oracle(sim, n)
    print("  |phi|_max initial = %.6e  (Poisson de adc actif)" % float(np.abs(phi0).max()))
    assert float(np.abs(phi0).max()) > 1e-8, "le potentiel self-consistent doit etre non nul"

    mass0 = float(n.sum()) * dx * dx
    n0 = n.copy()
    cfl, nsteps = 0.4, 200
    for step in range(nsteps):
        r1, speed = rhs(sim, n, dx, B0)
        dt = cfl * dx / max(speed, 1e-12)
        n1 = n + dt * r1                      # etage 1
        r2, _ = rhs(sim, n1, dx, B0)
        n = 0.5 * n + 0.5 * (n1 + dt * r2)    # SSPRK2, ecrit en Python
        assert_finite(n, "densite au pas %d" % step)

    mass1 = float(n.sum()) * dx * dx
    drel = relative_drift(mass1, mass0)
    moved = float(np.abs(n - n0).max())
    print("  derive de masse relative = %.3e  (flux upwind conservatif)" % drel)
    print("  evolution max|dn|        = %.3e  (dynamique non triviale)" % moved)
    assert drel < 1e-12, "masse non conservee : %.3e" % drel
    assert moved > 1e-3, "la bande n'a pas evolue"
    print("Schema spatial + temporel ecrit en Python ; adc ne fait que Poisson.")
    print("OK custom_scheme")


if __name__ == "__main__":
    main()
