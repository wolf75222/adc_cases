"""Cas "dsl_euler" : Euler compressible ECRIT EN FORMULES (mini-DSL symbolique adc.dsl).

Demonstration du principe "Python ecrit les equations, le coeur execute les boucles", version
PROTOTYPE interprete CPU. On ne declare AUCUNE brique nommee (pas d'adc.CompressibleFlux) : on ecrit
les variables, leurs formules, le flux et les valeurs propres comme des expressions symboliques.
adc.dsl construit l'arbre, l'interprete en numpy, et le branche sur le backend hote adc.PythonFlux
qui assemble -div(F*) par volumes finis (Rusanov, periodique).

Etat : interprete CPU (prototypage). Le MEME arbre alimenterait plus tard un codegen C++/Kokkos pour
la production (cf. adc_cpp/docs/ARCHITECTURE_CIBLE.md sect. 3). Le chemin de production reste les
briques compilees ; ce cas montre le bout "declaratif" cote utilisateur.

On verifie : masse conservee (continuite, domaine periodique), dynamique non triviale (la bulle de
pression genere des ondes acoustiques), etat physique (rho > 0, p > 0, fini).
"""
import os
import sys

import numpy as np

from adc import dsl

# Rend le depot importable si le paquet n'est pas installe (cf. adc_cases.ensure_importable).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases.common.checks import assert_finite, relative_drift  # noqa: E402
from adc_cases.common.grid import meshgrid_xy  # noqa: E402
from adc_cases.common.initial_conditions import euler_pressure  # noqa: E402

GAMMA = 1.4


def make_euler():
    """Euler 2D entierement declaratif : variables -> primitives -> flux -> valeurs propres."""
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")

    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))

    H = (E + p) / rho                 # enthalpie totale
    c = dsl.sqrt(GAMMA * p / rho)     # vitesse du son

    e.set_flux(
        x=[rhou, rhou * u + p, rhou * v, rho * H * u],
        y=[rhov, rhov * u, rhov * v + p, rho * H * v],
    )
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.check()  # verifie que toutes les variables referencees sont declarees
    return e


def pressure(U):
    return euler_pressure(U, gamma=GAMMA)


def main():
    euler = make_euler()
    print("modele declare en formules : %d variables %s" % (euler.n_vars, euler.cons_names))

    n, L = 64, 1.0
    h = L / n
    gx, gy = meshgrid_xy(n, L)
    r2 = (gx - 0.5) ** 2 + (gy - 0.5) ** 2

    # rho uniforme, vitesse nulle, bulle de pression au centre -> expansion acoustique
    p0 = 1.0 + 0.4 * np.exp(-r2 / 0.01)
    U = np.zeros((4, n, n))
    U[0] = 1.0
    U[3] = p0 / (GAMMA - 1.0)

    mass0 = float(U[0].sum())
    p_init = pressure(U).copy()

    pf = euler.to_python_flux()  # arbre symbolique -> flux numpy -> backend hote
    steps = 120
    for _ in range(steps):
        U = U + pf.cfl_dt(U, h, 0.4) * pf.residual(U, h)

    drel = relative_drift(float(U[0].sum()), mass0)
    moved = float(np.max(np.abs(pressure(U) - p_init)))

    print("apres %d pas : drho_max=%.3f  |v|_max=%.3f"
          % (steps, float(U[0].max() - U[0].min()),
             float(np.max(np.sqrt((U[1] / U[0]) ** 2 + (U[2] / U[0]) ** 2)))))
    print("masse : drel=%.2e   dynamique : max|dp|=%.3f" % (drel, moved))

    assert_finite(U, "etat")
    assert U[0].min() > 0 and pressure(U).min() > 0, "rho ou p negatif"
    assert drel < 1e-9, "masse non conservee (drel=%.2e)" % drel
    assert moved > 1e-3, "dynamique triviale (rien ne bouge)"
    print("OK dsl_euler")


if __name__ == "__main__":
    main()
