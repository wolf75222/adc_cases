#!/usr/bin/env python3
"""Effet TEMPOREL du complement de Schur sur un fluide magnetise CARTESIEN raide.

But (PR6)
---------
MESURER, sur un fluide isotherme magnetise CARTESIEN, l'effet temporel de l'etage
source condense par Schur (`adc.Split(Explicit, CondensedSchur)`, #118-128) face a
l'integration EXPLICITE de la meme source.

Le terme raide est la force de Lorentz `m x Omega` (Omega = omega_c e_z, omega_c =
B_z). Integree EXPLICITEMENT, la rotation cyclotronique impose la borne de
stabilite dt * omega_c < O(1) : a omega_c grand le pas explicite s'effondre. L'etage
`CondensedSchur` avance la source electrostatique/Lorentz IMPLICITEMENT (il assemble
et resout l'operateur condense A = I + theta^2 dt^2 alpha rho B^{-1}, B portant la
rotation de Lorentz), supprimant cette borne : le pas stable n'est plus limite que
par le transport hyperbolique.

Modele
------
Fluide isotherme magnetise (memes equations que `magnetic_isothermal_dsl`), ecrit
une seule fois en `adc.dsl.Model` :

    d_t rho + div(m) = 0
    d_t m  + div(m m^T/rho + cs2 rho I) = q rho (-grad phi) + m x Omega
    Delta phi = q rho

Deux VARIANTES de source partagent flux / valeurs propres / Poisson :

`local`   la source complete (electrostatique + Lorentz) est emise dans le C++
          genere ; avancee EXPLICITEMENT apres le transport.
`schur`   la source locale est nulle ; l'etage electrostatique/Lorentz est avance
          par `CondensedSchur` (set_source_stage), IMPLICITEMENT.

Pour rendre le TRANSPORT non limitant et isoler la raideur de la SOURCE, on prend
une vitesse du son lente (cs2 petit) : le pas explicite de transport ~ h/cs reste
large devant 1/omega_c.

Mesure
------
Pour chaque variante on cherche le PLUS GRAND dt qui reste stable (densite finie,
bornee, positive) jusqu'a t_end, par balayage geometrique de dt. On reporte le pas
stable explicite vs Schur (theta=0.5 Crank-Nicolson et theta=1.0 Euler retrograde),
le produit dt*omega_c et le gain.

Plateforme
----------
Le backend DSL `production` (natif zero-copie) ne se lie pas toujours (echec dlopen
selon la plateforme : c'est le cas sur macOS arm64). On utilise le backend `aot`
(host-marshale), qui supporte set_source_stage et la force de Lorentz via B_z. Le
chemin AOT n'expose que l'integrateur explicite SSPRK2 (pas SSPRK3) : on l'emploie
pour le transport, ce qui n'affecte pas la conclusion temporelle (le facteur mesure
vient de la source, pas du schema RK du transport).

Lancer
------
    PYTHONPATH=<adc_cpp>/build-master/python \
        python schur_magnetized_cartesian/run.py

Options : --n, --omega-c, --cs2, --alpha, --t-end, --csv.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

import adc
from adc import dsl

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402

CASE = "schur_magnetized_cartesian"
Q = -1.0  # charge (signe electron) ; le facteur global est absorbe dans alpha / B_z


def magnetized_model(local_source, cs2):
    """Fluide isotherme magnetise en formules. local_source=True -> source emise
    dans le C++ (variante explicite) ; False -> source nulle (l'etage Schur la porte)."""
    tag = "local" if local_source else "schur"
    m = dsl.Model("schurmag_%s" % tag)
    rho, mx, my = m.conservative_vars(
        "rho", "rho_u", "rho_v", roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    cs2p = m.param("cs2", cs2)
    q = m.param("charge", Q)

    m.flux(x=[mx, mx * u + cs2p * rho, mx * v],
           y=[my, my * u, my * v + cs2p * rho])
    cs = dsl.sqrt(cs2p)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])

    if local_source:
        # electrostatique q rho (-grad phi) + Lorentz (m x Omega) avec Omega = bz e_z :
        # composante x = +bz my, composante y = -bz mx.
        m.source([0.0, q * rho * (-gx) + bz * my, q * rho * (-gy) - bz * mx])
    else:
        # CondensedSchur possede l'etage complet : la source locale doit etre nulle
        # (sinon elle serait avancee deux fois).
        m.source([0.0 * rho, 0.0 * mx, 0.0 * my])

    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(q * rho)
    m.check()
    return m


def initial_state(n):
    """Densite perturbee (cosinus en x) + vitesse oblique (u=v=0.5) pour que la
    rotation de Lorentz soit active des le depart."""
    x = (np.arange(n) + 0.5) / n
    rho0 = 1.0 + 0.05 * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))
    half = 0.5 * np.ones((n, n))
    return rho0, half, half


def build(compiled, n, L, omega_c, alpha, schur, theta):
    sim = adc.System(n=n, L=L, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_equation(
        "plasma", model=compiled,
        spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov",
                                 variables="conservative"),
        time=adc.Explicit())  # AOT : SSPRK2 (transport) ; la source vit dans l'etage Schur
    sim.set_magnetic_field(omega_c * np.ones((n, n)))
    if schur:
        # adc.Split n'est pas cable sur backend AOT (l'ABI .so ne transporte pas
        # SSPRK3) ; on branche l'etage source condense directement (meme C++).
        sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta, alpha)
    rho0, u0, v0 = initial_state(n)
    sim.set_primitive_state("plasma", rho=rho0, u=u0, v=v0)
    sim.solve_fields()
    return sim


def is_stable(compiled, n, L, omega_c, alpha, dt, schur, theta, t_end):
    """True si la simulation reste finie / bornee / positive jusqu'a t_end."""
    sim = build(compiled, n, L, omega_c, alpha, schur, theta)
    nst = max(2, int(math.ceil(t_end / dt)))
    for _ in range(nst):
        sim.step(dt)
        d = np.asarray(sim.density("plasma"))
        if not np.isfinite(d).all() or np.abs(d).max() > 1.0e3 or d.min() < -1.0e-2:
            return False
    return True


def largest_stable_dt(compiled, n, L, omega_c, alpha, schur, theta, t_end,
                      dt_max=0.5):
    """Plus grand dt stable sur une echelle geometrique (quart de decade)."""
    best = 0.0
    for e in range(-16, 5):
        dt = 10.0 ** (e / 4.0)
        if dt > dt_max:
            continue
        if is_stable(compiled, n, L, omega_c, alpha, dt, schur, theta, t_end):
            best = dt
    return best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--omega-c", type=float, default=1.0e3,
                   help="frequence cyclotron B_z (terme raide de Lorentz)")
    p.add_argument("--cs2", type=float, default=1.0e-4,
                   help="vitesse du son^2 ; petite => transport non limitant")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="couplage electrostatique de l'etage source condense")
    p.add_argument("--t-end", type=float, default=1.0)
    p.add_argument("--csv", action="store_true",
                   help="ecrit out/%s/dt_stable.csv" % CASE)
    return p.parse_args()


def main():
    args = parse_args()
    inc = adc_include()
    so_dir = case_output_dir(CASE)

    cm_local = magnetized_model(True, args.cs2).compile(
        os.path.join(so_dir, "schurmag_local.so"), inc, backend="aot")
    cm_schur = magnetized_model(False, args.cs2).compile(
        os.path.join(so_dir, "schurmag_schur.so"), inc, backend="aot")

    n, L = args.n, args.L
    h = L / n
    transport_dt = 0.5 * h / math.sqrt(args.cs2 + 0.5)  # CFL transport approx (u,v=0.5)

    print("=== %s : effet temporel du Schur (fluide magnetise cartesien raide) ===" % CASE)
    print("n=%d, L=%g, h=%g | omega_c(B_z)=%g, cs2=%g, alpha=%g, t_end=%g"
          % (n, L, h, args.omega_c, args.cs2, args.alpha, args.t_end))
    print("transport-limited dt ~ %.3e ; source-limited explicit dt ~ 1/omega_c = %.3e"
          % (transport_dt, 1.0 / args.omega_c))
    print()

    de = largest_stable_dt(cm_local, n, L, args.omega_c, args.alpha,
                           schur=False, theta=0.5, t_end=args.t_end)
    ds05 = largest_stable_dt(cm_schur, n, L, args.omega_c, args.alpha,
                             schur=True, theta=0.5, t_end=args.t_end)
    ds10 = largest_stable_dt(cm_schur, n, L, args.omega_c, args.alpha,
                             schur=True, theta=1.0, t_end=args.t_end)

    rows = [
        ("explicit (Lorentz explicite)", de, de * args.omega_c),
        ("schur theta=0.5 (Crank-Nicolson)", ds05, ds05 * args.omega_c),
        ("schur theta=1.0 (Euler retrograde)", ds10, ds10 * args.omega_c),
    ]
    print("%-38s %12s %14s" % ("methode", "dt_stable", "dt*omega_c"))
    for name, dt, prod in rows:
        print("%-38s %12.3e %14.2f" % (name, dt, prod))
    print()
    if de > 0:
        print("gain en pas de temps du Schur sur l'explicite :")
        print("  theta=0.5 -> %.1fx ; theta=1.0 -> %.1fx" % (ds05 / de, ds10 / de))
    else:
        print("explicite instable a tout dt teste : gain Schur non borne par le bas")

    if args.csv:
        path = os.path.join(so_dir, "dt_stable.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "dt_stable", "dt_times_omega_c", "gain_over_explicit"])
            for name, dt, prod in rows:
                gain = dt / de if de > 0 else float("inf")
                w.writerow([name, dt, prod, gain])
        print("\nCSV :", path)


if __name__ == "__main__":
    main()
