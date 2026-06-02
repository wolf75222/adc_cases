#!/usr/bin/env python3
"""Cas "two_euler" : deux gaz d'Euler INDEPENDANTS (electrons + ions), non couples.

Etape "deux Euler, meme code" de l'echelle de tests du tuteur : le MEME schema
(CompressibleFlux + HLLC + reconstruction PRIMITIVE) tourne pour deux especes Euler sans aucun
code dedie ; seules les conditions initiales different. Les electrons sont 100x plus legers
(densite plus faible -> vitesse du son ~10x plus grande), donc ils s'etendent plus vite. Les
deux blocs ne sont PAS couples (charge nulle, NoSource). On verifie :
  - masse conservee par bloc (schema conservatif, domaine periodique) ;
  - positivite : rho > 0 et p > 0 (la reconstruction primitive y aide) ;
  - les electrons evoluent plus vite que les ions (front de detente plus etendu) ;
  - le multirate step_adaptive sous-cycle automatiquement les electrons (plus rapides).

Euler compressible 2D : d_t U + div F(U) = 0, U = (rho, rho u, rho v, E),
p = (gamma-1)(E - 1/2 rho |u|^2).
"""
import os
import sys

import numpy as np

import adc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models  # noqa: E402  (compositions de briques nommees, cote application)

GAMMA = 1.4


def blob(n, L, rho0, p0, dp):
    """Gaz au repos, surpression gaussienne centrale (detente radiale). U = (rho, 0, 0, E)."""
    coord = (np.arange(n) + 0.5) / n * L
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    r2 = (xx - 0.5 * L) ** 2 + (yy - 0.5 * L) ** 2
    U = np.zeros((4, n, n))
    U[0] = rho0
    p = p0 + dp * np.exp(-r2 / (0.02 * L * L))
    U[3] = p / (GAMMA - 1.0)  # u = v = 0 : E = p/(gamma-1)
    return U


def pressure(U):
    return (GAMMA - 1.0) * (U[3] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0])


def disturbed(U, U0, thr):
    """Fraction de cellules ou la pression a change de plus de thr (etendue du front)."""
    return float(np.mean(np.abs(pressure(U) - pressure(U0)) > thr))


def main():
    n, L = 64, 1.0
    sim = adc.System(n=n, L=L, periodic=True)
    spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")
    sim.add_block("electrons", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
    sim.add_block("ions", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
    sim.set_poisson()  # f = 0 (charge nulle) : blocs independants, juste pour solve_fields

    Ue0 = blob(n, L, rho0=0.01, p0=1.0, dp=0.5)  # electrons : legers -> c ~ 10x, rapides
    Ui0 = blob(n, L, rho0=1.0, p0=1.0, dp=0.5)   # ions : lourds, lents
    sim.set_state("electrons", Ue0.reshape(-1).tolist())
    sim.set_state("ions", Ui0.reshape(-1).tolist())

    me0, mi0 = sim.mass("electrons"), sim.mass("ions")
    print("== two_euler : deux Euler independants (meme schema HLLC + recon primitive) ==")
    print("  c_electrons/c_ions ~ %.1f (electrons 100x plus legers)" % np.sqrt(1.0 / 0.01))

    for _ in range(20):
        sim.step_adaptive(0.4)

    Ue = np.array(sim.get_state("electrons")).reshape(4, n, n)
    Ui = np.array(sim.get_state("ions")).reshape(4, n, n)
    dme = abs(sim.mass("electrons") - me0) / abs(me0)
    dmi = abs(sim.mass("ions") - mi0) / abs(mi0)
    pe, pi = pressure(Ue), pressure(Ui)
    fe, fi = disturbed(Ue, Ue0, 0.02), disturbed(Ui, Ui0, 0.02)
    print("  masse      : electrons drel=%.2e  ions drel=%.2e" % (dme, dmi))
    print("  positivite : rho_min e=%.3e i=%.3e ; p_min e=%.3e i=%.3e"
          % (Ue[0].min(), Ui[0].min(), pe.min(), pi.min()))
    print("  front (frac cellules perturbees) : electrons=%.3f ions=%.3f" % (fe, fi))

    assert dme < 1e-9 and dmi < 1e-9, "masse non conservee par bloc"
    assert Ue[0].min() > 0 and Ui[0].min() > 0, "densite negative"
    assert pe.min() > 0 and pi.min() > 0, "pression negative"
    assert fe > fi, "les electrons (plus legers) devraient s'etendre plus vite que les ions"
    assert np.isfinite(Ue).all() and np.isfinite(Ui).all(), "etat non fini"
    print("OK two_euler")


if __name__ == "__main__":
    main()
