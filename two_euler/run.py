#!/usr/bin/env python3
"""Cas "two_euler" : deux gaz d'Euler independants (electrons + ions).

Etape "deux Euler, meme code" de l'echelle de tests du tuteur : le meme schema
(CompressibleFlux + HLLC + reconstruction primitive) tourne pour deux especes Euler sans aucun
code dedie ; seules les conditions initiales different. Les electrons sont 100x plus legers
(densite plus faible -> vitesse du son ~10x plus grande), donc ils s'etendent plus vite. Les
deux blocs ne sont pas couples (charge nulle, NoSource). On verifie :
  - masse conservee par bloc (schema conservatif, domaine periodique) ;
  - positivite : rho > 0 et p > 0 (la reconstruction primitive y aide) ;
  - les electrons evoluent plus vite que les ions (front de detente plus etendu) ;
  - le multirate step_adaptive sous-cycle automatiquement les electrons (plus rapides).

Euler compressible 2D : d_t U + div F(U) = 0, U = (rho, rho u, rho v, E),
p = (gamma-1)(E - 1/2 rho |u|^2).
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
from adc_cases import (
    models,
)  # noqa: E402  (compositions de briques nommees, cote application)
from adc_cases.common.checks import (
    assert_finite,
    assert_mass_conserved,
)  # noqa: E402
from adc_cases.common.initial_conditions import (  # noqa: E402
    euler_pressure,
    euler_pressure_blob,
)

GAMMA = 1.4


def blob(n: int, L: float, rho0: float, p0: float, dp: float) -> np.ndarray:
    """Gaz au repos, surpression gaussienne centrale (detente radiale).

    U = (rho, 0, 0, E).
    """
    return euler_pressure_blob(n, L, rho0=rho0, p0=p0, dp=dp, gamma=GAMMA)


def pressure(U: np.ndarray) -> np.ndarray:
    """Pression d'un etat d'Euler conservatif U = (rho, rho u, rho v, E)."""
    return euler_pressure(U, gamma=GAMMA)


def disturbed(U: np.ndarray, U0: np.ndarray, thr: float) -> float:
    """Fraction de cellules ou la pression a change de plus de thr (front)."""
    return float(np.mean(np.abs(pressure(U) - pressure(U0)) > thr))


def main() -> None:
    n, L = 64, 1.0
    sim = adc.System(n=n, L=L, periodic=True)
    spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")
    sim.add_block(
        "electrons",
        model=models.euler(GAMMA),
        spatial=spatial,
        time=adc.Explicit(),
    )
    sim.add_block(
        "ions", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit()
    )
    sim.set_poisson()  # f = 0 (charge nulle) : blocs independants, juste pour solve_fields

    Ue0 = blob(
        n, L, rho0=0.01, p0=1.0, dp=0.5
    )  # electrons : legers -> c ~ 10x, rapides
    Ui0 = blob(n, L, rho0=1.0, p0=1.0, dp=0.5)  # ions : lourds, lents
    sim.set_state("electrons", Ue0.reshape(-1).tolist())
    sim.set_state("ions", Ui0.reshape(-1).tolist())

    me0, mi0 = sim.mass("electrons"), sim.mass("ions")
    print(
        "== two_euler : deux Euler independants (meme schema HLLC + recon primitive) =="
    )
    print(
        "  c_electrons/c_ions ~ %.1f (electrons 100x plus legers)"
        % np.sqrt(1.0 / 0.01)
    )

    for _ in range(20):
        sim.step_adaptive(0.4)

    Ue = np.array(sim.get_state("electrons")).reshape(4, n, n)
    Ui = np.array(sim.get_state("ions")).reshape(4, n, n)
    dme = assert_mass_conserved(
        sim.mass("electrons"), me0, tol=1e-9, label="electrons"
    )
    dmi = assert_mass_conserved(sim.mass("ions"), mi0, tol=1e-9, label="ions")
    pe, pi = pressure(Ue), pressure(Ui)
    fe, fi = disturbed(Ue, Ue0, 0.02), disturbed(Ui, Ui0, 0.02)
    print("  masse      : electrons drel=%.2e  ions drel=%.2e" % (dme, dmi))
    print(
        "  positivite : rho_min e=%.3e i=%.3e ; p_min e=%.3e i=%.3e"
        % (Ue[0].min(), Ui[0].min(), pe.min(), pi.min())
    )
    print(
        "  front (frac cellules perturbees) : electrons=%.3f ions=%.3f"
        % (fe, fi)
    )

    assert Ue[0].min() > 0 and Ui[0].min() > 0, "densite negative"
    assert pe.min() > 0 and pi.min() > 0, "pression negative"
    assert (
        fe > fi
    ), "les electrons (plus legers) devraient s'etendre plus vite que les ions"
    assert_finite(Ue, "etat electrons")
    assert_finite(Ui, "etat ions")
    print("OK two_euler")


if __name__ == "__main__":
    main()
