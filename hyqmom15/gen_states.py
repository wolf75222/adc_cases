#!/usr/bin/env python3
"""Genere golden/golden_states.csv, les etats echantillons de validation.

Etats FIGES de la validation HyQMOM, generes de facon deterministe (seed
fixe). Les goldens MATLAB (golden_fx/fy/vp.csv) sont produits a partir de CE
fichier par golden/gen/golden_gen.m (Octave) sur le code de reference RIEMOM2D : regenerer les etats impose
de regenerer les goldens (provenance dans le README). Etats choisis pour couvrir : maxwellienne
au repos / en derive / correlee / haut Mach (regime crossing Ma=20), melanges discrets fortement
asymetriques (S30 != 0), etat quasi-degenere (variance ~1e-6, test de cancellation sqrt) et etat
fortement anisotrope (C20/C02 = 100)."""

from __future__ import annotations

import os

import numpy as np

from model import GAUSSIAN_PARAMS, gaussian_state, mixture_state


def build_states() -> np.ndarray:
    """Construit les etats echantillons figes de la validation HyQMOM."""
    states = []
    # 1-4 : gaussiennes exactes (oracle Isserlis disponible pour l'ordre 5) -- parametres dans
    # model.GAUSSIAN_PARAMS (source unique, cross-verifiee par runs/run.py contre le CSV fige).
    for prm in GAUSSIAN_PARAMS:
        states.append(gaussian_state(*prm))
    # 5-9 : melanges discrets (realisables par construction), seed fige.
    rng = np.random.default_rng(20260611)
    for k in range(4):
        npts = int(rng.integers(4, 9))
        w = rng.uniform(0.1, 1.0, npts)
        vx = rng.normal(0.0, 1.0 + k, npts)
        vy = rng.normal(0.5 * k, 0.5 + 0.5 * k, npts)
        states.append(mixture_state(w, vx, vy))
    # quasi-degenere : points resserres en vx (C20 ~ 1e-6), disperses en vy.
    w = np.array([0.3, 0.3, 0.4])
    vx = 1.0 + 1e-3 * np.array([-1.0, 0.5, 0.4])
    vy = np.array([-1.2, 0.3, 0.9])
    states.append(mixture_state(w, vx, vy))
    # 10 : gaussienne fortement anisotrope.
    states.append(gaussian_state(1.0, 0.1, -0.1, 4.0, 0.0, 0.04))
    return np.array(states)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "golden")
    os.makedirs(out, exist_ok=True)
    states = build_states()
    path = os.path.join(out, "golden_states.csv")
    np.savetxt(path, states, delimiter=",", fmt="%.17g")
    print("ecrit %s : %d etats x %d moments" % (path, *states.shape))


if __name__ == "__main__":
    main()
