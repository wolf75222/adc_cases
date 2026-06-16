"""Conditions initiales reutilisees par plusieurs cas (ecrites en numpy, cote application).

Les CI sont l'unique endroit ou la physique d'un scenario est posee : aucune fonction C++ par
cas. Ce module factorise les profils partages (bande gaussienne diocotron, anneau, bulle de
pression Euler). Convention de grille : `field[j, i]` (cf. `adc_cases.common.grid`).
"""

from __future__ import annotations

import numpy as np

from .grid import meshgrid_xy


def band_density(
    n: int,
    L: float = 1.0,
    amp: float = 1.0,
    width: float = 0.05,
    mode: int = 2,
    disp: float = 0.02,
    floor: float = 1.0,
) -> np.ndarray:
    """Bande horizontale de charge perturbee sinusoidalement le long de x (mode azimutal).

        ne(x, y) = floor + amp * exp(-(y - y0)^2 / width^2),
        y0       = 0.5 L + disp * cos(2 pi mode x / L).

    Utilisee par les cas diocotron (grille uniforme et AMR) et custom_scheme. Renvoie un tableau
    (n, n) contigu, convention `ne[j, i]`.
    """
    X, Y = meshgrid_xy(n, L)
    y0 = 0.5 * L + disp * np.cos(2.0 * np.pi * mode * X / L)
    ne = floor + amp * np.exp(-((Y - y0) ** 2) / (width**2))
    return np.ascontiguousarray(ne)


def ring_density(
    n: int,
    L: float = 1.0,
    r0: float = 0.15,
    r1: float = 0.20,
    mode: int = 4,
    delta: float = 0.01,
    floor: float = 1e-3,
) -> np.ndarray:
    """Anneau de charge (colonne creuse) perturbe par un mode azimutal `mode`.

        ne ~ floor en dehors de l'anneau [r0, r1],
        ne ~ 1 - delta + delta sin(mode theta) dans l'anneau,
    centre au milieu du domaine [0, L]^2. CI du benchmark diocotron (arXiv:2510.11808).
    """
    X, Y = meshgrid_xy(n, L)
    r = np.hypot(X - 0.5 * L, Y - 0.5 * L)
    th = np.arctan2(Y - 0.5 * L, X - 0.5 * L)
    ne = np.full((n, n), floor)
    ring = (r > r0) & (r < r1)
    ne[ring] = 1.0 - delta + delta * np.sin(mode * th[ring])
    return ne


def euler_pressure_blob(
    n: int,
    L: float = 1.0,
    rho0: float = 1.0,
    p0: float = 1.0,
    dp: float = 0.5,
    sigma2: float = 0.02,
    gamma: float = 1.4,
) -> np.ndarray:
    """Gaz d'Euler au repos avec surpression gaussienne centrale (detente radiale).

    Renvoie l'etat conservatif U = (rho, rho u, rho v, E) de forme (4, n, n) avec u = v = 0,
    donc E = p / (gamma - 1) ; p = p0 + dp exp(-r^2 / (sigma2 L^2)). Utilise par two_euler et,
    en variante, dsl_euler.
    """
    X, Y = meshgrid_xy(n, L)
    r2 = (X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2
    U = np.zeros((4, n, n))
    U[0] = rho0
    p = p0 + dp * np.exp(-r2 / (sigma2 * L * L))
    U[3] = p / (gamma - 1.0)
    return U


def euler_pressure(U: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Pression d'un etat d'Euler conservatif U = (rho, rho u, rho v, E)."""
    return (gamma - 1.0) * (U[3] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0])
