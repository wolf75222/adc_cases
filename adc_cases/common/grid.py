"""Grilles a centres de cellules, partagees par les cas.

Convention de la facade `adc` : un champ scalaire est un tableau numpy (n, n) indexe
`field[j, i]`, ou la cellule (j, i) a pour centre x = (i + 0.5)/n * L (colonne, axis=1) et
y = (j + 0.5)/n * L (ligne, axis=0). Pour obtenir des grilles X, Y telles que X[j, i] = x_i et
Y[j, i] = y_j, on appelle `numpy.meshgrid(..., indexing="xy")`. Plusieurs cas reconstruisaient
ce meshgrid a la main ; il est centralise ici.
"""

from __future__ import annotations

import numpy as np


def cell_centers(n: int, L: float = 1.0) -> np.ndarray:
    """Coordonnees des centres de cellules le long d'un axe : (i + 0.5)/n * L, i = 0..n-1."""
    return (np.arange(n) + 0.5) / n * L


def meshgrid_xy(n: int, L: float = 1.0) -> list[np.ndarray]:
    """Grilles (X, Y) a centres de cellules, convention row-major de la facade `adc`.

    Renvoie X, Y de forme (n, n) avec X[j, i] = x_i et Y[j, i] = y_j (indexing="xy"), de sorte
    que `field[j, i]` corresponde bien au point (X[j, i], Y[j, i]).
    """
    coord = cell_centers(n, L)
    return np.meshgrid(coord, coord, indexing="xy")
