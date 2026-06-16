#!/usr/bin/env python3
"""Verification CI legere du cas tutorial : le diocotron construit de trois facons (helper specialise,
briques natives, formules DSL) donne un etat bit-identique. Pas de figures (matplotlib non requis),
moins de pas que run.py, juste le coeur du cas, assez rapide pour la CI (needs = ["cxx"] pour le DSL).

Reutilise les constructeurs de run.py (meme dossier) pour ne pas dupliquer la physique : la verite du
tutoriel et celle testee en CI sont alors le meme code.

Lancement : PYTHONPATH=<build>/python:. python3 tutorial/equivalence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import adc  # noqa: F401  (importe tot : echoue clairement si le module n'est pas sur le PYTHONPATH)

# run.py vit dans ce dossier : on l'ajoute au chemin d'import pour reutiliser ses constructeurs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (  # noqa: E402
    add_bricks_block,
    add_dsl_block,
    compile_dsl,
    diocotron_from_bricks,
    diocotron_from_dsl,
    make_system,
)

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import models  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402

B0, ALPHA = 1.0, 1.0


def _final_density(sim: adc.System, n_steps: int) -> np.ndarray:
    for _ in range(n_steps):
        sim.step_cfl(0.4)
    return np.asarray(sim.density("ne"))


def _build_and_run(
    add_block_fn, model_or_compiled, ne0: np.ndarray, n_steps: int
) -> np.ndarray:
    sim = make_system(ne0)
    add_block_fn(sim, model_or_compiled)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne0)
    return _final_density(sim, n_steps)


def main() -> None:
    n, n_steps = 64, 30
    ne0 = band_density(n, 1.0, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())

    d_helper = _build_and_run(
        add_bricks_block,
        models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0),
        ne0,
        n_steps,
    )
    d_bricks = _build_and_run(
        add_bricks_block, diocotron_from_bricks(n_i0), ne0, n_steps
    )
    compiled, backend = compile_dsl(diocotron_from_dsl(n_i0))
    d_dsl = _build_and_run(add_dsl_block, compiled, ne0, n_steps)

    eq_hb = bool(np.array_equal(d_helper, d_bricks))
    eq_bd = bool(np.array_equal(d_bricks, d_dsl))
    print(
        "backend DSL %r : helper==briques %s, briques==DSL %s "
        "(max|b-DSL| = %.3e)"
        % (backend, eq_hb, eq_bd, float(np.max(np.abs(d_bricks - d_dsl))))
    )
    assert eq_hb, "helper models.diocotron != composition de briques"
    assert (
        eq_bd
    ), "modele DSL != briques (une formule diverge des conventions du coeur)"
    print(
        "OK tutorial/equivalence (diocotron helper == briques == DSL, bit-identique)"
    )


if __name__ == "__main__":
    main()
