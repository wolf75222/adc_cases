#!/usr/bin/env python3
"""GIF de l'instabilite diocotron sur AMR : reproduit (en version locale) la figure hero du README
adc_cpp (`docs/anim_romeo_diocotron_amr3.gif`).

PORTEE HONNETE : la facade Python `adc.AmrSystem` raffine sur UN niveau fin multi-patch
(Berger-Rigoutsos, regrid d'union des tags), pas sur 3 niveaux. La figure hero du README a ete
produite par le MOTEUR C++ multi-niveaux (`advance_amr`) sur ROMEO (GH200), non expose dans la
facade Python. Ce script reproduit donc le VISUEL (l'instabilite diocotron suivie par un AMR
adaptatif, comparee a une grille uniforme), pas les 3 niveaux exacts du run ROMEO. Le binding
n'expose pas la GEOMETRIE des patchs (`patch_boxes`), seulement leur NOMBRE (`n_patches()`) : on
annote donc le compte de patchs, on ne dessine pas les rectangles.

Produit `figures/diocotron_amr_hero.gif` (uniforme | AMR cote a cote) + `figures/provenance.json`.
Lancement : PYTHONPATH=<build>/python:. python3 diocotron_amr/make_hero_gif.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import adc
from adc_cases import models
from adc_cases.common.initial_conditions import band_density

N, L = 128, 1.0
MODE, DISP, WIDTH = 4, 0.05, 0.05
B0, ALPHA = 1.0, 1.0
NFRAMES, STEPS_PER_FRAME, CFL = 48, 6, 0.4
THRESHOLD = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


def _diocotron(n_i0):
    return models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0)


def build_amr(ne, n_i0):
    sim = adc.AmrSystem(n=N, L=L, regrid_every=10, periodic=True)
    sim.add_block("ne", model=_diocotron(n_i0), spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_refinement(threshold=THRESHOLD)          # un niveau fin, tagge ou la densite varie
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def build_uniform(ne, n_i0):
    sim = adc.System(n=N, L=L, periodic=True)
    sim.add_block("ne", model=_diocotron(n_i0), spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ne = band_density(N, L, amp=1.0, width=WIDTH, mode=MODE, disp=DISP)
    n_i0 = float(ne.mean())
    su, sa = build_uniform(ne, n_i0), build_amr(ne, n_i0)

    frames_u, frames_a, npatch = [], [], []
    for _ in range(NFRAMES):
        frames_u.append(np.asarray(su.density("ne")).copy())
        frames_a.append(np.asarray(sa.density("ne")).copy())
        npatch.append(int(sa.n_patches()))
        for _ in range(STEPS_PER_FRAME):
            su.step_cfl(CFL); sa.step_cfl(CFL)

    vmax = max(max(f.max() for f in frames_u), max(f.max() for f in frames_a))
    fig, (axu, axa) = plt.subplots(1, 2, figsize=(8.4, 4.3))
    imu = axu.imshow(frames_u[0].T, origin="lower", cmap="inferno", vmin=1.0, vmax=vmax, extent=[0, L, 0, L])
    ima = axa.imshow(frames_a[0].T, origin="lower", cmap="inferno", vmin=1.0, vmax=vmax, extent=[0, L, 0, L])
    axu.set_title("grille uniforme"); axa.set_title("AMR (1 niveau fin, multi-patch)")
    for ax in (axu, axa):
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(ima, ax=axa, fraction=0.046, label=r"$n_e$")
    sup = fig.suptitle("")

    def update(k):
        imu.set_data(frames_u[k].T); ima.set_data(frames_a[k].T)
        sup.set_text("diocotron mode l=%d sur AMR : %d patchs fins (pas %d/%d)"
                     % (MODE, npatch[k], k * STEPS_PER_FRAME, NFRAMES * STEPS_PER_FRAME))
        return imu, ima, sup

    anim = animation.FuncAnimation(fig, update, frames=NFRAMES, interval=90, blit=False)
    gif = os.path.join(FIGDIR, "diocotron_amr_hero.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=11))
    plt.close(fig)

    # image de couverture PNG (derniere trame) pour les exports statiques
    figc, (a0, a1) = plt.subplots(1, 2, figsize=(8.4, 4.3))
    a0.imshow(frames_u[-1].T, origin="lower", cmap="inferno", vmin=1.0, vmax=vmax, extent=[0, L, 0, L])
    a1.imshow(frames_a[-1].T, origin="lower", cmap="inferno", vmin=1.0, vmax=vmax, extent=[0, L, 0, L])
    a0.set_title("uniforme (final)"); a1.set_title("AMR (final, %d patchs)" % npatch[-1])
    for ax in (a0, a1):
        ax.set_xticks([]); ax.set_yticks([])
    figc.suptitle("diocotron mode l=%d : uniforme vs AMR (etat final)" % MODE)
    figc.tight_layout(); figc.savefig(os.path.join(FIGDIR, "diocotron_amr_hero_cover.png"), dpi=120)
    plt.close(figc)

    adc_cpp_root = os.path.abspath(os.path.join(os.path.dirname(adc.__file__), "..", "..", ".."))
    prov = {
        "script": "diocotron_amr/make_hero_gif.py",
        "command": "python diocotron_amr/make_hero_gif.py",
        "produces": ["diocotron_amr_hero.gif", "diocotron_amr_hero_cover.png"],
        "reproduit": "version LOCALE de docs/anim_romeo_diocotron_amr3.gif (README adc_cpp)",
        "difference_avec_hero": "facade Python AmrSystem = 1 niveau fin multi-patch ; le hero ROMEO = moteur C++ multi-niveaux (3 niveaux), non expose en Python",
        "adc_cpp_sha": git_sha(adc_cpp_root),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.AmrSystem, briques models.diocotron)",
        "resolution": "%dx%d (base)" % (N, N),
        "mode": MODE, "nframes": NFRAMES, "steps_per_frame": STEPS_PER_FRAME, "cfl": CFL,
        "n_patches_final": npatch[-1], "n_patches_max": max(npatch),
        "python": "%d.%d.%d" % sys.version_info[:3], "adc_module": adc.__file__,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print("patchs fins : min=%d max=%d final=%d" % (min(npatch), max(npatch), npatch[-1]))
    print("ecrit : %s" % gif)
    print("ecrit : %s" % os.path.join(FIGDIR, "diocotron_amr_hero_cover.png"))


if __name__ == "__main__":
    main()
