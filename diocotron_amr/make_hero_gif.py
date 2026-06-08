#!/usr/bin/env python3
"""GIF hero de l'instabilite diocotron suivie par AMR : reproduit le TYPE de la figure hero du
README adc_cpp (`docs/anim_romeo_diocotron_amr3.gif`), avec les VRAIS patchs fins du solveur.

CE QUE MONTRE LA FIGURE : un seul panneau, bande de charge horizontale perturbee au mode l=2, qui
s'enroule en oeil-de-chat (Kelvin-Helmholtz du diocotron), suivie par les CADRES DE RAFFINEMENT AMR
REELS qui collent au coeur dense. Fond sombre, colormap inferno, titre "diocotron AMR : densite n_e".

PORTEE HONNETE (lire avant de citer la figure) :
  - LA PHYSIQUE EST REELLE : la bande est advectee par le vrai solveur (derive E x B du modele
    `models.diocotron`, Poisson de charge resolu par multigrille geometrique sur `adc.AmrSystem`).
    L'enroulement en vortex est la VRAIE sortie du code, pas une animation scriptee.
  - LES CADRES SONT REELS (plus de proxy). Ils sont la GEOMETRIE EXACTE des patchs fins, lue par
    `AmrSystem.patch_rectangles()` (binding `patch_boxes()`). Plus aucune reconstruction par seuils de
    densite ni scipy : chaque rectangle est un patch que le moteur a effectivement raffine. Le
    raffinement suit la bande (criteres `set_refinement(threshold)` au-dessus du plancher) et evolue
    a chaque regrid : les patchs se deplacent et se multiplient quand l'instabilite s'enroule.
  - DIFFERENCE AVEC LE HERO ROMEO : la facade Python `adc.AmrSystem` raffine sur UN niveau fin
    multi-patch (Berger-Rigoutsos). Le hero du README a ete produit par le MOTEUR C++ multi-niveaux
    (`advance_amr`, 3 niveaux, GH200). Ici les patchs sont colores PAR NIVEAU (cyan = niveau 1, vert =
    2, rouge = 3) : aujourd'hui seul le niveau 1 apparait (facade = 1 niveau fin), le code est pret si
    un futur expose plus de niveaux. Le nombre reel de patchs est aussi consigne dans provenance.json.

Produit `figures/diocotron_amr_hero.gif` + `figures/diocotron_amr_hero_cover.png` + `provenance.json`.
Lancement : PYTHONPATH=<build>/python:. python3 diocotron_amr/make_hero_gif.py
(Requiert un module `adc` exposant patch_boxes()/patch_rectangles() : adc_cpp >= PR patch-boxes.)
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

# --- physique / discretisation (mode l=2 = 2 vortices, comme le hero) ---
N, L = 144, 1.0
MODE, DISP, WIDTH = 2, 0.04, 0.06
B0, ALPHA = 1.0, 1.0
NFRAMES, STEPS_PER_FRAME, CFL = 44, 6, 0.4
REGRID_EVERY = 6          # regrid une fois par trame -> les patchs suivent la bande
# Seuil de tag AU-DESSUS du plancher (band_density: floor=1.0, pic ~2.0) : sans ca, toute la grille
# (densite >= 1 partout) serait taggee et le niveau fin tuilerait tout le domaine (non adaptatif).
THRESHOLD = 1.4
FLOOR = 1.0

# Couleur du cadre PAR NIVEAU AMR (cyan = niveau 1, vert = 2, rouge = 3). La facade ne produit que
# le niveau 1 aujourd'hui ; le dict couvre les niveaux superieurs au cas ou (lecture honnete : la
# couleur trace le VRAI niveau du patch, pas un seuil de densite invente).
LEVEL_COLORS = {1: "#00e5ff", 2: "#39ff14", 3: "#ff2d2d"}
LEVEL_LW = {1: 1.3, 2: 1.6, 3: 1.9}

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


def build_amr(ne, n_i0):
    """AmrSystem mono-bloc diocotron : derive E x B + Poisson de charge, 1 niveau fin multi-patch,
    regrid periodique (les patchs se replacent sur la bande a chaque regrid)."""
    sim = adc.AmrSystem(n=N, L=L, regrid_every=REGRID_EVERY, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0),
                  spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_refinement(threshold=THRESHOLD)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def patch_rects_by_level(sim):
    """Renvoie une liste de (level, x0, y0, w, h) : les VRAIS patchs fins, niveau + rectangle physique.
    Lit patch_boxes() (level + coins index) et convertit en [0, L]^2 (dx = L / (n << level))."""
    n = sim.nx()
    out = []
    for level, ilo, jlo, ihi, jhi in sim.patch_boxes():
        dx = L / (n << level)
        out.append((level, ilo * dx, jlo * dx, (ihi - ilo + 1) * dx, (jhi - jlo + 1) * dx))
    return out


def draw_panel(ax, ne, vmax, rects, Rectangle):
    """Trace un panneau hero : champ inferno sur fond sombre + VRAIS patchs AMR (colores par niveau)."""
    im = ax.imshow(ne, origin="lower", cmap="inferno", vmin=FLOOR, vmax=vmax, extent=[0, L, 0, L])
    for level, x0, y0, w, h in rects:
        col = LEVEL_COLORS.get(level, "#ffffff")
        lw = LEVEL_LW.get(level, 1.3)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=col, lw=lw, alpha=0.95))
    ax.set_xticks([])
    ax.set_yticks([])
    return im


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
    from matplotlib.patches import Rectangle

    ne0 = band_density(N, L, amp=1.0, width=WIDTH, mode=MODE, disp=DISP)
    n_i0 = float(ne0.mean())
    sim = build_amr(ne0, n_i0)

    fields, rects_per_frame, npatch = [], [], []
    for _ in range(NFRAMES):
        fields.append(np.asarray(sim.density("ne")).copy())
        rects_per_frame.append(patch_rects_by_level(sim))   # VRAIS patchs de la trame
        npatch.append(int(sim.n_patches()))
        for _ in range(STEPS_PER_FRAME):
            sim.step_cfl(CFL)
    vmax = max(float(f.max()) for f in fields)

    bg = "#0b0b1a"
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    draw_panel(ax, fields[0], vmax, rects_per_frame[0], Rectangle)
    ax.set_title("diocotron AMR : densite $n_e$", color="white", fontsize=13, pad=8)

    def update(k):
        ax.clear()
        ax.set_facecolor(bg)
        draw_panel(ax, fields[k], vmax, rects_per_frame[k], Rectangle)
        ax.set_title("diocotron AMR : densite $n_e$", color="white", fontsize=13, pad=8)
        return []

    anim = animation.FuncAnimation(fig, update, frames=NFRAMES, interval=90, blit=False)
    gif = os.path.join(FIGDIR, "diocotron_amr_hero.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=12), savefig_kwargs={"facecolor": bg})
    plt.close(fig)

    # cover PNG (derniere trame)
    figc, axc = plt.subplots(figsize=(5.2, 5.2))
    figc.patch.set_facecolor(bg)
    axc.set_facecolor(bg)
    draw_panel(axc, fields[-1], vmax, rects_per_frame[-1], Rectangle)
    axc.set_title("diocotron AMR : densite $n_e$ (etat final, %d patchs)" % npatch[-1],
                  color="white", fontsize=12, pad=8)
    cover = os.path.join(FIGDIR, "diocotron_amr_hero_cover.png")
    figc.savefig(cover, dpi=110, facecolor=bg)
    plt.close(figc)

    adc_cpp_root = os.path.abspath(os.path.join(os.path.dirname(adc.__file__), "..", "..", ".."))
    levels_seen = sorted({lvl for rs in rects_per_frame for (lvl, *_rest) in rs})
    prov = {
        "script": "diocotron_amr/make_hero_gif.py",
        "command": "python diocotron_amr/make_hero_gif.py",
        "produces": ["diocotron_amr_hero.gif", "diocotron_amr_hero_cover.png"],
        "reproduit": "le TYPE de docs/anim_romeo_diocotron_amr3.gif (README adc_cpp) : panneau unique, "
                     "bande mode l=2 enroulee en oeil-de-chat, cadres AMR suivant le coeur dense",
        "physique_reelle": "advection E x B + Poisson de charge (multigrille) par le vrai solveur "
                           "adc.AmrSystem (models.diocotron) ; l'enroulement KH est la sortie du code",
        "cadres": "REELS : geometrie exacte des patchs fins via AmrSystem.patch_boxes() / "
                  "patch_rectangles() (binding patch-boxes). AUCUN proxy de densite, AUCUN scipy. "
                  "Colores par niveau (1=cyan, 2=vert, 3=rouge) ; niveaux observes : %s." % levels_seen,
        "difference_avec_hero": "facade Python AmrSystem = 1 niveau fin multi-patch (niveaux observes "
                                "ci-dessus) ; le hero ROMEO = moteur C++ multi-niveaux (advance_amr, "
                                "3 niveaux reels, GH200)",
        "adc_cpp_sha": git_sha(adc_cpp_root),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.AmrSystem, brique models.diocotron, Poisson geometric_mg)",
        "resolution": "%dx%d (grille de base)" % (N, N),
        "mode": MODE, "disp": DISP, "width": WIDTH, "threshold": THRESHOLD, "regrid_every": REGRID_EVERY,
        "nframes": NFRAMES, "steps_per_frame": STEPS_PER_FRAME, "cfl": CFL,
        "n_patches_final": npatch[-1], "n_patches_max": max(npatch), "n_patches_min": min(npatch),
        "levels_observed": levels_seen, "vmax": vmax,
        "python": "%d.%d.%d" % sys.version_info[:3], "adc_module": adc.__file__,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("patchs fins REELS (via patch_boxes) : min=%d max=%d final=%d ; niveaux %s"
          % (min(npatch), max(npatch), npatch[-1], levels_seen))
    print("ecrit : %s" % gif)
    print("ecrit : %s" % cover)


if __name__ == "__main__":
    main()
