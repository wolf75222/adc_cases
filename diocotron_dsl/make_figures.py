#!/usr/bin/env python3
"""Figures du cas diocotron_dsl (equivalence DSL <-> natif).

Rejoue la MEME configuration que run.py (memes grille / CI / Poisson / nombre de pas), recupere
l'etat final des DEUX chemins (composition native de briques vs modele ecrit en formules adc.dsl),
et trace la SEULE figure qui prouve l'equivalence : la carte de |state_dsl - state_natif|, qui DOIT
etre identiquement noire (max = 0). Une seule cellule non noire = une formule DSL qui devie d'une
brique du coeur (ExBVelocity / BackgroundDensity). Ecrit aussi le panneau des deux densites finales
(controle visuel : meme dynamique) et figures/provenance.json (nombres mesures du run).

Lancement (meme interpreteur que celui qui a compile _adc) :
  PYTHONPATH=<build>/python:/private/tmp/adc_cases-deeptut python3.12 make_figures.py
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adc  # noqa: F401  (sert a localiser le module pour la provenance)

# run.py est dans le meme dossier : on importe ses fonctions (memes parametres, meme physique).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as case  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    # --- MEME configuration que run.main() ---
    n, L = 96, 1.0
    ne0 = case.band_density(n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())
    n_steps = 60

    dn, tn, mn = case.run_native(ne0, n_i0, n_steps)
    dd, td, md, backend = case.run_dsl(ne0, n_i0, n_steps)

    diff = np.abs(dd - dn)
    max_abs = float(diff.max())
    identical = bool(np.array_equal(dd, dn))

    # --- Figure 1 : carte d'equivalence |DSL - natif| (DOIT etre identiquement noire) ---
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # echelle fixe 0..1e-15 : a max = 0 toute la carte sature au noir ; un seul pixel au niveau
    # machine (1e-15) ressortirait clairement. On NE laisse PAS matplotlib auto-scaler (sinon le
    # bruit numerique serait dramatise sur une carte pourtant exactement nulle).
    im = ax.imshow(diff, origin="lower", cmap="inferno", vmin=0.0, vmax=1e-15,
                   extent=[0, L, 0, L])
    cb = fig.colorbar(im, ax=ax, label=r"$|n_{\mathrm{DSL}} - n_{\mathrm{natif}}|$ (echelle 0 a 1e-15)")
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.get_offset_text().set_visible(False)
    ax.set_title("Equivalence bit : carte de l'ecart DSL - natif\n"
                 r"max $= %.1e$ (bit-identique : %s, backend : %s)" % (max_abs, identical, backend))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    p1 = os.path.join(FIGDIR, "equivalence_heatmap.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # --- Figure 2 : controle visuel des deux etats finaux (meme dynamique diocotron) ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    vmax = max(float(dn.max()), float(dd.max()))
    for ax, field, title in (
            (axes[0], dn, "natif (briques ExB + BackgroundDensity)"),
            (axes[1], dd, "DSL (formules adc.dsl.Model)"),
            (axes[2], diff, "ecart |DSL - natif|")):
        if title.startswith("ecart"):
            im = ax.imshow(field, origin="lower", cmap="inferno", vmin=0.0, vmax=1e-15,
                           extent=[0, L, 0, L])
            ax.set_title(title + r" (max $= %.1e$)" % max_abs)
        else:
            im = ax.imshow(field, origin="lower", cmap="viridis", vmin=1.0, vmax=vmax,
                           extent=[0, L, 0, L])
            ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(r"diocotron_dsl : densite finale $n$ apres %d pas (grille $%d^2$, CFL 0.4)"
                 % (n_steps, n), y=1.02)
    fig.tight_layout()
    p2 = os.path.join(FIGDIR, "final_density.png")
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Provenance : nombres MESURES de ce run (rien d'invente) ---
    amp0 = case.perturbation_amplitude(ne0)
    amp_dsl = case.perturbation_amplitude(dd)
    mass0 = float(ne0.sum())
    mass_drift = case.relative_drift(md, mass0)
    prov = {
        "script": "diocotron_dsl/make_figures.py",
        "command": "python diocotron_dsl/make_figures.py",
        "produces": ["equivalence_heatmap.png", "final_density.png"],
        "adc_cpp_sha": git_sha("/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"),
        "adc_cases_sha": git_sha(HERE),
        "backend": backend,
        "backend_preference": ["production", "aot"],
        "resolution": "%dx%d" % (n, n),
        "n_steps": n_steps,
        "cfl": 0.4,
        "n_i0": n_i0,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "adc_module": adc.__file__,
        "max_abs_diff": max_abs,
        "bit_identical": identical,
        "t_native": tn,
        "t_dsl": td,
        "mass_native": mn,
        "mass_dsl": md,
        "mass_drift_rel_dsl": mass_drift,
        "amp_initial": amp0,
        "amp_final_dsl": amp_dsl,
        "amp_growth_factor": amp_dsl / amp0,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("backend %r | max|DSL - natif| = %.3e | bit-identique = %s" % (backend, max_abs, identical))
    print("ecrit : %s" % p1)
    print("ecrit : %s" % p2)
    print("ecrit : %s" % os.path.join(FIGDIR, "provenance.json"))


if __name__ == "__main__":
    main()
