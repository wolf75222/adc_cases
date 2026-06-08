#!/usr/bin/env python3
"""Figures du cas diocotron_dsl (equivalence DSL <-> natif).

Rejoue la meme configuration que run.py (memes grille / CI / Poisson / nombre de pas), recupere
l'etat final des deux chemins (composition native de briques vs modele ecrit en formules adc.dsl),
et trace une figure a 3 panneaux (densite natif, densite DSL, ecart |state_dsl - state_natif|) :
les deux premiers montrent des champs reels identiques, le troisieme l'ecart qui doit etre noir
(max = 0). Une seule cellule non noire = une formule DSL qui devie d'une brique du coeur. Ecrit
aussi figures/provenance.json (nombres mesures du run).

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
    # --- meme configuration que run.main() ---
    n, L = 96, 1.0
    ne0 = case.band_density(n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())
    n_steps = 60

    dn, tn, mn = case.run_native(ne0, n_i0, n_steps)
    dd, td, md, backend = case.run_dsl(ne0, n_i0, n_steps)

    diff = np.abs(dd - dn)
    max_abs = float(diff.max())
    identical = bool(np.array_equal(dd, dn))

    # --- Figure unique : 3 panneaux (natif | DSL | ecart). On montre les deux champs reels
    # (structures, identiques a l'oeil) puis l'ecart, plutot qu'un carre noir seul (qui aurait
    # l'air vide / casse). Les deux premiers panneaux prouvent que les champs sont reels et de
    # meme dynamique ; le troisieme prouve qu'ils sont bit-identiques (max = 0). ---
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    vmin, vmax = 1.0, max(float(dn.max()), float(dd.max()))
    im0 = axes[0].imshow(dn, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=[0, L, 0, L])
    axes[0].set_title("natif : briques ExB + BackgroundDensity")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$n$")
    im1 = axes[1].imshow(dd, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=[0, L, 0, L])
    axes[1].set_title("DSL : formules adc.dsl.Model")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$n$")
    # ecart : echelle fixe 0..1e-15 (a max = 0 le panneau est noir ; un seul pixel au niveau
    # machine ressortirait). On annote le max pour lever toute ambiguite "carte vide vs cassee".
    im2 = axes[2].imshow(diff, origin="lower", cmap="inferno", vmin=0.0, vmax=1e-15, extent=[0, L, 0, L])
    axes[2].set_title(r"$|n_{\mathrm{DSL}} - n_{\mathrm{natif}}|$" + "\n"
                      r"max $= %.1e$ (bit-identique : %s)" % (max_abs, identical))
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ecart (echelle 0 a 1e-15)")
    cb.formatter.set_powerlimits((0, 0)); cb.ax.yaxis.get_offset_text().set_visible(False)
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(r"diocotron_dsl : deux chemins, etat bit-identique apres %d pas (grille $%d^2$, "
                 r"CFL 0.4, backend %s)" % (n_steps, n, backend), y=1.03)
    fig.tight_layout()
    p1 = os.path.join(FIGDIR, "equivalence_heatmap.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    p2 = p1  # une seule figure desormais (l'ancien final_density.png est fusionne ici)

    # --- Provenance : nombres mesures de ce run (rien d'invente) ---
    amp0 = case.perturbation_amplitude(ne0)
    amp_dsl = case.perturbation_amplitude(dd)
    mass0 = float(ne0.sum())
    mass_drift = case.relative_drift(md, mass0)
    prov = {
        "script": "diocotron_dsl/make_figures.py",
        "command": "python diocotron_dsl/make_figures.py",
        "produces": ["equivalence_heatmap.png"],
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
    print("ecrit : %s" % os.path.join(FIGDIR, "provenance.json"))


if __name__ == "__main__":
    main()
