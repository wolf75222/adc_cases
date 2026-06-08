#!/usr/bin/env python3
"""Figures de diagnostic du cas two_species_dsl (equivalence DSL <-> natif par espece).

Re-joue EXACTEMENT la physique de run.py (memes modeles, meme CI, memes 15 pas, meme CFL, meme
backend), puis trace, PAR ESPECE et PAR COMPOSANTE conservative, la heatmap de |etat_DSL - etat_natif|.

Lecture attendue (cf. README sec. 5) :
  - ions : toutes les composantes IDENTIQUEMENT NOIRES (max|d| = 0, bit-identique) ;
  - electrons : composantes a ~1e-32 (reassociation flottante de l'accumulation du RHS de
    Poisson partage), tres en-dessous de la tolerance machine 1e-24.

Sorties : figures/equivalence_electrons.png, figures/equivalence_ions.png, figures/provenance.json
Tout sous le dossier du cas (assets versionnes du tutoriel d'equivalence).
"""

import json
import os
import platform
import subprocess

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import adc  # noqa: E402

# Import du cas lui-meme : on reutilise SES modeles et SA boucle, aucune divergence de parametre.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as case  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

N, NSTEPS = case.main.__defaults__ if case.main.__defaults__ else (48, 15)
# main() fixe n, n_steps = 48, 15 en dur ; on relit les memes valeurs explicitement.
N, NSTEPS = 48, 15

ELEC_VARS = ["rho", "rho_u", "rho_v", "E"]
ION_VARS = ["rho", "rho_u", "rho_v"]


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def heatmap_species(diff, var_names, title, png_path, tol):
    """diff : (n_vars, n, n) = |DSL - natif| par composante. Une rangee de heatmaps."""
    nv = diff.shape[0]
    fig, axes = plt.subplots(1, nv, figsize=(3.2 * nv, 3.6), squeeze=False)
    maxall = float(diff.max())
    for k in range(nv):
        ax = axes[0][k]
        d = diff[k]
        dmax = float(d.max())
        # echelle commune a toutes les composantes de l'espece (vmax = max global, ou tol si nul)
        vmax = maxall if maxall > 0 else tol
        im = ax.imshow(d, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax,
                       extent=[0, 1, 0, 1])
        ax.set_title("%s\nmax|d| = %.3e" % (var_names[k], dmax), fontsize=10)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if k == 0:
            ax.set_ylabel("y")
        ax.set_xlabel("x")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("|etat_DSL - etat_natif|")
    fig.suptitle(title, fontsize=12)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return maxall


def main():
    ne2d, ni2d = case.initial_conditions(N)

    # Reference native + chemin DSL, EXACTEMENT comme run.py (meme fallback de backend).
    en, inn, me_n, mi_n = case.run_native(N, ne2d, ni2d, NSTEPS)
    ed, idd, me_d, mi_d, backend = case.run_dsl(N, ne2d, ni2d, NSTEPS)

    diff_e = np.abs(np.asarray(ed) - np.asarray(en))  # (4, n, n)
    diff_i = np.abs(np.asarray(idd) - np.asarray(inn))  # (3, n, n)

    max_e = heatmap_species(
        diff_e, ELEC_VARS,
        "electrons (Euler compressible, 4 var) : |DSL - natif| par composante",
        os.path.join(FIGDIR, "equivalence_electrons.png"), tol=1e-24)
    max_i = heatmap_species(
        diff_i, ION_VARS,
        "ions (Euler isotherme, 3 var) : |DSL - natif| par composante",
        os.path.join(FIGDIR, "equivalence_ions.png"), tol=1e-24)

    bit_e = bool(np.array_equal(ed, en))
    bit_i = bool(np.array_equal(idd, inn))

    prov = {
        "script": "two_species_dsl/make_figures.py",
        "command": "python two_species_dsl/make_figures.py",
        "produces": ["equivalence_electrons.png", "equivalence_ions.png"],
        "adc_cpp_sha": _git_sha(os.path.dirname(adc.__file__) + "/../../.."),
        "adc_cases_sha": _git_sha(os.path.dirname(HERE)),
        "backend_dsl": backend,
        "backend_native": "natif serie (adc.System, models.electron_euler + ion_isothermal)",
        "resolution": "%dx%d" % (N, N),
        "n_steps": NSTEPS,
        "cfl": 0.4,
        "tol_equivalence": 1e-24,
        "q_e": case.Q_E, "q_i": case.Q_I,
        "gamma_e": case.GAMMA_E, "cs2_i": case.CS2_I,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "compiler": _compiler_string(),
        "adc_module": adc.__file__,
        "max_abs_diff_electrons": float(max_e),
        "max_abs_diff_ions": float(max_i),
        "electrons_bit_identical": bit_e,
        "ions_bit_identical": bit_i,
        "max_abs_diff_per_var_electrons": {ELEC_VARS[k]: float(diff_e[k].max())
                                           for k in range(diff_e.shape[0])},
        "max_abs_diff_per_var_ions": {ION_VARS[k]: float(diff_i[k].max())
                                      for k in range(diff_i.shape[0])},
        "mass_drift_rel_electrons": float(case.relative_drift(me_d, float(ne2d.sum()))),
        "mass_drift_rel_ions": float(case.relative_drift(mi_d, float(ni2d.sum()))),
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("backend DSL : %r" % backend)
    print("electrons : max|DSL - natif| = %.3e (bit-identique = %s)" % (max_e, bit_e))
    print("ions      : max|DSL - natif| = %.3e (bit-identique = %s)" % (max_i, bit_i))
    print("figures + provenance.json ecrits dans %s" % FIGDIR)


def _compiler_string():
    for cxx in (os.environ.get("CXX"), "c++", "clang++", "g++"):
        if not cxx:
            continue
        try:
            return subprocess.check_output([cxx, "--version"], text=True,
                                           stderr=subprocess.DEVNULL).splitlines()[0]
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


if __name__ == "__main__":
    main()
