#!/usr/bin/env python3
"""Figures de diagnostic du cas two_species_dsl (equivalence DSL <-> natif par espece).

Re-joue exactement la physique de run.py (memes modeles, meme CI, memes 15 pas, meme CFL, meme
backend), puis trace, par espece, une figure a 3 panneaux sur la composante densite rho :
  [natif rho] | [DSL rho] | [|rho_DSL - rho_natif|]
Les deux premiers panneaux (viridis) montrent le champ reel structure (cosinus module advecte) ;
le troisieme (inferno, echelle fixe) montre l'ecart. On voit alors deux densites identiques a l'oeil
puis la preuve qu'elles matchent (ecart noir, max annote), plutot qu'un carre noir seul (qui aurait
l'air vide / casse).

Lecture attendue (cf. README sec. 4) :
  - ions  : rho_natif et rho_DSL identiques a l'oeil (modulation induite ~1e-4) ; ecart exactement
    noir (max|d| = 0), comme tout l'etat ionique (bit-identique) ;
  - electrons : rho_natif et rho_DSL identiques a l'oeil (cosinus +/-1.6 %) ; ecart sur rho exactement
    noir (rho bit-identique). L'epsilon machine 4.93e-32 de l'etat electron vit dans la composante
    rho_v (pas rho) ; on l'annote sur la figure (max|d| etat complet) pour qu'il reste documente.

Le panneau d'ecart electron utilise l'echelle fixe 0..1e-30 (au niveau du bruit de reassociation FP
du RHS de Poisson partage) : si un jour rho cessait d'etre bit-identique, le residu y ressortirait ;
a max = 0 il reste noir. Les ions gardent l'echelle stricte 0..1e-15 (bit-identique attendu).

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

# Import du cas lui-meme : on reutilise ses modeles et sa boucle, aucune divergence de parametre.
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


def density_triptych(rho_native, rho_dsl, species_label, state_max_abs, png_path,
                     diff_vmax, backend):
    """Figure a 3 panneaux sur la densite rho d'une espece.

    Panneau 0 : rho natif (viridis, champ reel structure).
    Panneau 1 : rho DSL  (viridis, meme echelle -> identique a l'oeil).
    Panneau 2 : |rho_DSL - rho_natif| (inferno, echelle fixe 0..diff_vmax -> noir si bit-identique).

    @p state_max_abs : max|DSL - natif| sur l'etat complet de l'espece (toutes composantes), annote
    sur le panneau d'ecart pour exposer l'epsilon machine meme s'il vit hors de rho.
    Renvoie max|rho_DSL - rho_natif|.
    """
    diff = np.abs(rho_dsl - rho_native)
    rho_max_abs = float(diff.max())

    # Echelle viridis commune aux deux champs (sinon un decalage de colormap simulerait une difference).
    vmin = float(min(rho_native.min(), rho_dsl.min()))
    vmax = float(max(rho_native.max(), rho_dsl.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    im0 = axes[0].imshow(rho_native, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                         extent=[0, 1, 0, 1])
    axes[0].set_title("natif : composition de briques\n" + r"$\rho$ (champ reel)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\rho$")

    im1 = axes[1].imshow(rho_dsl, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                         extent=[0, 1, 0, 1])
    axes[1].set_title("DSL : formules adc.dsl.Model\n" + r"$\rho$ (champ reel)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\rho$")

    im2 = axes[2].imshow(diff, origin="lower", cmap="inferno", vmin=0.0, vmax=diff_vmax,
                         extent=[0, 1, 0, 1])
    axes[2].set_title(
        r"$|\rho_{\mathrm{DSL}} - \rho_{\mathrm{natif}}|$" + "\n"
        + r"max$|d|_\rho = %.2e$   (etat complet : %.2e)" % (rho_max_abs, state_max_abs))
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04,
                      label="ecart (echelle 0 a %.0e)" % diff_vmax)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.get_offset_text().set_visible(False)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    fig.suptitle(
        "%s : deux chemins, densite bit-identique apres %d pas "
        "(grille $%d^2$, CFL 0.4, backend %s)" % (species_label, NSTEPS, N, backend),
        y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return rho_max_abs


def main():
    ne2d, ni2d = case.initial_conditions(N)

    # Reference native + chemin DSL, exactement comme run.py (meme fallback de backend).
    en, inn, me_n, mi_n = case.run_native(N, ne2d, ni2d, NSTEPS)
    ed, idd, me_d, mi_d, backend = case.run_dsl(N, ne2d, ni2d, NSTEPS)

    en = np.asarray(en)
    ed = np.asarray(ed)
    inn = np.asarray(inn)
    idd = np.asarray(idd)

    diff_e = np.abs(ed - en)   # (4, n, n)
    diff_i = np.abs(idd - inn)  # (3, n, n)
    max_e = float(diff_e.max())
    max_i = float(diff_i.max())

    # Figures 3 panneaux sur rho (index 0 = densite conservative pour les deux modeles).
    # Electrons : echelle d'ecart 0..1e-30 (niveau du bruit de reassociation FP partage) ;
    # ions : echelle stricte 0..1e-15 (bit-identique attendu).
    rho_max_e = density_triptych(
        en[0], ed[0], "electrons (Euler compressible)", max_e,
        os.path.join(FIGDIR, "equivalence_electrons.png"), diff_vmax=1e-30, backend=backend)
    rho_max_i = density_triptych(
        inn[0], idd[0], "ions (Euler isotherme)", max_i,
        os.path.join(FIGDIR, "equivalence_ions.png"), diff_vmax=1e-15, backend=backend)

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
        "max_abs_diff_rho_electrons": float(rho_max_e),
        "max_abs_diff_rho_ions": float(rho_max_i),
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
    print("electrons : max|DSL - natif| (etat) = %.3e | sur rho = %.3e (bit-identique etat = %s)"
          % (max_e, rho_max_e, bit_e))
    print("ions      : max|DSL - natif| (etat) = %.3e | sur rho = %.3e (bit-identique etat = %s)"
          % (max_i, rho_max_i, bit_i))
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
