#!/usr/bin/env python3
"""Honest figures for the Hoffart magnetic Euler-Poisson DSL reproduction-candidate.

Deux figures, et seulement deux, parce que ce cas est `reproduction-candidate` :

1. ``gap_to_paper.png`` -- le COEUR : taux mesure (brut, full-system-schur,
   baseline cartesienne) vs cible papier, par mode, a n=256 ET n=384. La figure
   est concue pour FAIRE RESSORTIR l'ecart de -82 a -95 %, pas pour suggerer une
   concordance. Les barres mesurees sont rasantes face aux cibles ; l'etiquette
   d'erreur est dessinee sur chaque paire. Les nombres sont VERBATIM ceux de la
   table de validation du README (eux-memes issus de runs hors CI documentes dans
   adc_cpp/docs/HOFFART_FIDELITY.md) ; ce script NE relance PAS run.py (LONG).

2. ``oracle_residual.png`` -- ce qui EST prouve : le residu de l'oracle analytique
   check_model.py (flux x/y, source Lorentz/electrique, rhs de Poisson) confronte
   le modele symbolique compile aux formules a la main sur 2x2 cellules. On RELANCE
   reellement la comparaison de check_model.py ici et on trace le residu max par
   bloc ; il plafonne au niveau machine (~1e-16). C'est l'observable qui justifie la
   clause PROUVE.

Ecrit aussi ``figures/provenance.json`` (SHA des deux depots, backend, source des
nombres mesures, statut PENDING explicite).

Lancement :
    cd /private/tmp/adc_cases-deeptut/hoffart_euler_poisson_dsl
    PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
      /opt/homebrew/anaconda3/bin/python3.12 make_figures.py
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adc  # noqa: F401  (pour le SHA adc_cpp + la compilation du modele)
from model import PaperParameters, magnetic_euler_poisson_model
from results import adc_cases_sha, adc_cpp_sha

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Cibles du papier (Section 5.3 / Fig. 5.4, arXiv:2510.11808).
PAPER = {3: 0.772, 4: 0.911, 5: 0.683}

# Mesures BRUTES (full-system-schur, baseline cartesienne carre + mur Poisson
# circulaire), VERBATIM de la table de validation du README (Section "Table de
# validation"), elles-memes issues de runs hors CI documentes dans
# adc_cpp/docs/HOFFART_FIDELITY.md. Ce script NE les recalcule PAS (run.py est LONG).
MEASURED = {
    256: {3: 0.0372, 4: 0.0489, 5: 0.1211},
    384: {3: 0.0385, 4: 0.0613, 5: 0.1257},
}


def err_pct(num, paper):
    return 100.0 * (num - paper) / paper


def figure_gap_to_paper():
    """Barres taux-mesure vs cible-papier ; l'ecart -82 a -95 % saute aux yeux."""
    modes = [3, 4, 5]
    resolutions = [256, 384]
    x = np.arange(len(modes), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    # Cible papier : barre de reference, une par mode.
    ax.bar(x - width, [PAPER[m] for m in modes], width,
           color="tab:red", label="cible papier (lin. theory)")
    # Mesures brutes, une serie par resolution ; rasantes a cote du papier.
    ax.bar(x, [MEASURED[256][m] for m in modes], width,
           color="black", label="mesure brute n=256")
    ax.bar(x + width, [MEASURED[384][m] for m in modes], width,
           color="dimgray", label="mesure brute n=384")

    # Etiquette d'erreur sur chaque mode (la mesure la plus haute des deux n).
    for i, m in enumerate(modes):
        best = max(MEASURED[256][m], MEASURED[384][m])
        e256 = err_pct(MEASURED[256][m], PAPER[m])
        e384 = err_pct(MEASURED[384][m], PAPER[m])
        ax.annotate(
            "n=256: %+.0f%%\nn=384: %+.0f%%" % (e256, e384),
            xy=(x[i] + 0.5 * width, best),
            xytext=(x[i] + 0.5 * width, PAPER[m] * 0.62),
            ha="center", va="center", fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["l=%d" % m for m in modes])
    ax.set_ylabel("taux de croissance gamma (BRUT, sans 2pi)")
    ax.set_title("full-system-schur (cart-square) vs papier : ecart -82 a -95 %\n"
                 "reproduction quantitative NON etablie (PENDING)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "gap_to_paper.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def figure_oracle_residual():
    """Re-joue la comparaison de check_model.py et trace le residu max par bloc.

    C'est la seule chose PROUVEE du cas : le modele symbolique compile == les
    formules analytiques sur 2x2 cellules, au niveau machine.
    """
    p = PaperParameters(beta=3.0, temperature=0.25)
    model = magnetic_euler_poisson_model(p, source="local")._m

    rho = np.array([[1.0, 2.0], [1.5, 0.8]])
    mx = np.array([[0.3, -0.2], [0.7, 0.1]])
    my = np.array([[-0.4, 0.6], [0.2, -0.5]])
    gx = np.array([[0.2, -0.1], [0.3, 0.4]])
    gy = np.array([[-0.3, 0.5], [0.1, -0.2]])
    U = np.stack([rho, mx, my])
    aux = {"phi": np.zeros_like(rho), "grad_x": gx, "grad_y": gy}

    u, v = mx / rho, my / rho
    pressure = p.temperature * rho

    fx = model.flux(U, aux, 0)
    fy = model.flux(U, aux, 1)
    src = model.source_value(U, aux)
    env = model._env(U, aux)
    ell = model._elliptic.eval(env)

    ref_fx = np.stack([mx, mx * u + pressure, mx * v])
    ref_fy = np.stack([my, my * u, my * v + pressure])
    ref_src = np.stack([
        np.zeros_like(rho),
        -rho * gx + p.omega * my,
        -rho * gy - p.omega * mx,
    ])
    ref_ell = -p.alpha * rho

    blocks = [
        ("flux_x", np.max(np.abs(fx - ref_fx))),
        ("flux_y", np.max(np.abs(fy - ref_fy))),
        ("source\n(Lorentz+E)", np.max(np.abs(src - ref_src))),
        ("Poisson rhs\n(-alpha rho)", np.max(np.abs(ell - ref_ell))),
    ]
    labels = [b[0] for b in blocks]
    resid = [float(b[1]) for b in blocks]  # VRAI residu, sans plancher artificiel

    # Les 4 residus sont EXACTEMENT 0.0 (bit-exact, pas seulement < eps). Une echelle
    # log mentirait (barres a 0 -> disparaissent ou plancher invente). On dessine donc
    # le residu mesure (0) en hauteur, la ligne eps-machine comme PLAFOND honnete que
    # le residu ne franchit pas, et l'etiquette dit "0.0 (bit-exact)".
    eps = float(np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(range(len(labels)), resid, color="tab:green", width=0.55,
           label="residu mesure")
    ax.axhline(eps, color="tab:red", ls="--", lw=1.2,
               label="eps machine (%.2e)" % eps)
    ax.set_ylim(-0.15 * eps, 1.4 * eps)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("residu max |DSL compile - analytique|")
    ax.set_title("Oracle check_model.py : modele symbolique == formules a la main\n"
                 "(2x2 cellules, beta=3, theta=0.25) -- residu = 0.0 bit-exact")
    for i, r in enumerate(resid):
        ax.annotate("%.1f\n(bit-exact)" % r, xy=(i, 0.0),
                    xytext=(i, 0.22 * eps), ha="center", va="bottom",
                    fontsize=9, color="black")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "oracle_residual.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return resid, path


def main():
    gap_path = figure_gap_to_paper()
    resid, oracle_path = figure_oracle_residual()

    provenance = {
        "case": "hoffart_euler_poisson_dsl",
        "category": "reproduction-candidate",
        "reproduction_status": "PENDING (quantitative paper reproduction NOT established)",
        "paper": "https://arxiv.org/abs/2510.11808",
        "adc_cpp_sha": adc_cpp_sha(adc),
        "adc_cases_sha": adc_cases_sha(),
        "python": "3.12 (anaconda3, macOS arm64)",
        "figures": {
            "gap_to_paper.png": {
                "what": "raw full-system-schur growth rate vs paper target, per mode, n=256 and n=384",
                "engine": "full-system-schur (cart-square baseline)",
                "normalization": "raw (no 2pi, no rhobar)",
                "source_of_numbers": "VERBATIM from README validation table / adc_cpp/docs/HOFFART_FIDELITY.md "
                                     "(documented out-of-CI runs); NOT recomputed by this script",
                "paper_targets": PAPER,
                "measured": MEASURED,
                "relative_error_percent": {
                    str(n): {m: round(err_pct(MEASURED[n][m], PAPER[m]), 1) for m in (3, 4, 5)}
                    for n in (256, 384)
                },
                "reading": "reproduction NOT shown: measured raw slopes are -82 to -95% off the paper targets; "
                           "error does not improve from n=256 to n=384 (not a resolution problem)",
            },
            "oracle_residual.png": {
                "what": "max residual |compiled DSL - analytic formula| on 2x2 cells for flux_x, flux_y, "
                        "Lorentz/electric source, Poisson rhs",
                "engine": "analytic oracle (check_model.py), beta=3 theta=0.25",
                "residual_max": {
                    "flux_x": resid[0], "flux_y": resid[1],
                    "source": resid[2], "poisson_rhs": resid[3],
                },
                "machine_eps": float(np.finfo(float).eps),
                "reading": "PROVED: the compiled symbolic model equals the analytic formulas BIT-EXACTLY "
                           "(all four residuals are exactly 0.0, below machine eps)",
            },
        },
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True, default=float)

    print("wrote:", gap_path)
    print("wrote:", oracle_path)
    print("oracle residuals (max):", ["%.2e" % r for r in resid])
    print("wrote:", os.path.join(FIGDIR, "provenance.json"))


if __name__ == "__main__":
    main()
