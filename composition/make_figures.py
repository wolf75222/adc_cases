#!/usr/bin/env python3
"""Figures de diagnostic du cas composition (categorie tutoriel).

Re-joue exactement la physique de run.py (memes parametres : Partie A grille 48,
electrons electron_euler() VanLeer+HLLC+IMEX(10), ions ion_isothermal()
Minmod+Rusanov+Explicit ; Parties B/D grille 32, diocotron) et produit deux figures
dans figures/ :

  1. density_maps.png : Partie A. Cartes des champs composes du systeme heterogene :
                        densite electron (CI puis finale, schema Euler/HLLC/IMEX,
                        10 sous-pas), densite ion (finale, isotherme/Rusanov/explicite),
                        et potentiel couple |phi| (Sum_s q_s n_s, Poisson de systeme).
                        Montre que la composition bloc-par-bloc produit des champs reels
                        couples ; aucun claim physique (tutoriel).
  2. determinism.png  : Parties B et D. Determinisme bit a bit, sur deux chemins :
                        (B) deux compositions independantes du meme diocotron (briques
                        figees C++) ; (D) deux executions de l'integrateur SSPRK2 ecrit
                        en python (adc.integrate.ssprk2_step). Chaque heatmap |a - b| est
                        identiquement noire (ecart == 0, array_equal True). Une seule
                        tache non noire trahirait du non-determinisme dans le chemin
                        compose (B) ou dans le pas Python (D).

Ce cas est un tutoriel : il demontre une capacite d'API, il ne valide aucun resultat
physique publie. Les figures sont des diagnostics de l'API (champs composes, egalite
bit), pas une reproduction. Versionnees avec figures/provenance.json (memes champs que
diocotron/figures/provenance.json).

Lancer (depuis composition/) :
  PYTHONPATH=<adc_build>/python:<deeptut> python make_figures.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # backend non interactif : ecrit des PNG sans serveur X
import matplotlib.pyplot as plt
import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import models  # noqa: E402
from adc_cases.common.grid import meshgrid_xy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Memes constantes que run.py (un seul jeu de parametres, pas de divergence).
N_A, L = 48, 1.0  # Partie A : composition heterogene
N_BD = 32  # Parties B et D : diocotron
DT_A, NSTEPS_A = 0.001, 8  # run.py partie_A : sim.advance(0.001, 8)
DT_B, NSTEPS_B = 0.002, 12  # run.py partie_B : s.advance(0.002, 12)
DT_D, NSTEPS_D = 0.001, 20  # run.py partie_D : 20 x ssprk2_step(sim, 0.001)


# --------------------------------------------------------------------------- #
# Partie A : champs composes du systeme heterogene
# --------------------------------------------------------------------------- #
def partie_A_fields() -> dict:
    """Reconstruit run.py:partie_A et renvoie les cartes (CI, finales, potentiel)."""
    sim = adc.System(n=N_A, L=L, periodic=True)
    sim.add_block(
        "electrons",
        model=models.electron_euler(),
        spatial=adc.Spatial(vanleer=True, flux="hllc"),
        time=adc.IMEX(substeps=10),
    )
    sim.add_block(
        "ions",
        model=models.ion_isothermal(),
        spatial=adc.Spatial(minmod=True, flux="rusanov"),
        time=adc.Explicit(),
    )
    X, _ = meshgrid_xy(N_A, L)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")
    sim.set_density(
        "electrons", (1.0 + 0.02 * np.cos(2.0 * np.pi * X / L)).copy()
    )
    sim.set_density("ions", np.ones((N_A, N_A)))
    sim.solve_fields()

    rho_e_ci = sim.density("electrons").copy()
    phi_amp0 = float(np.max(np.abs(sim.potential())))
    m_e0, m_i0 = sim.mass("electrons"), sim.mass("ions")
    sim.advance(DT_A, NSTEPS_A)
    rho_e_fin = sim.density("electrons").copy()
    rho_i_fin = sim.density("ions").copy()
    phi_fin = sim.potential().copy()
    return {
        "rho_e_ci": rho_e_ci,
        "rho_e_fin": rho_e_fin,
        "rho_i_fin": rho_i_fin,
        "phi_fin": phi_fin,
        "phi_amp0": phi_amp0,
        "de": abs(sim.mass("electrons") - m_e0),
        "di": abs(sim.mass("ions") - m_i0),
        "evol_e": float(np.max(np.abs(rho_e_fin - rho_e_ci))),
    }


def fig_density_maps(fa: dict) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))

    def panel(ax, field, title, cmap="viridis"):
        im = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            extent=[0, L, 0, L],
            aspect="equal",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, L])
        ax.set_yticks([0, L])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    panel(
        axes[0], fa["rho_e_ci"], "electrons : densite CI\n1 + 0.02 cos(2pi x)"
    )
    panel(
        axes[1],
        fa["rho_e_fin"],
        "electrons : finale (t=%.3f)\nEuler / HLLC / IMEX(10)"
        % (DT_A * NSTEPS_A),
    )
    panel(
        axes[2],
        fa["rho_i_fin"],
        "ions : finale\nisotherme / Rusanov / Explicit",
    )
    panel(
        axes[3],
        np.abs(fa["phi_fin"]),
        "|phi| couple (Poisson systeme)\nSum_s q_s n_s",
        cmap="magma",
    )
    fig.suptitle(
        "Partie A : champs d'un systeme compose bloc par bloc (un schema par bloc), tutoriel, aucun claim physique",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "density_maps.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Partie B : deux compositions independantes du meme diocotron (briques C++)
# --------------------------------------------------------------------------- #
def partie_B_compose() -> tuple[np.ndarray, np.ndarray]:
    X, _ = meshgrid_xy(N_BD, L)
    rho0 = (1.0 + 0.1 * np.cos(2.0 * np.pi * X / L)).copy()
    n_i0 = float(rho0.mean())

    def construire_et_avancer():
        s = adc.System(n=N_BD, L=L, periodic=True)
        s.add_block(
            "e",
            model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
            spatial=adc.Spatial(minmod=True, flux="rusanov"),
            time=adc.Explicit(substeps=1),
        )
        s.set_poisson()
        s.set_density("e", rho0.copy())
        s.advance(DT_B, NSTEPS_B)
        return s.density("e")

    da = construire_et_avancer()
    db = construire_et_avancer()
    return da, db


# --------------------------------------------------------------------------- #
# Partie D : deux executions de l'integrateur SSPRK2 ecrit en python
# --------------------------------------------------------------------------- #
def partie_D_pystep() -> tuple[np.ndarray, np.ndarray, float]:
    X, Y = meshgrid_xy(N_BD, L)
    rho0 = (
        1.0 + 0.1 * np.cos(2.0 * np.pi * X / L) * np.sin(2.0 * np.pi * Y / L)
    ).copy()
    n_i0 = float(rho0.mean())

    def run_python_loop():
        sim = adc.System(n=N_BD, L=L, periodic=True)
        sim.add_block(
            "e",
            model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
            spatial=adc.Spatial(minmod=True),
            time=adc.Explicit(),
        )
        sim.set_poisson()
        sim.set_density("e", rho0.copy())
        m0 = sim.mass("e")
        for _ in range(NSTEPS_D):
            adc.integrate.ssprk2_step(sim, DT_D)  # SSPRK2 ecrit en Python
        return sim.density("e"), abs(sim.mass("e") - m0)

    da, dm = run_python_loop()
    db, _ = run_python_loop()
    return da, db, dm


def fig_determinism(
    da_B: np.ndarray,
    db_B: np.ndarray,
    da_D: np.ndarray,
    db_D: np.ndarray,
) -> tuple:
    diff_B = np.abs(da_B - db_B)
    diff_D = np.abs(da_D - db_D)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (B) heatmap de l'ecart : doit etre identiquement noir.
    imB = axes[0].imshow(
        diff_B,
        origin="lower",
        cmap="inferno",
        extent=[0, L, 0, L],
        vmin=0.0,
        vmax=1e-15,
    )
    axes[0].set_title(
        "(B) |rho_compo1 - rho_compo2|\ndeux compositions C++ (diocotron)",
        fontsize=10,
    )
    axes[0].set_xticks([0, L])
    axes[0].set_yticks([0, L])
    fig.colorbar(imB, ax=axes[0], fraction=0.046, pad=0.04)

    # (D) heatmap de l'ecart : doit etre identiquement noir.
    imD = axes[1].imshow(
        diff_D,
        origin="lower",
        cmap="inferno",
        extent=[0, L, 0, L],
        vmin=0.0,
        vmax=1e-15,
    )
    axes[1].set_title(
        "(D) |rho_run1 - rho_run2|\ndeux pas SSPRK2 ecrits en python",
        fontsize=10,
    )
    axes[1].set_xticks([0, L])
    axes[1].set_yticks([0, L])
    fig.colorbar(imD, ax=axes[1], fraction=0.046, pad=0.04)

    # histogramme du residu : une seule barre a 0.
    allres = np.concatenate([diff_B.ravel(), diff_D.ravel()])
    axes[2].hist(
        allres, bins=np.linspace(-5e-16, 5e-16, 11), color="#444", edgecolor="k"
    )
    axes[2].axvline(0.0, color="crimson", lw=1.2, ls="--")
    axes[2].set_title(
        "residu |a - b| (B et D)\ntoute la masse a exactement 0", fontsize=10
    )
    axes[2].set_xlabel("ecart par cellule")
    axes[2].set_ylabel("nb de cellules")

    fig.suptitle(
        "Determinisme bit a bit : composition figee (B) et pas SSPRK2 ecrit en Python (D), ecart == 0",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGDIR, "determinism.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return (
        out,
        float(diff_B.max()),
        float(diff_D.max()),
        bool(np.array_equal(da_B, db_B)),
        bool(np.array_equal(da_D, db_D)),
    )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_sha(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    fa = partie_A_fields()
    out_maps = fig_density_maps(fa)

    da_B, db_B = partie_B_compose()
    da_D, db_D, dm_D = partie_D_pystep()
    out_det, max_B, max_D, eq_B, eq_D = fig_determinism(da_B, db_B, da_D, db_D)

    adc_cpp_root = os.path.dirname(
        os.path.dirname(os.path.dirname(adc.__file__))
    )
    prov = {
        "script": "composition/make_figures.py",
        "command": "python make_figures.py",
        "produces": ["density_maps.png", "determinism.png"],
        "adc_cpp_sha": git_sha(adc_cpp_root),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.System, blocs = compositions de briques models.*)",
        "resolution": {
            "partie_A": f"{N_A}x{N_A}",
            "parties_BD": f"{N_BD}x{N_BD}",
        },
        "periodic": True,
        "schemes": {
            "electrons": "electron_euler() | Spatial(vanleer, hllc) | IMEX(substeps=10)",
            "ions": "ion_isothermal() | Spatial(minmod, rusanov) | Explicit()",
            "diocotron_B": "diocotron() | Spatial(minmod, rusanov) | Explicit(substeps=1)",
            "diocotron_D": "diocotron() | Spatial(minmod) | integrateur SSPRK2 ecrit en python (adc.integrate.ssprk2_step)",
        },
        "dt_nsteps": {
            "A": [DT_A, NSTEPS_A],
            "B": [DT_B, NSTEPS_B],
            "D": [DT_D, NSTEPS_D],
        },
        "python": sys.version.split()[0],
        "adc_module": adc.__file__,
        "measured": {
            "phi_amp_initial_A": fa["phi_amp0"],
            "mass_drift_electrons_A": fa["de"],
            "mass_drift_ions_A": fa["di"],
            "electron_evolution_A": fa["evol_e"],
            "rho_e_min_fin_A": float(fa["rho_e_fin"].min()),
            "rho_e_max_fin_A": float(fa["rho_e_fin"].max()),
            "rho_i_min_fin_A": float(fa["rho_i_fin"].min()),
            "rho_i_max_fin_A": float(fa["rho_i_fin"].max()),
            "compose_bit_diff_B": max_B,
            "compose_array_equal_B": eq_B,
            "pystep_bit_diff_D": max_D,
            "pystep_array_equal_D": eq_D,
            "mass_drift_pystep_D": dm_D,
        },
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("Figures ecrites dans", FIGDIR)
    print("  ", out_maps)
    print("  ", out_det)
    print(
        f"  Partie A : |phi|_max = {fa['phi_amp0']:.6e}, evolution electrons = {fa['evol_e']:.3e}"
    )
    print(f"  Partie B : compose bit diff = {max_B:.3e}, array_equal = {eq_B}")
    print(
        f"  Partie D : pystep bit diff = {max_D:.3e}, array_equal = {eq_D}, mass drift = {dm_D:.3e}"
    )


if __name__ == "__main__":
    main()
