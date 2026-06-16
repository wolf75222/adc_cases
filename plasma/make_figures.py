#!/usr/bin/env python3
"""Figures de diagnostic du cas `plasma` (validation : Poisson + ionisation + collision).

Re-joue exactement la physique de `run.py` (memes CI, meme recette, meme nombre de pas et meme
CFL) mais instrumente la boucle pas-a-pas pour enregistrer l'historique des diagnostics, puis
trace trois figures sous `figures/` :
  1. densities.png   : densites moyennes e / i / n vs t (la modulation e- a 5 % se moyenne).
  2. ionization.png  : bilan d'ionisation (n_i monte, n_g descend, n_i + n_g plat) + erreur de
                       conservation en echelle log + impulsion totale (collision).
  3. density_map.png  : carte 2D de densite des trois especes a l'etat final.

Ecrit aussi figures/provenance.json (memes champs que diocotron/figures/provenance.json).

Aucune figure n'est versionnee dans le depot : ce cas est `validation`, pas `reproduction`. Les
PNG produits ici sont des diagnostics du tutoriel (le guide autorise `<cas>/figures/` versionne
seulement pour une reproduction). On les ecrit neanmoins dans figures/ pour le tutoriel, avec
provenance, en assumant qu'ils ne tournent pas en CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import recipes  # noqa: E402
from adc_cases.common.checks import relative_drift  # noqa: E402

PI = np.pi
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
N, L = 48, 1.0
NSTEPS, CFL = 20, 0.3
K_ION, K_COL = 0.3, 0.5


def build_system() -> adc.System:
    """Memes CI et meme recette que run.py (faible separation de charge electronique)."""
    x = (np.arange(N) + 0.5) / N
    ne = 1.0 + 0.05 * np.cos(2 * PI * x)[None, :] * np.ones((N, N))
    sim = adc.System(n=N, L=L, periodic=True)
    recipes.plasma(
        sim,
        ne=ne,
        ni=np.ones((N, N)),
        ng=np.ones((N, N)),
        ionization_rate=K_ION,
        collision_rate=K_COL,
    )
    return sim


def total_momentum(sim: adc.System, block: str) -> tuple[float, float]:
    """Impulsion totale (somme cellule) d'un bloc fluide : (sum rho u, sum rho v)."""
    st = np.array(
        sim._s.get_state(block)
    )  # (ncomp, n, n), comp 1 = rho u, comp 2 = rho v
    return float(st[1].sum()), float(st[2].sum())


def run_with_history() -> tuple[dict, dict]:
    """Avance NSTEPS pas, enregistre a chaque pas t, masses e/i/n, |phi|max, impulsion totale."""
    sim = build_system()
    sim.solve_fields()
    species = ("electrons", "ions", "neutrals")
    hist = {
        "t": [],
        "mass": {s: [] for s in species},
        "phimax": [],
        "px_tot": [],
        "py_tot": [],
    }

    def snapshot():
        hist["t"].append(float(sim.time()))
        for s in species:
            hist["mass"][s].append(float(sim.mass(s)))
        hist["phimax"].append(float(np.abs(np.array(sim.potential())).max()))
        # impulsion totale du systeme ferme ion+neutre (la collision ne fait que l'echanger ;
        # le champ E agit aussi sur les ions, donc px_tot global n'est pas conserve : on suit
        # le couple ion+neutre pour isoler la friction le mieux possible).
        pix, piy = total_momentum(sim, "ions")
        pnx, pny = total_momentum(sim, "neutrals")
        hist["px_tot"].append(pix + pnx)
        hist["py_tot"].append(piy + pny)

    snapshot()
    for _ in range(NSTEPS):
        sim.step_cfl(CFL)
        snapshot()
    # cartes finales de densite
    dens = {s: np.array(sim.density(s)) for s in species}
    return hist, dens


def fig_densities(hist: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(hist["t"])
    n2 = N * N  # mass() = somme des densites ; densite moyenne = mass / n^2
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {
        "electrons": "tab:blue",
        "ions": "tab:red",
        "neutrals": "tab:green",
    }
    labels = {
        "electrons": r"$\bar n_e$ (electrons)",
        "ions": r"$\bar n_i$ (ions)",
        "neutrals": r"$\bar n_g$ (neutres)",
    }
    for s in ("electrons", "ions", "neutrals"):
        ax.plot(
            t,
            np.array(hist["mass"][s]) / n2,
            "-o",
            ms=3,
            color=colors[s],
            label=labels[s],
        )
    ax.set_xlabel("t")
    ax.set_ylabel(r"densite moyenne $\bar n = \sum_{cell} n / N^2$")
    ax.set_title("Densites moyennes des trois especes (ionisation + collision)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "densities.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_ionization(hist: dict) -> tuple[str, float, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(hist["t"])
    mi = np.array(hist["mass"]["ions"])
    mg = np.array(hist["mass"]["neutrals"])
    s = mi + mg
    drel = np.abs(s - s[0]) / max(abs(s[0]), 1e-30)
    px = np.array(hist["px_tot"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    ax.plot(t, mi, "-o", ms=3, color="tab:red", label=r"$M_i=\sum n_i$ (ions)")
    ax.plot(
        t, mg, "-o", ms=3, color="tab:green", label=r"$M_g=\sum n_g$ (neutres)"
    )
    ax.plot(t, s, "-s", ms=3, color="black", label=r"$M_i+M_g$ (conserve)")
    ax.set_xlabel("t")
    ax.set_ylabel("masse (somme des densites)")
    ax.set_title("Transfert n_g -> n_i (ionisation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    eps = np.finfo(float).eps
    ax.semilogy(t, np.maximum(drel, eps), "-o", ms=3, color="tab:purple")
    ax.axhline(1e-7, ls="--", color="grey", label="tolerance assert (1e-7)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|M_i+M_g - (M_i+M_g)_0| / (M_i+M_g)_0$")
    ax.set_title("Derive de masse n_i + n_g (precision machine)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(
        t,
        px - px[0],
        "-o",
        ms=3,
        color="tab:orange",
        label=r"$\Delta(P_x^{ion}+P_x^{neutre})$",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("variation d'impulsion totale (x)")
    ax.set_title("Impulsion ion+neutre : friction + champ E sur ions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "ionization.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out, float(drel[-1]), float(px[-1] - px[0])


def fig_density_map(dens: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    titles = {
        "electrons": "electrons n_e (etat final)",
        "ions": "ions n_i (etat final)",
        "neutrals": "neutres n_g (etat final)",
    }
    for ax, s in zip(axes, ("electrons", "ions", "neutrals")):
        im = ax.imshow(
            dens[s], origin="lower", extent=[0, L, 0, L], cmap="viridis"
        )
        ax.set_title(titles[s])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "density_map.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def git_sha(path: str) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    hist, dens = run_with_history()

    f1 = fig_densities(hist)
    f2, drel_final, dpx_final = fig_ionization(hist)
    f3 = fig_density_map(dens)

    n2 = N * N
    mi0, mi1 = hist["mass"]["ions"][0], hist["mass"]["ions"][-1]
    mg0, mg1 = hist["mass"]["neutrals"][0], hist["mass"]["neutrals"][-1]
    me0, me1 = hist["mass"]["electrons"][0], hist["mass"]["electrons"][-1]
    prov = {
        "script": "plasma/make_figures.py",
        "command": (
            "PYTHONPATH=.../adc_cpp/build-master/python:.../adc_cases-deeptut "
            "python3.12 make_figures.py"
        ),
        "produces": ["densities.png", "ionization.png", "density_map.png"],
        "adc_cpp_sha": git_sha(
            "/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"
        ),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.System, recette models/recipes.plasma)",
        "resolution": "48x48",
        "species": ["electrons", "ions", "neutrals"],
        "nsteps": NSTEPS,
        "cfl": CFL,
        "ionization_rate": K_ION,
        "collision_rate": K_COL,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "adc_module": adc.__file__,
        "t_final": hist["t"][-1],
        "phimax_initial": hist["phimax"][0],
        "mass_electrons": [me0, me1],
        "mass_ions": [mi0, mi1],
        "mass_neutrals": [mg0, mg1],
        "n_ion_mean": [mi0 / n2, mi1 / n2],
        "n_neutral_mean": [mg0 / n2, mg1 / n2],
        "drel_ni_plus_ng_final": drel_final,
        "delta_px_ion_plus_neutral_final": dpx_final,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    for f in (f1, f2, f3):
        sz = os.path.getsize(f)
        print("  wrote %-40s %8d bytes" % (os.path.relpath(f, HERE), sz))
    print("== diagnostics mesures ==")
    print("  t_final              = %.5f" % hist["t"][-1])
    print("  |phi|_max (t=0)       = %.4e" % hist["phimax"][0])
    print("  n_i mean : %.5f -> %.5f" % (mi0 / n2, mi1 / n2))
    print("  n_g mean : %.5f -> %.5f" % (mg0 / n2, mg1 / n2))
    print("  n_e mean : %.5f -> %.5f" % (me0 / n2, me1 / n2))
    print("  drel(n_i+n_g) final  = %.3e" % drel_final)
    print("  delta Px(ion+neutre) = %.3e" % dpx_final)


if __name__ == "__main__":
    main()
