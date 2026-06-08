#!/usr/bin/env python3
"""Figures du cas two_fluid_ap : la propriete asymptotic-preserving.

Re-joue la physique du cas (meme solveur C++ compile a la volee que run.py) et produit :

  1. ap_vs_explicit.png : balayage de la raideur s = dt*omega_pe. Le schema AP
     (stabilize=True) reste BORNE quand s -> infini (limite asymptotique preservee) ;
     le schema explicite (stabilize=False) reste fini sous s ~ 1 puis EXPLOSE (NaN).
  2. final_state.png : etat final des deux fluides au run raide de reference
     (n_e, n_i, charge nette n_i - n_e) a s = dt*omega_pe = 5.

Ecrit figures/*.png + figures/provenance.json (versionnes : ce sont les assets du cas).

ATTENTION : le diagnostic C++ tfap_max_dev fait std::fmax sur le champ et propage mal les
NaN (un champ explose peut rendre 0.0). On NE s'y fie PAS pour le schema explicite : on lit le
champ via density_e()/density_i() cote Python et on teste np.isfinite (detection robuste de
l'explosion). C'est l'observable qui distingue "borne" de "explose".
"""

import json
import os
import platform
import subprocess
import sys

import numpy as np

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run as R  # noqa: E402  (le pilote du cas : _build_lib, _bind, TwoFluidAP)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

N = 64           # grille n x n (identique a run.py)
DT = 5.0e-3      # pas de temps fixe (identique au run raide de run.py)
NSTEPS = 200     # horizon (identique au run raide)
RATIO_PI = 0.02  # omega_pi / omega_pe = 20/1000, comme run.py


def _field_diag(solver):
    """Lit n_e, n_i cote Python et calcule (finite, max|n_e-1|, max|n_i-n_e|).

    Robuste a l'explosion : si un champ contient un NaN/Inf, finite=False et les deviations
    valent +inf. NE PAS utiliser le tfap_max_dev C++ (fmax sur NaN = non fiable)."""
    ne = solver.density_e()
    ni = solver.density_i()
    finite = bool(np.isfinite(ne).all() and np.isfinite(ni).all())
    if not finite:
        return False, float("inf"), float("inf")
    return True, float(np.max(np.abs(ne - 1.0))), float(np.max(np.abs(ni - ne)))


def sweep_stiffness(lib):
    """Balaye s = dt*omega_pe, mesure la deviation AP vs explicite a dt et horizon fixes."""
    svals = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, 10.0, 25.0, 50.0]
    rows = []
    for s in svals:
        wpe = s / DT
        wpi = RATIO_PI * wpe
        row = {"s": s, "omega_pe": wpe}
        for tag, stab in (("ap", True), ("exp", False)):
            solver = R.TwoFluidAP(lib, n=N, omega_pe=wpe, omega_pi=wpi, stabilize=stab)
            solver.advance(DT, NSTEPS)
            finite, dev, chg = _field_diag(solver)
            row[tag + "_finite"] = finite
            # JSON strict n'admet pas Infinity : on serialise une explosion par null.
            row[tag + "_dev"] = dev if finite else None
            row[tag + "_chg"] = chg if finite else None
        rows.append(row)
    return rows


def final_state(lib):
    """Etat final du run raide de reference (s = dt*omega_pe = 5, AP)."""
    wpe, wpi = 1.0e3, 20.0
    solver = R.TwoFluidAP(lib, n=N, omega_pe=wpe, omega_pi=wpi, stabilize=True)
    solver.advance(DT, NSTEPS)
    ne = solver.density_e()
    ni = solver.density_i()
    return ne, ni


def plot_ap_vs_explicit(rows, path):
    s = np.array([r["s"] for r in rows])
    ap_dev = np.array([r["ap_dev"] for r in rows])
    exp_dev = np.array([r["exp_dev"] if r["exp_finite"] else np.nan for r in rows], dtype=float)
    exp_blown = np.array([not r["exp_finite"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(s, ap_dev, "o-", color="#1f77b4", lw=2, ms=6,
              label="AP (implicite raide) : borne")
    finite_exp = ~exp_blown
    ax.loglog(s[finite_exp], exp_dev[finite_exp], "s-", color="#d62728", lw=2, ms=6,
              label="explicite : fini sous s~1")
    # marqueurs d'explosion (NaN) du schema explicite, places en haut de l'axe
    ylo, yhi = ap_dev.min() * 0.3, max(np.nanmax(exp_dev), ap_dev.max()) * 30
    for sv in s[exp_blown]:
        ax.plot([sv], [yhi * 0.5], marker="x", color="#d62728", ms=11, mew=3)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.text(1.05, ylo * 2.0, "borne explicite\n$s=\\Delta t\\,\\omega_{pe}\\approx 1$",
            color="gray", fontsize=8.5, va="bottom")
    ax.text(s[exp_blown][0], yhi * 0.62, "explicite = NaN", color="#d62728",
            fontsize=9, ha="left")
    ax.set_xlabel(r"raideur  $s = \Delta t\,\omega_{pe}$  (limite asymptotique : $s\to\infty$)")
    ax.set_ylabel(r"$\max|n_e - 1|$  (ecart a la quasi-neutralite)")
    ax.set_title("Propriete asymptotic-preserving : AP borne, explicite explose")
    ax.set_ylim(ylo, yhi)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_final_state(ne, ni, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ext = [0, 2 * np.pi, 0, 2 * np.pi]
    for ax, field, title, cmap in (
        (axes[0], ne, r"$n_e$ (electrons)", "viridis"),
        (axes[1], ni, r"$n_i$ (ions)", "viridis"),
        (axes[2], ni - ne, r"charge nette $n_i - n_e$", "RdBu_r"),
    ):
        im = ax.imshow(field.T, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("y")
    fig.suptitle(r"Etat final, run raide de reference ($s=\Delta t\,\omega_{pe}=5$, AP, 200 pas)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    lib = R._bind(R._build_lib())

    rows = sweep_stiffness(lib)
    ne, ni = final_state(lib)

    fig1 = os.path.join(FIGDIR, "ap_vs_explicit.png")
    fig2 = os.path.join(FIGDIR, "final_state.png")
    plot_ap_vs_explicit(rows, fig1)
    plot_final_state(ne, ni, fig2)

    # provenance : champs reels (cf. ../diocotron/figures/provenance.json)
    import adc
    prov = {
        "script": "two_fluid_ap/make_figures.py",
        "command": "python make_figures.py",
        "produces": ["ap_vs_explicit.png", "final_state.png"],
        "adc_cpp_sha": _sha("/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"),
        "adc_cases_sha": _sha("/private/tmp/adc_cases-deeptut"),
        "backend": "scenario C++ sur mesure (two_fluid_ap.hpp), compile JIT via ctypes, "
                   "TwoFluidAP2D<GeometricMG>, CPU serie",
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "resolution": "64x64",
        "dt": DT,
        "nsteps": NSTEPS,
        "omega_pi_over_omega_pe": RATIO_PI,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "adc_module": adc.__file__,
        "stiffness_sweep": rows,
        "explicit_stability_bound_s": 1.0,
        "explicit_first_nan_at_s": 1.2,
        "ap_dev_plateau": rows[-1]["ap_dev"],
        "base_run_s5_ap_dev": next(r["ap_dev"] for r in rows if r["s"] == 5.0),
        "base_run_s5_ap_chg": next(r["ap_chg"] for r in rows if r["s"] == 5.0),
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("figures ecrites dans", FIGDIR)
    print("  AP dev plateau (s=50) =", rows[-1]["ap_dev"])
    print("  base run s=5 AP dev   =", prov["base_run_s5_ap_dev"],
          " chg =", prov["base_run_s5_ap_chg"])
    print("  explicite NaN des s >=", prov["explicit_first_nan_at_s"])


if __name__ == "__main__":
    main()
