#!/usr/bin/env python3
"""Figures de diagnostic du cas two_euler (validation d'invariant).

Re-joue EXACTEMENT la physique de run.py (meme grille, meme schema, memes CI, meme boucle
step_adaptive(0.4) x 20), mais instrumente chaque macro-pas pour produire les series temporelles
que run.py n'enregistre pas (il ne garde que l'etat final). On trace :

  (1) density_maps.png   : rho finale des 2 gaz (electrons / ions) cote a cote.
  (2) masses.png         : masse(t) des 2 especes (derive relative en insert) -> invariant 1.
  (3) positivity.png     : rho_min(t) et p_min(t) des 2 especes -> invariant 2 (positivite).
  (4) multirate.png      : macro_dt(t) et nombre de sous-cycles n_e(t) du bloc rapide -> multirate.

Aucune valeur n'est inventee : les nombres cites dans le README viennent de figures/provenance.json,
ecrit par ce script a partir des memes mesures.

  cd /private/tmp/adc_cases-deeptut/two_euler && \
  PYTHONPATH=<adc build>/python:/private/tmp/adc_cases-deeptut python3.12 make_figures.py
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # backend non interactif (pas de DISPLAY)
import matplotlib.pyplot as plt
import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402
from adc_cases.common.initial_conditions import (  # noqa: E402
    euler_pressure, euler_pressure_blob)

GAMMA = 1.4
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def pressure(U):
    return euler_pressure(U, gamma=GAMMA)


def max_wave_speed(U):
    """max sur la grille de |v| + c, c = sqrt(gamma p / rho) : la vitesse d'onde du bloc Euler
    compressible (cf. adc_cpp include/adc/physics/euler.hpp:121 max_wave_speed)."""
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    c = np.sqrt(GAMMA * pressure(U) / rho)
    speed = np.sqrt(u * u + v * v) + c
    return float(speed.max())


def build():
    n, L = 64, 1.0
    sim = adc.System(n=n, L=L, periodic=True)
    spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")
    sim.add_block("electrons", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
    sim.add_block("ions", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())
    sim.set_poisson()
    Ue0 = euler_pressure_blob(n, L, rho0=0.01, p0=1.0, dp=0.5, gamma=GAMMA)
    Ui0 = euler_pressure_blob(n, L, rho0=1.0, p0=1.0, dp=0.5, gamma=GAMMA)
    sim.set_state("electrons", Ue0.reshape(-1).tolist())
    sim.set_state("ions", Ui0.reshape(-1).tolist())
    return sim, n, L, Ue0, Ui0


def state(sim, name, n):
    return np.array(sim.get_state(name)).reshape(4, n, n)


def run_instrumented():
    sim, n, L, Ue0, Ui0 = build()
    h = L / n
    cfl = 0.4
    me0, mi0 = sim.mass("electrons"), sim.mass("ions")

    rec = {k: [] for k in ("t", "me", "mi", "rho_e", "rho_i", "p_e", "p_i",
                           "macro_dt", "n_e", "n_i", "we", "wi")}
    # pas 0 : etat initial (t = 0)
    Ue, Ui = state(sim, "electrons", n), state(sim, "ions", n)
    rec["t"].append(0.0)
    rec["me"].append(sim.mass("electrons")); rec["mi"].append(sim.mass("ions"))
    rec["rho_e"].append(float(Ue[0].min())); rec["rho_i"].append(float(Ui[0].min()))
    rec["p_e"].append(float(pressure(Ue).min())); rec["p_i"].append(float(pressure(Ui).min()))
    rec["macro_dt"].append(np.nan); rec["n_e"].append(np.nan); rec["n_i"].append(np.nan)
    rec["we"].append(max_wave_speed(Ue)); rec["wi"].append(max_wave_speed(Ui))

    for _ in range(20):
        # vitesses d'onde AVANT le macro-pas : reproduisent wmin et n_b du C++
        Ue, Ui = state(sim, "electrons", n), state(sim, "ions", n)
        we, wi = max_wave_speed(Ue), max_wave_speed(Ui)
        wmin = min(we, wi)
        # n_b = ceil(stride_b * w_b / wmin), stride = 1 (cf. system_stepper.hpp:338)
        n_e = max(1, int(np.ceil(we / wmin)))
        n_i = max(1, int(np.ceil(wi / wmin)))
        dt = sim.step_adaptive(cfl)  # macro_dt = cfl*h/wmin (renvoye)
        Ue, Ui = state(sim, "electrons", n), state(sim, "ions", n)
        rec["t"].append(float(sim.time()))
        rec["me"].append(sim.mass("electrons")); rec["mi"].append(sim.mass("ions"))
        rec["rho_e"].append(float(Ue[0].min())); rec["rho_i"].append(float(Ui[0].min()))
        rec["p_e"].append(float(pressure(Ue).min())); rec["p_i"].append(float(pressure(Ui).min()))
        rec["macro_dt"].append(float(dt)); rec["n_e"].append(n_e); rec["n_i"].append(n_i)
        rec["we"].append(we); rec["wi"].append(wi)

    for k in rec:
        rec[k] = np.array(rec[k], dtype=float)
    Ue, Ui = state(sim, "electrons", n), state(sim, "ions", n)
    return rec, Ue, Ui, Ue0, Ui0, me0, mi0, n, L, h, cfl


def fig_density_maps(Ue, Ui, L):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ext = [0, L, 0, L]
    for a, U, title in ((ax[0], Ue, "electrons (rho0 = 0.01)"),
                        (ax[1], Ui, "ions (rho0 = 1.0)")):
        im = a.imshow(U[0], origin="lower", extent=ext, cmap="viridis", aspect="equal")
        a.set_title("rho finale, %s" % title)
        a.set_xlabel("x"); a.set_ylabel("y")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("two_euler : densite finale des 2 gaz independants (t fixe, meme schema HLLC)")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "density_maps.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_masses(rec, me0, mi0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    t = rec["t"]
    ax[0].plot(t, rec["me"], "o-", color="tab:blue", label="electrons")
    ax[0].plot(t, rec["mi"], "s-", color="tab:red", label="ions")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("masse totale du bloc")
    ax[0].set_title("masse(t) : chaque bloc plat (conservation)")
    ax[0].legend()
    # derive relative |m(t) - m0| / m0 en echelle log : doit rester au bruit machine.
    # On plancher a eps_machine (~2.2e-16) : une derive EXACTEMENT nulle (m(t) == m0 au bit pres)
    # est portee au plancher, pas a -inf, pour rester lisible.
    eps = np.finfo(float).eps
    de = np.maximum(np.abs(rec["me"] - me0) / abs(me0), eps)
    di = np.maximum(np.abs(rec["mi"] - mi0) / abs(mi0), eps)
    ax[1].semilogy(t, de, "o-", color="tab:blue", label="electrons")
    ax[1].semilogy(t, di, "s-", color="tab:red", label="ions")
    ax[1].axhline(1e-9, color="k", ls="--", lw=1.2, label="tol assert = 1e-9")
    ax[1].axhline(eps, color="0.5", ls=":", lw=1, label="eps machine ~ 2.2e-16")
    ax[1].set_ylim(eps * 0.3, 1e-7)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("derive relative |m(t)-m0|/m0")
    ax[1].set_title("derive de masse vs tolerance (bruit machine)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("two_euler : masse conservee par espece (invariant 1)")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "masses.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    # max drift RAW (non plancher) pour la provenance : la vraie derive mesuree
    de_raw = float(np.max(np.abs(rec["me"] - me0) / abs(me0)))
    di_raw = float(np.max(np.abs(rec["mi"] - mi0) / abs(mi0)))
    return p, de_raw, di_raw


def fig_positivity(rec):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    t = rec["t"]
    # echelle log : les deux blocs differant d'un facteur 100 en densite, seul le log montre que
    # rho_min des electrons (~0.007, rho0=0.01) reste STRICTEMENT positif tout du long.
    ax[0].semilogy(t, rec["rho_e"], "o-", color="tab:blue", label="electrons (rho0=0.01)")
    ax[0].semilogy(t, rec["rho_i"], "s-", color="tab:red", label="ions (rho0=1.0)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("rho_min(t)  (echelle log)")
    ax[0].set_title("densite minimale (reste > 0, jamais 0)")
    ax[0].legend()
    ax[1].plot(t, rec["p_e"], "o-", color="tab:blue", label="electrons")
    ax[1].plot(t, rec["p_i"], "s-", color="tab:red", label="ions")
    ax[1].axhline(0.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("p_min(t)")
    ax[1].set_title("pression minimale (reste > 0)")
    ax[1].legend()
    fig.suptitle("two_euler : positivite rho_min / p_min vs t (invariant 2)")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "positivity.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_multirate(rec):
    t = rec["t"][1:]  # le pas 0 (t=0) n'a pas de macro_dt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(t, rec["n_e"][1:], "o-", color="tab:blue", label="n_electrons (sous-cycles)")
    ax[0].plot(t, rec["n_i"][1:], "s-", color="tab:red", label="n_ions (sous-cycles)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("nombre de sous-cycles par macro-pas")
    ax[0].set_title("multirate : n_b = ceil(w_b / wmin)")
    ax[0].set_ylim(0, max(rec["n_e"][1:].max(), 2) + 1)
    ax[0].legend()
    ax[1].plot(t, rec["macro_dt"][1:], "d-", color="tab:green", label="macro_dt = cfl h / wmin")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("macro_dt")
    ax[1].set_title("macro-pas cale sur le bloc le plus LENT (ions)")
    ax[1].legend()
    fig.suptitle("two_euler : le bloc rapide (electrons) est sous-cycle automatiquement")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "multirate.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    rec, Ue, Ui, Ue0, Ui0, me0, mi0, n, L, h, cfl = run_instrumented()
    p1 = fig_density_maps(Ue, Ui, L)
    p2, de_max, di_max = fig_masses(rec, me0, mi0)
    p3 = fig_positivity(rec)
    p4 = fig_multirate(rec)

    adc_mod = getattr(adc, "__file__", "unknown")
    adc_cpp_root = os.path.normpath(os.path.join(os.path.dirname(adc_mod), "..", "..", ".."))

    prov = {
        "script": "two_euler/make_figures.py",
        "command": "python make_figures.py",
        "produces": ["density_maps.png", "masses.png", "positivity.png", "multirate.png"],
        "adc_cpp_sha": git_sha(adc_cpp_root),
        "adc_cases_sha": git_sha(HERE),
        "backend": "natif serie (adc.System, 2 blocs models.euler, HLLC + recon primitive)",
        "resolution": "64x64",
        "periodic": True,
        "nsteps_macro": 20,
        "cfl": cfl,
        "gamma": GAMMA,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "matplotlib_backend": matplotlib.get_backend(),
        "adc_module": adc_mod,
        "measured": {
            "mass_drift_rel_electrons_max": de_max,
            "mass_drift_rel_ions_max": di_max,
            "rho_min_electrons_final": float(Ue[0].min()),
            "rho_min_ions_final": float(Ui[0].min()),
            "p_min_electrons_final": float(pressure(Ue).min()),
            "p_min_ions_final": float(pressure(Ui).min()),
            "rho_min_electrons_overrun": float(rec["rho_e"].min()),
            "p_min_electrons_overrun": float(rec["p_e"].min()),
            "wave_speed_electrons_init": float(rec["we"][0]),
            "wave_speed_ions_init": float(rec["wi"][0]),
            "wave_speed_ratio_init": float(rec["we"][0] / rec["wi"][0]),
            "n_electrons_subcycle_first": int(rec["n_e"][1]),
            "n_electrons_subcycle_last": int(rec["n_e"][-1]),
            "n_ions_subcycle_const": int(rec["n_i"][1]),
            "macro_dt_first": float(rec["macro_dt"][1]),
            "macro_dt_last": float(rec["macro_dt"][-1]),
            "t_final": float(rec["t"][-1]),
        },
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("figures ecrites :")
    for p in (p1, p2, p3, p4):
        print("  %s  (%d octets)" % (p, os.path.getsize(p)))
    print("provenance :", os.path.join(FIGDIR, "provenance.json"))
    print(json.dumps(prov["measured"], indent=2))


if __name__ == "__main__":
    main()
