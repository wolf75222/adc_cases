#!/usr/bin/env python3
"""Figures de diagnostic du cas multispecies (categorie validation).

Re-joue EXACTEMENT la physique de run.py (memes briques models.electron_euler /
models.ion_isothermal, meme Poisson de systeme charge_density, meme CI cosinus,
memes 20 pas de dt = 0.001) et enregistre l'historique pas a pas de la masse de
chaque espece. Produit trois figures sous figures/ :

  1. masse.png     : mass_e(t) et mass_i(t) en ECHELLE ABSOLUE, fenetre etroite
                     autour de la masse initiale, pour rendre visible la derive
                     ~1e-11 (du bruit d'arrondi flottant, pas de la physique).
  2. densite.png   : cartes finales n_e(x,y) et n_i(x,y).
  3. potentiel.png : carte de |phi(x,y)|, solution du Poisson couple, et coupe
                     phi(x) comparee a la solution analytique mono-mode.

Lancer (depuis multispecies/) :
  PYTHONPATH=<adc_build>/python:<repo> python3 make_figures.py
"""

import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")  # backend sans affichage : ecrit des PNG, jamais de fenetre
import matplotlib.pyplot as plt
import numpy as np

import adc
import adc_cases.models as models

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --- Parametres : STRICTEMENT ceux de run.py --------------------------------
N = 48
L = 1.0
DT = 0.001
NSTEPS = 20
DELTA = 0.02  # amplitude du cosinus electronique (run.py:80)


def build_system():
    """Compose le systeme multispecies de run.py (deux blocs heterogenes + Poisson couple)."""
    sim = adc.System(n=N, L=L, periodic=True)
    sim.add_block(
        "electrons",
        model=models.electron_euler(charge=-1.0, gamma=5.0 / 3.0),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    sim.add_block(
        "ions",
        model=models.ion_isothermal(charge=+1.0, cs2=1.0),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    n = sim.nx()
    x = (np.arange(n) + 0.5) / n * L
    ne = 1.0 + DELTA * np.cos(2.0 * np.pi * x / L)
    sim.set_density("electrons", np.broadcast_to(ne, (n, n)).copy())
    sim.set_density("ions", np.ones((n, n)))
    return sim


def run_with_history():
    """Avance 20 pas en relevant la masse par espece a CHAQUE pas (historique du diagnostic)."""
    sim = build_system()
    me0 = sim.mass("electrons")
    mi0 = sim.mass("ions")
    t = [0.0]
    me = [me0]
    mi = [mi0]
    for _ in range(NSTEPS):
        sim.advance(DT, 1)
        t.append(sim.time())
        me.append(sim.mass("electrons"))
        mi.append(sim.mass("ions"))
    return (sim, me0, mi0,
            np.array(t), np.array(me), np.array(mi))


def fig_mass(t, me, mi, me0, mi0):
    """Figure 1 : conservation de masse par espece, ECHELLE ABSOLUE (montre la derive d'arrondi)."""
    fig, (axe, axi) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Echelle absolue : on trace la DERIVE M(t) - M(0) (l'offset M0 est dans le titre/legende),
    # graduee en multiples de l'epsilon machine relatif eps*M0 pour montrer que la derive y vit.
    eps_e = np.finfo(float).eps * me0  # 1 ulp sur M0 ~ 5.1e-13 (= grain du flottant a cette echelle)
    span_e = max(np.max(np.abs(me - me0)), eps_e)
    axe.plot(t, me - me0, "o-", color="C0", ms=4, lw=1.2)
    axe.axhline(0.0, color="0.6", ls="--", lw=1.0)
    axe.set_ylim(-4 * span_e, 4 * span_e)
    axe.set_title(f"electrons : Euler compressible (4 var), $M_e(0)$ = {me0:.6g}")
    axe.set_xlabel("t")
    axe.set_ylabel("$M_e(t) - M_e(0)$")
    axe.text(0.02, 0.06,
             f"$\\max_t |M_e(t)-M_e(0)|$ = {np.max(np.abs(me-me0)):.2e}\n"
             f"1 ulp($M_e0$) $\\approx$ {eps_e:.1e}   (tol assert = 1e-9, bien au-dessus du cadre)",
             transform=axe.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    eps_i = np.finfo(float).eps * mi0
    span_i = max(np.max(np.abs(mi - mi0)), eps_i)
    axi.plot(t, mi - mi0, "s-", color="C3", ms=4, lw=1.2)
    axi.axhline(0.0, color="0.6", ls="--", lw=1.0)
    axi.set_ylim(-4 * span_i, 4 * span_i)
    axi.set_title(f"ions : Euler isotherme (3 var), $M_i(0)$ = {mi0:.6g}")
    axi.set_xlabel("t")
    axi.set_ylabel("$M_i(t) - M_i(0)$")
    axi.text(0.02, 0.06,
             f"$\\max_t |M_i(t)-M_i(0)|$ = {np.max(np.abs(mi-mi0)):.2e}\n"
             f"1 ulp($M_i0$) $\\approx$ {eps_i:.1e}   (tol assert = 1e-9, bien au-dessus du cadre)",
             transform=axi.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.suptitle("Conservation de masse PAR ESPECE (tol assert = 1e-9, en absolu)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "masse.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_density(sim):
    """Figure 2 : cartes finales de densite n_e et n_i (echelles propres a chaque espece)."""
    de = np.asarray(sim.density("electrons"))
    di = np.asarray(sim.density("ions"))
    fig, (axe, axi) = plt.subplots(1, 2, figsize=(11, 4.6))
    ext = [0, L, 0, L]

    ime = axe.imshow(de, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    axe.set_title(f"$n_e$ (electrons)  [{de.min():.4f}, {de.max():.4f}]")
    axe.set_xlabel("x")
    axe.set_ylabel("y")
    fig.colorbar(ime, ax=axe, fraction=0.046, pad=0.04)

    imi = axi.imshow(di, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    axi.set_title(f"$n_i$ (ions)  [{di.min():.6f}, {di.max():.6f}]")
    axi.set_xlabel("x")
    axi.set_ylabel("y")
    fig.colorbar(imi, ax=axi, fraction=0.046, pad=0.04)

    fig.suptitle("Densites finales (t = 0.02) : modulee en x, invariante en y")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "densite.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, de, di


def fig_potential(sim):
    """Figure 3 : carte de |phi| (Poisson couple) + coupe phi(x) vs solution analytique mono-mode."""
    phi = np.asarray(sim.potential())
    n = phi.shape[0]
    x = (np.arange(n) + 0.5) / n * L

    # Solution analytique du Poisson lap(phi) = f sur le tore, mode unique k = 2pi/L :
    # f = q_e n_e + q_i n_i = (-1)(1 + delta cos(kx)) + (+1)(1) = -delta cos(kx).
    # lap(phi) = f  =>  -k^2 phi_hat = -delta  =>  phi(x) = (delta/k^2) cos(kx).
    k = 2.0 * np.pi / L
    phi_analytic = (DELTA / k**2) * np.cos(k * x)
    phi_cut = phi[n // 2, :]  # une ligne y = const : phi ne depend que de x

    fig, (axmap, axcut) = plt.subplots(1, 2, figsize=(11, 4.6))

    im = axmap.imshow(np.abs(phi), origin="lower", extent=[0, L, 0, L],
                      cmap="magma", aspect="equal")
    axmap.set_title(f"$|\\phi(x,y)|$  (max = {np.abs(phi).max():.3e})")
    axmap.set_xlabel("x")
    axmap.set_ylabel("y")
    fig.colorbar(im, ax=axmap, fraction=0.046, pad=0.04)

    axcut.plot(x, phi_cut, "o", color="C0", ms=4, label="$\\phi$ adc (coupe y = L/2)")
    axcut.plot(x, phi_analytic, "-", color="C1", lw=1.6,
               label="$(\\delta/k^2)\\cos(kx)$ analytique")
    axcut.axhline(0, color="0.7", lw=0.8)
    axcut.set_title("coupe $\\phi(x)$ vs solution mono-mode du Poisson")
    axcut.set_xlabel("x")
    axcut.set_ylabel("$\\phi$")
    axcut.legend(loc="upper right", fontsize=9)
    resid = float(np.max(np.abs(phi_cut - phi_analytic)))
    axcut.text(0.02, 0.04, f"$\\max|\\phi_{{adc}}-\\phi_{{ana}}|$ = {resid:.2e}",
               transform=axcut.transAxes, fontsize=9,
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.suptitle("Potentiel couple : un seul Poisson agrege les charges des deux especes")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "potentiel.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, phi, phi_analytic, phi_cut, resid


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    sim, me0, mi0, t, me, mi = run_with_history()

    f_mass = fig_mass(t, me, mi, me0, mi0)
    f_dens, de, di = fig_density(sim)
    f_pot, phi, phi_ana, phi_cut, phi_resid = fig_potential(sim)

    # Separation de charge initiale et finale (max|f|), comme run.py.
    charge1 = (-1.0) * de + (+1.0) * di
    qmax1 = float(np.max(np.abs(charge1)))

    drift_e = float(np.max(np.abs(me - me0)))
    drift_i = float(np.max(np.abs(mi - mi0)))

    prov = {
        "script": "multispecies/make_figures.py",
        "command": "python make_figures.py",
        "produces": ["masse.png", "densite.png", "potentiel.png"],
        "adc_cpp_sha": git_sha("/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.System, deux blocs heterogenes + Poisson de systeme charge_density)",
        "resolution": "48x48",
        "periodic": True,
        "species": {
            "electrons": "Euler compressible (4 var), charge -1, gamma = 5/3",
            "ions": "Euler isotherme (3 var), charge +1, cs2 = 1.0",
        },
        "dt": DT,
        "nsteps": NSTEPS,
        "delta_cos": DELTA,
        "python": "3.12.2",
        "adc_module": adc.__file__,
        "measured": {
            "mass_e0": me0,
            "mass_i0": mi0,
            "drift_e_max_abs": drift_e,
            "drift_i_max_abs": drift_i,
            "drift_e_final_abs": float(abs(me[-1] - me0)),
            "drift_i_final_abs": float(abs(mi[-1] - mi0)),
            "tol_mass_assert": 1e-9,
            "ne_min": float(de.min()),
            "ne_max": float(de.max()),
            "ni_min": float(di.min()),
            "ni_max": float(di.max()),
            "phi_absmax": float(np.abs(phi).max()),
            "phi_analytic_absmax": float(np.abs(phi_ana).max()),
            "phi_vs_analytic_resid_max": phi_resid,
            "qmax_final": qmax1,
        },
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("figures ecrites :")
    for f in (f_mass, f_dens, f_pot):
        print("  ", f, os.path.getsize(f), "octets")
    print("mesures cles :")
    print(f"  M_e0 = {me0:.16e}  M_i0 = {mi0:.16e}")
    print(f"  max|dM_e| = {drift_e:.3e}  max|dM_i| = {drift_i:.3e}")
    print(f"  n_e in [{de.min():.6f}, {de.max():.6f}]  n_i in [{di.min():.6f}, {di.max():.6f}]")
    print(f"  |phi|max = {np.abs(phi).max():.6e}  analytic = {np.abs(phi_ana).max():.6e}  "
          f"resid = {phi_resid:.3e}")
    print(f"  max|f| final = {qmax1:.6e}")


if __name__ == "__main__":
    main()
