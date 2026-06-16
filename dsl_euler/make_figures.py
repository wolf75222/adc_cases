"""Figures du cas dsl_euler (prototype experimental, mini-DSL interprete numpy).

Re-joue exactement la physique de run.py (meme modele declare en formules, meme CI, meme schema
Rusanov ordre 1 / Euler avant, 120 pas) en instrumentant a chaque pas. Ecrit deux figures + un
provenance.json sous figures/. Backend matplotlib Agg (sans affichage).

Le cas etant `experimental` (cf. cases_manifest.toml), ces figures ne sont pas un asset de
reproduction versionne : ce sont des diagnostics du prototype (un etat fini et coherent + la
relaxation de la bulle). Aucune cible publiee, aucune tolerance sur une valeur physique.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adc import dsl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases.common.grid import meshgrid_xy  # noqa: E402
from adc_cases.common.initial_conditions import euler_pressure  # noqa: E402

GAMMA = 1.4
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def make_euler() -> dsl.HyperbolicModel:
    """Modele Euler 2D declare en formules (copie conforme de run.py:make_euler)."""
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(
        x=[rhou, rhou * u + p, rhou * v, rho * H * u],
        y=[rhov, rhov * u, rhov * v + p, rho * H * v],
    )
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.check()
    return e


def main() -> None:
    euler = make_euler()
    n, L = 64, 1.0
    h = L / n
    gx, gy = meshgrid_xy(n, L)
    r2 = (gx - 0.5) ** 2 + (gy - 0.5) ** 2
    p0 = 1.0 + 0.4 * np.exp(-r2 / 0.01)
    U = np.zeros((4, n, n))
    U[0] = 1.0
    U[3] = p0 / (GAMMA - 1.0)

    mass0 = float(U[0].sum())
    p_init = euler_pressure(U, gamma=GAMMA).copy()
    pc0 = float(p_init[n // 2, n // 2])

    pf = euler.to_python_flux()
    steps = 120
    t = []
    tt = 0.0
    p_center = []  # pression au centre (sommet de la bulle)
    amp = []  # max|p - p_init| (le diagnostic 'moved' de run.py)
    for _ in range(steps):
        dt = pf.cfl_dt(U, h, 0.4)
        U = U + dt * pf.residual(U, h)
        tt += dt
        pr = euler_pressure(U, gamma=GAMMA)
        t.append(tt)
        p_center.append(float(pr[n // 2, n // 2]))
        amp.append(float(np.max(np.abs(pr - p_init))))
    t = np.array(t)
    p_center = np.array(p_center)
    amp = np.array(amp)
    pr_final = euler_pressure(U, gamma=GAMMA)
    rho_final = U[0]
    drel = abs(float(U[0].sum()) - mass0) / abs(mass0)
    p_mean = float(pr_final.mean())

    # ---- Figure 1 : carte finale (densite + pression) ----
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.3))
    ext = [0, L, 0, L]
    im0 = ax[0].imshow(rho_final.T, origin="lower", extent=ext, cmap="viridis")
    ax[0].set_title(r"densite finale $\rho$ (t=%.3f, 120 pas)" % tt)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(pr_final.T, origin="lower", extent=ext, cmap="magma")
    ax[1].set_title(r"pression finale $p$ (anneau acoustique)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        "dsl_euler (prototype DSL interprete) : etat fini, anneau radial, "
        r"$\rho>0$, $p>0$",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = os.path.join(FIGDIR, "final_state.png")
    fig.savefig(f1, dpi=120)
    plt.close(fig)

    # ---- Figure 2 : relaxation de la bulle (decroissance de la perturbation) ----
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax[0].plot(t, p_center, color="C3", lw=1.8)
    ax[0].axhline(
        p_mean,
        color="0.5",
        ls="--",
        lw=1.0,
        label=r"$\bar p=%.3f$ (moyenne finale)" % p_mean,
    )
    ax[0].axhline(
        pc0, color="C3", ls=":", lw=1.0, alpha=0.6, label=r"$p_c(0)=%.3f$" % pc0
    )
    ax[0].set_title("relaxation du sommet de la bulle")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$p$ au centre")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, amp, color="C0", lw=1.8)
    kmax = int(np.argmax(amp))
    ax[1].plot(t[kmax], amp[kmax], "o", color="C0", ms=5)
    ax[1].annotate(
        r"pic %.3f a t=%.3f" % (amp[kmax], t[kmax]),
        (t[kmax], amp[kmax]),
        textcoords="offset points",
        xytext=(8, -2),
        fontsize=8,
    )
    ax[1].set_title(r"amplitude $\max|p-p_0|$ (diagnostic 'moved')")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$\max|p-p_0|$")
    fig.suptitle(
        "dsl_euler : la bulle se detend (front sortant), puis se dilue",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f2 = os.path.join(FIGDIR, "bubble_decay.png")
    fig.savefig(f2, dpi=120)
    plt.close(fig)

    prov = {
        "script": "dsl_euler/make_figures.py",
        "command": "python make_figures.py",
        "category": "experimental",
        "note": "prototype DSL interprete numpy (PythonFlux), hors CI ; figures de diagnostic, "
        "pas un asset de reproduction versionne",
        "produces": ["final_state.png", "bubble_decay.png"],
        "adc_cpp_sha": "018732997c02a17ade387fa99a74267f37e252c1",
        "adc_cases_sha": "1affec1d209e26d5ee422cac255d2cc3f149247a",
        "backend": "interprete CPU numpy (adc.dsl.HyperbolicModel.flux + adc.PythonFlux Rusanov ordre 1)",
        "resolution": "64x64",
        "scheme": "Rusanov ordre 1 + Euler avant, CFL 0.4, periodique",
        "nsteps": steps,
        "python": "3.12.2",
        "numpy": "1.26.4",
        "platform": "Darwin arm64 (Apple Silicon)",
        "adc_module": "/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python/adc/__init__.py",
        "measured": {
            "mass0": mass0,
            "drel": drel,
            "rho_min": float(rho_final.min()),
            "rho_max": float(rho_final.max()),
            "drho_max": float(rho_final.max() - rho_final.min()),
            "p_min_final": float(pr_final.min()),
            "p_max_final": float(pr_final.max()),
            "p_mean_final": p_mean,
            "p_center_t0": pc0,
            "p_center_final": float(p_center[-1]),
            "vmax_final": float(
                np.sqrt((U[1] / U[0]) ** 2 + (U[2] / U[0]) ** 2).max()
            ),
            "moved_final": float(amp[-1]),
            "amp_peak": float(amp[kmax]),
            "t_amp_peak": float(t[kmax]),
            "t_final": float(tt),
        },
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("ecrit :", f1)
    print("ecrit :", f2)
    print("ecrit :", os.path.join(FIGDIR, "provenance.json"))
    print(
        "drel=%.2e  drho_max=%.4f  p_center %.4f -> %.4f (moyenne %.4f)  amp_peak=%.4f @ t=%.3f"
        % (
            drel,
            prov["measured"]["drho_max"],
            pc0,
            p_center[-1],
            p_mean,
            amp[kmax],
            t[kmax],
        )
    )


if __name__ == "__main__":
    main()
