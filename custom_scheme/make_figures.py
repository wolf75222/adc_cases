"""Figures du cas custom_scheme : transport ExB ecrit 100 % en numpy, Poisson par adc.

Rejoue exactement la physique de run.py (memes briques, memes parametres, meme schema
SSPRK2 + upwind ecrit en Python) en instrumentant l'evolution :
  - densite n a 4 instants (CI -> fin) : le transport numpy advecte la bande (mode 4) ;
  - |phi| resolu par adc aux memes 4 instants : le seul appel a la lib (l'oracle Poisson) ;
  - series temporelles : derive de masse relative, |phi|_max, vitesse ExB max, max|dn|.

Ecrit figures/{density_evolution,phi_evolution,diagnostics}.png + figures/provenance.json
(SHA adc_cpp/adc_cases, backend, resolution, nombres mesures). matplotlib Agg, aucune fenetre.
"""

from __future__ import annotations

import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from adc_cases import models  # noqa: E402
from adc_cases.common.checks import relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402

# Memes fonctions numpy que run.py (copie locale pour instrumenter l'evolution).
from run import drift, divergence_upwind, poisson_oracle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Parametres identiques a run.py.
NX, L, B0 = 96, 1.0, 1.0
DX = L / NX
CFL, NSTEPS = 0.4, 200
SNAP_STEPS = [0, 40, 100, 200]  # instants captures (en pas)


def rhs(sim: adc.System, n: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    phi = poisson_oracle(sim, n)
    vx, vy = drift(phi, DX, B0)
    speed = float(np.hypot(vx, vy).max())
    return divergence_upwind(n, vx, vy, DX), speed, phi


def main() -> None:
    n = band_density(NX, L, amp=1.0, width=0.05, mode=4, disp=0.02)
    n_i0 = float(n.mean())

    sim = adc.System(n=NX, L=L, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=B0, alpha=1.0, n_i0=n_i0))
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")

    mass0 = float(n.sum()) * DX * DX
    n0 = n.copy()

    n_snaps, phi_snaps, snap_t = [], [], []
    t = 0.0
    times, mass_drift, phimax, speedmax, dnmax = [], [], [], [], []

    for step in range(NSTEPS + 1):
        r1, speed, phi = rhs(sim, n)
        # diagnostics avant le pas (etat courant)
        times.append(t)
        mass_drift.append(relative_drift(float(n.sum()) * DX * DX, mass0))
        phimax.append(float(np.abs(phi).max()))
        speedmax.append(speed)
        dnmax.append(float(np.abs(n - n0).max()))
        if step in SNAP_STEPS:
            n_snaps.append(n.copy())
            phi_snaps.append(phi.copy())
            snap_t.append(t)
        if step == NSTEPS:
            break
        dt = CFL * DX / max(speed, 1e-12)
        n1 = n + dt * r1
        r2, _, _ = rhs(sim, n1)
        n = 0.5 * n + 0.5 * (n1 + dt * r2)
        t += dt

    # ---- Figure 1 : evolution de la densite (numpy) ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    vmin = min(float(s.min()) for s in n_snaps)
    vmax = max(float(s.max()) for s in n_snaps)
    for ax, s, tt in zip(axes, n_snaps, snap_t):
        im = ax.imshow(
            s,
            origin="lower",
            extent=[0, L, 0, L],
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title("n  (t = %.3f)" % tt)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="densite n")
    fig.suptitle(
        "custom_scheme : densite advectee par le transport ExB ecrit en numpy "
        "(bande mode 4)",
        fontsize=12,
    )
    fig.savefig(
        os.path.join(FIGDIR, "density_evolution.png"),
        dpi=110,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 2 : |phi| resolu par adc ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    pvmax = max(float(np.abs(p).max()) for p in phi_snaps)
    for ax, p, tt in zip(axes, phi_snaps, snap_t):
        im = ax.imshow(
            np.abs(p),
            origin="lower",
            extent=[0, L, 0, L],
            cmap="viridis",
            vmin=0.0,
            vmax=pvmax,
            aspect="equal",
        )
        ax.set_title("|phi|  (t = %.3f)" % tt)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(
        im, ax=axes, fraction=0.012, pad=0.02, label="|phi| (resolu par adc)"
    )
    fig.suptitle(
        "custom_scheme : |phi| self-consistant resolu par adc (le seul appel a la lib)",
        fontsize=12,
    )
    fig.savefig(
        os.path.join(FIGDIR, "phi_evolution.png"), dpi=110, bbox_inches="tight"
    )
    plt.close(fig)

    # ---- Figure 3 : diagnostics temporels ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    ax[0].semilogy(times, np.maximum(mass_drift, 1e-18), color="C0")
    ax[0].axhline(1e-12, color="r", ls="--", label="tolerance 1e-12")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("derive de masse relative")
    ax[0].set_title("masse : flux upwind conservatif")
    ax[0].legend()
    ax[1].plot(times, dnmax, color="C1")
    ax[1].axhline(1e-3, color="r", ls="--", label="seuil 1e-3")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("max|n(t) - n(0)|")
    ax[1].set_title("evolution : dynamique non triviale")
    ax[1].legend()
    ax[2].plot(times, phimax, color="C2", label="|phi|_max (adc)")
    ax[2].plot(times, speedmax, color="C3", label="vitesse ExB max")
    ax[2].set_xlabel("t")
    ax[2].set_title("couplage Poisson actif")
    ax[2].legend()
    fig.suptitle(
        "custom_scheme : diagnostics du schema Python (Poisson par adc)",
        fontsize=12,
    )
    fig.savefig(
        os.path.join(FIGDIR, "diagnostics.png"), dpi=110, bbox_inches="tight"
    )
    plt.close(fig)

    # ---- provenance ----
    def sha(path: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            return "unknown"

    prov = {
        "script": "custom_scheme/make_figures.py",
        "command": "python custom_scheme/make_figures.py",
        "produces": [
            "density_evolution.png",
            "phi_evolution.png",
            "diagnostics.png",
        ],
        "adc_cpp_sha": sha(adc.__file__.split("/python/adc/")[0]),
        "adc_cases_sha": sha(os.path.dirname(HERE)),
        "backend": "natif serie (adc.System : un bloc models.diocotron, Poisson geometric_mg)",
        "role_adc": "oracle de Poisson uniquement (set_density + solve_fields + potential)",
        "resolution": "%dx%d" % (NX, NX),
        "mode_band": 4,
        "nsteps": NSTEPS,
        "cfl": CFL,
        "python": "3.12.2",
        "adc_module": adc.__file__,
        "mass_drift_final": float(mass_drift[-1]),
        "phi_max_initial": float(phimax[0]),
        "phi_max_final": float(phimax[-1]),
        "speed_max_initial": float(speedmax[0]),
        "speed_max_final": float(speedmax[-1]),
        "dn_max_final": float(dnmax[-1]),
        "t_final": float(times[-1]),
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("figures ecrites dans", FIGDIR)
    print("  |phi|_max initial = %.6e  final = %.6e" % (phimax[0], phimax[-1]))
    print(
        "  vitesse ExB max   initial = %.6e  final = %.6e"
        % (speedmax[0], speedmax[-1])
    )
    print(
        "  derive masse finale = %.3e   max|dn| finale = %.3e   t_final = %.4f"
        % (mass_drift[-1], dnmax[-1], times[-1])
    )


if __name__ == "__main__":
    main()
