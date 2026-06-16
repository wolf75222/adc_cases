"""Genere les figures de diagnostic du cas diocotron_amr.

Re-joue la physique du cas (meme bande de charge, meme modele diocotron, meme schema
NoSlope+Rusanov, CFL=0.4) sur deux chemins :
  - AMR   : adc.AmrSystem (hierarchie grossier + 1 niveau fin, regrid Berger-Rigoutsos,
            reflux conservatif) ; mass()/density()/n_patches() sans argument de bloc ;
  - uniforme : adc.System a la meme resolution de base 64x64 ; mass("ne")/density("ne").

Les deux runs partagent band_density / models.diocotron / n_i0 = <n_e>, donc le seul
facteur change est la presence de l'AMR. On en tire :
  fig 1  density_compare.png : carte de densite finale uniforme | AMR | difference ;
  fig 2  patch_map.png       : footprint des cellules taggees (proxy de la couverture
          des patchs fins) a 3 instants, + n_patches(t) ;
  fig 3  mass_conservation.png : derive relative de masse vs t, uniforme vs AMR (reflux).

Chaque run mesure ses propres nombres ; ils sont ecrits dans figures/provenance.json.
Backend matplotlib Agg (pas d'affichage). Aucun nombre n'est invente : tout vient des runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
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
from adc_cases.common.checks import relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Memes constantes que run.py.
N, L = 64, 1.0
MODE, REFINE_FRAC = 4, 0.15
NSTEPS = 40
CFL = 0.4


def build_amr(ne: np.ndarray, n_i0: float, threshold: float) -> adc.AmrSystem:
    sim = adc.AmrSystem(n=N, L=L, regrid_every=10, periodic=True)
    sim.add_block(
        "ne",
        model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
        spatial=adc.Spatial(none=True),
    )
    sim.set_refinement(threshold=threshold)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def build_uniform(ne: np.ndarray, n_i0: float) -> adc.System:
    """Meme bloc diocotron, meme schema, meme Poisson, grille uniforme 64x64."""
    sim = adc.System(n=N, L=L, periodic=True)
    sim.add_block(
        "ne",
        model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
        spatial=adc.Spatial(none=True),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def main() -> None:
    ne = band_density(N, L, amp=1.0, width=0.05, mode=MODE, disp=0.02)
    n_i0 = float(ne.mean())
    threshold = n_i0 + REFINE_FRAC

    # --- AMR : mass(), density(), n_patches() sans argument de bloc ---
    amr = build_amr(ne, n_i0, threshold)
    m0_amr = amr.mass()
    t_amr, drel_amr, npatch_t = [0.0], [0.0], [amr.n_patches()]
    snap_steps = {1, NSTEPS // 2, NSTEPS - 1}
    tagged_snaps, tagged_times = [], []
    d_amr_init = np.asarray(amr.density()).copy()
    for k in range(NSTEPS):
        amr.step_cfl(CFL)
        d = np.asarray(amr.density())
        t_amr.append(amr.time())
        drel_amr.append(relative_drift(amr.mass(), m0_amr))
        npatch_t.append(amr.n_patches())
        if k in snap_steps:
            tagged_snaps.append((d > threshold).astype(float))
            tagged_times.append(amr.time())
    d_amr = np.asarray(amr.density())

    # --- uniforme : mass("ne"), density("ne") par nom de bloc ---
    uni = build_uniform(ne, n_i0)
    m0_uni = uni.mass("ne")
    t_uni, drel_uni = [0.0], [0.0]
    for _ in range(NSTEPS):
        uni.step_cfl(CFL)
        t_uni.append(uni.time())
        drel_uni.append(relative_drift(uni.mass("ne"), m0_uni))
    d_uni = np.asarray(uni.density("ne"))

    diff = d_amr - d_uni
    gap_uni_amr = float(np.abs(diff).max())

    extent = [0.0, L, 0.0, L]

    # ============ figure 1 : densite finale uniforme | AMR | difference ============
    vmin = float(min(d_uni.min(), d_amr.min()))
    vmax = float(max(d_uni.max(), d_amr.max()))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    im0 = ax[0].imshow(
        d_uni,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        aspect="equal",
    )
    ax[0].set_title("uniforme 64x64 (adc.System)\n$n_e$ a t=%.2f" % t_uni[-1])
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="$n_e$")
    im1 = ax[1].imshow(
        d_amr,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        aspect="equal",
    )
    ax[1].set_title("AMR base 64x64 + 1 niveau fin\n$n_e$ a t=%.2f" % t_amr[-1])
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="$n_e$")
    amax = float(np.abs(diff).max())
    im2 = ax[2].imshow(
        diff,
        origin="lower",
        extent=extent,
        vmin=-amax,
        vmax=amax,
        cmap="RdBu_r",
        aspect="equal",
    )
    ax[2].set_title("AMR - uniforme\n$\\max|\\Delta n_e|$ = %.3e" % gap_uni_amr)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="$\\Delta n_e$")
    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
    fig.suptitle(
        "Meme dynamique diocotron : l'AMR concentre la resolution sur la bande",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "density_compare.png"), dpi=110)
    plt.close(fig)

    # ============ figure 2 : footprint des cellules taggees + n_patches(t) ============
    fig, ax = plt.subplots(
        1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]}
    )
    for j, (tag, tt) in enumerate(zip(tagged_snaps, tagged_times)):
        ax[j].imshow(
            tag,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        ax[j].set_title(
            "cellules taggees (proxy patch fin)\nt=%.2f, %d cellules"
            % (tt, int(tag.sum()))
        )
        ax[j].set_xlabel("x")
        ax[j].set_ylabel("y")
    ax[3].step(t_amr, npatch_t, where="post", color="C3", lw=2)
    ax[3].set_ylim(0, max(npatch_t) + 1)
    ax[3].set_yticks(range(0, max(npatch_t) + 2))
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("n_patches()")
    ax[3].set_title("nombre de patchs fins au cours du temps")
    ax[3].grid(True, alpha=0.3)
    fig.suptitle(
        "Carte AMR : le critere tagge la bande de charge (seuil $n_{i0}+0.15$ = %.3f), "
        "couverte par n_patches >= 2" % threshold,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "patch_map.png"), dpi=110)
    plt.close(fig)

    # ============ figure 3 : conservation de masse uniforme vs AMR ============
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    floor = 1e-18  # plancher pour tracer un drel exactement nul en echelle log
    ax.semilogy(
        t_amr,
        np.maximum(drel_amr, floor),
        "o-",
        color="C3",
        ms=4,
        label="AMR (reflux conservatif)",
    )
    ax.semilogy(
        t_uni,
        np.maximum(drel_uni, floor),
        "s-",
        color="C0",
        ms=4,
        label="uniforme 64x64",
    )
    ax.axhline(
        1e-9, color="k", ls="--", lw=1, label="tolerance TOL_MASS = 1e-9"
    )
    ax.set_xlabel("t")
    ax.set_ylabel("derive relative de masse  |m(t) - m0| / |m0|")
    ax.set_title(
        "Conservation de masse : le reflux maintient l'AMR a l'arrondi machine"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mass_conservation.png"), dpi=110)
    plt.close(fig)

    # ============ provenance.json ============
    def sha(repo: str) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", repo, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    prov = {
        "script": "diocotron_amr/make_figures.py",
        "command": (
            "PYTHONPATH=<adc_build>:<adc_cases> python diocotron_amr/make_figures.py"
        ),
        "produces": [
            "density_compare.png",
            "patch_map.png",
            "mass_conservation.png",
        ],
        "adc_cpp_sha": sha(
            os.path.dirname(os.path.dirname(adc.__file__))
            if "adc_cpp" not in adc.__file__
            else "/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"
        ),
        "adc_cases_sha": sha(os.path.dirname(HERE)),
        "backend": "natif (adc.AmrSystem + adc.System, composition models.diocotron, "
        "NoSlope+Rusanov, geometric_mg periodique)",
        "matplotlib_backend": matplotlib.get_backend(),
        "resolution": "base 64x64 (AMR : + 1 niveau fin) ; uniforme 64x64",
        "regrid_every": 10,
        "nsteps": NSTEPS,
        "cfl": CFL,
        "band_mode": MODE,
        "python": sys.version.split()[0],
        "adc_module": adc.__file__,
        "n_i0_background": n_i0,
        "refine_threshold": threshold,
        "patches_observed": sorted(set(npatch_t)),
        "amr_mass0": m0_amr,
        "amr_mass_drel_final": float(drel_amr[-1]),
        "amr_mass_drel_max": float(max(drel_amr)),
        "uniform_mass0": m0_uni,
        "uniform_mass_drel_final": float(drel_uni[-1]),
        "uniform_mass_drel_max": float(max(drel_uni)),
        "amr_density_min": float(d_amr.min()),
        "amr_density_max": float(d_amr.max()),
        "uniform_density_min": float(d_uni.min()),
        "uniform_density_max": float(d_uni.max()),
        "gap_amr_vs_uniform_sup": gap_uni_amr,
        "tagged_cells_init": int((d_amr_init > threshold).sum()),
        "t_final_amr": float(t_amr[-1]),
        "t_final_uniform": float(t_uni[-1]),
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    # Trace console (les nombres a citer dans le README).
    print("== make_figures diocotron_amr ==")
    print("n_i0 (fond)            = %.6f" % n_i0)
    print("seuil de raffinement   = %.6f" % threshold)
    print("patchs observes (AMR)  = %s" % sorted(set(npatch_t)))
    print(
        "AMR  : mass0 = %.12e  drel_final = %.3e  drel_max = %.3e"
        % (m0_amr, drel_amr[-1], max(drel_amr))
    )
    print(
        "uni  : mass0 = %.12e  drel_final = %.3e  drel_max = %.3e"
        % (m0_uni, drel_uni[-1], max(drel_uni))
    )
    print("AMR density  min/max   = %.6f / %.6f" % (d_amr.min(), d_amr.max()))
    print("uni density  min/max   = %.6f / %.6f" % (d_uni.min(), d_uni.max()))
    print("max|AMR - uniforme|    = %.6e" % gap_uni_amr)
    print("t_final AMR / uni      = %.4f / %.4f" % (t_amr[-1], t_uni[-1]))
    print("figures ecrites dans   : %s" % FIGDIR)


if __name__ == "__main__":
    main()
