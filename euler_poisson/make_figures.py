"""Figures de diagnostic du cas euler_poisson (categorie validation).

Re-joue la physique du cas (memes parametres que run.py : N=64, L=1, gamma=1.4,
four_pi_G=1, rho0=1, dt=0.004, 20 pas) et produit trois figures dans figures/ :

  1. energy_vs_t.png   : E_tot(t) = U[3].sum() pour gravite (sign=+1) et PLASMA
                         (sign=-1) cote a cote. Lit visuellement le contraste de
                         signe asserte par run.py:177-180 (dE_grav<0, dE_plas>0).
  2. de_vs_eps.png     : |dE| vs eps en log-log, balayage de l'amplitude de
                         perturbation. La prediction falsifiable de la
                         linearisation est |dE| ~ eps^2 : on ajuste la pente et on
                         la compare a 2 (doubler eps quadruple |dE|).
  3. density_map.png   : carte 2D de la densite finale (gravite et plasma) pour la
                         CI initiale rho = rho0 (1 + eps cos(2 pi x / L)).

Ecrit aussi figures/provenance.json (memes champs que diocotron/figures/provenance.json :
adc_cpp_sha, adc_cases_sha, backend, resolution, python, + les nombres mesures).

Lancer (depuis euler_poisson/) :
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

# Memes constantes que run.py (un seul jeu de parametres, pas de divergence).
N, L, GAMMA, RHO0, DT, NSTEPS = 64, 1.0, 1.4, 1.0, 0.004, 20
EPS_REF = 0.01  # amplitude de reference (celle de run.py)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def make_sim(sign: float) -> adc.System:
    """Construit le systeme Euler-Poisson de reference (cf. run.py:93-99).

    Un seul bloc "gas" (modele euler_poisson) avec Poisson de systeme
    (geometric_mg). Le parametre `sign` fixe le signe du couplage : +1
    pour la gravite (attractif), -1 pour le plasma (repulsif).
    """
    sim = adc.System(n=N, L=L, periodic=True)
    sim.add_block(
        "gas",
        model=models.euler_poisson(
            sign=sign, gamma=GAMMA, four_pi_G=1.0, rho0=RHO0
        ),
        spatial=adc.Spatial(vanleer=True, flux="hllc"),
        time=adc.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    return sim


def initial_density(eps: float) -> np.ndarray:
    """Condition initiale de densite parametree par eps (cf. run.py:75-79).

    Renvoie rho = rho0 (1 + eps cos(2 pi x / L)), constante en y.
    """
    x = (np.arange(N) + 0.5) * L / N
    xx, _ = np.meshgrid(x, x, indexing="ij")
    return RHO0 * (1.0 + eps * np.cos(2.0 * np.pi * xx / L))


def run_energy_trace(
    sign: float, eps: float = EPS_REF
) -> tuple[np.ndarray, np.ndarray]:
    """Avance NSTEPS pas et trace l'energie fluide a chaque pas.

    Returns:
        Couple (t, E_tot) ou chaque entree couvre l'etat initial puis les
        NSTEPS pas. E_tot = U[3].sum() est l'energie fluide totale.
    """
    sim = make_sim(sign)
    sim.set_density("gas", initial_density(eps))
    ts = [0.0]
    es = [float(sim.get_state("gas")[3].sum())]
    for _ in range(NSTEPS):
        sim.advance(DT, 1)
        ts.append(float(sim.time()))
        es.append(float(sim.get_state("gas")[3].sum()))
    return np.array(ts), np.array(es)


def run_dE(sign: float, eps: float) -> float:
    """dE = E_tot(fin) - E_tot(debut) pour une amplitude eps donnee."""
    sim = make_sim(sign)
    sim.set_density("gas", initial_density(eps))
    e0 = float(sim.get_state("gas")[3].sum())
    for _ in range(NSTEPS):
        sim.advance(DT, 1)
    e1 = float(sim.get_state("gas")[3].sum())
    return e1 - e0


def final_density(sign: float, eps: float = EPS_REF) -> np.ndarray:
    """Densite rho(x,y) apres NSTEPS pas (composante 0 de l'etat)."""
    sim = make_sim(sign)
    sim.set_density("gas", initial_density(eps))
    for _ in range(NSTEPS):
        sim.advance(DT, 1)
    return sim.get_state("gas")[0].copy()


# --------------------------------------------------------------------------- #
# Figure 1 : E_tot(t) gravite vs plasma
# --------------------------------------------------------------------------- #
tg, eg = run_energy_trace(+1.0)
tp, ep = run_energy_trace(-1.0)
dE_grav = eg[-1] - eg[0]
dE_plas = ep[-1] - ep[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, (t, e, dE, title, col) in zip(
    axes,
    [
        (
            tg,
            eg,
            dE_grav,
            f"GRAVITE (sign=+1, attractif)\ndE = {dE_grav:+.3e}",
            "tab:blue",
        ),
        (
            tp,
            ep,
            dE_plas,
            f"PLASMA (sign=-1, repulsif)\ndE = {dE_plas:+.3e}",
            "tab:red",
        ),
    ],
):
    ax.plot(t, e - e[0], "o-", color=col, ms=4)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("temps t")
    ax.set_ylabel(
        r"$E_{tot}(t) - E_{tot}(0)$  (energie fluide $U[3].\mathrm{sum}()$)"
    )
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
fig.suptitle(
    r"Contraste energetique : $E_{tot}$ diminue pour la gravite, augmente pour le plasma"
    "\n(signe asserte par run.py:177-180, pas deduit du travail $v\\cdot g$ qui est >0 des deux cotes)",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIGDIR, "energy_vs_t.png"), dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2 : |dE| vs eps en log-log (prediction eps^2, pente attendue 2)
# --------------------------------------------------------------------------- #
EPS_SWEEP = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
dEg = np.array([run_dE(+1.0, e) for e in EPS_SWEEP])
dEp = np.array([run_dE(-1.0, e) for e in EPS_SWEEP])
slope_g = float(np.polyfit(np.log(EPS_SWEEP), np.log(np.abs(dEg)), 1)[0])
slope_p = float(np.polyfit(np.log(EPS_SWEEP), np.log(np.abs(dEp)), 1)[0])
# controle eps = 0 : dE doit etre exactement 0 (machine), borne basse de TOL_DE.
dEg0 = run_dE(+1.0, 0.0)
dEp0 = run_dE(-1.0, 0.0)

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.loglog(
    EPS_SWEEP,
    np.abs(dEg),
    "o-",
    color="tab:blue",
    label=f"|dE| gravite  (pente {slope_g:.3f})",
)
ax.loglog(
    EPS_SWEEP,
    np.abs(dEp),
    "s-",
    color="tab:red",
    label=f"|dE| plasma   (pente {slope_p:.3f})",
)
# droite de reference pente 2 ancree au point eps=0.01
ref = np.abs(dEg[1]) * (EPS_SWEEP / EPS_SWEEP[1]) ** 2
ax.loglog(
    EPS_SWEEP, ref, "--", color="0.4", label=r"pente 2 ($\propto \epsilon^2$)"
)
ax.axvline(EPS_REF, color="0.7", lw=0.8, ls=":")
ax.text(
    EPS_REF * 1.05,
    np.abs(dEg).min(),
    r"$\epsilon=0.01$ (run.py)",
    fontsize=8,
    color="0.4",
)
ax.set_xlabel(r"amplitude de perturbation $\epsilon$")
ax.set_ylabel(r"$|dE| = |E_{tot}(\mathrm{fin}) - E_{tot}(0)|$")
ax.set_title(
    r"Prediction falsifiable : $|dE| \propto \epsilon^2$ (pente 2 en log-log)"
)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "de_vs_eps.png"), dpi=130)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 : carte 2D de densite finale (gravite vs plasma)
# --------------------------------------------------------------------------- #
def run_conservation(sign: float, eps: float = EPS_REF) -> tuple[float, float]:
    """Mesure les invariants de conservation sur les NSTEPS pas.

    Memes diagnostics que run.py.

    Returns:
        Couple (max_rel_mass, max_mom) : la plus grande derive relative de
        masse et la plus grande impulsion |p| rencontrees sur le run.
    """
    sim = make_sim(sign)
    sim.set_density("gas", initial_density(eps))
    mass0 = float(sim.mass("gas"))
    max_rel_mass, max_mom = 0.0, 0.0
    for _ in range(NSTEPS):
        sim.advance(DT, 1)
        m = float(sim.mass("gas"))
        U = sim.get_state("gas")
        max_rel_mass = max(
            max_rel_mass, abs(m - mass0) / max(abs(mass0), 1e-30)
        )
        max_mom = max(max_mom, abs(float(U[1].sum())), abs(float(U[2].sum())))
    return max_rel_mass, max_mom


rho_g = final_density(+1.0)
rho_p = final_density(-1.0)
rho0_map = initial_density(EPS_REF)
mass_drift_g, mom_g = run_conservation(+1.0)
mass_drift_p, mom_p = run_conservation(-1.0)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
extent = [0, L, 0, L]
# echelle d'ecart au fond, commune, symetrique
amp = max(
    np.abs(rho_g - RHO0).max(),
    np.abs(rho_p - RHO0).max(),
    np.abs(rho0_map - RHO0).max(),
)
for ax, (field, title) in zip(
    axes,
    [
        (rho0_map, r"CI : $\rho_0(1+\epsilon\cos 2\pi x/L)$, $\epsilon=0.01$"),
        (rho_g, "gravite finale (t=0.08)"),
        (rho_p, "PLASMA finale (t=0.08)"),
    ],
):
    im = ax.imshow(
        (field - RHO0).T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-amp,
        vmax=amp,
        aspect="equal",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho - \rho_0$")
amp_ci = float(np.abs(rho0_map - RHO0).max() * 2)
amp_g = float(rho_g.max() - rho_g.min())
amp_p = float(rho_p.max() - rho_p.min())
fig.suptitle(
    "Densite finale : perturbation 1D en x (std-en-y ~ 3.8e-16). "
    f"Amplitude max-min : CI {amp_ci:.2e}, grav {amp_g:.2e}, plasma {amp_p:.2e}.\n"
    "Les deux s'aplatissent ; le contraste de signe n'est pas visible ici, il vit dans dE.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(FIGDIR, "density_map.png"), dpi=130)
plt.close(fig)


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


ADC_BUILD = os.path.dirname(
    os.path.dirname(adc.__file__)
)  # .../build-master/python
# adc_cpp = remonter depuis build-master/python/adc
adc_cpp_root = os.path.dirname(os.path.dirname(os.path.dirname(adc.__file__)))

prov = {
    "script": "euler_poisson/make_figures.py",
    "command": "python make_figures.py",
    "produces": ["energy_vs_t.png", "de_vs_eps.png", "density_map.png"],
    "adc_cpp_sha": git_sha(adc_cpp_root),
    "adc_cases_sha": git_sha(os.path.dirname(HERE)),
    "backend": "natif serie (adc.System, un bloc models.euler_poisson, Poisson geometric_mg)",
    "resolution": f"{N}x{N}",
    "periodic": True,
    "nsteps": NSTEPS,
    "dt": DT,
    "gamma": GAMMA,
    "four_pi_G": 1.0,
    "rho0": RHO0,
    "eps_ref": EPS_REF,
    "eps_sweep": EPS_SWEEP.tolist(),
    "python": sys.version.split()[0],
    "adc_module": adc.__file__,
    "measured": {
        "dE_grav": dE_grav,
        "dE_plas": dE_plas,
        "dE_grav_eps0": dEg0,
        "dE_plas_eps0": dEp0,
        "slope_dE_grav_vs_eps": slope_g,
        "slope_dE_plas_vs_eps": slope_p,
        "abs_dE_grav_sweep": np.abs(dEg).tolist(),
        "abs_dE_plas_sweep": np.abs(dEp).tolist(),
        "rho_amp_ci": float(np.abs(rho0_map - RHO0).max() * 2),
        "rho_amp_grav_final": float(rho_g.max() - rho_g.min()),
        "rho_amp_plas_final": float(rho_p.max() - rho_p.min()),
        "rho_std_in_y_grav": float(rho_g.std(axis=1).mean()),
        "max_rel_mass_grav": mass_drift_g,
        "max_rel_mass_plas": mass_drift_p,
        "max_mom_grav": mom_g,
        "max_mom_plas": mom_p,
    },
}
with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
    json.dump(prov, f, indent=2)

print("Figures ecrites dans", FIGDIR)
print(f"  dE_grav = {dE_grav:+.6e}   dE_plas = {dE_plas:+.6e}")
print(f"  pente |dE| vs eps : gravite {slope_g:.4f}, plasma {slope_p:.4f}")
print(f"  dE(eps=0) : gravite {dEg0:.3e}, plasma {dEp0:.3e}")
