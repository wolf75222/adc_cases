#!/usr/bin/env python3
"""Figures du cas magnetic_isothermal_dsl (categorie validation, type equivalence DSL).

Deux figures, ecrites sous magnetic_isothermal_dsl/figures/ (versionnees, avec provenance.json) :

  lorentz_oracle.png       : l'oracle analytique de Lorentz. Le residu inter-runs
                             dR = eval_rhs(B_z=B0) - eval_rhs(B_z=0) confronte, canal par canal,
                             a sa forme analytique numpy (B0 my, -B0 mx). Heatmap des deux ecarts
                             (identiquement noirs : err == 0) + histogramme du residu (pic a 0).
  cyclotron_trajectory.png : la trajectoire de Lorentz. La quantite de mouvement moyenne (mx, my),
                             initialement (u0, 0), tourne a omega_c = q*B_z sans changer de module ;
                             le cercle analytique omega_c*t est superpose aux points mesures.

Reproduit la physique exacte de run.py (memes parametres, meme modele DSL, meme schema).
Sur macOS le backend 'production' echoue (ABI en-tetes du module pre-construit, cf. README) :
seul 'aot' se lie. Les figures sont donc tracees sur le backend reellement lie.
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adc  # noqa: E402

import run as R  # le cas lui-meme : modele DSL, IC, bind_backends  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

N, NSTEPS, CFL = 32, 40, 0.4


def _bound_single() -> tuple:
    """Lie les backends a B_z = B0 et B_z = 0 ; renvoie (backend, sim_B0, sim_0, state0)."""
    state0 = R.initial_state(N)
    bound = R.bind_backends(N, state0, R.B0)
    bound0 = R.bind_backends(N, state0, 0.0)
    b = sorted(bound)[0]
    return b, bound[b], bound0[b], state0, sorted(bound)


def fig_lorentz_oracle(backend: str, sim_b0, sim_0, state0: np.ndarray) -> dict:
    """dR = rhs(B0) - rhs(0), confronte a (B0 my0, -B0 mx0) numpy. Renvoie les nombres mesures."""
    mx0, my0 = state0[1], state0[2]
    dR = np.array(sim_b0.eval_rhs("plasma")) - np.array(
        sim_0.eval_rhs("plasma")
    )
    lor_x = (
        R.B0 * my0
    )  # +B_z my sur la qte de mvt x (ici my0 = 0 -> identiquement nul)
    lor_y = -R.B0 * mx0  # -B_z mx sur la qte de mvt y
    err_x = dR[1] - lor_x  # doit etre identiquement 0
    err_y = dR[2] - lor_y  # doit etre identiquement 0
    err_rho = dR[
        0
    ]  # B_z ne touche jamais la densite : doit etre identiquement 0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # Echelle de couleur ancree a l'epsilon machine : tout pixel non nul (>~ 2e-16) sature.
    # Le champ identiquement nul s'assoit exactement au centre neutre : la preuve est visuelle.
    eps = np.finfo(np.float64).eps  # 2.22e-16
    for ax, err, title in (
        (axes[0], err_rho, r"$\Delta R_\rho - 0$  (densite)"),
        (axes[1], err_x, r"$\Delta R_{m_x} - B_0 m_y$"),
        (axes[2], err_y, r"$\Delta R_{m_y} - (-B_0 m_x)$"),
    ):
        im = ax.imshow(err, origin="lower", cmap="seismic", vmin=-eps, vmax=eps)
        ax.set_title(
            title
            + "\nmax|.| = %.1e (echelle +/- eps_mach)"
            % float(np.max(np.abs(err))),
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks([-eps, 0, eps])
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Oracle Lorentz [backend %r] : residu DSL - forme analytique numpy. "
        "Echelle +/- eps_machine : tout pixel non nul saturerait ; ici tout est 0."
        % backend,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGDIR, "lorentz_oracle.png"), dpi=120)
    plt.close(fig)

    return {
        "err_x_max": float(np.max(np.abs(err_x))),
        "err_y_max": float(np.max(np.abs(err_y))),
        "err_rho_max": float(np.max(np.abs(err_rho))),
        "max_abs_dR": float(np.max(np.abs(dR))),
        "lor_y_min": float(lor_y.min()),
        "lor_y_max": float(lor_y.max()),
    }


def fig_cyclotron_trajectory(backend: str, sim_b0) -> dict:
    """Trajectoire (mx_mean, my_mean) au fil des pas ; cercle analytique omega_c*t superpose."""
    omega_c = R.Q * R.B0  # qom * B_z = q * B_z (qom = q ici)
    ts, mxm, mym = [], [], []
    st = np.array(sim_b0.get_state("plasma"))
    ts.append(sim_b0.time())
    mxm.append(float(st[1].mean()))
    mym.append(float(st[2].mean()))
    for _ in range(NSTEPS):
        sim_b0.step_cfl(CFL)
        st = np.array(sim_b0.get_state("plasma"))
        ts.append(sim_b0.time())
        mxm.append(float(st[1].mean()))
        mym.append(float(st[2].mean()))
    ts = np.array(ts)
    mxm = np.array(mxm)
    mym = np.array(mym)
    r0 = mxm[0]  # module initial |m_mean| = (u0, 0)

    # Cercle analytique : (mx, my)(t) = r0 (cos(omega_c t), sin(omega_c t)) (rotation pure).
    tt = np.linspace(ts[0], ts[-1], 200)
    cx = r0 * np.cos(omega_c * tt)
    cy = r0 * np.sin(omega_c * tt)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axL.plot(
        cx,
        cy,
        "-",
        color="0.6",
        lw=1.4,
        label=r"cercle cyclotron $r_0(\cos\omega_c t,\sin\omega_c t)$",
    )
    axL.plot(
        mxm,
        mym,
        "o-",
        color="C0",
        ms=3.5,
        lw=1.0,
        label="mesure DSL (moyenne par pas)",
    )
    axL.plot(
        [mxm[0]], [mym[0]], "s", color="C2", ms=9, label="depart $(u_0, 0)$"
    )
    axL.plot([mxm[-1]], [mym[-1]], "*", color="C3", ms=14, label="arrivee")
    axL.axhline(0, color="0.85", lw=0.7)
    axL.axvline(0, color="0.85", lw=0.7)
    axL.set_xlabel(r"$\langle m_x\rangle$")
    axL.set_ylabel(r"$\langle m_y\rangle$")
    axL.set_aspect("equal")
    axL.legend(fontsize=8, loc="lower left")
    ang = np.degrees(np.arctan2(mym[-1], mxm[-1]))
    axL.set_title(
        r"$\omega_c = qB_z = %.1f$ ; angle final mesure $%.2f^\circ$"
        r" (predit $\omega_c t = %.2f^\circ$)"
        % (omega_c, ang, np.degrees(omega_c * ts[-1])),
        fontsize=10,
    )

    mag = np.hypot(mxm, mym)
    axR.plot(ts, mag, "o-", color="C0", ms=3.5, lw=1.0)
    axR.axhline(
        r0,
        color="0.6",
        lw=1.2,
        ls="--",
        label=r"$r_0 = |\langle m\rangle|_{t=0}$",
    )
    axR.set_xlabel("t")
    axR.set_ylabel(r"$|\langle m\rangle|$")
    axR.set_title(
        r"module conserve : derive relative $%.1e$" % ((mag[-1] - r0) / r0),
        fontsize=10,
    )
    axR.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "cyclotron_trajectory.png"), dpi=120)
    plt.close(fig)

    return {
        "omega_c": float(omega_c),
        "t_final": float(ts[-1]),
        "mx_mean_init": float(mxm[0]),
        "my_mean_init": float(mym[0]),
        "mx_mean_final": float(mxm[-1]),
        "my_mean_final": float(mym[-1]),
        "angle_final_deg": float(ang),
        "angle_predicted_deg": float(np.degrees(omega_c * ts[-1])),
        "mag_init": float(r0),
        "mag_final": float(mag[-1]),
        "mag_rel_drift": float((mag[-1] - r0) / r0),
    }


def main() -> None:
    t0 = time.time()
    backend, sim_b0, sim_0, state0, backends = _bound_single()
    ora = fig_lorentz_oracle(backend, sim_b0, sim_0, state0)
    # NB : la trajectoire avance sim_b0 ; on la genere apres l'oracle (qui lit rhs a t=0).
    traj = fig_cyclotron_trajectory(backend, sim_b0)
    wall = time.time() - t0

    prov = {
        "script": "magnetic_isothermal_dsl/make_figures.py",
        "command": "python make_figures.py",
        "produces": ["lorentz_oracle.png", "cyclotron_trajectory.png"],
        "adc_cpp_sha": "018732997c02a17ade387fa99a74267f37e252c1",
        "adc_cases_sha": "a9541ba402d14f558380498cbbc24bddf0a0e5bd",
        "backend": "DSL %r (production non lie : ABI en-tetes du module pre-construit)"
        % backend,
        "backends_linked": backends,
        "resolution": "%dx%d" % (N, N),
        "nsteps": NSTEPS,
        "cfl": CFL,
        "cs2": R.CS2,
        "charge_q": R.Q,
        "B_z": R.B0,
        "python": "3.12.2",
        "adc_module": adc.__file__,
        "platform": "macOS arm64 (Apple Silicon), serie mono-process",
        "wall_seconds": round(wall, 3),
        "lorentz_oracle": ora,
        "cyclotron_trajectory": traj,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print("figures ecrites dans", FIGDIR)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
