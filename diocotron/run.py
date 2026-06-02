#!/usr/bin/env python3
r"""Reproduction du benchmark diocotron de Hoffart-Maier-Shadid-Tomas (arXiv:2510.11808),
Section 5.3, AVEC NOTRE solveur `adc` (bindings Python de la facade compilee), 100 % Python.

Le papier valide son schema "structure-preserving" pour les equations magnetic Euler-Poisson
dans la LIMITE DE DERIVE MAGNETIQUE (omega_d << omega_p << omega_c) en reproduisant le TAUX DE
CROISSANCE de l'instabilite diocotron d'une colonne creuse, compare a la relation de dispersion
analytique. Ce modele reduit de derive E x B (la limite visee) se COMPOSE ici GENERIQUEMENT
depuis Python, un bloc `diocotron` + un Poisson de systeme a paroi conductrice circulaire,
via `adc.System`, SANS aucun solveur C++ dedie au diocotron. On reproduit donc directement la
figure-cle du papier (gamma numerique vs analytique vs mode l), en pur Python sur la lib.

Ce qui est produit (dans diocotron/figures/) :
  1. dispersion.png  : taux de croissance gamma_l vs mode azimutal l, courbe analytique
     (probleme aux valeurs propres radial de Petri/Davidson-Felice, resolu ici en numpy) +
     points mesures par notre solveur + cibles du papier (gamma_3=0.772, 4=0.911, 5=0.683).
  2. amplitude.png   : |c_l|(t) en echelle log, phase exponentielle + ajustement, modes 3/4/5.
  3. diocotron.gif   : evolution de la densite (mode l=4) ; l'anneau developpe l lobes qui
     s'enroulent (instabilite diocotron non lineaire).
  4. snapshots.png   : 4 instantanes de densite (t croissant) du meme run.

Geometrie (reproduit la cible analytique du papier) : anneau r0:r1:Rwall = 0.15:0.20:0.40
(= 6:8:16), paroi conductrice circulaire (Dirichlet), B0 = 1, couplage Poisson alpha = 1.

Usage : PYTHONPATH=<adc_cpp/build-py/python> python3 diocotron/run.py
"""
import math
import os
import sys

import numpy as np

import adc  # notre solveur (facade compilee)

# ---------------------------------------------------------------------------
# Geometrie de l'anneau (cibles analytiques du papier reproduites a cette geometrie).
R0, R1, RWALL = 0.15, 0.20, 0.40   # rayons interne / externe / paroi  (6:8:16)
L = 1.0                            # domaine carre [0,L]^2, centre (L/2, L/2)
B0, ALPHA = 1.0, 1.0
RHOBAR = 1.0                       # densite moyenne de l'anneau (convention papier)
WIDTH = 0.05                       # lissage du profil radial (eigenvalue)
MODES = [3, 4, 5]                  # modes du papier
PAPER = {3: 0.772, 4: 0.911, 5: 0.683}   # cibles analytiques (Section 5.3)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ===========================================================================
# 1. ANALYTIQUE : probleme aux valeurs propres radial de Petri (numpy, "full python").
#    omega L_m phi = m Omega L_m phi + q_m phi,  phi(0)=phi(Rw)=0,  Omega = -(1/r^2) int rho r'.
#    Port EXACT de adc_cases/include/adc/analysis/diocotron_growth.hpp (cf. ce fichier).
# ===========================================================================
# Geometrie ABSOLUE pour le probleme aux valeurs propres (memes RATIOS 6:8:16 que la simu,
# mais a une echelle ou le lissage w=0.05 represente un anneau NET ; le taux NORMALISE est
# invariant d'echelle, donc directement comparable a la simu qui tourne en r0:r1:wall =
# 0.15:0.20:0.40). C'est l'echelle exacte de adc_cases/analysis/diocotron_growth.hpp.
ANA_A, ANA_B, ANA_RW, ANA_W, ANA_N = 6.0, 8.0, 16.0, 0.05, 2000


def diocotron_density(r, a=ANA_A, b=ANA_B, rhobar=RHOBAR, w=ANA_W):
    # annulus lisse : ~rhobar entre a et b, ~0 ailleurs (port exact du C++).
    return 0.5 * rhobar * (np.tanh((r - a) / w) - np.tanh((r - b) / w))


def diocotron_eigenvalue(m, a=ANA_A, b=ANA_B, Rw=ANA_RW, rhobar=RHOBAR, w=ANA_W, N=ANA_N):
    h = Rw / N
    r = np.arange(N + 1) * h
    rho = 0.5 * rhobar * (np.tanh((r - a) / w) - np.tanh((r - b) / w))  # annulus lisse
    integrand = rho * r
    C = np.zeros(N + 1)
    C[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * h)       # int_0^r rho r' dr'
    Om = np.zeros(N + 1)
    Om[1:] = -C[1:] / (r[1:] ** 2)                                      # rotation azimutale
    n = N - 1
    ri = r[1:N]
    Lmat = np.zeros((n, n))
    np.fill_diagonal(Lmat, -2.0 / h ** 2 - (m * m) / (ri * ri))
    for k in range(1, n):
        Lmat[k, k - 1] = 1.0 / h ** 2 - 1.0 / (2 * h * r[k + 1])
    for k in range(n - 1):
        Lmat[k, k + 1] = 1.0 / h ** 2 + 1.0 / (2 * h * r[k + 1])
    Q = (m / ri) * ((rho[2:N + 1] - rho[0:N - 1]) / (2 * h))            # (m/r) drho/dr
    A = (m * Om[1:N])[:, None] * Lmat                                   # diag(m Omega) @ L
    A[np.arange(n), np.arange(n)] += Q
    M = np.linalg.solve(Lmat, A)
    ev = np.linalg.eigvals(M)
    dom = ev[np.argmax(ev.imag)]
    return (2.0 * math.pi / rhobar) * dom                              # normalisation papier


# ===========================================================================
# 2. NUMERIQUE : notre solveur. Mesure du taux de croissance du mode l.
# ===========================================================================
def bilinear_on_circle(field, n, radius, l_samples=256):
    """Echantillonne `field` (n x n, centres de cellules) sur un cercle de rayon `radius`
    centre au milieu du domaine, par interpolation bilineaire. Retourne (theta, valeurs)."""
    dx = L / n
    cx = cy = 0.5 * L
    theta = np.linspace(0.0, 2 * math.pi, l_samples, endpoint=False)
    px = cx + radius * np.cos(theta)
    py = cy + radius * np.sin(theta)
    fi = px / dx - 0.5
    fj = py / dx - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    ti = fi - i0
    tj = fj - j0
    # field[j, i] (row-major numpy de la facade : (n,n) avec [j*n+i])
    f00 = field[j0, i0]; f10 = field[j0, i0 + 1]
    f01 = field[j0 + 1, i0]; f11 = field[j0 + 1, i0 + 1]
    val = (f00 * (1 - ti) * (1 - tj) + f10 * ti * (1 - tj) +
           f01 * (1 - ti) * tj + f11 * ti * tj)
    return theta, val


def mode_l_amplitude(field, n, radius, l):
    """Amplitude du mode azimutal l de `field` sur le cercle (coef de Fourier)."""
    _, val = bilinear_on_circle(field, n, radius, 256)
    ck = np.fft.rfft(val) / len(val)
    return 2.0 * abs(ck[l])


def fit_linear_phase(t, a):
    """gamma = pente de log(a) sur la phase de croissance avant saturation (1.3 a0 -> 0.85 pic)."""
    t = np.asarray(t); a = np.asarray(a)
    good = a > 0
    t, a = t[good], a[good]
    if len(a) < 8:
        return float("nan"), (0, 0)
    ipk = int(np.argmax(a))
    a0 = a[0]
    lo = int(np.searchsorted(a[:ipk + 1], 1.3 * a0)) if ipk > 0 else 0
    hi = max(lo + 4, int(0.85 * ipk))
    hi = min(hi, len(a) - 1)
    if hi - lo < 4:
        lo, hi = 0, len(a) - 1
    coef = np.polyfit(t[lo:hi + 1], np.log(a[lo:hi + 1]), 1)
    return coef[0], (t[lo], t[hi])


def ring_density(n, l, delta):
    """CI anneau (mode l) ecrite EN PYTHON/numpy : ~1 entre R0 et R1, ~0 ailleurs,
    avec une perturbation azimutale sin(l*theta). C'est l'unique endroit ou la CI
    est definie, plus aucune fonction C++ par cas."""
    coord = (np.arange(n) + 0.5) / n * L
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
    th = np.arctan2(yy - 0.5 * L, xx - 0.5 * L)
    ne = np.full((n, n), 1e-3)
    ring = (r > R0) & (r < R1)
    ne[ring] = 1.0 - delta + delta * np.sin(l * th[ring])
    return ne


def make_ring_system(n, l, delta):
    """Compose le diocotron GENERIQUEMENT depuis Python : un bloc 'diocotron', un
    Poisson de systeme avec paroi conductrice circulaire. Aucun DiocotronSolver C++."""
    sim = adc.System(n=n, L=L, B0=B0, alpha=ALPHA, n_i0=0.0, periodic=False)
    sim.add_block("ne", model="diocotron", charge=1.0,
                  spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                    wall="circle", wall_radius=RWALL)
    sim.set_density("ne", ring_density(n, l, delta))
    return sim


def measure_growth(l, n=192, delta=0.01, cfl=0.4, nsteps=900):
    """Lance le diocotron mode l et mesure gamma_norm = gamma * 2 pi / rhobar."""
    sim = make_ring_system(n, l, delta)
    rm = 0.5 * (R0 + R1)
    ts, amp = [], []
    for step in range(nsteps + 1):
        phi = sim.potential()
        if not np.isfinite(phi).all():
            print(f"  [mode {l}] NaN au pas {step} -> arret", file=sys.stderr)
            break
        ts.append(sim.time())
        amp.append(mode_l_amplitude(phi, n, rm, l))
        if step == nsteps:
            break
        sim.step_cfl(cfl)
    ts = np.asarray(ts); amp = np.asarray(amp)
    gamma_raw, win = fit_linear_phase(ts, amp)
    gamma_norm = gamma_raw * 2.0 * math.pi / RHOBAR
    return ts, amp, gamma_norm, win


# ===========================================================================
# 3. GIF + snapshots de l'instabilite (mode l=4, perturbation visible).
# ===========================================================================
def run_evolution(l=4, n=192, delta=0.1, cfl=0.4, nframes=60, steps_per_frame=12):
    sim = make_ring_system(n, l, delta)
    frames, times = [], []
    for f in range(nframes):
        d = sim.density("ne")
        if not np.isfinite(d).all():
            break
        frames.append(d.copy()); times.append(sim.time())
        for _ in range(steps_per_frame):
            sim.step_cfl(cfl)
    return frames, times


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    print("=" * 74)
    print("Reproduction arXiv:2510.11808 (diocotron) avec le solveur adc : full Python")
    print("=" * 74)

    # --- (1) analytique : courbe de dispersion + verification des cibles du papier ---
    l_curve = np.arange(2, 8)
    gamma_ana = np.array([diocotron_eigenvalue(int(m)).imag for m in l_curve])
    print("\n[analytique] taux de croissance (eigenvalue de Petri, numpy) :")
    for m, g in zip(l_curve, gamma_ana):
        tag = f"  (papier {PAPER[m]:.3f})" if m in PAPER else ""
        print(f"   l={m}: gamma = {g:.3f}{tag}")

    # --- (2) numerique : mesure par notre solveur, modes 3/4/5 ---
    print("\n[numerique] mesure du taux par adc.System (diocotron compose, n=192, delta=0.01) :")
    runs = {}
    for l in MODES:
        ts, amp, gnum, win = measure_growth(l)
        runs[l] = (ts, amp, gnum)
        gana = diocotron_eigenvalue(l).imag
        err = 100 * (gnum - gana) / gana
        print(f"   l={l}: gamma_num = {gnum:.3f}  | analytique {gana:.3f} | papier "
              f"{PAPER[l]:.3f} | ecart {err:+.0f}%  (fenetre t in [{win[0]:.1f},{win[1]:.1f}])")

    # --- figure dispersion ---
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(l_curve, gamma_ana, "-o", color="0.4", label="analytique (Petri, eigenvalue numpy)")
    ax.scatter(list(PAPER), [PAPER[m] for m in PAPER], marker="s", s=70, facecolors="none",
               edgecolors="tab:red", linewidths=1.6, label="papier (arXiv:2510.11808)", zorder=5)
    ax.scatter(MODES, [runs[l][2] for l in MODES], marker="*", s=160, color="tab:blue",
               label="adc (notre solveur, Python)", zorder=6)
    ax.set_xlabel("mode azimutal $l$"); ax.set_ylabel(r"taux de croissance $\gamma$ (norm. $\omega_D$)")
    ax.set_title("Instabilite diocotron : taux de croissance\n(reproduction arXiv:2510.11808 avec adc)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dispersion.png"), dpi=130)
    plt.close(fig)

    # --- figure amplitude(t) log ---
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for l in MODES:
        ts, amp, gnum = runs[l]
        ax.semilogy(ts, amp, label=f"$l={l}$ (mesure)")
    ax.set_xlabel("temps"); ax.set_ylabel(r"amplitude du mode $|c_l|$ (de $\phi$)")
    ax.set_title("Croissance exponentielle du mode azimutal (phase lineaire)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "amplitude.png"), dpi=130)
    plt.close(fig)

    # --- (3) evolution -> gif + snapshots ---
    print("\n[evolution] run mode l=4 (delta=0.1) pour le gif...")
    frames, times = run_evolution(l=4)
    if frames:
        vmax = max(f.max() for f in frames)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        im = ax.imshow(frames[0], origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                       extent=[0, L, 0, L])
        ax.set_title("diocotron l=4 (adc)"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$n_e$")

        def update(k):
            im.set_data(frames[k]); ax.set_xlabel(f"t = {times[k]:.2f}")
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
        anim.save(os.path.join(OUT, "diocotron.gif"), writer=animation.PillowWriter(fps=12))
        plt.close(fig)

        idx = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, k in zip(axes, idx):
            ax.imshow(frames[k], origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                      extent=[0, L, 0, L])
            ax.set_title(f"t = {times[k]:.2f}"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Instabilite diocotron mode l=4 : densite (adc, reproduction arXiv:2510.11808)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "snapshots.png"), dpi=130)
        plt.close(fig)
        print(f"   {len(frames)} frames, t_final = {times[-1]:.2f}")

    print(f"\nFigures + gif ecrits dans : {OUT}")
    print("OK repro_paper_2510_11808")


if __name__ == "__main__":
    main()
