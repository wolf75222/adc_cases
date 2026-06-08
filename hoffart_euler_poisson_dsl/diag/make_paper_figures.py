#!/usr/bin/env python3
"""Figures style Hoffart et al. (arXiv:2510.11808, Fig 5.1-5.4) pour le diocotron ADC.

Produit, dans la palette du papier (disque blanc, exterieur ardoise #3C4358, schlieren en
colormap Blues blanc->bleu marine) :
  - snapshots_l{3,4,5}.png : grille 3x3 de snapshots schlieren de la densite (Fig 5.1/5.2/5.3) ;
  - diocotron_l{3,4,5}.gif : animation du rollup diocotron (memes couleurs) ;
  - growth_rate.png        : amplitudes |c_l(t)|/|c_l(0)| semilog + fenetre de fit MAPPEE +
                             pente theorique papier, et (d) gamma vs cible (Fig 5.4).

Modele : densite advectee par la derive ExB NORMALISEE (Scalar + ExB(B0=1) + ChargeDensity(
charge=1)). C'est le champ que le full system-schur advecte (alpha/|Omega|=1/rho_max=1, cf.
T2_NORMALIZATION_AUDIT.md) -- representation de la limite de derive magnetique du papier. Le
panneau (d) porte AUSSI le taux du full system-schur (T3, RESULTS_SYSTEM_SCHUR.md section 9).
Temps final t_f = 10 periodes diocotron = 2pi (en temps sim, rhobar=rho_max=1).

Lancer :
    PYTHONPATH=<adc_cpp>/build-master/python \\
        python hoffart_euler_poisson_dsl/diag/make_paper_figures.py [l ...] [--out DIR]
(par defaut l=3 4 5, sortie ./hoffart_figures/). Necessite matplotlib + pillow (GIF).
"""
import math
import os
import sys
import time

import numpy as np

import adc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import to_rgba

# --- geometrie de l'anneau (echelle papier 6:8:16) ---
R0, R1, RW = 6.0, 8.0, 16.0
RHO_MIN, RHO_MAX, DELTA = 1e-6, 1.0, 0.1
TWO_PI = 2.0 * math.pi
SLATE = to_rgba("#3C4358")              # exterieur du disque (fond ardoise du papier)
BLUES = plt.get_cmap("Blues")
PAPER_WIN = {3: (0.40, 0.70), 4: (0.60, 0.75), 5: (1.15, 1.35)}   # fenetres papier (temps T_d)
PAPER_GAMMA = {3: 0.772, 4: 0.911, 5: 0.683}
FULL_T3 = {3: 0.702, 4: 0.894, 5: 0.683}    # full system-schur, n=96, T3 (RESULTS section 9)
SNAP_FRAC = (0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
FIG_NUM = {3: 1, 4: 2, 5: 3}


def ring_ic(n, l):
    h = 2 * RW / n
    x = (np.arange(n) + 0.5) * h - RW
    X, Y = np.meshgrid(x, x, indexing="xy")
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    dper = RHO_MAX * (1.0 - DELTA + DELTA * np.sin(l * th))
    return np.where((r >= R0) & (r <= R1), dper, RHO_MIN), r


def schlieren_rgba(rho, rmask, n):
    """log(1+|grad rho|) normalise -> Blues ; exterieur du disque -> ardoise. Style papier."""
    h = 2 * RW / n
    gy, gx = np.gradient(rho, h, h, edge_order=2)
    g = np.hypot(gx, gy)
    img = np.log1p(20.0 * g / max(float(g.max()), 1e-30))
    img = img / max(float(img.max()), 1e-30)
    rgba = BLUES(0.15 + 0.85 * img)
    rgba[img < 1e-3] = BLUES(0.0)
    rgba[rmask > RW] = SLATE
    return rgba


def sample_cl(phi2d, n, l):
    h = 2 * RW / n
    th = np.linspace(0.0, TWO_PI, 1024, endpoint=False)
    xs = RW + R0 * np.cos(th)
    ys = RW + R0 * np.sin(th)
    fi = xs / h - 0.5
    fj = ys / h - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    tx, ty = fi - i0, fj - j0
    v = (phi2d[j0, i0] * (1 - tx) * (1 - ty) + phi2d[j0, i0 + 1] * tx * (1 - ty)
         + phi2d[j0 + 1, i0] * (1 - tx) * ty + phi2d[j0 + 1, i0 + 1] * tx * ty)
    return abs((np.fft.rfft(v) / v.size)[l])


def run_mode(l, n=128, cfl=0.4, nframes=120):
    t_end = TWO_PI * 10.0     # t_f = 10 periodes diocotron (papier), en temps sim (rhobar=1)
    sim = adc.System(n=n, L=2 * RW, periodic=False)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                    wall="circle", wall_radius=RW)
    sim.add_block("ne", model=adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.ChargeDensity(charge=1.0)),
                  spatial=adc.Spatial(weno5=True), time=adc.Explicit(method="ssprk3"))
    rho0, rmask = ring_ic(n, l)
    sim.set_density("ne", rho0.reshape(-1))
    targets = sorted(set(round(x, 6) for x in
                         list(np.linspace(0.0, t_end, nframes)) + [f * t_end for f in SNAP_FRAC]))
    fi = 0
    frames, ftimes, ts, cs = [], [], [], []
    while True:
        t = float(sim.time())
        phi = np.asarray(sim.potential(), float).reshape(n, n)
        if not np.isfinite(phi).all():
            break
        ts.append(t)
        cs.append(sample_cl(phi, n, l))
        while fi < len(targets) and t >= targets[fi] - 1e-9:
            frames.append(np.asarray(sim.density("ne"), float).reshape(n, n).copy())
            ftimes.append(t)
            fi += 1
        if t >= t_end:
            break
        sim.step_cfl(cfl)
    return dict(l=l, n=n, t_end=t_end, rmask=rmask, frames=frames,
                ftimes=np.array(ftimes), ts=np.array(ts), cs=np.array(cs))


def make_snapshots(res, out):
    l, n, t_end, rmask, ft = res["l"], res["n"], res["t_end"], res["rmask"], res["ftimes"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.6))
    fig.patch.set_facecolor("white")
    for k, frac in enumerate(SNAP_FRAC):
        ax = axes.flat[k]
        idx = int(np.argmin(np.abs(ft - frac * t_end)))
        ax.imshow(schlieren_rgba(res["frames"][idx], rmask, n), origin="lower", extent=(-RW, RW, -RW, RW))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        lbl = "0.01" if abs(frac - 0.01) < 1e-9 else ("%d/8" % round(frac * 8) if frac < 1 else "1")
        ax.set_title(r"(%s) $t=%s\,t_f$" % ("abcdefghi"[k], lbl), fontsize=11, color="#222")
    fig.suptitle(r"Mode $\ell=%d$ -- schlieren densite (derive ExB), style Hoffart et al. Fig 5.%d"
                 % (l, FIG_NUM[l]), fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = os.path.join(out, "snapshots_l%d.png" % l)
    fig.savefig(p, dpi=160, facecolor="white"); plt.close(fig)
    return p


def make_gif(res, out, fps=15):
    l, n, rmask, tf = res["l"], res["n"], res["rmask"], res["t_end"]
    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    fig.patch.set_facecolor("white")
    im = ax.imshow(schlieren_rgba(res["frames"][0], rmask, n), origin="lower", extent=(-RW, RW, -RW, RW))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ttl = ax.set_title("", fontsize=12, color="#222")

    def upd(k):
        im.set_data(schlieren_rgba(res["frames"][k], rmask, n))
        ttl.set_text(r"diocotron $\ell=%d$   $t=%.2f\,t_f$" % (l, res["ftimes"][k] / tf))
        return im, ttl

    mov = animation.FuncAnimation(fig, upd, frames=len(res["frames"]), interval=1000 / fps)
    p = os.path.join(out, "diocotron_l%d.gif" % l)
    mov.save(p, writer=animation.PillowWriter(fps=fps)); plt.close(fig)
    return p


def gfit(ts, cs, lo, hi):
    a = np.abs(cs); m = (ts >= lo) & (ts <= hi) & (a > 0)
    return float(np.polyfit(ts[m], np.log(a[m]), 1)[0]) if m.sum() > 4 else float("nan")


def make_growth(results, out):
    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 4, wspace=0.32)
    exb = {}
    for j, l in enumerate((3, 4, 5)):
        r = next(x for x in results if x["l"] == l)
        ts, cs = r["ts"], np.abs(r["cs"])
        nz = np.flatnonzero(cs > 1e-12)             # t=0 : champ pas encore resolu -> ~0
        i0 = int(nz[0]) if len(nz) else 0
        a0 = cs[i0]
        tt = ts / TWO_PI
        gr = gfit(ts, cs, PAPER_WIN[l][0] * TWO_PI, PAPER_WIN[l][1] * TWO_PI)
        exb[l] = gr * TWO_PI
        ax = fig.add_subplot(gs[0, j])
        ax.semilogy(tt[i0:], cs[i0:] / a0, color="#1f3b73", lw=1.6)
        lo, hi = PAPER_WIN[l]
        m = (tt >= lo) & (tt <= hi)
        if m.any():
            anc = np.flatnonzero(m)[m.sum() // 2]
            ax.semilogy(tt[i0:], (cs[anc] / a0) * np.exp(PAPER_GAMMA[l] * (tt[i0:] - tt[anc])),
                        color="#c0392b", ls="--", lw=1.3, label=r"pente papier $\gamma_%d$" % l)
        ax.axvspan(lo, hi, color="#1f3b73", alpha=0.12)
        ax.set_ylim(0.5, 5e3)
        ax.set_xlabel(r"$t/t_f$")
        ax.set_title(r"(%s) $\ell=%d$:  $\gamma_{ExB}=%.3f$" % ("abc"[j], l, exb[l]), fontsize=10)
        if j == 0:
            ax.set_ylabel(r"$|c_\ell(t)|/|c_\ell(0)|$")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="lower right")
    axd = fig.add_subplot(gs[0, 3])
    ls = [3, 4, 5]
    axd.plot(ls, [PAPER_GAMMA[l] for l in ls], "s-", color="#c0392b", label="papier (theorie [13])")
    axd.plot(ls, [FULL_T3[l] for l in ls], "o-", color="#1f3b73", label="full system-schur (T3, n=96)")
    axd.plot(ls, [exb[l] for l in ls], "^--", color="#2a9d8f", label=r"ExB-derive ($\times2\pi$, n=128)")
    axd.set_xticks(ls); axd.set_xlabel(r"mode $\ell$")
    axd.set_title(r"(d) taux $\gamma_\ell$ ($\times2\pi/\bar\rho$)", fontsize=10)
    axd.grid(alpha=0.25); axd.legend(fontsize=7.5)
    fig.suptitle("Taux de croissance diocotron -- style Hoffart et al. Fig 5.4 (mesure paper-faithful T3)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(out, "growth_rate.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


def main():
    argv = [a for a in sys.argv[1:]]
    out = "hoffart_figures"
    if "--out" in argv:
        k = argv.index("--out")
        out = argv[k + 1]
        del argv[k:k + 2]
    modes = [int(x) for x in argv] or [3, 4, 5]
    os.makedirs(out, exist_ok=True)
    results = []
    for l in modes:
        t0 = time.time()
        r = run_mode(l)
        results.append(r)
        print("l=%d : %d frames, tf_sim=%.1f, %.0fs" % (l, len(r["frames"]), r["ts"][-1], time.time() - t0))
        print("  ", make_snapshots(r, out))
        print("  ", make_gif(r, out))
    if len(results) == 3:
        print("  ", make_growth(results, out))


if __name__ == "__main__":
    main()
