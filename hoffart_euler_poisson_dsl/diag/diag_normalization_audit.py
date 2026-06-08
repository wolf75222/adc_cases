#!/usr/bin/env python3
"""T2 -- audit de NORMALISATION du chemin hoffart ``system-schur``.

But (plan user T2)
-----------------
Expliquer le deficit -95 % du run ``system-schur`` (``gamma_raw ~ 0.032``, fenetres
papier, ``alpha = omega = 1e12``) PUREMENT par des facteurs DIMENSIONNELS derives a
l'avance -- pas ajustes apres coup. La conclusion (cf. ``../RESULTS_SYSTEM_SCHUR.md``
sections 7ter / 8 et ``../NORMALIZATION.md``) : le deficit n'est PAS la geometrie
cartesienne ; c'est (a) la FENETRE DE FIT et (b) le facteur de temps ``T_d = 2 pi``.

Cle dimensionnelle (``../model.py``)
------------------------------------
::

    alpha = beta^2 / rho_max ,   omega = beta^2  (= |Omega| = omega_c, le champ B_z)

La vitesse de derive du run COMPLET est ``v = grad(phi)/omega`` avec ``-Delta phi =
alpha rho``. En posant ``phi = alpha phi~`` (``-Delta phi~ = rho``) :

    v = (alpha/omega) grad(phi~) ,   alpha/omega = 1/rho_max = 1

=> ``v == grad(phi~)`` == EXACTEMENT la derive ExB NORMALISEE (``alpha=1``, ``B=1``).
Le ``1e12`` de ``alpha`` et le ``1e12`` de ``omega`` se SIMPLIFIENT dans le transport.
Donc le run complet et le reduit ExB NORMALISE (``B0=1``, ``charge=1``) advectent rho
avec le MEME champ de vitesse, dans les MEMES unites de temps de simulation, et
``gamma_raw`` est directement comparable. La seule difference possible entre 0.032
(run complet, RESULTS section 7) et ~0.10 (reduit cart, section 7ter) est la FENETRE.

Echelles diocotron (params papier, ``rho_max = 1``)
---------------------------------------------------
::

    omega_c = |Omega|              = beta^2                 (cyclotron)
    omega_d = rho_max alpha/|Omega| = rho_max (beta^2/rho_max)/beta^2 = 1   (diocotron, O(1))
    T_d     = 2 pi / omega_d        = 2 pi                  (periode diocotron == le "2 pi" du depot)

Comme ``omega_d = 1`` et ``alpha/omega = 1``, TOUS les candidats de scaling
dimensionnellement honnetes s'effondrent sur ``gamma_raw * 2 pi`` :

    c1 = gamma_raw * 2 pi
    c2 = gamma_raw * 2 pi * (alpha/omega)   (= c1, alpha/omega = 1)
    c3 = gamma_raw / omega_d                (= gamma_raw, omega_d = 1 -- no-op)
    c4 = gamma_raw * T_d                    (= c1, T_d = 2 pi)

Il n'existe AUCUN facteur ~3 supplementaire au niveau dimensionnel. Le "residu ~3x"
au-dela du 2 pi est la FENETRE : ``run.py:fit_growth`` masque le temps de SIMULATION
directement avec ``PAPER_FIT_WINDOWS`` ([0.40,0.70]...) alors que temps_papier =
``T_d`` x temps_sim ; la fenetre papier appliquee en temps SIM tombe dans le
TRANSITOIRE (taux encore en rampe), pas dans le regime exponentiel etabli.

Ce que mesure ce script
-----------------------
UN run ExB normalise par mode (champ de vitesse IDENTIQUE au run complet), fit dans
DEUX fenetres : la fenetre papier appliquee en temps sim (comme run.py, transitoire)
ET une fenetre etablie [3, 12]. Le ratio etabli/papier EST le facteur fenetre. Puis
on applique les 4 candidats au ``gamma_raw`` etabli et on montre qu'ils s'effondrent
sur ``* 2 pi``.

Lancer
------
::

    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_normalization_audit.py [n]

n par defaut = 128. Le module ``adc`` doit etre construit (chemin cartesien :
``adc.System(n, L)`` + transport ExB scalaire + Poisson ``geometric_mg`` paroi
circulaire). Reproduit ``gamma_raw(l=3, fenetre papier) ~ 0.031`` == la mesure du
run COMPLET (RESULTS section 1 : 0.0321), confirmant l'equivalence full == reduit.
"""

import math
import sys
import time

import numpy as np

import adc


# --- params papier (../model.py) ---
BETA = 1.0e6
RHO_MAX = 1.0
ALPHA = BETA * BETA / RHO_MAX            # 1e12
OMEGA = BETA * BETA                      # 1e12  (= omega_c = |Omega|)
OMEGA_C = OMEGA
OMEGA_D = RHO_MAX * ALPHA / abs(OMEGA)   # = 1   (frequence diocotron, echelle LENTE)
T_D = 2.0 * math.pi / OMEGA_D            # = 2 pi (periode diocotron)
FACTOR_ALPHA_OMEGA = ALPHA / OMEGA       # = 1/rho_max = 1 (seule combo a-dimensionnee)

# --- geometrie de l'anneau (echelle papier 6:8:16) ---
R0, R1, RW = 6.0, 8.0, 16.0
RHO_MIN, DELTA = 1.0e-6, 0.1
PAPER = {3: 0.772, 4: 0.911, 5: 0.683}
PAPER_WIN = {3: (0.40, 0.70), 4: (0.60, 0.75), 5: (1.15, 1.35)}
ESTABLISHED_WIN = (3.0, 12.0)
TWO_PI = 2.0 * math.pi


def exb_model():
    """Derive ExB scalaire NORMALISEE -- MEME champ de vitesse que le run complet (alpha/omega=1)."""
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(), elliptic=adc.ChargeDensity(charge=1.0))


def ring_ic_cart(n, l):
    """Anneau cartesien [R0, R1] perturbe au mode l, centre au milieu du carre L = 2 RW."""
    L = 2.0 * RW
    h = L / n
    x = (np.arange(n) + 0.5) * h - RW
    X, Y = np.meshgrid(x, x, indexing="xy")
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    dper = RHO_MAX * (1.0 - DELTA + DELTA * np.sin(l * th))
    return np.where((r >= R0) & (r <= R1), dper, RHO_MIN)


def run_cart(l, n, dt, t_end):
    """Run ExB normalise cartesien, dt FIXE (comme run.py). Renvoie (t, c_l(phi a r0))."""
    L = 2.0 * RW
    sim = adc.System(n=n, L=L, periodic=False)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                    wall="circle", wall_radius=RW)
    sim.add_block("ne", model=exb_model(), spatial=adc.Spatial(weno5=True),
                  time=adc.Explicit(method="ssprk3"))
    sim.set_density("ne", ring_ic_cart(n, l).reshape(-1))
    h = L / n
    th = np.linspace(0.0, 2.0 * math.pi, 1024, endpoint=False)
    xs = 0.5 * L + R0 * np.cos(th)
    ys = 0.5 * L + R0 * np.sin(th)
    fi = xs / h - 0.5
    fj = ys / h - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    tx, ty = fi - i0, fj - j0
    ts, cs = [], []
    while True:
        t = float(sim.time())
        phi = np.asarray(sim.potential(), float).reshape(n, n)
        if not np.isfinite(phi).all():
            break
        vals = (phi[j0, i0] * (1 - tx) * (1 - ty) + phi[j0, i0 + 1] * tx * (1 - ty)
                + phi[j0 + 1, i0] * (1 - tx) * ty + phi[j0 + 1, i0 + 1] * tx * ty)
        ts.append(t)
        cs.append((np.fft.rfft(vals) / vals.size)[l])
        if t >= t_end:
            break
        sim.step(min(dt, t_end - t))
    return np.array(ts), np.array(cs)


def gfit(ts, cs, lo, hi):
    """Pente de log|c_l| dans [lo, hi]. Renvoie (gamma, n_points)."""
    a = np.abs(cs)
    m = (ts >= lo) & (ts <= hi) & (a > 0)
    if m.sum() <= 4:
        return float("nan"), int(m.sum())
    return float(np.polyfit(ts[m], np.log(a[m]), 1)[0]), int(m.sum())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    dt, t_end = 2.0e-3, 15.0

    sep = "=" * 96
    print(sep)
    print("ECHELLES DIMENSIONNELLES (params papier beta=%.0e, rho_max=%.0f)" % (BETA, RHO_MAX))
    print("  alpha             = beta^2/rho_max     = %.3e" % ALPHA)
    print("  omega_c = |Omega| = beta^2             = %.3e   (champ B_z du run complet)" % OMEGA_C)
    print("  omega_d = rho_max*alpha/|Omega|        = %.6f      (diocotron, O(1) -- echelle LENTE)" % OMEGA_D)
    print("  T_d     = 2 pi / omega_d               = %.6f   (periode diocotron == le 2 pi du depot)" % T_D)
    print("  alpha/omega (seule combo a-dimensionnee) = %.6f  (= 1/rho_max ; le 1e12 se SIMPLIFIE)"
          % FACTOR_ALPHA_OMEGA)
    print(sep)
    print("TEST FENETRE : reduit ExB NORMALISE (B0=1, charge=1) == champ de vitesse du run complet.")
    print("UN run/mode (dt=%.0e fixe, t_end=%.1f, n=%d), fit fenetre PAPIER (sim) vs ETABLIE." % (dt, t_end, n))
    print("-" * 96)
    print("  l | win_papier (sim)  g_raw   npts | win_etablie [3,12]  g_raw   npts | ratio etabli/papier")
    for l in (3, 4, 5):
        t0 = time.time()
        ts, cs = run_cart(l, n, dt, t_end)
        lo, hi = PAPER_WIN[l]
        g_pap, np_pap = gfit(ts, cs, lo, hi)
        g_eta, np_eta = gfit(ts, cs, *ESTABLISHED_WIN)
        ratio = g_eta / g_pap if (np.isfinite(g_pap) and g_pap != 0) else float("nan")
        print("  %d | [%.2f,%.2f]  %.4f  n=%3d | [3.0,12.0]  %.4f  n=%4d | %.2f   (%.0fs, tf=%.1f)"
              % (l, lo, hi, g_pap, np_pap, g_eta, np_eta, ratio, time.time() - t0,
                 ts[-1] if len(ts) else 0.0))
    print(sep)
    print("CANDIDATS DE SCALING (derives A L'AVANCE ; appliques au g_raw ETABLI l=4) :")
    ts, cs = run_cart(4, n, dt, t_end)
    g_eta, _ = gfit(ts, cs, *ESTABLISHED_WIN)
    g_pap, _ = gfit(ts, cs, *PAPER_WIN[4])
    c1 = g_eta * TWO_PI
    c2 = g_eta * TWO_PI * FACTOR_ALPHA_OMEGA
    c3 = g_eta / OMEGA_D
    c4 = g_eta * T_D
    print("  g_raw etabli (l=4, [3,12]) = %.4f   | g_raw fenetre-papier = %.4f" % (g_eta, g_pap))
    print("  candidat_1 = g_raw * 2pi          = %.4f" % c1)
    print("  candidat_2 = g_raw * 2pi * (a/o)  = %.4f   (a/o = %.3f -> == candidat_1)" % (c2, FACTOR_ALPHA_OMEGA))
    print("  candidat_3 = g_raw / omega_d      = %.4f   (omega_d = %.3f -> NO-OP)" % (c3, OMEGA_D))
    print("  candidat_4 = g_raw * T_d          = %.4f   (T_d = 2pi -> == candidat_1)" % c4)
    print("  cible papier l=4                  = %.4f" % PAPER[4])
    print("-" * 96)
    print("Tous les candidats DIMENSIONNELLEMENT honnetes s'effondrent sur g_raw*2pi (a/o=1, omega_d=1,")
    print("T_d=2pi). Le seul facteur libre est 2 pi = T_d. Le 'residu ~3x' est la FENETRE (run.py fitte")
    print("le transitoire en temps sim), PAS une echelle manquante. Residu final ~20%% = grille cart vs")
    print("polaire (section 7ter), seule part NON metrologique.")


if __name__ == "__main__":
    main()
