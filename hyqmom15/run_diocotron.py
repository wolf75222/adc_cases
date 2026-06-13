#!/usr/bin/env python3
"""Cas "hyqmom15/run_diocotron" : couplage Vlasov-Poisson complet du modele HyQMOM 15 moments,
instabilite diocotron (anneau + perturbation azimutale), I/O cadences.

Pourquoi ce cas
---------------
Troisieme brique de l'integration HyQMOM  : le champ electrique cesse d'etre nul
(run_crossing) -- le Poisson du systeme est resolu sur M00 a chaque pas, E = -grad(phi) retro-
agit sur les 15 moments via la source electrique. Scenario de reference :
main_electrostatic_wave.m, section dicotron (anneau r0 = 0.35..r1 = 0.40, mode 4, omega_p = 25,
omega_c = -30 ; a l'execution la branche electrostatique SEULE est active, B n'entre que par la
derive ExB de la condition initiale -- fidele au driver MATLAB).

Validation
----------
  (1) oracle de Poisson analytique : densite 1 + eps cos(k x) uniforme en y -> avec
      Delta(phi) = (rho - fond) / lambda^2, phi = -eps cos(k x) / (lambda^2 k^2). On verifie
      sim.potential() (8e-4 a n = 64, discretisation) -- SIGNE ET ECHELLE epingles. Le fond
      neutralisant est EXPLICITE (param rho_background = moyenne du scenario, constante car la
      masse est conservee) : un rhs periodique a moyenne non nulle rend le MG singulier
      (constate : damier de Nyquist + re-solve divergent) ;
  (2) source electrique de bout en bout : Ex implicite = (rhs - advection)[M10]/M00 == gradient
      CENTRE de phi a 1e-16 (la table compilee lit exactement le champ resolu) == analytique a
      8e-4 ;
  (3) diocotron smoke : IC anneau + perturbation mode 4 + derive ExB (port de
      initialize_dicotron.m, Poisson IC en numpy/FFT fidele a poisson_fft.m), robust, rusanov,
      10 pas : etat fini, M00 > 0, masse conservee a 1e-12, phi fini ;
  (4) checkpoint/restart bit-stable : 2 pas apres restart == 2 pas sans interruption ;
  (5) snapshots cadences : write npz tous les 5 pas (3 fichiers), 15 moments + phi presents.

Ne prouve pas : le taux de croissance diocotron (le driver MATLAB de reference tourne en HLLC +
relaxation15 ; la comparaison quantitative attend avec vitesses exactes et un
golden HLL re-genere) ; la realisabilite long-terme sans relaxation15 ; r != 0.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import build_moment_model, gaussian_state  # noqa: E402

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

import adc  # noqa: E402

# Parametres du scenario de reference (main_electrostatic_wave.m, section dicotron).
R0, R1 = 0.35, 0.40
RHO_MIN, RHO_MAX = 1e-4, 1.0
EPS, MODE = 0.1, 4
OMEGA_P = 25.0
DEBYE = 1.0 / OMEGA_P          # longueur de Debye adimensionnee (domaine de cote 1)
OMEGA_C = -30.0
T = 1.0                        # temperature de la maxwellienne de base


def poisson_fft_ref(rho, lam, dx, dy):
    """Transcription numpy de poisson_fft.m (IC seulement) : Delta(phi) = (rho - moyenne)/lam^2,
    resolu en Fourier periodique. Sert a construire la derive ExB initiale, comme le MATLAB."""
    nx, ny = rho.shape
    rho = rho - rho.mean()
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    rho_hat = np.fft.fft2(rho)
    phi_hat = np.zeros_like(rho_hat)
    mask = K2 != 0
    phi_hat[mask] = -rho_hat[mask] / (lam ** 2 * K2[mask])  # signe electron (poisson_fft.m)
    return np.real(np.fft.ifft2(phi_hat))


def diocotron_state(n):
    """Port d'initialize_dicotron.m : anneau perturbe + derive ExB calculee sur le Poisson IC.
    @return (15, n, n), axe x en dernier (x = colonnes, y = lignes)."""
    h = 1.0 / n
    xm = -0.5 + (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xm, xm, indexing="xy")     # X varie sur l'axe 1 (colonnes) = x
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    rho = RHO_MIN * np.ones_like(R)
    mask = (R >= R0) & (R <= R1)
    rho[mask] = RHO_MAX * (1.0 - EPS + EPS * np.sin(MODE * theta[mask]))
    # Poisson IC (transpose : poisson_fft_ref attend (nx, ny) avec x en premier axe)
    phi = poisson_fft_ref(rho.T, DEBYE, h, h).T
    gphi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * h)
    gphi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * h)
    vx = -gphi_y / OMEGA_C                        # derive ExB (initialize_dicotron.m)
    vy = gphi_x / OMEGA_C
    U = np.empty((15, n, n))
    base = gaussian_state(1.0, 0.0, 0.0, T, 0.0, T)
    for j in range(n):
        for i in range(n):
            U[:, j, i] = gaussian_state(rho[j, i], vx[j, i], vy[j, i], T, 0.0, T)
    assert np.all(np.isfinite(U)) and base is not None
    return U


def build_sim(n, rho_bg, name="mom", riemann="rusanov", exact_speeds=False,
              solver="fft"):
    """System periodique + modele avec sources electriques et Poisson
    (Delta phi = (M00 - rho_bg)/lam^2, fond neutralisant = moyenne du scenario : un rhs
    periodique a moyenne non nulle rend le MG singulier), rusanov + borne bring-up."""
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    # robust=False sur le chemin exact : fidele au MATLAB (aucune garde) ; les planchers du
    # mode robust sont par ailleurs derivables depuis diff(Abs) si besoin.
    m = build_moment_model(name="hyqmom15_vp" + ("_ex" if exact_speeds else ""),
                           robust=not exact_speeds, with_sources=True,
                           q_over_m=1.0, omega_c=0.0, debye=DEBYE, rho_background=rho_bg,
                           omega_p=OMEGA_P, exact_speeds=exact_speeds)
    compiled = m.compile(os.path.join(case_output_dir("hyqmom15"),
                                      "hyqmom15_vp%s.so" % ("_ex" if exact_speeds else "")),
                         adc_include(), backend="aot")
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(name, model=compiled,
                     spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
                     time=adc.Explicit())
    # solver : "fft" par defaut : solveur DIRECT periodique, l'analogue
    # de poisson_fft.m -- meme operateur discret que le MG, zero tolerance iterative) ;
    # "fft_spectral" = symbole continu ;
    # "geometric_mg" disponible (cas non periodiques / comparaisons).
    sim.set_poisson(rhs="charge_density", solver=solver)
    return sim


def check_poisson_oracle():
    """(1) + (2) : sinusoide analytique. Le signe et l'echelle sont LUS sur l'assert qui passe
    (convention : ADC resout Delta(phi) = rhs, rhs = (M00 - rho_bg)/lam^2, fond explicite)."""
    n, eps = 64, 1e-3
    k = 2.0 * np.pi  # premier mode du domaine de cote 1
    sim = build_sim(n, rho_bg=1.0)  # moyenne de 1 + eps cos(kx)
    xm = -0.5 + (np.arange(n) + 0.5) / n
    rho = 1.0 + eps * np.cos(k * xm)[None, :] * np.ones((n, n))
    # etat gaussien au repos module en densite : M_pq lineaire en rho (les moments d'une
    # gaussienne a (u, T) fixes sont proportionnels a rho)
    base = gaussian_state(1.0, 0.0, 0.0, T, 0.0, T)
    U = base[:, None, None] * rho[None, :, :]
    sim.set_state("mom", U)
    sim.solve_fields()
    phi = np.array(sim.potential())
    # analytique : phi = -eps cos(kx)/(lam^2 k^2) (moyenne deflatee par le solveur)
    phi_ref = -(eps / (DEBYE ** 2 * k ** 2)) * np.cos(k * xm)[None, :] * np.ones((n, n))
    phi0 = phi - phi.mean()
    err = float(np.max(np.abs(phi0 - phi_ref)) / np.max(np.abs(phi_ref)))
    assert err < 2e-2, "Poisson : phi != analytique (err rel %.3e) -- signe ou echelle" % err
    print("(1) oracle Poisson : phi == -eps cos(kx)/(lam^2 k^2) a %.1e pres (n = %d) -- OK"
          % (err, n))

    # (2) la source compilee lit le champ resolu : Ex implicite = (rhs - advection)[M10] / M00,
    # confronte au gradient centre de phi (E = -grad phi) puis a l'analytique.
    rhs = np.array(sim.eval_rhs("mom"))
    m_free = build_moment_model(name="hyqmom15_free", robust=True)
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include
    free = m_free.compile(os.path.join(case_output_dir("hyqmom15"), "hyqmom15_free.so"),
                          adc_include(), backend="aot")
    sim0 = adc.System(n=n, L=1.0, periodic=True)
    sim0.add_equation("mom", model=free,
                      spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                      time=adc.Explicit())
    sim0.set_state("mom", U)
    rhs0 = np.array(sim0.eval_rhs("mom"))
    ex_implied = (rhs[1] - rhs0[1]) / U[0]    # S[M10] = qm M00 Ex, qm = 1
    h = 1.0 / n
    ex_fd = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * h)
    d = float(np.max(np.abs(ex_implied - ex_fd)) / np.max(np.abs(ex_fd)))
    assert d < 1e-10, ("source compilee : Ex implicite != -grad_x(phi) centre (err rel %.3e ; "
                       "le gradient du solveur differerait du centre)" % d)
    ex_ref = -(eps / (DEBYE ** 2 * k)) * np.sin(k * xm)[None, :] * np.ones((n, n))
    d2 = float(np.max(np.abs(ex_implied - ex_ref)) / np.max(np.abs(ex_ref)))
    assert d2 < 2e-2, "Ex implicite != analytique (err rel %.3e)" % d2
    print("(2) source compilee : Ex implicite == -grad phi centre (%.1e) == analytique (%.1e) "
          "-- OK" % (d, d2))


def check_diocotron():
    """(3) + (4) + (5) : smoke anneau diocotron, checkpoint/restart bit-stable, snapshots."""
    from adc_cases.common.io import case_output_dir

    n, nsteps, every = 64, 10, 5
    U0 = diocotron_state(n)
    sim = build_sim(n, rho_bg=float(U0[0].mean()))  # fond neutralisant = moyenne de l'anneau
    sim.set_state("mom", U0)
    sim.solve_fields()
    mass0 = float(U0[0].sum())
    out_dir = case_output_dir("hyqmom15")
    snaps = []
    for s in range(nsteps):
        sim.step_cfl(0.4)
        if (s + 1) % every == 0:
            snaps.append(sim.write(os.path.join(out_dir, "diocotron"), format="npz", step=s + 1))
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "etat non fini"
    assert np.all(U[0] > 0), "M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "masse non conservee (%.2e)" % drift
    phi = np.array(sim.potential())
    assert np.all(np.isfinite(phi)), "phi non fini"
    print("(3) diocotron : %d pas finis, M00 > 0, derive de masse %.1e, phi fini -- OK"
          % (nsteps, drift))

    # (4) checkpoint -> 2 pas ; vs 2 pas directs : bit-stable
    ck = os.path.join(out_dir, "diocotron_ck.h5")
    sim.checkpoint(ck)
    sim2 = build_sim(n, rho_bg=float(U0[0].mean()))
    sim2.restart(ck)
    for _ in range(2):
        sim.step_cfl(0.4)
        sim2.step_cfl(0.4)
    Ua = np.array(sim.get_state("mom"))
    Ub = np.array(sim2.get_state("mom"))
    assert np.array_equal(Ua, Ub), "restart non bit-stable (dmax %.2e)" % float(
        np.max(np.abs(Ua - Ub)))
    print("(4) checkpoint/restart : 2 pas apres restart bit-identiques -- OK")

    assert len(snaps) == nsteps // every, "snapshots cadences manquants"
    last = np.load(snaps[-1])
    assert any("state" in k or "mom" in k for k in last.files), "snapshot sans etat"
    assert any("phi" in k for k in last.files), "snapshot sans phi"
    print("(5) snapshots cadences : %d fichiers npz avec etat + phi -- OK" % len(snaps))


def check_diocotron_hll_exact():
    """(6) bascule : le diocotron complet (Poisson + source electrique) tourne en
    riemann='hll' avec les vitesses EXACTES (exact_speeds, ) -- la cible fidele au MATLAB.
    Smoke : 10 pas stables, masse conservee, phi fini. Le taux de croissance quantitatif vs un
    golden MATLAB-HLL long est un suivi (campagne dediee : le run de reference dure des heures
    sous Octave)."""
    n, nsteps = 64, 10
    U0 = diocotron_state(n)
    sim = build_sim(n, rho_bg=float(U0[0].mean()), riemann="hll", exact_speeds=True)
    sim.set_state("mom", U0)
    sim.solve_fields()
    mass0 = float(U0[0].sum())
    for _ in range(nsteps):
        sim.step_cfl(0.4)
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "diocotron HLL exact : etat non fini"
    assert np.all(U[0] > 0), "diocotron HLL exact : M00 non positif"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "diocotron HLL exact : masse non conservee (%.2e)" % drift
    assert np.all(np.isfinite(np.array(sim.potential()))), "phi non fini"
    print("(6) diocotron en riemann='hll' + vitesses exactes : %d pas stables, masse conservee "
          "%.1e, phi fini (cible fidele au MATLAB ; taux de croissance = campagne dediee) -- OK"
          % (nsteps, drift))


def check_poisson_solvers():
    """(7) fidelite Poisson : le MEME oracle sinusoidal a travers les trois
    solveurs. fft (stencil discret diagonalise) == geometric_mg au residu MG pres (meme
    operateur) et tous deux a O(h^2) du continu ; fft_spectral (symbole continu, l'exact
    poisson_fft.m de RIEMOM2D) atteint la solution continue a ~1e-12 : la MEME mesure
    discrimine le symbole ET prouve le no-default-change des chemins existants."""
    n, eps = 64, 1e-3
    k = 2.0 * np.pi
    xm = -0.5 + (np.arange(n) + 0.5) / n
    rho = 1.0 + eps * np.cos(k * xm)[None, :] * np.ones((n, n))
    base = gaussian_state(1.0, 0.0, 0.0, T, 0.0, T)
    U = base[:, None, None] * rho[None, :, :]
    phi_ref = -(eps / (DEBYE ** 2 * k ** 2)) * np.cos(k * xm)[None, :] * np.ones((n, n))
    errs = {}
    phis = {}
    for solver in ("fft", "fft_spectral", "geometric_mg"):
        sim = build_sim(n, rho_bg=1.0, solver=solver)
        sim.set_state("mom", U)
        sim.solve_fields()
        phi = np.array(sim.potential())
        phis[solver] = phi - phi.mean()
        errs[solver] = float(np.max(np.abs(phis[solver] - phi_ref)) / np.max(np.abs(phi_ref)))
    assert errs["fft_spectral"] < 1e-11, (
        "fft_spectral devrait etre exact sur la sinusoide (err %.2e)" % errs["fft_spectral"])
    for sname in ("fft", "geometric_mg"):
        assert 1e-5 < errs[sname] < 1e-2, (
            "%s : fenetre O(h^2) attendue (err %.2e)" % (sname, errs[sname]))
    dmg = float(np.max(np.abs(phis["fft"] - phis["geometric_mg"])) / np.max(np.abs(phi_ref)))
    assert dmg < 1e-5, "fft != geometric_mg (%.2e) : meme operateur attendu" % dmg
    print("(7) Poisson : fft_spectral == continu a %.1e ; fft == MG (%.1e) a O(h^2) du "
          "continu (%.1e) -- symbole discrimine, defauts intacts -- OK"
          % (errs["fft_spectral"], dmg, errs["fft"]))


def main():
    print("=== hyqmom15/run_diocotron : Vlasov-Poisson 15 moments, anneau diocotron ===")
    check_poisson_oracle()
    check_poisson_solvers()
    check_diocotron()
    check_diocotron_hll_exact()
    print("hyqmom15/run_diocotron : OK")


if __name__ == "__main__":
    main()
