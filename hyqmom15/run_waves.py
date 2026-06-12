#!/usr/bin/env python3
"""Cas "hyqmom15/run_waves" : vitesses d'onde HLL EXACTES du modele 15 moments, validees contre
le chemin de production du MATLAB de reference (eigenvalues15_2D.m, flagsym = 1).

Pourquoi ce cas
---------------
Le verrou HLL de l'integration HyQMOM  : HLL exige smin/smax signees, obtenues
dans le MATLAB par les valeurs propres de la jacobienne de flux symbolique (jacobian15.m,
400 lignes generees) extraites PAR BLOCS. Ici, la meme verite sort de l'AUTODIFF du flux DSL
declare (dsl.diff via m.wave_speeds_from_jacobian, adc_cpp ) + eig numerique par
sous-blocs (adc::real_eig_minmax, ) : aucune jacobienne generee a la main, aucun
SymPy -- le jacobien ne peut pas se desynchroniser du flux. Les partitions de blocs
(model.HYQMOM_BLOCKS) sont le miroir exact du flagsym = 1 : en x les chaines contiguës
1:5 / 6:9 / 13:15 (10:12 sciemment saute), en y les images par la symetrie x<->y -- listes
d'indices NON CONTIGUES [0,5,9,12,14] / [1,6,10,13] / [3,8,4] sur le dFy/dU DIRECT,
strictement equivalentes au swap d'arguments de jacobian15 du MATLAB.

Validation (golden/golden_vp.csv : eigenvalues15_2D(M, 1) execute par Octave sur RIEMOM2D,
meme provenance que les goldens de flux)
-----------------------------------------
  (1) [vpxmin, vpxmax, vpymin, vpymax] reproduits a rtol <= 1e-8 sur les 9 etats bien
      conditionnes -- la direction y est un GATE DUR (c'est la que la structure de blocs se
      rate silencieusement) ; en pratique l'accord observe est ~1e-15 (precision machine) ;
  (2) l'etat quasi-degenere (variance ~1e-6, paires de valeurs propres quasi-defectives) a une
      tolerance DEDUITE EN TEST : on mesure la sensibilite du probleme aux perturbations
      d'arrondi des entrees du jacobien (1e-12 relatif -> deplacement des extremes), et on
      exige |ecart| <= 100 x sensibilite mesuree. L'ecart observe (~8e-4) est DANS le
      conditionnement du probleme aux valeurs propres lui-meme, pas un defaut du chemin ;
  (3) la borne CFL exacte (max_wave_speed, memes blocs) COUVRE les vraies vitesses sur les 10
      etats -- y compris les 2 etats asymetriques qui DEPASSAIENT la borne bring-up k*sqrt(C)
      (run.py, invariant 6) : la faille de surete du bring-up est FERMEE par le chemin exact.

Ne prouve pas : l'execution compilee de ce chemin dans un System (couverte par les tests
adc_cpp de : eval_rhs HLL == reference numpy a 8e-15 sur jouet non lineaire) ; la
bascule des drivers en riemann='hll'. 
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import HYQMOM_BLOCKS, build_moment_model  # noqa: E402

NEAR_DEGENERATE = 8  # etat quasi-degenere du jeu golden (melange resserre, C20 ~ 1e-6)


def load_golden_vp():
    g = os.path.join(HERE, "golden")
    states = np.loadtxt(os.path.join(g, "golden_states.csv"), delimiter=",")
    vp = np.loadtxt(os.path.join(g, "golden_vp.csv"), delimiter=",")
    return states, vp


def measured_sensitivity(m, M, blocks_x):
    """Sensibilite du probleme aux valeurs propres de l'etat @p M : perturbation relative 1e-12
    des entrees du jacobien -> plus grand deplacement relatif des extremes (20 tirages). C'est
    la borne d'incertitude INTRINSEQUE entre deux jacobiennes algebriquement identiques
    arrondies differemment (autodiff DSL vs jacobian15.m symbolique)."""
    env = m._m._env(M[:, None], {})
    rows = m._m._ws_jacobian["rows"]["x"]
    n = len(rows)
    J = np.array([[float(np.asarray(rows[i][j].eval(env)).ravel()[0]) for j in range(n)]
                  for i in range(n)])
    rng = np.random.default_rng(0)
    worst = 0.0
    for b in blocks_x:
        idx = np.asarray(b)
        A = J[np.ix_(idx, idx)]
        lam = np.sort(np.linalg.eigvals(A).real)
        for _ in range(20):
            Ap = A * (1.0 + 1e-12 * rng.standard_normal(A.shape))
            lp = np.sort(np.linalg.eigvals(Ap).real)
            den = max(abs(lam[0]), abs(lam[-1]), 1e-30)
            worst = max(worst, abs(lp[0] - lam[0]) / den, abs(lp[-1] - lam[-1]) / den)
    return worst


def main():
    print("=== hyqmom15/run_waves : vitesses HLL exactes vs eigenvalues15_2D.m (flagsym=1) ===")
    states, vp = load_golden_vp()
    m = build_moment_model(name="hyqmom15_exact", exact_speeds=True)

    U = states.T
    lox, hix = m.eval_wave_speeds(U, {}, 0)
    loy, hiy = m.eval_wave_speeds(U, {}, 1)
    got = np.stack([lox, hix, loy, hiy], axis=1)
    err = np.abs(got - vp) / np.maximum(1e-30, np.abs(vp))

    # (1) etats bien conditionnes : rtol 1e-8, x ET y (gate dur en y)
    well = [i for i in range(states.shape[0]) if i != NEAR_DEGENERATE]
    for i in well:
        assert err[i, :2].max() < 1e-8, "etat %d : vpx (err %.2e)" % (i, err[i, :2].max())
        assert err[i, 2:].max() < 1e-8, "etat %d : vpy (err %.2e) -- GATE DUR y" % (
            i, err[i, 2:].max())
    print("(1) 9 etats bien conditionnes : [vpxmin vpxmax vpymin vpymax] == MATLAB, pire "
          "err rel %.1e (x) / %.1e (y) -- OK"
          % (max(err[i, :2].max() for i in well), max(err[i, 2:].max() for i in well)))

    # (2) etat quasi-degenere : tolerance deduite de la sensibilite mesuree du probleme
    sens = measured_sensitivity(m, states[NEAR_DEGENERATE], HYQMOM_BLOCKS["x"])
    tol = 100.0 * sens
    e8 = err[NEAR_DEGENERATE].max()
    assert e8 < tol, ("etat quasi-degenere : ecart %.2e > 100 x sensibilite mesuree %.2e -- "
                      "ce ne serait PLUS du conditionnement" % (e8, sens))
    print("(2) etat quasi-degenere : ecart %.1e <= 100 x sensibilite mesuree %.1e (paires de "
          "valeurs propres quasi-defectives : conditionnement du probleme, pas du chemin) -- OK"
          % (e8, sens))

    # (3) la borne CFL exacte couvre les vraies vitesses sur TOUS les etats (faille bring-up
    # fermee : run.py invariant 6 montrait 2 etats au-dela de k*sqrt(C)). Marge relative :
    # 1e-6 sur les etats bien conditionnes ; sur le quasi-degenere, la MEME incertitude de
    # conditionnement que (2) s'applique aux deux cotes de la comparaison (mon extreme et celui
    # d'Octave different chacun de O(sensibilite) de la 'vraie' valeur) -- la couverture y est
    # exigee a 100 x sensibilite pres, marge sans consequence pour la CFL (facteur ~0.4-0.5).
    for i in range(states.shape[0]):
        slack = (100.0 * sens) if i == NEAR_DEGENERATE else 1e-6
        mws_x = m._m.max_wave_speed(U[:, i:i + 1], {}, 0)
        mws_y = m._m.max_wave_speed(U[:, i:i + 1], {}, 1)
        need_x = max(abs(vp[i, 0]), abs(vp[i, 1]))
        need_y = max(abs(vp[i, 2]), abs(vp[i, 3]))
        assert need_x <= mws_x * (1 + slack) + 1e-12, "etat %d : borne x insuffisante" % i
        assert need_y <= mws_y * (1 + slack) + 1e-12, "etat %d : borne y insuffisante" % i
    print("(3) max_wave_speed exact >= vraies vitesses sur les 10 etats (faille de la borne "
          "bring-up fermee ; quasi-degenere a l'incertitude de conditionnement pres) -- OK")

    print("hyqmom15/run_waves : OK")


if __name__ == "__main__":
    main()
