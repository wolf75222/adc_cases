#!/usr/bin/env python3
"""Projection de realisabilite relaxation15 : port Python vs Octave + crossing Ma = 20.

Verites de reference et perimetre :
 (1) golden : relaxation.relax15 == relaxation15.m EXECUTE (Octave sur RIEMOM2D,
     golden_relax_gen.m) sur 12 etats couvrant les 5 branches (0 identite, 1 clamp s30/s03,
     2 bord univarie, 3 clamp s11, 4 projection collision15), rtol 1e-12 + atol echelle ;
     la COUVERTURE est assertee (chaque code de branche present dans le jeu).
 (2) branche 0 = identite A L'ARRONDI du round-trip M -> C -> S -> C -> M pres (rtol 1e-12 ;
     le MATLAB reconstruit AUSSI dans ce cas, verifie sous Octave) ; invariants structurels
     sur tout le jeu : M00, M10, M01 (masse, impulsion) et C20, C02 (la relaxation reecrit
     les moments standardises d'ordre >= 3 et s11, a m00/u/v/c20/c02 FIGES -- M11 et les
     ordres 3-4 peuvent bouger, c'est son role). NB : relaxation15 n'est PAS idempotente
     (verifie sous Octave : re-relaxer relaxe encore -- c'est une RELAXATION vers une cible,
     pas un projecteur) ; aucune assertion d'idempotence, fidele au comportement MATLAB.
 (3) crossing Ma = 20, HLL exact SANS gardes (fidele au MATLAB : flagrelax = 1 est la seule
     protection), 50 pas, projection PAR CELLULE entre pas (relax_field). Le verrou « Ma = 20
     exige relaxation15 » ne se manifeste PAS en NaN chez nous (mesure : le run nu reste fini
     500 pas en Rusanov, 300 en HLL -- nos schemas plus diffusifs que le MATLAB encaissent) :
     le critere executable est la REALISABILITE, lambda_min(p2p2) par cellule.
     Calibre (n = 32, 50 pas, post-pas sans re-projection) : projete lambda_min ~ -1.1 et
     ~13 % de cellules < -1e-9 (les violations d'UN pas, bornees) ; nu lambda_min ~ -12.8 et
     ~52 % (accumulation). Asserts a marges : projete >= -5 / < 30 %, nu <= -5 / > 35 % ;
     et l'etat re-projete final est realisable partout (>= -1e-6). + fini, M00 > 0, masse.

Application par macro-pas via System.get_state/set_state (round-trip bit-stable, cf. tests
checkpoint d'ADC-85). Chemin compile = ADC-177 (backlog).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in (os.path.dirname(HERE),):
    if d not in sys.path:
        sys.path.insert(0, d)

import adc  # noqa: E402
from model import MOMENT_NAMES, build_moment_model, crossing_state  # noqa: E402
from relaxation import make_corner_eigs, relax15, relax_field  # noqa: E402


def check_golden():
    g = os.path.join(HERE, "golden")
    inm = np.loadtxt(os.path.join(g, "golden_relax_in.csv"), delimiter=",")
    outm = np.loadtxt(os.path.join(g, "golden_relax_out.csv"), delimiter=",")
    meta = np.loadtxt(os.path.join(g, "golden_relax_meta.csv"), delimiter=",")
    branches = sorted(set(int(b) for b in meta[:, 2]))
    assert branches == [0, 1, 2, 3, 4], ("couverture des branches incomplete : %s "
                                         "(regenerer golden_relax_gen.m)" % branches)
    fn = make_corner_eigs()
    worst = 0.0
    for t in range(inm.shape[0]):
        lamin, ma = float(meta[t, 0]), float(meta[t, 1])
        got = relax15(inm[t], lamin, ma, corner_eigs=fn)
        scale = np.maximum(np.abs(outm[t]), 1e-13)
        err = float((np.abs(got - outm[t]) / scale).max())
        worst = max(worst, err)
        np.testing.assert_allclose(
            got, outm[t], rtol=1e-12, atol=1e-13 * float(np.abs(outm[t]).max()),
            err_msg="relax15 != relaxation15.m (etat %d, branche %d)" % (t, int(meta[t, 2])))
    print("(1) port == Octave sur %d etats, 5 branches couvertes, pire err rel %.2e -- OK"
          % (inm.shape[0], worst))

    nb_id = 0
    for t in range(inm.shape[0]):
        lamin, ma = float(meta[t, 0]), float(meta[t, 1])
        out = relax15(inm[t], lamin, ma, corner_eigs=fn)
        if int(meta[t, 2]) == 0:
            np.testing.assert_allclose(
                out, inm[t], rtol=1e-12, atol=1e-14 * float(np.abs(inm[t]).max()),
                err_msg=("branche 0 : identite attendue a l'arrondi du round-trip "
                         "M->C->S->C->M pres (etat %d)" % t))
            nb_id += 1
        # invariants : masse, quantite de mouvement, covariances (la relaxation n'agit que
        # sur les moments standardises d'ordre >= 3 et s11/s22 ; m00, u, v, C20, C02 figes)
        for k, nm in ((0, "M00"), (1, "M10"), (5, "M01")):
            assert abs(out[k] - inm[t][k]) <= 1e-13 * max(1.0, abs(inm[t][k])), \
                "%s non conserve (etat %d)" % (nm, t)
        for (k2, k1) in ((2, 1), (9, 5)):  # C20 = M20/M00 - u^2, C02 sym.
            cin = inm[t][k2] / inm[t][0] - (inm[t][k1] / inm[t][0]) ** 2
            cout = out[k2] / out[0] - (out[k1] / out[0]) ** 2
            assert abs(cout - cin) <= 1e-11 * max(1.0, abs(cin)), \
                "covariance non conservee (etat %d)" % t
    print("(2) branche 0 = identite au round-trip pres (%d etats) ; masse / impulsion / covariances"
          " conservees sur les 12 -- OK" % nb_id)


def lam_min_field(U):
    """lambda_min(p2p2) par cellule : la mesure de realisabilite du jeu de moments."""
    from relaxation import m2cs4, p2p2_2d
    nn = U.shape[1]
    out = np.empty((nn, U.shape[2]))
    for j in range(nn):
        for i in range(U.shape[2]):
            _, S = m2cs4(U[:, j, i])
            out[j, i] = np.sort(np.real(np.linalg.eigvals(p2p2_2d(
                S[(0, 3)], S[(0, 4)], S[(1, 1)], S[(1, 2)], S[(1, 3)],
                S[(2, 1)], S[(2, 2)], S[(3, 0)], S[(3, 1)], S[(4, 0)]))))[0]
    return out


def check_crossing_ma20():
    from adc_cases.common.native import adc_include
    from adc_cases.common.io import case_output_dir

    n, ma, nsteps, dt = 32, 20.0, 50, 2e-4
    # AUCUNE garde (robust=False) + HLL exact : fidele au MATLAB, ou flagrelax = 1 est la
    # seule protection. La projection est appliquee AVANT chaque pas (etat de depart
    # realisable), comme le MATLAB.
    m = build_moment_model(name="hyqmom15_ma20ex", robust=False, exact_speeds=True)
    compiled = m.compile(os.path.join(case_output_dir("hyqmom15"), "hyqmom15_ma20ex.so"),
                         adc_include(), backend="aot")
    fn = make_corner_eigs()

    def run(project):
        sim = adc.System(n=n, L=1.0, periodic=True)
        sim.add_equation("mom", model=compiled,
                         spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                         time=adc.Explicit())
        sim.set_state("mom", crossing_state(n, ma))
        with np.errstate(all="ignore"):
            for _ in range(nsteps):
                if project:
                    sim.set_state("mom", relax_field(np.array(sim.get_state("mom")),
                                                     1e-12, ma, corner_eigs=fn))
                sim.step(dt)
        return np.array(sim.get_state("mom"))

    U = run(project=True)
    assert np.all(np.isfinite(U)), "Ma=20 + relaxation : etat non fini"
    m00 = U[MOMENT_NAMES.index("M00")]
    assert np.all(m00 > 0), "Ma=20 + relaxation : M00 <= 0"
    mass0 = float(crossing_state(n, ma)[0].sum())
    drift = abs(float(m00.sum()) - mass0) / mass0
    assert drift < 1e-12, "derive de masse %.2e" % drift

    lp = lam_min_field(U)  # post-pas : les violations fraiches d'UN pas seulement
    Uraw = run(project=False)
    lr = lam_min_field(Uraw) if np.all(np.isfinite(Uraw)) else None

    fp = float(np.mean(lp < -1e-9))
    assert lp.min() > -5.0 and fp < 0.30, (
        "projete : violations post-pas hors fenetre calibree (lam_min %.2e, frac %.1f %%)"
        % (lp.min(), 100 * fp))
    if lr is None:
        print("(3) nu : NaN avant %d pas (contraste maximal)" % nsteps)
    else:
        fr = float(np.mean(lr < -1e-9))
        assert lr.min() < -5.0 and fr > 0.35, (
            "le contraste attendu a disparu : run nu lam_min %.2e, frac %.1f %% "
            "(re-calibrer ou re-evaluer le verrou)" % (lr.min(), 100 * fr))
    Ufinal = relax_field(U, 1e-12, ma, corner_eigs=fn)
    lf = lam_min_field(Ufinal)
    assert lf.min() > -1e-6, "etat re-projete non realisable (lam_min %.2e)" % lf.min()

    raw_txt = ("NaN avant la fin" if lr is None else
               "lam_min %.2f, %.0f %% (accumulation)" % (lr.min(), 100 * float(np.mean(lr < -1e-9))))
    print("(3) crossing Ma=20 HLL exact, %d pas : projete sain (masse %.1e), violations "
          "post-pas bornees (lam_min %.2f, %.0f %% de cellules) ; nu : %s ; etat "
          "re-projete realisable partout -- OK" % (nsteps, drift, lp.min(), 100 * fp, raw_txt))


def main():
    check_golden()
    check_crossing_ma20()
    print("hyqmom15/run_relaxation : OK")


if __name__ == "__main__":
    main()
