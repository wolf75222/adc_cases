#!/usr/bin/env python3
"""Cas hyqmom15 : modele 2D a 15 moments (HyQMOM), flux valide vs RIEMOM2D.

Modele ecrit en formules (fermeture HyQMOM) dont le flux est valide contre le
code MATLAB de reference (RIEMOM2D).

Pourquoi ce cas
---------------
Premiere brique de l'integration HyQMOM  : le vecteur d'etat porte les 15 moments
cartesiens M00..M04, le flux physique reconstruit localement les 6 moments d'ordre 5 manquants
(M50, M41, M32, M23, M14, M05) par la fermeture HyQMOM (M -> C -> S -> fermeture -> C5 -> M5),
le tout en expressions DSL compilees une fois (aucun callback Python par cellule). Le coeur
adc_cpp n'apprend rien de "HyQMOM" : le modele est un bloc hyperbolique ordinaire.

Validation (trois oracles independants du pipeline DSL)
--------------------------------------------------------
  (1) golden MATLAB : eval_flux reproduit Flux_closure15_2D.m (execute par Octave sur le depot
      RIEMOM2D, goldens commites dans golden/) sur 10 etats couvrant maxwelliennes (repos,
      derive, correlee, haut Mach ~ Ma 20), melanges discrets asymetriques, etat quasi-degenere
      (C20 ~ 1e-6) et etat fortement anisotrope -- rtol 1e-12 ;
  (2) oracle gaussien exact : sur les etats gaussiens, les 6 entrees de flux d'ordre 5 egalent
      les moments bruts exacts d'Isserlis (la fermeture HyQMOM est exacte sur une gaussienne :
      les moments standardises d'ordre 5 retournes sont nuls) -- rtol 1e-12, oracle calcule par
      binome SANS passer par le pipeline ;
  (3) structure : les 20 entrees de recopie (ordre <= 4) sont BIT-identiques aux composantes
      de U correspondantes (le DSL les assemble comme references directes, sans le round-trip
      C5toM5 du MATLAB, algebriquement identique).
  (4) compilation : le modele compile en backend AOT (et production si la plateforme le lie),
      check_model passe sur les etats realisables ; cout de compilation AFFICHE, sans assert
      mural (un budget d'horloge en CI serait un flake) ;
  (5) contraste robust : sur un etat degenere (C20 = 0), le flux bit_match DIVERGE (fidele au
      MATLAB sans gardes) et le flux robust reste fini ;
  (6) borne de vitesse confrontee a golden_vp.csv (vraies vitesses, eigenvalues15_2D flagsym=1) :
      les gaussiennes s'etendent EXACTEMENT a u +- sqrt(6)*sqrt(C) (couvertes par k = 3) et au
      moins un melange asymetrique DEPASSE k*sqrt(C) -- le danger de la borne bring-up est
      DEMONTRE, pas seulement documente.
  (7) parite des conditions initiales correlees : gaussian_state == InitializeM4_15.m (golden
      golden_crossing.csv, Octave sur RIEMOM2D) pour r = 0, 0.5, -0.5 et un etat anisotrope
      C20 != C02 -- rtol 1e-12 ; plus un controle des zones de crossing_state (fond, anti-
      diagonale, jets haut/bas) qui valide aussi que r != 0 ne leve plus d'exception.

Ne prouve pas : la stabilite d'une evolution temporelle (drivers = ), les vitesses
d'onde exactes pour HLL,
le mode robust au-dela de la finitude (gardes hors MATLAB, qui n'en a aucune).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import (  # noqa: E402
    GAUSSIAN_PARAMS,
    K_SPEED,
    MOMENT_NAMES,
    IDX,
    build_moment_model,
    crossing_state,
    gaussian_raw_moment,
    gaussian_state,
    mixture_state,
)

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

# Indices des 6 entrees de fermeture dans Fx et Fy ; le reste = recopies de U.
CLOSURE_FX = {
    4: (5, 0),
    8: (4, 1),
    11: (3, 2),
    13: (2, 3),
    14: (1, 4),
}  # k -> (p, q) de M_pq
CLOSURE_FY = {4: (4, 1), 8: (3, 2), 11: (2, 3), 13: (1, 4), 14: (0, 5)}
PASSTHROUGH_FX = {
    0: "M10",
    1: "M20",
    2: "M30",
    3: "M40",
    5: "M11",
    6: "M21",
    7: "M31",
    9: "M12",
    10: "M22",
    12: "M13",
}
PASSTHROUGH_FY = {
    0: "M01",
    1: "M11",
    2: "M21",
    3: "M31",
    5: "M02",
    6: "M12",
    7: "M22",
    9: "M03",
    10: "M13",
    12: "M04",
}

# rtol par-etat du golden de flux (check_matlab_golden). Mesure (eval_flux vs golden, atol
# proportionnel a l'echelle) : l'ecart requis vaut <= 1.1e-16 sur 9 etats sur 10 et 6.4e-13
# sur le seul etat 8 (quasi-degenere C20 ~ 1e-6 : la standardisation divise par sqrt(C20),
# annulation catastrophique). Un rtol GLOBAL 1e-12 calibre sur l'etat 8 laisse une regression
# x80 invisible sur les 9 autres. Seuils :
#   - defaut 1e-14 : ~90x de marge sur l'ecart mesure des etats sains (<= 1.1e-16), serre assez
#     pour voir une derive x80 que le 1e-12 global masquait ;
#   - etat 8 : 1e-12 (ecart mesure 6.4e-13, marge 1.6x) -- LACHE et DOCUMENTE, la borne du
#     conditionnement de l'etat lui-meme, pas du schema.
RTOL_GOLDEN_DEFAULT = 1e-14
RTOL_GOLDEN_LOOSE = {8: 1e-12}


def load_goldens() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge les etats figes et les goldens MATLAB commites.

    Provenance : golden_gen.m, Octave sur RIEMOM2D ; voir README. Verifie au
    passage l'ANTI-DERIVE : les 4 premiers etats du CSV sont exactement
    gaussian_state(GAUSSIAN_PARAMS[i]) -- si gen_states.py et les goldens sont
    regeneres sans mettre a jour model.GAUSSIAN_PARAMS (ou inversement), on
    echoue ICI au lieu de laisser l'oracle d'Isserlis valider silencieusement
    d'autres etats que ceux du golden.

    Returns:
        (states, fx, fy, vp), tableaux numpy charges depuis golden/*.csv.
    """
    g = os.path.join(HERE, "golden")
    states = np.loadtxt(os.path.join(g, "golden_states.csv"), delimiter=",")
    fx = np.loadtxt(os.path.join(g, "golden_fx.csv"), delimiter=",")
    fy = np.loadtxt(os.path.join(g, "golden_fy.csv"), delimiter=",")
    vp = np.loadtxt(os.path.join(g, "golden_vp.csv"), delimiter=",")
    for i, prm in enumerate(GAUSSIAN_PARAMS):
        np.testing.assert_allclose(
            states[i],
            gaussian_state(*prm),
            rtol=1e-13,
            err_msg="derive : golden_states.csv[%d] != gaussian_state(GAUSSIAN_PARAMS[%d]) -- "
            "regenerer les goldens ET model.GAUSSIAN_PARAMS en meme temps"
            % (i, i),
        )
    return states, fx, fy, vp


def check_matlab_golden(
    m, states: np.ndarray, gold_fx: np.ndarray, gold_fy: np.ndarray
) -> None:
    """(1) eval_flux == Flux_closure15_2D.m sur chaque etat golden.

    Tolerance mixte PAR-ETAT : rtol serre (RTOL_GOLDEN_DEFAULT = 1e-14, lache et
    documente sur l'etat 8 quasi-degenere via RTOL_GOLDEN_LOOSE) + atol
    proportionnel a l'echelle des moments de l'etat (les formes algebriques
    different legerement du MATLAB -- regroupements ux/uy -- donc egalite a
    l'arrondi pres, pas au bit). Le rtol par-etat remplace l'ancien 1e-12 global,
    calibre sur le seul etat 8 : sur les 9 autres il laissait une regression x80
    invisible.
    """
    U = states.T  # (15, N)
    for d, gold in ((0, gold_fx), (1, gold_fy)):
        F = m.eval_flux(U, {}, d)  # (15, N)
        for i in range(states.shape[0]):
            scale = np.max(np.abs(states[i]))
            rtol = RTOL_GOLDEN_LOOSE.get(i, RTOL_GOLDEN_DEFAULT)
            np.testing.assert_allclose(
                F[:, i],
                gold[i],
                rtol=rtol,
                atol=1e-13 * scale,
                err_msg="flux %s, etat golden #%d (rtol %.0e)"
                % ("xy"[d], i, rtol),
            )
    print(
        "(1) golden MATLAB : 10 etats x {Fx, Fy} reproduits (rtol par-etat : %.0e, "
        "etat 8 quasi-degenere a %.0e) -- OK"
        % (RTOL_GOLDEN_DEFAULT, RTOL_GOLDEN_LOOSE[8])
    )


def check_gaussian_oracle(m) -> None:
    """(2) sur des gaussiennes, les 6 entrees de fermeture == moments d'Isserlis.

    Les moments bruts exacts d'Isserlis sont calcules par binome, jamais par le
    pipeline DSL.
    """
    for rho, ux, uy, c20, c11, c02 in GAUSSIAN_PARAMS:
        Uvec = gaussian_state(rho, ux, uy, c20, c11, c02)
        U = Uvec[:, None]
        for d, closure_map in ((0, CLOSURE_FX), (1, CLOSURE_FY)):
            F = m.eval_flux(U, {}, d)[:, 0]
            for k, (p, q) in closure_map.items():
                exact = gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q)
                scale = max(abs(exact), np.max(np.abs(Uvec)))
                assert (
                    abs(F[k] - exact) <= 1e-12 * scale
                ), "oracle gaussien : F%s[%d] = M%d%d : %r != exact %r" % (
                    "xy"[d],
                    k,
                    p,
                    q,
                    F[k],
                    exact,
                )
    print(
        "(2) oracle gaussien (Isserlis) : 4 etats x 6 moments d'ordre 5 exacts -- OK"
    )


# Moments bruts d'ordre 5 d'une gaussienne, CODES EN DUR (oracle INDEPENDANT de model.py).
# check_gaussian_oracle ci-dessus tire son oracle de gaussian_raw_moment, qui repose sur
# _gaussian_central : la MEME table sert a fabriquer l'etat gaussien ET a le verifier (angle
# mort correle -- une coquille dans _gaussian_central corromprait entree et oracle ensemble).
# Les valeurs ci-dessous sont calculees A LA MAIN par Isserlis en arithmetique rationnelle
# exacte (fractions), hors de tout chemin model.py, puis figees en litteraux. Une derive de
# _gaussian_central qui passerait check_gaussian_oracle echoue ICI.
#
# Recette (M_pq = rho * somme_ij C(p,i) C(q,j) ux^(p-i) uy^(q-j) Cent_ij ; Cent gaussien :
# Cent_00=1, Cent_20=c20, Cent_11=c11, Cent_02=c02, Cent_40=3 c20^2, Cent_31=3 c20 c11,
# Cent_22=c20 c02 + 2 c11^2, Cent_13=3 c02 c11, Cent_04=3 c02^2, tout ordre impair = 0).
# Etat A = GAUSSIAN_PARAMS[1] = (2, 1/2, -3/10, 1, 0, 2) ; etat B = GAUSSIAN_PARAMS[2] =
# (3/2, -1/5, 2/5, 1, 9/20, 1/2). Exemple M50(A) = 2*(ux^5 + 10 ux^3 c20 + 15 ux c20^2)
# = 2*(1/32 + 10/8 + 15/2) = 2*281/32 = 281/16 = 17.5625.
ISSERLIS_LITERALS = {
    (2.0, 0.5, -0.3, 1.0, 0.0, 2.0): {  # etat A (GAUSSIAN_PARAMS[1])
        (5, 0): 281.0 / 16.0,
        (4, 1): -219.0 / 80.0,
        (3, 2): 2717.0 / 400.0,
        (2, 3): -1827.0 / 400.0,
        (1, 4): 130881.0 / 10000.0,
        (0, 5): -1854243.0 / 50000.0,
    },
    (1.5, -0.2, 0.4, 1.0, 0.45, 0.5): {  # etat B (GAUSSIAN_PARAMS[2], c11 != 0)
        (5, 0): -14439.0 / 3125.0,
        (4, 1): 948.0 / 3125.0,
        (3, 2): 35919.0 / 50000.0,
        (2, 3): 7689.0 / 6250.0,
        (1, 4): 35403.0 / 25000.0,
        (0, 5): 34317.0 / 12500.0,
    },
}


def check_isserlis_literals(m) -> None:
    """(2b) spot-check de l'oracle d'Isserlis sur des LITTERAUX (ISSERLIS_LITERALS).

    Verifie sans passer par _gaussian_central : casse l'angle mort correle
    (model.py fabrique l'etat ET fournit l'oracle d'ordre 5). Les 6 moments
    d'ordre 5 sont couverts sur 2 etats (un correle).
    """
    seen = set()
    n_lit = 0
    for params, lits in ISSERLIS_LITERALS.items():
        Uvec = gaussian_state(*params)
        U = Uvec[:, None]
        for d, closure_map in ((0, CLOSURE_FX), (1, CLOSURE_FY)):
            F = m.eval_flux(U, {}, d)[:, 0]
            for k, (p, q) in closure_map.items():
                want = lits[(p, q)]
                scale = max(abs(want), np.max(np.abs(Uvec)))
                assert abs(F[k] - want) <= 1e-13 * scale, (
                    "oracle litteral : F%s[%d] = M%d%d : %r != litteral %r (params %r)"
                    % ("xy"[d], k, p, q, F[k], want, params)
                )
                seen.add((p, q))
                n_lit += 1
    assert seen == {(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)}, (
        "couverture litterale incomplete : %s (les 6 moments d'ordre 5 doivent etre verifies)"
        % sorted(seen)
    )
    print(
        "(2b) oracle litteral (Isserlis a la main, hors _gaussian_central) : %d verifications, "
        "6 moments d'ordre 5 couverts sur 2 etats -- OK" % n_lit
    )


def check_passthrough(m, states: np.ndarray) -> None:
    """(3) les recopies d'ordre <= 4 sont bit-identiques aux composantes de U."""
    U = states.T
    for d, pmap in ((0, PASSTHROUGH_FX), (1, PASSTHROUGH_FY)):
        F = m.eval_flux(U, {}, d)
        for k, name in pmap.items():
            assert np.array_equal(F[k], U[IDX[name]]), (
                "recopie F%s[%d] != U[%s] (devrait etre une reference directe)"
                % ("xy"[d], k, name)
            )
    print("(3) structure : 20 recopies bit-identiques a U -- OK")


def check_compile_and_model(m, states: np.ndarray) -> list[str]:
    """(4) check_model sur etats realisables + compilation AOT (production si liable).

    Codegen + compilation chronometres ; renvoie la liste des backends qui ont
    effectivement compile le modele.
    """
    report = m.check_model(samples=states.T, raise_on_error=True)
    print(
        "(4a) check_model : %d etats realisables, ok = %s"
        % (report["n_samples"], report["ok"])
    )

    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    include = adc_include()
    so_dir = case_output_dir("hyqmom15")
    bound = []
    for cand in ("production", "aot"):
        t0 = time.time()
        try:
            m.compile(
                os.path.join(so_dir, "hyqmom15_%s.so" % cand),
                include,
                backend=cand,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 (diagnostic : compilation/dlopen selon plateforme)
            print(
                "(4b) backend %r indisponible (%s)" % (cand, type(exc).__name__)
            )
            continue
        dt = time.time() - t0
        bound.append(cand)
        # suivi SOUPLE du cout (pas d'assert : sur un runner CI froid/charge, une compilation
        # correcte peut depasser un budget mural -- coupler pass/fail a l'horloge serait un flake).
        slow = "  [LENT : > 60 s, verifier le cache .so]" if dt > 60.0 else ""
        print("(4b) backend %r compile en %.1f s%s" % (cand, dt, slow))
    assert bound, "aucun backend DSL n'a pu compiler le modele hyqmom15"
    return bound


def check_robust_smoke() -> None:
    """(5) smoke du mode robust : bit_match diverge, robust fini sur etat degenere.

    Sur un etat DEGENERE (tous les points du melange a la meme vitesse vx =>
    C20 = 0 exactement), le flux bit_match DIVERGE (division par sqrt(0), fidele
    au MATLAB sans gardes -- CONTRASTE verifie, pas seulement affirme) tandis que
    le flux robust reste fini.
    """
    Udeg = mixture_state([0.5, 0.5], [1.0, 1.0], [-1.0, 1.0])[:, None]
    m_bit = build_moment_model(name="hyqmom15_bit")
    with np.errstate(divide="ignore", invalid="ignore"):
        Fbit = np.concatenate([m_bit.eval_flux(Udeg, {}, d) for d in (0, 1)])
    assert not np.all(
        np.isfinite(Fbit)
    ), "etat degenere : le flux bit_match devrait diverger (sinon l'etat ne teste rien)"
    m_robust = build_moment_model(name="hyqmom15_robust", robust=True)
    for d in (0, 1):
        F = m_robust.eval_flux(Udeg, {}, d)
        assert np.all(np.isfinite(F)), (
            "mode robust : flux non fini sur l'etat degenere (dir %d)" % d
        )
    print(
        "(5) smoke robust : bit_match diverge ET robust fini sur etat degenere (C20 = 0) -- OK"
    )


def check_speed_bound(m, states: np.ndarray, vp: np.ndarray) -> None:
    """(6) borne bring-up k*sqrt(C) confrontee aux vraies vitesses (golden_vp.csv).

    Confronte la borne bring-up k*sqrt(C) aux VRAIES vitesses d'onde
    (eigenvalues15_2D flagsym=1, jacobien symbolique + eig par blocs).
    (a) Etats gaussiens : l'etendue vraie vaut EXACTEMENT u +- sqrt(6)*sqrt(C)
        (verifie a 1e-9) et k = 3 la couvre.
    (b) Le danger documente est DEMONTRE : au moins un melange asymetrique du jeu
        DEPASSE la borne k*sqrt(C) -- la borne n'est pas production, le chemin
        exact l'est.
    """
    U = states.T
    over = []
    for i in range(states.shape[0]):
        M = states[i]
        ux, uy = M[1] / M[0], M[5] / M[0]
        sC20 = np.sqrt(M[2] / M[0] - ux * ux)
        sC02 = np.sqrt(M[9] / M[0] - uy * uy)
        rx = max(abs(vp[i, 0] - ux), abs(vp[i, 1] - ux)) / sC20
        ry = max(abs(vp[i, 2] - uy), abs(vp[i, 3] - uy)) / sC02
        if i < len(GAUSSIAN_PARAMS):
            np.testing.assert_allclose(
                [rx, ry],
                np.sqrt(6.0),
                rtol=1e-9,
                err_msg="etat gaussien %d : etendue != sqrt(6)*sqrt(C)" % i,
            )
        # la borne emise est max(|u|+k*sC, ...) par direction : couverte ssi ratio <= k
        if max(rx, ry) > K_SPEED:
            over.append((i, max(rx, ry)))
        mws_x = m._m.max_wave_speed(U[:, i : i + 1], {}, 0)
        mws_y = m._m.max_wave_speed(U[:, i : i + 1], {}, 1)
        if i < len(GAUSSIAN_PARAMS):
            assert max(abs(vp[i, 0]), abs(vp[i, 1])) <= mws_x + 1e-12, (
                "gaussienne %d : borne x" % i
            )
            assert max(abs(vp[i, 2]), abs(vp[i, 3])) <= mws_y + 1e-12, (
                "gaussienne %d : borne y" % i
            )
    assert over, (
        "aucun etat ne depasse k*sqrt(C) : le jeu golden ne demontre plus le danger "
        "documente de la borne bring-up (ajouter un melange plus asymetrique)"
    )
    worst = max(r for _, r in over)
    print(
        "(6) golden_vp : gaussiennes a sqrt(6)*sqrt(C) exactement (couvertes par k = %g) ; "
        "%d melange(s) DEPASSENT la borne (pire ratio %.2f) -- bring-up seulement, "
        "chemin production = jacobienne exacte -- OK"
        % (K_SPEED, len(over), worst)
    )


def check_crossing_ic_parity() -> None:
    """(7a) parite des conditions initiales correlees vs InitializeM4_15.m.

    Pour chaque ligne du golden golden_crossing.csv (parametres M00, u, v, C20,
    C11, C02 + 15 moments produits par InitializeM4_15.m, Octave sur RIEMOM2D),
    gaussian_state(formule d'Isserlis) reproduit les 15 moments. Le golden couvre
    r = 0, r = 0.5, r = -0.5 (isotrope C20 = C02 = 1, au repos puis jets du
    croisement Ma = 20) et un etat anisotrope C20 != C02 ; voir
    golden_crossing_gen.m. Tolerance d'echelle (rtol 1e-12 + atol proportionnel a
    l'amplitude des moments) : pour les jets a Ma = 20 les moments d'ordre 4
    valent ~ 4e4, donc l'ecart absolu ~ 7e-12 est a 1e-16 pres en relatif --
    l'arrondi, pas une divergence de modele.
    """
    g = os.path.join(HERE, "golden", "golden_crossing.csv")
    data = np.loadtxt(g, delimiter=",")
    n_states = 0
    for row in data:
        m00, u, v, c20, c11, c02 = row[:6]
        gold = row[6:]
        ours = gaussian_state(m00, u, v, c20, c11, c02)
        scale = np.max(np.abs(gold))
        r = c11 / np.sqrt(c20 * c02)
        np.testing.assert_allclose(
            ours,
            gold,
            rtol=1e-12,
            atol=1e-12 * scale,
            err_msg="parite IC : gaussian_state != InitializeM4_15 (r = %.2f, C20 = %.2f, "
            "C02 = %.2f)" % (r, c20, c02),
        )
        n_states += 1
    print(
        "(7a) parite IC correlees : gaussian_state == InitializeM4_15 sur %d etats "
        "(r = 0, +/-0.5, anisotrope) -- rtol 1e-12 -- OK" % n_states
    )


def check_crossing_zones() -> None:
    """(7b) zones de crossing_state : grille exacte par zone, |r| >= 1 rejete.

    Pour r = 0 et r = 0.5 (ne leve plus d'exception), la grille (15, n, n) doit
    valoir exactement gaussian_state dans chaque zone :
    - fond hors du carre central [3n/8, 5n/8) : densite rho_out, repos ;
    - anti-diagonale i + j == n - 1 dans le carre : densite rho_in, repos ;
    - au-dessus de l'anti-diagonale (i + j > n - 1) : jet (-Uc, -Uc) ;
    - en dessous (i + j < n - 1) : jet (+Uc, +Uc), Uc = ma / sqrt(2).
    Hors domaine (|r| >= 1, covariance non definie positive) doit lever
    ValueError.
    """
    n, ma = 32, 20.0
    rho_in, rho_out, T = 1.0, 1e-3, 1.0
    uc = ma / np.sqrt(2.0)
    for r in (0.0, 0.5):
        c11 = r * T
        U = crossing_state(n, ma, rho_in=rho_in, rho_out=rho_out, T=T, r=r)
        assert U.shape == (
            15,
            n,
            n,
        ), "crossing_state : forme %r (attendu (15, %d, %d))" % (U.shape, n, n)
        m_out = gaussian_state(rho_out, 0.0, 0.0, T, c11, T)
        m_mid = gaussian_state(rho_in, 0.0, 0.0, T, c11, T)
        m_top = gaussian_state(rho_in, -uc, -uc, T, c11, T)
        m_bot = gaussian_state(rho_in, uc, uc, T, c11, T)
        lo, hi = 3 * n // 8, 5 * n // 8
        # un point representatif par zone (bornes 0-based [lo, hi))
        assert np.array_equal(U[:, 0, 0], m_out), "zone fond (r = %.1f)" % r
        # anti-diagonale : i + j == n - 1 dans le carre central
        jmid = lo
        imid = n - 1 - jmid
        assert lo <= imid < hi, "point anti-diagonale hors carre (n = %d)" % n
        assert np.array_equal(U[:, jmid, imid], m_mid), (
            "zone anti-diagonale (r = %.1f)" % r
        )
        assert np.array_equal(U[:, hi - 1, hi - 1], m_top), (
            "zone jet haut (r = %.1f)" % r
        )
        assert np.array_equal(U[:, lo, lo], m_bot), (
            "zone jet bas (r = %.1f)" % r
        )
    bad = 0
    for r in (1.0, -1.0, 1.5, -2.0):
        try:
            crossing_state(n, ma, r=r)
        except ValueError:
            bad += 1
    assert bad == 4, (
        "crossing_state : |r| >= 1 doit lever ValueError (%d/4 leves)" % bad
    )
    print(
        "(7b) zones crossing_state (fond / anti-diagonale / jets) r = 0 et r = 0.5 exactes, "
        "|r| >= 1 rejete -- OK"
    )


def main() -> None:
    print(
        "=== hyqmom15 : modele 15 moments HyQMOM en formules, flux valide vs RIEMOM2D ==="
    )
    t0 = time.time()
    m = build_moment_model()
    print(
        "modele construit (%d conservatives) en %.2f s"
        % (len(MOMENT_NAMES), time.time() - t0)
    )

    states, gold_fx, gold_fy, gold_vp = load_goldens()
    check_matlab_golden(m, states, gold_fx, gold_fy)
    check_gaussian_oracle(m)
    check_isserlis_literals(m)
    check_passthrough(m, states)
    backends = check_compile_and_model(m, states)
    check_robust_smoke()
    check_speed_bound(m, states, gold_vp)
    check_crossing_ic_parity()
    check_crossing_zones()

    print("hyqmom15 : OK (backends compiles : %s)" % ", ".join(backends))


if __name__ == "__main__":
    main()
