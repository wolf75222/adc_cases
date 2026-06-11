#!/usr/bin/env python3
"""Cas "hyqmom15" : modele 2D a 15 moments (fermeture HyQMOM) ecrit en formules, flux valide
contre le code MATLAB de reference (RIEMOM2D).

Pourquoi ce cas
---------------
Premiere brique de l'integration HyQMOM (epic ADC-81) : le vecteur d'etat porte les 15 moments
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

Ne prouve pas : la stabilite d'une evolution temporelle (drivers = ADC-84/85), les vitesses
d'onde exactes pour HLL (ADC-87/88 : golden_vp.csv ne sert ici qu'a encadrer la borne bring-up),
le mode robust au-dela de la finitude (gardes hors MATLAB, qui n'en a aucune).
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import (GAUSSIAN_PARAMS, K_SPEED, MOMENT_NAMES, IDX,  # noqa: E402
                   build_moment_model, gaussian_raw_moment, gaussian_state, mixture_state)

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

# Indices des 6 entrees de fermeture dans Fx et Fy ; le reste = recopies de U.
CLOSURE_FX = {4: (5, 0), 8: (4, 1), 11: (3, 2), 13: (2, 3), 14: (1, 4)}   # k -> (p, q) de M_pq
CLOSURE_FY = {4: (4, 1), 8: (3, 2), 11: (2, 3), 13: (1, 4), 14: (0, 5)}
PASSTHROUGH_FX = {0: "M10", 1: "M20", 2: "M30", 3: "M40", 5: "M11", 6: "M21", 7: "M31",
                  9: "M12", 10: "M22", 12: "M13"}
PASSTHROUGH_FY = {0: "M01", 1: "M11", 2: "M21", 3: "M31", 5: "M02", 6: "M12", 7: "M22",
                  9: "M03", 10: "M13", 12: "M04"}



def load_goldens():
    """Charge les etats figes et les goldens MATLAB commites (provenance : golden_gen.m, Octave
    sur RIEMOM2D ; voir README). Verifie au passage l'ANTI-DERIVE : les 4 premiers etats du CSV
    sont exactement gaussian_state(GAUSSIAN_PARAMS[i]) -- si gen_states.py et les goldens sont
    regeneres sans mettre a jour model.GAUSSIAN_PARAMS (ou inversement), on echoue ICI au lieu
    de laisser l'oracle d'Isserlis valider silencieusement d'autres etats que ceux du golden."""
    g = os.path.join(HERE, "golden")
    states = np.loadtxt(os.path.join(g, "golden_states.csv"), delimiter=",")
    fx = np.loadtxt(os.path.join(g, "golden_fx.csv"), delimiter=",")
    fy = np.loadtxt(os.path.join(g, "golden_fy.csv"), delimiter=",")
    vp = np.loadtxt(os.path.join(g, "golden_vp.csv"), delimiter=",")
    for i, prm in enumerate(GAUSSIAN_PARAMS):
        np.testing.assert_allclose(
            states[i], gaussian_state(*prm), rtol=1e-13,
            err_msg="derive : golden_states.csv[%d] != gaussian_state(GAUSSIAN_PARAMS[%d]) -- "
                    "regenerer les goldens ET model.GAUSSIAN_PARAMS en meme temps" % (i, i))
    return states, fx, fy, vp


def check_matlab_golden(m, states, gold_fx, gold_fy):
    """(1) eval_flux == Flux_closure15_2D.m sur chaque etat. Tolerance mixte : rtol 1e-12 +
    atol proportionnel a l'echelle des moments de l'etat (les formes algebriques different
    legerement du MATLAB -- regroupements ux/uy -- donc egalite a l'arrondi pres, pas au bit)."""
    U = states.T  # (15, N)
    for d, gold in ((0, gold_fx), (1, gold_fy)):
        F = m.eval_flux(U, {}, d)  # (15, N)
        for i in range(states.shape[0]):
            scale = np.max(np.abs(states[i]))
            np.testing.assert_allclose(
                F[:, i], gold[i], rtol=1e-12, atol=1e-13 * scale,
                err_msg="flux %s, etat golden #%d" % ("xy"[d], i))
    print("(1) golden MATLAB : 10 etats x {Fx, Fy} reproduits (rtol 1e-12) -- OK")


def check_gaussian_oracle(m):
    """(2) sur des gaussiennes, les 6 entrees de fermeture egalent les moments bruts exacts
    d'Isserlis (calcul par binome, jamais par le pipeline)."""
    for (rho, ux, uy, c20, c11, c02) in GAUSSIAN_PARAMS:
        Uvec = gaussian_state(rho, ux, uy, c20, c11, c02)
        U = Uvec[:, None]
        for d, closure_map in ((0, CLOSURE_FX), (1, CLOSURE_FY)):
            F = m.eval_flux(U, {}, d)[:, 0]
            for k, (p, q) in closure_map.items():
                exact = gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q)
                scale = max(abs(exact), np.max(np.abs(Uvec)))
                assert abs(F[k] - exact) <= 1e-12 * scale, (
                    "oracle gaussien : F%s[%d] = M%d%d : %r != exact %r"
                    % ("xy"[d], k, p, q, F[k], exact))
    print("(2) oracle gaussien (Isserlis) : 4 etats x 6 moments d'ordre 5 exacts -- OK")


def check_passthrough(m, states):
    """(3) les recopies d'ordre <= 4 sont bit-identiques aux composantes de U."""
    U = states.T
    for d, pmap in ((0, PASSTHROUGH_FX), (1, PASSTHROUGH_FY)):
        F = m.eval_flux(U, {}, d)
        for k, name in pmap.items():
            assert np.array_equal(F[k], U[IDX[name]]), (
                "recopie F%s[%d] != U[%s] (devrait etre une reference directe)"
                % ("xy"[d], k, name))
    print("(3) structure : 20 recopies bit-identiques a U -- OK")


def check_compile_and_model(m, states):
    """(4) check_model sur etats realisables + compilation AOT (et production si liable),
    codegen + compilation chronometres."""
    report = m.check_model(samples=states.T, raise_on_error=True)
    print("(4a) check_model : %d etats realisables, ok = %s" % (report["n_samples"], report["ok"]))

    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include
    include = adc_include()
    so_dir = case_output_dir("hyqmom15")
    bound = []
    for cand in ("production", "aot"):
        t0 = time.time()
        try:
            m.compile(os.path.join(so_dir, "hyqmom15_%s.so" % cand), include, backend=cand)
        except Exception as exc:  # noqa: BLE001 (diagnostic : compilation/dlopen selon plateforme)
            print("(4b) backend %r indisponible (%s)" % (cand, type(exc).__name__))
            continue
        dt = time.time() - t0
        bound.append(cand)
        # suivi SOUPLE du cout (pas d'assert : sur un runner CI froid/charge, une compilation
        # correcte peut depasser un budget mural -- coupler pass/fail a l'horloge serait un flake).
        slow = "  [LENT : > 60 s, verifier le cache .so]" if dt > 60.0 else ""
        print("(4b) backend %r compile en %.1f s%s" % (cand, dt, slow))
    assert bound, "aucun backend DSL n'a pu compiler le modele hyqmom15"
    return bound


def check_robust_smoke():
    """Smoke du mode robust : sur un etat DEGENERE (tous les points du melange a la meme vitesse
    vx => C20 = 0 exactement), le flux bit_match DIVERGE (division par sqrt(0), fidele au MATLAB
    sans gardes -- CONTRASTE verifie, pas seulement affirme) tandis que le flux robust reste fini."""
    Udeg = mixture_state([0.5, 0.5], [1.0, 1.0], [-1.0, 1.0])[:, None]
    m_bit = build_moment_model(name="hyqmom15_bit")
    with np.errstate(divide="ignore", invalid="ignore"):
        Fbit = np.concatenate([m_bit.eval_flux(Udeg, {}, d) for d in (0, 1)])
    assert not np.all(np.isfinite(Fbit)), (
        "etat degenere : le flux bit_match devrait diverger (sinon l'etat ne teste rien)")
    m_robust = build_moment_model(name="hyqmom15_robust", robust=True)
    for d in (0, 1):
        F = m_robust.eval_flux(Udeg, {}, d)
        assert np.all(np.isfinite(F)), "mode robust : flux non fini sur l'etat degenere (dir %d)" % d
    print("(5) smoke robust : bit_match diverge ET robust fini sur etat degenere (C20 = 0) -- OK")


def check_speed_bound(m, states, vp):
    """(6) golden_vp.csv consomme : confronte la borne bring-up k*sqrt(C) aux VRAIES vitesses
    d'onde (eigenvalues15_2D flagsym=1, jacobien symbolique + eig par blocs).
    (a) Etats gaussiens : l'etendue vraie vaut EXACTEMENT u +- sqrt(6)*sqrt(C) (verifie a 1e-9)
        et k = 3 la couvre.
    (b) Le danger documente est DEMONTRE : au moins un melange asymetrique du jeu DEPASSE la
        borne k*sqrt(C) -- la borne n'est pas production, le chemin exact (ADC-87/88) l'est."""
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
            np.testing.assert_allclose([rx, ry], np.sqrt(6.0), rtol=1e-9,
                                       err_msg="etat gaussien %d : etendue != sqrt(6)*sqrt(C)" % i)
        # la borne emise est max(|u|+k*sC, ...) par direction : couverte ssi ratio <= k
        if max(rx, ry) > K_SPEED:
            over.append((i, max(rx, ry)))
        mws_x = m._m.max_wave_speed(U[:, i:i + 1], {}, 0)
        mws_y = m._m.max_wave_speed(U[:, i:i + 1], {}, 1)
        if i < len(GAUSSIAN_PARAMS):
            assert max(abs(vp[i, 0]), abs(vp[i, 1])) <= mws_x + 1e-12, "gaussienne %d : borne x" % i
            assert max(abs(vp[i, 2]), abs(vp[i, 3])) <= mws_y + 1e-12, "gaussienne %d : borne y" % i
    assert over, ("aucun etat ne depasse k*sqrt(C) : le jeu golden ne demontre plus le danger "
                  "documente de la borne bring-up (ajouter un melange plus asymetrique)")
    worst = max(r for _, r in over)
    print("(6) golden_vp : gaussiennes a sqrt(6)*sqrt(C) exactement (couvertes par k = %g) ; "
          "%d melange(s) DEPASSENT la borne (pire ratio %.2f) -- bring-up seulement, "
          "chemin production = jacobienne exacte -- OK" % (K_SPEED, len(over), worst))


def main():
    print("=== hyqmom15 : modele 15 moments HyQMOM en formules, flux valide vs RIEMOM2D ===")
    t0 = time.time()
    m = build_moment_model()
    print("modele construit (%d conservatives) en %.2f s" % (len(MOMENT_NAMES), time.time() - t0))

    states, gold_fx, gold_fy, gold_vp = load_goldens()
    check_matlab_golden(m, states, gold_fx, gold_fy)
    check_gaussian_oracle(m)
    check_passthrough(m, states)
    backends = check_compile_and_model(m, states)
    check_robust_smoke()
    check_speed_bound(m, states, gold_vp)

    print("hyqmom15 : OK (backends compiles : %s)" % ", ".join(backends))


if __name__ == "__main__":
    main()
