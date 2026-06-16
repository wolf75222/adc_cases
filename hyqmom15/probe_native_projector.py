#!/usr/bin/env python3
"""Sonde de faisabilite : relaxation15 en projecteur natif DSL (ADC-275).

Ce script N'EST PAS un projecteur de production. Il produit les preuves reproductibles de la
note notes/native_projector_feasibility.md : pourquoi relaxation15 ne s'emet PAS comme
projection DSL (m.projection d'ADC-177) en l'etat, et ce qu'il manque cote ADC-177/DSL.

Trois sections, sur les goldens de realisabilite (golden/golden_relax_*.csv, 12 etats,
branches 0-4) :

 (1) IDEMPOTENCE : la projection DSL exige P(P(U)) == P(U). On mesure ||relax(relax(U)) -
     relax(U)|| par etat. relax15 N'EST PAS idempotente (relaxation vers une cible) : ecart
     relatif jusqu'a ~6e1. -> blocage 1 (contrat de m.projection viole).
 (2) PREDICATS SPECTRAUX : le test des valeurs propres complexes (blocs d'ordre 3 du jacobien a
     l'etat standardise) et les portes de collision15_anisotropic (lambda_min(p2p2) <= lamin,
     det(p2[0:2,0:2]) < 0) tirent sur les goldens. Ils exigent numpy.linalg.eigvals / det, qui
     n'ont AUCUN noeud Expr dans la DSL. -> blocage 2 (primitive spectrale absente).
 (3) CLAMPS SANS BRANCHE : le sous-ensemble exprimable (clamps s30/s03 preservant H20, planchers
     H20/H02, clamp s11) en max/min/abs/sign. Idempotent partout, mais ne reproduit relax15 que
     sur la branche 0 (identite) : c'est un AUTRE operateur, pas une approximation de relax15.

Lancement (numpy seul ; la section 2 a besoin du modele DSL pour les valeurs propres de coin,
auto-sautee si l'extension adc n'est pas construite) :

    PYTHONPATH=<adc_cpp>/python python hyqmom15/probe_native_projector.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from relaxation import m2cs4, p2p2_2d, relax15  # noqa: E402


# --- helpers sans branche : exactement ce que abs_/sign donnent dans la DSL -------------------

def _mx(a, b):
    return 0.5 * (a + b + abs(a - b))   # max(a, b) -> (a + b + |a - b|) / 2


def _mn(a, b):
    return 0.5 * (a + b - abs(a - b))   # min(a, b) -> (a + b - |a - b|) / 2


def clamp_proj(M4, Ma):
    """Projection de realisabilite SANS branche, SANS spectre (s30/s03 + H20/H02 + s11).

    Mappe directement sur l'algebre de m.projection (Add/Sub/Mul/Div/Pow/Sqrt/Abs/Sign).
    Idempotente. Ce n'est PAS relax15 : ni collision, ni test de valeurs propres complexes."""
    M4 = np.asarray(M4, dtype=float)
    m00 = M4[0]
    u, v = M4[1] / m00, M4[5] / m00
    C, S = m2cs4(M4)
    c20, c02 = C[(2, 0)], C[(0, 2)]
    s30, s03 = S[(3, 0)], S[(0, 3)]
    s40, s04 = S[(4, 0)], S[(0, 4)]
    s11 = S[(1, 1)]
    small = 1.0e-6
    s3m = 4.0 + Ma / 2.0

    # clamp |s30| <= s3m en preservant H20 = s40 - s30^2 - 1 (s40 <- H20 + s30c^2 + 1).
    s30c = np.sign(s30) * _mn(abs(s30), s3m)
    s40 = s40 + (s30c ** 2 - s30 ** 2)
    s30 = s30c
    s03c = np.sign(s03) * _mn(abs(s03), s3m)
    s04 = s04 + (s03c ** 2 - s03 ** 2)
    s03 = s03c

    # plancher H20/H02 : porte = 1 si min(H20, H02) < small, sinon 0 (via sign, sans if).
    h20 = s40 - s30 ** 2 - 1.0
    h02 = s04 - s03 ** 2 - 1.0
    gate = 0.5 * (1.0 - np.sign(_mn(h20, h02) - small))
    s40 = (1.0 - gate) * s40 + gate * (s30 ** 2 + 1.0 + small)
    s04 = (1.0 - gate) * s04 + gate * (s03 ** 2 + 1.0 + small)
    s11 = (1.0 - gate) * s11 + gate * np.sign(s11)

    # de-standardisation et retour aux moments bruts (s21/s12/s22/s13/s31 inchanges ici).
    sx, sy = np.sqrt(c20), np.sqrt(c02)
    from relaxation import cs4_to_m4
    Cn = {(2, 0): c20, (1, 1): s11 * sx * sy, (0, 2): c02,
          (3, 0): s30 * sx ** 3, (2, 1): S[(2, 1)] * sx ** 2 * sy,
          (1, 2): S[(1, 2)] * sx * sy ** 2, (0, 3): s03 * sy ** 3,
          (4, 0): s40 * sx ** 4, (3, 1): S[(3, 1)] * sx ** 3 * sy,
          (2, 2): S[(2, 2)] * sx ** 2 * sy ** 2, (1, 3): S[(1, 3)] * sx * sy ** 3,
          (0, 4): s04 * sy ** 4}
    return cs4_to_m4(m00, u, v, Cn)


def _load_goldens():
    g = os.path.join(HERE, "golden")
    inm = np.loadtxt(os.path.join(g, "golden_relax_in.csv"), delimiter=",")
    outm = np.loadtxt(os.path.join(g, "golden_relax_out.csv"), delimiter=",")
    meta = np.loadtxt(os.path.join(g, "golden_relax_meta.csv"), delimiter=",")
    return inm, outm, meta


def _corner_eigs():
    """make_corner_eigs() si l'extension adc se charge, sinon None (section 2 sautee)."""
    try:
        from relaxation import make_corner_eigs
        return make_corner_eigs()
    except Exception as ex:  # noqa: BLE001
        print("  (section 2 sautee : modele DSL indisponible -- %s)" % ex)
        return None


def section_idempotence(inm, meta, fn):
    print("== (1) IDEMPOTENCE : P(P(U)) vs P(U) -- m.projection exige l'egalite ==")
    worst = 0.0
    for t in range(inm.shape[0]):
        lamin, ma, br = float(meta[t, 0]), float(meta[t, 1]), int(meta[t, 2])
        o1 = relax15(inm[t], lamin, ma, corner_eigs=fn)
        o2 = relax15(o1, lamin, ma, corner_eigs=fn)
        rel = float(np.max(np.abs(o2 - o1) / np.maximum(np.abs(o1), 1e-13)))
        worst = max(worst, rel)
        print("  etat %2d branche %d  ecart-idemp(rel) %.3e" % (t, br, rel))
    verdict = "" if worst < 1e-10 else "PAS "
    print("  PIRE ecart idempotence (rel) : %.3e  ->  relax15 %sIDEMPOTENTE" % (worst, verdict))
    return worst


def section_spectral(inm, meta, fn):
    print("== (2) PREDICATS SPECTRAUX : eigvals/det necessaires (aucun noeud Expr DSL) ==")
    n_eig = 0
    n_collide = 0
    for t in range(inm.shape[0]):
        lamin, br = float(meta[t, 0]), int(meta[t, 2])
        _, S = m2cs4(inm[t])
        sd = {"s03": S[(0, 3)], "s04": S[(0, 4)], "s11": S[(1, 1)], "s12": S[(1, 2)],
              "s13": S[(1, 3)], "s21": S[(2, 1)], "s22": S[(2, 2)], "s30": S[(3, 0)],
              "s31": S[(3, 1)], "s40": S[(4, 0)]}
        lamx, lamy = fn(sd)
        tol = 1e-9
        cx = bool(np.any(np.abs(np.imag(lamx)) > tol * np.maximum(1.0, np.abs(lamx))))
        cy = bool(np.any(np.abs(np.imag(lamy)) > tol * np.maximum(1.0, np.abs(lamy))))
        p2 = p2p2_2d(sd["s03"], sd["s04"], sd["s11"], sd["s12"], sd["s13"], sd["s21"],
                     sd["s22"], sd["s30"], sd["s31"], sd["s40"])
        lam0 = np.sort(np.real(np.linalg.eigvals(p2)))[0]
        collide = (max(0.0, lam0) <= lamin)
        n_eig += int(cx or cy)
        n_collide += int(collide and br == 4)
        print("  etat %2d br%d : val-propres-complexes=%s  lam0(p2p2)=%+.3e  collision-gate=%s"
              % (t, br, (cx or cy), lam0, collide))
    print("  branche valeurs propres complexes : %d/12 etats ; gate collision actif : %d etats br4"
          % (n_eig, n_collide))
    return n_eig, n_collide


def section_clamps(inm, meta, fn):
    print("== (3) CLAMPS SANS BRANCHE : exprimables, idempotents, MAIS != relax15 ==")
    worst_idemp = 0.0
    for t in range(inm.shape[0]):
        lamin, ma, br = float(meta[t, 0]), float(meta[t, 1]), int(meta[t, 2])
        o = clamp_proj(inm[t], ma)
        o2 = clamp_proj(o, ma)
        idemp = float(np.max(np.abs(o2 - o) / np.maximum(np.abs(o), 1e-13)))
        worst_idemp = max(worst_idemp, idemp)
        tag = ""
        if fn is not None:
            r = relax15(inm[t], lamin, ma, corner_eigs=fn)
            rel = float(np.max(np.abs(o - r) / np.maximum(np.abs(r), 1e-13)))
            tag = "  vs-relax15 rel %.2e %s" % (rel, "(identique)" if rel < 1e-9 else "")
        print("  etat %2d br%d  clamp-idemp %.2e%s" % (t, br, idemp, tag))
    print("  clamp-only idempotent partout (pire %.2e) ; ne reproduit relax15 que sur la branche 0"
          % worst_idemp)
    return worst_idemp


def main():
    inm, outm, meta = _load_goldens()
    print("goldens : %d etats, branches %s\n"
          % (inm.shape[0], sorted(set(int(b) for b in meta[:, 2]))))
    fn = _corner_eigs()  # None si le modele DSL ne se construit pas (extension adc absente)
    if fn is not None:
        # relax15 a besoin des valeurs propres de coin : sections 1 et 2 ne tournent qu'avec fn.
        section_idempotence(inm, meta, fn)
        print()
        section_spectral(inm, meta, fn)
        print()
    else:
        print("== (1) et (2) sautees : relax15 et le test des valeurs propres exigent le modele "
              "DSL ==\n")
    # section 3 (clamps) n'a pas besoin du spectre : tourne toujours (idempotence du clamp seul).
    section_clamps(inm, meta, fn)
    print("\nprobe_native_projector : faisabilite documentee "
          "(cf. notes/native_projector_feasibility.md)")


if __name__ == "__main__":
    main()
