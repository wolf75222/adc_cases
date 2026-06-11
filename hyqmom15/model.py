"""Modele 2D a 15 moments avec fermeture HyQMOM, ecrit en formules (adc.dsl.Model).

Structure semi-generale decidee en reunion (11/06/2026) : un builder de modele de moments
`build_moment_model(closure=...)` ou la fermeture est un callable Python qui recoit les moments
standardises et retourne les 6 moments d'ordre 5 fermes. La fermeture HyQMOM (`hyqmom_closure`,
transcription litterale de closureS5.m, forme polynomiale) est l'implementation fournie ; une
autre fermeture du meme contrat s'y branche sans toucher ni au builder ni au coeur adc_cpp.

Etat conservatif (ordre du document maths / MATLAB RIEMOM2D, 0-based) :

    U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
         0    1    2    3    4    5    6    7    8    9    10   11   12   13   14

Flux physiques (M50, M41, M32, M23, M14, M05 reconstruits par la fermeture) :

    Fx = [M10 M20 M30 M40 M50 M11 M21 M31 M41 M12 M22 M32 M13 M23 M14]
    Fy = [M01 M11 M21 M31 M41 M02 M12 M22 M32 M03 M13 M23 M04 M14 M05]

20 des 30 entrees sont des recopies directes de U (ordre <= 4) ; seules les 6 reconstructions
d'ordre 5 portent la fermeture. Le pipeline M -> C -> S -> fermeture -> C5 -> M5 est emis en
let-bindings `m.primitive(...)` : chaque intermediaire devient une variable locale C++ nommee,
les formules aval ne referencent que des feuilles Var (codegen lineaire, pas de re-expansion).

References : RIEMOM2D/{Flux_closure15_2D.m, M2CS4_15.m, M4toC4.m, closureS5.m, C5toM5.m} ;
document maths main.pdf eq. 1.8-1.12 (Bryngelson, Fox & Laurent 2025, hal-05398171).
"""

import numpy as np

from adc import dsl

# Noms et exposants (p, q) de M_pq, dans l'ordre du vecteur d'etat.
MOMENT_NAMES = ["M00", "M10", "M20", "M30", "M40", "M01", "M11", "M21", "M31",
                "M02", "M12", "M22", "M03", "M13", "M04"]
MOMENT_PQ = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1),
             (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (0, 4)]

# Indices (0-based) dans U de chaque moment, pour l'assemblage des flux.
IDX = {name: k for k, name in enumerate(MOMENT_NAMES)}

# Borne de vitesse bring-up : |u| + K_SPEED*sqrt(C). Les vraies valeurs propres (eigenvalues15_2D
# flagsym=1, cf. golden/golden_vp.csv) s'etendent a u +- sqrt(6)*sqrt(C20) ~ +-2.449*sqrt(C20)
# pour une gaussienne (ratio verifie EXACT sur les 4 etats gaussiens du jeu golden) : k = 3 les
# couvre avec ~22 % de marge. MAIS des etats realisables asymetriques DEPASSENT k*sqrt(C) (ratio
# jusqu'a 3.29 sur les melanges du jeu golden, non borne pres de la frontiere de realisabilite) :
# run.py le DEMONTRE en consommant golden_vp.csv. Borne de demarrage Rusanov uniquement ; le
# chemin production est la jacobienne exacte (ADC-87/ADC-88).
K_SPEED = 3.0

# Parametres (rho, ux, uy, C20, C11, C02) des etats gaussiens du jeu golden : SOURCE UNIQUE,
# consommee par gen_states.py (generation des etats figes) ET run.py (oracle d'Isserlis +
# verification anti-derive contre golden_states.csv).
GAUSSIAN_PARAMS = [
    (1.0, 0.0, 0.0, 1.0, 0.0, 1.0),     # repos isotrope
    (2.0, 0.5, -0.3, 1.0, 0.0, 2.0),    # derive anisotrope
    (1.5, -0.2, 0.4, 1.0, 0.45, 0.5),   # correlee (S11 != 0)
    (1.0, 14.1, 0.0, 0.5, 0.0, 0.5),    # haut Mach (Ma ~ 20)
]


def hyqmom_closure(S):
    """Fermeture HyQMOM d'ordre 5 (closureS5.m, transcription litterale de la forme polynomiale ;
    PAS les variantes Moments5.m / S5_2D.m qui different sur S32/S23).

    @p S : dict des moments standardises S11,S30,S21,S12,S03,S40,S31,S22,S13,S04 (Expr DSL ou
    numpy : seules les operations arithmetiques sont utilisees). @return dict S50..S05."""
    s11, s30, s21, s12, s03 = S["S11"], S["S30"], S["S21"], S["S12"], S["S03"]
    s40, s31, s22, s13, s04 = S["S40"], S["S31"], S["S22"], S["S13"], S["S04"]
    return {
        "S50": s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0) / 2.0,
        "S05": s03 * (5.0 * s04 - 3.0 * s03 * s03 - 1.0) / 2.0,
        "S41": (-s30 * (8.0 * s40 - 9.0 * s30 * s30 - 4.0) * s11 / 4.0
                + (10.0 * s40 - 15.0 * s30 * s30 - 6.0) * s21 / 4.0 + 2.0 * s30 * s31),
        "S14": (-s03 * (8.0 * s04 - 9.0 * s03 * s03 - 4.0) * s11 / 4.0
                + (10.0 * s04 - 15.0 * s03 * s03 - 6.0) * s12 / 4.0 + 2.0 * s03 * s13),
        "S32": (2.0 * s40 - 3.0 * s30 * s30) * s12 / 2.0 + (3.0 * s22 - 1.0) * s30 / 2.0,
        "S23": (2.0 * s04 - 3.0 * s03 * s03) * s21 / 2.0 + (3.0 * s22 - 1.0) * s03 / 2.0,
    }


def build_moment_model(name="hyqmom15", closure=hyqmom_closure, robust=False,
                       eps_m00=1e-12, eps_c=1e-12):
    """Construit le modele DSL 15 moments avec la fermeture @p closure.

    @p robust : False (defaut) = mode bit_match, AUCUNE garde, fidele au MATLAB qui n'en a
    aucune (division par M00, sqrt(C20), sqrt(C02) inconditionnels) -- requis pour la
    validation golden. True = planchers M00/C20/C02 (max lisse via |.|) dans les formules,
    cote cas uniquement, jamais dans le coeur. @return adc.dsl.Model pret a compiler."""
    m = dsl.Model(name)
    cons = m.conservative_vars(*MOMENT_NAMES)
    U = dict(zip(MOMENT_NAMES, cons))

    def let(nm, expr):
        return m.primitive(nm, expr)

    def floor(nm, x, eps):
        # max(x, eps) = ((x + eps) + |x - eps|) / 2 : plancher lisse, exprimable dans l'AST.
        return let(nm, ((x + eps) + dsl.abs_(x - eps)) / 2.0)

    # --- vitesses moyennes (et denominateurs proteges en mode robust) ---
    M00 = floor("M00f", U["M00"], eps_m00) if robust else U["M00"]
    ux = let("ux", U["M10"] / M00)
    uy = let("uy", U["M01"] / M00)
    ux2 = let("ux2", ux * ux)
    uy2 = let("uy2", uy * uy)
    ux3 = let("ux3", ux2 * ux)
    uy3 = let("uy3", uy2 * uy)

    # --- moments centres (M4toC4.m, reecrits en ux/uy : algebriquement identiques) ---
    c20_raw = U["M20"] / M00 - ux2
    c02_raw = U["M02"] / M00 - uy2
    C20 = floor("C20", c20_raw, eps_c) if robust else let("C20", c20_raw)
    C02 = floor("C02", c02_raw, eps_c) if robust else let("C02", c02_raw)
    C11 = let("C11", U["M11"] / M00 - ux * uy)
    C30 = let("C30", U["M30"] / M00 - 3.0 * ux * (U["M20"] / M00) + 2.0 * ux3)
    C03 = let("C03", U["M03"] / M00 - 3.0 * uy * (U["M02"] / M00) + 2.0 * uy3)
    C21 = let("C21", U["M21"] / M00 - uy * (U["M20"] / M00) - 2.0 * ux * (U["M11"] / M00)
              + 2.0 * ux2 * uy)
    C12 = let("C12", U["M12"] / M00 - ux * (U["M02"] / M00) - 2.0 * uy * (U["M11"] / M00)
              + 2.0 * uy2 * ux)
    C40 = let("C40", U["M40"] / M00 - 4.0 * ux * (U["M30"] / M00)
              + 6.0 * ux2 * (U["M20"] / M00) - 3.0 * ux2 * ux2)
    C04 = let("C04", U["M04"] / M00 - 4.0 * uy * (U["M03"] / M00)
              + 6.0 * uy2 * (U["M02"] / M00) - 3.0 * uy2 * uy2)
    C31 = let("C31", U["M31"] / M00 - uy * (U["M30"] / M00) - 3.0 * ux * (U["M21"] / M00)
              + 3.0 * ux * uy * (U["M20"] / M00) + 3.0 * ux2 * (U["M11"] / M00)
              - 3.0 * ux3 * uy)
    C13 = let("C13", U["M13"] / M00 - ux * (U["M03"] / M00) - 3.0 * uy * (U["M12"] / M00)
              + 3.0 * ux * uy * (U["M02"] / M00) + 3.0 * uy2 * (U["M11"] / M00)
              - 3.0 * uy3 * ux)
    C22 = let("C22", U["M22"] / M00 - 2.0 * ux * (U["M12"] / M00) - 2.0 * uy * (U["M21"] / M00)
              + ux2 * (U["M02"] / M00) + uy2 * (U["M20"] / M00)
              + 4.0 * ux * uy * (U["M11"] / M00) - 3.0 * ux2 * uy2)

    # --- standardisation (M2CS4_15.m) : S_ij = C_ij / (C20^(i/2) C02^(j/2)) ---
    sC20 = let("sC20", dsl.sqrt(C20))
    sC02 = let("sC02", dsl.sqrt(C02))
    S = {}
    S["S11"] = let("S11", C11 / (sC20 * sC02))
    S["S30"] = let("S30", C30 / (C20 * sC20))
    S["S21"] = let("S21", C21 / (C20 * sC02))
    S["S12"] = let("S12", C12 / (C02 * sC20))
    S["S03"] = let("S03", C03 / (C02 * sC02))
    S["S40"] = let("S40", C40 / (C20 * C20))
    S["S31"] = let("S31", C31 / (C20 * sC20 * sC02))
    S["S22"] = let("S22", C22 / (C20 * C02))
    S["S13"] = let("S13", C13 / (sC20 * C02 * sC02))
    S["S04"] = let("S04", C04 / (C02 * C02))

    # --- fermeture (callable, contrat : S -> 6 moments standardises d'ordre 5) ---
    S5 = closure(S)
    S50 = let("S50", S5["S50"])
    S41 = let("S41", S5["S41"])
    S32 = let("S32", S5["S32"])
    S23 = let("S23", S5["S23"])
    S14 = let("S14", S5["S14"])
    S05 = let("S05", S5["S05"])

    # --- de-standardisation (Flux_closure15_2D.m) : C_ij = S_ij sC20^i sC02^j ---
    C50 = let("C50", S50 * C20 * C20 * sC20)
    C41 = let("C41", S41 * C20 * C20 * sC02)
    C32 = let("C32", S32 * C20 * sC20 * C02)
    C23 = let("C23", S23 * C20 * C02 * sC02)
    C14 = let("C14", S14 * sC20 * C02 * C02)
    C05 = let("C05", S05 * C02 * C02 * sC02)

    # --- moments bruts d'ordre 5 (C5toM5.m, les 6 entrees d'ordre 5 uniquement ; les entrees
    # d'ordre <= 4 du round-trip MATLAB sont algebriquement identiques a U : recopies directes) ---
    M50 = let("M50", M00 * (ux3 * ux2 + C50 + 10.0 * C20 * ux3 + 10.0 * C30 * ux2
                            + 5.0 * C40 * ux))
    M41 = let("M41", M00 * (C41 + 4.0 * C11 * ux3 + 6.0 * C21 * ux2 + 4.0 * C31 * ux
                            + C40 * uy + ux2 * ux2 * uy + 6.0 * C20 * ux2 * uy
                            + 4.0 * C30 * ux * uy))
    M32 = let("M32", M00 * (C32 + C02 * ux3 + 3.0 * C12 * ux2 + C30 * uy2 + 3.0 * C22 * ux
                            + 2.0 * C31 * uy + ux3 * uy2 + 3.0 * C20 * uy2 * ux
                            + 6.0 * C11 * ux2 * uy + 6.0 * C21 * ux * uy))
    M23 = let("M23", M00 * (C23 + C03 * ux2 + C20 * uy3 + 3.0 * C21 * uy2 + 2.0 * C13 * ux
                            + 3.0 * C22 * uy + ux2 * uy3 + 6.0 * C11 * uy2 * ux
                            + 3.0 * C02 * ux2 * uy + 6.0 * C12 * ux * uy))
    M14 = let("M14", M00 * (C14 + 4.0 * C11 * uy3 + 6.0 * C12 * uy2 + C04 * ux
                            + 4.0 * C13 * uy + uy2 * uy2 * ux + 6.0 * C02 * uy2 * ux
                            + 4.0 * C03 * ux * uy))
    M05 = let("M05", M00 * (uy3 * uy2 + C05 + 10.0 * C02 * uy3 + 10.0 * C03 * uy2
                            + 5.0 * C04 * uy))

    # --- assemblage des flux : 20 recopies de U + 6 reconstructions (M41/M32/M23/M14 partages) ---
    m.flux(
        x=[U["M10"], U["M20"], U["M30"], U["M40"], M50,
           U["M11"], U["M21"], U["M31"], M41,
           U["M12"], U["M22"], M32,
           U["M13"], M23,
           M14],
        y=[U["M01"], U["M11"], U["M21"], U["M31"], M41,
           U["M02"], U["M12"], U["M22"], M32,
           U["M03"], U["M13"], M23,
           U["M04"], M14,
           M05])

    # Borne de vitesse bring-up (Rusanov / CFL) : voir K_SPEED. NON-production.
    k = m.param("k_speed", K_SPEED)
    m.eigenvalues(x=[ux - k * sC20, ux + k * sC20],
                  y=[uy - k * sC02, uy + k * sC02])

    # Layout primitif identite (pas de reconstruction en variables primitives pour ce modele) ;
    # to_conservative est l'identite correspondante.
    m.primitive_vars(*cons)
    m.conservative_from(list(cons))

    m.check()
    return m


# ---------------------------------------------------------------------------------------------
# Generateurs d'etats REALISABLES et oracle gaussien exact (independants du pipeline ci-dessus).
# ---------------------------------------------------------------------------------------------

def _binom(n, k):
    from math import comb
    return float(comb(n, k))


def _gaussian_central(c20, c11, c02, p, q):
    """Moment centre C_pq (p+q <= 5) d'une gaussienne 2D de covariance [[c20,c11],[c11,c02]]
    (Isserlis) : 0 si p+q impair ; ordres 0/2/4 en forme fermee ; ordre 5 = 0."""
    table = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0,
             (2, 0): c20, (1, 1): c11, (0, 2): c02,
             (3, 0): 0.0, (2, 1): 0.0, (1, 2): 0.0, (0, 3): 0.0,
             (4, 0): 3.0 * c20 ** 2, (3, 1): 3.0 * c20 * c11,
             (2, 2): c20 * c02 + 2.0 * c11 ** 2,
             (1, 3): 3.0 * c02 * c11, (0, 4): 3.0 * c02 ** 2}
    if (p, q) in table:
        return table[(p, q)]
    if (p + q) == 5:
        return 0.0  # tout moment centre gaussien d'ordre impair est nul
    raise ValueError("ordre non couvert : (%d, %d)" % (p, q))


def gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q):
    """Moment brut EXACT M_pq d'une gaussienne 2D (binome sur les moments centres d'Isserlis) :
    M_pq = rho * sum_ij binom(p,i) binom(q,j) ux^(p-i) uy^(q-j) C_ij. Oracle independant du
    pipeline de fermeture (la fermeture HyQMOM est exacte sur les gaussiennes : S30=S21=...=0,
    S40=S04=3 => les 6 moments standardises d'ordre 5 retournes sont exactement nuls)."""
    tot = 0.0
    for i in range(p + 1):
        for j in range(q + 1):
            tot += (_binom(p, i) * _binom(q, j) * ux ** (p - i) * uy ** (q - j)
                    * _gaussian_central(c20, c11, c02, i, j))
    return rho * tot


def gaussian_state(rho, ux, uy, c20, c11, c02):
    """Vecteur d'etat (15,) des moments bruts exacts d'une gaussienne 2D."""
    return np.array([gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q)
                     for (p, q) in MOMENT_PQ])


def mixture_state(weights, vxs, vys):
    """Vecteur d'etat (15,) d'un melange discret f = sum_k w_k delta(v - v_k) : moments exacts
    M_pq = sum_k w_k vx_k^p vy_k^q. Toujours realisable (c'est une distribution), permet des
    etats fortement asymetriques / quasi-degeneres hors de portee des gaussiennes."""
    w = np.asarray(weights, dtype=float)
    vx = np.asarray(vxs, dtype=float)
    vy = np.asarray(vys, dtype=float)
    return np.array([np.sum(w * vx ** p * vy ** q) for (p, q) in MOMENT_PQ])
