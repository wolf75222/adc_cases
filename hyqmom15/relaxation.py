"""Projection de realisabilite relaxation15 (oracle de reference Octave).

Port de relaxation15.m, collision15_anisotropic.m et p2p2_2D.m.

Pipeline par cellule : clamps des moments standardises (|s30| <= 4 + Ma/2, hyperbolicite
H2, |s11| < 1), suppression des valeurs propres de flux complexes (blocs d'ordre 3), puis
relaxation vers une cible realisable (Z1 = u+v et Z2 = u-v independants) quand la matrice
de realisabilite p2p2 a une valeur propre < lamin. Reconstruction M <- C <- S par les
transformations binomiales ; les references sont les paires in/out produites par
golden_relax_gen.m (Octave), tolerance 1e-12.

Le test des valeurs propres complexes evalue le jacobien autodiff du modele aux etats
standardises (seuil |Im| > 1e-9*max(1, |lambda|)). Les deux branches x et y appliquent la
meme correction (s21 = s12 = 0, s22 >= 1/3) : seul compte « l'un des blocs est complexe ».

En simulation, la projection s'applique par champ entre les pas (relax_field), comme le
flagrelax = 1 du MATLAB. Ce module est l'ORACLE de reference (== Octave a 4e-14) et la source
de verite du port natif : boucle Python par cellule (copie host, eigvals par cellule), pour
validation et runs moderes seulement, PAS dans un pas de temps device/MPI. Le chemin
production est le projecteur natif compile model.build_projection, emis via m.projection (hook
generique post-pas du coeur, ADC-275 / ADC-177) ; il reproduit relax15 branche par branche
(~1e-15 sur les goldens, cf. validate_native_projector.py) et tourne dans le System sans
callback Python par cellule. relax15 reste l'oracle de validation.
"""

from __future__ import annotations

import numpy as np

from model import MOMENT_NAMES, MOMENT_PQ

_EPS = np.finfo(float).eps  # eps MATLAB


# ---------------------------------------------------------------------------------------------
# Transformations M <-> C <-> S (M2CS4_15.m / C4toM4.m, via binomiales -- memes formules que
# le generateur adc.moments, reecrites ici en numpy pur pour des SCALAIRES par cellule).
# ---------------------------------------------------------------------------------------------


def _binom(n, k):
    from math import comb

    return float(comb(n, k))


def m2cs4(M4):
    """Moments bruts -> centres C et standardises S (M2CS4_15.m).

    Args:
        M4: Moments bruts (15,) dans l'ordre MOMENT_NAMES.

    Returns:
        Le couple ``(C, S)`` ou C (centres) et S (standardises) sont des
        dicts indexes par la paire d'exposants (p, q). S ne contient que
        les ordres p + q >= 2.
    """
    M4 = np.asarray(M4, dtype=float)
    m00 = M4[0]
    mn = {pq: M4[k] / m00 for k, pq in enumerate(MOMENT_PQ)}
    u, v = mn[(1, 0)], mn[(0, 1)]
    C = {}
    for p, q in MOMENT_PQ:
        acc = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                acc += (
                    _binom(p, i)
                    * _binom(q, j)
                    * (-u) ** (p - i)
                    * (-v) ** (q - j)
                    * mn[(i, j)]
                )
        C[(p, q)] = acc
    sx, sy = np.sqrt(C[(2, 0)]), np.sqrt(C[(0, 2)])
    S = {
        pq: C[pq] / (sx ** pq[0] * sy ** pq[1])
        for pq in MOMENT_PQ
        if pq[0] + pq[1] >= 2
    }
    return C, S


def cs4_to_m4(m00, u, v, C) -> np.ndarray:
    """Centres C -> moments bruts (binomiale inverse, C4toM4.m).

    Args:
        m00: Densite (moment d'ordre 0).
        u: Vitesse moyenne selon x.
        v: Vitesse moyenne selon y.
        C: Moments centres, dict indexe par (p, q), avec C00 = 1 et
            C10 = C01 = 0 implicites.

    Returns:
        Les moments bruts (15,) dans l'ordre MOMENT_NAMES.
    """
    out = np.empty(15)
    for k, (p, q) in enumerate(MOMENT_PQ):
        acc = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                cij = (
                    1.0
                    if (i, j) == (0, 0)
                    else (0.0 if (i, j) in ((1, 0), (0, 1)) else C[(i, j)])
                )
                if cij == 0.0:
                    continue
                acc += (
                    _binom(p, i)
                    * _binom(q, j)
                    * u ** (p - i)
                    * v ** (q - j)
                    * cij
                )
        out[k] = m00 * acc
    return out


# ---------------------------------------------------------------------------------------------
# p2p2_2D.m : matrice de realisabilite 3x3 (transcription LITTERALE du code genere par leur
# toolbox symbolique ; reshape MATLAB column-major).
# ---------------------------------------------------------------------------------------------


def p2p2_2d(a03, a04, a11, a12, a13, a21, a22, a30, a31, a40) -> np.ndarray:
    """Matrice de realisabilite 3x3 (transcription litterale de p2p2_2D.m).

    Args:
        a03, a04, a11, a12, a13, a21, a22, a30, a31, a40: Moments
            standardises d'ordre 3 et 4.

    Returns:
        La matrice de realisabilite 3x3 (reshape column-major, ordre MATLAB).
    """
    t2 = a03 * a12
    t3 = a03 * a21
    t4 = a12 * a21
    t5 = a12 * a30
    t6 = a21 * a30
    t7 = a11**2
    t8 = a11**3
    t9 = a12**2
    t10 = a21**2
    t12 = a03 * a11 * a30
    t15 = -a13
    t16 = -a31
    t11 = a11 * t3
    t13 = a11 * t4
    t14 = a11 * t5
    t17 = -t3
    t18 = -t5
    t19 = a11 * t9
    t20 = a13 * t7
    t21 = a11 * t10
    t22 = a22 * t7
    t23 = a31 * t7
    t24 = -t7
    t25 = -t8
    t26 = t7 - 1.0
    t27 = -t11
    t28 = -t14
    t29 = -t19
    t30 = -t21
    t31 = -t22
    t32 = 1.0 / t26
    t33 = a22 + t12 + t13 + t17 + t18 + t26 + t31
    t34 = a11 + t2 + t4 + t15 + t20 + t25 + t27 + t29
    t35 = a11 + t4 + t6 + t16 + t23 + t25 + t28 + t30
    t36 = t32 * t33
    t38 = t32 * t34
    t39 = t32 * t35
    t37 = -t36
    flat = [
        t32 * (-a40 + t10 + t24 - a11 * t6 * 2.0 + a40 * t7 + a30**2 + 1.0),
        t39,
        t37,
        t39,
        t32 * (-a22 + t7 + t9 + t10 - t13 * 2.0 + t22 + t7 * t24),
        t38,
        t37,
        t38,
        t32 * (-a04 + t9 + t24 + a04 * t7 - a11 * t2 * 2.0 + a03**2 + 1.0),
    ]
    return np.array(flat).reshape(3, 3, order="F")


# ---------------------------------------------------------------------------------------------
# collision15_anisotropic.m : relaxation vers la cible Z1 = u+v / Z2 = u-v independants.
# ---------------------------------------------------------------------------------------------


def collision15_anisotropic(
    s03, s04, s11, s12, s13, s21, s22, s30, s31, s40, lamin
) -> np.ndarray:
    """Relaxe les moments standardises vers une cible realisable.

    Port de collision15_anisotropic.m. Cible Z1 = u+v et Z2 = u-v
    independants, declenchee quand la plus petite valeur propre de p2p2
    tombe sous lamin ; sinon les entrees sont renvoyees inchangees.

    Args:
        s03, s04, s11, s12, s13, s21, s22, s30, s31, s40: Moments
            standardises d'ordre 3 et 4.
        lamin: Seuil de valeur propre p2p2 sous lequel on relaxe.

    Returns:
        Les 10 moments standardises (s03, s04, s11, s12, s13, s21, s22,
        s30, s31, s40), projetes ou identiques a l'entree.
    """
    small = 1.0e-6
    S = np.array([s03, s04, s11, s12, s13, s21, s22, s30, s31, s40])

    lam = np.sort(
        np.real(
            np.linalg.eigvals(
                p2p2_2d(s03, s04, s11, s12, s13, s21, s22, s30, s31, s40)
            )
        )
    )
    lam0 = max(0.0, lam[0])
    if lam0 > lamin:
        return S  # pas de projection requise

    Del1 = max(1.0 - s11**2, _EPS)

    # croisement de jets (deux deltas) : ne devrait pas arriver ici
    S3 = np.sign(s30) * np.sqrt(abs(s30 * s03))
    S4 = np.sqrt(s40 * s04)
    H2 = S4 - S3**2 - 1.0
    if H2 < small:
        S4 = S3**2 + 1.0 + small
    H20 = s40 - s30**2 - 1.0
    H02 = s04 - s03**2 - 1.0
    if H20 < small or H02 < small:
        S11 = np.sign(s11)
        return np.array(
            [S3 * S11, S4, S11, S3, S4 * S11, S3 * S11, S4, S3, S4 * S11, S4]
        )

    # cible : Z1 = u + v et Z2 = u - v independants
    S11 = s11
    S30 = 3.0 * s12 / 4.0 + s30 / 4.0
    S21 = s03 / 4.0 + 3.0 * s21 / 4.0
    S12 = S30
    S03 = S21
    S40 = s04 / 8.0 + (3.0 * s22) / 4.0 + s40 / 8.0 + 3.0 * (1.0 - s11**2) / 2.0
    S31 = (s13 + s31) / 2.0
    S22 = s04 / 8.0 + (3.0 * s22) / 4.0 + s40 / 8.0 - (1.0 - s11**2) / 2.0
    S13 = S31
    S04 = S40

    # realisabilite d'ordre 3 de la cible
    if abs(S21 - S11 * S30) >= np.sqrt(Del1 * (S40 - S30**2 - 1.0)):
        S21 = S11 * S30
    elif abs(S12 - S11 * S03) >= np.sqrt(Del1 * (S04 - S03**2 - 1.0)):
        S12 = S11 * S03

    # cible non realisable -> moments CJ
    p2 = p2p2_2d(S03, S04, S11, S12, S13, S21, S22, S30, S31, S40)
    if np.linalg.det(p2[0:2, 0:2]) < 0.0 or p2[0, 0] < 0.0:
        S4 = np.sqrt(s40 * s04)
        S3 = np.sqrt(abs(s03 * s30)) * np.sign(s30)
        S11 = np.sign(S11)
        H2 = S4 - S3**2 - 1.0
        if H2 < small:
            S4 = S3**2 + 1.0 + small
        S03 = S3 * S11
        S21 = S3 * S11
        S12 = S3
        S30 = S3
        S04 = S4
        S13 = S4 * S11
        S22 = S4
        S31 = S4 * S11
        S40 = S4
        S11 = S11 * (1.0 - small)

    # s22 minimal pour det(p2p2) >= 0 (transcription LITTERALE des deux racines)
    s22_1 = (
        (3.0 * S11) / 8.0
        - S04 / 32.0
        - S13 / 8.0
        - S31 / 8.0
        - S40 / 32.0
        + (3.0 * S03 * S12) / 32.0
        - (S04 * S11) / 32.0
        + (3.0 * S03 * S21) / 32.0
        - (S11 * S13) / 8.0
        + (S03 * S30) / 32.0
        + (9.0 * S12 * S21) / 32.0
        - (S11 * S31) / 8.0
        + (3.0 * S12 * S30) / 32.0
        - (S11 * S40) / 32.0
        + (3.0 * S21 * S30) / 32.0
        + S03**2 / 64.0
        + (3.0 * S11**2) / 8.0
        + S11**3 / 8.0
        + (9.0 * S12**2) / 64.0
        + (9.0 * S21**2) / 64.0
        + S30**2 / 64.0
        + 1.0 / 8.0
    ) / ((3.0 * S11) / 16.0 + 3.0 / 16.0)
    s22_2 = -(
        (
            S13 / 8.0
            - (3.0 * S11) / 8.0
            - S04 / 32.0
            + S31 / 8.0
            - S40 / 32.0
            - (3.0 * S03 * S12) / 32.0
            + (S04 * S11) / 32.0
            + (3.0 * S03 * S21) / 32.0
            - (S11 * S13) / 8.0
            - (S03 * S30) / 32.0
            - (9.0 * S12 * S21) / 32.0
            - (S11 * S31) / 8.0
            + (3.0 * S12 * S30) / 32.0
            + (S11 * S40) / 32.0
            - (3.0 * S21 * S30) / 32.0
            + S03**2 / 64.0
            + (3.0 * S11**2) / 8.0
            - S11**3 / 8.0
            + (9.0 * S12**2) / 64.0
            + (9.0 * S21**2) / 64.0
            + S30**2 / 64.0
            + 1.0 / 8.0
        )
        / ((3.0 * S11) / 16.0 - 3.0 / 16.0)
    )
    s22max = max(s22_1, s22_2)
    S22 = max(s22max, S22) * 1.001
    S22 = min(S22, S40)
    S22 = max(S22, 1.0 / 3.0)  # valeurs propres reelles

    return np.array([S03, S04, S11, S12, S13, S21, S22, S30, S31, S40])


# ---------------------------------------------------------------------------------------------
# relaxation15.m : la projection complete par cellule.
# ---------------------------------------------------------------------------------------------


def relax15(M4, lamin, Ma, corner_eigs=None) -> np.ndarray:
    """Projection par cellule -> moments realisables hyperboliques (relaxation15.m).

    Args:
        M4: Moments bruts (15,) dans l'ordre MOMENT_NAMES.
        lamin: Seuil minimal de valeur propre p2p2 declenchant la relaxation.
        Ma: Nombre de Mach (borne le clamp |s30| <= 4 + Ma/2).
        corner_eigs: callable (s_dict) -> (lamx, lamy) renvoyant les valeurs
            propres (complexes) des blocs d'ordre 3 du jacobien de flux a
            l'etat STANDARDISE. None = construit a la demande via le modele
            DSL (lent) ; les drivers passent une version vectorisee.

    Returns:
        Les moments bruts realisables (15,) dans l'ordre MOMENT_NAMES.
    """
    M4 = np.asarray(M4, dtype=float)
    m00 = M4[0]
    u = M4[1] / m00
    v = M4[5] / m00

    C, S = m2cs4(M4)
    c20, c02 = C[(2, 0)], C[(0, 2)]
    s30, s40 = S[(3, 0)], S[(4, 0)]
    s11 = S[(1, 1)]
    s21, s31 = S[(2, 1)], S[(3, 1)]
    s12, s22 = S[(1, 2)], S[(2, 2)]
    s03, s13, s04 = S[(0, 3)], S[(1, 3)], S[(0, 4)]

    # |s30|, |s03| trop grands : clamp a S3m en preservant H2
    S3m = 4.0 + Ma / 2.0
    if abs(s30) > S3m:
        H20 = s40 - s30**2 - 1.0
        s30 = np.sign(s30) * S3m
        s40 = H20 + s30**2 + 1.0
    if abs(s03) > S3m:
        H02 = s04 - s03**2 - 1.0
        s03 = np.sign(s03) * S3m
        s04 = H02 + s03**2 + 1.0

    # moments univaries en bord de realisabilite (croisement 2D)
    small = 1.0e-6
    H20 = s40 - s30**2 - 1.0
    H02 = s04 - s03**2 - 1.0
    if H20 < small or H02 < small:
        s40 = s30**2 + 1.0 + small
        s04 = s03**2 + 1.0 + small
        s11 = np.sign(s11)

    # s11 realisable
    flagS11 = 0
    if s11 >= 1.0 - small:
        s11 = 1.0 - small
        flagS11 = 1
        S3 = np.sqrt(abs(s03 * s30)) * np.sign(s30)
        S4 = np.sqrt(abs(s04 * s40))
        H2 = S4 - S3**2 - 1.0
        if H2 <= 0.0:
            S4 = S3**2 + 1.0 + small
        s03 = S3
        s21 = S3
        s12 = S3
        s30 = S3
        s04 = S4
        s13 = S4
        s22 = S4
        s31 = S4
        s40 = S4
    elif s11 <= -1.0 + small:
        s11 = -1.0 + small
        flagS11 = 1
        S3 = np.sqrt(abs(s03 * s30)) * np.sign(s30)
        S4 = np.sqrt(abs(s04 * s40))
        H2 = S4 - S3**2 - 1.0
        if H2 <= 0.0:
            S4 = S3**2 + 1.0 + small
        s03 = -S3
        s21 = -S3
        s12 = S3
        s30 = S3
        s04 = S4
        s13 = -S4
        s22 = S4
        s31 = -S4
        s40 = S4

    # valeurs propres de flux complexes (blocs d'ordre 3 du jacobien, etat standardise) :
    # les deux branches MATLAB (x puis y) font la MEME action -> seule compte « l'un des
    # blocs est complexe ».
    sd = {
        "s03": s03,
        "s04": s04,
        "s11": s11,
        "s12": s12,
        "s13": s13,
        "s21": s21,
        "s22": s22,
        "s30": s30,
        "s31": s31,
        "s40": s40,
    }
    lamx, lamy = (
        corner_eigs if corner_eigs is not None else make_corner_eigs()
    )(sd)
    tol = 1e-9
    cx = np.any(np.abs(np.imag(lamx)) > tol * np.maximum(1.0, np.abs(lamx)))
    cy = np.any(np.abs(np.imag(lamy)) > tol * np.maximum(1.0, np.abs(lamy)))
    if cx or cy:
        s21 = 0.0
        s12 = 0.0
        s22 = max(s22, 1.0 / 3.0)

    # relaxation vers la cible (sauf si s11 a ete substitue)
    S03, S04, S11, S12, S13 = s03, s04, s11, s12, s13
    S21, S22, S30, S31, S40 = s21, s22, s30, s31, s40
    if flagS11 != 1:
        out = collision15_anisotropic(
            s03, s04, s11, s12, s13, s21, s22, s30, s31, s40, lamin
        )
        S03, S04, S11, S12, S13, S21, S22, S30, S31, S40 = out

    # de-standardisation et retour aux moments bruts
    sx, sy = np.sqrt(c20), np.sqrt(c02)
    Cn = {
        (2, 0): c20,
        (1, 1): S11 * sx * sy,
        (0, 2): c02,
        (3, 0): S30 * sx**3,
        (2, 1): S21 * sx**2 * sy,
        (1, 2): S12 * sx * sy**2,
        (0, 3): S03 * sy**3,
        (4, 0): S40 * sx**4,
        (3, 1): S31 * sx**3 * sy,
        (2, 2): S22 * sx**2 * sy**2,
        (1, 3): S13 * sx * sy**3,
        (0, 4): S04 * sy**4,
    }
    return cs4_to_m4(m00, u, v, Cn)


# ---------------------------------------------------------------------------------------------
# Valeurs propres des blocs d'ordre 3 du jacobien a l'etat standardise : via la jacobienne
# AUTODIFF du modele (m.flux_jacobian, la meme matrice que jacobian15.m a l'arrondi pres,
# validee a 1e-11 par golden_vp). Blocs : x = [12,13,14] (chaine M03/M13/M04 de Jx),
# y = [3, 8, 4] (chaine d'exposants de Jy) -- les chaines d'ordre 3 de HYQMOM_BLOCKS.
# ---------------------------------------------------------------------------------------------

_CORNER_CACHE = {}


def _standardized_state(sd) -> np.ndarray:
    """Moments bruts de l'etat standardise (M00 = 1, u = v = 0, sigma = 1) : M_pq = s_pq."""
    vals = {
        "M00": 1.0,
        "M10": 0.0,
        "M01": 0.0,
        "M20": 1.0,
        "M02": 1.0,
        "M11": sd["s11"],
        "M30": sd["s30"],
        "M40": sd["s40"],
        "M21": sd["s21"],
        "M31": sd["s31"],
        "M12": sd["s12"],
        "M22": sd["s22"],
        "M03": sd["s03"],
        "M13": sd["s13"],
        "M04": sd["s04"],
    }
    return np.array([vals[nm] for nm in MOMENT_NAMES])


def make_corner_eigs():
    """Construit le callable des valeurs propres de coin du jacobien.

    Le callable renvoye, ``(s_dict) -> (lamx, lamy)``, donne les valeurs
    propres des blocs d'ordre 3 du jacobien de flux AUTODIFF
    (m.flux_jacobian : matrice d'Expr, primitives developpees -- la meme
    matrice que jacobian15.m a l'arrondi pres). Les 2 x 9 expressions de
    coin sont extraites UNE fois, puis evaluees par Expr.eval(env)
    (numpy/floats) a chaque appel.

    Returns:
        Le callable (s_dict) -> (lamx, lamy) decrit ci-dessus.
    """
    if "fn" in _CORNER_CACHE:
        return _CORNER_CACHE["fn"]
    from model import build_moment_model

    m = build_moment_model(
        "hyqmom15_relax_jac", robust=False, exact_speeds=True
    )
    IX = [12, 13, 14]
    IY = [3, 8, 4]
    Jx = m.flux_jacobian(0)
    Jy = m.flux_jacobian(1)
    EX = [[Jx[i][j] for j in IX] for i in IX]
    EY = [[Jy[i][j] for j in IY] for i in IY]

    def _ev(e, env):
        return float(e) if isinstance(e, (int, float)) else float(e.eval(env))

    def fn(sd):
        U = _standardized_state(sd)
        env = m._m._env(
            U, {}
        )  # cons + primitives derivees (les exprs gardent des symboles)
        bx = np.array(
            [[_ev(EX[a][b], env) for b in range(3)] for a in range(3)]
        )
        by = np.array(
            [[_ev(EY[a][b], env) for b in range(3)] for a in range(3)]
        )
        return np.linalg.eigvals(bx), np.linalg.eigvals(by)

    _CORNER_CACHE["fn"] = fn
    return fn


def relax_field(U, lamin, Ma, corner_eigs=None) -> np.ndarray:
    """Projette un champ (15, ny, nx) cellule par cellule.

    Reproduit l'application par pas du MATLAB (main_pb_2Dcrossing,
    flagrelax = 1).

    Args:
        U: Champ de moments (15, ny, nx).
        lamin: Seuil minimal de valeur propre p2p2 declenchant la relaxation.
        Ma: Nombre de Mach (borne le clamp des moments standardises).
        corner_eigs: Callable des valeurs propres de coin (cf. relax15) ;
            None = construit une fois ici via make_corner_eigs.

    Returns:
        Une copie du champ avec chaque cellule projetee, meme forme que U.
    """
    fn = corner_eigs if corner_eigs is not None else make_corner_eigs()
    out = np.array(U, dtype=float, copy=True)
    ny, nx = out.shape[1], out.shape[2]
    for j in range(ny):
        for i in range(nx):
            out[:, j, i] = relax15(out[:, j, i], lamin, Ma, corner_eigs=fn)
    return out


def maxwellian_state(M4):
    """15 moments bruts de la maxwellienne locale ajustee sur les bas moments de M4.

    Miroir numpy scalaire de moments.maxwellian_moments : densite M00, moyenne (u, v) et
    covariance (C20, C11, C02) des moments centres d'ordre 2, puis Isserlis (les moments
    centres pairs gaussiens jusqu'a l'ordre 4). Reutilise gaussian_state de model.py.

    Args:
        M4: vecteur de 15 moments bruts (ordre MOMENT_NAMES).

    Returns:
        ndarray (15,) des moments bruts de la maxwellienne d'equilibre.
    """
    from model import gaussian_state

    m00 = M4[0]
    u, v = M4[1] / m00, M4[5] / m00
    c20 = M4[2] / m00 - u * u
    c11 = M4[6] / m00 - u * v
    c02 = M4[9] / m00 - v * v
    return gaussian_state(m00, u, v, c20, c11, c02)


def bgk_field(U, nu, dt):
    """Relaxation BGK explicite sur un pas, par champ : U <- U + dt*nu*(M_eq - U).

    Oracle de la source BGK compilee (gmom.bgk_source emise par build_moment_model
    collision=True) : melange convexe de pas theta = nu*dt vers la maxwellienne locale.
    Masse et qdm sont exactement conservees (M_eq les egale). Boucle hote, pour la
    validation / des runs moderes seulement, comme relax_field.

    Args:
        U: champ de moments (15, ...).
        nu: frequence de collision.
        dt: pas de temps.

    Returns:
        Une copie du champ apres un pas BGK explicite, meme forme que U.
    """
    out = np.array(U, dtype=float, copy=True)
    flat = out.reshape(out.shape[0], -1)
    theta = nu * dt
    for c in range(flat.shape[1]):
        meq = maxwellian_state(flat[:, c])
        flat[:, c] = flat[:, c] + theta * (meq - flat[:, c])
    return out
