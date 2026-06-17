"""Modele 2D a 15 moments (fermeture HyQMOM) compose via adc.moments.

Etat (ordre partage avec la reference MATLAB RIEMOM2D) :

    U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]

Flux : F_x[M_pq] = M_{p+1,q}, F_y[M_pq] = M_{p,q+1}. Les six moments d'ordre 5
(M50, M41, M32, M23, M14, M05) sont reconstruits par la fermeture ; les 24 autres entrees
recopient une composante de U. L'algebre M -> C -> S -> fermeture -> C5 -> M5 est generee
par adc.moments ; ce fichier ne contient que la fermeture, la partition spectrale du
jacobien, la borne de vitesse de demarrage et le cablage plasma (sources, Poisson).

Reference mathematique : Bryngelson, Fox & Laurent 2025 (hal-05398171).
"""

from __future__ import annotations

import numpy as np

from adc import dsl
from adc import moments as gmom

# Noms et exposants (p, q) de M_pq, dans l'ordre du vecteur d'etat.
MOMENT_NAMES = [
    "M00",
    "M10",
    "M20",
    "M30",
    "M40",
    "M01",
    "M11",
    "M21",
    "M31",
    "M02",
    "M12",
    "M22",
    "M03",
    "M13",
    "M04",
]
MOMENT_PQ = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (0, 3),
    (1, 3),
    (0, 4),
]

# Indices (0-based) dans U de chaque moment, pour l'assemblage des flux.
IDX = {name: k for k, name in enumerate(MOMENT_NAMES)}

# L'ordre canonique d'adc.moments (q externe, p interne) doit etre celui du vecteur d'etat
# ci-dessus : verifie a l'import.
assert MOMENT_NAMES == gmom.moment_names(
    4
) and MOMENT_PQ == gmom.moment_indices(4)

# Borne de vitesse de demarrage : |u| + K_SPEED*sqrt(C). Une gaussienne atteint exactement
# u +- sqrt(6)*sqrt(C20) (k = 3 couvre avec ~22 % de marge), mais des etats realisables
# asymetriques DEPASSENT k*sqrt(C) sans borne pres de la frontiere de realisabilite (ratio
# 3.29 sur le jeu golden). Demarrage Rusanov uniquement ; pour HLL, exact_speeds=True.
K_SPEED = 3.0

# Partition du jacobien de flux pour les vitesses exactes par sous-blocs (la structure
# par chaines d'exposants d'eigenvalues15_2D.m, flagsym=1). En y, les chaines ne sont pas
# contigues : listes d'indices sur le dFy/dU direct (equivalent au swap d'arguments du
# MATLAB) ; la chaine [2, 7, 11] ne porte pas d'extreme et n'est pas calculee.
HYQMOM_BLOCKS = {
    "x": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [12, 13, 14]],
    "y": [[0, 5, 9, 12, 14], [1, 6, 10, 13], [3, 8, 4]],
}

# Chaines d'ordre 3 du jacobien de flux dont relaxation15.m teste les valeurs propres complexes
# (memes indices que make_corner_eigs / relaxation.py) : x = chaine M03/M13/M04, y = chaine
# M30/M31/M40. Les blocs 3x3 EX/EY sont evalues a l'etat STANDARDISE (M00=1, u=v=0, sigma=1) ;
# le projecteur natif les construit en Expr et passe leur temoin de VP complexes a dsl.eig_max_im.
CORNER_IX = [12, 13, 14]
CORNER_IY = [3, 8, 4]

# Seuils de relaxation15.m (memes valeurs que relaxation.py, l'oracle).
_PROJ_SMALL = 1.0e-6  # bord univarie / planchers H2
_PROJ_EIG_TOL = 1.0e-9  # |Im(lambda)| > tol*max(1,|lambda|) -> bloc complexe
_PROJ_EPS = float(np.finfo(float).eps)  # eps MATLAB (Del1)

# Etats gaussiens du jeu golden (rho, ux, uy, C20, C11, C02) : consommes par gen_states.py
# et par l'oracle d'Isserlis de run.py.
GAUSSIAN_PARAMS = [
    (1.0, 0.0, 0.0, 1.0, 0.0, 1.0),  # repos isotrope
    (2.0, 0.5, -0.3, 1.0, 0.0, 2.0),  # derive anisotrope
    (1.5, -0.2, 0.4, 1.0, 0.45, 0.5),  # correlee (S11 != 0)
    (1.0, 14.1, 0.0, 0.5, 0.0, 0.5),  # haut Mach (Ma ~ 20)
]


def hyqmom_closure(S) -> dict:
    """Fermeture HyQMOM d'ordre 5 (forme polynomiale de closureS5.m).

    Attention : les variantes Moments5.m / S5_2D.m du depot MATLAB different sur
    S32/S23.

    Args:
        S: dict des moments standardises S11..S04 (Expr DSL ou numpy).

    Returns:
        dict des six moments standardises d'ordre 5, S50..S05.
    """
    s11, s30, s21, s12, s03 = S["S11"], S["S30"], S["S21"], S["S12"], S["S03"]
    s40, s31, s22, s13, s04 = S["S40"], S["S31"], S["S22"], S["S13"], S["S04"]
    return {
        "S50": s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0) / 2.0,
        "S05": s03 * (5.0 * s04 - 3.0 * s03 * s03 - 1.0) / 2.0,
        "S41": (
            -s30 * (8.0 * s40 - 9.0 * s30 * s30 - 4.0) * s11 / 4.0
            + (10.0 * s40 - 15.0 * s30 * s30 - 6.0) * s21 / 4.0
            + 2.0 * s30 * s31
        ),
        "S14": (
            -s03 * (8.0 * s04 - 9.0 * s03 * s03 - 4.0) * s11 / 4.0
            + (10.0 * s04 - 15.0 * s03 * s03 - 6.0) * s12 / 4.0
            + 2.0 * s03 * s13
        ),
        "S32": (2.0 * s40 - 3.0 * s30 * s30) * s12 / 2.0
        + (3.0 * s22 - 1.0) * s30 / 2.0,
        "S23": (2.0 * s04 - 3.0 * s03 * s03) * s21 / 2.0
        + (3.0 * s22 - 1.0) * s03 / 2.0,
    }


def moment_sources(U, ex, ey, qm, oc) -> list:
    """Les 15 termes sources de la hierarchie sous force de Lorentz.

    Pour chaque moment :

        S[M_pq] = qm (p Ex M_{p-1,q} + q Ey M_{p,q-1}) + oc (p M_{p-1,q+1} - q M_{p+1,q-1})

    Le terme electrique abaisse l'ordre, le terme magnetique le conserve : la hierarchie
    d'ordre <= 4 est fermee. Adaptateur nom -> (p, q) au-dessus de
    adc.moments.lorentz_sources.

    Args:
        U: dict nom -> Expr/valeur des 15 moments.
        ex, ey: champ electrique (Expr ou valeurs).
        qm: rapport charge/masse.
        oc: pulsation cyclotron.

    Returns:
        Liste des 15 sources, ordonnee comme MOMENT_NAMES (S[M00] = 0).
    """
    return gmom.lorentz_sources(
        {pq: U[nm] for pq, nm in zip(MOMENT_PQ, MOMENT_NAMES)}, ex, ey, qm, oc
    )


def build_moment_model(
    name="hyqmom15",
    closure=hyqmom_closure,
    robust=False,
    eps_m00=1e-12,
    eps_c=1e-12,
    with_sources=False,
    q_over_m=1.0,
    omega_c=0.0,
    debye=None,
    rho_background=0.0,
    omega_p=None,
    exact_speeds=False,
    projection=False,
    Ma=4.0,
    lamin=1e-12,
    collision=False,
    nu_coll=0.0,
) -> dsl.Model:
    """Construit le modele DSL 15 moments avec la fermeture donnee.

    L'algebre des moments vient d'adc.moments.build_moment_model (order=4) ; ne restent ici
    que la fermeture, la partition spectrale, la borne de demarrage et le cablage plasma.

    Args:
        name: nom du modele DSL.
        closure: fermeture d'ordre 5 (par defaut hyqmom_closure).
        robust: False (defaut) = aucune garde, comme le MATLAB (divisions par M00 et
            racines inconditionnelles) -- requis pour comparer aux goldens. True =
            planchers lisses max(x, eps) sur M00, C20, C02, appliques la ou ils
            protegent (racines, divisions de standardisation).
        eps_m00, eps_c: planchers du mode robust sur M00 et sur les covariances.
        with_sources: ajoute la source de Lorentz (moment_sources) -- champ electrique
            lu dans les canaux aux grad_x/grad_y (E = -grad phi, rempli par le Poisson
            du systeme), champ magnetique par la constante omega_c (cuite a la
            compilation).
        q_over_m: rapport charge/masse de la source de Lorentz.
        omega_c: pulsation cyclotron (champ magnetique constant) de la source.
        debye: longueur de Debye adimensionnee (None = pas de Poisson). Emet
            elliptic_rhs((M00 - rho_background)/debye^2). En periodique, un second membre
            a moyenne non nulle rend le solveur singulier : rho_background doit valoir la
            moyenne de M00 du scenario (constante, la masse est conservee) -- l'equivalent
            de la soustraction de moyenne de poisson_fft.m.
        rho_background: moyenne de M00 soustraite au second membre du Poisson periodique.
        omega_p: frequence de la source (constante), borne le pas de temps via
            dt <= cfl/omega_p. None = pas de borne. ATTENTION (audit ADC-197) : ce n'est
            PAS equivalent au dt_source de compute_dt.m
            (= CFL*dx*lambda_flux*k_min^2/max_speed^2) ; la borne ADC est ~500x plus laxe
            et ne mord jamais (la borne transport gouverne) -- pas de fidelite dt MATLAB.
        exact_speeds: True = vitesses d'onde signees par valeurs propres du jacobien de
            flux (autodiff + sous-blocs HYQMOM_BLOCKS) -- requis pour riemann='hll' ; la
            meme verite spectrale sert a la CFL. False (defaut) = borne de demarrage
            k*sqrt(C) (cf. K_SPEED).
        projection: True = emet le projecteur natif relaxation15 via m.projection
            (ADC-275, hook ADC-177) -- chemin PRODUCTION applique par le System a la fin
            de chaque macro-pas entier, branche par branche fidele a relaxation.relax15
            (l'ORACLE). EXIGE exact_speeds=True (le test des VP complexes lit les
            sous-blocs d'ordre 3 du jacobien de flux). Le temoin de VP complexes passe par
            dsl.eig_max_im (ADC-289). False (defaut) = aucun hook emis, chemin
            bit-identique.
        Ma: nombre de Mach du clamp |s30|, |s03| <= 4 + Ma/2 (cuit dans le projecteur).
        lamin: seuil de la porte de collision lambda_min(p2p2) <= lamin (cuit dans le
            projecteur).
        collision: True = ajoute une collision BGK qui relaxe la hierarchie vers la
            maxwellienne locale (moments.bgk_source) ; elle se compose avec la source de
            Lorentz dans l'UNIQUE creneau de source, est echelonnee par dt via le splitting
            (explicite) ou l'IMEX, et est ORTHOGONALE a la projection (M_eq est realisable,
            donc la cascade transport -> BGK -> projection est sure). Les invariants
            collisionnels M00/M10/M01 sont nuls (masse et qdm conservees). False (defaut).
        nu_coll: frequence de collision BGK (constante). Si > 0, borne le pas de temps via
            dt <= cfl/nu_coll (composee en max avec omega_p en une seule source_frequency).

    Returns:
        adc.dsl.Model pret a compiler.

    Raises:
        ValueError: si projection=True sans exact_speeds=True.
    """
    src = None
    if with_sources:

        def src(m_, M_):
            # canaux aux canoniques grad_x/grad_y (Ex = -d phi/dx) + constantes cuites.
            gx = m_.aux("grad_x")
            gy = m_.aux("grad_y")
            qm = m_.param("q_over_m", q_over_m)
            oc = m_.param("omega_c", omega_c)
            return gmom.lorentz_sources(
                M_, -1.0 * gx, -1.0 * gy, qm, oc
            )  # E = -grad phi

    if collision:
        # BGK : relaxe la hierarchie vers la maxwellienne locale (gmom.bgk_source). Compose
        # avec la source de Lorentz dans l'UNIQUE creneau de source ; les invariants
        # collisionnels (M00/M10/M01) sont nuls -> masse et qdm conservees.
        base_src = src  # None (pas de Lorentz) ou la fermeture de Lorentz

        def _bgk_src(m_, M_):
            nu = m_.param("nu_coll", float(nu_coll))
            bgk = gmom.bgk_source(M_, nu)
            if base_src is None:
                return bgk
            return [a + b for a, b in zip(base_src(m_, M_), bgk)]

        src = _bgk_src

    m = gmom.build_moment_model(
        name,
        4,
        closure,
        blocks=HYQMOM_BLOCKS if exact_speeds else None,
        exact_speeds=exact_speeds,
        robust=robust,
        eps_m00=eps_m00,
        eps_cov=eps_c,
        sources=src,
    )

    # Poignees par NOM vers les conservatives declarees par le generateur (les Var de la DSL
    # s'identifient par nom a l'emission : aucune re-declaration).
    U = {nm: dsl.Var(nm, "cons") for nm in MOMENT_NAMES}

    if not exact_speeds:
        # Borne bring-up historique k*sqrt(C) (voir K_SPEED ; NON-production) : formules
        # inline strictement identiques au modele manuel d'origine (memes operations, memes
        # planchers en mode robust) -- le CSE du codegen les fusionne avec celles du flux.
        k = m.param("k_speed", K_SPEED)

        def _floor(x, eps):
            return ((x + eps) + dsl.abs_(x - eps)) / 2.0

        M00b = _floor(U["M00"], eps_m00) if robust else U["M00"]
        ux = U["M10"] / M00b
        uy = U["M01"] / M00b
        c20 = U["M20"] / M00b - ux * ux
        c02 = U["M02"] / M00b - uy * uy
        if robust:
            c20 = _floor(c20, eps_c)
            c02 = _floor(c02, eps_c)
        sx = dsl.sqrt(c20)
        sy = dsl.sqrt(c02)
        m.eigenvalues(
            x=[ux - k * sx, ux + k * sx], y=[uy - k * sy, uy + k * sy]
        )

    # Borne dt source : UNE seule source_frequency, le max des frequences declarees
    # (omega_p de la source electrique, nu_coll de la collision BGK). dt <= cfl/freq.
    freqs = []
    if with_sources and omega_p is not None:
        freqs.append(float(omega_p))
    if collision and nu_coll > 0.0:
        freqs.append(float(nu_coll))
    if freqs:
        m.source_frequency(max(freqs) + 0.0 * U["M00"])  # constante
    if debye is not None:
        inv_l2 = m.param("inv_debye2", 1.0 / float(debye) ** 2)
        rho_bg = m.param("rho_background", float(rho_background))
        m.elliptic_rhs(inv_l2 * (U["M00"] - rho_bg))

    if projection:
        if not exact_speeds:
            raise ValueError(
                "build_moment_model : projection=True exige exact_speeds=True "
                "(le test des VP complexes lit les sous-blocs d'ordre 3 du "
                "jacobien de flux)"
            )
        m.projection(build_projection(m, Ma=Ma, lamin=lamin))

    m.check()
    return m


# ---------------------------------------------------------------------------------------------
# Projecteur natif relaxation15 (ADC-275) : emis via m.projection (hook ADC-177), CHEMIN
# PRODUCTION. relaxation.relax15 reste l'ORACLE de reference ; ce code en est la transcription
# BRANCHE PAR BRANCHE en algebre DSL SANS branche dynamique (contrat pointwise d'ADC-177) :
# chaque branche relaxation15.m devient un melange par masque mask = 0.5*(sign(c)+1) (1 si c>0).
# Les operations s'expriment en Add/Sub/Mul/Div/Pow/Sqrt/Abs/Sign ; le temoin de valeurs propres
# complexes (blocs d'ordre 3 du jacobien) passe par dsl.eig_max_im (ADC-289), la porte de
# collision par dsl.eig_lmin (lambda_min de p2p2). Compile device-clean (foncteur nomme, pas de
# lambda etendue). Verifie bit-pour-bit (~1e-15) contre relax15 sur golden_relax_*.csv.
# ---------------------------------------------------------------------------------------------


def _proj_subst(e, repl):
    """Copie structurelle d'une Expr DSL en substituant des Var par valeur.

    Chaque Var dont le nom figure dans `repl` est remplacee par la valeur associee
    (float -> Const, ou Expr substituee telle quelle). Deux usages : (a) figer les
    primitives standardisantes (u=v=0, sx=sy=1, M00=1) pour evaluer les blocs de coin du
    jacobien a l'etat STANDARDISE ; (b) reinjecter les S_pq COURANTS (post-clamps, des
    Expr) dans ces blocs.

    Args:
        e: Expr DSL (ou int/float) a copier.
        repl: dict nom de Var -> remplacement (float ou Expr).

    Returns:
        Une nouvelle Expr ou les Var citees sont substituees et les constantes pliees.
    """
    if isinstance(e, (int, float)):
        return dsl.Const(float(e))
    if isinstance(e, dsl.Const):
        return e
    if isinstance(e, dsl.Var):
        if e.name in repl:
            r = repl[e.name]
            return r if isinstance(r, dsl.Expr) else dsl.Const(float(r))
        return e
    if isinstance(e, dsl._Bin):
        return _fold_bin(
            type(e), _proj_subst(e.a, repl), _proj_subst(e.b, repl)
        )
    if isinstance(e, (dsl.Neg, dsl.Sqrt, dsl.Abs, dsl.Sign)):
        return _fold_un(type(e), _proj_subst(e.a, repl))
    if isinstance(e, dsl.EigWitness):
        return dsl.EigWitness(
            [[_proj_subst(x, repl) for x in r] for r in e.rows], e.field
        )
    raise TypeError("_proj_subst : noeud DSL non gere %s" % type(e))


# Pliage de constantes pour _proj_subst : a l'etat standardise (u=v=0, sx=sy=1, M00=1) la plupart
# des termes des blocs de coin du jacobien s'annulent (0*x, 0+x...). Sans pliage l'arbre substitue
# reste enorme (codegen CSE non tractable) ; avec pliage il se reduit a une fonction COMPACTE des
# S_pq. Pliage ALGEBRIQUE exact (pas de reordonnancement flottant) : 0/1 absorbants, Const op Const.


def _is_c(e, val):
    return isinstance(e, dsl.Const) and e.value == val


def _fold_bin(cls, a, b):
    ca, cb = isinstance(a, dsl.Const), isinstance(b, dsl.Const)
    if ca and cb:
        if cls is dsl.Add:
            return dsl.Const(a.value + b.value)
        if cls is dsl.Sub:
            return dsl.Const(a.value - b.value)
        if cls is dsl.Mul:
            return dsl.Const(a.value * b.value)
        if cls is dsl.Div:
            if b.value != 0.0:
                return dsl.Const(a.value / b.value)
        if cls is dsl.Pow:
            return dsl.Const(a.value**b.value)
    if cls is dsl.Add:
        if _is_c(a, 0.0):
            return b
        if _is_c(b, 0.0):
            return a
    elif cls is dsl.Sub:
        if _is_c(b, 0.0):
            return a
        if _is_c(a, 0.0):
            return _fold_un(dsl.Neg, b)
    elif cls is dsl.Mul:
        if _is_c(a, 0.0) or _is_c(b, 0.0):
            return dsl.Const(0.0)
        if _is_c(a, 1.0):
            return b
        if _is_c(b, 1.0):
            return a
    elif cls is dsl.Div:
        if _is_c(a, 0.0):
            return dsl.Const(0.0)
        if _is_c(b, 1.0):
            return a
    elif cls is dsl.Pow:
        if _is_c(b, 1.0):
            return a
        if _is_c(b, 0.0):
            return dsl.Const(1.0)
    return cls(a, b)


def _fold_un(cls, a):
    if isinstance(a, dsl.Const):
        if cls is dsl.Neg:
            return dsl.Const(-a.value)
        if cls is dsl.Sqrt:
            return dsl.Const(float(np.sqrt(a.value)))
        if cls is dsl.Abs:
            return dsl.Const(abs(a.value))
        if cls is dsl.Sign:
            return dsl.Const(float(np.sign(a.value)))
    return cls(a)


def _mx(a, b):
    """max(a, b) sans branche : (a + b + |a - b|) / 2."""
    return (a + b + dsl.abs_(a - b)) / 2.0


def _mn(a, b):
    """min(a, b) sans branche : (a + b - |a - b|) / 2."""
    return (a + b - dsl.abs_(a - b)) / 2.0


def _gt0(c):
    """Masque 1 si c > 0, 0 si c < 0, 0.5 en c == 0 : 0.5*(sign(c) + 1).

    Les seuils sont choisis hors du fil du rasoir (cf. golden_relax_gen.m) -- l'egalite
    exacte ne tombe pas sur les goldens.
    """
    return 0.5 * (dsl.sign(c) + 1.0)


def _blend(mask, when_true, when_false):
    """when_false*(1 - mask) + when_true*mask : selection sans branche."""
    return when_false * (1.0 - mask) + when_true * mask


def _corner_blocks_std(m):
    """Blocs 3x3 EX/EY du jacobien de flux aux chaines d'ordre 3, etat standardise.

    Les chaines sont CORNER_IX/IY ; l'evaluation a l'etat standardise se fait par
    substitution u=v=0, sx=sy=1, M00=1 (memes blocs que make_corner_eigs).

    Returns:
        Couple (EX, EY) de listes de lignes d'Expr ne dependant que des primitives S_pq.
    """
    jx = m.flux_jacobian(0)
    jy = m.flux_jacobian(1)
    std = {"u": 0.0, "v": 0.0, "sx": 1.0, "sy": 1.0, "M00": 1.0, "M00f": 1.0}
    ex = [[_proj_subst(jx[a][b], std) for b in CORNER_IX] for a in CORNER_IX]
    ey = [[_proj_subst(jy[a][b], std) for b in CORNER_IY] for a in CORNER_IY]
    return ex, ey


def _p2p2_expr(a03, a04, a11, a12, a13, a21, a22, a30, a31, a40):
    """p2p2_2D.m en Expr DSL (transcription LITTERALE).

    Meme reshape column-major que relaxation.p2p2_2d.

    Returns:
        Liste de 3 lignes de 3 Expr (la matrice p2p2 3x3).
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
    t15 = -1.0 * a13
    t16 = -1.0 * a31
    t11 = a11 * t3
    t13 = a11 * t4
    t14 = a11 * t5
    t17 = -1.0 * t3
    t18 = -1.0 * t5
    t19 = a11 * t9
    t20 = a13 * t7
    t21 = a11 * t10
    t22 = a22 * t7
    t23 = a31 * t7
    t24 = -1.0 * t7
    t25 = -1.0 * t8
    t26 = t7 - 1.0
    t27 = -1.0 * t11
    t28 = -1.0 * t14
    t29 = -1.0 * t19
    t30 = -1.0 * t21
    t31 = -1.0 * t22
    t32 = 1.0 / t26
    t33 = a22 + t12 + t13 + t17 + t18 + t26 + t31
    t34 = a11 + t2 + t4 + t15 + t20 + t25 + t27 + t29
    t35 = a11 + t4 + t6 + t16 + t23 + t25 + t28 + t30
    t36 = t32 * t33
    t38 = t32 * t34
    t39 = t32 * t35
    t37 = -1.0 * t36
    # reshape(flat, (3, 3), order='F') : colonne-major. flat = [c00, c10, c20, c01, c11, ...].
    c00 = t32 * (
        -1.0 * a40 + t10 + t24 - a11 * t6 * 2.0 + a40 * t7 + a30**2 + 1.0
    )
    c10 = t39
    c20 = t37
    c01 = t39
    c11 = t32 * (-1.0 * a22 + t7 + t9 + t10 - t13 * 2.0 + t22 + t7 * t24)
    c21 = t38
    c02 = t37
    c12 = t38
    c22 = t32 * (
        -1.0 * a04 + t9 + t24 + a04 * t7 - a11 * t2 * 2.0 + a03**2 + 1.0
    )
    return [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]]


def _collision_expr(
    s03, s04, s11, s12, s13, s21, s22, s30, s31, s40, lamin, let, witnesses
):
    """collision15_anisotropic.m en Expr DSL SANS branche (cible Z1=u+v / Z2=u-v).

    Args:
        s03, s04, s11, s12, s13, s21, s22, s30, s31, s40: moments standardises d'entree.
        lamin: seuil de la porte de collision (lam0 <= lamin declenche la projection).
        let: callable (name, expr) -> Var let-binding. Chaque etape intermediaire est liee
            a une primitive du modele : sans cela l'arbre d'Expr explose (S22 se developpe
            en ~3e7 noeuds en arbre, meme s'il n'a que ~750 noeuds en DAG -- tout parcours
            sans memo, deps()/eval()/codegen, devient intractable). La liaison garde
            l'arbre PLAT (chaque reference = un Var, taille 1).
        witnesses: liste a laquelle on AJOUTE le temoin eig_lmin BRUT (non let-binde), pour
            qu'il soit surface dans _proj (le declarateur de foncteur
            _collect_eig_witnesses ne descend pas dans prim_defs).

    Returns:
        dict des 10 moments standardises projetes (S03, S04, S11, S12, S13, S21, S22, S30,
        S31, S40).
    """
    s_in = {
        "S03": s03,
        "S04": s04,
        "S11": s11,
        "S12": s12,
        "S13": s13,
        "S21": s21,
        "S22": s22,
        "S30": s30,
        "S31": s31,
        "S40": s40,
    }
    small = _PROJ_SMALL

    # porte : lam0 = max(0, lambda_min(p2p2)) ; collision si lam0 <= lamin. Le temoin eig_lmin est
    # mis de cote dans witnesses pour etre surface dans _proj ; mcoll est let-binde (consommateurs
    # plats).
    p2 = _p2p2_expr(s03, s04, s11, s12, s13, s21, s22, s30, s31, s40)
    lmin_wit = dsl.eig_lmin(p2)
    witnesses.append(lmin_wit)
    lam0 = _mx(0.0, lmin_wit)
    mcoll = let("c_mcoll", _gt0(lamin - lam0))  # 1 si lam0 <= lamin

    del1 = _mx(1.0 - s11**2, _PROJ_EPS)

    # bord univarie INTERNE (H20 < small ou H02 < small) -> retour CJ-like immediat.
    s3 = let("c_s3", dsl.sign(s30) * dsl.sqrt(dsl.abs_(s30 * s03)))
    s4 = dsl.sqrt(s40 * s04)
    h2 = s4 - s3**2 - 1.0
    mh2a = _gt0(small - h2)
    s4 = let("c_s4", _blend(mh2a, s3**2 + 1.0 + small, s4))
    h20 = s40 - s30**2 - 1.0
    h02 = s04 - s03**2 - 1.0
    minner = let("c_minner", _gt0(small - _mn(h20, h02)))
    s11s = let("c_s11s", dsl.sign(s11))
    early = {
        "S03": s3 * s11s,
        "S04": s4,
        "S11": s11s,
        "S12": s3,
        "S13": s4 * s11s,
        "S21": s3 * s11s,
        "S22": s4,
        "S30": s3,
        "S31": s4 * s11s,
        "S40": s4,
    }

    # cible Z1/Z2.
    cS11 = let("c_cS11_0", s11)
    cS30 = let("c_cS30", 3.0 * s12 / 4.0 + s30 / 4.0)
    cS21 = let("c_cS21_0", s03 / 4.0 + 3.0 * s21 / 4.0)
    cS12 = let("c_cS12_0", cS30)
    cS03 = let("c_cS03_0", cS21)
    cS40 = let(
        "c_cS40",
        s04 / 8.0 + 3.0 * s22 / 4.0 + s40 / 8.0 + 3.0 * (1.0 - s11**2) / 2.0,
    )
    cS31 = let("c_cS31", (s13 + s31) / 2.0)
    cS22 = let(
        "c_cS22_0",
        s04 / 8.0 + 3.0 * s22 / 4.0 + s40 / 8.0 - (1.0 - s11**2) / 2.0,
    )
    cS13 = cS31
    cS04 = cS40

    # realisabilite d'ordre 3 de la cible (if S21 ... elif S12 ...).
    c21 = dsl.abs_(cS21 - cS11 * cS30) - dsl.sqrt(
        _mx(del1 * (cS40 - cS30**2 - 1.0), 0.0)
    )
    m21 = let("c_m21", _gt0(c21))
    c12 = dsl.abs_(cS12 - cS11 * cS03) - dsl.sqrt(
        _mx(del1 * (cS04 - cS03**2 - 1.0), 0.0)
    )
    m12 = let("c_m12", (1.0 - m21) * _gt0(c12))
    cS21 = let("c_cS21_1", _blend(m21, cS11 * cS30, cS21))
    cS12 = let("c_cS12_1", _blend(m12, cS11 * cS03, cS12))

    # cible non realisable -> moments CJ (si det(p2[0:2,0:2]) < 0 ou p2[0,0] < 0).
    p2t = _p2p2_expr(cS03, cS04, cS11, cS12, cS13, cS21, cS22, cS30, cS31, cS40)
    det2 = p2t[0][0] * p2t[1][1] - p2t[0][1] * p2t[1][0]
    mcj = let("c_mcj", _gt0(_mx(-1.0 * det2, -1.0 * p2t[0][0])))
    s4c = dsl.sqrt(s40 * s04)
    s3c = let("c_s3c", dsl.sqrt(dsl.abs_(s03 * s30)) * dsl.sign(s30))
    s11c = let("c_s11c", dsl.sign(cS11))
    h2c = s4c - s3c**2 - 1.0
    mh2c = _gt0(small - h2c)
    s4c = let("c_s4c", _blend(mh2c, s3c**2 + 1.0 + small, s4c))
    cj = {
        "S03": s3c * s11c,
        "S21": s3c * s11c,
        "S12": s3c,
        "S30": s3c,
        "S04": s4c,
        "S13": s4c * s11c,
        "S22": s4c,
        "S31": s4c * s11c,
        "S40": s4c,
        "S11": s11c * (1.0 - small),
    }
    cS03 = let("c_cS03_1", _blend(mcj, cj["S03"], cS03))
    cS21 = let("c_cS21_2", _blend(mcj, cj["S21"], cS21))
    cS12 = let("c_cS12_2", _blend(mcj, cj["S12"], cS12))
    cS30 = let("c_cS30_1", _blend(mcj, cj["S30"], cS30))
    cS04 = let("c_cS04_1", _blend(mcj, cj["S04"], cS04))
    cS13 = let("c_cS13_1", _blend(mcj, cj["S13"], cS13))
    cS22 = let("c_cS22_1", _blend(mcj, cj["S22"], cS22))
    cS31 = let("c_cS31_1", _blend(mcj, cj["S31"], cS31))
    cS40 = let("c_cS40_1", _blend(mcj, cj["S40"], cS40))
    cS11 = let("c_cS11_1", _blend(mcj, cj["S11"], cS11))

    # s22 minimal pour det(p2p2) >= 0 (transcription LITTERALE des deux racines).
    s22_1 = (
        (3.0 * cS11) / 8.0
        - cS04 / 32.0
        - cS13 / 8.0
        - cS31 / 8.0
        - cS40 / 32.0
        + (3.0 * cS03 * cS12) / 32.0
        - (cS04 * cS11) / 32.0
        + (3.0 * cS03 * cS21) / 32.0
        - (cS11 * cS13) / 8.0
        + (cS03 * cS30) / 32.0
        + (9.0 * cS12 * cS21) / 32.0
        - (cS11 * cS31) / 8.0
        + (3.0 * cS12 * cS30) / 32.0
        - (cS11 * cS40) / 32.0
        + (3.0 * cS21 * cS30) / 32.0
        + cS03**2 / 64.0
        + (3.0 * cS11**2) / 8.0
        + cS11**3 / 8.0
        + (9.0 * cS12**2) / 64.0
        + (9.0 * cS21**2) / 64.0
        + cS30**2 / 64.0
        + 1.0 / 8.0
    ) / ((3.0 * cS11) / 16.0 + 3.0 / 16.0)
    s22_2 = -1.0 * (
        (
            cS13 / 8.0
            - (3.0 * cS11) / 8.0
            - cS04 / 32.0
            + cS31 / 8.0
            - cS40 / 32.0
            - (3.0 * cS03 * cS12) / 32.0
            + (cS04 * cS11) / 32.0
            + (3.0 * cS03 * cS21) / 32.0
            - (cS11 * cS13) / 8.0
            - (cS03 * cS30) / 32.0
            - (9.0 * cS12 * cS21) / 32.0
            - (cS11 * cS31) / 8.0
            + (3.0 * cS12 * cS30) / 32.0
            + (cS11 * cS40) / 32.0
            - (3.0 * cS21 * cS30) / 32.0
            + cS03**2 / 64.0
            + (3.0 * cS11**2) / 8.0
            - cS11**3 / 8.0
            + (9.0 * cS12**2) / 64.0
            + (9.0 * cS21**2) / 64.0
            + cS30**2 / 64.0
            + 1.0 / 8.0
        )
        / ((3.0 * cS11) / 16.0 - 3.0 / 16.0)
    )
    s22max = _mx(let("c_s22_1", s22_1), let("c_s22_2", s22_2))
    cS22 = _mx(s22max, cS22) * 1.001
    cS22 = _mn(cS22, cS40)
    cS22 = let("c_cS22_2", _mx(cS22, 1.0 / 3.0))

    general = {
        "S03": cS03,
        "S04": cS04,
        "S11": cS11,
        "S12": cS12,
        "S13": cS13,
        "S21": cS21,
        "S22": cS22,
        "S30": cS30,
        "S31": cS31,
        "S40": cS40,
    }
    # interne : early si bord univarie, sinon cible generale.
    collide = {
        k: let("c_collide_" + k, _blend(minner, early[k], general[k]))
        for k in general
    }
    # global : projection si lam0 <= lamin, sinon entree inchangee.
    return {
        k: let("c_out_" + k, _blend(mcoll, collide[k], s_in[k])) for k in s_in
    }


def build_projection(m, Ma=4.0, lamin=1e-12) -> list:
    """relaxation15.m -> liste de 15 Expr pour m.projection (ADC-275).

    Une Expr par composante conservative, ordre MOMENT_NAMES. Transcription BRANCHE PAR
    BRANCHE de relaxation.relax15 (l'oracle) en algebre DSL sans branche : clamps s30/s03
    (preservant H2), bord univarie H20/H02, clamp s11, suppression des VP complexes (temoin
    dsl.eig_max_im sur les blocs d'ordre 3 du jacobien a l'etat standardise), puis
    collision15_anisotropic (porte dsl.eig_lmin sur p2p2). M00, M10, M01 sont inchanges
    (relaxation15 ne touche que les moments d'ordre >= 2 ; u, v, M00 conserves).

    Args:
        m: le modele construit par build_moment_model(exact_speeds=True).
        Ma: Mach du clamp |s30| <= 4 + Ma/2.
        lamin: seuil de la porte de collision.

    Returns:
        Liste de 15 Expr, une par composante conservative dans l'ordre MOMENT_NAMES.
    """
    small = _PROJ_SMALL
    s3m = 4.0 + Ma / 2.0

    # Liaison let : chaque etape intermediaire devient une primitive du modele (un local C++ partage
    # + un Var de taille 1 pour les references). Sans cela l'arbre d'Expr de la projection explose
    # (parcours sans memo intractable : deps() de m.check, eval() du miroir numpy, ~3e7 noeuds).
    _ctr = [0]

    def let(name, e):
        _ctr[0] += 1
        return m.primitive("proj_%s_%d" % (name, _ctr[0]), e)

    # primitives du modele : moments standardises et facteurs de de-standardisation.
    def P(name):
        return dsl.Var(name, "prim")

    s30, s40 = P("S30"), P("S40")
    s11 = P("S11")
    s21, s31 = P("S21"), P("S31")
    s12, s22 = P("S12"), P("S22")
    s03, s13, s04 = P("S03"), P("S13"), P("S04")
    c20, c02 = P("C20"), P("C02")
    sx, sy = P("sx"), P("sy")
    u, v = P("u"), P("v")
    m00 = dsl.Var("M00", "cons")

    # --- clamp |s30| <= s3m en preservant H20 = s40 - s30^2 - 1.
    mc30 = let("mc30", _gt0(dsl.abs_(s30) - s3m))
    h20 = s40 - s30**2 - 1.0
    s30c = dsl.sign(s30) * s3m
    s40 = let("s40_a", _blend(mc30, h20 + s30c**2 + 1.0, s40))
    s30 = let("s30_a", _blend(mc30, s30c, s30))
    # --- clamp |s03| <= s3m en preservant H02.
    mc03 = let("mc03", _gt0(dsl.abs_(s03) - s3m))
    h02 = s04 - s03**2 - 1.0
    s03c = dsl.sign(s03) * s3m
    s04 = let("s04_a", _blend(mc03, h02 + s03c**2 + 1.0, s04))
    s03 = let("s03_a", _blend(mc03, s03c, s03))

    # --- bord univarie : si H20 < small ou H02 < small, plancher H2 et s11 <- sign(s11).
    h20 = s40 - s30**2 - 1.0
    h02 = s04 - s03**2 - 1.0
    muv = let("muv", _gt0(small - _mn(h20, h02)))
    s40 = let("s40_b", _blend(muv, s30**2 + 1.0 + small, s40))
    s04 = let("s04_b", _blend(muv, s03**2 + 1.0 + small, s04))
    s11 = let("s11_b", _blend(muv, dsl.sign(s11), s11))

    # --- clamp s11 (flagS11) : si |s11| >= 1 - small, substitution complete (et collision sautee).
    ms11 = let("ms11", _gt0(dsl.abs_(s11) - (1.0 - small)))
    sgn11 = let("sgn11", dsl.sign(s11))
    s3 = let("s11clamp_s3", dsl.sqrt(dsl.abs_(s03 * s30)) * dsl.sign(s30))
    s4 = dsl.sqrt(dsl.abs_(s04 * s40))
    h2 = s4 - s3**2 - 1.0
    mh2 = _gt0(
        -1.0 * h2
    )  # 1 si H2 < 0 (la branche MATLAB est H2 <= 0 ; egalite hors goldens)
    s4 = let("s11clamp_s4", _blend(mh2, s3**2 + 1.0 + small, s4))
    # branche + : s03=S3, s21=S3, s12=S3, s30=S3, s13=S4, s22=S4, s31=S4, s40=S4, s04=S4 ;
    # branche - : s03/s21/s13/s31 changent de signe (facteur sgn11). s12=S3, s30=S3, s22=S4, s40=S4.
    s11 = let("s11_c", _blend(ms11, (1.0 - small) * sgn11, s11))
    s03 = let("s03_c", _blend(ms11, sgn11 * s3, s03))
    s21 = let("s21_c", _blend(ms11, sgn11 * s3, s21))
    s12 = let("s12_c", _blend(ms11, s3, s12))
    s30 = let("s30_c", _blend(ms11, s3, s30))
    s04 = let("s04_c", _blend(ms11, s4, s04))
    s13 = let("s13_c", _blend(ms11, sgn11 * s4, s13))
    s22 = let("s22_c", _blend(ms11, s4, s22))
    s31 = let("s31_c", _blend(ms11, sgn11 * s4, s31))
    s40 = let("s40_c", _blend(ms11, s4, s40))

    # --- suppression des VP complexes (blocs d'ordre 3 du jacobien a l'etat standardise) :
    # temoin = max(|Im|) des deux blocs ; correction s21=s12=0, s22 >= 1/3 si > tol. Les blocs sont
    # batis sur les S_pq COURANTS (post-clamps) via _proj_subst des primitives standardisantes.
    ex0, ey0 = _corner_blocks_std(m)
    sub = {
        "S30": s30,
        "S40": s40,
        "S11": s11,
        "S21": s21,
        "S31": s31,
        "S12": s12,
        "S22": s22,
        "S03": s03,
        "S13": s13,
        "S04": s04,
    }
    ex = [[_proj_subst(e, sub) for e in row] for row in ex0]
    ey = [[_proj_subst(e, sub) for e in row] for row in ey0]
    # temoins eig_max_im des deux blocs : mis de cote dans `witnesses` (surface dans _proj plus bas
    # par un porteur nul, pour declarer le foncteur) ; le masque meig est let-binde (consommateurs
    # plats).
    witnesses = []
    wit_x = dsl.eig_max_im(ex)
    wit_y = dsl.eig_max_im(ey)
    witnesses += [wit_x, wit_y]
    max_im = _mx(wit_x, wit_y)
    meig = let("meig", _gt0(max_im - _PROJ_EIG_TOL))
    s21 = let("s21_d", _blend(meig, dsl.Const(0.0), s21))
    s12 = let("s12_d", _blend(meig, dsl.Const(0.0), s12))
    s22 = let("s22_d", _blend(meig, _mx(s22, 1.0 / 3.0), s22))

    # --- collision15_anisotropic (seulement si flagS11 != 1 : melange par 1 - ms11).
    coll = _collision_expr(
        s03, s04, s11, s12, s13, s21, s22, s30, s31, s40, lamin, let, witnesses
    )
    fS03 = let("fS03", _blend(ms11, s03, coll["S03"]))
    fS04 = let("fS04", _blend(ms11, s04, coll["S04"]))
    fS11 = let("fS11", _blend(ms11, s11, coll["S11"]))
    fS12 = let("fS12", _blend(ms11, s12, coll["S12"]))
    fS13 = let("fS13", _blend(ms11, s13, coll["S13"]))
    fS21 = let("fS21", _blend(ms11, s21, coll["S21"]))
    fS22 = let("fS22", _blend(ms11, s22, coll["S22"]))
    fS30 = let("fS30", _blend(ms11, s30, coll["S30"]))
    fS31 = let("fS31", _blend(ms11, s31, coll["S31"]))
    fS40 = let("fS40", _blend(ms11, s40, coll["S40"]))

    # --- de-standardisation C_pq = S_pq * sx^p sy^q, puis reconstruction binomiale M_pq.
    cn = {
        (2, 0): c20,
        (1, 1): fS11 * sx * sy,
        (0, 2): c02,
        (3, 0): fS30 * sx**3,
        (2, 1): fS21 * sx**2 * sy,
        (1, 2): fS12 * sx * sy**2,
        (0, 3): fS03 * sy**3,
        (4, 0): fS40 * sx**4,
        (3, 1): fS31 * sx**3 * sy,
        (2, 2): fS22 * sx**2 * sy**2,
        (1, 3): fS13 * sx * sy**3,
        (0, 4): fS04 * sy**4,
    }

    # Porteur nul : surface les temoins eig BRUTS dans _proj (top-level) sans rien changer
    # numeriquement (0 * temoin = 0 a l'eval ; en C++ l'appel du foncteur est multiplie par 0).
    # Necessaire car _collect_eig_witnesses (declarateur de foncteur) ne descend pas dans prim_defs,
    # ou meig/mcoll sont let-bindes. Somme des temoins (eig_max_im x, eig_max_im y, eig_lmin p2p2).
    carrier = None
    for w in witnesses:
        carrier = w if carrier is None else carrier + w
    m00_carried = m00 + dsl.Const(0.0) * carrier if carrier is not None else m00

    out = []
    for p, q in MOMENT_PQ:
        if (p, q) == (0, 0):
            out.append(m00_carried)
            continue
        if (p, q) == (1, 0):
            out.append(dsl.Var("M10", "cons"))
            continue
        if (p, q) == (0, 1):
            out.append(dsl.Var("M01", "cons"))
            continue
        # M_pq = M00 * sum_ij binom(p,i) binom(q,j) u^(p-i) v^(q-j) C_ij (C00=1, C10=C01=0).
        acc = None
        for i in range(p + 1):
            for j in range(q + 1):
                if (i, j) == (0, 0):
                    cij = dsl.Const(1.0)
                elif (i, j) in ((1, 0), (0, 1)):
                    continue
                else:
                    cij = cn[(i, j)]
                coef = _binom(p, i) * _binom(q, j)
                term = coef * (u ** (p - i)) * (v ** (q - j)) * cij
                acc = term if acc is None else acc + term
        out.append(m00 * acc)
    return out


# ---------------------------------------------------------------------------------------------
# Generateurs d'etats REALISABLES et oracle gaussien exact (independants du pipeline ci-dessus).
# ---------------------------------------------------------------------------------------------


def _binom(n, k):
    from math import comb

    return float(comb(n, k))


def _gaussian_central(c20, c11, c02, p, q):
    """Moment centre C_pq (p+q <= 5) d'une gaussienne 2D (Isserlis).

    Covariance [[c20, c11], [c11, c02]] : 0 si p+q impair ; ordres 0/2/4 en forme fermee ;
    ordre 5 = 0.
    """
    table = {
        (0, 0): 1.0,
        (1, 0): 0.0,
        (0, 1): 0.0,
        (2, 0): c20,
        (1, 1): c11,
        (0, 2): c02,
        (3, 0): 0.0,
        (2, 1): 0.0,
        (1, 2): 0.0,
        (0, 3): 0.0,
        (4, 0): 3.0 * c20**2,
        (3, 1): 3.0 * c20 * c11,
        (2, 2): c20 * c02 + 2.0 * c11**2,
        (1, 3): 3.0 * c02 * c11,
        (0, 4): 3.0 * c02**2,
    }
    if (p, q) in table:
        return table[(p, q)]
    if (p + q) == 5:
        return 0.0  # tout moment centre gaussien d'ordre impair est nul
    raise ValueError("ordre non couvert : (%d, %d)" % (p, q))


def gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q) -> float:
    """Moment brut EXACT M_pq d'une gaussienne 2D (oracle d'Isserlis).

    Binome sur les moments centres d'Isserlis :
    M_pq = rho * sum_ij binom(p,i) binom(q,j) ux^(p-i) uy^(q-j) C_ij. Oracle independant du
    pipeline de fermeture (la fermeture HyQMOM est exacte sur les gaussiennes : S30=S21=...=0,
    S40=S04=3 => les 6 moments standardises d'ordre 5 retournes sont exactement nuls).
    """
    tot = 0.0
    for i in range(p + 1):
        for j in range(q + 1):
            tot += (
                _binom(p, i)
                * _binom(q, j)
                * ux ** (p - i)
                * uy ** (q - j)
                * _gaussian_central(c20, c11, c02, i, j)
            )
    return rho * tot


def gaussian_state(rho, ux, uy, c20, c11, c02) -> np.ndarray:
    """Vecteur d'etat (15,) des moments bruts exacts d'une gaussienne 2D."""
    return np.array(
        [
            gaussian_raw_moment(rho, ux, uy, c20, c11, c02, p, q)
            for (p, q) in MOMENT_PQ
        ]
    )


def crossing_state(
    n: int, ma: float, rho_in=1.0, rho_out=1e-3, T=1.0, r=0.0
) -> np.ndarray:
    """Condition initiale du croisement de jets (main_pb_2Dcrossing_2DHyQMOM15.m).

    Fond au repos a basse densite rho_out sur [-0.5, 0.5]^2, carre central [3n/8, 5n/8)
    coupe par l'anti-diagonale -- jets gaussiens (+Uc, +Uc) sous la diagonale,
    (-Uc, -Uc) au-dessus, repos sur la diagonale exacte, Uc = ma / sqrt(2).

    Pour r, la covariance vaut [[C20, C11], [C11, C02]] avec C20 = C02 = T et
    C11 = r*sqrt(C20*C02) = r*T (meme convention que le driver MATLAB, qui pose
    C11 = r*sqrt(C20*C02) avant InitializeM4_15). Les 15 moments produits par
    gaussian_state (formule d'Isserlis) sont IDENTIQUES a InitializeM4_15 a l'arrondi pres
    (~1e-12) pour tout r dans le domaine : InitializeM4_15 part des moments standardises
    S22 = 1, S31 = S13 = 0 dans la base PRINCIPALE (decorrelee) puis S4toC4 reintroduit la
    correlation par une rotation dependant de C11 -- la gaussienne correlee exacte, pas une
    approximation (parite verifiee par golden_crossing_gen.m / golden/golden_crossing.csv).

    Args:
        n: nombre de cellules par axe (grille n x n).
        ma: nombre de Mach des jets ; Uc = ma / sqrt(2).
        rho_in: densite des jets et de l'anti-diagonale.
        rho_out: densite du fond au repos.
        T: temperature (C20 = C02 = T).
        r: coefficient de correlation initial de la gaussienne jointe (-1 < r < 1 ;
            |r| = 1 rend la covariance singuliere).

    Returns:
        Tableau (15, n, n), axe x en dernier (convention des cas adc).

    Raises:
        ValueError: si r n'est pas dans l'intervalle ouvert (-1, 1).
    """
    if not -1.0 < r < 1.0:
        raise ValueError(
            "crossing_state : r doit verifier -1 < r < 1 (covariance definie "
            "positive) ; recu r = %r" % (r,)
        )
    uc = ma / np.sqrt(2.0)
    c11 = r * T  # = r*sqrt(C20*C02) avec C20=C02=T
    m_out = gaussian_state(rho_out, 0.0, 0.0, T, c11, T)  # fond au repos
    m_mid = gaussian_state(
        rho_in, 0.0, 0.0, T, c11, T
    )  # anti-diagonale au repos
    m_top = gaussian_state(
        rho_in, -uc, -uc, T, c11, T
    )  # au-dessus : jet (-Uc, -Uc)
    m_bot = gaussian_state(
        rho_in, uc, uc, T, c11, T
    )  # en dessous : jet (+Uc, +Uc)
    U = np.empty((15, n, n))
    U[:] = m_out[:, None, None]
    lo, hi = 3 * n // 8, 5 * n // 8  # bornes 0-based [lo, hi)
    for j in range(lo, hi):  # j : indice y
        for i in range(lo, hi):  # i : indice x (dernier axe)
            if i + j == n - 1:
                U[:, j, i] = m_mid
            elif i + j > n - 1:
                U[:, j, i] = m_top
            else:
                U[:, j, i] = m_bot
    return U


def mixture_state(weights, vxs, vys) -> np.ndarray:
    """Vecteur d'etat (15,) d'un melange discret de Dirac.

    Pour f = sum_k w_k delta(v - v_k), les moments exacts sont
    M_pq = sum_k w_k vx_k^p vy_k^q. Toujours realisable (c'est une distribution), permet des
    etats fortement asymetriques / quasi-degeneres hors de portee des gaussiennes.
    """
    w = np.asarray(weights, dtype=float)
    vx = np.asarray(vxs, dtype=float)
    vy = np.asarray(vys, dtype=float)
    return np.array([np.sum(w * vx**p * vy**q) for (p, q) in MOMENT_PQ])
