"""Modele 2D a 15 moments (fermeture HyQMOM), compose via le generateur adc.moments.

Etat (ordre partage avec la reference MATLAB RIEMOM2D) :

    U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]

Flux : F_x[M_pq] = M_{p+1,q}, F_y[M_pq] = M_{p,q+1}. Les six moments d'ordre 5
(M50, M41, M32, M23, M14, M05) sont reconstruits par la fermeture ; les 24 autres entrees
recopient une composante de U. L'algebre M -> C -> S -> fermeture -> C5 -> M5 est generee
par adc.moments ; ce fichier ne contient que la fermeture, la partition spectrale du
jacobien, la borne de vitesse de demarrage et le cablage plasma (sources, Poisson).

Reference mathematique : Bryngelson, Fox & Laurent 2025 (hal-05398171).
"""

import numpy as np

from adc import dsl
from adc import moments as gmom

# Noms et exposants (p, q) de M_pq, dans l'ordre du vecteur d'etat.
MOMENT_NAMES = ["M00", "M10", "M20", "M30", "M40", "M01", "M11", "M21", "M31",
                "M02", "M12", "M22", "M03", "M13", "M04"]
MOMENT_PQ = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1),
             (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (0, 4)]

# Indices (0-based) dans U de chaque moment, pour l'assemblage des flux.
IDX = {name: k for k, name in enumerate(MOMENT_NAMES)}

# L'ordre canonique d'adc.moments (q externe, p interne) doit etre celui du vecteur d'etat
# ci-dessus : verifie a l'import.
assert MOMENT_NAMES == gmom.moment_names(4) and MOMENT_PQ == gmom.moment_indices(4)

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

# Etats gaussiens du jeu golden (rho, ux, uy, C20, C11, C02) : consommes par gen_states.py
# et par l'oracle d'Isserlis de run.py.
GAUSSIAN_PARAMS = [
    (1.0, 0.0, 0.0, 1.0, 0.0, 1.0),     # repos isotrope
    (2.0, 0.5, -0.3, 1.0, 0.0, 2.0),    # derive anisotrope
    (1.5, -0.2, 0.4, 1.0, 0.45, 0.5),   # correlee (S11 != 0)
    (1.0, 14.1, 0.0, 0.5, 0.0, 0.5),    # haut Mach (Ma ~ 20)
]


def hyqmom_closure(S):
    """Fermeture HyQMOM d'ordre 5 (forme polynomiale de closureS5.m ; attention, les
    variantes Moments5.m / S5_2D.m du depot MATLAB different sur S32/S23).

    @p S : dict des moments standardises S11..S04 (Expr DSL ou numpy). @return dict S50..S05."""
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


def moment_sources(U, ex, ey, qm, oc):
    """Les 15 termes sources de la hierarchie sous force de Lorentz :

        S[M_pq] = qm (p Ex M_{p-1,q} + q Ey M_{p,q-1}) + oc (p M_{p-1,q+1} - q M_{p+1,q-1})

    Le terme electrique abaisse l'ordre, le terme magnetique le conserve : la hierarchie
    d'ordre <= 4 est fermee. Adaptateur nom -> (p, q) au-dessus de
    adc.moments.lorentz_sources. @p U : dict nom -> Expr/valeur ; @return liste ordonnee
    comme MOMENT_NAMES (S[M00] = 0)."""
    return gmom.lorentz_sources({pq: U[nm] for pq, nm in zip(MOMENT_PQ, MOMENT_NAMES)},
                                ex, ey, qm, oc)


def build_moment_model(name="hyqmom15", closure=hyqmom_closure, robust=False,
                       eps_m00=1e-12, eps_c=1e-12, with_sources=False,
                       q_over_m=1.0, omega_c=0.0, debye=None, rho_background=0.0,
                       omega_p=None, exact_speeds=False):
    """Construit le modele DSL 15 moments avec la fermeture @p closure.

    L'algebre des moments vient d'adc.moments.build_moment_model (order=4) ; ne restent ici
    que la fermeture, la partition spectrale, la borne de demarrage et le cablage plasma.

    @p robust : False (defaut) = aucune garde, comme le MATLAB (divisions par M00 et racines
    inconditionnelles) -- requis pour comparer aux goldens. True = planchers lisses
    max(x, eps) sur M00, C20, C02, appliques la ou ils protegent (racines, divisions de
    standardisation).

    @p with_sources : ajoute la source de Lorentz (moment_sources) -- champ electrique lu
    dans les canaux aux grad_x/grad_y (E = -grad phi, rempli par le Poisson du systeme),
    champ magnetique par la constante @p omega_c (cuite a la compilation).

    @p debye : longueur de Debye adimensionnee (None = pas de Poisson). Emet
    elliptic_rhs((M00 - rho_background)/debye^2). En periodique, un second membre a moyenne
    non nulle rend le solveur singulier : @p rho_background doit valoir la moyenne de M00 du
    scenario (constante, la masse est conservee) -- l'equivalent de la soustraction de
    moyenne de poisson_fft.m.

    @p omega_p : frequence de la source, borne le pas de temps (la deuxieme CFL de
    compute_dt.m). None = pas de borne.

    @p exact_speeds : True = vitesses d'onde signees par valeurs propres du jacobien de flux
    (autodiff + sous-blocs HYQMOM_BLOCKS) -- requis pour riemann='hll' ; la meme verite
    spectrale sert a la CFL. False (defaut) = borne de demarrage k*sqrt(C) (cf. K_SPEED).
    @return adc.dsl.Model pret a compiler."""
    src = None
    if with_sources:
        def src(m_, M_):
            # canaux aux canoniques grad_x/grad_y (Ex = -d phi/dx) + constantes cuites.
            gx = m_.aux("grad_x")
            gy = m_.aux("grad_y")
            qm = m_.param("q_over_m", q_over_m)
            oc = m_.param("omega_c", omega_c)
            return gmom.lorentz_sources(M_, -1.0 * gx, -1.0 * gy, qm, oc)  # E = -grad phi

    m = gmom.build_moment_model(name, 4, closure,
                                blocks=HYQMOM_BLOCKS if exact_speeds else None,
                                exact_speeds=exact_speeds, robust=robust,
                                eps_m00=eps_m00, eps_cov=eps_c, sources=src)

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
        m.eigenvalues(x=[ux - k * sx, ux + k * sx], y=[uy - k * sy, uy + k * sy])

    if with_sources and omega_p is not None:
        m.source_frequency(omega_p + 0.0 * U["M00"])  # borne dt source (constante)
    if debye is not None:
        inv_l2 = m.param("inv_debye2", 1.0 / float(debye) ** 2)
        rho_bg = m.param("rho_background", float(rho_background))
        m.elliptic_rhs(inv_l2 * (U["M00"] - rho_bg))

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


def crossing_state(n, ma, rho_in=1.0, rho_out=1e-3, T=1.0, r=0.0):
    """Condition initiale du croisement de jets (main_pb_2Dcrossing_2DHyQMOM15.m) : fond au
    repos a basse densite @p rho_out sur [-0.5, 0.5]^2, carre central [3n/8, 5n/8) coupe par
    l'anti-diagonale -- jets gaussiens (+Uc, +Uc) sous la diagonale, (-Uc, -Uc) au-dessus,
    repos sur la diagonale exacte, Uc = ma / sqrt(2). @p r : correlation initiale (0 dans le
    driver de reference : gaussian_state == InitializeM4_15 exactement ; pour r != 0 le MATLAB
    fige S22 = 1, S31 = S13 = 0, ce qui n'est PAS la gaussienne correlee exacte -- non porte).
    @return tableau (15, n, n), axe x en dernier (convention des cas adc)."""
    if r != 0.0:
        raise NotImplementedError("crossing_state : r != 0 non porte (InitializeM4_15 fige "
                                  "S22=1, S31=S13=0, distinct de la gaussienne correlee exacte)")
    uc = ma / np.sqrt(2.0)
    c11 = r * T
    m_out = gaussian_state(rho_out, 0.0, 0.0, T, c11, T)     # fond au repos
    m_mid = gaussian_state(rho_in, 0.0, 0.0, T, c11, T)      # anti-diagonale au repos
    m_top = gaussian_state(rho_in, -uc, -uc, T, c11, T)      # au-dessus : jet (-Uc, -Uc)
    m_bot = gaussian_state(rho_in, uc, uc, T, c11, T)        # en dessous : jet (+Uc, +Uc)
    U = np.empty((15, n, n))
    U[:] = m_out[:, None, None]
    lo, hi = 3 * n // 8, 5 * n // 8                          # bornes 0-based [lo, hi)
    for j in range(lo, hi):          # j : indice y
        for i in range(lo, hi):      # i : indice x (dernier axe)
            if i + j == n - 1:
                U[:, j, i] = m_mid
            elif i + j > n - 1:
                U[:, j, i] = m_top
            else:
                U[:, j, i] = m_bot
    return U


def mixture_state(weights, vxs, vys):
    """Vecteur d'etat (15,) d'un melange discret f = sum_k w_k delta(v - v_k) : moments exacts
    M_pq = sum_k w_k vx_k^p vy_k^q. Toujours realisable (c'est une distribution), permet des
    etats fortement asymetriques / quasi-degeneres hors de portee des gaussiennes."""
    w = np.asarray(weights, dtype=float)
    vx = np.asarray(vxs, dtype=float)
    vy = np.asarray(vys, dtype=float)
    return np.array([np.sum(w * vx ** p * vy ** q) for (p, q) in MOMENT_PQ])
