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
d'ordre 5 portent la fermeture.

Depuis ADC-172, le pipeline M -> C -> S -> fermeture -> C5 -> M5 n'est PLUS transcrit a la
main : il est GENERE par adc.moments (adc_cpp ADC-164), qui derive l'algebre binomiale en
boucles sur l'AST de la DSL (let-bindings -> variables locales C++ nommees, codegen lineaire).
Ne restent manuels ici que : la fermeture (`hyqmom_closure`), les blocs spectraux
(`HYQMOM_BLOCKS`), la borne bring-up k*sqrt(C) et le cablage plasma (sources Lorentz via
adc.moments.lorentz_sources, Poisson, omega_p). Equivalence a l'ancien modele manuel verifiee
sur les goldens MATLAB (flux 7.7e-13, vitesses 8e-10 hors etat quasi-degenere) -- voir run.py.

References : RIEMOM2D/{Flux_closure15_2D.m, M2CS4_15.m, M4toC4.m, closureS5.m, C5toM5.m} ;
document maths main.pdf eq. 1.8-1.12 (Bryngelson, Fox & Laurent 2025, hal-05398171).
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

# Contrat de layout avec le generateur generique adc.moments (adc_cpp ADC-164) : son ordre
# canonique (q externe, p interne) EST l'ordre du document maths / MATLAB. Verifie a l'import.
assert MOMENT_NAMES == gmom.moment_names(4) and MOMENT_PQ == gmom.moment_indices(4)

# Borne de vitesse bring-up : |u| + K_SPEED*sqrt(C). Les vraies valeurs propres (eigenvalues15_2D
# flagsym=1, cf. golden/golden_vp.csv) s'etendent a u +- sqrt(6)*sqrt(C20) ~ +-2.449*sqrt(C20)
# pour une gaussienne (ratio verifie EXACT sur les 4 etats gaussiens du jeu golden) : k = 3 les
# couvre avec ~22 % de marge. MAIS des etats realisables asymetriques DEPASSENT k*sqrt(C) (ratio
# jusqu'a 3.29 sur les melanges du jeu golden, non borne pres de la frontiere de realisabilite) :
# run.py le DEMONTRE en consommant golden_vp.csv. Borne de demarrage Rusanov uniquement ; le
# chemin production est la jacobienne exacte (ADC-87/ADC-88).
K_SPEED = 3.0

# Partitions de sous-blocs du jacobien de flux pour les vitesses HLL EXACTES
# (m.wave_speeds_from_jacobian), miroir EXACT du chemin production du MATLAB
# (eigenvalues15_2D.m flagsym=1 : eig par blocs 1:5 / 6:9 / 13:15 de Jx, le bloc 10:12 est
# sciemment saute ; Jy y est obtenu en appelant jacobian15 avec les arguments x<->y permutes,
# ce qui revient EXACTEMENT a prendre, sur le dFy/dU DIRECT, les chaines par exposant en x --
# listes d'indices NON CONTIGUES ci-dessous ; la chaine sautee est [2, 7, 11]).
HYQMOM_BLOCKS = {
    "x": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [12, 13, 14]],
    "y": [[0, 5, 9, 12, 14], [1, 6, 10, 13], [3, 8, 4]],
}

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


def moment_sources(U, ex, ey, qm, oc):
    """Table des 15 termes sources de la hierarchie de moments (document maths eq. 1.2),
    generee PROGRAMMATIQUEMENT (les eq. explicites 1.3-1.7 du document servent d'oracle de
    verification dans run_crossing.py, jamais de source de copie) :

        S[M_pq] = qm (p Ex M_{p-1,q} + q Ey M_{p,q-1}) + oc (p M_{p-1,q+1} - q M_{p+1,q-1})

    avec qm = q/m, oc = Omega_c = qB/m. Le terme electrique ABAISSE l'ordre (les moments
    references existent toujours) ; le terme magnetique CONSERVE l'ordre total (rotation dans
    l'espace des vitesses : la hierarchie d'ordre <= 4 est fermee sous B). @p U : dict nom ->
    Expr/valeur ; @p ex, ey : champ electrique (Expr aux ou valeurs) ; @return liste de 15
    expressions dans l'ordre de MOMENT_NAMES (S[M00] = 0.0 : la masse n'a pas de source).

    Adaptateur nom -> (p, q) au-dessus de la hierarchie generique adc.moments.lorentz_sources
    (adc_cpp ADC-164, meme formule remontee dans le coeur car independante de la fermeture)."""
    return gmom.lorentz_sources({pq: U[nm] for pq, nm in zip(MOMENT_PQ, MOMENT_NAMES)},
                                ex, ey, qm, oc)


def build_moment_model(name="hyqmom15", closure=hyqmom_closure, robust=False,
                       eps_m00=1e-12, eps_c=1e-12, with_sources=False,
                       q_over_m=1.0, omega_c=0.0, debye=None, rho_background=0.0,
                       omega_p=None, exact_speeds=False):
    """Construit le modele DSL 15 moments avec la fermeture @p closure.

    Le corps est delegue au generateur generique adc.moments.build_moment_model (order=4) :
    seule la fermeture, les blocs, la borne bring-up et le cablage plasma restent ici.

    @p robust : False (defaut) = mode bit_match, AUCUNE garde, fidele au MATLAB qui n'en a
    aucune (division par M00, sqrt(C20), sqrt(C02) inconditionnels) -- requis pour la
    validation golden. True = planchers M00/C20/C02 (max lisse via |.|), cote cas uniquement,
    jamais dans le coeur. Nuance vs l'ancien modele manuel : le generateur n'utilise les
    planchers C20/C02 que la ou ils protegent (sqrt, divisions de standardisation), pas dans
    les termes polynomiaux de la reconstruction d'ordre 5 -- identique sur etat sain (les
    planchers y sont l'identite), differences uniquement pres de la degenerescence, ou les
    deux variantes restent finies.

    @p with_sources : ajoute la source de la hierarchie (moment_sources) -- electrique via les
    canaux aux canoniques grad_x/grad_y (Ex = -d phi/dx, rempli par le Poisson du systeme ou
    nul sans champ) et magnetique via la constante @p omega_c (q B / m, cuite au codegen).
    False (defaut) : aucun bloc source emis -- partie flux strictement identique au modele
    sans sources (valide par les goldens ADC-82).

    @p debye : longueur de Debye ADIMENSIONNEE lambda (None = pas de couplage Poisson). Emet
    elliptic_rhs((M00 - rho_background) / lambda^2) : ADC resout Delta(phi) = rhs SANS deflater
    la moyenne en periodique (un rhs a moyenne non nulle rend le systeme singulier : l'iteration
    MG derive, verifie experimentalement -- damier de Nyquist + re-solve divergent). Le fond
    NEUTRALISANT @p rho_background doit donc valoir la moyenne de M00 sur le domaine : la masse
    etant conservee, c'est une CONSTANTE du scenario, strictement equivalente a la soustraction
    de moyenne par pas du MATLAB poisson_fft. Le 'signe electron' du MATLAB est porte par
    q_over_m = +1 dans la source avec E = -grad phi (electric_source_term.m). Verifie par
    l'oracle sinusoidal analytique de run_diocotron.py (signe lu sur l'assert qui passe).

    @p omega_p : frequence locale de la SOURCE (m.source_frequency) bornant le pas de temps
    source (None = pas de borne) -- la 'deuxieme CFL' du MATLAB compute_dt, portee par la
    frequence plasma omega_p = 1/lambda.

    @p exact_speeds : True = vitesses d'onde signees EXACTES par valeurs propres du jacobien de
    flux (m.wave_speeds_from_jacobian : AUTODIFF du flux declare + blocs HYQMOM_BLOCKS, miroir
    du chemin flagsym=1 du MATLAB) -- ouvre riemann='hll' fidele et REMPLACE la borne bring-up
    k*sqrt(C) par la verite spectrale pour Rusanov/CFL aussi (max_wave_speed sur les memes
    blocs). Exige adc_cpp >= ADC-87. False (defaut) : borne bring-up historique (eigenvalues
    k*sqrt(C), formules identiques a l'ancien modele manuel, planchers compris en robust).
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
