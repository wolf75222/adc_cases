"""Magnetic isothermal Euler-Poisson model from Hoffart et al. (arXiv:2510.11808).

Note : ce module definit le modele symbolique complet (rho, rho*u, rho*v + flux +
source electrostatique/Lorentz). La validation de normalisation du taux de
croissance (docs/NORMALIZATION.md, diag/diag_polar_omega.py) porte sur un modele
different et reduit (derive ExB scalaire, sans quantite de mouvement). Aucune
reproduction quantitative du modele complet ci-dessous n'est encore etablie.

The physical model is written once with `adc.dsl.Model`.  Two source variants are
provided:

`schur`
    The local DSL source is zero.  The electrostatic/Lorentz source is advanced by
    `adc.CondensedSchur` in `adc.System`.

`local`
    The full source is emitted in the generated C++ model.  This is the variant used
    by `adc.AmrSystem` with its cell-local IMEX source step.

Both variants use the same conservative variables, physical flux, eigenvalues and
Poisson right-hand side.
"""

from dataclasses import asdict, dataclass

import numpy as np

from adc import dsl


@dataclass(frozen=True)
class PaperParameters:
    """Physical and geometrical parameters of section 5.3 of the paper."""

    radius: float = 16.0
    ring_inner: float = 6.0
    ring_outer: float = 8.0
    rho_min: float = 1.0e-6
    rho_max: float = 1.0
    beta: float = 1.0e6
    perturbation: float = 0.1
    final_time: float = 10.0
    temperature: float = 0.0

    @property
    def length(self):
        return 2.0 * self.radius

    @property
    def alpha(self):
        return self.beta * self.beta / self.rho_max

    @property
    def omega(self):
        return self.beta * self.beta

    def to_dict(self):
        out = asdict(self)
        out.update(length=self.length, alpha=self.alpha, omega=self.omega)
        return out


# Cibles du papier (Section 5.3). NB : la comparaison de normalisation (diag/diag_polar_omega.py)
# utilise le modele ExB scalaire reduit, pas le systeme complet defini ici ; seul l=4 colle.
PAPER_GROWTH_RATES = {3: 0.772, 4: 0.911, 5: 0.683}
PAPER_FIT_WINDOWS = {3: (0.40, 0.70), 4: (0.60, 0.75), 5: (1.15, 1.35)}
PAPER_SNAPSHOT_FRACTIONS = (0.01, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0)


def magnetic_euler_poisson_model(params=None, source="schur"):
    r"""Return the paper model as symbolic formulas.

    .. math::

       \partial_t \rho + \nabla\cdot m = 0,

       \partial_t m + \nabla\cdot(m m^T/\rho + pI)
       = -\rho\nabla\phi + m\times\Omega,

       -\Delta\phi = \alpha\rho,\qquad p=\theta\rho.

    In two dimensions, with `Omega = omega e_z`,
    `m x Omega = (omega*m_y, -omega*m_x)`.
    """
    params = params or PaperParameters()
    if source not in ("schur", "local"):
        raise ValueError("source must be 'schur' or 'local'")

    # Tout le modele s'ecrit ici en SYMBOLES, une seule fois. adc.dsl en derive le solveur de
    # Riemann, genere le noyau C++ device-ready et le compile : on n'ecrit jamais de boucle ni de
    # flux numerique a la main. Le reste de la fonction ne fait que poser les formules du papier.
    m = dsl.Model("hoffart_magnetic_euler_poisson_%s" % source)

    # 1. Les inconnues CONSERVATIVES, celles que le schema fait avancer : densite et les deux
    #    composantes de la quantite de mouvement m = rho*u. Pas d'energie (modele barotrope,
    #    fermeture isotherme p = theta*rho). Le role indique au moteur le sens physique du champ.
    rho, mx, my = m.conservative_vars(
        "rho", "rho_u", "rho_v",
        roles=["Density", "MomentumX", "MomentumY"],
    )

    # 2. Les variables PRIMITIVES, definies a partir des conservatives. On ecrit la relation
    #    physique (u = m_x/rho, p = theta*rho) ; le DSL gere les deux conversions prim<->cons.
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    pressure = m.primitive("p", params.temperature * rho)
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])

    # 3. Le FLUX hyperbolique d'Euler, composante par composante, comme au tableau :
    #      d_t rho      + div(m)             = 0       -> flux de masse = m
    #      d_t (rho u)  + div(rho u u + p)   = source  -> flux de qdm   = m*u + p (et m*v en y)
    #    On donne les colonnes x et y ; le DSL en fait un flux numerique conservatif.
    m.flux(
        x=[mx, mx * u + pressure, mx * v],
        y=[my, my * u, my * v + pressure],
    )

    # 4. Les VALEURS PROPRES (vitesses d'onde u, u +/- c avec c = sqrt(theta)) : le solveur de
    #    Riemann (Rusanov) s'en sert pour la dissipation. En limite froide theta=0 -> c=0 et les
    #    trois valeurs propres degenerent en u (advection pure).
    sound_speed = dsl.sqrt(params.temperature)
    m.eigenvalues(
        x=[u - sound_speed, u, u + sound_speed],
        y=[v - sound_speed, v, v + sound_speed],
    )

    # 5. Les champs AUXILIAIRES : le potentiel phi et son gradient. Ils ne sont pas avances par le
    #    flux ; le solveur de champ (Poisson) les remplit a chaque pas et le modele y a acces.
    m.aux("phi")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    # 6. La SOURCE : force electrique -rho*grad(phi) + force de Lorentz m x Omega = (omega*m_y, -omega*m_x).
    #    Deux variantes du MEME modele :
    #      - 'local' : la source est emise dans le noyau C++ (chemin AMR, etage IMEX cell-local) ;
    #      - 'schur' : source nulle ici, l'etage CondensedSchur l'avance implicitement (chemin de
    #        reference). La laisser non nulle l'avancerait deux fois.
    if source == "local":
        omega = m.param("omega", params.omega)
        m.source([
            0.0 * rho,
            -rho * grad_x + omega * my,
            -rho * grad_y - omega * mx,
        ])
    else:
        m.source([0.0 * rho, 0.0 * mx, 0.0 * my])

    # 7. La loi de GAUSS qui ferme le couplage : -Delta phi = alpha*rho. Le solveur resout
    #    Delta(phi) = rhs, d'ou le signe negatif. alpha = beta^2/rho_max est le couplage du papier.
    alpha = m.param("alpha", params.alpha)
    m.elliptic_rhs(-alpha * rho)

    # 8. check() valide la coherence (roles, dimensions, flux/source). check_model.py compare ensuite
    #    le noyau COMPILE aux formules a la main sur 2x2 cellules : residu exactement nul.
    m.check()
    return m


def paper_initial_density(n, mode, params=None):
    """Cell-centred annular density from equation (35) of the paper."""
    params = params or PaperParameters()
    if mode not in PAPER_GROWTH_RATES:
        raise ValueError("paper modes are 3, 4 and 5")

    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    radius = np.hypot(X, Y)
    angle = np.arctan2(Y, X)

    rho = np.full((n, n), params.rho_min, dtype=np.float64)
    ring = (radius >= params.ring_inner) & (radius <= params.ring_outer)
    rho[ring] = params.rho_max * (
        1.0 - params.perturbation
        + params.perturbation * np.sin(mode * angle[ring])
    )
    return rho


def drift_velocity_from_potential(phi, params=None):
    r"""Initial drift velocity `-(grad(phi) x Omega)/|Omega|^2`."""
    params = params or PaperParameters()
    h = params.length / phi.shape[0]
    grad_y, grad_x = np.gradient(phi, h, h, edge_order=2)
    u = -grad_y / params.omega
    v = grad_x / params.omega

    x = (np.arange(phi.shape[0]) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    outside = np.hypot(X, Y) > params.radius
    u[outside] = 0.0
    v[outside] = 0.0
    return u, v

