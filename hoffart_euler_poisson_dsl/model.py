"""Magnetic isothermal Euler-Poisson model from Hoffart et al. (arXiv:2510.11808).

Note : ce module definit le modele symbolique COMPLET (rho, rho*u, rho*v + flux +
source electrostatique/Lorentz). La validation de normalisation du taux de
croissance (NORMALIZATION.md, diag/diag_polar_omega.py) porte sur un modele
DIFFERENT et REDUIT (derive ExB scalaire, sans quantite de mouvement). Aucune
reproduction quantitative du modele complet ci-dessous n'est encore etablie.

The physical model is written once with ``adc.dsl.Model``.  Two source variants are
provided:

``schur``
    The local DSL source is zero.  The electrostatic/Lorentz source is advanced by
    ``adc.CondensedSchur`` in ``adc.System``.

``local``
    The full source is emitted in the generated C++ model.  This is the variant used
    by ``adc.AmrSystem`` with its cell-local IMEX source step.

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
# utilise le modele ExB scalaire REDUIT, pas le systeme complet defini ici ; seul l=4 colle.
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

    In two dimensions, with ``Omega = omega e_z``,
    ``m x Omega = (omega*m_y, -omega*m_x)``.
    """
    params = params or PaperParameters()
    if source not in ("schur", "local"):
        raise ValueError("source must be 'schur' or 'local'")

    m = dsl.Model("hoffart_magnetic_euler_poisson_%s" % source)
    rho, mx, my = m.conservative_vars(
        "rho", "rho_u", "rho_v",
        roles=["Density", "MomentumX", "MomentumY"],
    )

    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    pressure = m.primitive("p", params.temperature * rho)
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])

    m.flux(
        x=[mx, mx * u + pressure, mx * v],
        y=[my, my * u, my * v + pressure],
    )
    sound_speed = dsl.sqrt(params.temperature)
    m.eigenvalues(
        x=[u - sound_speed, u, u + sound_speed],
        y=[v - sound_speed, v, v + sound_speed],
    )

    m.aux("phi")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    if source == "local":
        omega = m.param("omega", params.omega)
        m.source([
            0.0 * rho,
            -rho * grad_x + omega * my,
            -rho * grad_y - omega * mx,
        ])
    else:
        # CondensedSchur owns the complete electrostatic/Lorentz source stage.
        # Keeping a non-zero local source here would advance it twice.
        m.source([0.0 * rho, 0.0 * mx, 0.0 * my])

    alpha = m.param("alpha", params.alpha)
    # ADC solves Delta(phi) = rhs.  The paper uses -Delta(phi) = alpha*rho.
    m.elliptic_rhs(-alpha * rho)
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
    r"""Initial drift velocity ``-(grad(phi) x Omega)/|Omega|^2``."""
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

