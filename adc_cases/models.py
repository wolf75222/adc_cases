"""Modeles d'espece nommes = compositions de briques generiques de `adc`.

C'est ici (cote application) que vivent les noms d'espece. adc_cpp ne connait que des briques
(ExB, CompressibleFlux, PotentialForce, ChargeDensity, BackgroundDensity...) ; un modele est une
composition `adc.Model(state, transport, source, elliptic)`, soit une espece.

Le niveau au-dessus (configurer un `sim` multi-especes : plusieurs blocs + Poisson + couplages)
vit dans `adc_cases.recipes` (two_fluid, plasma), pas ici.

Usage (depuis un cas) :
    import adc_cases.models as models
    sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=0.0), ...)
"""

from __future__ import annotations

import adc


def diocotron(
    B0: float = 1.0, alpha: float = 1.0, n_i0: float = 0.0
) -> adc.Model:
    """Derive E x B d'une densite scalaire, fond neutralisant n_i0 (Poisson alpha (n - n0))."""
    return adc.Model(
        state=adc.Scalar(),
        transport=adc.ExB(B0=B0),
        source=adc.NoSource(),
        elliptic=adc.BackgroundDensity(alpha=alpha, n0=n_i0),
    )


def electron_euler(charge: float = -1.0, gamma: float = 1.4) -> adc.Model:
    """Euler compressible + force electrostatique + densite de charge (electrons)."""
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.PotentialForce(charge=charge),
        elliptic=adc.ChargeDensity(charge=charge),
    )


def ion_isothermal(charge: float = 1.0, cs2: float = 0.5) -> adc.Model:
    """Euler isotherme + force electrostatique + densite de charge (ions)."""
    return adc.Model(
        state=adc.FluidState(kind="isothermal", cs2=cs2),
        transport=adc.IsothermalFlux(),
        source=adc.PotentialForce(charge=charge),
        elliptic=adc.ChargeDensity(charge=charge),
    )


def euler_poisson(
    sign: float = 1.0,
    gamma: float = 1.4,
    four_pi_G: float = 1.0,
    rho0: float = 1.0,
) -> adc.Model:
    """Euler compressible + champ self-consistant (sign +1 auto-gravite, -1 plasma)."""
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.GravityForce(),
        elliptic=adc.GravityCoupling(sign=sign, four_pi_G=four_pi_G, rho0=rho0),
    )


def euler(gamma: float = 1.4) -> adc.Model:
    """Euler compressible pur : un gaz isole, sans source ni couplage (f = 0).

    Sert aux cas multi-fluides non couples (meme flux pour chaque espece, seules
    les conditions initiales different).
    """
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.NoSource(),
        elliptic=adc.ChargeDensity(charge=0.0),
    )


def neutral_isothermal(cs2: float = 1.0) -> adc.Model:
    """Gaz neutre isotherme : pas de force (charge nulle, hors de Poisson).

    Sert d'espece de fond reactive (ionisation, collisions) dans un plasma.
    """
    return adc.Model(
        state=adc.FluidState(kind="isothermal", cs2=cs2),
        transport=adc.IsothermalFlux(),
        source=adc.NoSource(),
        elliptic=adc.ChargeDensity(charge=0.0),
    )
