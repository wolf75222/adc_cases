"""Modeles nommes = compositions de briques generiques de `adc`.

C'est ICI (cote application) que vivent les noms de scenarios. adc_cpp ne connait que des
briques (ExB, CompressibleFlux, PotentialForce, ChargeDensity, BackgroundDensity...) ; un
modele est une composition `adc.Model(state, transport, source, elliptic)`.

Usage (depuis un cas) :
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import models
    sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=0.0), ...)
"""

import adc


def diocotron(B0=1.0, alpha=1.0, n_i0=0.0):
    """Derive E x B d'une densite scalaire, fond neutralisant n_i0 (Poisson alpha (n - n0))."""
    return adc.Model(
        state=adc.Scalar(),
        transport=adc.ExB(B0=B0),
        source=adc.NoSource(),
        elliptic=adc.BackgroundDensity(alpha=alpha, n0=n_i0),
    )


def electron_euler(charge=-1.0, gamma=1.4):
    """Euler compressible + force electrostatique + densite de charge (electrons)."""
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.PotentialForce(charge=charge),
        elliptic=adc.ChargeDensity(charge=charge),
    )


def ion_isothermal(charge=1.0, cs2=0.5):
    """Euler isotherme + force electrostatique + densite de charge (ions)."""
    return adc.Model(
        state=adc.FluidState(kind="isothermal", cs2=cs2),
        transport=adc.IsothermalFlux(),
        source=adc.PotentialForce(charge=charge),
        elliptic=adc.ChargeDensity(charge=charge),
    )


def euler_poisson(sign=1.0, gamma=1.4, four_pi_G=1.0, rho0=1.0):
    """Euler compressible + champ self-consistant. sign = +1 auto-gravite, -1 plasma."""
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.GravityForce(),
        elliptic=adc.GravityCoupling(sign=sign, four_pi_G=four_pi_G, rho0=rho0),
    )
