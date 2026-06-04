"""Modeles nommes = compositions de briques generiques de `adc`.

C'est ICI (cote application) que vivent les noms de scenarios. adc_cpp ne connait que des
briques (ExB, CompressibleFlux, PotentialForce, ChargeDensity, BackgroundDensity...) ; un
modele est une composition `adc.Model(state, transport, source, elliptic)`.

Usage (depuis un cas) :
    import adc_cases
    models = adc_cases.bootstrap()   # met le depot sur le chemin d'import, renvoie ce module
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


def euler(gamma=1.4):
    """Euler compressible PUR : un gaz isole, sans source ni couplage (f = 0). Sert aux cas
    multi-fluides NON couples (meme flux pour chaque espece, seules les CI different)."""
    return adc.Model(
        state=adc.FluidState(kind="compressible", gamma=gamma),
        transport=adc.CompressibleFlux(),
        source=adc.NoSource(),
        elliptic=adc.ChargeDensity(charge=0.0),
    )


def neutral_isothermal(cs2=1.0):
    """Gaz neutre isotherme : pas de force (charge nulle, n'entre pas dans Poisson). Sert d'espece
    de fond reactive (ionisation, collisions) dans un plasma."""
    return adc.Model(
        state=adc.FluidState(kind="isothermal", cs2=cs2),
        transport=adc.IsothermalFlux(),
        source=adc.NoSource(),
        elliptic=adc.ChargeDensity(charge=0.0),
    )


# ---------------------------------------------------------------------------
# Recettes SYSTEME : un niveau au-dessus des modeles d'espece. Une recette configure un `sim`
# complet (plusieurs blocs + Poisson + couplages inter-especes) au lieu de renvoyer un seul
# adc.Model. C'est ici que vit la composition d'un scenario multi-especes.
# ---------------------------------------------------------------------------
def two_fluid(sim, ne, ni, qe=-1.0, qi=1.0, gamma=1.4, cs2=1.0):
    """Electrons (Euler) + ions (isothermes) couples par un Poisson de systeme (f = q_e n_e +
    q_i n_i). Configure `sim` (deux blocs + Poisson + densites) et le renvoie."""
    sim.add_block("electrons", electron_euler(charge=qe, gamma=gamma),
                  spatial=adc.Spatial(vanleer=True, flux="hllc"))
    sim.add_block("ions", ion_isothermal(charge=qi, cs2=cs2), spatial=adc.Spatial(minmod=True))
    sim.set_poisson()
    sim.set_density("electrons", ne)
    sim.set_density("ions", ni)
    return sim


def plasma(sim, ne, ni, ng, qe=-1.0, qi=1.0, gamma=5.0 / 3.0, cs2=1.0,
           ionization_rate=0.3, collision_rate=0.5):
    """Plasma a trois especes : electrons (Euler, HLLC + reconstruction primitive), ions et neutres
    (isothermes). Couplage par Poisson (f = q_e n_e + q_i n_i) + ionisation (n_g -> n_i + n_e) +
    collision ion-neutre. Configure `sim` entierement (blocs + couplages + densites) et le renvoie."""
    sim.add_block("electrons", electron_euler(charge=qe, gamma=gamma),
                  spatial=adc.Spatial(vanleer=True, flux="hllc", recon="primitive"))
    sim.add_block("ions", ion_isothermal(charge=qi, cs2=cs2), spatial=adc.Spatial(minmod=True))
    sim.add_block("neutrals", neutral_isothermal(cs2=cs2), spatial=adc.Spatial(minmod=True))
    sim.set_poisson()
    if ionization_rate:
        sim.add_ionization(electron="electrons", ion="ions", neutral="neutrals",
                           rate=ionization_rate)
    if collision_rate:
        sim.add_collision("ions", "neutrals", rate=collision_rate)
    sim.set_density("electrons", ne)
    sim.set_density("ions", ni)
    sim.set_density("neutrals", ng)
    return sim
