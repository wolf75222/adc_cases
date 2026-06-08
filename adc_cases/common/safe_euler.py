"""Le CAS SUR de la campagne de perf : Euler compressible PUR, periodique, bulle de pression lisse.

UNIQUE source de verite Python du cas sur (constantes, dt, CI, modeles briques & DSL). Importe par
`perf/frontend_compare.py` (mesure 3 fronts) ET `safe_euler_periodic/run.py` (validation/CI). Le
pendant C++ est `adc_cpp/bench/frontend_cpp.cpp` (namespace `safecase`) : les CONSTANTES et le
schema numerique (minmod / rusanov / reconstruction conservative / SSPRK2 / dt FIXE) DOIVENT y
coincider bit-a-bit, sinon l'identite numerique inter-fronts tombe.

Choix physiques : rho uniforme (rho>0 garanti), bulle de pression de faible amplitude (p>0
garanti), vitesse nulle a t=0 ; domaine periodique ; AUCUNE source, AUCUN couplage Poisson (transport
pur) -- le toggle Poisson de la campagne ajoute un solve elliptique inerte (charge=0) en option.

IMPORTANT : ce module n'importe PAS `adc` au niveau module. Les fabriques `bricks_model` / `dsl_model`
importent `adc` PARESSEUSEMENT, pour que `frontend_compare.py` puisse chronometrer un `import adc`
reellement FROID dans son worker.
"""

L = 1.0
GAMMA = 1.4
RHO0 = 1.0
P0 = 1.0
DP = 0.1
SIGMA2 = 0.02
CFL_FOR_DT = 0.4

# Reglages numeriques PINNES, identiques aux trois fronts (et a frontend_cpp.cpp).
LIMITER = "minmod"
FLUX = "rusanov"
RECON = "conservative"
TIMEINT = "ssprk2"
WORKLOAD = "euler_safe"


def dt(n):
    """dt FIXE deterministe : cfl * (L/n) / wmax, wmax = vitesse du son au pic de pression (v=0).

    Fixe (pas d'adaptatif CFL) pour que les trois fronts integrent EXACTEMENT le meme dt -> pas de
    boucle de retroaction dt qui briserait l'identite numerique."""
    wmax = (GAMMA * (P0 + DP) / RHO0) ** 0.5
    return CFL_FOR_DT * (L / n) / wmax


def ic(n):
    """Etat conservatif initial (4, n, n) : reutilise euler_pressure_blob (rho=RHO0, v=0, bulle p)."""
    from .initial_conditions import euler_pressure_blob
    return euler_pressure_blob(n, L=L, rho0=RHO0, p0=P0, dp=DP, sigma2=SIGMA2, gamma=GAMMA)


def bricks_model():
    """Front BRIQUES : Euler compressible pur (models.euler) = Model(FluidState, CompressibleFlux,
    NoSource, ChargeDensity(0))."""
    import adc_cases.models as models
    return models.euler(GAMMA)


def dsl_model():
    """Front DSL : Euler compressible pur ecrit en formules (adc.dsl.Model), SANS source ni
    elliptic_rhs (transport pur). MEME convention que la brique Euler du coeur (euler.hpp)."""
    from adc import dsl
    m = dsl.Model("safe_euler")
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    g = m.param("gamma", GAMMA)
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    c = dsl.sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, (E + p) * u],
           y=[rhov, rhov * u, rhov * v + p, (E + p) * v])
    m.eigenvalues(x=[u - c, u, u, u + c], y=[v - c, v, v, v + c])
    m.primitive_vars(rho, u, v, p)
    m.conservative_from([rho, rho * u, rho * v, p / (g - 1.0) + 0.5 * rho * (u * u + v * v)])
    m.check()
    return m


def spatial_bricks():
    """Schema spatial des briques (Spatial minmod / rusanov / conservative)."""
    import adc
    return adc.Spatial(limiter=LIMITER, flux=FLUX, recon=RECON)


def spatial_dsl():
    """Schema spatial du DSL (FiniteVolume -> meme Spatial minmod / rusanov / conservative)."""
    import adc
    return adc.FiniteVolume(limiter=LIMITER, riemann=FLUX, variables=RECON)


def pressure(U):
    """Pression d'un etat conservatif (4, n, n) : (gamma-1)(E - 1/2 (mx^2+my^2)/rho)."""
    return (GAMMA - 1.0) * (U[3] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0])
