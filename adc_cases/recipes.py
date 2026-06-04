"""Recettes SYSTEME : configurations multi-especes pretes a l'emploi.

Un niveau au-dessus des modeles d'espece (`adc_cases.models`). La ou un modele renvoie un seul
`adc.Model` (une espece), une recette configure un `sim` COMPLET : plusieurs blocs, le Poisson de
systeme et les couplages inter-especes (ionisation, collision). C'est ici que vit la composition
d'un scenario multi-especes, separee des briques d'espece.

Usage (depuis un cas) :
    import adc, adc_cases.recipes as recipes
    sim = adc.System(n=48, L=1.0, periodic=True)
    recipes.plasma(sim, ne=ne, ni=ni, ng=ng)   # cable blocs + Poisson + couplages, renvoie sim
"""

import adc

from . import models


def two_fluid(sim, ne, ni, qe=-1.0, qi=1.0, gamma=1.4, cs2=1.0):
    """Electrons (Euler) + ions (isothermes) couples par un Poisson de systeme (f = q_e n_e +
    q_i n_i). Configure `sim` (deux blocs + Poisson + densites) et le renvoie."""
    sim.add_block("electrons", models.electron_euler(charge=qe, gamma=gamma),
                  spatial=adc.Spatial(vanleer=True, flux="hllc"))
    sim.add_block("ions", models.ion_isothermal(charge=qi, cs2=cs2),
                  spatial=adc.Spatial(minmod=True))
    sim.set_poisson()
    sim.set_density("electrons", ne)
    sim.set_density("ions", ni)
    return sim


def plasma(sim, ne, ni, ng, qe=-1.0, qi=1.0, gamma=5.0 / 3.0, cs2=1.0,
           ionization_rate=0.3, collision_rate=0.5):
    """Plasma a trois especes : electrons (Euler, HLLC + reconstruction primitive), ions et neutres
    (isothermes). Couplage par Poisson (f = q_e n_e + q_i n_i) + ionisation (n_g -> n_i + n_e) +
    collision ion-neutre. Configure `sim` entierement (blocs + couplages + densites) et le renvoie."""
    sim.add_block("electrons", models.electron_euler(charge=qe, gamma=gamma),
                  spatial=adc.Spatial(vanleer=True, flux="hllc", recon="primitive"))
    sim.add_block("ions", models.ion_isothermal(charge=qi, cs2=cs2),
                  spatial=adc.Spatial(minmod=True))
    sim.add_block("neutrals", models.neutral_isothermal(cs2=cs2), spatial=adc.Spatial(minmod=True))
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
