"""Recettes systeme : configurations multi-especes pretes a l'emploi.

Un niveau au-dessus des modeles d'espece (`adc_cases.models`). La ou un modele renvoie un seul
`adc.Model` (une espece), une recette configure un `sim` complet : plusieurs blocs, le Poisson de
systeme et les couplages inter-especes (ionisation, collision). C'est ici que vit la composition
d'un scenario multi-especes, separee des briques d'espece.

Usage (depuis un cas) :
    import adc, adc_cases.recipes as recipes
    sim = adc.System(n=48, L=1.0, periodic=True)
    recipes.plasma(sim, ne=ne, ni=ni, ng=ng)   # cable blocs + Poisson + couplages, renvoie sim
"""

from __future__ import annotations

import adc

from . import models


def two_fluid(
    sim: adc.System,
    ne,
    ni,
    qe: float = -1.0,
    qi: float = 1.0,
    gamma: float = 1.4,
    cs2: float = 1.0,
) -> adc.System:
    """Electrons (Euler) + ions (isothermes) couples par un Poisson de systeme.

    Le couplage est f = q_e n_e + q_i n_i.

    Args:
        sim: Systeme `adc` a configurer (modifie en place).
        ne: Densite initiale des electrons.
        ni: Densite initiale des ions.
        qe: Charge des electrons.
        qi: Charge des ions.
        gamma: Indice adiabatique des electrons (Euler compressible).
        cs2: Vitesse du son au carre des ions (isothermes).

    Returns:
        Le `sim` configure (deux blocs + Poisson + densites).
    """
    sim.add_block(
        "electrons",
        models.electron_euler(charge=qe, gamma=gamma),
        spatial=adc.Spatial(vanleer=True, flux="hllc"),
    )
    sim.add_block(
        "ions",
        models.ion_isothermal(charge=qi, cs2=cs2),
        spatial=adc.Spatial(minmod=True),
    )
    sim.set_poisson()
    sim.set_density("electrons", ne)
    sim.set_density("ions", ni)
    return sim


def plasma(
    sim: adc.System,
    ne,
    ni,
    ng,
    qe: float = -1.0,
    qi: float = 1.0,
    gamma: float = 5.0 / 3.0,
    cs2: float = 1.0,
    ionization_rate: float = 0.3,
    collision_rate: float = 0.5,
) -> adc.System:
    """Plasma a trois especes : electrons (Euler, HLLC + recon primitive), ions, neutres.

    Ions et neutres sont isothermes. Couplage par Poisson (f = q_e n_e + q_i n_i)
    + ionisation (n_g -> n_i + n_e) + collision ion-neutre. L'ionisation et la
    collision ne sont cablees que si leur taux est non nul.

    Args:
        sim: Systeme `adc` a configurer (modifie en place).
        ne: Densite initiale des electrons.
        ni: Densite initiale des ions.
        ng: Densite initiale des neutres.
        qe: Charge des electrons.
        qi: Charge des ions.
        gamma: Indice adiabatique des electrons (Euler compressible).
        cs2: Vitesse du son au carre des ions et neutres (isothermes).
        ionization_rate: Taux d'ionisation n_g -> n_i + n_e (0 = desactive).
        collision_rate: Taux de collision ion-neutre (0 = desactive).

    Returns:
        Le `sim` configure (blocs + couplages + densites).
    """
    sim.add_block(
        "electrons",
        models.electron_euler(charge=qe, gamma=gamma),
        spatial=adc.Spatial(vanleer=True, flux="hllc", recon="primitive"),
    )
    sim.add_block(
        "ions",
        models.ion_isothermal(charge=qi, cs2=cs2),
        spatial=adc.Spatial(minmod=True),
    )
    sim.add_block(
        "neutrals",
        models.neutral_isothermal(cs2=cs2),
        spatial=adc.Spatial(minmod=True),
    )
    sim.set_poisson()
    if ionization_rate:
        sim.add_ionization(
            electron="electrons",
            ion="ions",
            neutral="neutrals",
            rate=ionization_rate,
        )
    if collision_rate:
        sim.add_collision("ions", "neutrals", rate=collision_rate)
    sim.set_density("electrons", ne)
    sim.set_density("ions", ni)
    sim.set_density("neutrals", ng)
    return sim
