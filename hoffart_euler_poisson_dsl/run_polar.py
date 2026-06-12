#!/usr/bin/env python3
"""Modele COMPLET Euler-Poisson isotherme magnetise sur GRILLE POLAIRE (anneau resolu).

Pendant POLAIRE du chemin cartesien ``run.py`` (engine ``system-schur``). Au lieu du
carre cartesien plein, le transport tourne sur un ANNEAU (r, theta) : la direction
RADIALE est portee par un axe de grille, ce qui leve le verrou des bords d'anneau
cartesiens (cf. docs/HOFFART_GEOMETRY_VERDICT.md : le chemin cartesien plafonne a
-82/-95% INDEPENDAMMENT de la resolution). Le diagnostic reduit ExB scalaire polaire
(diag/diag_polar_omega.py) recupere deja l=4 EXACT ; ce runner porte le MODELE COMPLET
(rho / rho v_r / rho v_theta + flux isotherme polaire + courbure + etage Schur condense
+ Lorentz) sur la MEME grille polaire.

Assemblage (briques 100% NATIVES, aucune compilation .so, CI-safe, MONO-RANG) :

  1. adc.System(mesh=adc.PolarMesh(r_min, r_max, nr, ntheta))   # anneau global
  2. set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")  # FFT-theta + Thomas-r
  3. set_magnetic_field(B0 * ones((ntheta, nr)))                 # Omega = B_z, AVANT le Schur
  4. adc.Model(state=FluidState(isothermal, cs2), transport=IsothermalFlux(),
              source=NoSource(),                                 # source = etage Schur (pas locale)
              elliptic=BackgroundDensity(alpha) ou ChargeDensity)
  5. add_equation(Split(hyperbolic=Explicit(ssprk3), source=CondensedSchur(theta, alpha)))
     -> dispatch polaire : IsothermalFluxPolar (#209) + PolarCondensedSchurSourceStepper (#215)
  6. init : top-hat annulaire + perturbation sin(l*theta) + EQUILIBRE ROTATIF (v_r=0, v_theta(r)
     racine du bilan radial centrifuge/pression/electrique/Lorentz) du PREMIER solve Poisson.
     L'ancienne derive ExB pure (v_theta=grad_r/B) n'etait PAS un equilibre du fluide complet : elle
     portait une quantite de mouvement non equilibree -> transient radial rapide -> NaN avant la
     fenetre de fit. v_theta(r) = (r/2)[-B + sqrt(B^2 + (4/r)(d_r phi + cs2 (d_r rho)/rho))] se reduit
     continument a la derive ExB grad_r/B quand la courbure -> 0. cs2 petit NON nul (defaut 1e-4) pour
     la stricte hyperbolicite (le papier autorise theta>=0).
  6bis. BILAN DISCRET BIEN POSE -- OPTION (c), --frozen-equilibrium (defaut ON) : l'IC d'equilibre
     rotatif (et la Newton #23) ne suffisent PAS a la raideur du papier (B_z=omega=1e12) : l'etage
     hyperbolique (WENO5+Rusanov au saut top-hat) et l'etage Schur n'annulent le bilan electrique/
     Lorentz QU'AU CONTINU, jamais au DISCRET -> derive parasite O(1) axisymetrique -> NaN ~ t 0.02.
     On PRECALCULE le residu d'equilibre GELE R_eq = step(U_eq) - U_eq UNE FOIS sur l'anneau
     axisymetrique (perturbation=0), puis on SOUSTRAIT R_eq a chaque pas (U <- step(U) - R_eq) :
     (step - R_eq)(U_eq) = U_eq est un POINT FIXE DISCRET EXACT a la precision machine, INDEPENDAMMENT
     du stencil ; la derive O(1) est annulee, seule la perturbation O(delta) (mode physique + petit
     residu O(delta)) evolue. PAS FIXE OBLIGATOIRE (le --cfl adaptatif s'effondre sur l'equilibre
     quasi-stationnaire -> dt explose -> NaN : il est ignore en mode frozen). Auto-test :
     --check-equilibrium (sous frozen, exige la STATIONARITE A LA PRECISION MACHINE de U_eq).
  7. observable : phi sur le cercle r=r0 (= colonne phi[:, i_r0], NATIVE, sans interpolation),
     FFT en theta, fenetres de fit VERBATIM du papier -> growth_rates.csv (BRUT, matplotlib optionnel).

Normalisation (T3 -- corrige) : on reporte gamma_raw_sim (pente brute, fenetre MAPPEE) ET
gamma_paper_units = gamma_raw_sim * 2pi/rhobar. Le facteur 2 pi est la conversion cyclique->
angulaire de l'horloge de derive et s'applique AU MODELE COMPLET aussi (alpha/|Omega|=1/rho_max=1
-> meme champ de derive que le reduit ExB) ; l'ancienne premisse "pente brute directement
comparable, aucun facteur" etait INCORRECTE (cf. docs/T2_NORMALIZATION_AUDIT.md). ATTENTION : la repro
quantitative du polaire COMPLET n'est PAS etablie (VOIE 1 diverge, non-positivite au bord d'anneau) ;
ce mapping rend seulement la metrologie coherente avec le cartesien, il ne valide pas le polaire.

GAP CONNU (reporte, non bloquant) : derive_aux_polar remplit l'aux (phi, grad_r, grad_theta)
mais AUCUN accesseur Python ne rend ces gradients. La derive ExB initiale est donc RECALCULEE
en Python a partir de phi avec EXACTEMENT le stencil du moteur (centre a l'interieur, decentre
d'ordre 2 aux parois radiales, enroulement periodique en theta ; grad_theta = (1/r) d phi/d theta).
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adc_cases.common.io import case_output_dir  # noqa: E402
from model import (  # noqa: E402
    PAPER_FIT_WINDOWS,
    PAPER_GROWTH_RATES,
    PaperParameters,
)
from results import (  # noqa: E402
    adc_cases_sha,
    adc_cpp_sha,
    build_record,
    engine_label,
    gamma_to_paper_units,
    paper_to_sim_time_window,
    verify_paper_windows,
    write_records,
)

# --- Geometrie de l'anneau (echelle papier 6:8:16, cf. diag/diag_polar_omega.py) -----------------
# L'anneau de densite vit dans [R0, R1] ; le domaine polaire va de RMIN (> 0, evite la singularite
# r=0 du solveur annulaire) a RW (= mur conducteur R). r < R0 et r > R1 portent rho_min (fond inerte).
DEFAULT_RMIN = 2.0


def i_radial(radius, r_min, dr, nr):
    """Indice radial de la cellule dont le CENTRE est le plus proche de @p radius.

    Centres : r_cell(i) = r_min + (i + 0.5) * dr (convention PolarGeometry::r_cell, mesh/geometry.hpp).
    """
    return max(0, min(nr - 1, int(round((radius - r_min) / dr - 0.5))))


def annular_density(nr, nth, mode, params, r_min):
    """Densite top-hat annulaire + perturbation azimutale sin(l*theta) (eq. (35) du papier).

    Layout polaire attendu par set_density : axe lent = theta (j), axe rapide = r (i), flat[j*nr+i].
    Renvoie un tableau (nth, nr) ; l'appelant l'aplatit (.ravel(), C-order).
    """
    dr = (params.radius - r_min) / nr
    dth = 2.0 * math.pi / nth
    rho = np.full((nth, nr), params.rho_min, dtype=np.float64)
    r = r_min + (np.arange(nr) + 0.5) * dr                       # centres radiaux (axe rapide i)
    ring = (r >= params.ring_inner) & (r <= params.ring_outer)   # masque radial de l'anneau
    for j in range(nth):
        th = (j + 0.5) * dth
        dper = 1.0 - params.perturbation + params.perturbation * math.sin(mode * th)
        rho[j, ring] = params.rho_max * dper
    return rho


def polar_gradient(phi, r_min, dr, nth, nr):
    """Gradient polaire de phi en base locale, EXACTEMENT le stencil de derive_aux_polar (C++).

    phi : tableau (nth, nr) = phi[theta, r]. Renvoie (grad_r, grad_theta), memes formes :
      grad_r     = d phi/dr     : centre a l'interieur, DECENTRE d'ordre 2 aux deux parois radiales
                   (i=0 : avant -3,4,-1 ; i=nr-1 : arriere 3,-4,1), phi SANS ghost radial.
      grad_theta = (1/r) d phi/d theta : centre avec ENROULEMENT periodique de l'indice theta.
    Le decentrage d'ordre 2 exige nr >= 3 (garanti par adc.PolarMesh / check_geometry).
    """
    dth = 2.0 * math.pi / nth
    r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]            # (1, nr) centres radiaux

    grad_r = np.empty_like(phi)
    # interieur : centre (p(i+1) - p(i-1)) / (2 dr)
    grad_r[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dr)
    # paroi interne i=0 : decentre avant ordre 2 (-3 p0 + 4 p1 - p2) / (2 dr)
    grad_r[:, 0] = (-3.0 * phi[:, 0] + 4.0 * phi[:, 1] - phi[:, 2]) / (2.0 * dr)
    # paroi externe i=nr-1 : decentre arriere ordre 2 (3 pN - 4 pN-1 + pN-2) / (2 dr)
    grad_r[:, -1] = (3.0 * phi[:, -1] - 4.0 * phi[:, -2] + phi[:, -3]) / (2.0 * dr)

    # grad_theta = (1/r) (p(j+1) - p(j-1)) / (2 dth), theta periodique (np.roll = enroulement d'indice).
    grad_theta = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dth * r)
    return grad_r, grad_theta


def polar_radial_derivative(field, r_min, dr, nth, nr):
    """d field/dr en base locale, MEME stencil radial que polar_gradient (branche grad_r).

    Reutilise pour d_r rho dans le terme de pression de l'equilibre. field : (nth, nr).
    Centre a l'interieur, decentre d'ordre 2 aux deux parois radiales (i=0 et i=nr-1).
    """
    d_r = np.empty_like(field)
    d_r[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dr)
    d_r[:, 0] = (-3.0 * field[:, 0] + 4.0 * field[:, 1] - field[:, 2]) / (2.0 * dr)
    d_r[:, -1] = (3.0 * field[:, -1] - 4.0 * field[:, -2] + field[:, -3]) / (2.0 * dr)
    return d_r


def equilibrium_v_theta(rho, grad_r, r_min, dr, nth, nr, params, cs2):
    r"""v_theta(r) de l'EQUILIBRE ROTATIF axisymetrique (bilan radial de quantite de mouvement).

    Derivation E1 (verifiee dans la source adc_cpp ; cf. en-tete de ce fichier). A l'etat
    stationnaire (v_r = 0, d_t = 0, axisymetrique) le membre de droite RADIAL des DEUX etages du
    moteur s'annule :

      etage HYPERBOLIQUE (assemble_rhs_polar + IsothermalFluxPolar::polar_geom_source) :
        R_hyp[1] = rho v_theta^2 / r - d_r p          (le +p/r de S_geom annule p/r de la divergence)
      etage SOURCE (CondensedSchur / Lorentz, dv_r/dt = -(d_r phi) + B_z v_theta) :
        R_src[1] = -rho d_r phi + rho B_z v_theta

    R_hyp[1] + R_src[1] = 0, p = cs2 rho, d_r p = cs2 d_r rho, donne la quadratique par cellule :

      (rho/r) v_theta^2 + (rho B_z) v_theta - (d_r p + rho d_r phi) = 0,

    soit, apres division par rho > 0 (vrai aussi sur le fond rho_min) :

      (1/r) v_theta^2 + B_z v_theta - ( d_r phi + cs2 (d_r rho)/rho ) = 0.

    Racine PHYSIQUE (continuation ExB) = branche +sqrt. La forme naive
    v_theta = (r/2)[ -B + sqrt(B^2 + (4/r) forcing) ] souffre d'une ANNULATION CATASTROPHIQUE quand
    B = omega = beta^2 est gigantesque (1e12) : B^2 = 1e24 + O(1) perd le terme de forcing dans la
    mantisse float64, sqrt(B^2 + small) = B exactement, et la difference -B + B = 0 ANEANTIT la derive
    physique grad_r/B ~ 1e-12. On utilise donc la forme MULTIPLIEE-CONJUGUEE, numeriquement stable
    pour b = B > 0 (racine + de a v^2 + b v - c, a = 1/r, b = B, c = forcing) :

      v_theta(r) = 2 forcing / ( B + sqrt( B^2 + (4/r) forcing ) ),   forcing = d_r phi + cs2 (d_r rho)/rho.

    (Equivalente algebriquement a (r/2)[-B + sqrt(...)] mais sans la soustraction de deux grands
    quasi-egaux : le numerateur porte directement le forcing, le denominateur ~ 2B reste exact.)

    Quand la courbure -> 0 (cs2 -> 0 ou grand r), v_theta -> forcing/B -> d_r phi/B = grad_r/B, donc
    elle se reduit CONTINUMENT a la derive azimutale ExB existante. La branche -sqrt (racine rapide
    contre-tournante parasite) est ecartee. Tout est calcule par cellule avec EXACTEMENT le stencil
    radial du moteur (polar_gradient / polar_radial_derivative) -> coherence avec l'operateur discret.

    @param grad_r  d phi/dr deja calcule (meme stencil que derive_aux_polar).
    @return v_theta : tableau (nth, nr), composante PHYSIQUE azimutale (base locale e_theta).
    """
    B = params.omega
    r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]            # (1, nr) centres radiaux
    # terme de pression cs2 (d_r rho)/rho (saut net aux bords du top-hat : physique, pas lisse).
    if cs2 != 0.0:
        d_r_rho = polar_radial_derivative(rho, r_min, dr, nth, nr)
        pressure_term = cs2 * d_r_rho / rho
    else:
        pressure_term = 0.0
    forcing = grad_r + pressure_term                            # d_r phi + cs2 (d_r rho)/rho
    disc = B * B + (4.0 / r) * forcing                          # discriminant de la quadratique reduite
    # Garde-fou : disc >= 0 (vrai pour le bump phi positif a paroi Dirichlet ; en pratique disc ~ B^2
    # >> 0). Borne a 0 pour ne jamais propager un NaN si une cellule de bord devie de peu.
    disc = np.maximum(disc, 0.0)
    # Forme conjuguee stable (pas de -B + B) : 2 forcing / (B + sqrt(disc)). B = omega > 0 garanti.
    return 2.0 * forcing / (B + np.sqrt(disc))


def build_polar_system(nr, nth, mode, params, args):
    """Anneau polaire + bloc isotherme natif + Poisson polaire + B_z + etage Schur condense.

    Init : (A) densite top-hat (+ perturbation seed) ; (B) 1er solve Poisson -> phi ; (C) v_r = ExB
    (-grad_theta/B) ET v_theta = EQUILIBRE ROTATIF (racine de la quadratique de bilan radial,
    equilibrium_v_theta) -- ce v_theta est un VRAI etat stationnaire du modele fluide complet (il
    equilibre centrifuge + pression + electrique + Lorentz), contrairement a l'ancienne derive ExB
    pure grad_r/B qui portait une quantite de mouvement non equilibree et explosait avant la fenetre
    de fit ; (D) injecte l'etat (rho, rho v_r, rho v_theta) ; (E) 2e solve_fields pour que le
    transport lise un phi coherent avec l'etat.
    """
    r_min = args.r_min
    dr = (params.radius - r_min) / nr

    sim = adc.System(mesh=adc.PolarMesh(r_min=r_min, r_max=params.radius, nr=nr, ntheta=nth))
    # Poisson polaire : -Delta phi = alpha rho (le solveur porte la metrique r dr dtheta ; le
    # second membre par cellule vient de la brique elliptique du bloc). Dirichlet phi=0 aux parois
    # radiales, theta toujours periodique cote solveur polaire.
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    # B_z constant = omega (Omega = B_z). REQUIS avant set_source_stage (l'etage Lorentz lit Omega).
    # Layout (ntheta, nr) C-order = flat[j*nr+i], coherent avec set_density / set_state polaire.
    sim.set_magnetic_field(params.omega * np.ones((nth, nr), dtype=np.float64))

    model = adc.Model(
        state=adc.FluidState(kind="isothermal", cs2=args.cs2),
        transport=adc.IsothermalFlux(),                  # -> IsothermalFluxPolar (3 var + courbure)
        source=adc.NoSource(),                           # la source est l'etage Schur condense
        elliptic=adc.BackgroundDensity(alpha=params.alpha, n0=0.0),  # -alpha rho au second membre
    )
    # Politique de splitting : Lie (adc.Split, defaut) ou Strang (adc.Strang, 2e ordre, --strang).
    split_factory = adc.Strang if args.strang else adc.Split
    sim.add_equation(
        "ne",
        model=model,
        spatial=adc.FiniteVolume(limiter=args.limiter, riemann="rusanov", variables="conservative"),
        time=split_factory(
            hyperbolic=adc.Explicit(method="ssprk3"),
            source=adc.CondensedSchur(kind="electrostatic_lorentz", theta=args.theta,
                                      alpha=params.alpha),
        ),
    )

    # (A) densite top-hat + perturbation ; vitesse au repos.
    rho = annular_density(nr, nth, mode, params, r_min)          # (nth, nr)
    sim.set_density("ne", rho.ravel())                          # flat[j*nr+i]
    # (B) premier solve Poisson -> phi.
    sim.solve_fields()
    phi = np.asarray(sim.potential(), dtype=np.float64).reshape(nth, nr)   # phi[theta, r]
    # (C) v_r = ExB (-grad_theta/B, point fixe de l'etage source) ; v_theta = EQUILIBRE ROTATIF
    # (racine de la quadratique de bilan radial : centrifuge + pression + electrique + Lorentz). Meme
    # stencil radial que le moteur (polar_gradient). L'ancien v_theta = grad_r/B (ExB pur) portait une
    # quantite de mouvement NON equilibree -> transient radial rapide -> NaN avant la fenetre de fit.
    grad_r, grad_theta = polar_gradient(phi, r_min, dr, nth, nr)
    B = params.omega
    v_r = -grad_theta / B
    # IC azimutale : "equilibrium" (defaut historique) = racine de la quadratique du bilan radial
    # (centrifuge + pression + electrique + Lorentz) ; "exb" = derive ExB PURE v_theta = grad_r/B
    # (la limite de derive du papier, sans inertie ni pression). L'IC exb avait ete abandonnee car
    # elle "explosait avant la fenetre de fit" -- mesure PRE-fix-seam (adc_cpp #289) : a re-tester,
    # la phase du mode (ADC-78) montre que l'equilibre rotatif a une rotation de fond quasi nulle
    # (omega_phase -0.07) la ou le cartesien paper-faithful tourne a +0.42.
    if getattr(args, "ic", "equilibrium") == "exb":
        v_theta = grad_r / B
    else:
        v_theta = equilibrium_v_theta(rho, grad_r, r_min, dr, nth, nr, params, args.cs2)
    # (D) etat conservatif (3, ntheta, nr) comp-major (rho, mom_r, mom_theta), aplati C-order.
    U = np.stack([rho, rho * v_r, rho * v_theta], axis=0)        # (3, nth, nr)
    sim.set_state("ne", U.ravel())
    # (E) re-solve : phi coherent avec l'etat drift avant le 1er pas de transport.
    sim.solve_fields()
    return sim


def compute_frozen_residual(params, args):
    r"""Residu d'equilibre GELE R_eq = step(U_eq) - U_eq (option c, bilan discret bien pose).

    Le modele complet polaire n'est PAS discretement bien pose a la raideur du papier
    (B_z = omega = 1e12) : l'etage hyperbolique (WENO5 + Rusanov au saut top-hat de l'anneau) et
    l'etage Schur n'annulent le bilan electrique/Lorentz/centrifuge/pression QU'AU CONTINU, jamais
    au DISCRET. Resultat : un seul pas du schema applique a l'etat d'equilibre axisymetrique U_eq
    produit une DERIVE PARASITE O(1) (R_eq != 0) qui fait diverger les modes azimutaux (NaN ~ t 0.02),
    quel que soit le pas de temps ou la tentative de Newton (#23 etait un NO-OP : jacobienne
    rho*B_z ~ 1e12 -> correction ~ 1e-12).

    Option c (soustraction du residu gele) : on PRECALCULE R_eq UNE FOIS sur l'etat axisymetrique
    U_eq (perturbation=0), puis on remplace la carte d'avancement step() par step()-R_eq dans la
    boucle perturbee. Par construction U_eq devient un POINT FIXE DISCRET EXACT :

        (step - R_eq)(U_eq) = step(U_eq) - (step(U_eq) - U_eq) = U_eq

    a la PRECISION MACHINE, INDEPENDAMMENT du stencil. La derive axisymetrique O(1) est donc annulee
    a chaque pas ; seule subsiste la perturbation (le mode physique O(delta) et son petit residu
    O(delta), car R_eq est evalue au phi d'equilibre alors que le phi perturbe differe : la
    soustraction enleve la derive axisymetrique dominante O(1) et laisse un residu O(delta) tolerable,
    le mode physique etant lui-meme O(delta) et croissant exponentiellement au-dessus).

    R_eq est calcule sur un System SONDE DEDIE (jamais le sim de production : step() avance time()
    de facon IRREVERSIBLE, ce qui decalerait les fenetres de fit absolues du papier). dt_fixed est le
    MEME pas que la boucle de production (R_eq est la derive PAR PAS A CE dt ; step()-R_eq n'a U_eq pour
    point fixe qu'a ce dt). La sonde est ensuite jetee.

    @return (U_eq, R_eq) : tableaux (3, ntheta, nr) comp-major (rho, mom_r, mom_theta).
    """
    nr, nth = args.nr, args.ntheta
    dt = args.dt
    # perturbation=0 -> annular_density ignore le terme sin(l theta) : anneau strictement axisymetrique
    # quel que soit l'argument mode (ici 1). Meme nr/ntheta/cs2/theta/B/dt que la production -> R_eq
    # correspond exactement a la derive du schema de production sur SON equilibre.
    flat_params = PaperParameters(final_time=params.final_time, temperature=params.temperature,
                                  perturbation=0.0)
    probe = build_polar_system(nr, nth, mode=1, params=flat_params, args=args)
    U_eq = np.asarray(probe.get_state("ne"), dtype=np.float64).reshape(3, nth, nr)
    # build_polar_system a deja pose U_eq et resolu phi ; on impose explicitement l'etat puis un
    # solve_fields pour garantir un phi coherent avec U_eq avant le pas sonde (step() re-resout aussi
    # en tete de macro-pas, mais on reste explicite pour la robustesse du contrat).
    probe.set_state("ne", U_eq.ravel())
    probe.solve_fields()
    probe.step(dt)                        # UN macro-pas complet : SSPRK3 hyperbolique + Schur polaire
    U1 = np.asarray(probe.get_state("ne"), dtype=np.float64).reshape(3, nth, nr)
    R_eq = U1 - U_eq                       # derive parasite O(1) du schema sur l'anneau axisymetrique
    return U_eq, R_eq


def step_frozen_subtracted(sim, dt, R_eq, nth, nr):
    """Un pas de la carte CORRIGEE U <- step(U) - R_eq (option c), puis solve_fields.

    Avance le sim d'un macro-pas (Split/Strang : SSPRK3 hyperbolique + etage Schur polaire), relit
    l'etat conservatif (3, ntheta, nr), SOUSTRAIT le residu d'equilibre GELE R_eq (constant, calcule
    une seule fois), re-impose l'etat corrige, puis re-resout Poisson pour que sim.potential() reflete
    l'etat corrige a l'echantillonnage. R_eq est le MEME tableau a chaque pas (gele).
    """
    sim.step(dt)
    U = np.asarray(sim.get_state("ne"), dtype=np.float64).reshape(3, nth, nr)
    sim.set_state("ne", (U - R_eq).ravel())
    sim.solve_fields()                     # phi coherent avec l'etat corrige (pour l'observable phi[:, i_r0])


def mode_amplitude_polar(sim, mode, i_r0, nth, nr):
    """Amplitude du mode l de phi sur le cercle r=r0 (colonne native phi[:, i_r0]).

    phi est rendu (ny, nx) = (ntheta, nr) (cf. to_2d(s.ny(), s.nx())). Le cercle r=r0 est EXACTEMENT
    la colonne radiale i_r0 (une ligne complete en theta a r fixe) : aucune interpolation bilineaire
    autour d'un cercle (contrairement au cartesien sample_circle). Convention diag_polar_omega :
    c_l = (rfft(phi[:, i_r0]) / ntheta)[l], amplitude = 2 |c_l|.
    """
    phi = np.asarray(sim.potential(), dtype=np.float64).reshape(nth, nr)
    line = phi[:, i_r0]
    coeffs = np.fft.rfft(line) / nth
    return 2.0 * abs(coeffs[mode])


def fit_growth(times, amplitudes, mode, rhobar=1.0):
    """Pente BRUTE sur la fenetre papier MAPPEE en temps sim (T3 : t_sim=2pi/rhobar * t_paper).

    Comme le chemin cartesien, le solveur polaire tourne en horloge ExB-naturelle ; la fenetre
    papier (T_d) doit etre mappee avant le fit (sinon transitoire). gamma_raw_sim ; conversion
    x2pi/rhobar a l'enregistrement. NB : la repro quantitative du polaire COMPLET n'est PAS
    etablie (VOIE 1 diverge) ; ce mapping rend juste la metrologie coherente avec le cartesien.
    """
    lo, hi = paper_to_sim_time_window(PAPER_FIT_WINDOWS[mode], rhobar)
    times = np.asarray(times)
    amplitudes = np.asarray(amplitudes)
    mask = (times >= lo) & (times <= hi) & (amplitudes > 0.0)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    return float(np.polyfit(times[mask], np.log(amplitudes[mask]), 1)[0])


def run_mode(mode, params, args, R_eq=None):
    """Avance le mode l sur l'anneau polaire et renvoie (temps, amplitudes, gamma, masse initiale).

    Si @p R_eq est fourni (mode --frozen-equilibrium), la carte d'avancement est step()-R_eq (option c,
    soustraction du residu d'equilibre gele) avec un pas FIXE args.dt ; le chemin CFL adaptatif
    (step_cfl) est alors INTERDIT car il s'effondre sur l'equilibre quasi-stationnaire (estimation de
    vitesse -> 0 -> dt explose -> NaN). Sinon, comportement historique (step() nu, --cfl autorise).
    """
    nr, nth = args.nr, args.ntheta
    r_min = args.r_min
    dr = (params.radius - r_min) / nr
    i_r0 = i_radial(params.ring_inner, r_min, dr, nr)
    frozen = R_eq is not None

    sim = build_polar_system(nr, nth, mode, params, args)
    mass0 = float(sim.mass("ne"))                              # masse FV polaire (sum rho r dr dtheta)

    times, amplitudes = [], []
    step = 0
    while True:
        t = float(sim.time())
        if step % args.sample_every == 0 or t >= args.t_end:
            amp = mode_amplitude_polar(sim, mode, i_r0, nth, nr)
            phi_finite = np.isfinite(np.asarray(sim.potential())).all()
            if not phi_finite or not np.isfinite(amp):
                raise FloatingPointError("potentiel/amplitude non fini a t=%g (mode l=%d)" % (t, mode))
            times.append(t)
            amplitudes.append(amp)
        if t >= args.t_end - 0.5 * args.dt:
            break
        if frozen:
            # Option c : pas FIXE + soustraction du residu gele (R_eq est calcule a ce dt).
            step_frozen_subtracted(sim, args.dt, R_eq, nth, nr)
        elif args.cfl > 0.0:
            sim.step_cfl(args.cfl)                            # pas stable CFL polaire (chemin historique)
        else:
            sim.step(min(args.dt, args.t_end - t))
        step += 1
        if step > args.max_steps:
            raise RuntimeError("max_steps atteint avant t_end (mode l=%d)" % mode)

    gamma = fit_growth(times, amplitudes, mode, rhobar=params.rho_max)
    mass1 = float(sim.mass("ne"))
    return dict(mode=mode, times=np.asarray(times), amplitudes=np.asarray(amplitudes),
                gamma=gamma, mass0=mass0, mass1=mass1)


def write_mode_amplitude(result, out):
    """amplitude.csv par mode (toujours, matplotlib optionnel) : temps, amplitude, amplitude/initiale."""
    mode_dir = os.path.join(out, "mode_%d" % result["mode"])
    os.makedirs(mode_dir, exist_ok=True)
    with open(os.path.join(mode_dir, "amplitude.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "amplitude", "amplitude_over_initial"])
        a0 = max(float(result["amplitudes"][0]), 1.0e-300)
        for t, a in zip(result["times"], result["amplitudes"]):
            writer.writerow([t, a, a / a0])

    # Figure OPTIONNELLE : amplitude.csv (ci-dessus) + growth_rates.csv (aval) sont les sorties
    # de verite. On tolere TOUTE panne matplotlib (absent, ou ABI numpy/matplotlib bancale comme
    # sur certains noeuds) sans jamais abandonner le run de donnees.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        normalized = result["amplitudes"] / max(float(result["amplitudes"][0]), 1.0e-300)
        ax.semilogy(result["times"], normalized, color="black", lw=1.5)
        lo, hi = PAPER_FIT_WINDOWS[result["mode"]]
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.12, label="paper fit window")
        ax.set(xlabel="time", ylabel=r"$|c_l(t)|/|c_l(0)|$",
               title="Polaire l=%d, gamma=%s" % (
                   result["mode"],
                   "n/a" if not np.isfinite(result["gamma"]) else "%.4f" % result["gamma"]))
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(mode_dir, "amplitude.png"), dpi=180)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 (figure optionnelle : ne jamais casser le run de donnees)
        print("  (figure amplitude ignoree : %s)" % exc)


def write_summary(results, out, params, args):
    """growth_rates.csv (BRUT) + measurement_record (CSV/JSON) + metadata.json. matplotlib optionnel."""
    # T3 : gamma_raw_sim (fenetre MAPPEE) + gamma_paper_units = raw*2pi/rhobar ; err vs paper_units.
    rhobar = params.rho_max
    rows = []
    for r in results:
        target = PAPER_GROWTH_RATES[r["mode"]]
        g_paper = gamma_to_paper_units(r["gamma"], rhobar)
        error = (100.0 * (g_paper - target) / target) if g_paper is not None else float("nan")
        rows.append((r["mode"], r["gamma"], ("" if g_paper is None else g_paper), target, error))

    with open(os.path.join(out, "growth_rates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "gamma_raw_sim", "gamma_paper_units", "gamma_paper",
                         "relative_error_percent"])
        writer.writerows(rows)

    # Enregistrement de mesure (graine de la table de validation Phase 2), engine='polar-schur'
    # -> label 'full-polar-schur'. T3 : meme conversion 2pi/rhobar que le cartesien (meme horloge
    # ExB-naturelle, alpha/|Omega|=1). NB : repro quantitative polaire complete NON etablie (diverge).
    cpp_sha = adc_cpp_sha(adc)
    cases_sha = adc_cases_sha()
    records = [
        build_record(
            engine="polar-schur",
            mode=r["mode"],
            gamma_raw_sim=r["gamma"],
            gamma_paper=PAPER_GROWTH_RATES[r["mode"]],
            fit_window=PAPER_FIT_WINDOWS[r["mode"]],
            n=args.nr,
            dt=args.dt,
            splitting=("Strang" if args.strang else "Lie"),
            schur_theta=args.theta,
            backend="kokkos-serial",   # mono-rang (l'etage Schur polaire l'exige)
            rhobar=rhobar,
            mpi_size=1,
            adc_cpp_sha_value=cpp_sha,
            adc_cases_sha_value=cases_sha,
        )
        for r in results
    ]
    write_records(records, out)

    metadata = {
        "paper": "https://arxiv.org/abs/2510.11808",
        "engine": "polar-schur",
        "engine_label": engine_label("polar-schur"),
        "geometry": "polar",
        "geometry_note": (
            "anneau polaire (r, theta) : la direction radiale est un axe de grille -> bords "
            "d'anneau resolus (leve le verrou cartesien, cf. docs/HOFFART_GEOMETRY_VERDICT.md). "
            "Transport IsothermalFluxPolar (#209, 3 var rho/rho v_r/rho v_theta + courbure), "
            "Poisson polaire direct (FFT-theta + Thomas-r), etage Schur condense polaire (#215, "
            "PolarCondensedSchurSourceStepper), mono-rang."
        ),
        "normalization": (
            "T3: gamma_paper_units = gamma_raw_sim * 2pi/rhobar (rhobar=rho_max=%g); the 2pi "
            "(cyclic->angular drift clock) applies to the full model too (alpha/|Omega|=1). Fit on "
            "paper windows MAPPED to sim time. NB: polar full-model quantitative reproduction is NOT "
            "established (VOIE 1 diverges); this only aligns metrology with the Cartesian path."
            % params.rho_max
        ),
        "adc_cpp_sha": cpp_sha,
        "adc_cases_sha": cases_sha,
        "parameters": params.to_dict(),
        "annulus": {
            "r_min": args.r_min, "r_max": params.radius,
            "ring_inner": params.ring_inner, "ring_outer": params.ring_outer,
            "nr": args.nr, "ntheta": args.ntheta,
        },
        "numerics": {
            "finite_volume": "%s + Rusanov (polaire)" % (
                "WENO5-Z" if args.limiter == "weno5" else args.limiter),
            "limiter": args.limiter,
            "time": "SSPRK3 + CondensedSchur(theta=%g) (%s)" % (
                args.theta, "Strang" if args.strang else "Lie"),
            "dt": args.dt, "cfl": args.cfl, "nr": args.nr, "ntheta": args.ntheta, "mpi_size": 1,
            "frozen_equilibrium": bool(args.frozen_equilibrium),
            "frozen_equilibrium_note": (
                "option c (well-balanced discret) : R_eq = step(U_eq) - U_eq precalcule UNE FOIS sur "
                "l'anneau axisymetrique (perturbation=0) puis SOUSTRAIT a chaque pas (U <- step(U) - "
                "R_eq), faisant de U_eq un point fixe discret EXACT et annulant la derive parasite "
                "O(1) du schema (qui produisait NaN ~ t 0.02). PAS FIXE OBLIGATOIRE (le CFL adaptatif "
                "s'effondre sur l'equilibre quasi-stationnaire)."
            ) if args.frozen_equilibrium else "desactive (chemin historique step() nu)",
        },
        "mass_conservation": {
            "mode_%d" % r["mode"]: {
                "mass0": r["mass0"], "mass1": r["mass1"],
                "rel_drift": (abs(r["mass1"] - r["mass0"]) / max(abs(r["mass0"]), 1e-300)),
            } for r in results
        },
        "fidelity": {
            "same_pde": True,
            "polar_grid": True,
            "rotating_equilibrium_ic": True,
            "frozen_equilibrium_residual_subtraction": bool(args.frozen_equilibrium),
            "paper_schur_source": True,
            "quantitative_comparison_enabled": True,
            "quantitative_paper_claim": False,
            "known_differences": [
                "finite-volume spatial discretisation (WENO5-Z + Rusanov)",
                "Lie splitting (unless --strang)",
                "initial v_theta is the rotating-equilibrium root of the radial momentum balance "
                "(centrifugal + pressure + electric + Lorentz), reducing to the ExB drift grad_r/B "
                "as curvature -> 0; v_r = ExB -grad_theta/B; recomputed in Python from phi with the "
                "same radial stencil as derive_aux_polar (no aux-gradient accessor)",
                "cs2 small but non-zero (default 1e-4) for strict hyperbolicity on the polar grid "
                "(paper allows theta >= 0)",
            ],
        },
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Figure recap OPTIONNELLE : growth_rates.csv + metadata.json + record sont deja ecrits.
    # On tolere toute panne matplotlib (absent / ABI bancale) sans casser le run.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        modes = [r[0] for r in rows]
        numeric = [r[1] for r in rows]
        target = [r[2] for r in rows]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(modes, target, "s-", color="tab:red", label="paper")
        ax.plot(modes, numeric, "o-", color="black", label="full-polar-schur")
        ax.set(xlabel="azimuthal mode l", ylabel="growth rate gamma",
               xticks=modes, title="Diocotron growth rates (polar grid)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "growth_rates.png"), dpi=180)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 (figure optionnelle : ne jamais casser le run de donnees)
        print("(figure growth_rates ignoree : %s)" % exc)


def mpi_size():
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS"):
        if key in os.environ:
            return int(os.environ[key])
    return 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--nr", type=int, default=256, help="cellules radiales (anneau)")
    parser.add_argument("--ntheta", type=int, default=256, help="cellules azimutales (periodiques)")
    parser.add_argument("--r-min", dest="r_min", type=float, default=DEFAULT_RMIN,
                        help="bord interne du domaine polaire (> 0 ; evite la singularite r=0)")
    parser.add_argument("--t-end", dest="t_end", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3,
                        help="pas de temps (utilise si --cfl <= 0 ; sinon step_cfl)")
    parser.add_argument("--cfl", type=float, default=0.0,
                        help="si > 0 : pas adaptatif step_cfl(cfl) ; sinon pas fixe --dt")
    parser.add_argument("--cs2", type=float, default=1.0e-4,
                        help="vitesse du son au carre. Defaut 1e-4 (petit mais NON nul) pour la "
                             "stricte hyperbolicite sur la grille polaire (le papier autorise "
                             "theta >= 0). Mettre 0 pour la limite froide exacte du papier.")
    parser.add_argument("--theta", type=float, default=0.5,
                        help="theta du CondensedSchur (0.5 Crank-Nicolson, 1 Euler retrograde)")
    parser.add_argument("--ic", choices=["equilibrium", "exb"], default="equilibrium",
                        help="IC azimutale : equilibrium = bilan radial exact (historique) ; exb = "
                             "derive ExB pure (limite du papier ; re-testable post-fix-seam, ADC-78)")
    parser.add_argument("--limiter", choices=["weno5", "minmod"], default="weno5",
                        help="reconstruction FV. weno5 = historique (overshoote au saut top-hat -> "
                             "rho<0, suspect de la divergence t=0.01, ADC-62) ; minmod = TVD, "
                             "preserve la positivite (fix valide cote cartesien, ADC-74)")
    parser.add_argument("--strang", action="store_true",
                        help="splitting de Strang (2e ordre) au lieu de Lie (defaut)")
    # --- Option c : soustraction du residu d'equilibre gele (well-balanced discret) -----------------
    parser.add_argument("--frozen-equilibrium", dest="frozen_equilibrium",
                        action="store_true", default=True,
                        help="DEFAUT ON : bilan discret bien pose par soustraction du residu "
                             "d'equilibre GELE R_eq = step(U_eq) - U_eq (option c). U_eq devient un "
                             "point fixe discret EXACT (step()-R_eq), la derive parasite O(1) "
                             "axisymetrique du schema (qui faisait NaN ~ t 0.02) est annulee a chaque "
                             "pas. EXIGE un pas FIXE --dt (le --cfl adaptatif s'effondre sur "
                             "l'equilibre quasi-stationnaire -> dt explose -> NaN ; il est ignore "
                             "dans ce mode).")
    parser.add_argument("--no-frozen-equilibrium", dest="frozen_equilibrium",
                        action="store_false",
                        help="DESACTIVE l'option c : chemin historique step() nu (NaN ~ t 0.02 a la "
                             "raideur du papier ; conserve pour comparaison / debogage seulement).")
    parser.add_argument("--frozen-check-const", dest="frozen_check_const", type=float, default=1.0e3,
                        help="constante C du critere de stationarite a la precision machine "
                             "(max_n ||U^n - U_eq||_inf <= C eps_mach ||U_eq||_inf, defaut 1e3). "
                             "N'agit que sur --check-equilibrium quand --frozen-equilibrium est ON.")
    parser.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    parser.add_argument("--perturbation", dest="perturbation", type=float, default=0.1,
                        help="amplitude delta du mode azimutal initial sin(l theta) (eq. 35 du "
                             "papier, defaut 0.1). N'agit PAS sur --check-equilibrium ni sur le "
                             "calcul du residu gele R_eq (qui imposent perturbation=0, anneau "
                             "axisymetrique). Diagnostic : varier delta discrimine le mecanisme "
                             "de divergence du chemin perturbe (derive parasite O(delta) residuelle "
                             "vs instabilite numerique exponentielle delta-independante).")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=2_000_000)
    parser.add_argument("--quick", action="store_true", help="smoke minuscule (parse/assemblage)")
    parser.add_argument("--check-equilibrium", dest="check_equilibrium", action="store_true",
                        help="auto-test de STATIONARITE : force perturbation=0 (anneau axisymetrique), "
                             "avance quelques pas et verifie que l'amplitude de CHAQUE mode azimutal "
                             "reste plate (pas de croissance, pas de NaN). C'est la validation cle que "
                             "l'IC d'equilibre est genuinement stationnaire AVANT de mesurer une "
                             "croissance. Code de sortie non nul si l'equilibre derive.")
    parser.add_argument("--check-tol", dest="check_tol", type=float, default=0.05,
                        help="tolerance relative de derive d'amplitude pour --check-equilibrium "
                             "(defaut 5%% : amplitude finale / initiale - 1 <= tol pour chaque mode).")
    parser.add_argument("--check-modes", dest="check_modes", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="modes azimutaux surveilles par --check-equilibrium (defaut 1..5).")
    parser.add_argument("--max-steps-check", dest="max_steps_check", type=int, default=200,
                        help="nombre de pas avances par --check-equilibrium (defaut 200).")
    return parser.parse_args()


def check_equilibrium_frozen(params, args):
    r"""STATIONARITE A LA PRECISION MACHINE (frozen-equilibrium ON, la VRAIE validation de l'option c).

    Le test laxiste base-amplitude (check_equilibrium ci-dessous) NORMALISE la derive par l'echelle de
    fond O(1) ; il MASQUE l'echec discret car la derive parasite O(1) y parait "petite" en relatif. La
    validation GENUINE est : avec soustraction du residu gele, U_eq DOIT etre un point fixe discret
    EXACT. On construit U_eq (perturbation=0), on calcule R_eq, puis on avance la carte CORRIGEE
    step()-R_eq SUR CE MEME U_eq. Par construction (step-R_eq)(U_eq)=U_eq, donc chaque pas doit
    reproduire U_eq a l'arrondi flottant pres.

    CRITERE (assertion reelle) : max_n ||U^n - U_eq||_inf <= C eps_mach ||U_eq||_inf sur N >= 200 pas,
    C ~ 1e3 (l'arrondi des etages SSPRK3 + Schur + Poisson + l'aller-retour get_state/set_state
    s'accumule lineairement en N ; C eps couvre largement). Floor ABSOLU calcule sur l'echelle de U_eq,
    PAS sur le fond 1e12 : c'est la correction precise de l'ancien check laxiste qui cachait l'echec.

    @return (ok, report) : ok bool ; report = liste d'un seul dict (max_dev, floor, n_steps, finite).
    """
    nr, nth = args.nr, args.ntheta

    flat_params = PaperParameters(final_time=params.final_time, temperature=params.temperature,
                                  perturbation=0.0)
    U_eq, R_eq = compute_frozen_residual(flat_params, args)
    sim = build_polar_system(nr, nth, mode=1, params=flat_params, args=args)
    # Repart EXACTEMENT de U_eq (le meme etat que celui ayant servi a calculer R_eq) -> point fixe.
    sim.set_state("ne", U_eq.ravel())
    sim.solve_fields()

    state_scale = max(float(np.max(np.abs(U_eq))), 1.0e-300)
    eps = float(np.finfo(np.float64).eps)
    # C ~ 1e3 : marge pour l'accumulation lineaire de l'arrondi sur N pas (SSPRK3 + Schur + Poisson).
    floor = args.frozen_check_const * eps * state_scale
    n_steps = max(args.max_steps_check, 200)

    max_dev = 0.0
    finite = True
    for _ in range(n_steps):
        step_frozen_subtracted(sim, args.dt, R_eq, nth, nr)
        U = np.asarray(sim.get_state("ne"), dtype=np.float64).reshape(3, nth, nr)
        if not np.isfinite(U).all() or not np.isfinite(np.asarray(sim.potential())).all():
            finite = False
            max_dev = float("inf")
            break
        max_dev = max(max_dev, float(np.max(np.abs(U - U_eq))))

    ok = finite and (max_dev <= floor)
    report = [dict(max_dev=max_dev, floor=floor, n_steps=n_steps, finite=finite,
                   state_scale=state_scale)]
    return ok, report


def check_equilibrium(params, args):
    """STATIONARITE : anneau d'equilibre SANS perturbation -> chaque mode azimutal reste plat.

    Pose perturbation=0 (rho axisymetrique top-hat), construit l'IC d'equilibre (v_r=0 a O(eps), v_theta
    = racine du bilan radial), avance args.max_steps_check pas et verifie pour chaque mode l de
    args.check_modes que (a) le potentiel reste fini a chaque echantillon et (b) la derive relative
    d'amplitude reste <= args.check_tol.

    ATTENTION : ce critere base-amplitude est LAXISTE (normalise par l'echelle de fond O(1)) et MASQUE
    l'echec discret a la raideur du papier ; il n'est conserve que pour le chemin HISTORIQUE sans
    soustraction de residu. Avec --frozen-equilibrium (defaut), main() route vers
    check_equilibrium_frozen qui exige la STATIONARITE A LA PRECISION MACHINE (le vrai test de l'option c).

    @return (ok, report) : ok bool ; report = liste de dicts par mode (amp0, amp_max, rel_drift, finite).
    """
    nr, nth = args.nr, args.ntheta
    r_min = args.r_min
    dr = (params.radius - r_min) / nr
    i_r0 = i_radial(params.ring_inner, r_min, dr, nr)

    # perturbation=0 -> anneau strictement axisymetrique (l'equilibre rotatif doit y etre stationnaire).
    flat_params = PaperParameters(final_time=params.final_time, temperature=params.temperature,
                                  perturbation=0.0)
    sim = build_polar_system(nr, nth, mode=1, params=flat_params, args=args)

    modes = sorted(set(args.check_modes))
    amp0 = {l: mode_amplitude_polar(sim, l, i_r0, nth, nr) for l in modes}
    amp_max = dict(amp0)
    n_steps = args.max_steps_check
    for _ in range(n_steps):
        if args.cfl > 0.0:
            sim.step_cfl(args.cfl)
        else:
            sim.step(args.dt)
        if not np.isfinite(np.asarray(sim.potential())).all():
            return False, [dict(mode=l, amp0=amp0[l], amp_max=float("inf"),
                                rel_drift=float("inf"), finite=False) for l in modes]
        for l in modes:
            amp_max[l] = max(amp_max[l], mode_amplitude_polar(sim, l, i_r0, nth, nr))

    report = []
    # Echelle de reference : amplitude du mode dominant initial (sinon, pour un anneau purement
    # axisymetrique, amp0 ~ bruit machine et toute derive parait enorme en relatif). On compare la
    # MONTEE absolue de chaque mode a cette echelle.
    scale = max(max(amp0.values()), 1.0e-300)
    ok = True
    for l in modes:
        rise = amp_max[l] - amp0[l]
        rel = rise / scale
        finite = np.isfinite(amp_max[l])
        mode_ok = finite and (rel <= args.check_tol)
        ok = ok and mode_ok
        report.append(dict(mode=l, amp0=amp0[l], amp_max=amp_max[l], rel_drift=rel, finite=finite))
    return ok, report


def main():
    args = parse_args()
    if any(mode not in PAPER_GROWTH_RATES for mode in args.modes):
        raise SystemExit("--modes doit etre choisi parmi 3, 4, 5")
    # MONO-RANG : l'etage Schur condense POLAIRE = boite unique couvrant l'anneau (solveur direct).
    if mpi_size() > 1:
        raise SystemExit(
            "run_polar.py est mono-rang : l'etage Schur condense polaire refuse n_ranks>1 "
            "(le solveur polaire = boite unique). Lancer avec 1 seul rang (ntasks=1).")
    # Pre-enregistrement : la comparaison du modele complet DOIT utiliser les fenetres verbatim du
    # papier (Fig. 5.4). Verrou contre toute fenetre adaptative (fit_growth lit PAPER_FIT_WINDOWS).
    verify_paper_windows(PAPER_FIT_WINDOWS)

    if args.quick:
        args.nr = 16
        args.ntheta = 16
        args.t_end = 0.004
        args.dt = 1.0e-3
        args.cfl = 0.0
        args.sample_every = 1
        args.modes = [3]
        args.max_steps_check = 5
        args.check_modes = [1, 2, 3]

    # --frozen-equilibrium (option c) IMPOSE un pas FIXE : le --cfl adaptatif est BRISE pour ce modele
    # (l'estimation de vitesse s'effondre a ~0 sur l'equilibre quasi-stationnaire -> dt ~ 1e28 -> NaN
    # instantane). On force donc args.cfl <= 0 et on previent si l'utilisateur avait demande --cfl.
    if args.frozen_equilibrium and args.cfl > 0.0:
        print("[frozen-equilibrium] --cfl=%g IGNORE : l'option c exige un pas fixe (le CFL adaptatif "
              "s'effondre sur l'equilibre quasi-stationnaire). Utilisation de --dt=%g."
              % (args.cfl, args.dt))
        args.cfl = 0.0

    # --strang bascule build_polar_system sur adc.Strang (2e ordre) ; sinon adc.Split (Lie, defaut).
    params = PaperParameters(final_time=args.t_end, temperature=args.cs2,
                             perturbation=args.perturbation)

    # --check-equilibrium : auto-test de stationarite (perturbation=0). C'est la validation cle de
    # l'IC d'equilibre AVANT toute mesure de croissance ; sort sans ecrire de growth_rates.csv.
    if args.check_equilibrium:
        if args.frozen_equilibrium:
            # VRAIE validation de l'option c : stationarite a la PRECISION MACHINE (U_eq point fixe
            # discret exact de step()-R_eq). Le critere base-amplitude (laxiste, normalise par le fond
            # O(1)) MASQUAIT l'echec discret ; ici floor = C eps_mach ||U_eq||_inf.
            print("[check-equilibrium] FROZEN : anneau axisymetrique (perturbation=0), "
                  ">= %d pas, critere PRECISION MACHINE (C=%g)" % (
                      max(args.max_steps_check, 200), args.frozen_check_const))
            ok, report = check_equilibrium_frozen(params, args)
            row = report[0]
            print("  max_dev=%.3e floor=%.3e (= C eps ||U_eq||_inf, ||U_eq||_inf=%.3e) "
                  "n_steps=%d finite=%s -> %s" % (
                      row["max_dev"], row["floor"], row["state_scale"], row["n_steps"], row["finite"],
                      "OK" if ok else "DERIVE"))
            if ok:
                print("[check-equilibrium] OK : U_eq est un point fixe discret a la precision machine "
                      "(option c : derive axisymetrique O(1) annulee par R_eq).")
                return
            raise SystemExit(
                "[check-equilibrium] ECHEC : ||U^n - U_eq||_inf=%.3e depasse le floor machine %.3e. "
                "La soustraction du residu gele ne rend PAS U_eq point fixe (bug dans R_eq ou la "
                "carte step()-R_eq)." % (row["max_dev"], row["floor"]))
        print("[check-equilibrium] anneau axisymetrique (perturbation=0), %d pas, modes %s, tol=%g"
              % (args.max_steps_check, args.check_modes, args.check_tol))
        ok, report = check_equilibrium(params, args)
        for row in report:
            print("  mode l=%d : amp0=%.3e amp_max=%.3e rel_drift=%.3e finite=%s -> %s" % (
                row["mode"], row["amp0"], row["amp_max"], row["rel_drift"], row["finite"],
                "OK" if (row["finite"] and row["rel_drift"] <= args.check_tol) else "DERIVE"))
        if ok:
            print("[check-equilibrium] OK : l'equilibre rotatif est stationnaire (chaque mode plat).")
            return
        raise SystemExit("[check-equilibrium] ECHEC : l'equilibre derive (un mode croit ou NaN). "
                         "L'IC n'est PAS un etat stationnaire du modele complet.")

    out = case_output_dir("hoffart_euler_poisson_dsl_polar_schur")

    # Option c : calcule R_eq UNE FOIS (anneau axisymetrique perturbation=0), avant la boucle des
    # modes. R_eq est GELE (constant) et reutilise pour CHAQUE mode/pas (step()-R_eq).
    R_eq = None
    if args.frozen_equilibrium:
        print("[frozen-equilibrium] precalcul du residu gele R_eq = step(U_eq) - U_eq "
              "(perturbation=0, dt=%g)" % args.dt)
        _, R_eq = compute_frozen_residual(params, args)
        print("  ||R_eq||_inf = %.3e (derive parasite O(1) du schema sur l'anneau axisymetrique)"
              % float(np.max(np.abs(R_eq))))

    results = []
    for mode in args.modes:
        print("[polar-schur] mode l=%d, nr=%d, ntheta=%d, t_end=%g, %s%s"
              % (mode, args.nr, args.ntheta, args.t_end,
                 ("cfl=%g" % args.cfl) if args.cfl > 0.0 else ("dt=%g" % args.dt),
                 " [frozen-eq]" if args.frozen_equilibrium else ""))
        result = run_mode(mode, params, args, R_eq=R_eq)
        results.append(result)
        target = PAPER_GROWTH_RATES[mode]
        print("  gamma = %s (paper %.3f) ; masse rel drift = %.2e" % (
            "n/a" if not np.isfinite(result["gamma"]) else "%.6f" % result["gamma"],
            target, abs(result["mass1"] - result["mass0"]) / max(abs(result["mass0"]), 1e-300)))
        write_mode_amplitude(result, out)

    write_summary(results, out, params, args)
    print("sorties :", out)


if __name__ == "__main__":
    main()
