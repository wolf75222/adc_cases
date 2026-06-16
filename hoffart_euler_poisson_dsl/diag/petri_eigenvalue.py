#!/usr/bin/env python3
"""Valeur propre analytique Petri/Davidson du diocotron (colonne creuse top-hat).

But
---
deriver (et non ajuster) le taux de croissance analytique gamma_l du mode diocotron
pour une colonne d'electrons creuse en top-hat, densite uniforme rho_max sur l'anneau
[r0, r1], a l'interieur d'un mur conducteur de rayon R, dans la geometrie 6:8:16 de
Hoffart et al. (arXiv:2510.11808, Sec 5.3), puis confirmer que

    gamma_3 ~ 0.772 ,  gamma_4 ~ 0.911 ,  gamma_5 ~ 0.683

en unites omega_d = 1 (la frequence diocotron du papier), sans aucun facteur 2 pi
applique a posteriori. C'est une confirmation independante de la resolution de
normalisation consignee dans ../../docs/HOFFART_FIDELITY.md pour le modele complet :
la pente de croissance brute du modele complet est directement comparable aux cibles,
et le facteur 2 pi / rhobar n'appartient qu'au chemin reduit ExB scalaire.

Reference [13] du papier : theorie lineaire diocotron de la colonne creuse non-neutre
en limite centre-guide (derive ExB). Voir Davidson, "Physics of Nonneutral Plasmas",
chap. 6 ; Levy, Phys. Fluids 8 (1965) 1288 ; Petri, J. Plasma Phys. (annulus).

Physique
--------
Equilibre axisymetrique en limite centre-guide. Le papier resout

    -Delta phi0 = alpha rho0 ,   v0 = derive ExB = -(grad phi0 x Omega)/|Omega|^2 .

Le Poisson radial donne r phi0'(r) = -alpha M(r), avec M(r) = int_0^r rho0(s) s ds la
"masse" enfermee. La vitesse azimutale de derive vaut v0_theta = -(1/|Omega|) phi0', soit
la frequence angulaire de derive d'equilibre

    omega_E(r) = v0_theta / r = (alpha / |Omega|) M(r) / r^2 .

Le papier (lignes 313-317) pose la frequence diocotron cyclique

    omega_d := rho_max * alpha / |Omega|  = 1 ,   periode T_d := 1 / omega_d = 1

(et tf = 10 = 10 T_d). Comme omega_d est cyclique (une revolution complete de 2 pi
radians par periode T_d = 1), l'echelle angulaire correspondante est

    Wd := 2 pi * omega_d        (= 2 pi pour omega_d = 1).

C'est la l'origine du 2 pi : il convertit la frequence cyclique du papier (cycles par
unite de temps) en frequence angulaire (radians par unite de temps) utilisee par la
relation de dispersion. Voir le pave "Ou est le 2 pi" plus bas et petri_eigenvalue.md.

Pour le top-hat rho0 = rho_max sur [r0, r1] (0 ailleurs) :

    r < r0     : omega_E = 0
    r0 < r < r1: omega_E(r) = (Wd / 2) (1 - r0^2 / r^2)
    r > r1     : omega_E(r) = (Wd / 2) (r1^2 - r0^2) / r^2 .

Deux ondes de surface (deplacements de bord eta_in en r0, eta_out en r1) de forme
exp(i l theta - i omega t) se couplent via le potentiel perturbe (harmonique l, Laplace
dans chaque anneau, regulier au centre, Dirichlet phi1(R) = 0). La condition cinematique
de bord et l'appariement du potentiel donnent le probleme aux valeurs propres 2x2 :

    (omega - l omega_E(r_k)) eta_k = (couplage de champ perturbe sur eta_in, eta_out)

dont les coefficients geometriques d'auto- et inter-couplage (mur Dirichlet en R) sont la
forme standard de la colonne creuse (Davidson) :

    s_in  = (1 / 2l) (1 - (r0/R)^{2l})              auto-couplage du bord interne
    s_out = (1 / 2l) (1 - (r1/R)^{2l})              auto-couplage du bord externe
    s_mut = (1 / 2l) (r0/r1)^l (1 - (r1/R)^{2l})    inter-couplage interne <-> externe .

Les signes de saut de densite (bord interne 0 -> rho_max : +1 ; bord externe
rho_max -> 0 : -1) rendent les inter-couplages de signes opposes : c'est le mecanisme de
Kelvin-Helmholtz/Rayleigh qui produit la paire de valeurs propres complexes conjuguees
(l'instabilite). Le taux de croissance est

    gamma_l = max Im(omega) ,

en unites omega_d = 1 grace a Wd = 2 pi omega_d.

Ou est le 2 pi (relation entre chemin reduit ExB et omega_d du modele complet)
-----------------------------------------------------------------------------
Le diagnostic du chemin reduit ExB scalaire (diag_polar_omega.py) mesure gamma_raw =
pente de log|c_l| dans l'horloge ExB-naturelle du solveur polaire. Dans cette horloge la
derive d'equilibre tourne en tours (un tour = 2 pi radians) : gamma_raw est le taux de
croissance par "tour-temps". Le papier rapporte gamma_l dans l'unite omega_d cyclique,
ou T_d = 1/omega_d = 1 designe un tour. Convertir un taux exprime par unite de temps
ExB-naturelle vers l'unite omega_d revient a multiplier par le nombre d'unites
ExB-naturelles dans un T_d, soit 2 pi (puisque l'angle parcouru en un T_d est 2 pi). D'ou

    gamma_norm = gamma_raw * (2 pi / rhobar) ,   rhobar = rho_max = 1 .

Le rhobar = rho_max apparait parce que l'echelle de derive omega_E est proportionnelle a
rho_max (via M(r) ~ rho_max) : normaliser par rhobar ramene a l'amplitude unitaire de
l'anneau. Le facteur global 2 pi / rhobar n'est donc pas un ajustement ; c'est la
conversion exacte horloge-ExB-naturelle (radian/tour) -> horloge-omega_d (cyclique). Le
present script construit directement la matrice en unites omega_d (le 2 pi est dans Wd),
donc Im(omega) brut est gamma_l du papier sans facteur applique apres coup.

Lancer
------
    python hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py

numpy seul, leger (deux 2x2 par mode). Aucune dependance au moteur adc.
"""

from __future__ import annotations

import math

import numpy as np


# Geometrie 6:8:16 du papier (Sec 5.3). r0, r1 = bords de l'anneau ; R = mur conducteur.
R0, R1, RW = 6.0, 8.0, 16.0
RHO_MAX = 1.0  # rhobar = rho_max dans la conversion 2pi/rhobar
OMEGA_D = 1.0  # frequence diocotron cyclique du papier (lignes 313-317)

# Cibles du papier (Sec 5.3, eq (5.1), Fig 5.4(d), theorie lineaire [13]).
PAPER = {3: 0.772, 4: 0.911, 5: 0.683}

# Parties reelles analytiques |Re(omega)| en unite ExB-naturelle (rotation propre du mode)
# et ratio invariant d'echelle Im/Re ; valeurs publiees dans ../docs/NORMALIZATION.md /
# diag_polar_omega.py. Servent de controle croise independant de l'unite.
RE_ANA = {3: 0.33144, 4: 0.43859, 5: 0.54747}
RATIO_ANA = {3: 0.3708, 4: 0.3309, 5: 0.1998}


def equilibrium_drift(r, r0, r1, wd) -> float:
    """Frequence angulaire de derive ExB d'equilibre omega_E(r) du top-hat [r0, r1].

    omega_E(r) = (alpha/|Omega|) M(r)/r^2 avec M(r) = int_0^r rho0 s ds (top-hat). En
    posant wd = 2 pi omega_d on travaille en unites omega_d (le 2 pi est dans wd).
    """
    if r <= r0:
        return 0.0
    if r < r1:
        return 0.5 * wd * (1.0 - r0 * r0 / (r * r))
    return 0.5 * wd * (r1 * r1 - r0 * r0) / (r * r)


def diocotron_matrix(l, r0, r1, R, omega_d=OMEGA_D) -> np.ndarray:
    """Matrice 2x2 du probleme aux valeurs propres diocotron (colonne creuse top-hat).

    Les deux degres de liberte sont les deplacements des bords interne (r0) et externe
    (r1). La frequence propre omega est valeur propre de cette matrice ; gamma_l =
    max Im(omega). Construite directement en unites omega_d via Wd = 2 pi omega_d (donc
    aucun facteur 2 pi a appliquer ensuite).
    """
    if l < 1:
        raise ValueError("le mode azimutal l doit etre >= 1")
    if not (0.0 < r0 < r1 < R):
        raise ValueError("geometrie invalide : exige 0 < r0 < r1 < R")

    wd = (
        2.0 * math.pi * omega_d
    )  # echelle angulaire de derive (cyclique -> angulaire)

    w_in = equilibrium_drift(r0, r0, r1, wd)  # = 0 (rien d'enferme sous r0)
    w_out = equilibrium_drift(r1, r0, r1, wd)  # rotation au bord externe

    # geometrie des ondes de surface, mur Dirichlet en R (forme standard colonne creuse)
    s_in = (1.0 / (2 * l)) * (1.0 - (r0 / R) ** (2 * l))
    s_out = (1.0 / (2 * l)) * (1.0 - (r1 / R) ** (2 * l))
    s_mut = (1.0 / (2 * l)) * ((r0 / r1) ** l) * (1.0 - (r1 / R) ** (2 * l))

    # sauts de densite : bord interne 0 -> rho_max (+), bord externe rho_max -> 0 (-)
    # ces signes opposes rendent les inter-couplages antisymetriques -> instabilite.
    return np.array(
        [
            [l * w_in + l * wd * s_in, -l * wd * s_mut],
            [l * wd * s_mut, l * w_out - l * wd * s_out],
        ]
    )


def growth_rate(l, r0=R0, r1=R1, R=RW, omega_d=OMEGA_D) -> float:
    """gamma_l = max Im(omega) en unites omega_d (sans 2 pi applique apres coup)."""
    eigs = np.linalg.eigvals(diocotron_matrix(l, r0, r1, R, omega_d))
    return float(np.max(np.abs(eigs.imag)))


def real_frequency_exb(l, r0=R0, r1=R1, R=RW) -> float:
    """|Re(omega)| en unite ExB-naturelle (matrice avec wd = omega_d, sans 2 pi).

    Sert au controle croise : doit retrouver RE_ANA (rotation propre du mode publiee).
    """
    eigs = np.linalg.eigvals(
        diocotron_matrix(l, r0, r1, R, omega_d=1.0 / (2.0 * math.pi))
    )
    return float(np.max(np.abs(eigs.real)))


def scale_invariant_ratio(l, r0=R0, r1=R1, R=RW) -> float:
    """Im/Re : invariant d'echelle (independant de l'unite de temps), controle du MODE."""
    eigs = np.linalg.eigvals(diocotron_matrix(l, r0, r1, R, omega_d=OMEGA_D))
    im = np.max(np.abs(eigs.imag))
    re = np.max(np.abs(eigs.real))
    return float(im / re) if re else float("nan")


# Tolerance relative : la cible papier est annoncee a +/- 0.024 (eq 5.1 / Fig 5.4) ;
# on exige mieux que 1% relatif, largement dans la marge du papier.
TOL = 0.01


def self_check() -> None:
    """Verifie les trois cibles analytiques par assertions (leve si hors marge).

    (1) gamma_l brut (unite omega_d, sans 2 pi) = cible papier a <1% ;
    (2) Re(ExB) = RE_ANA a <1% (controle croise) ; (3) Im/Re = RATIO_ANA a <2%
    (invariant d'echelle, prouve que le MODE complexe est le bon).
    """
    for l in (3, 4, 5):
        g = growth_rate(l)
        assert abs(g - PAPER[l]) / PAPER[l] < TOL, (
            "gamma_%d=%.5f hors tolerance vs papier %.4f (unite omega_d, sans 2pi)"
            % (l, g, PAPER[l])
        )
        re = real_frequency_exb(l)
        assert (
            abs(re - RE_ANA[l]) / RE_ANA[l] < TOL
        ), "Re_%d=%.5f hors tolerance vs RE_ANA %.5f" % (l, re, RE_ANA[l])
        ratio = scale_invariant_ratio(l)
        assert (
            abs(ratio - RATIO_ANA[l]) / RATIO_ANA[l] < 2.0 * TOL
        ), "Im/Re_%d=%.4f hors tolerance vs RATIO_ANA %.4f" % (
            l,
            ratio,
            RATIO_ANA[l],
        )


def main() -> None:
    """Affiche la table gamma_l vs papier (Re, Im/Re) puis lance self_check()."""
    print(
        "valeur propre analytique Petri/Davidson, colonne creuse top-hat [%g,%g], mur R=%g"
        % (R0, R1, RW)
    )
    print(
        "  unites omega_d=1 (cyclique, T_d=1) ; le 2 pi est dans Wd=2pi*omega_d, "
        "rien n'est multiplie apres coup."
    )
    print()
    print(
        "%3s %14s %10s %10s %8s | %10s %10s %8s | %8s %8s"
        % (
            "l",
            "gamma_l (Im)",
            "papier",
            "abs-err",
            "rel%",
            "Re(ExB)",
            "RE_ANA",
            "rel%",
            "Im/Re",
            "rat_ana",
        )
    )
    for l in (3, 4, 5):
        g = growth_rate(l)
        re = real_frequency_exb(l)
        ratio = scale_invariant_ratio(l)
        rel = 100.0 * abs(g - PAPER[l]) / PAPER[l]
        rel_re = 100.0 * abs(re - RE_ANA[l]) / RE_ANA[l]
        print(
            "%3d %14.5f %10.4f %10.5f %7.3f%% | %10.5f %10.5f %7.3f%% | %8.4f %8.4f"
            % (
                l,
                g,
                PAPER[l],
                abs(g - PAPER[l]),
                rel,
                re,
                RE_ANA[l],
                rel_re,
                ratio,
                RATIO_ANA[l],
            )
        )
    print()
    # asserts reels : leve AssertionError (donc sortie non nulle / CI rouge) si hors marge.
    self_check()
    print(
        "confirme : gamma_3/4/5 = 0.772 / 0.911 / 0.683 reproduits a < 1% en unites "
        "omega_d, sans facteur 2 pi applique apres coup. Re(ExB) retrouve RE_ANA et "
        "Im/Re (invariant d'echelle) retrouve RATIO_ANA : le MODE complexe est correct."
    )


if __name__ == "__main__":
    main()
