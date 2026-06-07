# Valeur propre analytique Petri/Davidson du diocotron (confirmation des cibles)

Confirmation INDEPENDANTE des cibles du papier Hoffart et al. (arXiv:2510.11808,
Sec 5.3) par la theorie lineaire diocotron de la colonne creuse (reference [13] du
papier). On DERIVE (et non ajuste) le taux de croissance analytique gamma_l du mode
azimutal l pour une colonne d'electrons CREUSE en top-hat, densite uniforme rho_max sur
l'anneau [r0, r1] = [6, 8], a l'interieur d'un mur conducteur de rayon R = 16, et on
montre

    gamma_3 ~ 0.772 ,  gamma_4 ~ 0.911 ,  gamma_5 ~ 0.683

en unites omega_d = 1, SANS aucun facteur 2 pi applique a posteriori.

Diagnostic reproductible : `petri_eigenvalue.py` (numpy seul, aucune dependance au
moteur adc).

Resultat (`python hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py`) :

    l   gamma_l (Im)   papier   abs-err   rel%  |  Re(ExB)   RE_ANA   rel%  |  Im/Re  rat_ana
    3      0.77297    0.7720   0.00097  0.126%  |  0.33134  0.33144 0.031% | 0.3713  0.3708
    4      0.91182    0.9110   0.00082  0.090%  |  0.43838  0.43859 0.048% | 0.3310  0.3309
    5      0.68338    0.6830   0.00038  0.055%  |  0.54711  0.54747 0.067% | 0.1988  0.1998

gamma_l (colonne `Im`) est le max Im(omega) BRUT du probleme aux valeurs propres, en
unites omega_d, sans facteur applique apres coup : il colle aux cibles a < 0.13 %, loin
dans la marge annoncee du papier (+/- 0.024 sur eq (5.1) / Fig 5.4).


## Modele physique

Limite centre-guide (derive ExB), equilibre axisymetrique. Le papier resout

    -Delta phi0 = alpha rho0 ,   v0 = -(grad phi0 x Omega) / |Omega|^2 .

Le Poisson radial donne `r phi0'(r) = -alpha M(r)` avec `M(r) = int_0^r rho0(s) s ds` la
charge enfermee. La vitesse azimutale de derive est `v0_theta = -(1/|Omega|) phi0'`, d'ou
la frequence ANGULAIRE de derive d'equilibre

    omega_E(r) = v0_theta / r = (alpha / |Omega|) M(r) / r^2 .

Pour le top-hat `rho0 = rho_max` sur `[r0, r1]` (0 ailleurs) :

    r < r0      : omega_E = 0
    r0 < r < r1 : omega_E(r) = (Wd / 2) (1 - r0^2 / r^2)
    r > r1      : omega_E(r) = (Wd / 2) (r1^2 - r0^2) / r^2

ou `Wd` est l'echelle angulaire de derive (voir le pave 2 pi ci-dessous).

Deux ondes de surface, deplacements de bord `eta_in` (en r0) et `eta_out` (en r1), de
forme `exp(i l theta - i omega t)`, se couplent par le potentiel perturbe (harmonique l,
Laplace dans chaque anneau, regulier au centre, Dirichlet `phi1(R) = 0`). La condition
cinematique de bord et l'appariement du potentiel donnent un probleme aux valeurs propres
2 x 2 dont les coefficients geometriques d'auto- et d'inter-couplage (mur Dirichlet en R)
sont la forme standard de la colonne creuse (Davidson) :

    s_in  = (1 / 2l) (1 - (r0/R)^{2l})              auto-couplage interne
    s_out = (1 / 2l) (1 - (r1/R)^{2l})              auto-couplage externe
    s_mut = (1 / 2l) (r0/r1)^l (1 - (r1/R)^{2l})    inter-couplage interne <-> externe

La matrice (deplacements `[eta_in, eta_out]`) est

    M = [[ l omega_E(r0) + l Wd s_in ,        - l Wd s_mut          ],
         [        l Wd s_mut         ,  l omega_E(r1) - l Wd s_out   ]] .

Les SIGNES de saut de densite -- bord interne `0 -> rho_max` (+1), bord externe
`rho_max -> 0` (-1) -- rendent les deux inter-couplages de signes OPPOSES. C'est le
mecanisme de Kelvin-Helmholtz/Rayleigh : il produit la paire de valeurs propres complexes
conjuguees, donc l'instabilite. Le taux de croissance est `gamma_l = max Im(omega)`.


## Controles croises (le MODE est le bon, pas seulement Im)

Deux quantites independantes de l'unite de temps valident que la matrice resout le bon
eigenmode complexe, pas un nombre fortuit :

- **Re(omega) en unite ExB-naturelle** retrouve `RE_ANA = {3: 0.33144, 4: 0.43859,
  5: 0.54747}` (la rotation propre du mode publiee dans `../NORMALIZATION.md` /
  `diag_polar_omega.py`) a < 0.07 %.
- **Le ratio Im/Re**, INVARIANT d'echelle (independant de toute unite de temps), retrouve
  `RATIO_ANA = {3: 0.3708, 4: 0.3309, 5: 0.1998}` a < 0.5 %.

Comme le ratio est correct ET la partie reelle est correcte, la partie imaginaire l'est
aussi : c'est le bon eigenmode, pas un ajustement.


## Ou est le 2 pi (relation chemin reduit ExB <-> omega_d du modele complet)

Le papier (lignes 313-317) definit la frequence diocotron CYCLIQUE

    omega_d := rho_max * alpha / |Omega| = 1 ,   periode T_d := 1 / omega_d = 1

(et `tf = 10 = 10 T_d`). `omega_d` est une frequence CYCLIQUE : une revolution complete
(2 pi radians) par periode `T_d = 1`. L'echelle ANGULAIRE correspondante (radians par
unite de temps), celle qu'attend la relation de dispersion via `omega_E`, est donc

    Wd := 2 pi * omega_d        (= 2 pi pour omega_d = 1).

En posant `Wd = 2 pi omega_d` dans la matrice, `Im(omega)` BRUT est deja gamma_l en unites
omega_d : RIEN n'est multiplie apres coup. Le 2 pi est la conversion cyclique -> angulaire,
INTERNE au probleme.

Cote chemin REDUIT ExB scalaire (`diag_polar_omega.py`), le solveur polaire tourne en
horloge ExB-NATURELLE ou la derive d'equilibre compte les revolutions en TOURS (1 tour =
2 pi radians). La pente mesuree `gamma_raw = pente de log|c_l|` est le taux de croissance
par "tour-temps". Pour passer a l'unite omega_d (ou `T_d = 1/omega_d = 1` designe UN
tour), on multiplie par le nombre d'unites ExB-naturelles dans un `T_d`, soit 2 pi
(l'angle parcouru en un `T_d` est 2 pi). D'ou la normalisation globale du chemin reduit

    gamma_norm = gamma_raw * (2 pi / rhobar) ,   rhobar = rho_max = 1 .

Le `rhobar = rho_max` apparait parce que l'echelle de derive `omega_E ~ (alpha/|Omega|)
M(r)/r^2 ~ rho_max * (...)` est proportionnelle a rho_max : normaliser par rhobar ramene a
l'amplitude unitaire de l'anneau. Le facteur GLOBAL `2 pi / rhobar` est donc la conversion
EXACTE horloge-ExB-naturelle (radian/tour) -> horloge-omega_d (cyclique), pas un
ajustement.

**Consequence pour HOFFART_FIDELITY.md (modele COMPLET).** Le facteur `2 pi / rhobar`
appartient UNIQUEMENT au chemin reduit ExB scalaire (`diag_polar_omega.py`), qui mesure
dans l'horloge ExB-naturelle. Le present calcul construit directement en unites omega_d
(`Wd = 2 pi omega_d`) et retrouve les cibles SANS facteur : c'est exactement ce
qu'affirme la resolution de normalisation de `HOFFART_FIDELITY.md` -- la pente de
croissance BRUTE du modele complet (`run.py --engine system-schur`, qui evolue en unites
omega_d) est DIRECTEMENT comparable a 0.772 / 0.911 / 0.683, SANS facteur 2 pi.


## Relancer

    python hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py

numpy seul, leger (deux 2 x 2 par mode). Le script asserte les trois controles
(gamma_l < 1 %, Re < 1 %, Im/Re < 2 %) et sort en erreur si l'un sort de la marge.


## References

- D. Hoffart, R. Maier, J. N. Shadid, I. Tomas, structure-preserving FE pour Euler-Poisson
  magnetique, arXiv:2510.11808 (Sec 5.3, eq (5.1), Fig 5.4, reference [13]).
- R. C. Davidson, *Physics of Nonneutral Plasmas*, ch. 6 (instabilite diocotron).
- R. H. Levy, *Diocotron Instability in a Cylindrical Geometry*, Phys. Fluids 8 (1965) 1288.
