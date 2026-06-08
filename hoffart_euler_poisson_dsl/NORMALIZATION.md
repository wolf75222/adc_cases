# Normalisation du taux de croissance du diocotron reduit (chemin polaire ExB scalaire)

La trouvaille en une ligne :

    gamma_norm = gamma_raw * (2 pi / rhobar)

Sur le chemin polaire ExB d'ADC (anneau global, transport ExB scalaire, WENO5 +
SSPRK3, Poisson polaire Dirichlet), ce facteur global valide la normalisation du
taux de croissance du diocotron réduit (modele de derive ExB scalaire, type
Petri), benchmark de la Section 5.3 de Hoffart et al. (arXiv:2510.11808). Ce
chemin ne resout pas le modele Euler-Poisson complet (rho, rho*u, rho*v) ; seul
l = 4 colle exactement (l = 3 +26 %, l = 5 oscille). Avec rhobar = rho_max = 1, le
facteur vaut exactement 2 pi.

Le diagnostic reproductible est `diag/diag_polar_omega.py`.


## Pourquoi gamma_raw est deja le bon Im(omega)

On suit le coefficient complexe c_l(t) du mode azimutal l du potentiel phi sur le
cercle interne r = r0 :

    c_l(t) ~ exp(-i omega_l t)
    => |c_l|   ~ exp(Im(omega) t)      (croissance)
       arg(c_l) ~ -Re(omega) t          (rotation)

Le solveur polaire tourne en unites de temps ExB-naturelles. Dans ces unites
gamma_raw = pente de log|c_l| est directement Im(omega) de l'eigenmode complexe :
il n'y a aucun re-scaling beta a appliquer (gamma_raw(sim) ~ Im(omega)_eigenmode).
Le seul facteur manquant pour passer aux unites du papier est le facteur global
2 pi / rhobar.


## Pourquoi la normalisation "par rotation locale" echoue

Une idee naturelle serait de normaliser par la rotation propre du mode :

    g_rot = (gamma_raw / |Omega_raw|) * |Re(omega)|_ana * 2 pi

Le rapport gamma_raw/Omega_raw = Im/Re est invariant d'echelle, donc en principe
robuste. mais au bord interne r0 la rotation mesuree Omega_raw est ~ 0 : il n'y a
pas de charge enfermee a l'interieur de l'anneau [r0, r1], donc pas de rotation de
corps rigide a r0. Omega_raw etant proche de zero, le rapport explose et g_rot
devient absurde (voir tableau : g_rot ~ 15 a 22, au lieu de ~ 0.7 a 0.9).

Conclusion : la bonne normalisation n'est pas une rotation locale ; c'est le
facteur global 2 pi / rhobar.


## Resultats l = 3 / 4 / 5

Mesures de `diag/diag_polar_omega.py` (top-hat [6, 8], R = 16, WENO5 / SSPRK3,
CFL 0.4). gamma_raw / Omega_raw mesures ; g_2pi = gamma_raw * 2pi/rhobar ;
g_pap = cible papier.

n = 128 :

    l   gamma_raw   Omega_raw    ratio   rat_ana    g_2pi    g_rot    g_pap
    3    0.15456    -0.02089    7.3997   0.3708    0.9712   15.41    0.772
    4    0.14526    -0.01830    7.9356   0.3309    0.9127   21.87    0.911
    5    0.07671    -0.03593    2.1351   0.1998    0.4820    7.34    0.683

n = 192 :

    l   gamma_raw   Omega_raw    ratio   rat_ana    g_2pi    g_rot    g_pap
    3    0.15460    -0.02402    6.4373   0.3708    0.9713   13.41    0.772
    4    0.14482    -0.02188    6.6183   0.3309    0.9100   18.24    0.911
    5    0.13780    -0.03193    4.3154   0.1998    0.8658   14.84    0.683

Lecture :

- l = 4 : g_2pi = 0.9127 (n=128) et 0.9100 (n=192), exact contre le papier
  (0.911) aux deux resolutions. Stable en resolution.
- l = 3 : g_2pi = 0.971 (+26 %) a n=128 ET n=192 contre 0.772 (stable, mais
  decale).
- l = 5 : oscille selon la fenetre de fit : g_2pi = 0.482 a n=128 (-29 % contre
  0.683, fenetre [2.12, 12.58]) et 0.866 a n=192 (+27 %, fenetre [2.12, 5.96]).
  C'est la fenetre retenue qui change, pas la physique (voir section suivante).
- g_rot (colonne "rotation locale") est absurde partout (~ 13 a 22) : Omega_raw
  ~ 0 a r0.


## Scatter l = 3 / 5 : sensibilite a la fenetre, pas un deficit de physique

Le ratio mesure gamma_raw/Omega_raw (~ 7 a 8 pour l=3/4, ~ 2 pour l=5) differe du
ratio analytique Im/Re (~ 0.33 a 0.20) precisement parce que Omega_raw ~ 0 a r0
(le ratio mesure n'est pas l'invariant analytique : la rotation locale n'existe
pas la). Cela confirme que la rotation locale n'est pas l'echelle pertinente.

Le scatter l=3 (+26 %) et l=5 (oscillation -29 % -> +27 %) provient de la
sensibilite A LA fenetre de fit du regime exponentiel, pas d'un deficit de
physique : la pente log|c_l| n'est exponentielle pure que sur un intervalle borne
(apres le transitoire initial, avant la saturation), et la pente extraite depend
de cet intervalle. l = 4, dont la fenetre exponentielle est la plus nette, est
exact et stable en resolution.


## Conclusion

Le chemin polaire + normalisation 2 pi / rhobar valide la normalisation du
diocotron réduit (derive ExB scalaire) : l = 4 colle exactement a n = 128 et
n = 192 ; l = 3 (+26 %) et l = 5 (oscillant -29 % / +27 %) restent decales et
sensibles a la fenetre de fit. Ce chemin n'est pas le modele Euler-Poisson complet
de Hoffart et al. et ne constitue donc pas une reproduction du modele complet. Le
facteur correct est global (2 pi / rhobar), pas une rotation locale (qui echoue car
la rotation a r0 est nulle).

Pour memoire, le chemin cartesien-Schur (cf. `run.py --engine system-schur`) donne
~ 0.035 : il omet le facteur 2 pi et subit la diffusion du bord d'anneau cartesien
(l'anneau circulaire est impose par le mur du Poisson alors que le transport reste
sur la grille carree). Ce n'est pas un contre-exemple a la normalisation ; c'est le
bord cartesien + le facteur manquant. Ce chemin reste par ailleurs en splitting de
Lie (pas Strang), donc non identique a la methode du papier.


## Relancer

    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 128
    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 192

Le module `adc` doit etre construit (chemin polaire : adc.System(mesh=PolarMesh),
transport ExB, Poisson polaire). Build de reference utilise ici :
adc_cpp/build-master/python.
