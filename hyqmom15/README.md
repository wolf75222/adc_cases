# hyqmom15 : modele 2D a 15 moments (fermeture HyQMOM), flux valide contre RIEMOM2D

## 0. Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` |
| Entrees | 10 etats figes `golden/golden_states.csv` (4 gaussiennes dont correlee et haut Mach ~ 20, 5 melanges discrets dont un quasi-degenere C20 ~ 1e-6, 1 gaussienne anisotrope C20/C02 = 100) ; aucun maillage : validation ponctuelle du flux. |
| Sorties | aucun fichier produit (asserts) ; `.so` production et aot compiles dans `out/hyqmom15/`. |
| Invariants garantis | (1) `eval_flux` == `Flux_closure15_2D.m` sur 10 etats x {Fx, Fy}, rtol 1e-12 + atol 1e-13 x echelle de l'etat (`run.py:89-101`) ; (2) les 6 entrees d'ordre 5 == moments bruts exacts d'Isserlis sur 4 gaussiennes, rtol 1e-12 (`run.py:104-118`) ; (3) les 20 recopies d'ordre <= 4 bit-identiques a U, `np.array_equal` (`run.py:121-130`) ; (4) `check_model` passe sur les 10 etats realisables, compilation AOT chronometree SANS assert mural (`run.py:132-150`) ; (5) contraste degenere C20 = 0 : flux `bit_match` divergent ET flux `robust` fini (`run.py:153-168`) ; (6) anti-derive goldens (`run.py:70-78`) + borne k*sqrt(C) confrontee aux vraies vitesses de `golden_vp.csv` : gaussiennes a sqrt(6)*sqrt(C) exactement, au moins un melange DEPASSE la borne (`run.py:170-202`). |
| Prouve | le pipeline de fermeture (M -> C -> S -> closureS5 -> C5 -> M5), GENERE par `adc.moments` (adc_cpp ADC-164) depuis la seule fermeture, est correct au sens du code MATLAB de reference, execute reellement (Octave) et non re-transcrit ; la fermeture est exacte sur les gaussiennes ; le modele compile et se lie par les chemins production et aot ; l'equivalence a l'ancien modele transcrit a la main est etablie (flux 2.6e-13, borne bring-up et sources bit-exactes, ADC-172). |
| Ne prouve pas | la stabilite d'une evolution temporelle (aucun pas de temps ici : drivers ADC-84/85) ; les vitesses d'onde HLL exactes (ADC-87/88 ; `golden_vp.csv` ne sert ici qu'a ENCADRER la borne bring-up, pas a valider un calcul de vitesses du modele) ; la SURETE de la borne bring-up `|u| + 3 sqrt(C)` hors gaussiennes : l'invariant (6) DEMONTRE qu'elle est depassee par des melanges realisables (pire ratio 3.29 sur le jeu, non borne pres de la frontiere de realisabilite) -- demarrage Rusanov uniquement, jamais production ; le mode `robust` au-dela de la finitude (les planchers n'existent pas dans le MATLAB) ; l'execution du `.so` dans un `System` (compilation et chargement seulement, pas de pas de temps) ; les chemins device GPU/Kokkos et MPI. |
| Provenance | RIEMOM2D `0f2a196`, GNU Octave 11.3.0 (aarch64-darwin), adc_cpp `4bb7cec`, backends production + aot, macOS ; cout : ~10 s (compilations comprises). |

## 1. Physique : pourquoi 15 moments et une fermeture

Le systeme est la hierarchie des moments cartesiens d'ordre <= 4 de l'equation de Vlasov 2D
(document maths `main.pdf`, eq. 1.2) : pour chaque moment $M_{pq} = \int f\, v_x^p v_y^q\, dv$,

$$\partial_t M_{pq} + \partial_x M_{p+1,q} + \partial_y M_{p,q+1} = \text{sources},$$

le flux du moment d'ordre maximal fait donc apparaitre des moments d'ordre 5 qui ne sont pas
dans le vecteur d'etat : c'est le probleme de fermeture. La fermeture HyQMOM (Bryngelson, Fox
et Laurent 2025, hal-05398171, eq. 1.10-1.12 du document) exprime les six moments standardises
d'ordre 5 en fonction des moments d'ordre inferieur et rend le systeme globalement hyperbolique
(valeurs propres reelles). Justifie la clause Prouve (1) : ce cas verifie la transcription de
cette fermeture, pas sa physique.

## 2. Equations et table des couches

Etat (ordre du document et du MATLAB, 0-based) :

```
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
```

Flux : `Fx = [M10 M20 M30 M40 M50 M11 M21 M31 M41 M12 M22 M32 M13 M23 M14]` et
`Fy = [M01 M11 M21 M31 M41 M02 M12 M22 M32 M03 M13 M23 M04 M14 M05]`. 20 entrees sur 30
recopient une composante de U (justifie l'invariant (3)) ; les 6 moments d'ordre 5 distincts
(M50, M41, M32, M23, M14, M05) sont reconstruits par la fermeture.

| Ligne | Couche | Ce qui se passe |
|---|---|---|
| `run.py` `build_moment_model()` | Python compose | construction du modele symbolique, fermeture choisie par callable |
| `adc.moments.build_moment_model(order=4, closure, ...)` | generateur (adc_cpp ADC-164) | le pipeline M -> C -> S -> closureS5 -> C5 -> M5 DERIVE en boucles sur l'AST (let-bindings -> locales C++ nommees) |
| `m.compile(..., backend="aot")` (`run.py`) | brique compilee | flux evalue par cellule sans callback Python |

## 3. Code : qui fait quoi depuis l'adoption du generateur (ADC-172)

Le pipeline n'est PLUS transcrit a la main : `build_moment_model` de [model.py](model.py) est
un emballage de `adc.moments.build_moment_model(order=4, closure=hyqmom_closure,
blocks=HYQMOM_BLOCKS, robust=, sources=)` (adc_cpp ADC-164). Le generateur derive l'algebre
en boucles sur l'AST de la DSL -- memes etages qu'avant, chacun en let-bindings (variables
locales nommees du C++ genere, codegen lineaire) :

- vitesses moyennes `u = M10/M00`, `v = M01/M00` ; moments centres C20..C04 par transformation
  binomiale (l'equivalent de `M4toC4.m`, derive et non transcrit -- algebriquement identique,
  d'ou la tolerance d'arrondi rtol 1e-12 du golden et non l'egalite bit) ; standardisation
  `S_ij = C_ij/(sx^i sy^j)` (`M2CS4_15.m`) via `sx = sqrt(C20)` et produits entiers ;
  de-standardisation et moments bruts d'ordre 5 par binomiale inverse (`Flux_closure15_2D.m`
  lignes 55-62 + `C5toM5.m`) ; les entrees d'ordre <= 4 du flux restent des recopies directes
  de U.
- RESTE MANUEL dans model.py, et seulement cela :
  - `hyqmom_closure` : transcription LITTERALE de `closureS5.m` (forme polynomiale).
    Attention : les variantes `Moments5.m` et `S5_2D.m` du depot MATLAB different sur S32/S23 ;
    le chemin de flux de reference appelle `closureS5`, c'est elle qui est transcrite et le
    golden (1) detecterait toute derive. C'est l'UNIQUE physique du modele : une autre
    fermeture du meme contrat (S -> ordre 5 standardise) se branche par simple callable.
  - `HYQMOM_BLOCKS` : partition spectrale du chemin production MATLAB (section 7).
  - la borne bring-up `u +- 3 sqrt(C)` (`m.eigenvalues`, exprimee inline avec les MEMES
    operations que l'ancien modele manuel -- valeurs bit-identiques) : les vraies vitesses
    (`golden_vp.csv`) valent EXACTEMENT `u +- sqrt(6) sqrt(C)` sur une gaussienne (k = 3
    couvre, marge ~22 %) mais des melanges asymetriques realisables depassent k = 3 (pire
    ratio 3.29) -- voir Ne prouve pas et l'invariant (6).
  - le cablage plasma : sources Lorentz (section 5), Poisson (section 6), omega_p.
- Mode `robust=True` : plancher lisse `max(x, eps) = ((x+eps)+|x-eps|)/2` sur M00, C20, C02
  (cote generateur). Hors MATLAB (qui ne protege rien : division par M00 et sqrt(C20)
  inconditionnels, `closureS5.m` test p2p2 commente) ; smoke de finitude seulement. Nuance vs
  l'ancien modele manuel : les planchers C20/C02 ne sont appliques que la ou ils protegent
  (sqrt, divisions de standardisation), pas dans les termes polynomiaux de la reconstruction
  d'ordre 5 -- identique sur etat sain, fini des deux cotes pres de la degenerescence.

## 4. Maths : les deux oracles independants

Golden MATLAB : les goldens sont produits par `golden_gen.m` qui EXECUTE le depot de reference
(Octave, `--path RIEMOM2D`), pas par une re-transcription Python (une re-transcription
partagerait ses fautes avec le modele et ne prouverait rien). Commande exacte :

```
python3 gen_states.py
octave --no-gui --path /chemin/vers/RIEMOM2D golden_gen.m
```

Oracle gaussien : pour une gaussienne 2D de covariance $[[C_{20}, C_{11}], [C_{11}, C_{02}]]$,
tout moment centre d'ordre impair est nul (Isserlis), donc les six $C$ d'ordre 5 sont nuls et
les moments bruts $M_{pq}$ d'ordre 5 ont la forme fermee du binome
$M_{pq} = \rho \sum_{ij} \binom{p}{i}\binom{q}{j} u_x^{p-i} u_y^{q-j} C_{ij}$
(`model.py` `gaussian_raw_moment`, calcule sans le pipeline). La fermeture HyQMOM est exacte sur ce cas :
avec $S_{30} = S_{21} = S_{12} = S_{03} = 0$, chaque formule de `closureS5` s'annule terme a
terme, y compris pour $C_{11} \neq 0$ (etat correle n. 3). L'oracle verifie donc le pipeline
complet de bout en bout sur une famille a 6 parametres, independamment du MATLAB.

Realisabilite des etats de test : les melanges discrets $f = \sum_k w_k \delta(v - v_k)$
(`model.py` `mixture_state`) sont des distributions, leurs moments sont realisables par construction ;
c'est ainsi qu'on obtient des etats fortement asymetriques (S30 != 0) et le quasi-degenere
(trois points resserres a 1e-3 en vx : C20 ~ 1e-6, test de cancellation des sqrt).

## 5. Evolution temporelle : sources et croisement de jets (run_crossing.py)

Second script du cas (manifeste separe, `validation`, CI). Trois apports :

- Sources de la hierarchie (document maths eq. 1.2) generees PROGRAMMATIQUEMENT
  (`adc.moments.lorentz_sources`, hierarchie generique en l'ordre remontee dans adc_cpp
  (ADC-164) car independante de la fermeture ; `model.py` `moment_sources` = adaptateur
  nom -> (p, q), bit-exact a l'ancienne boucle locale) et verifiees contre les 15 equations
  EXPLICITES 1.3-1.7 transcrites a la main (`run_crossing.py`, oracle : 20 tirages
  aleatoires, rtol 1e-14). S[M00] = 0 et conservation de M20 + M02 par B verifies.
- Rotation de Larmor a travers le System COMPLET (`run_crossing.py:100-138`) : etat gaussien
  uniforme derivant, omega_c = 2, E = 0 -> M10/M01 suivent l'analytique cos/sin a 1e-3 apres
  1/8 de tour (source compilee dans la brique, ssprk2) ; M20 + M02 conserve a 1e-10.
- Croisement de jets de reference (`main_pb_2Dcrossing_2DHyQMOM15.m`) : IC portee dans
  `model.py` `crossing_state` (fond rho = 1e-3 au repos, carre central [3n/8, 5n/8) coupe par
  l'anti-diagonale, jets +-Uc, Uc = Ma/sqrt(2)) ; smoke a Ma = 2 MODERE (le Ma = 20 passe par
  la projection de realisabilite `relaxation15`, PORTEE depuis ADC-176 : run_relaxation.py),
  mode
  `robust`, Rusanov + borne bring-up, 10 pas CFL 0.4 : etat fini, M00 > 0, C20/C02 >= 0,
  masse conservee (0.0) ; snapshot npz des 15 moments ecrit (`System.write`).

Ne prouve pas : la fidelite quantitative au MATLAB (schema different : Rusanov + borne
bring-up vs HLL exact + relaxation15 a Ma = 20 ; comparaison fidele = ADC-89 apres ADC-87/88) ;
le couplage Poisson (E = 0 partout ici : ADC-85) ; r != 0 (InitializeM4_15 fige S22 = 1,
S31 = S13 = 0, distinct de la gaussienne correlee exacte : non porte, refus explicite).

## 6. Couplage Vlasov-Poisson : diocotron (run_diocotron.py)

Troisieme script (manifeste separe, `validation`, CI). Le Poisson du systeme est resolu sur M00
a chaque pas et E = -grad(phi) retro-agit par la source electrique. Scenario de reference :
`main_electrostatic_wave.m` section dicotron (anneau 0.35..0.40, mode 4, omega_p = 25,
omega_c = -30, branche electrostatique SEULE a l'execution -- B n'entre que par la derive ExB
initiale, fidele au MATLAB).

- Convention epinglee par oracle analytique (`run_diocotron.py:121-180`) : ADC resout
  Delta(phi) = rhs avec rhs = (M00 - rho_background)/lambda^2 ; sur 1 + eps cos(kx),
  phi == -eps cos(kx)/(lambda^2 k^2) a 8e-4 (n = 64). Le fond neutralisant est un parametre
  EXPLICITE (= moyenne du scenario, constante car la masse est conservee, equivalent strict de
  la soustraction de moyenne par pas de `poisson_fft.m`) : un rhs periodique a moyenne non
  nulle rend le MG singulier (constate : damier de Nyquist + re-solve divergent).
- La source compilee lit EXACTEMENT le champ resolu : Ex implicite (rhs[M10]/M00) == gradient
  centre de phi a 1.4e-16, == analytique a 8e-4.
- IC diocotron (`run_diocotron.py:77-100`) : port d'`initialize_dicotron.m` (anneau perturbe
  mode 4, Poisson IC en numpy/FFT transcrit de `poisson_fft.m`, derive ExB
  v = (-d_y phi, +d_x phi)/omega_c, moments gaussiens derives par cellule).
- Smoke 10 pas (robust, rusanov, CFL 0.4) : fini, M00 > 0, masse conservee a 2.6e-16, phi fini ;
  checkpoint/restart BIT-identique sur 2 pas ; snapshots npz cadences (etat + phi).
- `m.source_frequency(omega_p)` borne le pas source (la 'deuxieme CFL' de `compute_dt.m`).

Ne prouve pas : le taux de croissance diocotron (reference MATLAB en HLLC + relaxation15 ;
quantitatif = ADC-89 apres ADC-87/88) ; la realisabilite long-terme sans relaxation15.

## 7. Vitesses HLL exactes : autodiff + eig par blocs (run_waves.py)

Quatrieme script (manifeste separe, `validation`, CI ; exige adc_cpp >= ADC-87). Le verrou HLL
est leve SANS jacobienne generee a la main ni SymPy : `exact_speeds=True` branche
`m.wave_speeds_from_jacobian(blocks=HYQMOM_BLOCKS)` -- AUTODIFF du flux declare (dsl.diff) +
valeurs propres numeriques par sous-blocs (adc::real_eig_minmax), le jacobien ne pouvant pas se
desynchroniser du flux. `HYQMOM_BLOCKS` est le miroir exact du chemin production MATLAB
(`eigenvalues15_2D.m` flagsym = 1) : chaines x contiguës 1:5 / 6:9 / 13:15 (10:12 saute) et,
en y, les listes d'indices NON CONTIGUES [0,5,9,12,14] / [1,6,10,13] / [3,8,4] sur le dFy/dU
direct -- strictement equivalentes au swap d'arguments de `jacobian15` (`model.py`, commentaire
de `HYQMOM_BLOCKS`).

- 9 etats bien conditionnes : [vpxmin, vpxmax, vpymin, vpymax] == golden Octave a 1.2e-11 (x)
  et 2.2e-15 (y) -- la direction y, ou une mauvaise partition se rate silencieusement, est un
  gate dur (`run_waves.py:93-101`).
- Etat quasi-degenere (C20 ~ 1e-6, paires de valeurs propres quasi-defectives) : ecart 8e-4
  IMPUTE AU CONDITIONNEMENT et PROUVE en test (`run_waves.py:58-79` : une perturbation 1e-12
  des entrees du jacobien deplace les extremes de ~7e-3 mesure ; tolerance = 100 x sensibilite
  mesuree, auto-justifiee).
- La borne CFL exacte (max_wave_speed, memes blocs) couvre les vraies vitesses sur les 10
  etats : la faille de surete de la borne bring-up (section 3 / invariant 6 de run.py) est
  FERMEE par le chemin exact (`run_waves.py:113-128`).

Ne prouve pas : l'execution compilee dans un System (couverte cote adc_cpp par les tests
d'ADC-87 : eval_rhs HLL == reference numpy a 8e-15) ; la bascule des drivers (ADC-89).

## 8. Bascule HLL : fidelite au schema MATLAB (ADC-89)

La cible fidele est ATTEINTE : les drivers tournent en `riemann='hll'` avec les vitesses
exactes (`exact_speeds=True`, section 7).

- Golden HLL matche : `golden_hll_gen.m` (Octave sur RIEMOM2D) fait tourner le crossing Ma = 2
  (Np = 64, 20 pas, sans relaxation) avec le SCHEMA du depot de reference -- vitesses
  `eigenvalues15_2D(M, 1)`, flux `Flux_closure15_2D`, HLL de Davis `pas_HLL`, split
  dimensionnel ADDITIF + Euler explicite, ghosts periodiques -- et enregistre la SEQUENCE de
  dt. adc REJOUE ces dt (`run_crossing.py`, check 6) avec le meme modele en non-splite ssprk2 :
  l'ecart residuel mesure la difference de SCHEMA seule : **L2 relatif 4.4 %** apres 20 pas
  (premier ordre, attendu a ce niveau). adc-Rusanov sur les memes dt : 5.9 % > HLL -- le
  « HLL moins diffusif » est quantifie contre la reference.
- Diocotron complet (Poisson + source electrique) en HLL exact (`run_diocotron.py`, check 6) :
  stable, masse conservee a 2.6e-16, phi fini ; `robust=False` sur ce chemin (fidele au MATLAB
  sans gardes ; les planchers du mode robust restent derivables -- diff(Abs), adc_cpp ADC-87).
- Note bit-match : le cas degenere |sL - sR| < 1e-10 differe par construction (MATLAB force
  W* = 0, adc rend FL/FR) -- mesure nulle, hors des etats compares.
- **Fidelite EXACTE au schema (ADC-176)** : le split dimensionnel ADDITIF + Euler du MATLAB
  est algebriquement le Euler non-splite (Mx + My - M = M + dt(Lx + Ly)) ; en rejouant les
  dt golden avec `time='euler'` (adc_cpp ADC-174), l'ecart de schema disparait (~arrondi,
  assert < 1e-9 dans run_crossing check 6) -- le 4.4 % de ssprk2 etait bien le 2e etage,
  pas une infidelite. ssprk2 reste le mode science (ordre 2).
- **Poisson fidele (ADC-175/176)** : `build_sim` est en `solver='fft'` par defaut (solveur
  direct periodique, l'analogue de `poisson_fft.m` -- meme operateur discret que le MG sans
  tolerance iterative) ; `solver='fft_spectral'` (symbole continu -(kx^2+ky^2), l'EXACT
  `poisson_fft.m`) atteint la solution continue a ~1e-12 sur l'oracle sinusoidal
  (run_diocotron check 7 : la meme mesure discrimine le symbole et fige les chemins
  existants). Le champ E reste un gradient centre des deux cotes (`electric_source_term.m`).

Ne prouve pas : le TAUX DE CROISSANCE diocotron vs un golden MATLAB-HLL long (le run de
reference dure des heures sous Octave : campagne dediee, suivi d'ADC-89) ; la convergence de
l'ecart de schema en maillage/pas (un second golden a Np = 128 le permettrait).

## 9. Limites et suite

La validation de run.py est ponctuelle (flux en un etat) ; l'evolution temporelle, Poisson et
les vitesses HLL sont couverts par les sections 5-8. L'epic ADC-81 (82-89) est SOLDE : flux
golden (82), HLL sans primitive p (83), sources + crossing (84), Poisson + diocotron (85),
eig dense generique (86), vitesses exactes autodiff + blocs (87/88), bascule HLL fidele (89).
Depuis ADC-172, le pipeline est genere par `adc.moments` (adc_cpp ADC-164) : ecrire un AUTRE
systeme de moments 2D = fournir une fermeture (callable S -> ordre N+1 standardise) et, si
besoin, une partition spectrale -- l'algebre, les sources Lorentz et les vitesses exactes
viennent du generateur. La projection de realisabilite
`relaxation15` est PORTEE (run_relaxation.py + [relaxation.py](relaxation.py), ADC-176) :
port verbatim valide contre Octave (12 etats, 5 branches, 3.9e-14), crossing Ma = 20 en HLL
exact sans gardes avec projection par pas -- le contraste projete/nu est mesure en
REALISABILITE (lambda_min(p2p2) par cellule : ~-1 / 13 % de cellules en un pas vs -12.8 /
52 % en accumulation ; nos schemas, plus diffusifs que le MATLAB, ne produisent pas de NaN a
cet horizon). Application par pas en Python (System.get/set_state) ; chemin compile =
ADC-177 (backlog). Restent ouverts : taux de croissance diocotron vs golden MATLAB-HLL long
(campagne dediee), golden Np = 128, terme de collision BGK (`collision15.m`, hors scope).
