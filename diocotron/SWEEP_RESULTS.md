# Balayage diocotron : ordre x resolution x mode (mesure, O1/O2 + haut ordre O5)

Quantification de l'ecart de taux de croissance diocotron (numerique `adc` vs cible analytique
de Petri) en fonction de la **resolution** et de l'**ordre de reconstruction**, pour decider la
PR-A "transport-wall". Aucune physique modifiee : ce balayage REUTILISE tel quel le pipeline de
[`run.py`](run.py) (CI anneau partagee, FFT azimutale du mode `l` de `phi`, ajustement de la
phase lineaire `exp(gamma t)`, normalisation par `omega_D`, cible analytique de Petri en numpy).
Script : [`sweep.py`](sweep.py). Donnees brutes : `out/diocotron/sweep_results.csv`.

Cette version etend PR-0 (qui balayait {O1, O2-minmod, O2-vanleer}) avec l'axe **haut ordre O5 =
WENO5-Z + SSPRK3**, desormais atteignable depuis Python (adc_cpp #88, master `ca803dc`). Le but de
l'axe O5 est d'ECLAIRER la question laissee ouverte par PR-0 : le residu l-dependant a O2 est-il de
la diffusion (refermable par l'ordre) ou un verrou structurel du bord d'anneau cartesien ?

## Protocole

- **Cas** : `adc.System` compose `models.diocotron` + Poisson paroi conductrice circulaire
  (`wall="circle"`, `wall_radius=0.40`), identique a `run.py`. CI anneau `r0:r1:Rwall =
  0.15:0.20:0.40`, perturbation azimutale `delta=0.01`, CFL=0.4.
- **Cible** : valeurs propres de Petri (numpy, `run.py`), invariantes en `n` :
  `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687`.
- **gamma_num** : pente de `log|c_l(phi)|` sur la phase lineaire (diagnostic existant
  `fit_linear_phase`), normalisee par `2 pi / rhobar`. `%err = 100 (gamma_num - gamma_ana) / gamma_ana`.
- **Horizon physique commun `t_end = 48`** : a `nsteps` fixe le pas `dt ~ CFL dx` decroit avec
  `n`, donc le temps physique final decroit avec `n` (n=256, 900 pas -> t ~ 35, phase non
  saturee, sur-lecture de gamma). On avance donc jusqu'a `t_end = 48` (horizon du calage valide
  de `run.py` : n=192, 900 pas -> t ~ 48). A cet horizon le balayage **reproduit le README a
  n=192** (l=3 -22 %, l=4 -27 %, l=5 -5 %), ce qui ANCRE la mesure. C'est un reglage de boucle,
  pas un nouvel observable.

## Axe ordre : ce qui est atteignable depuis Python

L'axe ordre vise par le cahier des charges etait {O2, **O5 WENO5-Z**}. Au moment de PR-0 (master
`adc_cpp` 30f6dfd) WENO5-Z et SSPRK3 n'etaient PAS atteignables depuis le chemin du cas diocotron.
**Depuis adc_cpp #88 (master `ca803dc`), ils le sont** par le chemin natif `add_block` :

- `adc.Spatial(limiter="weno5", flux="rusanov")` : reconstruction WENO5-Z (ordre 5 en zone lisse,
  stencil 5 points, 3 ghosts) ; `make_block` instancie maintenant la politique `Weno5`. Seul le
  chemin natif `add_block` l'expose (les chemins .so AOT/JIT allouent 2 ghosts et la rejettent).
- `adc.Explicit(method="ssprk3")` : integrateur SSPRK3 (Shu-Osher 3 etages, ordre 3). On l'apparie
  a WENO5-Z car appairer un ordre eleve en espace a SSPRK2 (ordre 2 en temps) briderait l'ordre
  effectif. La cle d'ordre `weno5` du balayage aiguille donc les DEUX briques (cf. `sweep.py`
  `make_system` : `limiter == "weno5"` selectionne `Explicit(method="ssprk3")`).

Verification empirique (master `ca803dc`) : `add_block(spatial=adc.Spatial(limiter="weno5"),
time=adc.Explicit(method="ssprk3"))` se compose et avance sans NaN (la chaine `"weno5"`, qui levait
`System : limiter inconnu 'weno5'` sur 30f6dfd, est maintenant acceptee). Les ordres O1/O2 gardent
l'integrateur historique SSPRK2 (`adc.Explicit()` par defaut) : leurs lignes ci-dessous reproduisent
exactement PR-0 (meme observable, meme calage `t_end=48`).

**Axe ordre effectivement balaye** : `{O1 none, O2 minmod, O2 vanleer, O5 weno5}`. La diffusion
numerique decroit (a) en montant la resolution et (b) en montant l'ordre / en baissant la
dissipation (none -> minmod -> vanleer, puis WENO5-Z + SSPRK3 a l'ordre 5). L'axe O5 est ce qui
permet d'eclairer la question diffusion-vs-structurel : a l'ordre 5 la diffusion residuelle est
fortement bornee, donc ce qui RESTE a O5 est le candidat le plus credible au plancher structurel.

## Resultats : gamma_num (%err vs analytique), `t_end = 48`

| n | ordre | l=3 (cible 0.772) | l=4 (cible 0.912) | l=5 (cible 0.687) |
|---|---|---|---|---|
| 128 | O1 none      | 0.263 (-66.0 %) | 0.005 (-99.5 %) | 0.111 (-83.9 %) |
| 192 | O1 none      | 0.325 (-57.9 %) | 0.371 (-59.3 %) | 0.343 (-50.0 %) |
| 256 | O1 none      | 0.463 (-40.0 %) | 0.452 (-50.5 %) | 0.418 (-39.2 %) |
| 128 | O2 minmod    | 0.506 (-34.4 %) | 0.613 (-32.8 %) | 0.519 (-24.4 %) |
| 192 | O2 minmod    | 0.599 (-22.4 %) | 0.662 (-27.4 %) | 0.652 (-5.1 %) |
| 256 | O2 minmod    | 0.639 (-17.3 %) | 0.801 (-12.1 %) | 0.703 (+2.3 %) |
| 384 | O2 minmod    | 0.679 (-12.1 %) | 0.798 (-12.5 %) | 0.705 (+2.6 %) |
| 128 | O2 vanleer   | 0.606 (-21.5 %) | 0.781 (-14.4 %) | 0.685 (-0.4 %) |
| 192 | O2 vanleer   | 0.658 (-14.8 %) | 0.752 (-17.5 %) | 0.714 (+3.9 %) |
| 256 | O2 vanleer   | 0.684 (-11.4 %) | 0.862 (-5.5 %) | 0.744 (+8.3 %) |
| 384 | O2 vanleer   | 0.702 (-9.1 %) | 0.825 (-9.5 %) | 0.710 (+3.3 %) |
| 128 | O5 weno5     | 0.659 (-14.6 %) | 0.874 (-4.2 %)  | 0.735 (+6.9 %) |
| 192 | O5 weno5     | 0.677 (-12.4 %) | 0.768 (-15.8 %) | 0.700 (+1.8 %) |
| 256 | O5 weno5     | 0.692 (-10.3 %) | 0.875 (-4.1 %)  | 0.719 (+4.7 %) |

(n=192 O2 minmod = ligne d'ancrage, reproduit le README. Les lignes O1/O2 sont rejouees ici et
reproduisent PR-0 a l'identique. O5 = WENO5-Z + SSPRK3, lance en local n=128/192/256 ; n=384 O5
saute, cf. "lance vs saute". n=384 O2 = sonde au-dela de la grille principale, 1 run ~ 143 s.)

### Tracabilite : fenetre de fit des trois points O5 l=4

Le CSV (`out/diocotron/sweep_results.csv`) ecrit pour CHAQUE ligne les bornes de la fenetre de
`fit_linear_phase` : indices `fit_i0..fit_i1` et temps `fit_t0..fit_t1`. Voici ces bornes pour les
trois points O5 l=4 (re-lances en local avec le logging de fenetre, master `adc_cpp` 28198b4) :

| n | gamma_num | %err | fenetre i0..i1 | fenetre t0..t1 |
|---|---|---|---|---|
| 128 | 0.8736 | -4.2 % | 269..528 | 20.8..41.1 |
| 192 | 0.7675 | -15.8 % | 108..703 | **5.4**..35.1 |
| 256 | 0.8745 | -4.1 % | 357..1095 | 13.3..40.8 |

La ligne n=192 est directement verifiable : sa fenetre s'OUVRE a t0 = 5.4 (contre 20.8 a n=128 et
13.3 a n=256), donc bien avant la phase exponentielle propre. C'est cette ouverture precoce qui
sous-lit la pente sur ce seul run.

**Note de lecture sur le point n=192 O5 l=4 (-15.8 %).** C'est un ARTEFACT de la fenetre
d'ajustement, pas une regression physique. La fenetre lineaire de `fit_linear_phase` s'ouvre a
t=5.4 sur ce run (contre t=20.8 a n=128 et t=13.3 a n=256), donc elle capte une transitoire
pre-asymptotique et sous-lit la pente. Les deux points O5 l=4 dont la fenetre s'ouvre sur la phase
exponentielle propre (n=128 et n=256) donnent un gamma coherent ~0.874 (soit -4.1 / -4.2 %). C'est
cette valeur ~ -4 % qui est representative de l=4 a l'ordre 5 sur ces deux points propres ; le
-15.8 % de n=192 est du bruit d'ajustement sur ce seul run. (Meme diagnostic de fenetre que la
sur-lecture deja signalee a `nsteps` fixe dans le protocole.)

## Lecture diffusion-vs-structurel, par mode

Methode : pour un ordre donne, si l'`|%err|` **decroit nettement** avec `n` (et avec l'ordre),
la part diffuse est dominante ; s'il **plafonne** en resolution, le residu est structurel (bord
d'anneau cartesien advecte sur grille pleine, cf. `docs/PAPER_ROADMAP.md`).

- **l = 3 : part MAJORITAIREMENT diffuse ; l'ordre reduit fortement le gap.** L'`|%err|` se referme
  de facon monotone avec la resolution ET avec l'ordre. minmod : -34 % -> -22 % -> -17 % -> -12 %
  (128->384) ; vanleer (moins dissipatif) : -21 % -> -15 % -> -11 % -> -9 % ; **O5 : -14.6 % ->
  -12.4 % -> -10.3 %** (n=128->256). A chaque n, O5 ameliore strictement vanleer (le meilleur O2) :
  -14.6 vs -21.5 (n=128), -10.3 vs -11.4 (n=256). L'ordre reduit donc fortement l'ecart l=3 ; le
  residu O5 ~ -10 % decroit encore et ne plafonne pas a n=256. **Verdict : l'ordre reduit fortement
  le gap observe a O2 sur l=3 ; pas de plancher structurel visible (residu diffuse non encore epuise
  en local), mais cela reste a confirmer a plus haute resolution.**
- **l = 4 : le mode-cle. L'ordre reduit fortement le gap observe a O2 ~12 %.** A O2,
  l=4 semblait buter sur ~10-12 % qui cessait de se refermer en resolution (minmod -12.1 % a n=256
  puis -12.5 % a n=384 ; vanleer non monotone -5.5 % -> -9.5 %), ce que PR-0 lisait comme un verrou
  structurel candidat. **L'ordre 5 suggere fortement l'autre lecture : O5 amene l=4 a -4.1 % a n=256**
  (et -4.2 % a n=128), nettement SOUS le plateau O2-minmod (-12 %) et sous O2-vanleer (-5.5 %) a meme
  resolution. Mais cette lecture ne tient que sur DEUX points propres (n=128 et n=256) : le point
  intermediaire n=192 O5 donne -15.8 %, et c'est un artefact de la fenetre de fit (fenetre ouverte a
  t0=5.4, cf. tableau de tracabilite ci-dessus), pas une mesure exploitable. Sur les deux points
  propres, le residu l=4 de ~12 % vu a O2 apparait MAJORITAIREMENT comme de la diffusion residuelle
  reduite par l'ordre : le plateau ~12 % de PR-0 ressemble plutot a un plateau de diffusion d'ordre 2
  qu'a un plancher structurel dur. **Verdict : l'ordre reduit fortement le gap l=4 sous le plateau O2,
  sur deux points propres ; un plancher structurel eventuel est inferieur aux ~12 % vus a O2, mais
  reste a confirmer par n=384/512 (le point n=192 etant inutilisable a cause de sa fenetre).**
- **l = 5 : part diffuse FAIBLE, deja resolu ; O5 le confirme.** Deja a la cible des O2 a n=192
  (minmod -5 %, vanleer +4 %), l'erreur traverse zero et reste petite et de signe variable.
  **O5 : +6.9 % -> +1.8 % -> +4.7 %** (n=128->256), du meme ordre de grandeur (quelques %, signe
  variable) que les O2. Le residu est domine par le bruit de mesure / un leger sur-tir, PAS par un
  plancher structurel. **Verdict : aucun gap notable a refermer ; O5 ne fait pas apparaitre de
  plancher sur l=5.**

**Conclusion globale (avec l'axe O5).** L'ajout de l'ordre 5 (WENO5-Z + SSPRK3) suggere fortement
une lecture de la question laissee ouverte par PR-0 : **le residu l-dependant a O2 est majoritairement
de la diffusion numerique, reduite par l'ordre, plutot qu'un verrou structurel a ~12 %.** Le cas le
plus net est l=4 : son plateau apparent ~12 % a O2 tombe a ~4 % a O5 sur les deux points propres
(n=128 et n=256 ; le point n=192 a -15.8 % est un artefact de fenetre de fit, cf. tableau de
tracabilite), ce qui suggere fortement que l'essentiel de l'ecart l=4 etait de la dissipation
d'ordre 2 plutot qu'un plancher du bord d'anneau cartesien. l=3 se reduit aussi avec l'ordre
(O5 < O2 a tout n), et l=5 etait deja resolu. Autrement dit, **l'ordre reduit fortement le gap
observe a O2** : a l'horizon de mesure et aux resolutions locales, on ne voit pas de plancher
structurel residuel net (le residu O5 le plus grand est l=3 ~ -10 %, encore decroissant, donc
encore compatible avec de la diffusion). Cela AFFAIBLIT (sans la refuter) l'hypothese de PR-0 d'un
plancher structurel ~12 % derriere la PR-A "transport-wall" : aux resolutions {128,192,256} et a
l'ordre 5, le plateau O2 n'est pas robuste a la montee en ordre, et un plancher structurel eventuel
est inferieur aux ~12 % vus a O2, mais reste a confirmer par n=384/512. Deux mesures sont donc
REQUISES avant de reecrire la roadmap papier, sur ROMEO/GH200 : (1) **n=384 / n=512** (incluant O5)
pour voir si le residu O5 l=3/l=4 plafonne enfin a haute resolution (-> plancher structurel pur) ou
continue de decroitre (-> diffusion encore) ; (2) un balayage O5 a plus fin pour borner le plancher
structurel l-dependant sous le bruit de mesure.

## Ce qui a ete lance vs saute

- **Lance (grille principale O1/O2, 27 runs)** : `n in {128,192,256} x {O1 none, O2 minmod,
  O2 vanleer} x l in {3,4,5}`, `t_end=48`. Rejoue ici a l'identique de PR-0 (memes valeurs), donc
  les lignes O1/O2 du tableau sont stables. Rejouable par `sweep.py --orders none,minmod,vanleer`.
- **Lance (axe haut ordre O5, 9 runs)** : `n in {128,192,256} x O5 (weno5 + ssprk3) x l in {3,4,5}`,
  `t_end=48`. Couts par run mesures en local (CPU mono-thread) : n=128 ~ 9 s, n=192 ~ 30 s,
  n=256 ~ 88 s (WENO5-Z = stencil 5 points + 3 ghosts, SSPRK3 = 3 etages, donc plus lourd que O2 a
  meme n). L'ensemble du balayage 4 ordres x 3 n x 3 l = **36 runs en ~16,5 min** (CPU local).
  CSV complet, rejouable par `sweep.py` (defaut, qui inclut maintenant `weno5`).
- **Lance (sonde n=384 O2, 6 runs)** : `n=384 x {O2 minmod, O2 vanleer} x l in {3,4,5}` (heritee
  de PR-0 ; un run n=384 O2 ~ 143 s ; rejouable via `sweep.py --ns 384 --orders minmod,vanleer`).
  O1 saute a n=384 (l'ordre 1 reste domine par la diffusion a toute resolution).
- **Saute (a basculer sur ROMEO / GH200)** :
  - **n=384 O5** : volontairement saute en local. Extrapolation du cout n=256 O5 (~88 s) en
    (384/256)^3 ~ 3,4x -> ~300 s par run x 3 modes ~ 15 min rien que pour n=384 O5, AU-DELA de la
    limite "quelques minutes par run" du cahier des charges. A lancer sur GH200 pour prolonger la
    courbe O5 d'un cran en resolution. **FLAG ROMEO.**
  - **n=512 (tous ordres, surtout O5)** : trop lourd en local (n=512 ~ 4x le cout n=256 par run x
    le nombre de runs). Le point le plus utile maintenant : voir si le residu O5 l=3 (~ -10 % a
    n=256, encore decroissant) et l=4 (~ -4 %) PLAFONNENT enfin a n=512 (-> plancher structurel pur)
    ou continuent de se refermer (-> diffusion encore non epuisee). **FLAG ROMEO.**

## Reproduire

```bash
cd ../adc_cpp && cmake -S . -B build-py -DADC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-py -j4
cd ../adc_cases
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py        # O2 + O5 (defaut)
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --orders weno5  # O5 seul
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --quick # fumee rapide
```

Le defaut de `--orders` est maintenant `minmod,vanleer,weno5` (O5 = WENO5-Z + SSPRK3). Ajouter
`none` pour la ligne O1. L'axe O5 exige adc_cpp #88 ou plus recent (master `ca803dc`).
