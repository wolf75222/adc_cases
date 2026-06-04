# Balayage diocotron : ordre x resolution x mode (mesure PR-0)

Quantification de l'ecart de taux de croissance diocotron (numerique `adc` vs cible analytique
de Petri) en fonction de la **resolution** et de l'**ordre de reconstruction**, pour decider la
PR-A "transport-wall". Aucune physique modifiee : ce balayage REUTILISE tel quel le pipeline de
[`run.py`](run.py) (CI anneau partagee, FFT azimutale du mode `l` de `phi`, ajustement de la
phase lineaire `exp(gamma t)`, normalisation par `omega_D`, cible analytique de Petri en numpy).
Script : [`sweep.py`](sweep.py). Donnees brutes : `out/diocotron/sweep_results.csv`.

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

## Axe ordre : ce qui est REELLEMENT atteignable depuis Python

L'axe ordre vise par le cahier des charges etait {O2, **O5 WENO5-Z**}. Verification faite sur le
solveur (master `adc_cpp` 30f6dfd) : **WENO5-Z n'est PAS atteignable depuis Python** par le
chemin du cas diocotron.

- `adc.System.add_block` route le limiteur vers la dispatch runtime `make_block`
  (`include/adc/runtime/block_builder.hpp`), qui n'instancie QUE `NoSlope` (ordre 1), `Minmod`
  (ordre 2 TVD) et `VanLeer` (ordre 2) pour le flux Rusanov. La politique `Weno5` (ordre 5,
  `weno5z`) existe bien dans le coeur (`numerics/reconstruction.hpp`, `numerics/spatial_operator.hpp`)
  mais **aucun `build_block<Weno5, ...>` n'est instancie** : aucune chaine `"weno5"` n'est acceptee.
- Les chemins `add_dynamic_block` et `add_compiled_block` (DSL) plafonnent aussi a l'ordre 2
  (`recon_id` 0/1/2 = none/minmod/vanleer ; le compiled block delegue a `make_block`).
- `adc.Explicit` est SSPRK2 (ordre 2 en temps) ; seul `substeps` est reglable, pas SSPRK3.

Verification empirique : `add_block(spatial=adc.Spatial(limiter="weno5"))` leve
`System : limiter inconnu 'weno5'`. La doc roadmap (`docs/PAPER_ROADMAP.md` panier 1, "adc.Spatial
expose deja WENO5-Z") **anticipe** ce cablage ; il n'est pas present sur master. Cabler `Weno5`
(et SSPRK3) dans `make_block` est du code COEUR, hors perimetre de cette PR mesure.

**Axe ordre effectivement balaye** : `{O1 none, O2 minmod, O2 vanleer}`. C'est suffisant pour
isoler diffusion-vs-structurel : la diffusion numerique decroit (a) en montant la resolution et
(b) en montant l'ordre / en baissant la dissipation du limiteur (none -> minmod -> vanleer, du
plus diffusif au moins diffusif a ordre 2). Un point O5 reste a obtenir apres cablage coeur
(suivi ROMEO, ci-dessous).

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

(n=192 O2 minmod = ligne d'ancrage, reproduit le README. n=384 = sonde au-dela de la grille
principale, 1 run ~ 143 s ; voir "lance vs saute".)

## Lecture diffusion-vs-structurel, par mode

Methode : pour un ordre donne, si l'`|%err|` **decroit nettement** avec `n` (et avec l'ordre),
la part diffuse est dominante ; s'il **plafonne** en resolution, le residu est structurel (bord
d'anneau cartesien advecte sur grille pleine, cf. `docs/PAPER_ROADMAP.md`).

- **l = 3 : part MAJORITAIREMENT diffuse.** L'`|%err|` se referme de facon monotone avec la
  resolution ET avec l'ordre. minmod : -34 % -> -22 % -> -17 % -> -12 % (128->384) ; vanleer,
  moins dissipatif a meme ordre : -21 % -> -15 % -> -11 % -> -9 %. La courbe ne plafonne pas
  encore a n=384 : l'essentiel de l'ecart est de la diffusion numerique, le plancher structurel
  eventuel est < 10 % et non encore atteint en local.
- **l = 4 : part diffuse forte A BASSE resolution, mais PLATEAU structurel a haute resolution.**
  C'est le mode le plus diffuse au depart (O1 n=128 quasi nul, 0.005). La resolution + l'ordre
  referment beaucoup (minmod -33 % -> -27 % -> -12 % de 128 a 256), MAIS de n=256 a n=384 minmod
  PLAFONNE (-12.1 % -> -12.5 %, plat / leger recul). Van Leer confirme l'absence de convergence
  propre a haute resolution (-5.5 % a n=256 puis -9.5 % a n=384, non monotone). Donc apres avoir
  paye le gros de la diffusion, **l=4 bute sur un plancher ~10-12 % qui ne se referme plus en
  resolution** : signature du **verrou structurel** (bord d'anneau cartesien) sur le mode le plus
  instable, exactement le l-dependant attendu par la roadmap.
- **l = 5 : part diffuse FAIBLE, pas de plancher.** Deja a la cible des n=192 (minmod -5 %,
  vanleer +4 %), l'erreur TRAVERSE zero et reste petite et de signe variable a n=256/384 (minmod
  +2.3 % -> +2.6 %, vanleer +8.3 % -> +3.3 %). Le residu est domine par le bruit de mesure / un
  leger sur-tir, PAS par un plancher structurel : la diffusion est negligeable des le depart pour
  ce mode a plus grande longueur d'onde radiale.

**Conclusion globale.** L'ordre et la resolution referment une part SUBSTANTIELLE de l'ecart sur
les trois modes (la majeure partie est de la diffusion numerique, comme attendu d'un FV d'ordre
2). Mais le residu est l-dependant : **l=4 conserve un plancher ~10-12 % qui cesse de se refermer
au-dela de n=256**, ce qui est le signal quantitatif d'un verrou STRUCTUREL (advection de l'anneau
sur grille cartesienne pleine, transport sans bord embedded), alors que l=3 reste diffusion-limite
et l=5 est deja resolu. Ce plancher l=4 est l'argument chiffre justifiant la PR-A "transport-wall"
(porter le cut-cell du Poisson vers l'operateur hyperbolique). Deux confirmations restent a faire,
toutes deux sur ROMEO/GH200 : (1) **n=512** pour verrouiller le plateau l=4 a haute resolution ;
(2) un point **O5 WENO5-Z** une fois l'ordre 5 cable cote coeur, le test le plus discriminant pour
borner la diffusion residuelle et isoler le plancher structurel pur.

## Ce qui a ete lance vs saute

- **Lance (grille principale, 27 runs)** : `n in {128,192,256} x {O1 none, O2 minmod, O2 vanleer}
  x l in {3,4,5}`, `t_end=48`. 27 runs en ~10 min (CPU local, mono-thread). CSV complet, entierement
  rejouable par `sweep.py` (defaut).
- **Lance (sonde n=384, 6 runs)** : `n=384 x {O2 minmod, O2 vanleer} x l in {3,4,5}` pour tester
  la convergence au-dela de n=256 (un run unique n=384 ~ 143 s, dans la limite "quelques minutes"
  du cahier des charges ; rejouable via `sweep.py --ns 384 --orders minmod,vanleer`). Decisif pour
  voir le PLATEAU l=4. O1 saute a n=384 (peu informatif : l'ordre 1 est domine par la diffusion a
  toute resolution).
- **Saute (a basculer sur ROMEO)** :
  - **n=512** (et le complement n=384) : trop lourds pour un balayage complet en local
    (n=512 ~ 4x le cout n=256 par run x 9-27 runs). A lancer sur GH200 (l'integrateur peut le
    faire), pour confirmer / infirmer le plateau structurel l-dependant a haute resolution.
  - **O5 WENO5-Z + SSPRK3** : NON atteignable depuis Python sur master (voir ci-dessus). Exige
    d'abord un cablage `build_block<Weno5,...>` (et SSPRK3) cote COEUR `adc_cpp`, hors perimetre
    de cette PR mesure. Une fois cable, rejouer ce meme balayage avec l'ordre O5 donnera le point
    manquant le plus discriminant pour separer diffusion residuelle et plancher structurel.

## Reproduire

```bash
cd ../adc_cpp && cmake -S . -B build-py -DADC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-py -j4
cd ../adc_cases
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py        # grille principale
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --quick # fumee rapide
```
