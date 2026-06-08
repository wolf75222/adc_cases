# Balayage diocotron : ordre x resolution x mode (mesure, O1/O2 + haut ordre O5)

Quantification de l'ecart de taux de croissance diocotron (numerique `adc` vs cible analytique
de Petri) en fonction de la **resolution** et de l'**ordre de reconstruction**, pour decider la
PR-A "transport-wall". Aucune physique modifiee : ce balayage reutilise tel quel le pipeline de
[`run.py`](run.py) (CI anneau partagee, FFT azimutale du mode `l` de `phi`, ajustement de la
phase lineaire `exp(gamma t)`, normalisation par `omega_D`, cible analytique de Petri en numpy).
Script : [`sweep.py`](sweep.py). Donnees brutes : `out/diocotron/sweep_results.csv`.

Cette version etend PR-0 (qui balayait {O1, O2-minmod, O2-vanleer}) avec l'axe **haut ordre O5 =
WENO5-Z + SSPRK3**, desormais atteignable depuis Python (adc_cpp #88, master `ca803dc`). Le but de
l'axe O5 est d'eclairer la question laissee ouverte par PR-0 : le residu l-dependant a O2 est-il de
la diffusion (refermable par l'ordre) ou un plancher structurel du bord d'anneau cartesien ?

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
  n=192** (l=3 -22 %, l=4 -27 %, l=5 -5 %), ce qui ancre la mesure. C'est un reglage de boucle,
  pas un nouvel observable.

## Axe ordre : ce qui est atteignable depuis Python

L'axe ordre vise par le cahier des charges etait {O2, **O5 WENO5-Z**}. Au moment de PR-0 (master
`adc_cpp` 30f6dfd) WENO5-Z et SSPRK3 n'etaient pas atteignables depuis le chemin du cas diocotron.
**Depuis adc_cpp #88 (master `ca803dc`), ils le sont** par le chemin natif `add_block` :

- `adc.Spatial(limiter="weno5", flux="rusanov")` : reconstruction WENO5-Z (ordre 5 en zone lisse,
  stencil 5 points, 3 ghosts) ; `make_block` instancie maintenant la politique `Weno5`. Seul le
  chemin natif `add_block` l'expose (les chemins .so AOT/JIT allouent 2 ghosts et la rejettent).
- `adc.Explicit(method="ssprk3")` : integrateur SSPRK3 (Shu-Osher 3 etages, ordre 3). On l'apparie
  a WENO5-Z car appairer un ordre eleve en espace a SSPRK2 (ordre 2 en temps) briderait l'ordre
  effectif. La cle d'ordre `weno5` du balayage aiguille donc les deux briques (cf. `sweep.py`
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
fortement bornee, donc ce qui reste a O5 est le candidat le plus credible au plancher structurel.

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
| 512 | O2 minmod    | 0.684 (-11.4 %) | 0.807 (-11.4 %) | 0.727 (+5.9 %) |
| 128 | O2 vanleer   | 0.606 (-21.5 %) | 0.781 (-14.4 %) | 0.685 (-0.4 %) |
| 192 | O2 vanleer   | 0.658 (-14.8 %) | 0.752 (-17.5 %) | 0.714 (+3.9 %) |
| 256 | O2 vanleer   | 0.684 (-11.4 %) | 0.862 (-5.5 %) | 0.744 (+8.3 %) |
| 384 | O2 vanleer   | 0.702 (-9.1 %) | 0.825 (-9.5 %) | 0.710 (+3.3 %) |
| 512 | O2 vanleer   | 0.700 (-9.4 %) | 0.821 (-10.0 %) | 0.721 (+4.9 %) |
| 128 | O5 weno5     | 0.659 (-14.6 %) | 0.874 (-4.2 %)  | 0.735 (+6.9 %) |
| 192 | O5 weno5     | 0.677 (-12.4 %) | 0.768 (-15.8 %) | 0.700 (+1.8 %) |
| 256 | O5 weno5     | 0.692 (-10.3 %) | 0.875 (-4.1 %)  | 0.719 (+4.7 %) |
| 384 | O5 weno5     | 0.706 (-8.6 %)  | 0.828 (-9.2 %)  | 0.702 (+2.2 %) |
| 512 | O5 weno5     | 0.704 (-8.8 %)  | 0.823 (-9.7 %)  | 0.715 (+4.0 %) |

(n=192 O2 minmod = ligne d'ancrage, reproduit le README. Les lignes O1/O2 sont rejouees ici et
reproduisent PR-0 a l'identique : en particulier n=384 et n=512 O2 sur ROMEO redonnent les valeurs
n=384 O2 deja au tableau au centieme pres, ce qui ancre la chaine de build ROMEO sur PR-0. O5 =
WENO5-Z + SSPRK3, lance en local n=128/192/256 ; **les lignes n=384 et n=512 (tous ordres) sont
mesurees sur ROMEO** (x64cpu amd EPYC, job SLURM 639912, cf. "lance vs saute"). n=384 O2 = sonde au-dela
de la grille principale heritee de PR-0.)

> **Lecture rapide des lignes haute resolution (n=384/512).** Le point qui change la lecture est
> O5 l=4 : a basse resolution il valait ~ -4 % sur les deux points propres (n=128, n=256), ce que
> PR-0 lisait comme "diffusion presque epuisee". A haute resolution O5 l=4 ne descend pas vers 0 %
> dans ces mesures :
> il vaut -9.2 % (n=384) puis -9.7 % (n=512). attention : ces deux points ont une fenetre de fit qui
> s'ouvre tot (t0 = 6.3 et 5.4, cf. tableau de tracabilite), exactement le meme defaut que le point
> n=192 deja signale ; ils ne sont donc pas directement comparables aux deux points propres basse
> resolution. Le signal le plus propre est l=3 O5, dont la fenetre s'ouvre de facon homogene
> (t0 ~ 6.5) a tout n : il passe de -10.3 % (n=256) a -8.6 % (n=384) puis -8.8 % (n=512), soit un
> aplatissement net autour de -9 % a haute resolution. Voir le verdict prudent plus bas.

### Tracabilite : fenetre de fit des points O5 l=4 (basse ET haute resolution)

Le CSV (`out/diocotron/sweep_results.csv`) ecrit pour chaque ligne les bornes de la fenetre de
`fit_linear_phase` : indices `fit_i0..fit_i1` et temps `fit_t0..fit_t1`. Voici ces bornes pour les
points O5 l=4. Les trois premiers (n=128/192/256) sont les valeurs locales de PR-0 ; les deux
derniers (n=384/512) viennent du job ROMEO 639912 (colonnes de fenetre directement issues du CSV
fusionne `sweep_hires_merged.csv`) :

| n | gamma_num | %err | fenetre i0..i1 | fenetre t0..t1 | fenetre propre ? |
|---|---|---|---|---|---|
| 128 | 0.8736 | -4.2 % | 269..528 | 20.8..41.1 | oui (t0 tardif) |
| 192 | 0.7675 | -15.8 % | 108..703 | **5.4**..35.1 | non (t0 precoce) |
| 256 | 0.8745 | -4.1 % | 357..1095 | 13.3..40.8 | oui (t0 tardif) |
| 384 | 0.8282 | -9.2 % | 257..1594 | **6.3**..38.7 | non (t0 precoce) |
| 512 | 0.8229 | -9.7 % | 295..2126 | **5.4**..38.6 | non (t0 precoce) |

Lecture directe de la colonne fenetre : les deux points haute resolution (n=384, n=512) ont une
fenetre qui s'ouvre tot (t0 = 6.3 et 5.4), exactement comme le point n=192 deja signale comme
artefact (t0 = 5.4). Autrement dit, le point intermediaire n=192 n'etait pas un accident isole :
des qu'on monte en n, la fenetre lineaire de `fit_linear_phase` tend a s'ouvrir tot pour O5 l=4.
Les deux seuls points O5 l=4 a fenetre tardive (n=128 t0=20.8 et n=256 t0=13.3) restent donc les
seuls "propres" du jeu l=4 ; les nouveaux points n=384/512 ne peuvent ni confirmer ni infirmer
proprement la valeur ~ -4 % de ces deux-la.

**Pourquoi l=3 O5 est le signal le plus fiable a haute resolution.** Contrairement a l=4, le mode
l=3 a une fenetre de fit qui s'ouvre de facon homogene a tout n (t0 ~ 6.5 : n=384 [t6.5..38.1],
n=512 [t6.5..39.9]). Sa courbe en n est donc comparable point a point, sans le biais de fenetre qui
affecte l=4. C'est sur l=3 que la lecture diffusion-vs-structurel est la moins ambigue (cf. verdict).

## Lecture diffusion-vs-structurel, par mode

Methode : pour un ordre donne, si l'`|%err|` **decroit nettement** avec `n` (et avec l'ordre),
la part diffuse est dominante ; s'il **plafonne** en resolution, le residu est structurel (bord
d'anneau cartesien advecte sur grille pleine, cf. `docs/PAPER_ROADMAP.md`).

- **l = 3 : le signal le plus propre. L'ordre reduit le gap, mais a haute resolution O5 aplatit
  autour de -9 %.** L'`|%err|` se referme avec la resolution et avec l'ordre. minmod : -34 % -> -22 %
  -> -17 % -> -12 % -> -11 % (128->512) ; vanleer (moins dissipatif) : -21 % -> -15 % -> -11 % ->
  -9 % -> -9 % ; **O5 : -14.6 % -> -12.4 % -> -10.3 % -> -8.6 % -> -8.8 %** (n=128->512). A chaque n,
  O5 ameliore strictement vanleer (le meilleur O2). Le point neuf de cette mesure : entre n=384 et
  n=512, **le residu O5 l=3 ne montre plus de fermeture nette** (-8.6 % puis -8.8 %, soit un ecart dans le bruit de
  mesure) ; et c'est le mode dont la fenetre de fit est homogene a tout n (t0 ~ 6.5), donc cet
  aplatissement n'est pas un artefact de fenetre. **Verdict : sur le mode le mieux mesure, l'ordre
  reduit d'abord fortement le gap puis l'`|%err|` O5 semble plafonner autour de -9 % a haute resolution. Cela
  Suggéré un plancher residuel de l'ordre de ~9 % qui ne se referme pas par la resolution (candidat
  structurel), plutot qu'une diffusion encore en train de s'epuiser ; reste a confirmer (un seul cran
  n=384 -> n=512 plat ne suffit pas a exclure une convergence tres lente).**
- **l = 4 : le mode-cle. Le -4 % O5 basse resolution ne se reproduit pas a haute resolution ; mais
  les points haute resolution sont biaises par leur fenetre.** A O2, l=4 bute sur ~10-12 % qui ne se
  referme pas en resolution (minmod -12.1 % a n=256, -12.5 % a n=384, -11.4 % a n=512 ; vanleer
  -5.5 % -> -9.5 % -> -10.0 %), ce que PR-0 lisait comme candidat structurel. PR-0 esperait
  l'autre lecture via O5 (-4.1 % a n=256, -4.2 % a n=128, "diffusion presque epuisee"). **La mesure
  haute resolution ROMEO ne reproduit pas ce -4 % : O5 l=4 vaut -9.2 % a n=384 et -9.7 % a n=512** -
  c'est-a-dire qu'il remonte vers la bande des O2 (~-10 a -11 %) et de l=3 O5 (~-9 %) au lieu de
  tendre vers 0. prudence de premiere importance : ces deux points haute resolution ont une fenetre
  de fit qui s'ouvre tot (t0 = 6.3 et 5.4, cf. tableau de tracabilite), exactement le defaut qui
  rendait le point n=192 inutilisable ; ils sous-lisent donc probablement un peu la pente, comme
  n=192. On ne peut donc pas affirmer que -9.5 % est la "vraie" valeur asymptotique de l=4 O5. Ce
  qu'on peut dire honnetement : (a) le -4 % observe sur les deux seuls points propres (n=128, n=256)
  ne se reproduit a aucune des deux resolutions superieures ; (b) des qu'on monte en n, la fenetre de
  l=4 s'ouvre tot et le %err mesure se loge dans la meme bande ~ -9 a -10 % que l=3 et que les O2.
  **Verdict : la lecture optimiste de PR-0 (l=4 O5 -> ~-4 %, donc diffusion) est affaiblie - elle ne
  tient que sur deux points propres basse resolution et ne survit pas a la montee en resolution. Les
  points n=384/512 (~-9.5 %) sont coherents avec un plancher de l'ordre de ~9-10 % du meme ordre que
  l=3, mais leur fenetre precoce interdit d'en faire une mesure de plancher fiable. Conclusion l=4 :
  pas de fermeture vers 0 % a haute resolution -> le residu l=4 ne se comporte pas comme une diffusion
  qui s'epuise ; un plancher (structurel) de ~9-10 % est le candidat le plus coherent, a confirmer
  par un diagnostic de fenetre robuste sur l=4 (ouvrir le fit plus tard).**
- **l = 5 : part diffuse faible, deja resolu ; la haute resolution le confirme.** Deja a la cible
  des O2 a n=192 (minmod -5 %, vanleer +4 %), l'erreur traverse zero et reste petite et de signe
  variable. **O5 : +6.9 % -> +1.8 % -> +4.7 % -> +2.2 % -> +4.0 %** (n=128->512), du meme ordre de
  grandeur (quelques %, signe variable, sans tendance nette) que les O2 (n=512 : minmod +5.9 %,
  vanleer +4.9 %). Le residu est domine par le bruit de mesure / un leger sur-tir, pas par un
  plancher structurel ni par une diffusion residuelle. **Verdict : aucun gap notable a refermer ;
  ni l'ordre ni la haute resolution ne font apparaitre de plancher sur l=5.**

**Conclusion globale (avec l'axe O5 ET la confirmation haute resolution ROMEO).** La mesure
n=384/512 (job ROMEO 639912, x64cpu) reverse la lecture optimiste que PR-0 tirait des deux points
propres basse resolution. PR-0 lisait : "l=4 O5 tombe a ~-4 % -> le residu est de la diffusion, pas
un plancher". **La haute resolution ne reproduit pas ce -4 %.** Sur les deux modes ou la mesure est
exploitable a haute resolution :

- **l = 3 (le mieux mesure, fenetre homogene a tout n)** : l'`|%err|` O5 plafonne autour de -9 %
  (-10.3 % a n=256, puis -8.6 % a n=384 et -8.8 % a n=512 - plat entre les deux derniers crans). Ce
  n'est pas le comportement d'une diffusion qui s'epuise (qui continuerait de se refermer), mais
  plutot celui d'un plancher residuel candidat (pas encore une preuve definitive).
- **l = 4** : le -4 % O5 basse resolution ne se reproduit a aucune des deux resolutions superieures
  (O5 l=4 = -9.2 % a n=384, -9.7 % a n=512). Il remonte dans la meme bande ~ -9 a -10 % que l=3 et
  que les O2. La reserve majeure : ces deux points l=4 ont une fenetre de fit precoce (t0 = 6.3 et
  5.4, comme le point n=192 deja ecarte), donc on ne peut pas en faire une valeur de plancher fiable
  - on peut seulement dire que l=4 ne tend pas vers 0 % dans ces mesures a haute resolution.
- **l = 5** : reste petit et de signe variable (quelques %), deja resolu, sans plancher.

**Verdict prudent (l=4, la question posee).** A la question "le residu l=4 O5 continue-t-il de se
refermer vers 0 % (-> c'etait de la diffusion, pas de plancher dur) ou plafonne-t-il a un certain %
(-> plancher de cette taille) ?", la mesure haute resolution repond : **il ne montre pas de fermeture
vers 0 % dans ces mesures**. Le -4 % de PR-0 etait un artefact des deux seuls points propres basse
resolution ; a n=384 et n=512 l=4 O5 se loge a ~-9.5 %, du meme ordre que le palier l=3 (~-9 %) qui,
lui, est mesure proprement et plat sur un cran (n=384 -> n=512). La lecture la plus coherente avec
l'ensemble des donnees est donc que les donnees SuggéréNT un plancher residuel l-dependant de l'ordre
de **~9-10 %** a l'ordre 5, qui ne montre pas de fermeture vers 0 par la resolution dans ces mesures -
candidat structurel du bord d'anneau cartesien vise par la PR-A "transport-wall", pas encore une
preuve definitive. Cette lecture re-ouvre l'hypothese de plancher de PR-0 (que l'axe O5 basse
resolution avait semble affaiblir), en situant sa taille plausible vers ~9-10 % a l'ordre 5 (contre
~12 % vus a O2), sous reserve des deux limites ci-dessous.

Ce verdict reste a confirmer et NE justifie a lui seul aucune reecriture de la roadmap papier :
(1) le plateau l=3 ne tient pour l'instant que sur un cran plat (n=384 -> n=512) ; il faudrait soit
un n=768/1024, soit deux horizons `t_end` differents, pour exclure une convergence simplement tres
lente ; (2) les points l=4 haute resolution sont biaises par leur fenetre de fit precoce - avant de
chiffrer un plancher l=4, il faut un diagnostic de fenetre robuste (ouvrir le fit plus tard, ou caler
la fenetre sur la phase exponentielle propre comme aux points n=128/256). Autrement dit la question
"diffusion vs structurel" penche maintenant, sur la base de cette mesure, du cote **structurel
(plancher ~9-10 % a l'ordre 5)** plutot que diffusion - mais ce basculement par rapport au verdict
prudent de PR-0 Suggéré seulement, et reste a confirmer.

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
  de PR-0 ; un run n=384 O2 ~ 143 s en local ; rejouable via `sweep.py --ns 384 --orders minmod,vanleer`).
  O1 saute a n=384 (l'ordre 1 reste domine par la diffusion a toute resolution).
- **Lance sur ROMEO (haute resolution n=384/512, 18 runs)** : `n in {384, 512} x {O2 minmod,
  O2 vanleer, O5 weno5} x l in {3,4,5}`, `t_end=48`, job SLURM **639912** (partition `short`,
  `--constraint=x64cpu`, amd EPYC 192 coeurs, compte `r250127`). Les 18 runs (mono-thread chacun)
  lances en parallele sur un noeud ; tous rc=0, aucun NaN, `t_final = 48` atteint sans toucher le
  garde-fou `max_steps=4000` (n=512 demande ~2600 pas, donc marge confortable). Le module `_adc` est
  rebati sur le noeud de login ROMEO (Spack : python@3.10.14 + numpy@1.26.4 + pybind11@2.13.5 +
  cmake@3.31.8 + gcc@11.4.1) a partir de `adc_cpp` master `5bb7208` ; **les lignes n=384 et n=512 O2
  reproduisent les valeurs n=384 O2 de PR-0 au centieme pres**, ce qui valide la chaine ROMEO.
  Temps de paroi par run mesures (CPU EPYC, 18 puis 9 puis 3 runs concurrents -> contention de bande
  passante a n=512) :
  - n=384 O2 (minmod/vanleer) : ~172-182 s ; n=384 O5 (weno5+ssprk3) : ~249-258 s.
  - n=512 O2 (minmod/vanleer) : ~653-716 s ; n=512 O5 : ~833-880 s (point le plus lourd : n=512 O5
    l=4 = 880 s).
  - Job complet (18 runs en parallele) : ~14,5 min de paroi. Rejouable par `sweep.py --ns 384,512
    --orders minmod,vanleer,weno5`.
- **Toujours saute** : **n=384/512 O1 none** (l'ordre 1 reste domine par la diffusion a toute
  resolution, sans interet pour la question diffusion-vs-structurel). Pistes ouvertes par cette
  mesure, non faites ici (cf. verdict) : (a) **n=768/1024** ou **deux horizons `t_end`** pour exclure
  que le plateau l=3 O5 ~ -9 % soit une convergence tres lente ; (b) un **diagnostic de fenetre
  robuste pour l=4** (fenetre de fit calee tard, comme aux points propres n=128/256) avant de chiffrer
  un plancher l=4 - les points l=4 haute resolution ont une fenetre precoce qui sous-lit la pente.

## Reproduire

```bash
cd ../adc_cpp && cmake -S . -B build-py -DADC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-py -j4
cd ../adc_cases
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py        # O2 + O5 (defaut)
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --orders weno5  # O5 seul
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --quick # fumee rapide
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py \
  --ns 384,512 --orders minmod,vanleer,weno5         # haute resolution (lourd : ROMEO)
```

Le defaut de `--orders` est maintenant `minmod,vanleer,weno5` (O5 = WENO5-Z + SSPRK3). Ajouter
`none` pour la ligne O1. L'axe O5 exige adc_cpp #88 ou plus recent (master `ca803dc`). `sweep.py`
n'a pas ete modifie pour la haute resolution : il parametre deja `--ns` et `--orders`, et le
garde-fou `max_steps=4000` couvre n=512 (~2600 pas a `t_end=48`).

Sur ROMEO (haute resolution, job 639912 ci-dessus) : le balayage est CPU (Poisson + transport, pas
de GPU), partition `short` `--constraint=x64cpu`, et le module `_adc` se rebatit sur le noeud de
login depuis `adc_cpp` (Spack : python@3.10.14 + numpy@1.26.4 + pybind11@2.13.5 + cmake@3.31.8 +
gcc@11.4.1) via la meme commande `cmake -S . -B build-py -DADC_BUILD_PYTHON=ON
-DCMAKE_BUILD_TYPE=Release`. Les 18 runs (n=384/512 x 3 ordres x 3 modes) tournent mono-thread en
parallele sur un noeud (192 coeurs).
