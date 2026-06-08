# Hoffart diocotron -- premiere mesure quantitative sur le chemin `system-schur`

Premiere table de taux de croissance produite sur le chemin **fidele au papier**
(arXiv:2510.11808, section 5.3) : volumes finis uniformes, **Strang(SSPRK3 + CondensedSchur,
theta=0.5)**, source electrostatique/Lorentz condensee par Schur, vitesse de derive initiale du
papier. **Statut : reproduction non etablie -- deficit structurel confirme.**

## Setup
- Moteur : `system-schur` (System uniforme, mono-rang), geometrie `square` (cartesienne pleine ;
  le verdict adc_cpp acte que la cut-cell est **sans effet** sur le taux).
- Schema temporel : `adc.Strang(hyperbolic=adc.Explicit(method="ssprk3"),
  source=adc.CondensedSchur(theta=0.5, alpha=alpha))` -- splitting symetrique d'ordre 2 du papier
  + RK3 peu dissipatif (le chemin production accepte ssprk3 via adc_cpp PR #230).
- Spatial : WENO5-Z + Rusanov, variables conservatives. `dt = 1e-3`. Limite froide (`theta_p=0`).
- Parametres papier : `R=16, r0=6, r1=8, rho_max=1, rho_min=1e-6, beta=1e6, delta=0.1`,
  `alpha = omega = beta^2 = 1e12`.
- Observable : `|c_l(t)|` = amplitude du coefficient de Fourier azimutal `l` de `phi` sur le
  cercle `r=r0=6` ; taux = regression lineaire de `log|c_l|` dans la fenetre verbatim du papier.
- SHA : `adc_cpp` `06e3b90` (branche `feat/ssprk3-production-path`, PR #230) ;
  `adc_cases` `a50b539` (branche `feat/hoffart-strang-fidelity`, PR #21). Normalisation brute
  (aucun facteur `2pi/rhobar` : modele complet, pas le chemin reduit ExB).

## 1. Scan de resolution (l=3, fenetre [0.40, 0.70])

| n | 64 | 96 | 128 | 192 |
|---|---|---|---|---|
| gamma_3 mesure | 0.0198 | 0.0270 | 0.0321 | 0.0351 |

`gamma_3` **converge vers ~0.035** (papier 0.772) : il monte legerement avec la resolution
(diffusion numerique qui diminue) mais **plafonne ~22x en dessous du papier**. Ce n'est donc
pas un probleme de sous-resolution.

## 2. Table complete (n=192, fenetres papier)

| l | fenetre fit | gamma mesure | gamma papier | erreur |
|---|---|---|---|---|
| 3 | [0.40, 0.70] | 0.0351 | 0.772 | -95.5% |
| 4 | [0.60, 0.75] | 0.0329 | 0.911 | -96.4% |
| 5 | [1.15, 1.35] | 0.1153 | 0.683 | -83.1% |

Tous les modes accusent un deficit de **~85 a 96 %** : les taux mesures sont un ordre de
grandeur trop petits.

## 3. Diagnostic de trajectoire (taux non sature, onset retarde)

Facteur de croissance `|c_3(t)| / |c_3(0)|` et taux local en fenetre glissante (l=3, n=128,
jusqu'a t=3.0) :

| t | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| `|c_3|/|c_3(0)|` | 1.01 | 1.03 | 1.06 | 1.11 | 1.17 | 1.24 |
| taux local `d log|c_3|/dt` | 0.029 | 0.058 | 0.081 | 0.097 | 0.108 | -- |

Le taux local **augmente regulierement** (0.03 -> 0.11) mais reste **~7x sous 0.772 a t=2.5**,
alors que le papier attend deja 0.77 dans [0.40, 0.70]. La croissance n'est donc pas non plus un
simple decalage de fenetre : l'onset de l'instabilite est **fortement retarde / amorti** dans la
phase precoce, puis s'accelere graduellement sans atteindre le taux papier.

## Verdict

Le deficit est **structurel**, pas un artefact de :
- **resolution** -- gamma_3 converge vers ~0.035 (ne tend pas vers 0.77 quand n croit) ; **concorde
  avec le verdict GH200 du depot** (`adc_cpp docs/HOFFART_GEOMETRY_VERDICT.md`, #236 : -95% a n=256 ET
  n=384, plateau ~0.037 resolution-independant). Cette mesure laptop reproduit donc le resultat GH200.
- **geometrie** -- #236 acte que square == staircase == cutcell donnent le même taux (cut-cell sans effet) ;
- **fenetre / timing** -- meme en fenetre tardive [1.0, 1.4] ou glissante jusqu'a t=2.5, le taux
  reste ~10x sous le papier.

Causes écartées par ce travail (donnees nouvelles vs #236, qui ne testait ni Gauss ni temperature) :
1. ~~**Politique de Gauss (R0)**~~ -- **écartée** (section 4 : `evolve ~= restart`).
2. ~~**Limite froide**~~ (`theta_p`) -- **écartée** : scan temperature (l=3, n=128) `theta_p=0 -> 0.0321`,
   `0.25 -> 0.0290`, `1.0 -> 0.0280` : ajouter de la pression empire legerement, ne recupere rien.

Cause restante : **sur-amortissement de l'operateur spatial** sur le chemin cartesien (pas une
non-positivite -- distinction importante vs le chemin polaire). Diagnostic complet :
`adc_cpp docs/HOFFART_SPATIAL_DIAGNOSTICS.md` (#238, workflow + revue adversariale).

correction (vs version precedente de ce doc qui parlait de "reconstruction non positive") : sur le
chemin cartesien, **min(rho) reste positif** au cours du run (mesure : 6.7e-7 au plancher 1e-6, jamais
negatif, jamais NaN). Le cartesien NE diverge pas -- il est sur-amorti. La non-positivite / le blow-up
appartiennent au chemin polaire (`IsothermalFluxPolar`, metrique `1/r`, source `1/rho`), pas ici.
Consequence (preuve dans #238 section 3) : un correctif de positivite (plancher / Zhang-Shu) est
**inerte sur le taux cartesien** (les cellules sont deja positives ; Zhang-Shu ne mord que sur le fond
sans signal). Le candidat reel du sur-amortissement = **dissipation Rusanov** `~ alpha*(U_R-U_L)`
proportionnelle au saut 1e6 au contact d'anneau (le flux numerique, pas la reconstruction WENO5-Z qui
ne s'effondre pas). Tester un flux moins dissipatif a 3-var = chantier C++ (`hll` non expose par le
DSL ; hllc/roe exigent une pression absente en isotherme froid). Le modele réduit ExB scalaire
reproduit la cible (+0.2% l=4) car il n'a NI reconstruction de moment NI Rusanov sur le moment.
**Reserve metrologique** : un facteur 2 pi est omis sur le chemin cartesien-Schur (`NORMALIZATION.md`)
-> une part du deficit pourrait etre non physique, a clore avant toute conclusion causale.

## 4. Experience R0 : GaussPolicy `restart` vs `evolve` -> **R0 écarté**

Le finding R0 du design (`adc_cpp docs/AMR_CONDENSED_SCHUR_DESIGN.md`, qualifie de "decisive fact" et
**gate de toute la Phase C**) postule que le `solve_fields` de tete de pas re-resout Gauss
(`-Delta phi = alpha rho`) et **ecrase** le `phi` evolue par l'etage Schur, tuant la dynamique
restart-free `-Delta phi` du papier. On a implemente le mecanisme `System.set_gauss_policy` :
- `restart` (defaut) : re-resout Gauss a chaque pas (historique, bit-identique) ;
- `evolve` : apres `phi^0`, `solve_fields` NE re-resout plus le Poisson -- l'etage Schur fait evoluer
  `phi` in-place dans `ell_phi()`, reproduisant l'evolution `-Delta phi` sans restart du papier.

Mesure (n=128, fenetres papier ; cf. `adc_cpp` PR GaussPolicy) :

| l | restart gamma | evolve gamma | papier | evolve/restart |
|---|---|---|---|---|
| 3 | 0.0321 | 0.0357 | 0.772 | 1.11x |
| 4 | ~0 (-0.005) | ~0 (-0.008) | 0.911 | -- |
| 5 | 0.1070 | 0.1091 | 0.683 | 1.02x |

**Verdict R0 : écarté.** `evolve` ne releve le taux que de 1.0 a 1.6x sur tous les modes, restant
~10-20x sous le papier. La contrainte de Gauss discrete est approximativement conservee par le
transport, donc la re-imposer (`restart`) est quasi un no-op vs l'evolution `-Delta phi`. **Le
deficit structurel n'est pas la politique de Gauss.** Consequence forte : **conditionner la Phase C
(Schur-sur-AMR) a R0 etait mal oriente -- l'AMR-Schur ne corrigera pas le taux**. Note : `restart`
est bit-identique a la baseline sans GaussPolicy (gamma_3=0.0321 a l'identique) -> NO-default-change.

**Verdict R0 : écarté** (ne corrige pas le taux).

## 5. Scans de robustesse : contraste et beta -> deux causes de plus écartées

Deux scans supplementaires (l=3, n=128, Strang+Schur) isolent encore le verrou. **Lire la tendance,
pas la valeur absolue** (la cible 0.772 ne vaut que pour les parametres papier).

**Contraste** (rho_min varie, rho_max=1) :

| rho_min | contraste | gamma_3 | min(rho) sur le run |
|---|---|---|---|
| 1e-6 | 1e6 | 0.0321 | 6.7e-7 (positif) |
| 1e-4 | 1e4 | 0.0321 | 6.7e-5 |
| 1e-2 | 1e2 | 0.0300 | 6.7e-3 |
| 1e-1 | 1e1 | 0.0214 | 6.3e-2 |

gamma_3 ne remonte pas quand le contraste baisse (plat, voire plus bas), et **min(rho) reste positif**.
Reserve (revue #238) : ce scan est confondu (monter rho_min change aussi la charge de fond
`alpha*rho_min`) -- mais le resultat plat + positif confirme : **pas de non-positivite cartesienne**.

**Beta** (omega=alpha=beta^2 ; `w=theta*dt*omega`, le det de Lorentz est `1+w^2`) :

| beta | omega | w^2 | gamma_3 |
|---|---|---|---|
| 1e2 | 1e4 | 25 | 0.0321 |
| 1e3 | 1e6 | 2.5e5 | 0.0320 |
| 1e4 | 1e8 | 2.5e9 | 0.0321 |
| 1e6 | 1e12 | 2.5e17 | 0.0321 |

gamma_3 **exactement plat** sur 4 ordres de grandeur d'omega (w^2 de 25 a 2.5e17, ce dernier au bord
de la precision float64). -> la **raideur / omega / precision de l'eliminateur de Lorentz est écartée**
comme cause. Le deficit est invariant aux deux parametres extremes du probleme (contraste ET omega).

## Synthese des causes écartées (cette session + #236)

| cause | verdict | preuve |
|---|---|---|
| resolution | écartée | converge n=64->192->256/384 (#236) |
| geometrie de bord (cut-cell) | écartée | square==staircase==cutcell (#236) |
| schema temporel / dt | écartée | dt-sweep GH200 (#236) ; Strang/ssprk3 livre |
| politique de Gauss (R0) | écartée | evolve~=restart (section 4) |
| limite froide (temperature) | écartée | scan theta_p empire (section 3) |
| **contraste de densite** | **écartée** | gamma_3 plat, min(rho)>0 (section 5) |
| **raideur / omega / precision** | **écartée** | gamma_3 plat de w^2=25 a 2.5e17 (section 5) |
| non-positivite (cartesien) | écartée | min(rho)>0, pas de NaN (section 5) |
| **dissipation de flux (Rusanov)** | **écartée** | HLL ~= Rusanov (section 6, adc_cpp #239) |

## 6. Test HLL vs Rusanov -> dissipation de flux écartée

HLL (Harten-Lax-van Leer, 2 ondes, moins diffusif que Rusanov) a ete expose pour le modele 3-var
isotherme (adc_cpp **#239** : `riemann="hll"`, sans exiger de pression, gate sur `model.wave_speeds`).
Mesure system-schur cartesien (n=128, Strang+Schur, fenetres papier) :

| | rusanov | hll |
|---|---|---|
| l=3 froid (theta_p=0) | 0.0321 | 0.0316 |
| l=3 chaud (theta_p=0.5) | 0.0285 | 0.0290 |

**HLL ~= Rusanov** (a ~2% pres, dans les deux sens). Le candidat "dissipation Rusanov au contact"
(hypothese n1 du playbook #238) est donc **écarté** : reduire la dissipation de flux ne recupere pas
le taux. Le plateau ~0.032 est invariant au flux comme au contraste, a beta, a la politique de Gauss
et a la temperature -- une robustesse remarquable qui pointe vers une cause non locale au flux/recon.

## Verrou restant (apres 9 causes ecartees)

Le deficit cartesien n'est ni temporel, ni geometrique, ni Gauss/R0, ni temperature, ni contraste, ni
raideur/omega, ni non-positivite, ni dissipation de flux. Suspects restants (branche "HLL ne change
rien" du plan user) :
1. **Couplage Schur** : la facon dont l'etage source condense reconstruit/applique la derive E×B
   (vs le ExB reduit qui advecte rho directement) -- le full transporte un moment compressible re-derive
   a chaque pas, le reduit non.
2. **Observable / normalisation 2 pi** : reserve metrologique (`NORMALIZATION.md`). partielle au mieux :
   meme x2pi (~6.28), 0.035 -> 0.22, encore 3-4x sous 0.772. Ne ferme pas le facteur seul.
3. **Structure complete vs reduite** : le ExB scalaire reproduit la cible (+0.2%), le full Euler-Poisson
   +moment+Schur donne ~0.032 quoi qu'on fasse -> la difference est dans la chaine moment/Schur/derive.

## 7. full vs réduit ExB sur le même setup cartesien -> structure du modele écartée (10e cause)

Test decisif : sur le même cartesien (meme IC anneau, meme observable |c_l(phi)| sur r0, meme n/dt/
fenetre), comparer le full (rho, m_x, m_y + Strang + CondensedSchur) au réduit ExB scalaire (n advecte
par la derive v=(-d_y phi/omega, d_x phi/omega), phi=Gauss(alpha n) ; pas de moment, pas de Schur) :

| l | full (rho,m,Schur) | reduit ExB scalaire | reduit/full |
|---|---|---|---|
| 3 | 0.0321 | 0.0309 | 1.0x |
| 4 | -0.0048 | -0.0036 | 0.8x |

**Le réduit ExB cartesien donne le même ~0.032 que le full.** La chaine moment/Schur/derive du full
N'est donc pas la cause (structure du modele écartée, 10e cause). Le deficit est commun au plus simple
ExB scalaire sur grille carree.

### Decomposition finale du deficit cartesien
- **Normalisation 2 pi** (`diag/diag_polar_omega.py:35` : rhobar=rho_max=1 -> facteur = 2 pi ~= 6.28
  exactement, pas davantage). Cartesien brut 0.032 x 2 pi = 0.20 -> encore ~3.8x sous 0.772.
- **Geometrie cartesien vs polaire** : le ExB reduit polaire (diag_polar_omega) + 2 pi reproduit le
  papier (l=4 exact 0.913 vs 0.911 ; l=3 +26%, l=5 -29%), avec g_raw polaire l=3=0.155, l=4=0.145
  (echelle de temps polaire tf~33, fenetres [2.4,12.5]). Le même modele ExB sur grille cartesienne donne
  g_raw ~0.032 (l=3) -> la grille carree ne capte pas la dynamique azimutale de l'anneau tournant aussi
  bien que la grille polaire. Le cut-cell (#236) et la resolution (converge 0.035) ne corrigent pas ce
  facteur geometrique -> limitation fondamentale du FV cartesien pour cette instabilite d'anneau tournant.

### 7bis. Ratio Im/Re (scale-invariant) -> LE mecanisme : rotation captee, croissance etouffee
Mesure du complexe c_l(t) sur r0 (cartesien, full ET reduit) : on extrait gamma_raw (pente de log|c_l|)
ET Omega_raw (pente de la phase = frequence de rotation azimutale du mode), fenetres papier, n=128.

| l | gamma_raw | Omega_raw | ratio gamma/Om |
|---|---|---|---|
| 3 | 0.032 | **0.527** | 0.06 |
| 4 | -0.004 | **0.664** | -0.01 |
| 5 | 0.107 | **0.706** | 0.15 |

(full ~= reduit, encore.) LE fait decisif : **Omega_raw ~ 0.5-0.7** -- le mode tourne fortement (la
derive E×B azimutale autour de l'anneau est captee par la grille carree), mais **gamma_raw ~ 0.03**
(vs ~0.77 attendu). **Le cartesien capte la rotation mais etouffe la croissance** (facteur ~15-25x).
L'instabilite diocotron croit du cisaillement radial d_rho/d_r aux bords d'anneau r0/r1 ; sur grille
carree ces bords radiaux sont diffuses -> la retro-action de croissance meurt, l'advection azimutale
survit. Ce n'est pas une normalisation de temps (le ratio differe du polaire/analytique -> pas que des
unites) ni la chaine moment/Schur (full=reduit). C'est l'incapacite de la grille carree a resoudre la
structure radiale de bord d'anneau qui pilote la croissance -> exactement pourquoi la grille polaire
(bords d'anneau = lignes radiales du maillage) reproduit et la carree non.

### conclusion : l'investigation cartesienne est close
Le deficit cartesien = la grille carree **capte la rotation E×B mais diffuse la structure radiale de
bord d'anneau qui pilote la croissance** (+ normalisation 2 pi). commun a tous les modeles cartesiens
(full = reduit). Ce n'est aucun knob fixable du moteur (10 causes ecartees : resolution, geometrie de
bord, temps, Gauss, temperature, contraste, beta/omega, non-positivite, dissipation de flux, structure
du modele ; + ratio Im/Re : rotation OK, croissance etouffee). **La repro N'est pas atteignable
proprement en FV cartesien** -- il faut un maillage qui resout le cisaillement radial de l'anneau (polaire).

Voies de repro restantes (toutes hors cartesien) :
1. **ExB reduit polaire + 2 pi** : reproduit l=4 exact, l=3/l=5 partiels. **Voie credible etablie** (l'objet
   du papier est la derive ExB ; le reduit la capture sur grille polaire). cf. `diag/diag_polar_omega.py`.
2. **Modele complet polaire** (voie 1, adc_cpp #236) : diverge (non-positivite au bord d'anneau, 1/rho) ->
   exige le redesign spatial positivite (chantier separe, hors cartesien).

Un correctif de positivite ou de flux est inerte sur le cartesien (min rho>0 ; HLL~=Rusanov).

## Reproduire

```bash
# adc_cpp : build avec ssprk3 sur le chemin production (PR #230) ; adc_cases : run.py Strang (PR #21)
python hoffart_euler_poisson_dsl/run.py --engine system-schur \
  --n 192 --t-end 1.4 --modes 3 4 5 --dt 1e-3 --no-gif
# scan de resolution : repeter avec --n 64 96 128 192 --modes 3 --t-end 0.8
```
