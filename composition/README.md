# composition : composer un systeme multi-blocs heterogene, un bloc d'equation a la fois

Tutoriel de l'API de composition de `adc` : depuis Python on assemble un systeme multi-especes
**un bloc a la fois**, en choisissant pour CHAQUE bloc, independamment, son modele physique, sa
reconstruction spatiale, son flux de Riemann, son traitement temporel et son nombre de sous-pas. Le
script demontre quatre capacites : (A) composition heterogene de deux fluides au schema different ;
(B) determinisme bit a bit de la composition de briques ; (C) rejet des combinaisons invalides ;
(D) un integrateur temporel SSPRK2 **ecrit en Python** par-dessus les primitives de la lib. Ce cas
**demontre une capacite d'API, il ne valide aucun resultat physique publie**.

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `tutoriel` (`cases_manifest.toml`, `composition/run.py`, `ci = true`, `needs = []`) |
| Entrees | A : grille $48^2$, $L=1$, **periodique** ; electrons $\rho=1+0.02\cos(2\pi x/L)$ (Euler $\gamma=1.4$, $q=-1$), ions $\rho=1$ (isotherme $c_s^2=0.5$, $q=+1$) ; $dt=0.001$, 8 pas. B/D : grille $32^2$ periodique, diocotron ($B_0=1$, $\alpha=1$, fond $n_{i0}=\overline{\rho}$) ; B : $dt=0.002$, 12 pas ; D : $dt=0.001$, 20 pas Python |
| Sorties | diagnostics imprimes (aucun fichier produit par `run.py`) ; les figures sont generees a part par `make_figures.py` dans `figures/` + `figures/provenance.json` |
| Invariants garantis | les `assert` de `run.py` : A masse electrons/ions $<$ `MASS_TOL=1e-10` ET $\|\phi\|_\infty>10^{-8}$ ET evolution electrons $>10^{-9}$ (`run.py:124-137`) ; B ecart deux compositions $==0$ exactement (`run.py:166`) ; C les 3 combinaisons invalides levent ET `n_species()==0` (`run.py:185-209`) ; D masse $<10^{-9}$ ET etat fini (`run.py:243-244`) |
| PROUVE | (A) deux fluides au schema **different** coexistent dans un meme `adc.System`, chacun conserve sa masse ($2.7\times10^{-12}$ electrons, $1.8\times10^{-12}$ ions), le Poisson couple est actif ($\|\phi\|_\infty=5.06\times10^{-4}$), les electrons evoluent ($3.5\times10^{-5}$) ; (B) un MEME modele compose deux fois donne un etat **identique au bit** (ecart $=0$, `np.array_equal` vrai) ; (C) HLLC sur transport scalaire, source fluide sur scalaire, et modele incoherent sont **rejetes a la composition/a l'ajout**, sans bloc ajoute ; (D) un integrateur SSPRK2 ecrit en Python conserve la masse ($2.3\times10^{-13}$) et reste fini |
| NE PROUVE PAS | **demontre une capacite d'API, ne valide aucun resultat physique publie**. Aucun nombre n'est confronte a un article ; les CI sont des cosinus simples, les horizons sont courts (8/12/20 pas), aucune dynamique physique n'est interpretee. Le determinisme bit (B, D) est une propriete d'**implementation** (memes briques C++ figees, memes operations flottantes dans le meme ordre), PAS une garantie cross-plateforme : il peut casser entre BLAS, ordre de sommation ou architecture. La conservation de masse mesuree n'est pas une validation de schema : c'est le minimum attendu d'un volumes-finis conservatif |
| Provenance | adc_cpp `01873299`, adc_cases (deeptut) `a9541ba4`, backend natif serie, $48^2$ (A) / $32^2$ (B,D), ~1 s 1 coeur CPU ; `figures/provenance.json` |

A la fin tu sauras : comment `adc.System` compose un systeme bloc par bloc, ce que signifie "chaque
bloc fige son schema a l'ajout" (table 3 couches), pourquoi la composition est deterministe au bit,
ce que les garde-fous attrapent, et comment ecrire son propre integrateur temporel en Python sans
sortir le calcul par cellule du C++.

---

## 1. Le niveau d'abstraction demontre (justifie PROUVE : composition heterogene)

Le mot d'ordre est **Python compose, le C++ calcule**. La config du systeme ne porte que le
**maillage** ; toute la physique est portee par les **blocs** :

```python
sim = adc.System(n=48, L=1.0, periodic=True)            # config = MAILLAGE seul (run.py:94)
sim.add_block("electrons", model=models.electron_euler(),
              spatial=adc.Spatial(vanleer=True, flux="hllc"),
              time=adc.IMEX(substeps=10))                # run.py:100-102
sim.add_block("ions", model=models.ion_isothermal(),
              spatial=adc.Spatial(minmod=True, flux="rusanov"),
              time=adc.Explicit())                       # run.py:104-106
```
- `adc.System(n, L, periodic)` (`run.py:94`) ne porte AUCUN parametre physique ($\gamma$, $c_s^2$,
  $B_0$, charge) : seulement le maillage. La physique migre dans les blocs.
- `models.electron_euler()` et `models.ion_isothermal()` (`models.py:28-45`) sont des
  **compositions de briques** generiques (section 3), pas des chaines magiques. Le mot "electron"
  vit dans `adc_cases`, jamais cote coeur.
- chaque `add_block` choisit **independamment** la reconstruction (`adc.Spatial`), le flux
  (`"hllc"` vs `"rusanov"`) et le traitement temporel (`adc.IMEX(substeps=10)` vs `adc.Explicit()`).
  Les deux blocs ne partagent QUE le maillage et le Poisson de systeme.

Le choix "un schema par bloc" est physiquement motive ici : les electrons (legers, raides) sont
sous-cycles 10 fois par macro-pas et leur force electrostatique est traitee en implicite (IMEX) ;
les ions (lents) avancent en explicite simple. C'est une **capacite** (multirate + IMEX par bloc),
pas une affirmation de validite physique.

---

## 2. Les modeles = compositions de briques (justifie : la physique est figee en C++)

Chaque modele d'espece est un `adc.Model(state, transport, source, elliptic)`. Les trois modeles
employes, et la brique exacte de chaque slot :

| Modele (`models.py`) | state | transport | source | elliptic |
|---|---|---|---|---|
| `electron_euler()` (l.28) | `FluidState(compressible, gamma=1.4)` | `CompressibleFlux` | `PotentialForce(charge=-1)` | `ChargeDensity(charge=-1)` |
| `ion_isothermal()` (l.38) | `FluidState(isothermal, cs2=0.5)` | `IsothermalFlux` | `PotentialForce(charge=+1)` | `ChargeDensity(charge=+1)` |
| `diocotron()` (l.18) | `Scalar` | `ExB(B0=1)` | `NoSource` | `BackgroundDensity(alpha=1, n0=n_i0)` |

Chaque brique est une physique **ponctuelle, device-callable**, definie une fois cote coeur :

- `CompressibleFlux` = `Euler` (`hyperbolic.hpp:118`) : flux Euler 4 variables $(\rho,\rho u,\rho v,E)$.
- `IsothermalFlux` (`hyperbolic.hpp:127`) : 3 variables, fermeture $p=c_s^2\rho$, onde $\sqrt{c_s^2}$.
- `ExBVelocity` (`hyperbolic.hpp:27`) : 1 variable, derive $v=(-\partial_y\phi,\partial_x\phi)/B_0$.
- `PotentialForce` (`source.hpp:33`) : $s[1]=q\rho E_x$, $s[2]=q\rho E_y$, travail $s[3]$ **seulement
  si** `State::size()==4` (Euler). Sur un transport a 3 variables (isotherme) il n'y a pas de
  composante energie ; sur un **scalaire** (1 variable) la force est sans objet (rejet, section 5).
- `ChargeDensity` (`elliptic.hpp:19`) : second membre $f=q\,n$, signe de $q$ inclus.
- `BackgroundDensity` (`elliptic.hpp:31`) : $f=\alpha(n-n_0)$, fond neutralisant pour Poisson periodique.

Le second membre du Poisson de systeme est $\sum_s f_s = \sum_s q_s n_s$ (somme generique des
briques elliptiques de chaque bloc). Avec electrons $q=-1$ perturbes et ions $q=+1$ uniformes
($\overline{q n}\approx 0$), le Poisson periodique est solvable et $\phi\neq 0$.

### La table 3 couches : qui calcule quoi, et OU le schema est fige

| Ligne `run.py` | Couche | Ce qui se passe |
|---|---|---|
| `sim.add_block("electrons", model=..., spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))` (`run.py:100-102`) | Python **compose** | choix du modele (briques), du schema spatial (recon + flux), du traitement temporel (IMEX + 10 sous-pas) ; lecture de l'etat via `density`/`potential`/`get_state` |
| `models.electron_euler()` -> briques `CompressibleFlux` / `PotentialForce` / `ChargeDensity` (`hyperbolic.hpp:118`, `source.hpp:33`, `elliptic.hpp:19`) | brique C++ **fige la physique** | la convention exacte du flux Euler, de la force $q\rho E$, du second membre $q n$ |
| `System::add_block(... spatial.limiter, spatial.flux, spatial.recon, time.kind, time.substeps, time.stride ...)` (facade `__init__.py:831-833`) puis `assemble_rhs<Limiter,Flux>` + Newton local IMEX + Poisson de systeme | noyau **par cellule** (device) | le calcul reel, sans callback Python dans le hot path |

Le point cle est la troisieme ligne : la facade `add_block` (`__init__.py:819-833`) passe
`spatial.limiter`, `spatial.flux`, `spatial.recon`, `time.kind`, `time.substeps`, `time.stride` au
`System::add_block` C++ **a l'instant de l'ajout**. Le schema (VanLeer+HLLC+IMEX+10) est alors
**fige dans le bloc** sous forme d'une fermeture d'avancee compilee ; il n'est plus reconfigurable
sans re-ajouter le bloc, et il est type-erased SEULEMENT au niveau de la liste de blocs. C'est ce
qui rend (B) deterministe : memes briques + meme schema fige -> meme calcul C++.

---

## 3. La prediction falsifiable : determinisme bit a bit (B et D)

Pour un tutoriel, la "prediction" testable n'est pas un nombre physique mais une propriete
d'implementation **falsifiable** : recomposer le MEME modele avec la MEME CI, ou rejouer le MEME pas
Python, donne un etat **identique au bit**. `run.py:166` l'affirme par `assert ecart == 0.0` (B) ; la
figure `determinism.png` (section 6) le confronte sur les deux chemins (B compose C++, D pas Python).
Une seule cellule non nulle dans l'ecart trahirait du non-determinisme : un etat global cache entre
deux compositions, un parcours de cellules dependant de l'allocation, ou un ordre de reduction non
reproductible dans le Poisson. L'egalite stricte ($==0$, pas $<\varepsilon$) est l'observable :
elle ne tolere AUCUN bruit, contrairement aux tolerances de conservation (section 4).

---

## 4. Les tolerances, justifiees par un ordre de grandeur (justifie 8 de la checklist)

| Tolerance | Valeur | Pourquoi cette valeur |
|---|---|---|
| `MASS_TOL` | $10^{-10}$ | Les flux (CompressibleFlux, IsothermalFlux, ExB a divergence nulle) sont **conservatifs** : la masse est un invariant exact, la seule derive est l'arithmetique flottante sur 8/12 pas. Mesure A : $2.7\times10^{-12}$ (electrons) / $1.8\times10^{-12}$ (ions), ~2 ordres sous la tolerance (`run.py:82`, `135-136`) |
| ecart B | $==0$ **exactement** | PAS une tolerance : une egalite stricte. Deux compositions du meme modele avec les memes briques figees executent les memes operations flottantes dans le meme ordre, donc le resultat est bit-identique. Toute valeur $>0$ serait un bug de determinisme, pas du bruit acceptable (`run.py:166`) |
| masse D | $10^{-9}$ | L'integrateur SSPRK2 Python combine des etats par $\frac12 U_0+\frac12(U_1+dt\,R)$ : combinaison **affine** d'etats conservatifs, donc conservative aux erreurs flottantes pres sur 20 pas. Mesure : $2.3\times10^{-13}$, ~4 ordres sous la tolerance (`run.py:243`) |
| $\|\phi\|_\infty>10^{-8}$ (A) | borne BASSE | Garantit que le Poisson couple est **actif** (le bloc contribue vraiment au second membre). Mesure : $5.06\times10^{-4}$, ~4 ordres au-dessus : le couplage est franc, pas un residu numerique (`run.py:124`) |
| evolution electrons $>10^{-9}$ (A) | borne BASSE | Garantit que le bloc electron **bouge** (la force et le transport agissent). Mesure : $3.5\times10^{-5}$, ~4 ordres au-dessus du seuil : la dynamique est non triviale (`run.py:137`) |

---

## 5. Les garde-fous : combinaisons invalides rejetees (justifie PROUVE : C)

`partie_C` (`run.py:169-209`) verifie que trois compositions invalides levent une erreur claire au
lieu de produire un calcul faux. `doit_lever(fn, why)` (`run.py:175-181`) execute `fn`, attend une
exception (pybind traduit `std::runtime_error` en `RuntimeError`), et echoue si rien ne leve.

1. **HLLC sur transport scalaire** (`run.py:185-188`). HLLC exige un transport compressible (4
   variables + pression) ; le diocotron transporte un scalaire par ExB. Rejet **a l'ajout du bloc**.
   Message reel : `System : flux 'hllc' exige un transport compressible (4 variables + pr...`.

2. **Source fluide sur transport scalaire** (`run.py:194-200`). Un `adc.Model(Scalar, ExB,
   PotentialForce, BackgroundDensity)` **se compose** (state Scalar et transport ExB sont coherents),
   mais `PotentialForce` agit sur une quantite de mouvement fluide ($s[1], s[2]$) absente d'un
   scalaire (1 variable). Rejet **a l'ajout du bloc**. Message reel : `source 'potential' invalide
   ici (exige un transport fluide >= 3 variab...`.

3. **Modele incoherent des la composition** (`run.py:204-207`). Un `adc.Model(Scalar,
   CompressibleFlux, ...)` melange un etat scalaire (1 var) et un flux Euler (4 var) :
   `adc.Model(...)` **leve directement**, avant tout ajout. Message reel : `Scalar exige
   transport=ExB(...)`.

L'assert final `sim.n_species() == 0` (`run.py:209`) garantit qu'**aucun bloc invalide n'a ete
ajoute** : les rejets sont propres (pas d'etat partiellement mute). La difference entre 1/2 (rejet a
l'ajout) et 3 (rejet a la composition) est le **moment** ou l'incoherence est detectee : un transport
incompatible avec l'etat est attrape par `adc.Model`, une source/flux incompatible avec le transport
choisi par `add_block`.

---

## 6. Partie D : un integrateur temporel ecrit en Python (justifie PROUVE : D)

Au lieu d'appeler `sim.advance(...)` (boucle en temps compilee), `partie_D` (`run.py:212-244`) ecrit
sa propre boucle avec `adc.integrate.ssprk2_step` (`integrate.py:27-44`), un SSPRK2 (Heun fort-stable)
assemble **en Python** par-dessus quatre primitives exposees par `System` :

```python
sim.solve_fields()                                    # Poisson + aux = grad(phi)   (C++)
U0 = {n: sim.get_state(n) for n in names}             # etat par bloc (ncomp,n,n)    (lecture)
for n in names:                                       # etage 1 : U1 = U0 + dt R(U0)
    sim.set_state(n, U0[n] + dt * sim.eval_rhs(n))    # eval_rhs = -div F + S         (C++ par cellule)
sim.solve_fields()                                    # Poisson RE-RESOLU per-stage   (C++)
for n in names:                                       # etage 2 : 1/2 U0 + 1/2 (U1 + dt R(U1))
    U1 = sim.get_state(n)
    sim.set_state(n, 0.5 * U0[n] + 0.5 * (U1 + dt * sim.eval_rhs(n)))
```
- `eval_rhs(n)` calcule le residu $-\nabla\cdot F + S$ du bloc **par cellule en C++** : aucun callback
  Python dans le hot path. Seul l'**assemblage des etages RK** (les combinaisons affines $U_0+dt\,R$
  et $\frac12 U_0+\frac12(\dots)$) est en Python, par PAS.
- `solve_fields()` re-resout Poisson **entre les deux etages** : couplage hyperbolique/elliptique
  per-stage, plus precis que le couplage fige par pas de `advance`.
- la boucle (`run.py:234-235`) appelle `ssprk2_step(sim, 0.001)` 20 fois. Mesure :
  derive masse $2.3\times10^{-13}$, etat fini : l'integrateur Python conserve et reste stable.

C'est la capacite la plus forte du tutoriel : **le schema en temps lui-meme peut etre ecrit en
Python** (par pas), le calcul du residu et le Poisson restant en C++ (par cellule). Et il reste
**bit-deterministe** : deux executions de cette meme boucle Python donnent un etat identique au bit
(section 6, figure D), car les primitives C++ et les combinaisons numpy sont elles-memes
deterministes a operations et ordre constants.

---

## 7. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (memes parametres que `run.py`), versionnees avec
`figures/provenance.json`. Commande exacte en section 9. `run.py` lui-meme ne produit AUCUN fichier
(diagnostics imprimes) ; les figures sont un diagnostic d'API a part.

### `density_maps.png` : les champs composes du systeme heterogene (Partie A)

![Cartes : densite electron CI et finale, densite ion finale, potentiel |phi| couple](figures/density_maps.png)

- **PROUVE** (asserte `run.py:124-137`) : le Poisson couple est actif ($\|\phi\|_\infty=5.06\times
  10^{-4}$, panneau 4) et les electrons evoluent (la densite finale, $[0.980, 1.020]$, differe de la
  CI ; ecart max $3.5\times10^{-5}$). Les deux blocs au schema **different** coexistent : le panneau
  electron porte la perturbation Euler/HLLC/IMEX, le panneau ion porte la reponse isotherme.
- **SUGGERE (non assere)** : la densite ion finale s'ecarte de l'uniforme par $\sim6\times10^{-7}$
  (echelle `1e-6+1`) : le couplage Poisson pousse les ions, mais l'effet est minuscule sur 8 pas et
  aucun assert ne le quantifie (l'assert ne teste que la conservation de masse ionique). La structure
  reste 1D selon $x$ (la CI ne depend pas de $y$).
- **NON MONTRE** : aucun resultat physique. Les profils sont des cosinus a horizon court (8 pas) ;
  rien ici ne reproduit ni ne valide un regime publie. C'est un tutoriel : les cartes montrent que
  la composition **produit des champs reels couples**, pas qu'ils sont physiquement significatifs.

### `determinism.png` : egalite bit, composition (B) ET pas Python (D)

![Deux heatmaps |a-b| identiquement noires (B compose, D pas Python) + histogramme du residu a zero](figures/determinism.png)

- **PROUVE** (asserte `run.py:166` pour B ; mesure pour D) : les deux heatmaps $|a-b|$ sont
  **identiquement noires** (echelle $[0,10^{-15}]$). Panneau B : deux compositions independantes du
  meme diocotron, ecart max $0.0$, `np.array_equal` vrai. Panneau D : deux executions de
  l'integrateur SSPRK2 **ecrit en Python**, ecart max $0.0$, `np.array_equal` vrai. L'histogramme
  concentre toutes les cellules ($32^2$ par chemin) **exactement** a $0$.
- **SUGGERE** : que l'egalite reste vraie pour d'autres schemas/CI est plausible (memes briques
  figees, memes operations), mais le cas ne teste que ces deux configurations.
- **NON MONTRE** : ce determinisme est une propriete de l'implementation **sur cette plateforme**, pas
  une garantie cross-plateforme. Il peut casser entre BLAS differentes, ordres de sommation ou
  architectures (cf. caveat plateforme, section 9). Une seule cellule non noire signalerait un etat
  global cache entre compositions, un parcours dependant de l'allocation, ou une reduction Poisson
  non reproductible.

---

## 8. Ce que le tutoriel ne capture pas (analyse honnete des limites)

- **Aucun resultat physique valide.** Categorie `tutoriel` : on demontre une **capacite d'API**
  (composition bloc-par-bloc, multirate, IMEX par bloc, garde-fous, integrateur Python), pas une
  courbe d'article ni un invariant physique non trivial. La conservation de masse ($10^{-12}$) est le
  minimum attendu d'un volumes-finis conservatif, pas une validation.
- **Le determinisme bit est de l'implementation, pas de la physique.** Il prouve que le chemin compose
  (B) et le pas Python (D) sont reproductibles a operations et ordre constants. Il n'est PAS garanti
  entre plateformes (BLAS, sommation, architecture).
- **Horizons courts, CI triviales.** A : 8 pas, cosinus 1D. B : 12 pas. D : 20 pas. Rien n'est
  integre assez longtemps pour qu'une dynamique non lineaire emerge ; ce n'est pas le but.
- **Le couplage ion est minuscule** ($\sim6\times10^{-7}$ sur 8 pas) : visible sur la carte mais non
  asserte. Le seul invariant teste sur les ions est la conservation de masse.
- **Le diocotron ici n'est pas l'etude physique** : c'est l'etude de reference
  [`../diocotron/`](../diocotron/) (taux de croissance, figures, gif) qui porte la physique ; ici le
  diocotron sert juste de modele scalaire simple pour les tests de determinisme (B, D).

---

## 9. Reproduire (justifie 14 de la checklist : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/composition
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~1 s
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 figures + provenance.json
```

Prerequis : `numpy` (et `matplotlib` pour les figures, hors `needs` du cas lui-meme), module `adc`
compile et importe **avec le meme interpreteur** que celui qui l'a compile (suffixe ABI
`cpython-312`). Le premier chemin du `PYTHONPATH` fournit le module C++ ; le second rend `adc_cases`
importable sans installation (le cas a aussi un fallback `sys.path`, `run.py:72-77`).

Sortie attendue de `run.py` (capturee, machine de dev macOS arm64) :

```
== Partie (A) : un schema (modele/spatial/temps/sous-pas) par bloc ==
  n_species              = 2
  |phi|_max (initial)    = 5.062437e-04
  derive masse electrons = 2.728e-12  (Euler/HLLC/IMEX, 10 sous-pas)
  derive masse ions      = 1.819e-12  (isotherme/Rusanov/explicite)
  evolution electrons    = 3.506e-05  (dynamique non triviale)
== Partie (B) : determinisme de la composition de briques (bit pour bit) ==
  ecart max (deux compositions independantes) = 0.000e+00
== Partie (C) : garde-fous des combinaisons invalides ==
  rejete (hllc sur diocotron (transport scalaire)) : System : flux 'hllc' exige un transport compressible ...
  rejete (source PotentialForce sur transport scalaire) : source 'potential' invalide ici (exige un transport fluide >= 3 ...
  rejete (modele incoherent (Scalar + CompressibleFlux)) : Scalar exige transport=ExB(...)
== Partie (D) : integrateur temporel custom en Python (SSPRK2) ==
  pas Python (SSPRK2)    = 20  (Poisson re-resolu per-stage)
  derive masse           = 2.274e-13  (integrateur ecrit en Python)
  etat fini              = True
OK composition_api
```

Cout : ~1 s temps mur (import numpy inclus), 4 parties (A 8 pas $48^2$ + Poisson par etage ;
B 2$\times$12 pas $32^2$ ; C 3 rejets ; D 20 pas Python $32^2$). **Caveat plateforme** : les
verdicts (masse conservee, ecart $=0$, rejets, `OK`) et les ordres de grandeur sont stables ; les
derniers chiffres des derives ($2.7\times10^{-12}$, etc.) varient avec la BLAS et l'ordre de
sommation, et l'egalite bit (B, D) elle-meme n'est garantie que **sur la meme plateforme/le meme
build** (cf. `figures/provenance.json`).

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | le cas : 4 parties (A composition heterogene, B determinisme compose, C garde-fous, D integrateur Python) |
| `make_figures.py` | re-joue A/B/D ; ecrit `density_maps.png`, `determinism.png` + `provenance.json` |
| `figures/density_maps.png` | champs composes du systeme heterogene (electrons, ions, $\|\phi\|$) |
| `figures/determinism.png` | egalite bit (B compose C++, D pas Python) + histogramme du residu a 0 |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, backend, resolution, schemas par bloc, nombres mesures |
| `../adc_cases/models.py` | `electron_euler`, `ion_isothermal`, `diocotron` = compositions de briques (l.18-45) |
| `../adc_cases/common/grid.py` | `meshgrid_xy` (grille a centres de cellules, convention de la facade) |
| `<adc>/integrate.py` | `ssprk2_step` = SSPRK2 ecrit en Python sur les primitives `solve_fields`/`eval_rhs`/`get_state`/`set_state` |
