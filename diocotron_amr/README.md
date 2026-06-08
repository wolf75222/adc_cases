# diocotron_amr : instabilite diocotron sur grille AMR multi-patch

Cas de **validation** (CI). Porte un bloc diocotron natif sur une hierarchie de raffinement
adaptatif `adc.AmrSystem` (grossier + un niveau fin re-decoupe dynamiquement par regrid
Berger-Rigoutsos, avec reflux conservatif aux interfaces grossier/fin), et **prouve par asserts**
que le raffinement observe vient bien du critere de tagging (pas du simple build de la hierarchie)
et qu'il modifie mesurablement la solution, tout en conservant la masse a l'arrondi machine.

Categorie manifeste : `validation`, `ci = true`, `needs = []` (cf. `cases_manifest.toml`).
Ce n'est PAS une reproduction d'un resultat publie : c'est un test de la capacite AMR de l'API.

---

## 1. Objectif du cas

Verifier, sur un scenario diocotron (bande de charge qui s'enroule sous derive E x B), que la
brique AMR de `adc` fonctionne reellement et de bout en bout depuis Python :

- **vrai raffinement adaptatif** : la bande de charge est taggee et couverte par **plusieurs**
  patchs fins (`n_patches() >= 2`) a chaque pas, et un run de **controle** avec un seuil de
  raffinement inatteignable ne produit qu'un patch degenere (1), strictement moins. Le raffinement
  vient donc du **tagging**, pas du build de la hierarchie ;
- **effet sur la solution** : la solution raffinee (projetee sur la grille de base) differe
  mesurablement de celle du run de controle non raffine (ecart sup mesure > seuil) ;
- **conservation de masse sur AMR (reflux)** a l'arrondi machine : `drel < 1e-9` ;
- **integrite numerique** : densite finie partout (ni NaN ni Inf).

Tout est verifie par `assert` : si un invariant casse, le cas sort en `AssertionError` et la CI
devient rouge.

---

## 2. Equations

Le bloc evolue une **densite scalaire** `n_e(x, y, t)` advectee par la derive E x B, le champ
electrique `E = -grad(phi)` provenant d'un Poisson periodique a fond neutralisant :

```
d n_e / dt + div( v_ExB n_e ) = 0          (transport conservatif de la densite)
v_ExB = (E x B) / B0^2,  E = -grad(phi)     (derive electrique, B = B0 e_z)
-Delta phi = alpha (n_e - n_i0)             (Poisson periodique, fond neutralisant)
```

Le membre de droite du Poisson est la brique `BackgroundDensity` : `f = alpha (n_e - n_i0)` avec
`alpha = 1.0`. Le fond `n_i0` est fixe a la **moyenne** de la densite initiale (`n_i0 = <n_e>`) :
un Poisson periodique exige un second membre a **moyenne nulle** pour etre soluble, ce que le fond
neutralisant garantit.

---

## 3. Modele physique

Le modele d'espece est `models.diocotron(B0=1.0, alpha=1.0, n_i0=<n_e>)`
(`adc_cases/models.py`), qui compose quatre briques generiques de `adc` :

| Slot       | Brique `adc`                              | Role |
|------------|-------------------------------------------|------|
| `state`    | `adc.Scalar()`                            | 1 variable : la densite `n_e` |
| `transport`| `adc.ExB(B0=1.0)`                         | advection par la derive E x B (champ `B0`) |
| `source`   | `adc.NoSource()`                          | pas de terme source |
| `elliptic` | `adc.BackgroundDensity(alpha=1.0, n0=n_i0)`| second membre Poisson `alpha (n - n0)` |

C'est un modele **purement cinematique de transport** : la densite est convectee par une vitesse
self-consistante (issue du potentiel resolu a chaque sous-pas). Pas de pression, pas de force sur
une quantite de mouvement (un seul champ scalaire). Le diocotron y apparait comme l'enroulement de
la bande de charge sous sa propre derive.

---

## 4. Methode numerique

- **Schema spatial** : `adc.Spatial(none=True)` -> limiteur **NoSlope** (reconstruction d'ordre 1,
  "none") + flux numerique **Rusanov** (defaut de `Spatial`). Volumes finis conservatifs, ordre 1,
  robuste : choisi pour une bande qui s'enroule sur une grille AMR grossiere (`n_base = 64`) sans
  oscillations.
- **Schema temporel** : explicite (defaut `adc.Explicit` -> SSPRK2, Shu-Osher 2 etages ordre 2),
  active par `step_cfl(0.4)` : pas de temps choisi a **CFL = 0.4** (`dt = CFL * h / w_max`, ou
  `w_max` borne la vitesse de derive). `step_cfl` renvoie le `dt` reellement effectue.
- **Poisson** : `set_poisson(rhs="charge_density", solver="geometric_mg")`. Le solveur est un
  **multigrille geometrique** ; les conditions aux limites sont **periodiques** (heritees de
  `periodic=True` a la construction). Le token `rhs="charge_density"` selectionne le second membre
  charge/densite (ici porte par la brique `BackgroundDensity` du modele).
- **AMR** : un niveau de base (grossier) plus un niveau fin. Toutes les `regrid_every = 10`
  iterations, les cellules dont la densite depasse le `threshold` sont taggees, et le niveau fin
  est re-decoupe en patchs rectangulaires par **clustering Berger-Rigoutsos** pour suivre la bande.
  Le couplage grossier/fin est conservatif grace au **reflux** (correction des flux aux interfaces
  de niveau), d'ou la conservation de masse a l'arrondi machine.

---

## 5. Architecture ADC utilisee

```
Python (run.py)                C++ (adc._adc.AmrSystem, header-only Kokkos)
---------------                ---------------------------------------------
adc.AmrSystem(n,L,             AmrSystemConfig {n, L, periodic, regrid_every, ...}
   regrid_every,periodic) ---> moteur AMR (hierarchie grossier + fin, regrid, reflux)
   .add_block("ne",model, ---> 1 bloc natif sur la hierarchie AMR
              spatial)          (transport ExB + reconstruction NoSlope + Rusanov)
   .set_refinement(thr)   ---> seuil de tagging des cellules a raffiner
   .set_poisson(rhs,solver)--> Poisson de systeme (geometric_mg, periodique)
   .set_density("ne",ne)  ---> projette la CI numpy sur la hierarchie
   .step_cfl(0.4)         ---> 1 macro-pas explicite a CFL=0.4 (regrid tous les 10 pas)
   .mass()/.n_patches()/  ---> diagnostics (lus depuis la hierarchie)
   .density()/.time()
```

Python ne fait que **composer** la config (briques, IC en numpy, criteres) et **piloter** la CI ;
tout le calcul (transport, Poisson, regrid, reflux) reste en C++. `adc.AmrSystem` est le pendant
raffine de `adc.System` ; les methodes `set_*`, `step_cfl`, `mass`, `density`, `n_patches`, `time`
traversent la facade Python (`__getattr__`) vers le binding C++ `adc._adc.AmrSystem`.

`AmrSystem` est ici utilise en **mono-bloc** (1 seul `add_block`) : c'est le chemin historique
`AmrCouplerMP` qui supporte le **regrid dynamique** (`regrid_every > 0`). Le mode multi-blocs
existe mais fige la hierarchie (il refuse `regrid_every > 0`) ; ce cas n'en a pas besoin.

---

## 6. Carte des fichiers

| Fichier | Role |
|---------|------|
| `diocotron_amr/run.py` | le cas : construit l'`AmrSystem`, boucle, asserts, run de controle |
| `adc_cases/models.py` | `diocotron(B0, alpha, n_i0)` = composition des 4 briques |
| `adc_cases/common/initial_conditions.py` | `band_density(...)` : la bande gaussienne perturbee |
| `adc_cases/common/grid.py` | `meshgrid_xy` / `cell_centers` : grille a centres de cellules |
| `adc_cases/common/checks.py` | `assert_finite`, `relative_drift` : invariants |
| `adc/__init__.py` (build) | facade `adc.AmrSystem`, `adc.Spatial`, briques `Scalar/ExB/...` |
| `cases_manifest.toml` | declare le cas : `category="validation"`, `ci=true`, `needs=[]` |

Le cas n'a **pas** de C++ propre (pas de `native.py`/`build_shared`) : il ne compose que des
briques natives deja compilees dans le binding `_adc`.

---

## 7. Prerequis

- Le paquet Python `adc` (binding `_adc.cpython-*.so`) doit etre importable. Dans cet
  environnement, il est fourni pre-compile sous
  `/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python`.
- `numpy` (seule dependance externe du cas ; `needs = []` au manifeste signifie : aucun extra
  comme `matplotlib` ni compilateur C++).
- Le paquet partage `adc_cases` doit etre importable : soit installe, soit sur `PYTHONPATH`
  (le `run.py` ajoute automatiquement la racine du depot au `sys.path` en repli si l'import echoue).
- **Aucun** compilateur C++ requis a l'execution (rien n'est compile a la volee), **aucun**
  matplotlib (le cas ne produit pas de figure).

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/diocotron_amr
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le premier element du `PYTHONPATH` apporte le paquet `adc` (binding C++), le second apporte le
paquet `adc_cases` (modeles + IC + checks partages).

---

## 9. Explication du code par etapes

`run.py` se lit en deux temps : une fonction d'usine `build_sim` et un `main`.

**`build_sim(ne, n_i0, threshold)`** construit un `AmrSystem` identique au nominal au seuil pres
(reutilise par le run de controle) :

1. `sim = adc.AmrSystem(n=64, L=1.0, regrid_every=10, periodic=True)` : hierarchie AMR, niveau de
   base 64x64, regrid tous les 10 pas, BC periodiques.
2. `sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0), spatial=adc.Spatial(none=True))`
   : le bloc diocotron (NoSlope + Rusanov) porte sur la hierarchie.
3. `sim.set_refinement(threshold=threshold)` : seuil de tagging des cellules a raffiner.
4. `sim.set_poisson(rhs="charge_density", solver="geometric_mg")` : Poisson multigrille periodique.
5. `sim.set_density("ne", ne)` : projette la CI numpy sur la hierarchie.

**`main()`** :

1. construit la CI `ne = band_density(64, 1.0, amp=1.0, width=0.05, mode=4, disp=0.02)` ;
2. fixe `n_i0 = ne.mean()` (fond neutralisant pour Poisson periodique) ;
3. construit le run nominal avec `threshold = n_i0 + 0.15` ;
4. capture `mass0 = sim.mass()` et verifie qu'elle est finie et positive ;
5. **boucle de 40 pas** : `sim.step_cfl(0.4)`, puis a chaque pas
   - `npatch = sim.n_patches()` -> assert `npatch >= 2` (la bande est couverte par >= 2 patchs) ;
   - `drel = relative_drift(mass, mass0)` -> assert `drel < 1e-9` (masse conservee) ;
   - `assert_finite(sim.density(), ...)` (pas de NaN/Inf) ;
   - impression d'une ligne tous les 10 pas (et au dernier pas) ;
6. apres la boucle, assert `min(patches_seen) >= 2` ;
7. **run de controle** : meme CI, mais `threshold = 1e30` (inatteignable) -> aucune cellule taggee.
   On avance 40 pas, on lit `npatch_ctrl = ctrl.n_patches()` et la densite finale `dens_ctrl` ;
8. `gap = max|dens - dens_ctrl|` (ecart sup nominal vs controle) ;
9. asserts finaux : `min(patches_seen) > npatch_ctrl` (le seuil discrimine) et
   `gap > 1e-3` (le raffinement change la solution) ;
10. impression `OK diocotron_amr`.

---

## 10. Conditions initiales

CI = `band_density` (`adc_cases/common/initial_conditions.py`) : bande horizontale gaussienne de
charge, perturbee sinusoidalement le long de x (mode azimutal `mode`).

```
n_e(x, y) = floor + amp * exp( -(y - y0)^2 / width^2 )
y0        = 0.5 L + disp * cos( 2 pi mode x / L )
```

Parametres du cas (`run.py`) : `N = 64`, `L = 1.0`, `amp = 1.0`, `width = 0.05`, `mode = 4`,
`disp = 0.02`, `floor = 1.0` (defaut). La bande est centree en `y = 0.5`, ondulee 4 fois en x.
Convention de grille `ne[j, i]` a centres de cellules (`meshgrid_xy`).

Le fond neutralisant `n_i0 = ne.mean()` (mesure : `1.0886`) assure la moyenne nulle du second
membre de Poisson periodique.

> Cette IC `band_density` est partagee avec le cas `diocotron` (grille uniforme) et `custom_scheme`.
> Elle differe de l'**anneau** `ring_density` du benchmark publie (`arXiv:2510.11808`) : le cas
> `diocotron_amr` n'est pas une reproduction de ce papier (voir section 16).

---

## 11. Invariants et assertions

Tous verifies a l'execution (valeurs reelles capturees ci-dessous) :

| Assertion (`run.py`) | Seuil | Valeur mesuree |
|----------------------|-------|----------------|
| `mass0` finie et `> 0` | -- | `mass0 = 1.088622692545e+00` |
| `npatch >= 2` a chaque pas | `>= 2` | `2` a tous les pas (patchs observes : `[2]`) |
| `drel < TOL_MASS` (masse conservee) | `< 1e-9` | `drel` final `= 8.159e-16` (arrondi machine) |
| `assert_finite(density)` | -- | densite finie partout (min `1.0`, max `1.966797`) |
| `min(patches_seen) >= 2` | `>= 2` | `2` |
| `min(patches_seen) > npatch_ctrl` (le seuil discrimine) | `2 > 1` | nominal `2`, controle `1` |
| `gap > MIN_SOLUTION_GAP` (le raffinement change la solution) | `> 1e-3` | `gap = 6.395745e-02` |

La conservation de masse a ~1e-16 atteste que le **reflux** corrige bien les flux aux interfaces
grossier/fin (sans reflux, la masse deriverait a chaque pas de regrid). Le contraste
nominal (2 patchs) vs controle (1 patch) atteste que les patchs fins viennent du **tagging**.

---

## 12. Sorties attendues

Le cas n'ecrit que sur stdout (pas de fichier). Sortie reelle observee :

```
# cas diocotron_amr : instabilite diocotron sur AMR multi-patch (adc.AmrSystem)
# n_base=64 regrid_every=10 band_mode=4  n_i0=1.0886
# step  t        patches  mass          drel
  0     0.1642   2        1.08862269e+00 4.079e-16
  10    1.8096   2        1.08862269e+00 2.040e-16
  20    3.4538   2        1.08862269e+00 8.159e-16
  30    5.1082   2        1.08862269e+00 1.428e-15
  39    6.6204   2        1.08862269e+00 8.159e-16
# patchs observes : [2]
# masse : init=1.088622692545e+00 final=1.088622692545e+00 drel=8.159e-16
# densite : min=1.000000e+00 max=1.966797e+00
# controle (seuil 1e+30) : patches=1  ecart_sup solution=6.395745e-02
OK diocotron_amr
```

Points cles : 40 pas avancent jusqu'a `t ~ 6.62`, la masse reste a `1.088622692545` (12 chiffres
stables), le run nominal a **2 patchs** a tous les pas, le controle **1 patch**, et l'ecart sup
entre les deux solutions vaut `6.40e-2` (bien au-dessus du seuil `1e-3`). La ligne `OK diocotron_amr`
signale que tous les asserts sont passes.

---

## 13. Generation figures/GIF

**Aucune.** Ce cas est un test de validation pur (`needs = []`, pas de `matplotlib`) : il ne
produit ni figure ni GIF, seulement une trace stdout. Le cas frere `diocotron/run.py` (categorie
`reproduction`, `needs = ["matplotlib"]`, hors CI) est celui qui genere figures + gif sur grille
uniforme.

---

## 14. Backends reellement supportes

- **CPU host** : c'est le chemin exerce ici. Le binding `_adc.cpython-312-darwin.so` (Kokkos
  backend hote) tourne en multi-thread (run mesure a ~511 % CPU, soit ~5 coeurs).
- **GPU / MPI** : le coeur `adc_cpp` cible aussi des backends Kokkos device (GH200) et MPI, et la
  brique AMR (regrid B_z device, reflux multi-box) y est validee dans le projet amont. **Mais ce
  cas precis** ne fait que composer des briques natives via la facade Python sur le binding fourni :
  il s'execute sur le backend du `_adc.so` charge (ici CPU host). Aucun chemin GPU/MPI n'est
  exerce ni requis par `run.py`.
- **Pas de compilation a la volee** : contrairement aux cas `*_dsl` / `two_fluid_ap`
  (`needs = ["cxx"]`), ce cas n'invoque aucun compilateur.

---

## 15. Cout approximatif

Mesure reelle (Apple Silicon, `/usr/bin/time -p`, binding CPU host) :

| Grandeur | Valeur |
|----------|--------|
| Temps mur total (run nominal 40 pas + controle 40 pas + import) | **~0.36-0.38 s** |
| Temps utilisateur cumule | ~1.6 s (multi-thread, ~5 coeurs, ~511 % CPU) |
| Grille de base | 64x64 + 1 niveau fin |
| Pas de temps | 40 (nominal) + 40 (controle) = 80 macro-pas, regrid tous les 10 |

Cas tres leger, adapte a la CI. La majeure partie du temps est l'import du binding C++ et la
construction des deux hierarchies AMR ; les 80 pas eux-memes sont quasi instantanes.

---

## 16. Limites et differences avec les references

- **Ce n'est pas une reproduction d'un resultat publie.** Categorie `validation` au manifeste : on
  teste la **capacite AMR** de l'API (vrai raffinement, reflux conservatif, effet sur la solution),
  pas la reproduction quantitative d'un taux de croissance ou d'une figure d'article.
- **IC differente du benchmark diocotron publie.** Le cas utilise une **bande** gaussienne
  (`band_density`) sur domaine periodique, pas l'**anneau** (`ring_density`) du benchmark
  `arXiv:2510.11808`. Ne pas presenter ce cas comme la reproduction de Hoffart/diocotron : la
  reproduction etablie (figures + gif) vit dans `diocotron/run.py` (grille uniforme, hors CI), et la
  candidate Euler-Poisson magnetisee complete vit dans `hoffart_euler_poisson_dsl/` (statut
  `reproduction-candidate` PENDING, hors CI).
- **Schema d'ordre 1.** NoSlope + Rusanov est volontairement dissipatif (robustesse sur grille
  grossiere) : il lisse les petits details de l'enroulement. Ce n'est pas le schema d'ordre eleve
  qu'on emploierait pour une etude de convergence ou de taux de croissance.
- **Mono-bloc, 1 niveau fin.** Le cas exerce un bloc unique et un seul niveau de raffinement
  (grossier + fin). Le multi-blocs partage et les hierarchies a plusieurs niveaux ne sont pas
  couverts ici. Le nombre de patchs observe est exactement 2 (suffisant pour discriminer le tagging
  du build degenere a 1 patch du controle), pas un grand nombre de patchs.
- **Le seuil de raffinement est cale empiriquement** (`threshold = n_i0 + 0.15`) pour que la bande
  soit taggee mais pas tout le domaine ; il n'est pas issu d'un critere d'erreur a posteriori.

---

## 17. Tests/CI associes

- **Manifeste** (`cases_manifest.toml`) : ce cas est declare
  `path = "diocotron_amr/run.py"`, `category = "validation"`, `ci = true`, `needs = []`,
  `desc = "Instabilite diocotron sur AMR multi-patch (adc.AmrSystem, reflux conservatif)."`.
- **CI** : la CI (`.github/workflows/ci.yml`) ne lance que les cas legers marques `ci = true`.
  Ce cas en fait partie : il s'execute tel quel (commande de la section 8), et le succes est defini
  par l'absence d'`AssertionError` et l'impression de `OK diocotron_amr` en derniere ligne.
- **Le test EST le run.py** : il n'y a pas de fichier de test separe. Les invariants de la section 11
  sont les criteres de validation ; un retour non nul ou l'absence de `OK diocotron_amr` signe un
  echec.
