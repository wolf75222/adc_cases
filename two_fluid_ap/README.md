# two_fluid_ap — bi-fluide isotherme raide (asymptotic-preserving)

Solveur deux-fluides isotherme 2D en regime RAIDE (frequence plasma elevee), integre
par un schema IMEX / asymptotic-preserving (AP). Le coeur du terme raide (couplage
plasma au champ electrique) est traite de maniere implicite, ce qui rend le schema
stable et conservatif meme quand `dt * omega_pe >> 1`, regime ou un schema explicite
exploserait.

Categorie manifeste : **validation** (`ci = true`, `needs = ["cxx"]`). Le solveur AP
n'est PAS une brique composable `adc.System` : l'integrateur AP a quitte le coeur
adc_cpp et vit ICI, en C++ sur mesure (`two_fluid_ap.hpp` + `_two_fluid_ap.cpp`),
compile a la volee et charge via ctypes.

---

## 1. Objectif du cas

Demontrer la propriete asymptotic-preserving (AP) d'un schema IMEX sur un modele
bi-fluide isotherme charge (electrons + ions) couple par un champ electrique de Poisson.

Concretement, le cas verifie par assertions que, pour un PAS DE TEMPS volontairement
grand devant l'echelle raide (`dt * omega_pe = 5` au run 1), le schema :

- ne diverge pas (toutes les diagnostics restent finies) ;
- maintient la quasi-neutralite (ecart densite electronique au fond `< 0.1`) ;
- maintient une charge nette locale faible (`< 0.1`) ;
- conserve la masse electronique (erreur relative `< 1e-7`).

Un schema explicite serait limite par `dt * omega_pe = O(1)` et exploserait des le
premier grand pas. Le cas n'est PAS une reproduction d'un resultat publie : c'est un
test de validation numerique de la stabilite/conservation AP.

---

## 2. Equations

Deux fluides isothermes charges (espece `s` ∈ {electron `e`, ion `i`}), densite `n_s`
et quantite de mouvement `m_s = (m_{s,x}, m_{s,y})`, couples au potentiel electrique
`phi` :

```
dt n_s   + div(m_s) = 0                                       (continuite)
dt m_s   + div( m_s ⊗ m_s / n_s + c_s^2 n_s I )
                     = z_s n_s E   (+ z_s n_s (u_s × B_z))     (quantite de mouvement isotherme)
lap(phi) = (n_e - n_i)                                        (Poisson, z_e = -1, z_i = +1)
E        = - grad phi
```

avec `c_s^2` la vitesse du son isotherme au carre par espece (`cse2`, `csi2` dans le
code), `z_e = -1`, `z_i = +1`, densite de fond `n0 = 1`. Le champ magnetique est
uniforme hors-plan `B = B_z ẑ` (terme de rotation cyclotron optionnel).

Raideur : le couplage `z_s n_s E` avec `E` issu de Poisson fait apparaitre les
frequences plasma `omega_pe`, `omega_pi`. Dans le code, le solveur stocke
`ce = omega_pe^2` et `ci = omega_pi^2` (coefficients de couplage `coup` dans
`tfap_lorentz` / `tfap_boris`).

**Reformulation AP de l'elliptique.** Au lieu de resoudre `lap(phi) = ne* - ni*` puis
d'appliquer la force explicitement, le schema STABILISE le Poisson en absorbant le
terme raide dans l'elliptique. Avec `n0 = 1`, le facteur de stabilisation est constant
`beta0 = dt^2 (omega_pe^2 + omega_pi^2) = dt^2 (ce + ci)` et le Poisson devient :

```
lap(phi) = (ne* - ni*) / (1 + beta0)
```

C'est exactement ce que la composition `adc.System` ne sait pas exprimer (le pas de
temps `dt` apparait DANS le membre de droite de l'elliptique), d'ou le solveur sur
mesure. Voir `two_fluid_ap.hpp:258-266`.

---

## 3. Modele physique

- **Isotherme** : pas d'equation d'energie ; la pression est `p_s = c_s^2 n_s`
  (gaz isotherme), `c_s` constant par espece. Etat conservatif a 3 composantes par
  espece : `(n, m_x, m_y)`.
- **Quasi-neutralite** : au repos `n_e = n_i = n0 = 1`. La perturbation initiale ne
  porte que sur la densite electronique (run 1). La propriete AP maintient
  `n_e ≈ n_i ≈ 1` malgre le grand pas de temps.
- **Couplage electrique** : champ auto-consistant `E = -grad phi` avec `phi` solution
  de Poisson sur l'ecart de densite. C'est ce couplage qui rend le systeme raide.
- **Magnetisation optionnelle** (run 2) : champ `B_z` uniforme hors-plan. La force de
  Lorentz magnetique fait TOURNER `(m_x, m_y)` a la frequence cyclotron
  `omega_ce` (electrons) / `omega_ci` (ions), sans changer `|m|` ni `n`. La rotation
  exacte est inconditionnellement stable (pas de limite `Omega * dt`).

Geometrie : domaine carre `[0, L]^2` periodique dans les deux directions
(`Periodicity{true, true}`), grille `n × n` a centres de cellules.

---

## 4. Methode numerique

Schema IMEX / asymptotic-preserving, scinde en sous-etapes (un pas `step(dt)` dans
`TwoFluidAP2D::step`, `two_fluid_ap.hpp:239-290`) :

1. **Remplissage des ghosts** periodiques (`fill_boundary`).
2. **Predicteur quantite de mouvement** `m*` : flux d'Euler isotherme par Rusanov
   (local Lax-Friedrichs) dimensionnellement scinde, vitesse d'onde `a = |u| + c_s`,
   `Fxx = m_x^2/n + c^2 n`, `Fyy = m_y^2/n + c^2 n`, etc. (`tfap_mstar`).
3. **Predicteur densite** `n*` : `n - dt div(m*)`. Deux variantes selectionnables :
   - **centree** (defaut, `upwind_continuity = false`) : divergence centree d'ordre 2,
     dissipation nulle (`tfap_div_update`) ;
   - **upwind** (`upwind_continuity = true`) : flux de masse Rusanov reconstruit a la
     face avec pente MINMOD (MUSCL ordre 2), coherent avec la vitesse d'onde du flux de
     quantite de mouvement ; anti-Gibbs sur les fronts raides (`tfap_div_update_up`,
     2 ghosts sur `n`). NON utilise par run.py (les deux runs gardent le defaut centre).
4. **Poisson AP** : RHS `(ne* - ni*) / (1 + beta0)` puis `ell.solve()`
   (`EllipticSolver`). Stabilisation `beta0 = dt^2 (ce + ci)` si `stabilize = true`
   (toujours vrai ici), `0` sinon.
5. **Champ electrique** `E = -grad phi` (differences centrees, `tfap_efield`).
6. **Mise a jour quantite de mouvement implicite (terme raide)** :
   - non magnetise : `m^{n+1} = m* + dt z coup E` (`tfap_lorentz`) ;
   - magnetise : push de Boris symetrique (demi-impulsion E -> rotation B complete
     d'angle `theta = z omega_c dt` -> demi-impulsion E, `tfap_boris`), qui reproduit
     exactement la derive E×B et conserve `|m|` sous B seul.
7. **Correcteur densite** : `n^{n+1} = n - dt div(m^{n+1})` (meme variante qu'a
   l'etape 3), puis recopie de `(m_x, m_y)` dans l'etat (`copy_mom`).

L'elliptique est templatise sur un `EllipticSolver` ; la facade `_two_fluid_ap.cpp`
instancie concretement `TwoFluidAP2D<GeometricMG>` (multigrille geometrique, lisseur
Gauss-Seidel rouge-noir + V-cycle, entierement on-device, `geometric_mg.hpp`). Le
header documente que `PoissonFFT` conviendrait aussi en CPU, mais le code REELLEMENT
compile par ce cas utilise `GeometricMG` sur les deux backends.

---

## 5. Architecture ADC utilisee

Point cle : ce cas n'utilise du coeur `adc_cpp` que des BRIQUES GENERIQUES (header-only,
incluses via `-I adc_cpp/include`), JAMAIS un scenario nomme. La physique deux-fluides
AP vit entierement dans le cas.

Briques generiques du coeur reellement incluses (`two_fluid_ap.hpp:11-20` et
`_two_fluid_ap.cpp:13-16`) :

| Brique du coeur | Header | Role dans le cas |
|---|---|---|
| `MultiFab` / `Fab2D` / `Array4` | `adc/mesh/multifab.hpp`, `fab2d.hpp` | conteneurs d'etat (n, m_x, m_y) avec ghosts |
| `BoxArray` / `DistributionMapping` / `Geometry` | `adc/mesh/box_array.hpp`, ... | decoupage du domaine et coordonnees cellules |
| `for_each_cell` (`ADC_HD`) | `adc/mesh/for_each.hpp` | kernels device-portables (CPU/GPU) |
| `fill_boundary` / `Periodicity` | `adc/mesh/fill_boundary.hpp` | ghosts periodiques |
| `EllipticSolver` (concept) + `GeometricMG` | `adc/numerics/elliptic/...` | Poisson AP (RHS reformule) |
| `adc::sum`, `device_fence` | `adc/mesh/multifab.hpp`, `for_each.hpp` | reductions (masse) et barriere host/device |
| `comm` (`n_ranks`, `n_ranks()`) | `adc/parallel/comm.hpp` | parallelisme MPI (boites/rangs) |

Le solveur est ecrit "device-clean" : tous les kernels passent par `for_each_cell` avec
des lambdas `ADC_HD`, `|x|`/`max`/`minmod` via ternaires (`std::fabs`/`std::fmax` ne sont
pas device-safe), `cos`/`sin`/`sqrt` calcules cote hote pour les champs uniformes. La
facade compile donc telle quelle pour le GPU si on lui passe les flags/include adequats.

Cote Python (`run.py`), aucun binding C++ du cas : la classe `TwoFluidAP` est un pilote
pur ctypes qui remplace l'ancien echappatoire interne `adc._adc._TwoFluidAP` (retire du
coeur), avec la meme API (`step` / `advance` / `mass_e` / `max_charge` / `max_dev` / ...).

---

## 6. Carte des fichiers

```
two_fluid_ap/
├── README.md            ← ce fichier
├── run.py               ← pilote Python : build JIT, ctypes, 2 scenarios, asserts, prints
├── two_fluid_ap.hpp     ← physique AP (kernels Rusanov/MUSCL, Boris, Poisson AP, TwoFluidAP2D)
└── _two_fluid_ap.cpp    ← ABI extern "C" (tfap_create/step/advance/diagnostics) + GeometricMG
```

Dependances sur le paquet partage `adc_cases/` (a la racine du depot) :

```
adc_cases/
├── __init__.py          ← REPO_ROOT, ensure_importable()
└── common/
    ├── native.py        ← build_shared() (cache hors source + cle d'ABI), load_symbols()
    └── io.py            ← case_output_dir() -> out/<cas>/ (gitignore)
```

Le cas n'utilise PAS `adc_cases/models.py`, `recipes.py`, `common/grid.py`,
`common/initial_conditions.py`, `common/checks.py` ni `common/native.py:adc_include`
au-dela de ce que `build_shared` appelle : la CI initiale en densite est codee
directement en C++ (`TwoFluidAP2D::init`, `two_fluid_ap.hpp:227-237`).

En-tetes du coeur (hors worktree) : `adc_cpp/include/adc/...`, localises par
`native.adc_include()` (via `$ADC_INCLUDE`, sinon depuis le paquet `adc` installe,
sinon `../adc_cpp/include`).

---

## 7. Prerequis

- **Python 3** avec **numpy** (seul module Python tiers requis par le cas).
- Le module C++ **`adc`** (bindings pybind11 d'adc_cpp) accessible via `PYTHONPATH`.
  Il n'est utilise que par `native.adc_include()` pour localiser `adc_cpp/include`
  (et par le paquet `adc_cases`) ; le calcul AP ne passe PAS par les bindings.
- Un **compilateur C++20** (`needs = ["cxx"]` au manifeste) : `c++` / `g++` / `clang++`,
  ou `$CXX`. Le solveur AP est compile a la volee en bibliotheque partagee.
- Les **en-tetes du coeur** `adc_cpp/include/` (header-only) trouvables par
  `native.adc_include()`.

Environnement de validation reel (capture ci-dessous) :

- macOS arm64 (Darwin 25.5.0), Apple clang 21.0.0 (`/usr/bin/c++`), C++20.
- Python 3.12 (`/opt/homebrew/anaconda3/bin/python3.12`).
- `adc_cpp/include` present, bindings `adc` fournis par
  `adc_cpp/build-master/python`.

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/two_fluid_ap
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Forme generique (paquet installe via `pip install -e .` a la racine) :

```bash
PYTHONPATH=<adc_cpp>/build-py/python python3 two_fluid_ap/run.py
```

En CI, c'est exactement `python3 two_fluid_ap/run.py` qui est lance (voir section 17),
avec `PYTHONPATH` pointant le build adc_cpp.

---

## 9. Explication du code par etapes

`run.py` (pilote Python) :

1. **Import du paquet** : `import adc_cases` ; a defaut, ajoute la racine du depot au
   `sys.path` puis importe `adc_cases.common.native` (`run.py:50-54`).
2. **Build JIT + chargement** (`_build_lib`, `run.py:69-78`) :
   `native.build_shared("two_fluid_ap", [_two_fluid_ap.cpp, two_fluid_ap.hpp])` compile
   la lib dans `out/two_fluid_ap/build/` (jamais a cote du `.cpp`), indexee par une cle
   d'ABI (hash compilateur + flags + sources + signature de l'arbre d'en-tetes du coeur).
   Recompilation UNIQUEMENT si la cle change. Puis `native.load_symbols` verifie que les
   12 symboles `tfap_*` attendus existent (ABI mismatch = erreur explicite).
3. **Liaison ctypes** (`_bind`, `run.py:81-104`) : declare `restype`/`argtypes` des
   fonctions `extern "C"` (notamment `tfap_create` a 11 arguments).
4. **Classe pilote `TwoFluidAP`** (`run.py:107-160`) : wrappe le handle opaque,
   expose `step`/`advance`/`mass_e`/`mass_i`/`max_charge`/`max_dev`/`density_e`/
   `density_i`. `__del__` libere via `tfap_destroy`.
5. **Run 1 raide** (`run_stiff`, `run.py:168-204`) : voir section 10/11.
6. **Run 2 magnetise** (`run_magnetized`, `run.py:207-237`).
7. **`main`** : build, run 1, run 2, impression de la conclusion et de `OK two_fluid_ap`.

`_two_fluid_ap.cpp` (cote C++) :

- `tfap_create(...)` construit un `Solver` (handle opaque) encapsulant
  `TwoFluidAP2D<GeometricMG>` + le drapeau `stabilize`, et appelle `d.init(eps)`.
- `tfap_step` / `tfap_advance` appellent `d.step(dt, stabilize)` une ou N fois.
- Diagnostics : `tfap_mass_e/i` = `adc::sum(...)`, `tfap_max_charge` =
  `max |n_i - n_e|`, `tfap_max_dev` = `max |n_e - 1|`, avec `device_fence()` avant
  toute lecture hote (memoire unifiee GPU). `tfap_density_e/i` recopient la composante
  densite (row-major) dans un buffer fourni par Python.

`two_fluid_ap.hpp` (physique) : kernels `tfap_mstar`, `tfap_div_update`,
`tfap_div_update_up`, `tfap_efield`, `tfap_lorentz`, `tfap_boris`, `tfap_rotate_mom`,
`tfap_copy_n`, et la classe `TwoFluidAP2D<Elliptic>` (`init` + `step`).

---

## 10. Conditions initiales

Codees en C++ dans `TwoFluidAP2D::init(eps)` (`two_fluid_ap.hpp:227-237`), en boucle
hote (memoire unifiee) :

```
n_e(x, y) = 1 + eps * cos(k x + k y),   k = 2 pi / L
m_e       = 0
n_i(x, y) = 1                            (fond uniforme)
m_i       = 0
```

`eps` est passe depuis Python via `tfap_create` ; le pilote `TwoFluidAP` utilise le
defaut `eps = 1e-3`. Les deux runs construisent le solveur avec ce defaut. La
perturbation ne porte donc QUE sur la densite electronique ; la charge nette initiale
`n_i - n_e = -eps cos(...)` est de l'ordre de `1e-3`.

Parametres des deux scenarios (constructeur `TwoFluidAP`, defauts `n=64`,
`L=2*pi`, `cse2=1.0`, `csi2=0.04`) :

| | run 1 (raide, non magnetise) | run 2 (raide magnetise) |
|---|---|---|
| `omega_pe` / `omega_pi` | `1e3` / `20` | defaut `5.0` / `1.0` |
| `omega_ce` / `omega_ci` | `0` / `0` | `4.0` / `0.2` |
| `dt` | `5e-3` (= `5/1e3`) | `1e-2` |
| `nsteps` | `200` | `100` |
| `dt * omega_pe` | `5` (explicite exploserait) | — |
| `stabilize` | `true` | `true` |

---

## 11. Invariants et assertions

Les assertions du code (et leurs valeurs REELLES mesurees, cf. section 12) :

**Run 1** (`run_stiff`, `run.py:195-202`) :

- `np.isfinite(max_dev)`, `np.isfinite(max_charge)`, `np.isfinite(mass_e)` — le grand
  pas n'a pas explose. **Mesure** : toutes finies.
- `max_dev < 0.1` (quasi-neutralite maintenue). **Mesure** : `max_dev = 5.325451e-07`.
- `max_charge < 0.1` (charge nette locale faible). **Mesure** : `6.697598e-11`.
- `mass_rel < 1e-7` (masse electronique conservee). **Mesure** : `2.276e-14`
  (`mass_e` : `4.096000e+03 -> 4.096000e+03`).

**Run 2** (`run_magnetized`, `run.py:232-235`) :

- `np.isfinite(max_dev)`, `np.isfinite(mass_e)`. **Mesure** : finies.
- `mass_rel < 1e-7` (masse conservee sous rotation cyclotron). **Mesure** : `1.665e-14`
  (`mass_e` : `4.096000e+03 -> 4.096000e+03`).

Note : la masse electronique vaut `4096 = 64 x 64 x 1` (densite de fond 1 integree sur
toutes les cellules ; la perturbation `cos` est de moyenne nulle). La conservation a
`~1e-14` est au niveau du bruit d'arrondi IEEE754.

Au niveau C++, `device_fence()` garantit que les reductions hote (`max_charge`,
`max_dev`, `adc::sum`) lisent un etat device synchronise (memoire unifiee GPU).

---

## 12. Sorties attendues

Le cas n'ecrit aucune figure ni fichier de donnees : il imprime des diagnostics
numeriques et termine par `OK two_fluid_ap`. Sortie REELLE capturee (premier run, avec
la ligne de (re)compilation JIT) :

```
=== Demo two_fluid_ap : bi-fluide isotherme raide (asymptotic-preserving) ===
two_fluid_ap : (re)compilation du solveur natif
  /usr/bin/c++ -shared -fPIC -std=c++20 -O2 -I .../adc_cpp/include .../_two_fluid_ap.cpp -o .../out/two_fluid_ap/build/_two_fluid_ap.dylib
[run 1 - raide, non magnetise]
  n=64  omega_pe=1.000e+03  omega_pi=2.000e+01
  dt=5.000e-03  nsteps=200  dt*omega_pe=5.0  (explicite EXPLOSERAIT)
  max_dev()    = 5.325451e-07   (ecart a la quasi-neutralite)
  max_charge() = 6.697598e-11   (charge nette locale)
  mass_e: 4.096000e+03 -> 4.096000e+03   (err. relative 2.276e-14)
[run 2 - raide magnetise]
  n=64  omega_ce=4.000e+00  omega_ci=2.000e-01
  dt=1.000e-02  nsteps=100
  max_dev()    = 9.447867e-04
  max_charge() = 7.732753e-04
  mass_e: 4.096000e+03 -> 4.096000e+03   (err. relative 1.665e-14)
Conclusion : schema IMEX / asymptotic-preserving stable et conservatif
pour un plasma raide, magnetise ou non (un schema explicite echouerait).
OK two_fluid_ap
```

Sur un run ulterieur (cache de build a jour), la ligne de (re)compilation disparait
(la `.dylib` est reutilisee). La presence de `OK two_fluid_ap` et un code de retour 0
signent le succes ; toute assertion violee leverait `AssertionError` avec un message
explicite (ex. `quasi-neutralite non maintenue`).

Artefact produit (hors source, gitignore), conforme a la note du manifeste :

```
out/two_fluid_ap/build/_two_fluid_ap.dylib        (sur Linux : _two_fluid_ap.so)
out/two_fluid_ap/build/_two_fluid_ap.dylib.abikey (cle d'ABI du cache)
```

---

## 13. Generation figures/GIF

Aucune. Ce cas ne genere ni figure ni GIF : c'est un test de validation a sortie
textuelle. Les fonctions `density_e()` / `density_i()` du pilote exposent bien les
champs `n_e` / `n_i` en tableaux numpy `n × n` (via `tfap_density_e/i`), mais `run.py`
ne les utilise pas (aucune dependance matplotlib ; `needs` ne contient que `cxx`).

---

## 14. Backends reellement supportes

- **CPU serie** : valide reellement ici (Apple clang 21, C++20). C'est le backend par
  defaut quand `-I adc_cpp/include` est passe sans flag de parallelisme. Le solveur
  utilise `n_ranks()` (1 rang sans MPI) et `GeometricMG` purement CPU.
- **GPU / OpenMP / Kokkos** : le code est ecrit device-clean (kernels `for_each_cell`
  `ADC_HD`, `GeometricMG` on-device, `device_fence()` avant lectures hote), donc la
  facade compile pour ces backends SI on passe les flags/include adequats a la
  compilation (le backend est herite de la facon dont `-I` + flags sont passes via
  `build_shared`). Ce cas ne fixe AUCUN flag de backend : `build_shared` compile avec
  `-shared -fPIC -std=c++20 -O2` uniquement. La validation GPU/Kokkos N'A PAS ete
  exercee par ce cas dans cet environnement (CPU serie seulement).
- **Elliptique** : seul `GeometricMG` est reellement instancie
  (`TwoFluidAP2D<GeometricMG>`). Le header evoque `PoissonFFT` comme alternative CPU,
  mais elle n'est pas compilee ici.
- **Compilateur** : C++20 requis (concept `EllipticSolver`, `static_assert`). `$CXX`
  prioritaire, sinon `c++`/`g++`/`clang++`.

---

## 15. Cout approximatif

Mesures reelles (macOS arm64, Apple clang 21, Python 3.12, `n = 64`, CPU serie) :

| | temps mur | detail |
|---|---|---|
| 1er run (avec compilation JIT) | **~5.5 s** (`5.541 total`, 98% CPU) | compile la `.dylib` + 2 scenarios |
| run suivant (cache a jour) | **~3.6 s** (`3.574 total`) | aucune recompilation, lib reutilisee |

Le calcul AP lui-meme (300 pas de `step` au total : 200 + 100, grille `64 x 64`,
2 fluides, Poisson multigrille par pas) est de l'ordre de quelques secondes ;
l'essentiel du surcout du premier appel est la compilation C++. Recompilation
declenchee uniquement si la cle d'ABI change (modif compilateur, flags, sources, ou
en-tetes du coeur).

---

## 16. Limites et differences avec les references

- **Pas une brique composable.** Le solveur AP n'est PAS un `adc.System` : l'integrateur
  AP a quitte le coeur adc_cpp parce que la stabilisation couple `dt` a l'elliptique
  (`lap(phi) = (ne* - ni*)/(1 + dt^2(wpe^2+wpi^2))`), ce que la composition bloc-a-bloc
  ne sait pas exprimer. Il vit dans `adc_cases` et n'emprunte au coeur que des briques
  generiques.
- **Pas une reproduction publiee.** Categorie manifeste = `validation`, pas
  `reproduction` : le cas verifie des invariants AP (stabilite, quasi-neutralite,
  conservation), il ne reproduit AUCUNE figure ni table d'un article. Ne pas le
  presenter comme une reproduction.
- **Stabilisation a coefficient constant.** La reformulation AP suppose `n0 = 1`
  constant, donc `beta0 = dt^2 (ce + ci)` est uniforme (Poisson a coefficient constant,
  resoluble par MG/FFT). Une densite de fond variable demanderait un elliptique a
  coefficient variable, non implemente ici.
- **Schema spatial bas ordre.** Continuite centree (defaut) d'ordre 2 sans dissipation,
  flux de quantite de mouvement Rusanov (LLF) ordre 1. La variante MUSCL/minmod
  (`upwind_continuity`) existe mais N'est PAS activee par run.py. Petites grilles
  (`n = 64`), perturbation lineaire (`eps = 1e-3`) : le cas teste la PROPRIETE AP, pas
  une dynamique non lineaire fine.
- **Backend valide = CPU serie seulement** dans cet environnement (cf. section 14). La
  portabilite GPU/Kokkos est une propriete du code (device-clean) non exercee ici.
- **Magnetisation simplifiee** : champ `B_z` uniforme hors-plan, rotation cyclotron
  exacte (Boris). Pas de champ magnetique auto-consistant ni de dynamique 3D.

---

## 17. Tests/CI associes

- Manifeste (`cases_manifest.toml`) : `path = "two_fluid_ap/run.py"`,
  `category = "validation"`, `ci = true`, `needs = ["cxx"]`.
- CI (`.github/workflows/ci.yml`) : le workflow lit le manifeste, selectionne les cas
  `ci = true` et lance `python3 two_fluid_ap/run.py` (avec
  `PYTHONPATH=<workspace>/adc_cpp/build-py/python`). Le besoin `cxx` est satisfait par
  le compilateur systeme du runner. Le cas est donc un test de non-regression : il echoue
  (code de retour non nul) si une assertion AP est violee, si la compilation JIT echoue,
  ou si l'ABI des symboles `tfap_*` ne correspond plus.
- Le cas est auto-suffisant : pas de fixture externe, conditions initiales codees en C++,
  diagnostics et asserts dans `run.py`. Le succes est la ligne `OK two_fluid_ap`.
- Lien d'historique : ce solveur remplace l'ancien echappatoire interne
  `adc._adc._TwoFluidAP` (retire du coeur) ; la verification d'ABI de
  `native.load_symbols` garantit qu'une lib en cache perimee leve une erreur explicite
  plutot qu'un `AttributeError` opaque.
```
