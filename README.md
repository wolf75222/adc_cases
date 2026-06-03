# adc_cases : cas d'utilisation Python de la lib `adc`

Ce dépôt contient les **cas d'utilisation** du solveur `adc` (advection-diffusion /
couplage hyperbolique-elliptique). Toute la physique générique et tout le calcul cellule
par cellule vivent dans la bibliothèque **[adc_cpp](../adc_cpp)** et sont exposés par son
module Python `adc` (bindings pybind11). Ici, **Python compose et pilote** : un dossier par
cas, chacun important `adc`, écrivant ses conditions initiales en numpy, et lançant la
simulation. La quasi-totalité des cas est donc du Python pur.

> **Principe** : *Python dit QUOI assembler, le C++ compilé fait le calcul.*

Un cas **sur mesure** peut porter son propre C++ (un scénario qui n'est pas une brique
générique du cœur) : il est alors compilé **à la volée** contre les en-têtes génériques
d'`adc_cpp` et chargé dans le process (`ctypes`). C'est le cas de [`two_fluid_ap/`](two_fluid_ap/)
(solveur asymptotic-preserving). Cela exige un compilateur C++20 ; le dossier `include/`
d'`adc_cpp` est localisé automatiquement depuis le module `adc`, ou via `ADC_INCLUDE`.

## Prérequis : construire le module `adc`

Le module Python est compilé depuis `adc_cpp` :

```bash
cd ../adc_cpp
cmake -B build-py -DADC_BUILD_PYTHON=ON
cmake --build build-py --target _adc -j
```

Puis on met le paquet sur le `PYTHONPATH` (le dossier qui contient le paquet `adc/`) :

```bash
export PYTHONPATH=$PWD/build-py/python      # depuis adc_cpp
```

Lancer les cas avec le **même interpréteur Python** que celui ayant compilé le module
(l'extension porte un suffixe ABI `cpython-3XY`). En cas de Python multiples, pinner à la
configuration : `-DPython_EXECUTABLE=$(which python3.12)`.

(Dépendances des cas : `numpy`, `matplotlib` pour la repro diocotron, et un compilateur C++20 pour `two_fluid_ap` (solveur compilé à la volée).)

## Lancer un cas

```bash
cd ../adc_cases
python3 diocotron/run.py          # reproduction du papier arXiv:2510.11808 (figures + gif)
python3 composition/run.py        # composition hétérogène + intégrateur temporel Python
python3 euler_poisson/run.py      # auto-gravité vs plasma (Langmuir)
python3 multispecies/run.py       # électrons Euler + ions isothermes, Poisson de système
python3 two_euler/run.py          # deux Euler indépendants, même schéma (HLLC + recon primitif)
python3 plasma/run.py             # électrons + ions + neutres : Poisson + ionisation + collision
python3 two_fluid_ap/run.py       # bi-fluide raide asymptotic-preserving
python3 diocotron_amr/run.py      # diocotron sur AMR multi-patch
python3 custom_scheme/run.py      # schéma spatial + temporel écrit en Python, Poisson par adc
```

## Les cas (un dossier par cas)

| Dossier | Cas | Ce qu'il montre |
|---|---|---|
| [`diocotron/`](diocotron/) | Instabilité diocotron (dérive E×B) | **Reproduction de [arXiv:2510.11808](https://arxiv.org/abs/2510.11808)** : taux de croissance analytique (Petri, numpy) vs mesuré, composé génériquement via `adc.System` (briques `ExB` + `BackgroundDensity` + paroi conductrice), figures + gif. Voir [diocotron/README.md](diocotron/README.md). `band_instability.py` : variante périodique minimale. |
| [`composition/`](composition/) | Composition multi-blocs | Électrons (Euler, VanLeer+HLLC, IMEX, 10 sous-pas) + ions (isotherme, Minmod+Rusanov, explicite) ; choix implicite/explicite par bloc **réversible** ; garde-fous ; **intégrateur temporel écrit en Python** (`adc.integrate.ssprk2_step`). |
| [`euler_poisson/`](euler_poisson/) | Euler + champ auto-consistant | Auto-gravité (attractif) vs plasma/Langmuir (répulsif) ; un seul signe de couplage les sépare ; masse et impulsion conservées. |
| [`multispecies/`](multispecies/) | Deux fluides hétérogènes | Électrons Euler (4 var) + ions isothermes (3 var) couplés par **un** Poisson de système `f = Σ q_s n_s` ; masse conservée par espèce. |
| [`two_euler/`](two_euler/) | Deux Euler indépendants | Électrons + ions, **deux gaz d'Euler non couplés**, mêmes briques (`CompressibleFlux` + HLLC + **reconstruction primitive**) ; seules les CI diffèrent (électrons plus légers donc plus rapides) ; multirate `step_adaptive`. Illustre « deux Euler, même code ». |
| [`plasma/`](plasma/) | Plasma couplé (e + i + n) | Trois espèces partageant un Poisson de système (`f = Σ q_s n_s`), couplées par **sources inter-espèces** : ionisation (`add_ionization`, n_g→n_i+n_e) et collision ion-neutre (`add_collision`) ; électrons en HLLC + reconstruction primitive. Conservation n_i+n_g à l'arrondi machine. |
| [`two_fluid_ap/`](two_fluid_ap/) | Bi-fluide raide AP | Intégrateur AP **sur mesure**, non composable bloc à bloc (stabilisation AP couplée au pas de temps dans l'elliptique) : schéma asymptotic-preserving stable quand `dt·ω_pe ≫ 1` (un explicite exploserait). **Scénario**, pas une brique générique : sa physique C++ (`two_fluid_ap.hpp` + `_two_fluid_ap.cpp`) vit ici, compilée à la volée contre les en-têtes génériques d'`adc_cpp` puis pilotée depuis Python (`ctypes`). |
| [`diocotron_amr/`](diocotron_amr/) | Diocotron sur AMR | Composé via `adc.AmrSystem` (pendant raffiné de `System` : `add_block` + `set_refinement`) : hiérarchie de patchs raffinés dynamiquement, reflux conservatif. |
| [`custom_scheme/`](custom_scheme/) | Méthode numérique en Python | Transport diocotron (reconstruction, flux upwind, SSPRK2) **écrit en numpy** ; `adc` ne sert que d'**oracle de Poisson** (`set_density` + `solve_fields` + `potential`). Masse conservée à l'arrondi machine. |

## L'API en deux niveaux

- **Composition générique** (`adc.System`) : on ajoute des **blocs**, chacun avec son
  modèle, son schéma spatial (`adc.Spatial(limiter, flux)`), son traitement temporel
  (`adc.Explicit` / `adc.IMEX` / `adc.Implicit`) et son sous-cyclage ; ils partagent un
  Poisson de système. Idéal pour coupler ions/électrons/neutres. Un modèle est une
  **composition de briques** `adc.Model(state, transport, source, elliptic)` ; les compositions
  nommées (`diocotron`, `electron_euler`, ...) sont définies dans [`models.py`](models.py).
- **Composition sur AMR** (`adc.AmrSystem`) : un bloc porté sur une hiérarchie raffinée
  (même API que `System`, plus `set_refinement`).
- **Scénario sur mesure** (AP deux-fluides) : schéma asymptotic-preserving non composable
  bloc à bloc (la stabilisation AP couple la raideur au pas de temps dans l'elliptique). Ce
  n'est **pas** une brique générique du cœur : sa physique C++ vit dans `two_fluid_ap/`,
  compilée à la volée contre les en-têtes génériques d'`adc_cpp` et chargée via `ctypes`.

Détails de l'API et de l'architecture : [adc_cpp/README.md](../adc_cpp/README.md) et
[adc_cpp/docs/ARCHITECTURE.md](../adc_cpp/docs/ARCHITECTURE.md).
