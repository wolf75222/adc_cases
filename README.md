# adc_cases : cas d'utilisation Python de la lib `adc`

Ce dépôt ne contient **que du Python** : des **cas d'utilisation** du solveur
`adc` (advection-diffusion / couplage hyperbolique-elliptique). Toute la physique
et tout le calcul cellule par cellule vivent dans la bibliothèque **[adc_cpp](../adc_cpp)**
et sont exposés par son module Python `adc` (bindings pybind11). Ici, **Python ne fait
que composer et piloter** : un dossier par cas, chacun important `adc`, écrivant ses
conditions initiales en numpy, et lançant la simulation.

> **Principe** : *Python dit QUOI assembler, le C++ compilé fait le calcul.* Aucun
> binding ni code C++ dans ce dépôt : la lib et ses bindings sont dans `adc_cpp`.

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

(Dépendances Python des cas : `numpy`, et `matplotlib` pour la repro diocotron.)

## Lancer un cas

```bash
cd ../adc_cases
python3 diocotron/run.py          # reproduction du papier arXiv:2510.11808 (figures + gif)
python3 composition/run.py        # composition hétérogène + intégrateur temporel Python
python3 euler_poisson/run.py      # auto-gravité vs plasma (Langmuir)
python3 multispecies/run.py       # électrons Euler + ions isothermes, Poisson de système
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
| [`two_fluid_ap/`](two_fluid_ap/) | Bi-fluide raide AP | Solveur **spécialisé** `adc.TwoFluidAP` : schéma asymptotic-preserving stable quand `dt·ω_pe ≫ 1` (un explicite exploserait). |
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
- **Solveur spécialisé** (`adc.TwoFluidAP`) : intégrateur asymptotic-preserving sur mesure,
  exposé comme façade, non composable bloc à bloc.

Détails de l'API et de l'architecture : [adc_cpp/README.md](../adc_cpp/README.md) et
[adc_cpp/docs/ARCHITECTURE.md](../adc_cpp/docs/ARCHITECTURE.md).
