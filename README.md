# adc_cases : cas d'utilisation Python de la lib `adc`

Ce dépôt contient les cas d'utilisation du solveur `adc` (advection-diffusion /
couplage hyperbolique-elliptique). Toute la physique générique et tout le calcul cellule
par cellule vivent dans la bibliothèque [adc_cpp](../adc_cpp) et sont exposés par son
module Python `adc` (bindings pybind11). Ici, Python compose et pilote : un dossier par
cas, chacun important `adc`, écrivant ses conditions initiales en numpy, et lançant la
simulation. La quasi-totalité des cas est donc du Python pur.

> **Principe** : *Python dit quoi assembler, le C++ compilé fait le calcul.*

Un cas sur mesure peut porter son propre C++ (un scénario qui n'est pas une brique
générique du cœur) : il est alors compilé à la volée contre les en-têtes génériques
d'`adc_cpp` et chargé dans le process (`ctypes`). C'est le cas de [`two_fluid_ap/`](two_fluid_ap/)
(solveur asymptotic-preserving). Cela exige un compilateur C++20 ; le dossier `include/`
d'`adc_cpp` est localisé automatiquement depuis le module `adc`, ou via `ADC_INCLUDE`. La
mécanique commune (localisation des en-têtes, compilation, chargement) vit dans
`adc_cases.common.native` : la bibliothèque compilée va dans `out/<cas>/build/` (jamais à côté
du `.cpp`), elle est recompilée dès que sa clé d'ABI change (compilateur, flags, sources,
en-têtes du cœur) et toute incompatibilité d'ABI (symbole attendu absent) lève une erreur
explicite au chargement plutôt qu'une panne opaque au premier appel.

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

Lancer les cas avec le même interpréteur Python que celui ayant compilé le module
(l'extension porte un suffixe ABI `cpython-3XY`). En cas de Python multiples, pinner à la
configuration : `-DPython_EXECUTABLE=$(which python3.12)`.

## Le paquet `adc_cases`

Ce dépôt est lui-même un paquet Python importable `adc_cases` qui centralise ce qui est
commun aux cas (modèles nommés, conditions initiales, grilles, invariants, sorties) :

| Module | Contenu |
|---|---|
| `adc_cases.models` | modèles d'espèce nommés = compositions de briques `adc` (electron_euler, ion_isothermal, diocotron, euler_poisson, euler, neutral_isothermal). Un modèle = une espèce (`adc.Model`). |
| `adc_cases.recipes` | recettes système = configurations multi-espèces prêtes à l'emploi (`two_fluid`, `plasma` : blocs + Poisson + couplages). Un niveau au-dessus des modèles d'espèce. |
| `adc_cases.common.grid` | grilles à centres de cellules (`meshgrid_xy`), convention `field[j, i]` de la façade `adc`. |
| `adc_cases.common.initial_conditions` | CI réutilisées : bande gaussienne (`band_density`), anneau (`ring_density`), bulle de pression Euler (`euler_pressure_blob`). |
| `adc_cases.common.checks` | invariants vérifiés par plusieurs cas (`assert_mass_conserved`, `assert_finite`, `assert_positive`, `relative_drift`). |
| `adc_cases.common.io` | répertoire de sortie `out/` (hors source, ignoré par git). |
| `adc_cases.common.native` | compilation à la volée + chargement `ctypes` des scénarios C++ sur mesure (cache hors source dans `out/`, contrôle d'ABI explicite). |

Installer le paquet en editable une fois (les cas font alors `import adc_cases` sans toucher
à `sys.path`) :

```bash
cd adc_cases
pip install -e .                 # numpy tiré automatiquement
pip install -e '.[figures]'      # + matplotlib (repro diocotron)
```

Sans installation, chaque cas reste lançable directement (`python3 diocotron/run.py`) : un
court préambule tente `import adc_cases` et, seulement s'il échoue (paquet non installé), met
la racine du dépôt sur le chemin d'import. Une fois le paquet installé, ce préambule ne touche
plus du tout à `sys.path` : les cas s'appuient sur le paquet importable, pas sur un bricolage
de chemin.

(Dépendances des cas : `numpy` ; `matplotlib` pour la repro diocotron ; un compilateur C++20
pour `two_fluid_ap` (solveur compilé à la volée).)

## Sorties

Les cas qui produisent des fichiers éphémères (figures de travail, gif, `.so`) écrivent sous
`out/<cas>/` à la racine du dépôt, pas dans leur dossier source. Idem pour les artefacts de
compilation à la volée (`two_fluid_ap`), placés sous `out/<cas>/build/`. `out/` est ignoré par git.
On peut surcharger la racine via `ADC_CASES_OUT=<chemin>`.

**Exception, figures canoniques versionnées.** `diocotron/run.py` écrit ses figures directement
dans `diocotron/figures/` (tracké) et y dépose un `provenance.json` (SHA `adc_cpp`/`adc_cases`,
backend, résolution, commande, taux mesurés) : une re-exécution les rafraîchit en place. La
provenance et le statut de chaque asset (committé vs éphémère) sont décrits dans
[`ASSETS.md`](ASSETS.md). Les figures de `hoffart_euler_poisson_dsl` restent éphémères et non
committées (cas `reproduction-candidate` PENDING : ne pas les présenter comme une reproduction).

## Manifeste des cas et CI

[`cases_manifest.toml`](cases_manifest.toml) classe chaque cas par catégorie (`validation`,
`tutoriel`, `reproduction`, `reproduction-candidate`, `experimental`) et indique s'il tourne en
CI (`ci = true`). La CI ne lance que les cas légers (`ci = true`) ; les cas longs
(reproduction `diocotron/run.py` avec figures/gif), les reproductions-candidates non encore
établies (`hoffart_euler_poisson_dsl/run.py`, table PENDING) et les expérimentaux (`dsl_euler`
DSL interprété, `schur_magnetized_cartesian`) restent hors CI et se lancent à la main.

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
python3 dsl_euler/run.py          # Euler écrit en formules (mini-DSL adc.dsl, expérimental)

# Tutoriel : le même diocotron écrit en helper / briques / formules (équivalence prouvée) + figures
python3 tutorial/run.py                   # tutoriel complet (3 fronts équivalents) + figures + gif
python3 tutorial/equivalence.py           # smoke CI : helper == briques == formules, bit-identique

# Cas écrits en formules via adc.dsl (génération + compilation C++ : compilateur C++20 requis)
python3 diocotron_dsl/run.py              # diocotron en formules, prouvé bit-identique au natif
python3 two_species_dsl/run.py            # électrons + ions en formules, Poisson couplé
python3 magnetic_isothermal_dsl/run.py    # fluide isotherme magnétisé (Lorentz via B_z) en formules
python3 schur_magnetized_cartesian/run.py # étage source Schur vs explicite (mesure, expérimental)
python3 hoffart_euler_poisson_dsl/run.py  # Euler-Poisson magnétisé (Hoffart) : reproduction-candidate PENDING
```

## Les cas (un dossier par cas)

Chaque dossier de cas dispose désormais d'un `README.md` rédigé selon le gabarit
commun (objectif, équations, modèle, méthode, architecture, carte des fichiers,
commande exacte, conditions initiales, invariants, sorties, coût, limites, CI). La table
ci-dessous couvre les 16 cas ; la colonne *Catégorie* est celle du
[manifeste](cases_manifest.toml) (`validation` / `tutoriel` / `reproduction` /
`reproduction-candidate` / `experimental`). Les descriptions sont honnêtes : un cas
`reproduction-candidate` n'est pas une reproduction établie, un cas `experimental` est
un prototype non finalisé.

| Dossier | Catégorie | Cas | Ce qu'il montre |
|---|---|---|---|
| [`diocotron/`](diocotron/README.md) | reproduction | Instabilité diocotron (dérive E×B) | Reproduction de [arXiv:2510.11808](https://arxiv.org/abs/2510.11808) : taux de croissance analytique (Petri, numpy) vs mesuré, composé génériquement via `adc.System` (briques `ExB` + `BackgroundDensity` + paroi conductrice), figures + gif ; long, hors CI. Le sous-script `band_instability.py` est une variante périodique minimale (croissance de l'instabilité, sans figures) classée `validation` (en CI). |
| [`tutorial/`](tutorial/README.md) | tutoriel | Le diocotron écrit de 3 façons | La même physique diocotron construite via le helper `models.diocotron`, via les briques natives reconstruites à la main, et via les formules `adc.dsl.Model` : les trois états finals sont bit-identiques (`np.array_equal`). Mirroir côté `adc_cases` du tutoriel Sphinx d'adc_cpp ; figures (croissance + gif + briques vs DSL). La variante CI `tutorial/equivalence.py` (`validation`, `cxx`) verrouille l'équivalence sans figures. |
| [`composition/`](composition/README.md) | tutoriel | Composition multi-blocs | Électrons (Euler, VanLeer+HLLC, IMEX, 10 sous-pas) + ions (isotherme, Minmod+Rusanov, explicite) ; choix implicite/explicite par bloc réversible ; garde-fous ; intégrateur temporel écrit en Python (`adc.integrate.ssprk2_step`). |
| [`euler_poisson/`](euler_poisson/README.md) | validation | Euler + champ auto-consistant | Auto-gravité (attractif) vs plasma/Langmuir (répulsif) ; un seul signe de couplage les sépare ; masse et impulsion conservées. |
| [`multispecies/`](multispecies/README.md) | validation | Deux fluides hétérogènes | Électrons Euler (4 var) + ions isothermes (3 var) couplés par un Poisson de système `f = Σ q_s n_s` ; masse conservée par espèce. |
| [`two_euler/`](two_euler/README.md) | validation | Deux Euler indépendants | Électrons + ions, deux gaz d'Euler non couplés, mêmes briques (`CompressibleFlux` + HLLC + reconstruction primitive) ; seules les CI diffèrent (électrons plus légers donc plus rapides) ; multirate `step_adaptive`. Illustre « deux Euler, même code ». |
| [`plasma/`](plasma/README.md) | validation | Plasma couplé (e + i + n) | Trois espèces partageant un Poisson de système (`f = Σ q_s n_s`), couplées par sources inter-espèces : ionisation (`add_ionization`, n_g->n_i+n_e) et collision ion-neutre (`add_collision`) ; électrons en HLLC + reconstruction primitive. Conservation n_i+n_g à l'arrondi machine. |
| [`two_fluid_ap/`](two_fluid_ap/README.md) | validation (needs `cxx`) | Bi-fluide raide AP | Intégrateur AP sur mesure, non composable bloc à bloc (stabilisation AP couplée au pas de temps dans l'elliptique) : schéma asymptotic-preserving stable quand `dt·ω_pe ≫ 1` (un explicite exploserait). Scénario, pas une brique générique : sa physique C++ (`two_fluid_ap.hpp` + `_two_fluid_ap.cpp`) vit ici, compilée à la volée contre les en-têtes génériques d'`adc_cpp` puis pilotée depuis Python (`ctypes`). |
| [`diocotron_amr/`](diocotron_amr/README.md) | validation | Diocotron sur AMR | Composé via `adc.AmrSystem` (pendant raffiné de `System` : `add_block` + `set_refinement`) : hiérarchie de patchs raffinés dynamiquement, reflux conservatif. |
| [`custom_scheme/`](custom_scheme/README.md) | tutoriel | Méthode numérique en Python | Transport diocotron (reconstruction, flux upwind, SSPRK2) écrit en numpy ; `adc` ne sert que d'oracle de Poisson (`set_density` + `solve_fields` + `potential`). Masse conservée à l'arrondi machine. |
| [`diocotron_dsl/`](diocotron_dsl/README.md) | validation (needs `cxx`) | Diocotron écrit en formules (DSL) | La physique diocotron (transport E×B + fond neutralisant) écrite entièrement en formules `adc.dsl.Model` au lieu de briques natives ; `adc.dsl` génère le C++, le compile et l'installe via `add_equation`. Coeur du cas : l'état produit est bit-identique à la composition native (`np.array_equal`), sur même grille / CI / Poisson. Backend `production` (natif zéro-copie) sinon `aot` (host-marshalé, numérique identique). |
| [`two_species_dsl/`](two_species_dsl/README.md) | validation (needs `cxx`) | Électrons + ions en formules (DSL) | Deux espèces en formules (`adc.dsl.Model`) : électrons Euler (4 var) + ions isothermes (3 var), chacune avec source électrostatique (lit grad φ via le canal aux) et densité de charge, couplées par un même Poisson. Équivalence au natif par espèce : ions bit-identiques, électrons à ε-machine (réassociation flottante dans l'accumulation du RHS de Poisson partagé, tolérance 1e-24). |
| [`magnetic_isothermal_dsl/`](magnetic_isothermal_dsl/README.md) | validation (needs `cxx`) | Fluide isotherme magnétisé en formules (DSL) | Fluide isotherme magnétisé en formules avec force de Lorentz `q ρ E + v×B` pilotée par un champ B_z constant lu sur le canal `aux` étendu (indice 3, peuplé depuis Python via `set_magnetic_field`). Aucun modèle natif de référence : correction prouvée par parité inter-backend (production == aot quand les deux se lient) + oracle Lorentz numpy + invariants (masse, positivité, rotation de la quantité de mouvement). |
| [`schur_magnetized_cartesian/`](schur_magnetized_cartesian/README.md) | experimental (needs `cxx`) | Étage source Schur vs explicite | Mesure de l'effet temporel de l'étage source condensé par Schur (`adc.Split(Explicit, CondensedSchur)`) face à l'intégration explicite de la même source de Lorentz raide, sur un fluide isotherme magnétisé cartésien (même DSL que `magnetic_isothermal_dsl`). Balaie le plus grand `dt` stable explicite vs Schur (θ=0.5 / θ=1.0), reporte `dt·ω_c` et le gain. Prototype de mesure, hors CI. |
| [`dsl_euler/`](dsl_euler/README.md) | experimental | Euler en formules (DSL interprété) | Euler compressible 2D en formules (mini-DSL `adc.dsl`), version prototype interprétée CPU : l'arbre symbolique est évalué en numpy et branché sur le backend hôte `adc.PythonFlux` (Rusanov, périodique). Démonstrateur déclaratif côté utilisateur, pas le chemin de production (qui reste les briques compilées). Vérifie masse conservée, dynamique acoustique non triviale, état physique. |
| [`hoffart_euler_poisson_dsl/`](hoffart_euler_poisson_dsl/README.md) | **reproduction-candidate** PENDING | Euler-Poisson magnétisé (Hoffart), étage Schur | Vise [arXiv:2510.11808](https://arxiv.org/abs/2510.11808) (système Euler-Poisson magnétisé complet, étage source Schur) écrit en `adc.dsl`. La reproduction quantitative n'est pas encore établie (table de validation PENDING) : baseline cartésienne loin du papier, géométrie suspecte (cf. `adc_cpp/docs/HOFFART_FIDELITY.md`). Hors CI. Le sous-script `check_model.py` (`validation`, en CI) est un oracle analytique du modèle : flux, source Lorentz/électrique, valeurs propres et RHS de Poisson vérifiés par assert. |

## L'API en deux niveaux

- **Composition générique** (`adc.System`) : on ajoute des blocs, chacun avec son
  modèle, son schéma spatial (`adc.Spatial(limiter, flux)`), son traitement temporel
  (`adc.Explicit` / `adc.IMEX` / `adc.Implicit`) et son sous-cyclage ; ils partagent un
  Poisson de système. Idéal pour coupler ions/électrons/neutres. Un modèle est une
  composition de briques `adc.Model(state, transport, source, elliptic)` ; les compositions
  nommées (`diocotron`, `electron_euler`, ...) sont définies dans [`models.py`](models.py).
- **Composition sur AMR** (`adc.AmrSystem`) : un bloc porté sur une hiérarchie raffinée
  (même API que `System`, plus `set_refinement`).
- **Scénario sur mesure** (AP deux-fluides) : schéma asymptotic-preserving non composable
  bloc à bloc (la stabilisation AP couple la raideur au pas de temps dans l'elliptique). Ce
  n'est pas une brique générique du cœur : sa physique C++ vit dans `two_fluid_ap/`,
  compilée à la volée contre les en-têtes génériques d'`adc_cpp` et chargée via `ctypes`.

Détails de l'API et de l'architecture : [adc_cpp/README.md](../adc_cpp/README.md) et
[adc_cpp/docs/ARCHITECTURE.md](../adc_cpp/docs/ARCHITECTURE.md).
