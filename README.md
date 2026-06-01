# adc_cases

Applications et cas physiques du solveur **adc**. Ce dépôt contient les modèles
(diocotron, Euler-Poisson, two-fluid), les façades compilées (`libadc`), les
bindings Python, les exemples, scripts et tutoriels. Le cœur générique (concepts
`PhysicalModel`/`EllipticSolver`, coupleur, elliptique, AMR, seams CPU/OpenMP/MPI/GPU)
vit dans le dépôt séparé [`adc_cpp`](../adc_cpp), tiré ici via FetchContent.

## Découpage

| Dépôt | Rôle |
|---|---|
| `adc_cpp` | cœur de la bibliothèque (générique, templates, parallélisme, AMR) |
| `adc_cases` | applications : modèles physiques, façades, exemples, Python |

```
include/adc/model/      modeles physiques (diocotron, euler, euler_poisson, charged_fluid, ...)
include/adc/solver/     facades PIMPL (Diocotron, EulerPoisson, TwoFluidAP, DiocotronAmr,
                        MultiSpecies, Simulation)
include/adc/analysis/   diagnostics applicatifs (invariants, taux de croissance, HDF5)
include/adc/integrator/ integrateurs specifiques a un cas (magnetic_euler_poisson, two_fluid_ap)
src/                    instanciation des facades (libadc)
examples/               pilotes de demonstration (diocotron, AMR, MPI, multispecies)
python/                 bindings pybind11 + demos (python/demos/) + test_bindings.py
tests/                  tests applicatifs (modeles, facades, integration coeur+modele)
scripts/ romeo/ tutorials/   outils, runs HPC, tutoriels
```

## Build

```
cmake -B build                          # tire adc_cpp (../adc_cpp), serie
cmake --build build -j
ctest --test-dir build
```

Backends (propagés au cœur) : `-DADC_USE_KOKKOS=ON` (**recommandé** — CPU OpenMP + GPU),
`-DADC_USE_MPI=ON`, `-DADC_USE_HDF5=ON` ; `-DADC_USE_OPENMP=ON` est **déprécié** (Kokkos le
couvre). Bindings Python : `-DADC_CASES_BUILD_PYTHON=ON`. Chemin du cœur surchargeable :
`-DADC_CPP_DIR=/chemin/vers/adc_cpp`.

## Multi-espèces et composition depuis Python

Le cœur sait coupler N espèces (ions, électrons, neutres…), chacune avec son modèle, son
schéma spatial et sa politique temporelle ; elles interagissent dans le second membre de
Poisson (`f = Σ_s q_s n_s`) et dans la source. Deux façades exposent ça :

- **`MultiSpeciesSolver`** (compilée) : système deux fluides figé (électrons Euler + ions
  isothermes + Poisson de système).
- **`Simulation`** (composition à l'exécution) : `add_species(name, model, charge)` ajoute
  des espèces à la volée (`"diocotron"`, `"electron_euler"`, `"ion_isothermal"`), partageant
  un même Poisson. La physique reste en C++ compilé — aucun callback Python dans le hot path.

```python
import adc
sim = adc.Simulation(adc.SimulationConfig())
sim.add_species("electrons", "electron_euler", -1.0)
sim.add_species("ions",      "ion_isothermal", +1.0)
sim.set_density("electrons", ne); sim.set_density("ions", ni)
sim.advance(1e-3, 100)          # masse conservée par espèce, Poisson de système
```

**Démos Python** (`python/demos/`) : un script par capacité (diocotron, AMR, Euler-Poisson
gravité/plasma, two-fluide AP raide/magnétisé, multi-espèces composé), pilotant la façade
depuis Python et produisant diagnostics + sorties. Lancer : `python python/demos/<nom>.py`
(avec le module `adc` sur le `PYTHONPATH`, cf. `python/README` ou le build `-DADC_CASES_BUILD_PYTHON=ON`).
