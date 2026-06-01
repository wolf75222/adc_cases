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
include/adc/model/      modeles physiques (diocotron, euler, euler_poisson, ...)
include/adc/solver/     facades PIMPL (Diocotron, EulerPoisson, TwoFluidAP, DiocotronAmr)
include/adc/analysis/   diagnostics applicatifs (invariants, taux de croissance, HDF5)
include/adc/integrator/ integrateurs specifiques a un cas (magnetic_euler_poisson, two_fluid_ap)
src/                    instanciation des facades (libadc)
examples/               pilotes de demonstration (diocotron, AMR, MPI, GPU)
python/                 bindings pybind11
tests/                  tests applicatifs (modeles, facades, integration coeur+modele)
scripts/ romeo/ tutorials/   outils, runs HPC, tutoriels
```

## Build

```
cmake -B build                          # tire adc_cpp (../adc_cpp), serie
cmake --build build -j
ctest --test-dir build
```

Backends (propagés au cœur) : `-DADC_USE_OPENMP=ON`, `-DADC_USE_MPI=ON`,
`-DADC_USE_KOKKOS=ON` (GPU), `-DADC_USE_HDF5=ON`. Bindings Python :
`-DADC_CASES_BUILD_PYTHON=ON`. Chemin du cœur surchargeable :
`-DADC_CPP_DIR=/chemin/vers/adc_cpp`.
