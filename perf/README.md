# `perf/` — campagne de mesure de performance

Deux axes, mesures uniquement en delta depuis `origin/master` des deux dépôts (`adc_cpp`,
`adc_cases`). Le cas physique est le **cas sûr** : Euler compressible pur, périodique, bulle de
pression lisse de faible amplitude (`rho > 0`, `p > 0` garantis), transport pur. Source de vérité
du cas : [`adc_cases/common/safe_euler.py`](../adc_cases/common/safe_euler.py) ; pendant C++ :
`adc_cpp/bench/frontend_cpp.cpp` (namespace `safecase`). La **validation** du cas (équivalence
des fronts + invariants, sans mesure) est le cas registré [`safe_euler_periodic/`](../safe_euler_periodic).

## Axe 1 — fronts : C++ direct vs Python briques vs Python DSL

`frontend_compare.py` joue la **même** physique avec les **mêmes** réglages numériques
(minmod / rusanov / reconstruction conservative / SSPRK2 / `dt` FIXE) sur trois fronts ; le seul
écart mesuré est le coût du front, à calcul identique.

- **C++ direct** : binaire `adc_cpp/bench/frontend_cpp` (sous-processus).
- **Python briques** : `adc.System` + `add_block(models.euler)` + `step(dt)`.
- **Python DSL** : `adc.dsl.Model(...).compile(backend="production")` + `add_equation` + `step(dt)`.

Méthodologie **cold-cache** : chaque front Python tourne dans un **sous-processus frais** (import
`adc` réellement froid, cache DSL maîtrisé). Le DSL est mesuré **froid** (`so_dir` vide → compilation
`g++`) puis **chaud** (même `so_dir` → cache touché). Chronométrage par étage : `import` /
`model_build` / `dsl_compile` / `addblock` / `state_init` / `first_step` / `warmup` / `run_loop` /
`diag` ; plus la boucle chaude (`median/p10/p90/cv`), `advance(dt,nsteps)` (un appel Python, isole
le crossing par pas) et, si Poisson actif, `solve_fields` isolé.

```bash
# depuis adc_cases, avec le build sur le PYTHONPATH
PYTHONPATH=<adc_cpp>/build-master/python:. python3 perf/frontend_compare.py \
    --n 256 --steps 50 --warmup 5 --poisson off \
    --cpp-bin <adc_cpp>/build-bench-serie/bin/frontend_cpp
python3 perf/plot_frontend.py   # figures dans out/safe_euler_periodic/figures/
```

`--poisson off` (défaut) = transport pur, signal frontend propre. `--poisson on` = solve elliptique
**inerte** (charge=0) à chaque pas, régime MG-dominé (idiome `two_euler`). Les deux modes restent
symétriques sur les trois fronts.

**Asymétrie de granularité (assumée).** `System` n'a aucun timer interne : le front C++ donne le
détail 7-phases (poisson/aux/halos/transport/réduction/fence/alloc, via la machinerie de
`profile_step`), les fronts Python ne donnent que `total + solve_fields`. La comparaison croisée
reste valide sur le **temps total cold-cache** et le **hot ms/pas**.

## Axe 2 — scaling CPU/GPU/MPI

Piloté côté `adc_cpp/bench/scaling_step.cpp` via `bench/run_scaling.sh` (multi-box, vrais halos
MPI). Charges : `transport` (4096²), `poisson` (1024²), `amr` (non câblé dans ce binaire → ligne de
diagnostic explicite). Le JSONL produit (un par point du balayage) est tracé par
`plot_frontend.py --scaling <fichier.jsonl>` (strong speedup/efficacité, weak efficacité, débit).

## Local vs cluster

- **Mac (série)** : validation/plomberie SEULEMENT — vérifie le câblage des API, la compilation DSL
  `production`, l'identité numérique, les invariants, le schéma JSONL, les figures. Les temps série
  sont étiquetés `machine=<mac>, backend=serial` et **exclus** de toute affirmation de scaling.
- **ROMEO (GH200/MPI/OpenMP)** : SEULE source de chiffres de scaling valides (CV<5%, cells/s,
  p10/p90, ratios). `bench/run_frontend.sh` et `bench/run_scaling.sh` portent les recettes de build
  par backend.

## Schéma JSONL (`adc_perf_v1`)

Chaque ligne porte : `adc_cpp_sha`/`adc_cpp_branch`/`adc_cases_sha`/`adc_cases_branch`, `backend`,
`machine`, `ranks`/`threads`/`gpus`, `nx`/`ny`/`boxes`/`max_grid`, `workload` + réglages numériques,
`stages{...}`, `total_cold_user_s`, `hot_ms_per_step{median,p10,p90,cv}`, `advance_ms_per_step`,
`phases_ms_per_step{...}`, `cells_per_s`, `invariants{mass,rho_min,p_min,nan}`.

## Acceptation

Un résultat n'est publiable que si : invariants OK, pas de NaN, `cv < 5 %`, réglages numériques
identiques entre fronts, SHA exacts dans le JSON, et **aucun graphe ne mélange master et PR**
(`plot_frontend.py` refuse de tracer deux `(adc_cpp_sha, adc_cpp_branch)` dans une même figure).
