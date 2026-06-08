# `safe_euler_periodic/` - cas sûr de référence (validation)

Euler compressible **pur**, domaine périodique, **bulle de pression lisse** de faible amplitude :

- `rho ≡ rho0 = 1` (densité uniforme → `rho > 0` garanti) ;
- `v = 0` à `t = 0` ; `p = p0 + dp·exp(-r²/(σ²L²))` avec `p0 = 1`, `dp = 0.1` (→ `p > 0` garanti) ;
- `E = p/(γ-1)`, `γ = 1.4` ; AUCUNE source, AUCUN couplage Poisson (transport pur).

La bulle de pression se détend en ondes acoustiques : dynamique non triviale mais douce, sans choc,
sans perte de positivité. C'est le cas de référence de la campagne de perf (`perf/`), choisi pour
être **sûr** (pas de Poisson physique, pas de Schur, pas de géométrie disque) et donc isoler le coût
des fronts à calcul identique.

## Ce que ce cas valide (CI)

- **Équivalence briques ↔ DSL** : état final **bit-identique** (`np.array_equal`, tolérance 1e-10),
  comme `diocotron_dsl` / `two_species_dsl`. Mêmes réglages : minmod / rusanov / reconstruction
  conservative / SSPRK2 / `dt` fixe.
- **Invariants** : masse conservée (transport, périodique), `rho > 0`, `p > 0`, état fini.
- **Dynamique** : `max|Δp| > 1e-4` (la bulle évolue, le cas n'est pas trivial).

## Source de vérité

Le modèle (briques & DSL), les CI, le `dt` et les réglages vivent dans
[`adc_cases/common/safe_euler.py`](../adc_cases/common/safe_euler.py) - partagés avec
`perf/frontend_compare.py`. Le pendant C++ direct est `adc_cpp/bench/frontend_cpp.cpp` (namespace
`safecase`) : les constantes et le schéma numérique **doivent** y coïncider bit-à-bit.

## Lancement

```bash
PYTHONPATH=<adc_cpp>/build-master/python:. python3 safe_euler_periodic/run.py --n 64 --steps 40
```

Nécessite un compilateur C++20 (`needs = ["cxx"]`) pour la compilation DSL `production`/`aot`. La
**mesure** de performance (3 fronts, temps, figures) n'est PAS ici - voir [`perf/`](../perf).
