# Campagne de performance adc_cpp / adc_cases — Rapport

**Sans Hoffart.** Deux axes : (1) coût des fronts C++ vs Python briques vs Python DSL sur un cas
sûr, (2) scaling CPU/GPU/MPI sur cas synthétiques contrôlés. Cas physique = **Euler compressible
pur, périodique, bulle de pression lisse** (`rho>0`, `p>0` garantis, transport pur, pas de Poisson
physique, pas de Schur, pas de géométrie disque).

## Provenance (SHA exacts)

| Donnée | Dépôt | SHA | Machine |
|---|---|---|---|
| Frontend (3 fronts) | adc_cpp `feat/perf-campaign-bench` | `5e17c7a` | ROMEO x64cpu (nœud propre, série) |
| Scaling CPU + GPU + AMR | adc_cpp `feat/perf-campaign-bench` | `0162d5f` | ROMEO x64cpu / GH200 |
| Harnais Python | adc_cases `feat/perf-campaign-harness` | `80c3049` | — |

Le commit `0162d5f` (durcissement JSONL + AMR) **ne touche pas** le chemin frontend
(`frontend_cpp`, `frontend_compare`, `safe_euler`) : les chiffres frontend `5e17c7a` valent pour
`0162d5f`. PR : **adc_cpp #246**, **adc_cases #33**. Aucune figure ne mélange deux SHA (garde-fou
`plot_frontend.py`).

---

## Axe 1 — Fronts C++ / Python briques / Python DSL

Cas sûr, `n=256`, 50 pas, **dt fixe**, mêmes réglages numériques sur les trois fronts
(minmod / rusanov / reconstruction conservative / SSPRK2). Nœud x86 propre, **CV < 1.2 %**.

| Front | hot ms/pas | ratio vs C++ | `advance` ms/pas | cold-cache total | backend DSL |
|---|---|---|---|---|---|
| **C++ direct** | 21.0 | 1.00× | — | 1.18 s | — |
| **Python briques** | 20.2 | **0.96×** | 20.2 | 1.18 s | — |
| **Python DSL (froid)** | 31.1 | **1.48×** | 31.1 | **17.4 s** | production |
| **Python DSL (chaud)** | 31.1 | 1.48× | 31.0 | 1.77 s | production |

Identité numérique briques ↔ DSL : `max|Δ| = 8.9e-16` (epsilon machine). Masse conservée à
`2e-15`, `rho>0`, `p>0` sur les trois fronts. `poisson=on` quasi identique (le solve inerte
charge=0 ajoute peu à `n=256`).

**Où part le temps, et pourquoi :**

1. **Python briques ne coûte rien dans le hot path** (0.96×, dans le bruit). `step(dt)` en boucle
   Python ≈ `advance(dt, nsteps)` en un appel ≈ le même noyau C++. → *Python n'est pas sur le
   chemin chaud* ; l'orchestration par pas (un appel pybind par `step`) est négligeable devant le
   calcul FV.
2. **Python DSL était ~1.5× plus lent en hot loop — cause trouvée et CORRIGÉE.** Ce n'est pas un
   coût de crossing (`advance` ≈ `step` = 31 ms) ni le codegen (le `.so` est en parité stricte,
   flux inliné dans `assemble_rhs`) : c'était les **flags de compilation du `.so`**. `compile_native`
   compilait en **`-O2` sans `-DNDEBUG`** (asserts vivants dans le hot-loop + vectorisation faible),
   alors que le natif est `-O3 -DNDEBUG` (CMake Release). Expérience contrôlée (ROMEO x64cpu, nœud
   propre, CV<1%, job 648019) :

   | flags du `.so` production | hot ms/pas | ratio vs C++ |
   |---|---|---|
   | `-O2` (avant) | 31.0 | 1.48× |
   | `-O3 -DNDEBUG` (corrigé, **adc_cpp PR #253**) | 21.8 | **1.04× (parité)** |
   | `-O3 -DNDEBUG -march=native -funroll-loops` | 18.5 | **0.88× (bat le natif)** |

   → `-O3 -DNDEBUG` ramène le DSL à **parité** avec la brique (dans le bruit de briques=0.96×).
   Et comme le `.so` est **JIT-compilé sur la machine cible**, `-march=native` (opt-in via
   `$ADC_DSL_OPTFLAGS`) lui donne l'AVX-512/NEON que le binaire natif générique n'a pas → le DSL
   **dépasse** la brique (0.88×). Le codegen lui-même n'avait rien à corriger.

   **Régime THREADÉ (Kokkos OpenMP) — 2ᵉ cause, plus profonde (diagnostic croisé Codex).** Sur un
   module `_adc` Kokkos-OpenMP, le DSL warm restait **341 ms INVARIANT** à threads=1/4/8 (ratio qui
   se dégradait 1.17→1.43→1.92) : `compile_native` ne propageait **pas** `-DADC_HAS_KOKKOS`/`-fopenmp`
   → les templates header-only du loader s'instanciaient sur le **fallback série**, le bloc DSL ne
   scalait pas. Fix (PR #253) : compiler le `.so` avec les en-têtes Kokkos + `-fopenmp` quand
   `ADC_KOKKOS_ROOT` est défini, **sans linker libkokkos** (sinon 2ᵉ runtime Kokkos → `SIGABRT` à la
   finalize) — les symboles se résolvent depuis le module déjà chargé (runtime unique). Validation
   ROMEO (module Kokkos-OpenMP, n=256, EXIT=0) :

   | OMP threads | briques ms/pas | DSL ms/pas | ratio |
   |---|---|---|---|
   | 1 | 24.3 | 27.3 | 1.12 |
   | 4 | 19.3 | 19.9 | 1.03 |
   | 8 | 10.5 | 10.7 | **1.02** |

   → le DSL **scale désormais avec les threads** et suit les briques (ratio → 1.02), au lieu des
   341 ms plats. Sans `ADC_KOKKOS_ROOT`, comportement historique (série) inchangé.
3. **Le coût propre du DSL est la compilation** : `dsl_compile` froid = ~15 s (sous-processus
   `g++`), entièrement amorti par le cache hors source (`adc_cache_dir`, clé `model_hash+abi_key`)
   → warm = 1.8 s (≈ briques). L'`import adc` (chargement de `_adc`) domine le cold-cache des
   fronts non-DSL (~1.2 s).

Figures : `frontend_cold_stages.png` (étages cold), `frontend_hot_ms.png` (hot ± p10/p90),
`frontend_step_vs_advance.png`, `frontend_ratio.png`.

---

## Axe 2 — Scaling CPU / GPU / MPI

### CPU OpenMP — strong (transport 4096², poisson 1024², CV < 1 %)

| threads | transport ms/pas | speedup | poisson ms/pas |
|---|---|---|---|
| 1  | 5239 | 1.00× | 71.2 |
| 2  | 5107 | 1.03× | **319.5** |
| 4  | 2736 | 1.91× | 214.0 |
| 8  | 1483 | 3.53× | 121.3 |
| 16 | 913  | **5.74×** (eff 36 %) | **82.4** |

- **Transport monte à 5.74× sur 16 threads** (efficacité 36 %) : noyau FV *memory-bound*, sous-
  linéaire attendu. Décomposition à 16t : transport 836 ms (dominant), alloc_tmp 57 ms, diag 38 ms,
  halos 13 ms, réduction 5 ms.
- **Poisson (V-cycle GeometricMG) est le MUR de scaling** : 16 threads (82 ms) est *plus lent* que
  1 thread (71 ms) — anti-scaling. Le V-cycle est latence/synchro-bound, les niveaux grossiers
  sérialisent. Le pic à 2 threads (319 ms, reproductible sur 2 jobs) est un effet de placement NUMA
  (`OMP_PROC_BIND=spread` sur 2 domaines). C'est une propriété du solveur, pas du harnais.

### CPU — weak scaling (transport 512²/u, poisson 256²/u)

| unités (threads) | transport ms/pas | poisson ms/pas |
|---|---|---|
| 1  (512² / 256²)  | 80.8 | 4.2 |
| 4  (1024² / 512²) | 194  | 54  |
| 16 (2048² / 1024²)| 252  | 80  |

Efficacité weak (temps constant idéal) : transport ≈ 32 %, **poisson ≈ 5 %** — confirme que le MG
ne tire aucun bénéfice du parallélisme CPU.

### GPU GH200 — mono-GPU (Kokkos CUDA)

| workload | taille | ms/pas | cells/s |
|---|---|---|---|
| transport | 1024² | 30.1 | 3.48e7 |
| transport | 2048² | 124  | 3.39e7 |
| transport | 4096² | 497  | 3.37e7 |
| poisson   | 512²  | 6.3  | 4.13e7 |
| poisson   | 1024² | 19.5 | 5.37e7 |

Débit transport **plat ~3.4e7 cells/s** (GPU saturé dès 1024²). **GH200 vs 16 threads x86** : à
taille égale, transport 4096² → **1.84×** (497 vs 913 ms), poisson 1024² → **4.2×** (19.5 vs
82 ms). Modeste sur le transport : noyau *bandwidth-bound*, un socket 16 threads n'est pas loin du
HBM3 du GH200 à ces tailles. (CV poisson GPU ~10 % aux petites tailles — à muscler par plus de
pas/tailles.)

### GPU — AMR synthétique multi-GPU (4 bulles, np = 1/2/4, masse bit-identique cross-rang)

| n | np=1 (répliqué) | np=2 (réparti) | np=4 (réparti) | dérive de masse |
|---|---|---|---|---|
| 128 | 215 ms | 1013 ms | 1390 ms | 0 / 2e-16 |
| 256 | 233 ms | 1080 ms | 1485 ms | 0 |

Le harnais AMR (chemin `AmrSystem` compilé via `add_compiled_model`) **tourne correctement en
multi-GPU** : masse bit-identique cross-rang (`drift = 0`). Mais il **anti-scale à ces tailles** :
le grossier réparti (MG distribué + halos inter-GPU) domine un calcul minuscule (`n=128/256` = 4 à
16 patchs sur 4 GH200 quasi inactifs). → *le problème AMR est trop petit pour bénéficier du
multi-GPU* ; un cas représentatif demanderait `n≥1024`. C'est le verrou de scaling AMR à ces
tailles (comm-bound), pas un défaut de correction.

Figures : `scaling_strong_transport_kokkos-omp.png`, `scaling_strong_poisson_kokkos-omp.png`,
`scaling_weak_transport_kokkos-omp.png`, `scaling_weak_poisson_kokkos-omp.png`,
`scaling_strong_amr_kokkos-cuda.png`.

### MPI multi-rang — DÉBLOQUÉ (deux bugs, corrigés)

Le **MPI×OpenMP hybride** et le **multi-GPU transport/poisson** deadlockaient à np≥2. Deux bugs
superposés, tous deux corrigés :
1. **GPU CUDA-IPC** (`fill_boundary` donnait des buffers UVM à OpenMPI → BTL smcuda routait en
   CUDA-IPC, impossible entre GPU cgroup-isolés) → corrigé par buffers `SharedHostPinnedSpace`
   (**adc_cpp #254**, sur master).
2. **Collective MPI sous `if(rank0)`** dans le bench lui-même : les agrégations `rmax() =
   all_reduce_max()` étaient évaluées dans le `printf` du rang 0 → le rang 0 faisait ~6
   `MPI_Allreduce` de plus que les autres → deadlock. Localisé par checkpoints rang-taggés
   (`fill_boundary` se termine ; le hang est dans l'émission JSON). Corrigé en hissant les `rmax`
   hors du bloc rang 0 (**adc_cpp #258**). `scaling_amr` était déjà correct → marchait en multi-GPU.

**Multi-GPU GH200 — transport 4096² strong scaling** (np = 1 GPU/rang, après fix) :

| GPUs | ms/pas | cells/s | speedup | efficacité |
|---|---|---|---|---|
| 1 | 490 | 3.4e7 | 1.00× | 100 % |
| 2 | 282 | 5.9e7 | 1.73× | 87 % |
| 4 | **141** | **1.19e8** | **3.47×** | **87 %** |

→ **excellent strong scaling multi-GPU** (87 % d'efficacité à 4 GH200). Poisson 1024² anti-scale en
multi-GPU (trop petit, MG comm-bound). **CPU hybride MPI×OpenMP** (transport 4096²) : 1×16=839 ms,
2×8=812 ms, 4×4=465 ms (meilleure localité NUMA à 4 rangs).

---

## Tests et acceptation

| Critère | État |
|---|---|
| Invariants OK (masse conservée, `rho>0`, `p>0`) | ✅ partout |
| Pas de NaN | ✅ (`nan=false` sur tous les runs) |
| CV < 5 % | ✅ CPU (transport/poisson/weak < 1 %), GPU transport (< 1 %), frontend (< 1.2 %), AMR mono (< 1 %) — ⚠️ **exception** : poisson GPU petites tailles (~10 %), AMR np=4 n=128 (15 %, jitter de regrid) |
| Mêmes paramètres numériques entre fronts | ✅ (identité briques↔DSL 8.9e-16) |
| SHA exacts dans le JSON | ✅ (frontend `5e17c7a`, scaling `0162d5f`) |
| Aucun graphe ne mélange master/PR | ✅ (garde-fou `plot_frontend.py` : une seule paire (SHA, branche) par figure) |

## Conclusions

1. **Les fronts Python ne pénalisent pas le calcul** : briques ≈ C++ (0.96×). Le DSL `production`
   était plus lent pour **deux raisons de compilation du `.so`** (ni codegen, ni orchestration
   Python), toutes deux corrigées dans **adc_cpp PR #253** : (a) flags `-O2` sans `-DNDEBUG` →
   `-O3 -DNDEBUG` = parité série (1.04×) ; (b) **pas de propagation Kokkos** (`-DADC_HAS_KOKKOS`/
   `-fopenmp`) → le bloc DSL tombait en **série** sur un module threadé (341 ms invariant) ; corrigé,
   il **scale** désormais et suit les briques (ratio 1.02 à 8 threads). `-march=native` (opt-in, `.so`
   JIT-local) le fait même **dépasser** le natif générique (0.88×). Reste le coût de compilation DSL
   froide (~15 s, entièrement caché → 1.8 s warm).
2. **Le solveur elliptique est le mur de scaling CPU** : le V-cycle MG n'accélère pas (voire
   ralentit) avec les threads ; le transport FV, lui, monte à ~6× sur 16 threads. Sur GPU, le GH200
   bat un socket de 1.8× (transport) à 4.2× (Poisson) — gains modestes car *bandwidth-bound* à ces
   tailles.
3. **Le multi-GPU transport scale bien** : 3.47× sur 4 GH200 (87 % d'efficacité), une fois levés
   les deux deadlocks (CUDA-IPC #254 + collective-sous-`if(rank0)` #258). L'AMR multi-GPU est correct
   (bit-identique) mais comm-bound aux tailles du plan (`n=128/256`) ; Poisson multi-GPU anti-scale
   pareillement (trop petit). Le mur de scaling n'est donc ni Python ni le transport, mais le
   **solveur elliptique** (CPU) et la **taille trop faible** (multi-GPU Poisson/AMR).

## Reproduire

```bash
# Frontend (3 fronts) :
PYTHONPATH=<adc_cpp>/build/python:. python3 perf/frontend_compare.py --n 256 --steps 50 \
    --poisson off --cpp-bin <adc_cpp>/build/bin/frontend_cpp
# Scaling (ROMEO) : bench/run_scaling.sh kokkos-omp|kokkos-cuda|mpi-cuda + scaling_amr
# Figures : python3 perf/plot_frontend.py --frontend <…>/frontend_compare.jsonl --scaling <…>/scaling.jsonl
```
