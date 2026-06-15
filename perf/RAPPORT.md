# adc_cpp / adc_cases performance campaign - report

**Without Hoffart.** Two axes: (1) the cost of the C++ vs Python-bricks vs Python-DSL frontends on a
safe case, (2) CPU/GPU/MPI scaling on controlled synthetic cases. The physical case is **pure
compressible Euler, periodic, smooth pressure bubble** (`rho>0`, `p>0` guaranteed, pure transport, no
physical Poisson, no Schur, no disk geometry).

## Provenance (exact SHAs)

| Data | Repo | SHA | Machine |
|---|---|---|---|
| Frontend (3 frontends) | adc_cpp `feat/perf-campaign-bench` | `5e17c7a` | ROMEO x64cpu (clean node, serial) |
| CPU + GPU + AMR scaling | adc_cpp `feat/perf-campaign-bench` | `0162d5f` | ROMEO x64cpu / GH200 |
| Python harness | adc_cases `feat/perf-campaign-harness` | `80c3049` | - |

Commit `0162d5f` (JSONL hardening + AMR) **does not touch** the frontend path (`frontend_cpp`,
`frontend_compare`, `safe_euler`): the `5e17c7a` frontend numbers hold for `0162d5f`. PRs: **adc_cpp
#246**, **adc_cases #33**. No figure mixes two SHAs (`plot_frontend.py` guard).

---

## Axis 1 - C++ / Python-bricks / Python-DSL frontends

Safe case, `n=256`, 50 steps, **fixed dt**, same numerical settings on all three frontends
(minmod / rusanov / conservative reconstruction / SSPRK2). Clean x86 node, **CV < 1.2 %**.

| Frontend | hot ms/step | ratio vs C++ | `advance` ms/step | cold-cache total | DSL backend |
|---|---|---|---|---|---|
| **C++ direct** | 21.0 | 1.00x | - | 1.18 s | - |
| **Python bricks** | 20.2 | **0.96x** | 20.2 | 1.18 s | - |
| **Python DSL (cold)** | 31.1 | **1.48x** | 31.1 | **17.4 s** | production |
| **Python DSL (warm)** | 31.1 | 1.48x | 31.0 | 1.77 s | production |

Numerical identity bricks vs DSL: `max|delta| = 8.9e-16` (machine epsilon). Mass conserved to
`2e-15`, `rho>0`, `p>0` on all three frontends. `poisson=on` is nearly identical (the inert solve at
charge=0 adds little at `n=256`).

**Where the time goes, and why:**

1. **Python bricks cost nothing in the hot path** (0.96x, within the noise). `step(dt)` in a Python
   loop ~= `advance(dt, nsteps)` in a single call ~= the same C++ kernel. -> *Python is not on the hot
   path*; per-step orchestration (one pybind call per `step`) is negligible next to the FV
   computation.
2. **Python DSL was ~1.5x slower in the hot loop - cause found and FIXED.** This is not a crossing
   cost (`advance` ~= `step` = 31 ms) nor codegen (the `.so` is in strict parity, flux inlined in
   `assemble_rhs`): it was the **compilation flags of the `.so`**. `compile_native` was compiling at
   **`-O2` without `-DNDEBUG`** (asserts live in the hot loop + weak vectorization), whereas the
   native build is `-O3 -DNDEBUG` (CMake Release). Controlled experiment (ROMEO x64cpu, clean node,
   CV<1%, job 648019):

   | production `.so` flags | hot ms/step | ratio vs C++ |
   |---|---|---|
   | `-O2` (before) | 31.0 | 1.48x |
   | `-O3 -DNDEBUG` (fixed, **adc_cpp PR #253**) | 21.8 | **1.04x (parity)** |
   | `-O3 -DNDEBUG -march=native -funroll-loops` | 18.5 | **0.88x (beats native)** |

   -> `-O3 -DNDEBUG` brings the DSL back to **parity** with the brick (within the brick noise of
   0.96x). And since the `.so` is **JIT-compiled on the target machine**, `-march=native` (opt-in via
   `$ADC_DSL_OPTFLAGS`) gives it the AVX-512/NEON that the generic native binary lacks -> the DSL
   **outperforms** the brick (0.88x). The codegen itself had nothing to fix.

   **THREADED regime (Kokkos OpenMP) - 2nd, deeper cause (cross-diagnosis with Codex).** On a Kokkos-
   OpenMP `_adc` module, the warm DSL stayed **341 ms INVARIANT** at threads=1/4/8 (a ratio that
   degraded 1.17->1.43->1.92): `compile_native` did **not** propagate `-DADC_HAS_KOKKOS`/`-fopenmp`
   -> the header-only loader templates instantiated against the **serial fallback**, and the DSL block
   did not scale. Fix (PR #253): compile the `.so` with the Kokkos headers + `-fopenmp` when
   `ADC_KOKKOS_ROOT` is defined, **without linking libkokkos** (otherwise a 2nd Kokkos runtime ->
   `SIGABRT` at finalize) - the symbols resolve from the already-loaded module (single runtime).
   ROMEO validation (Kokkos-OpenMP module, n=256, EXIT=0):

   | OMP threads | bricks ms/step | DSL ms/step | ratio |
   |---|---|---|---|
   | 1 | 24.3 | 27.3 | 1.12 |
   | 4 | 19.3 | 19.9 | 1.03 |
   | 8 | 10.5 | 10.7 | **1.02** |

   -> the DSL **now scales with the threads** and tracks the bricks (ratio -> 1.02), instead of the
   flat 341 ms. Without `ADC_KOKKOS_ROOT`, the historical (serial) behavior is unchanged.
3. **The DSL's own cost is compilation**: cold `dsl_compile` = ~15 s (`g++` subprocess), fully
   amortized by the out-of-source cache (`adc_cache_dir`, key `model_hash+abi_key`) -> warm = 1.8 s
   (~= bricks). The `import adc` (loading `_adc`) dominates the cold-cache of the non-DSL frontends
   (~1.2 s).

Figures: `frontend_cold_stages.png` (cold stages), `frontend_hot_ms.png` (hot +/- p10/p90),
`frontend_step_vs_advance.png`, `frontend_ratio.png`.

---

## Axis 2 - CPU / GPU / MPI scaling

### CPU OpenMP - strong (transport 4096^2, poisson 1024^2, CV < 1 %)

| threads | transport ms/step | speedup | poisson ms/step |
|---|---|---|---|
| 1  | 5239 | 1.00x | 71.2 |
| 2  | 5107 | 1.03x | **319.5** |
| 4  | 2736 | 1.91x | 214.0 |
| 8  | 1483 | 3.53x | 121.3 |
| 16 | 913  | **5.74x** (eff 36 %) | **82.4** |

- **Transport reaches 5.74x on 16 threads** (36 % efficiency): the FV kernel is *memory-bound*, so
  sub-linear is expected. Breakdown at 16t: transport 836 ms (dominant), alloc_tmp 57 ms, diag 38 ms,
  halos 13 ms, reduction 5 ms.
- **Poisson (V-cycle GeometricMG) is the scaling WALL**: 16 threads (82 ms) is *slower* than 1 thread
  (71 ms) - anti-scaling. The V-cycle is latency/sync-bound, and the coarse levels serialize. The peak
  at 2 threads (319 ms, reproducible across 2 jobs) is a NUMA placement effect (`OMP_PROC_BIND=spread`
  over 2 domains). This is a property of the solver, not of the harness.

### CPU - weak scaling (transport 512^2/u, poisson 256^2/u)

| units (threads) | transport ms/step | poisson ms/step |
|---|---|---|
| 1  (512^2 / 256^2)  | 80.8 | 4.2 |
| 4  (1024^2 / 512^2) | 194  | 54  |
| 16 (2048^2 / 1024^2)| 252  | 80  |

Weak efficiency (constant time is ideal): transport ~= 32 %, **poisson ~= 5 %** - confirms that MG
draws no benefit from CPU parallelism.

### GPU GH200 - single-GPU (Kokkos CUDA)

| workload | size | ms/step | cells/s |
|---|---|---|---|
| transport | 1024^2 | 30.1 | 3.48e7 |
| transport | 2048^2 | 124  | 3.39e7 |
| transport | 4096^2 | 497  | 3.37e7 |
| poisson   | 512^2  | 6.3  | 4.13e7 |
| poisson   | 1024^2 | 19.5 | 5.37e7 |

Transport throughput is **flat at ~3.4e7 cells/s** (GPU saturated from 1024^2 onward). **GH200 vs 16
x86 threads**: at equal size, transport 4096^2 -> **1.84x** (497 vs 913 ms), poisson 1024^2 ->
**4.2x** (19.5 vs 82 ms). Modest on transport: the kernel is *bandwidth-bound*, and a 16-thread socket
is not far from the GH200 HBM3 at these sizes. (GPU poisson CV ~10 % at small sizes - to be tightened
with more steps/sizes.)

### GPU - synthetic AMR multi-GPU (4 bubbles, np = 1/2/4, mass bit-identical cross-rank)

| n | np=1 (replicated) | np=2 (distributed) | np=4 (distributed) | mass drift |
|---|---|---|---|---|
| 128 | 215 ms | 1013 ms | 1390 ms | 0 / 2e-16 |
| 256 | 233 ms | 1080 ms | 1485 ms | 0 |

The AMR harness (the `AmrSystem` path compiled via `add_compiled_model`) **runs correctly on
multi-GPU**: mass is bit-identical cross-rank (`drift = 0`). But it **anti-scales at these sizes**: the
distributed coarse level (distributed MG + inter-GPU halos) dominates a tiny computation (`n=128/256` =
4 to 16 patches on 4 nearly-idle GH200). -> *the AMR problem is too small to benefit from multi-GPU*; a
representative case would need `n>=1024`. This is the AMR scaling lock at these sizes (comm-bound), not
a correctness defect.

Figures: `scaling_strong_transport_kokkos-omp.png`, `scaling_strong_poisson_kokkos-omp.png`,
`scaling_weak_transport_kokkos-omp.png`, `scaling_weak_poisson_kokkos-omp.png`,
`scaling_strong_amr_kokkos-cuda.png`.

### MPI multi-rank - UNBLOCKED (two bugs, both fixed)

The **hybrid MPI x OpenMP** and the **multi-GPU transport/poisson** were deadlocking at np>=2. Two
stacked bugs, both fixed:
1. **GPU CUDA-IPC** (`fill_boundary` handed UVM buffers to OpenMPI -> the smcuda BTL routed over
   CUDA-IPC, impossible between cgroup-isolated GPUs) -> fixed with `SharedHostPinnedSpace` buffers
   (**adc_cpp #254**, on master).
2. **MPI collective under `if(rank0)`** in the bench itself: the `rmax() = all_reduce_max()`
   aggregations were evaluated inside the rank-0 `printf` -> rank 0 issued ~6 more `MPI_Allreduce`
   than the others -> deadlock. Located via rank-tagged checkpoints (`fill_boundary` completes; the
   hang is in the JSON emission). Fixed by hoisting the `rmax` calls out of the rank-0 block (**adc_cpp
   #258**). `scaling_amr` was already correct -> it worked on multi-GPU.

**Multi-GPU GH200 - transport 4096^2 strong scaling** (np = 1 GPU/rank, after the fix):

| GPUs | ms/step | cells/s | speedup | efficiency |
|---|---|---|---|---|
| 1 | 490 | 3.4e7 | 1.00x | 100 % |
| 2 | 282 | 5.9e7 | 1.73x | 87 % |
| 4 | **141** | **1.19e8** | **3.47x** | **87 %** |

-> **excellent multi-GPU strong scaling** (87 % efficiency on 4 GH200). Poisson 1024^2 anti-scales on
multi-GPU (too small, MG comm-bound). **Hybrid CPU MPI x OpenMP** (transport 4096^2): 1x16=839 ms,
2x8=812 ms, 4x4=465 ms (better NUMA locality at 4 ranks).

---

## Tests and acceptance

| Criterion | Status |
|---|---|
| Invariants OK (mass conserved, `rho>0`, `p>0`) | OK everywhere |
| No NaN | OK (`nan=false` on all runs) |
| CV < 5 % | OK CPU (transport/poisson/weak < 1 %), GPU transport (< 1 %), frontend (< 1.2 %), AMR single (< 1 %) - WARN **exception**: GPU poisson at small sizes (~10 %), AMR np=4 n=128 (15 %, regrid jitter) |
| Same numerical parameters across frontends | OK (bricks vs DSL identity 8.9e-16) |
| Exact SHAs in the JSON | OK (frontend `5e17c7a`, scaling `0162d5f`) |
| No graph mixes master/PR | OK (`plot_frontend.py` guard: a single (SHA, branch) pair per figure) |

## Conclusions

1. **The Python frontends do not penalize the computation**: bricks ~= C++ (0.96x). The `production`
   DSL was slower for **two `.so` compilation reasons** (neither codegen nor Python orchestration),
   both fixed in **adc_cpp PR #253**: (a) `-O2` flags without `-DNDEBUG` -> `-O3 -DNDEBUG` = serial
   parity (1.04x); (b) **no Kokkos propagation** (`-DADC_HAS_KOKKOS`/`-fopenmp`) -> the DSL block fell
   back to **serial** on a threaded module (341 ms invariant); fixed, it now **scales** and tracks the
   bricks (ratio 1.02 at 8 threads). `-march=native` (opt-in, JIT-local `.so`) even makes it
   **outperform** the generic native build (0.88x). What remains is the cold DSL compilation cost
   (~15 s, fully hidden -> 1.8 s warm).
2. **The elliptic solver is the CPU scaling wall**: the MG V-cycle does not speed up (and even slows
   down) with threads; the FV transport, on the other hand, reaches ~6x on 16 threads. On GPU, the
   GH200 beats a socket by 1.8x (transport) to 4.2x (Poisson) - modest gains because the workload is
   *bandwidth-bound* at these sizes.
3. **Multi-GPU transport scales well**: 3.47x on 4 GH200 (87 % efficiency), once the two deadlocks
   were lifted (CUDA-IPC #254 + collective-under-`if(rank0)` #258). Multi-GPU AMR is correct
   (bit-identical) but comm-bound at the plan's sizes (`n=128/256`); multi-GPU Poisson anti-scales
   likewise (too small). The scaling wall is therefore neither Python nor transport, but the **elliptic
   solver** (CPU) and the **too-small size** (multi-GPU Poisson/AMR).

## Reproduce

```bash
# Frontend (3 frontends):
PYTHONPATH=<adc_cpp>/build/python:. python3 perf/frontend_compare.py --n 256 --steps 50 \
    --poisson off --cpp-bin <adc_cpp>/build/bin/frontend_cpp
# Scaling (ROMEO): bench/run_scaling.sh kokkos-omp|kokkos-cuda|mpi-cuda + scaling_amr
# Figures: python3 perf/plot_frontend.py --frontend <…>/frontend_compare.jsonl --scaling <…>/scaling.jsonl
```
