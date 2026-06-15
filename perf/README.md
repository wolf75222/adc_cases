# `perf/` - performance measurement campaign

Two axes, measured only as a delta from `origin/master` of both repositories (`adc_cpp`,
`adc_cases`). The physics case is the **safe case**: pure compressible Euler, periodic, a smooth
low-amplitude pressure bubble (`rho > 0`, `p > 0` guaranteed), pure transport. Source of truth for
the case: [`adc_cases/common/safe_euler.py`](../adc_cases/common/safe_euler.py); C++ counterpart:
`adc_cpp/bench/frontend_cpp.cpp` (namespace `safecase`). The case **validation** (front equivalence
plus invariants, without measurement) is the registered case [`safe_euler_periodic/`](../safe_euler_periodic).

## Axis 1 - fronts: C++ direct vs Python bricks vs Python DSL

`frontend_compare.py` runs the **same** physics with the **same** numerical settings
(minmod / rusanov / conservative reconstruction / SSPRK2 / FIXED `dt`) across three fronts; the only
difference measured is the cost of the front itself, at identical computation.

- **C++ direct**: the `adc_cpp/bench/frontend_cpp` binary (subprocess).
- **Python bricks**: `adc.System` + `add_block(models.euler)` + `step(dt)`.
- **Python DSL**: `adc.dsl.Model(...).compile(backend="production")` + `add_equation` + `step(dt)`.

**Cold-cache** methodology: each Python front runs in a **fresh subprocess** (the `adc` import is
genuinely cold, the DSL cache is controlled). The DSL is measured **cold** (empty `so_dir` -> `g++`
compilation) then **hot** (same `so_dir` -> cache hit). Per-stage timing: `import` /
`model_build` / `dsl_compile` / `addblock` / `state_init` / `first_step` / `warmup` / `run_loop` /
`diag`; plus the hot loop (`median/p10/p90/cv`), `advance(dt,nsteps)` (a single Python call, isolates
the per-step crossing), and, when Poisson is active, `solve_fields` in isolation.

```bash
# from adc_cases, with the build on PYTHONPATH
PYTHONPATH=<adc_cpp>/build-master/python:. python3 perf/frontend_compare.py \
    --n 256 --steps 50 --warmup 5 --poisson off \
    --cpp-bin <adc_cpp>/build-bench-serie/bin/frontend_cpp
python3 perf/plot_frontend.py   # figures in out/safe_euler_periodic/figures/
```

`--poisson off` (default) = pure transport, a clean frontend signal. `--poisson on` = an **inert**
elliptic solve (charge=0) at every step, an MG-dominated regime (the `two_euler` idiom). Both modes
stay symmetric across the three fronts.

**Granularity asymmetry (assumed).** `System` has no internal timer: the C++ front gives the
7-phase breakdown (poisson/aux/halos/transport/reduction/fence/alloc, via the `profile_step`
machinery), the Python fronts give only `total + solve_fields`. The cross-comparison stays valid on
the **cold-cache total time** and the **hot ms/step**.

## Axis 2 - CPU/GPU/MPI scaling

Driven on the `adc_cpp/bench/scaling_step.cpp` side via `bench/run_scaling.sh` (multi-box, real MPI
halos). Workloads: `transport` (4096^2), `poisson` (1024^2), `amr` (not wired into this binary -> an
explicit diagnostic line). The JSONL produced (one per sweep point) is plotted by
`plot_frontend.py --scaling <file.jsonl>` (strong speedup/efficiency, weak efficiency, throughput).

## Local vs cluster

- **Mac (serial)**: validation/plumbing ONLY - checks the API wiring, the `production` DSL
  compilation, numerical identity, the invariants, the JSONL schema, the figures. Serial timings are
  labeled `machine=<mac>, backend=serial` and **excluded** from any scaling claim.
- **ROMEO (GH200/MPI/OpenMP)**: the ONLY source of valid scaling numbers (CV<5%, cells/s,
  p10/p90, ratios). `bench/run_frontend.sh` and `bench/run_scaling.sh` carry the per-backend build
  recipes.

## JSONL schema (`adc_perf_v1`)

Each line carries: `adc_cpp_sha`/`adc_cpp_branch`/`adc_cases_sha`/`adc_cases_branch`, `backend`,
`machine`, `ranks`/`threads`/`gpus`, `nx`/`ny`/`boxes`/`max_grid`, `workload` + numerical settings,
`stages{...}`, `total_cold_user_s`, `hot_ms_per_step{median,p10,p90,cv}`, `advance_ms_per_step`,
`phases_ms_per_step{...}`, `cells_per_s`, `invariants{mass,rho_min,p_min,nan}`.

## Acceptance

A result is publishable only if: invariants OK, no NaN, `cv < 5 %`, numerical settings identical
across fronts, exact SHAs in the JSON, and **no graph mixes master and a PR**
(`plot_frontend.py` refuses to plot two `(adc_cpp_sha, adc_cpp_branch)` in the same figure).
