# `safe_euler_periodic/` - safe reference case (validation)

**Pure** compressible Euler, periodic domain, low-amplitude **smooth pressure bubble**:

- `rho ≡ rho0 = 1` (uniform density -> `rho > 0` guaranteed);
- `v = 0` at `t = 0`; `p = p0 + dp·exp(-r²/(σ²L²))` with `p0 = 1`, `dp = 0.1` (-> `p > 0` guaranteed);
- `E = p/(γ-1)`, `γ = 1.4`; NO source, NO Poisson coupling (pure transport).

The pressure bubble expands into acoustic waves: nontrivial but smooth dynamics, no shock,
no loss of positivity. This is the reference case for the performance campaign (`perf/`), chosen to be
**safe** (no physical Poisson, no Schur, no disk geometry) so as to isolate the cost
of the fronts at identical compute.

## What this case validates (CI)

- **Bricks <-> DSL equivalence**: **bit-identical** final state (`np.array_equal`, tolerance 1e-10),
  like `diocotron_dsl` / `two_species_dsl`. Same settings: minmod / rusanov / conservative
  reconstruction / SSPRK2 / fixed `dt`.
- **Invariants**: conserved mass (transport, periodic), `rho > 0`, `p > 0`, finite state.
- **Dynamics**: `max|Δp| > 1e-4` (the bubble evolves, the case is not trivial).

## Source of truth

The model (bricks and DSL), the CI, the `dt`, and the settings live in
[`adc_cases/common/safe_euler.py`](../adc_cases/common/safe_euler.py) - shared with
`perf/frontend_compare.py`. The direct C++ counterpart is `adc_cpp/bench/frontend_cpp.cpp` (namespace
`safecase`): the constants and the numerical scheme **must** match it bit for bit.

## Running

```bash
PYTHONPATH=<adc_cpp>/build-master/python:. python3 safe_euler_periodic/run.py --n 64 --steps 40
```

Requires a C++20 compiler (`needs = ["cxx"]`) for the `production`/`aot` DSL compilation. The
performance **measurement** (3 fronts, timings, figures) is NOT here - see [`perf/`](../perf).
