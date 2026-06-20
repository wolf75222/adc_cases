# tests/: case guardrails

Tests that lock down the assembly, sign, and flag contracts against the **real** `adc`
extension (no fake/mock module). Each one adds the case root (`..`) to `sys.path` so it can
import `model` / `run` / `run_polar`, and builds a real `adc.System` (native bricks, no DSL
compile) or calls the pure helpers directly.

Run them from the repository root, the same way CI does (the real `adc` must be importable;
`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` for the Kokkos-OpenMP build):
`PYTHONPATH=<adc_cpp>/build/python python3 hoffart_euler_poisson_dsl/tests/<test>.py`

`test_geometry_flag.py` additionally **compiles the DSL model** (it drives the real
`build_uniform`), so it needs a compile-capable `adc` build (`ADC_INCLUDE=<adc_cpp>/include`
and a C++ compiler); the other three only need a real `adc`.

| File | Category | Checks |
|---|---|---|
| `test_polar_assembly.py` | validation (CI) | The real polar path (`run_polar.py`): `build_polar_system` yields a valid `(3, ntheta, nr)` state + finite polar-Dirichlet phi; annular top-hat density; `v_r`/`v_theta` are the ExB/rotating-equilibrium drifts of the solved phi; the rotating equilibrium is stationary; the frozen-equilibrium residual `R_eq = step(U_eq) - U_eq` makes `U_eq` a machine-precision fixed point over >= 200 steps; `polar_gradient`/`fit_growth` exactness; multi-rank rejected. 15 checks. |
| `test_signs.py` | guardrail | The sign conventions of the model.py numpy helpers (electric/Lorentz force, Poisson RHS `-alpha*rho`). 6 checks. |
| `test_geometry_flag.py` | guardrail | The `--geometry {square,staircase,cutcell}` flag of `run.py`: drives the real `build_uniform` and asserts on `sim.disc_mask()` -- square keeps the full Cartesian mask (n*n), staircase/cutcell restrict it to the disc R centered at L/2; unknown geometry raises; staircase+amr-imex rejected at the argument layer. 5 checks. |
| `test_dump_npz.py` | guardrail | The `--dump-npz` raw-state dump of `run.py`: the flag defaults off and parses; `resolve_dump_npz` enables it single-rank and disables it under MPI (np>1); a real `adc.System` writes `mode_<l>/state_<NNNNNN>.npz` via `sim.write(format='npz', step=idx)` carrying the real state/phi/clock. 7 checks. |
