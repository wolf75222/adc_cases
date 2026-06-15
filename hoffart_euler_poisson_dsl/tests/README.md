# tests/: case guardrails

Build-free tests (they install a minimal fake `adc` module) that lock down the
assembly and sign contracts without running the heavy simulation. Each one adds the case root
(`..`) to `sys.path` so it can import `model` / `run` / `run_polar`.

Run them from the repository root, the same way CI does:
`PYTHONPATH=<adc_cpp>/build/python python3 hoffart_euler_poisson_dsl/tests/<test>.py`

| File | Category | Checks |
|---|---|---|
| `test_polar_assembly.py` | validation (CI) | The order of facade calls on the polar path (`run_polar.py`): polar/Dirichlet Poisson -> magnetic field before the Schur stage -> WENO5+Rusanov+SSPRK3+CondensedSchur -> annular top-hat density -> steady rotating equilibrium. 16 assertions. |
| `test_signs.py` | guardrail | The sign conventions of the DSL model (electric/Lorentz force, Poisson RHS `-alpha*rho`). 6 assertions. |
| `test_geometry_flag.py` | guardrail | The `--geometry {square,staircase}` flag of `run.py` (AMR-IMEX path; needs an AMR build, so it is excluded from lightweight CI). |
