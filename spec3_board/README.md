# spec3_board: write a model "at the blackboard" and watch it lower to the operator-first kernel

Spec 3 adds a physico-mathematical facade (`adc.physics.Model` + `adc.math`) that reads like
equations, without replacing the Spec 2 operator-first kernel (`adc.model.Module` +
`adc.time.Program`). The facade only builds the same objects: it has no registry, scheduler or
codegen of its own. This case proves that from the application side, by introspection, compiling
nothing: it authors an Euler-Poisson-Lorentz model at the blackboard, checks it lowers to a typed
`adc.model.Module`, and asserts a board-written time step produces the exact same Program IR as the
explicit operator-first step. This is not a reproduction of a published result.

## Contract

| Field | Content |
|---|---|
| Category (manifest) | `tutoriel` (`cases_manifest.toml`, `spec3_board/run.py`, `ci = true`, `needs = []`) |
| Inputs | none numeric: the case builds IR only (no grid, no time integration, no compile) |
| Outputs | stdout: `m.dump_module_ir()` (the typed Module) and `T.dump_operator_ir()` (the lowered Program IR) |
| Guaranteed invariants | the `assert` in `run.py`: the board model lowers to `StateSpace U(rho, mx, my)`, a `FieldSpace`, and operators `explicit_rate` (kind `local_rate`, `(U, Fields) -> Rate(U)`) and `implicit_operator` (kind `local_linear_operator`, `Fields -> LocalLinearOperator(U, U)`); and the board time step IR equals the operator-first step IR node-for-node |
| Proves | the facade is a thin lowering, not a second system: `m.physics.Model` -> `adc.model.Module`, and `T.fields`/`T.define`/`T.solve`/`T.commit` -> `P.solve_fields`/`P.linear_combine`/`P.solve_local_linear`/`P.commit` with identical IR (the anti-duplication guarantee, Spec 3 criterion 8) |
| Does not prove | nothing numerical: no simulation runs here. A case that COMPILES and RUNS a board-authored model end to end (Python describes, C++ executes) is a follow-up (it needs a C++ compiler + Kokkos, `needs = ["cxx"]`) |
| Requires | an `adc` that exposes the Spec 3 surface (`adc.physics`, `adc.math`, `adc.time`); built from adc_cpp master by the CI |

By the end you will know: how a blackboard model maps to the typed operator-first IR, and how to
read the lowering yourself with `Model.dump_module_ir()` / `Program.dump_operator_ir()`.

## 1. The board model

The model is written close to the equations (`build_board_model` in `run.py`):

```
d_t U = -div F(U) + A(E)U        (m.rate)
-Delta phi = alpha (rho - rho_ref)   (m.solve_field)
E = -grad phi                    (m.vector_field)
C(B): U -> U  (Lorentz)          (m.local_linear_operator + m.operator)
```

`m.local_linear_operator(...)` builds a math object; `m.operator("implicit_operator", returns=...)`
registers the callable typed operator (a non-registered math object is not callable, by design).

## 2. The lowering (what the case checks)

`m.module` is the real `adc.model.Module`. The case asserts the expected spaces and operator kinds,
then builds the implicit step two ways and compares the full Program IR:

- board: `T.fields` / `T.rhs` / `T.solve((I - dt*C) @ unknown("U*") == U_n + dt*R_n)` / `T.commit`;
- operator-first: `P.solve_fields` / `P.rhs` / `P.linear_combine` + `P.solve_local_linear` / `P.commit`.

The two IRs are identical, so the board notation adds readability without adding semantics.

## 3. Run

```
python spec3_board/run.py
```

Exit 0 means the lowering and the IR equivalence both hold.
