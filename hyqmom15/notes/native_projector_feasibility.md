# Feasibility note: relaxation15 as a native compiled projector (ADC-275)

ADC-275 asks for a production path for `relaxation15`, compiled and device-safe, wired onto the
generic post-step projection hook of ADC-177 (`m.projection`), to replace the per-cell Python
loop `relaxation.py::relax_field` while keeping `relax_field` as the reference oracle.

The framing comment proposes emitting `relaxation15` as a DSL projection through `m.projection`
rather than as a hand-written C++ brick. This note records why that emission is **not achievable
on ADC-177 as merged** (`adc_cpp` master, commit 396ac64), with reproducible evidence, and
states precisely what ADC-177 / the DSL must add before the implementation PR. No production
projector is shipped: per ADC-273 (this issue is `blockedBy` ADC-273) the architecture choice is
gated on the multi-agent vote, and the task brief mandates a documented blocker over invalid code.

The evidence below is reproduced by `hyqmom15/probe_native_projector.py` (numpy only, no `_adc`
build required) against the existing realizability goldens (`golden/golden_relax_*.csv`).

## What ADC-177 `m.projection` provides

`HyperbolicModel.projection(exprs)` (adc_cpp `python/adc/dsl.py`, symbol
`HyperbolicModel.projection`) takes one expression per conservative component and emits a C++
`project(U, aux) -> State` (trait `HasPointwiseProjection`), applied by the System at the end of
each whole macro-step on the valid cells of each block. Its contract:

- **idempotent**: the docstring requires `P(P(U)) == P(U)` (a projection, not a relaxation);
- **pointwise**: no neighbor read;
- **branchless**: the realizability clamps must be written in `max` / `min` via `abs_` / `sign`,
  with no `if` (the documented ADC-177 doctrine).

The expression node set usable inside a projection is the full DSL `Expr` algebra and nothing
more: `Const`, `Var`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, `Sqrt`, `Abs`, `Sign`,
`StateRef` (Roe only), `RuntimeParamRef`. There is **no eigenvalue node, no determinant node, no
conditional node**, and the `project(U, aux)` codegen body emits only the cons / prim / aux
locals followed by the expressions (adc_cpp `python/adc/dsl.py`, the `if self._proj is not None`
block of `emit_cpp_brick`): it has no access to a per-cell flux jacobian or its spectrum.

## Blocker 1: relaxation15 is not idempotent

`relax_field` applies `relax15` per cell, and `relax15` is a relaxation toward a target, not a
projection onto a set. `run_relaxation.py` already documents this ("relaxation15 n'est PAS
idempotente ... c'est une RELAXATION vers une cible, pas un projecteur") and asserts no
idempotence, faithful to MATLAB.

Measured on the 12 realizability goldens (`probe_native_projector.py`, idempotence section):

```
state  0 branch 0  idemp-gap(rel) 3.0e-16
state  2 branch 4  idemp-gap(rel) 4.4e-16
state  3 branch 1  idemp-gap(rel) 9.3e-01
state  8 branch 4  idemp-gap(rel) 5.9e+01
state 10 branch 4  idemp-gap(rel) 3.3e-01
WORST idempotence gap (rel): 5.9e+01  ->  relax15 is NOT idempotent
```

Re-applying `relax15` keeps relaxing: the worst relative gap between `P(P(U))` and `P(U)` is
`5.9e+01` (state 8, collision branch). Even a "branch 1" state (the `s30/s03` clamp) is
non-idempotent, because the clamped state routes through the collision step on the second pass
(`s21`, `s12` change on re-apply). Emitting `relax15` as `m.projection` therefore **violates the
`P(P(U)) == P(U)` contract**: the System applies the hook once per whole step, so a single pass
matches `relax_field`, but the hook is declared a projection and any robustness policy that
re-applies it (the issue mentions a possible `after_stage` mode) would diverge from `relax_field`,
and `check_model` style idempotence checks on the emitted projection would fail by construction.

A faithful native `relaxation15` needs a post-step hook with **relaxation (non-idempotent)
semantics**, not the projection contract: either a relaxed `HasPointwiseProjection` that drops the
idempotence requirement (and is renamed accordingly), or a separate `after_step` pointwise
operator. This is an ADC-177 scope item.

## Blocker 2: the spectral predicates have no DSL expression

`relax15` and its callee `collision15_anisotropic` take two spectral decisions that have no
counterpart in the projection expression algebra:

1. **complex flux eigenvalues** (`relax15`): the test
   `any(|Im(lambda)| > 1e-9 max(1, |lambda|))` on the eigenvalues of the order-3 jacobian
   sub-blocks (`x = [12,13,14]`, `y = [3,8,4]`) **evaluated at the standardized state**
   (`M00 = 1`, `u = v = 0`, `sigma = 1`), via `numpy.linalg.eigvals`. When true,
   `s21 = s12 = 0`, `s22 = max(s22, 1/3)`.
2. **realizability of the p2p2 matrix** (`collision15_anisotropic`): the gate
   `max(0, lambda_min(p2p2)) <= lamin` via `numpy.linalg.eigvals` of the 3x3 `p2p2_2d`, and the
   CJ-fallback gate `det(p2[0:2,0:2]) < 0 or p2[0,0] < 0` via `numpy.linalg.det`.

Both are load-bearing on the goldens (`probe_native_projector.py`, spectral section):

```
complex-eig branch (cx|cy) fires on 3/12 states (3, 4, 10)
collision gate (lam0(p2p2) <= lamin) active on every branch-4 state and others
```

The DSL has no eigenvalue or determinant `Expr` node. The C++ primitives exist
(`adc::real_eig_minmax` and `EigBounds.max_im`, `include/adc/numerics/dense_eig.hpp`, device-safe
`ADC_HD`), but they are wired into one place only: the `set_wave_speeds_from_jacobian` codegen
(`adc_cpp python/adc/dsl.py`, the `adc::real_eig_minmax(Jb_)` emission). They are **not exposed as
a composable expression** that a projection body could call on a sub-block and read `max_im` /
`lmin` from. Two further obstacles compound this:

- the complex-eig test needs the jacobian at the **standardized** state, not at `U`; `project(U,
  aux)` has no access to a flux jacobian at all, let alone at a transformed state;
- `det` of a 2x2 has no node either (expressible as `a*d - b*c` once the entries are available,
  but the entries are p2p2 matrix elements built from the standardized moments, which is fine; it
  is `eigvals` of the full 3x3 that has no node).

A faithful native projector needs the DSL to expose, as expressions usable in `project`:

- an eigen-bound node over a small fixed matrix of expressions, returning `lmin` / `lmax` /
  `max_im` (a thin `Expr` wrapper over `adc::real_eig_minmax`), and
- a way to build the order-3 jacobian sub-blocks at the standardized state (or, equivalently, to
  precompute the two `3x3` blocks as expressions of the standardized moments and feed them to the
  eigen-bound node).

These are ADC-177 / DSL scope items.

## What IS expressible (and why it is not relaxation15)

The spectral-free realizability clamps map cleanly to the projection algebra and are idempotent:
the `s30/s03` clamp `|s30| <= 4 + Ma/2` preserving `H20` (via `sign` and `min`), the `H20/H02`
floors, and the `s11` clamp, all in `max` / `min` / `abs_` / `sign` with no branch. The probe
builds this `clamp_proj` and measures (clamp section):

```
clamp-only projector: idempotent everywhere (gap ~1e-14)
matches relax15 only on branch 0 (identity); differs by 80-160x on branches 1, 2, 4
```

It matches `relax15` only on the identity branch. On every other golden state, `relax15` routes
through the collision and / or complex-eig path, so the clamp-only operator is a **different
operator**, not an approximation of `relaxation15`. Shipping it as "the production `relaxation15`
projector" would be misleading: it would not match the oracle, and the Ma=20 realizability gain it
buys is the clamp's, not `relaxation15`'s. It is kept here only as the feasibility probe.

## Decision

Out of short-term scope until ADC-177 / the DSL gains the two facilities above (relaxation-typed
post-step hook + eigen-bound expression node with standardized-state jacobian access), and gated on
the ADC-273 multi-agent vote (API / ABI choice). The oracle `relaxation.py::relax_field` stays the
reference and the production validation path; GPU / MPI campaigns apply no in-step projection until
the native path lands (already stated in the `relaxation.py` header and the README).

Follow-ups to file:

- ADC-177 (adc_cpp): post-step pointwise hook with **relaxation** (non-idempotent) semantics, and
  an **eigen-bound `Expr` node** over a small matrix of expressions exposing `lmin` / `lmax` /
  `max_im`, usable inside `m.projection` / the new hook; plus standardized-state jacobian access
  for the order-3 sub-blocks.
- ADC-275 (this issue): once the above lands, port `relax15` branch by branch onto the hook
  (clamps already prototyped here; collision and complex-eig via the eigen-bound node), validate
  native vs `relax15` on the goldens (1e-12 to 1e-10 by branch) and vs `relax_field` on a
  `(15, ny, nx)` field, plus the no-projection bit-identity and the Ma=20 realizability criteria.

## Provenance

- adc_cpp master inspected at commit 396ac64 ("ADC-177 Hook generique de projection ponctuelle
  post-pas (DSL m.projection)"). Symbols cited: `HyperbolicModel.projection`,
  `HyperbolicModel.projection_value`, the `emit_cpp_brick` projection block,
  `set_wave_speeds_from_jacobian`, `adc::real_eig_minmax`, `EigBounds.max_im`
  (`include/adc/numerics/dense_eig.hpp`).
- Evidence script: `hyqmom15/probe_native_projector.py` (numpy only). Goldens:
  `hyqmom15/golden/golden_relax_in.csv`, `golden_relax_out.csv`, `golden_relax_meta.csv`
  (12 states, branches 0-4, from `golden_relax_gen.m`).
- Oracle: `hyqmom15/relaxation.py` (`relax15`, `collision15_anisotropic`, `p2p2_2d`,
  `make_corner_eigs`), port of `relaxation15.m` / `collision15_anisotropic.m` / `p2p2_2D.m`.
