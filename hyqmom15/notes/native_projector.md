# Native relaxation15 projector (ADC-275)

ADC-275 asks for a production path for `relaxation15`, compiled and device-safe, wired onto the
generic post-step projection hook of ADC-177 (`m.projection`), to replace the per-cell Python loop
`relaxation.py::relax_field` while keeping `relax_field` as the reference oracle.

This note records the design of the native projector now shipped in
`model.py::build_projection` and the one fidelity caveat (idempotence). The earlier version of this
note recorded why the emission was blocked on ADC-177 as first merged; both blockers are now
closed, so this note documents the implementation instead.

## How it is emitted

`build_moment_model(..., projection=True, Ma=..., lamin=...)` calls `build_projection(m)` and feeds
the result to `m.projection([...])` (one expression per conservative component, ADC-177). The
System applies the emitted `project(U, aux)` at the end of each whole macro-step on the valid cells
of each block (post-step semantics, never per RK stage), compiled like the flux (CSE, device-clean;
backends `aot` via `add_compiled_block` and `production` via `add_native_block`; the `prototype`
backend and `amr_system` target reject a projection hook). Without `projection=True`, no hook is
emitted and the transport path is bit-identical.

`build_projection` is a branch-by-branch transcription of `relaxation.relax15` (the oracle) into
the DSL `Expr` algebra, with no dynamic branch (the pointwise contract of ADC-177). Each MATLAB
branch becomes a branchless blend by a mask `mask = 0.5*(sign(c) + 1)` (1 when `c > 0`), with
`max(a,b) = (a + b + |a-b|)/2` and `min(a,b) = (a + b - |a-b|)/2`. The standardized moments
`S_pq`, the de-standardization factors `sx`, `sy`, and `u`, `v`, `M00` are reused directly from the
moment model's named primitives (no re-derivation): the same algebra as the flux.

The two formerly missing facilities are now available:

- the **complex flux eigenvalues** test (order-3 jacobian sub-blocks `x = [12,13,14]`,
  `y = [3,8,4]` at the standardized state) uses `dsl.eig_max_im(rows)` (ADC-289, a thin `Expr`
  over `adc::real_eig_minmax`); the 3x3 blocks are built once from `m.flux_jacobian` with the
  standardizing primitives frozen (`u=v=0`, `sx=sy=1`, `M00=1`) so they reduce to functions of the
  `S_pq` only (verified: the block witness matches `make_corner_eigs` to machine precision on the
  goldens), then the current post-clamp `S_pq` are substituted back in;
- the **p2p2 realizability** gate of `collision15_anisotropic` uses `dsl.eig_lmin(p2)` (the
  smallest real eigenvalue of the 3x3 `p2p2` matrix, transcribed in `Expr`), and the CJ-fallback
  gate `det(p2[0:2,0:2]) < 0 or p2[0,0] < 0` is the explicit `a*d - b*c` and `p2[0,0]` on the same
  matrix expressions.

`M00`, `M10`, `M01` are passed through unchanged (`relaxation15` only touches order >= 2 moments;
`u`, `v`, `M00` are conserved).

## Application policy

`relaxation15` follows the MATLAB `flagrelax = 1`: applied per whole step. The ADC-177 hook runs
post-step, which matches. An `after_stage` (per RK stage) application would trade MATLAB fidelity
for extra robustness; it is not the emitted policy.

## Idempotence caveat

`relaxation15` is a relaxation toward a target, not a strict projection: `P(P(U)) != P(U)` in
general (re-relaxing relaxes again; the worst re-apply gap on the goldens is large on the collision
branch). The ADC-177 `m.projection` contract documents idempotence as the intended property, but
the System applies the hook once per macro-step, so a single pass reproduces `relax_field` exactly,
which is what the acceptance criteria require and what the drivers assert. Do not enable an
`after_stage` / repeated application expecting it to match `relax_field` bit for bit: it will keep
relaxing, faithful to MATLAB but not idempotent.

## Validation

Acceptance criteria of the issue, checked by `validate_native_projector.py` (compiled brick
`project(U, aux)`, against the relax goldens and the Python oracle):

- cell test: compiled `project` == `relax15` on the 12 goldens (branches 0-4), worst absolute gap
  ~1e-15, well inside the 1e-12 to 1e-10 per-branch tolerance;
- field test: compiled projection over a `(15, ny, nx)` field == `relax_field`;
- Ma=20 realizability: the native non-realizable cell rate drops as with `relax_field`;
- no-regression: `projection=False` emits no `project` hook (bit-identical transport).

`relaxation.py` stays the **oracle** of reference (`relax15` === Octave to ~4e-14,
`run_relaxation.py`); the native projector is the **production** path.

## Provenance

- Oracle: `hyqmom15/relaxation.py` (`relax15`, `collision15_anisotropic`, `p2p2_2d`,
  `make_corner_eigs`), port of `relaxation15.m` / `collision15_anisotropic.m` / `p2p2_2D.m`.
- adc_cpp prerequisites on master: `m.projection` (ADC-177) and `dsl.eig_max_im` /
  `dsl.eig_lmin` / `dsl.eig_lmax` (ADC-289, witnesses over `adc::real_eig_minmax`,
  `include/adc/numerics/dense_eig.hpp`).
- Goldens: `hyqmom15/golden/golden_relax_{in,out,meta}.csv` (12 states, branches 0-4, from
  `golden/gen/golden_relax_gen.m`).
