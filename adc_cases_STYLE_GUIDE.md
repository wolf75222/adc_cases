# Style guide: adc_cases case tutorials

## 1. Guiding principle

You aim for a README that makes a case reproducible and understood by a reader with no prior knowledge: they must be able to re-run the case, read the real code line by line, follow the math derivations with no gaps, understand the physical mechanism, and know exactly what is proven and what is not. You refuse three things: promotional bullshit (empty words, sales tone), over-selling (presenting a validation as a reproduction, over-reading a diagnostic figure), and redundancy (re-stating in section 12 what section 1 already said). Every sentence carries a fact, a number, a symbol, a sign, or a verifiable reason; otherwise it goes. Honesty about the limits is not a concession, it is the heart of the contract: the `Does not prove` clause is as detailed as the positive guarantees.

## 2. Canonical structure of a case README

The README always opens with the Contract block (section 2.0), then chains the substantive sections. Each substantive section must name the contract clause it justifies; a section that justifies no clause is cut.

### 2.0 Contract block (mandatory, at the top, all types)

A dense table, the hardest to pad with bullshit. Columns / rows:

| Field | Content |
|---|---|
| Category (manifest) | the exact value from `cases_manifest.toml` (`validation` / `tutoriel` / `reproduction` / `reproduction-candidate` / `experimental`). It dictates the tone of Does not prove (see 2.x). |
| Inputs | grid (n, L, periodic?), IC, physical parameters with values and units (or "dimensionless", e.g. `four_pi_G=1`). |
| Outputs | state(s) read, diagnostics computed, figures/CSV produced and their location. |
| Guaranteed invariants | the list of the real `assert` statements in `run.py`, each with its tolerance. |
| Proves | what an `assert` actually establishes (e.g. sign of dE, mass conserved to 1e-9, bit-exact equality). |
| Does not prove | as detailed as Proves: name the proxy, the regime, what is not tested (see 2.x). |
| Provenance | adc_cpp + adc_cases SHA, backend, resolution, measured cost, platform. |

The coupling rule: each downstream section writes explicitly "justifies clause X of the contract". If a Proves / Does not prove clause is justified by no section, you either write the section or remove the clause.

### 2.x The tone of Does not prove is dictated by the category

- `reproduction` (e.g. `diocotron/run.py`): state what is reproduced (analytic oracle == paper to 3 digits) and what is not (FV rate underestimated by -22 to -27 %). No pending: the reproduction is established for the announced part.
- `reproduction-candidate` (e.g. `hoffart_euler_poisson_dsl/run.py`): must write pending. The validation table is explicitly not established; presenting it as reproduced is forbidden.
- `validation` (e.g. `euler_poisson/run.py`): must state "this is not a published reproduction". You verify invariants, not a curve from the paper.
- `tutoriel` (e.g. `composition/run.py`): the clause states "demonstrates an API capability, validates no published physical result".
- `experimental` (e.g. `schur_magnetized_cartesian/run.py`, `dsl_euler/run.py`): must flag prototype / unfinished path (interpreted DSL, platform-dependent timing measurement).

### Substantive sections, by case type

The common skeleton (after the contract):

1. Physics: the mechanism (section 5).
2. Equations solved + brick table (3 layers, section 3).
3. Math / derivation of the prediction (section 4).
4. Code, function by function, anchored to line numbers (section 3).
5. Initial conditions.
6. Figures, generated and analyzed (section 6).
7. Honest analysis of the discrepancies / limits.
8. Provenance, exact command, cost (re-detailed).

The falsifiable prediction is stated from the contract on (section 0) and the artifact that confronts it is produced in section 6. The nature of the prediction shifts by type, and the skeleton adapts:

- Physical reproduction (`diocotron`, `hoffart_*`, `plasma`, `euler_poisson` on the contrast side): prediction = a physical number (growth rate, sign and magnitude of dE). Keep all sections; section 7 (analysis of the gap to the paper / to the analytic) is the heart.

- Invariant validation (`euler_poisson`, `multispecies`, `two_euler`, `plasma`, `diocotron_amr`): prediction = a structural invariant (mass conserved, net momentum zero, opposite signs). The math derivation (4) explains why the invariant holds (periodic potential -> zero net force sum). Trimmed physics section, reinforced "why this invariant" section. Section 7 becomes "what the invariant does not capture".

- DSL-vs-native equivalence (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`): prediction = bit-for-bit path equality (`np.array_equal`). Never reproduce the physics already derived in the parent case: link (`../diocotron/`), do not copy. Sections 1/4/5 shrink to a cross-reference; the heart becomes "which core conventions are reproduced" (ExBVelocity / BackgroundDensity table anchored to `include/adc/physics/*.hpp`) and "how the bit-exact equality is verified and what a divergence would betray".

- Timing study (`schur_magnetized_cartesian`): prediction = a speedup factor (stable Schur step / stable explicit step) and the `dt*omega_c` bound. The heart is the measurement methodology (`largest_stable_dt`, geometric sweep, finite/bounded/positive density stability criterion) and the platform caveats (AOT backend, `set_source_stage` instead of `adc.Split`). No physical figure: a method / dt_stable / speedup table.

- Prototype (`dsl_euler`, `experimental` paths): prediction = "the declarative path produces a finite, coherent state", not a target number. State plainly that it is an interpreted prototype, not in CI, and what is missing to promote it.

## 3. How to treat the code

Hard rules:

- Real anchoring: cite the exact stable symbol (`run_case`, `diocotron_eigenvalue`, `largest_stable_dt`, `assert_opposite_sign`, `TOL_DE`) and its file, in the form `SYMBOL in run.py`, not a drift-prone line range. A theoretical claim with no symbol implementing it is cut. No non-trivial construct without its justification.
- Never paraphrase a trivial line (`import numpy as np`, `sys.path.insert`). You gloss only the lines that carry physics or algorithm.
- Inline vs link: you show inline the key-physics functions (the flux, the eigenvalues, the operator assembly, the diagnostic measurement) as 5-to-15-line blocks of the real `run.py`, followed by bullets explaining each non-trivial variable (`rho`, `Om`, `Lmat`, `Q`, `dE_grav`). You link (without copying) the plumbing: the `adc_cases` import try/except block, the backend fallback machinery, the argparse.
- Granularity by role: a key-physics function (e.g. `mode_l_amplitude`, `diocotron_eigenvalue`, `magnetized_model`) deserves the line-by-line comment; a plumbing function (e.g. `make_system` that only does `adc.System(...)`) deserves one sentence.
- The 3-layer "who computes what" table, mandatory for brick-based cases. Three rows, each pinned to a real symbol of `run.py`:

| run.py symbol | Layer | What happens |
|---|---|---|
| `add_block(...)` / `add_equation(...)` | Python composes and diagnoses | choice of model, scheme, integrator; reading the state |
| `models.euler_poisson(...)` / `ExBVelocity` brick / `BackgroundDensity` | compiled C++ brick | the frozen physical choice (flux, eigenvalues, elliptic RHS) |
| `assemble_rhs<Limiter,Flux>`, local Newton, system Poisson | per-cell kernel (device) | the actual computation, with no Python callback in the hot path |

For a DSL case, the middle layer is no longer a named brick but the expressions (`m.flux(...)`, `m.eigenvalues(...)`, `m.elliptic_rhs(...)`) that `adc.dsl` compiles; anchor the table on those calls.

## 4. How to treat the math

- Derive, do not assert: for a falsifiable prediction, you show the steps. Mandatory example for `diocotron`: go from the linearization $\phi'=\hat\phi(r)e^{i(m\theta-\omega t)}$ to the eigenvalue problem $\omega\mathcal{L}_m\hat\phi=(m\Omega\mathcal{L}_m+Q)\hat\phi$, then to the standard form $\omega\hat\phi=\mathcal{L}_m^{-1}(\dots)\hat\phi=M\hat\phi$, and state that `eigvals(M)` returns the spectrum. Each symbol in the formula points to the variable that computes it in `diocotron_eigenvalue` (`Om` = $\Omega(r)$, `Q` = $\frac{m}{r}\frac{dn_0}{dr}$, `Lmat` = $\mathcal{L}_m$).
- Admit cleanly: what is not re-derivable in a few lines (the exact paper convention, the $\times 2\pi/\bar\rho$ normalization) is cited with its source, not reconstructed by hand.
- Notation: GitHub LaTeX, `$...$` inline and `$$...$$` in block. French accents OK. No em-dash (U+2014); use a colon, parentheses, or periods.
- The quantitative falsifiable prediction is preferred. For `euler_poisson`, the real testable prediction from the linearization is $|dE|\propto\epsilon^2$: a log-log plot of $|dE|$ vs $\epsilon$ must have slope 2; doubling $\epsilon$ quadruples $|dE|$. This is verifiable and turns a boolean assert into a convergence curve. State it, and say what a different slope would betray (slope ~1 = spurious linear term, background `rho0` poorly subtracted; slope > 2 at large $\epsilon$ = nonlinear onset).
- Verify the sign by behavior, never by a textbook convention pasted on top. The Poisson solver (`poisson_operator.hpp`) has several sign layers plus a `GradSign` in post-processing. Writing "$-\nabla^2\phi=+4\pi G(\rho-\rho_0)$ hence attractive gravity" without checking is wrong (may yield repulsion). The physical sign is read off the assert that passes: for `euler_poisson`, `main` in run.py imposes `dE_grav < 0` (attractive) and `dE_plas > 0` (repulsive); that is the reference, not a textbook formula.
- Name the paradoxes, do not fabricate the derivation. For `euler_poisson`, $E_{tot}=U[3].sum()$ is the fluid energy alone (no field potential) and it decreases for gravity even though $v\cdot g>0$. State the tension openly and attribute it to the coupling convention, without manufacturing a boxed theorem. A wrong boxed sign is worse than an honest report.

## 5. How to treat the physics

- The mechanism before the result. For `diocotron`: the differential rotation $\Omega(r)=-\frac{1}{r^2}\int_0^r n_e r'dr'$ creates a shear, the shear is a Kelvin-Helmholtz instability of a vorticity ring ($n_e$ plays the vorticity, $\phi$ the stream function), so the ring develops $l$ lobes that roll up. The rate $\gamma_l$ comes after, as a quantified consequence.
- Relate the reduced model to the full model, explicitly. `diocotron` solves the E x B drift limit; the full magnetized Euler-Poisson system is `hoffart_euler_poisson_dsl`. State "this case reproduces only the drift limit, not the full system", with the link.
- Honesty about what is modeled: name the simplifications (a single conserved variable, no momentum nor energy for the diocotron; `four_pi_G=1` dimensionless; quasi-linear regime $\epsilon=0.01$, 20 steps, no Jeans collapse for `euler_poisson`).

## 6. Figures: which figures, how to generate them, how to analyze them

### Table by case type

| Type | Diagnostic figures to generate | What you read in them |
|---|---|---|
| Physical reproduction (rate) | `dispersion.png` (gamma vs mode: analytic + paper points + adc measurements); `amplitude.png` (semilog $|c_l|(t)$, straight line = exponential); `snapshots.png` + `*.gif` (nonlinear roll-up) | mode ranking, measured/analytic gap, visual signature with $l$ lobes |
| Invariant validation | conservation vs t (mass, momentum in absolute scale); energy contrast (dE of the two runs side by side); $|dE|$ vs $\epsilon$ convergence in log-log (expected slope 2); 2D map of the perturbation | the invariant holds to tolerance; the sign is clean and above noise; the slope confirms the regime |
| DSL-vs-native equivalence | $|state_{dsl}-state_{natif}|$ heatmap that must be identically black; residual histogram capping at ~1e-15 (or exactly 0) | a single non-black pixel = failure; the machine-level residual is the observable that proves determinism |
| Multi-species / coupled | masses per species vs t (each flat); density map per species; coupled potential $|\phi|$ | conservation per species, active Poisson coupling |
| Uniform vs AMR (`diocotron_amr`) | side-by-side uniform/AMR comparison of the same diagnostic; patch map | the conservative reflux preserves the invariant; the AMR follows the same dynamics |
| Timing (`schur`) | dt_stable vs method (bars or table); dt*omega_c; speedup | the explicit source collapses as omega_c grows; the Schur lifts the bound |

### Generation convention

- Versioned reproduction assets: only `reproduction` cases commit their figures into `<case>/figures/` with a `figures/provenance.json` (real fields: `adc_cpp_sha`, `adc_cases_sha`, `backend`, `resolution`, `nsteps_growth`, `cfl`, `python`, and the measured numbers such as `gamma_num_mesure`).
- Transient diagnostics: everything else writes under `out/<case>/` via `case_output_dir(<case>)` (see `adc_cases/common/io.py`), a git-ignored directory. Never write a throwaway diagnostic into the source tree.
- Workflow for a case with no figure today: state explicitly which figure to generate (run + plot + commit), with the exact command and the location (`out/<case>/` to explore, `<case>/figures/` only if the case becomes a versioned reproduction).

### Analysis rules

- Each figure is embedded (`![meaningful alt](figures/xxx.png)`) then followed by 2 to 4 sentences that interpret what it shows physically. Never an empty caption ("here is the density").
- Partition the reading into Proves / Suggests / Not shown:
  - Proves: what an assert tests (opposite signs of dE, flat mass, bit-exact equality).
  - Suggests (not asserted): what is plausible to the eye but not tested (e.g. the ~5 % gravity/plasma mirror symmetry is visible but no assert checks it; state it as a suggestion).
  - Not shown: what the figure does not cover (no nonlinear dynamics over 20 steps; no Jeans collapse).
- Diagnostic reading, not decorative. A slope != 2 on $|dE|$ vs $\epsilon$ betrays (slope ~1 = spurious linear, slope > 2 = nonlinear). On an equivalence heatmap, state what a non-black spot would signal (a DSL formula that diverges from a core brick, e.g. a wrong sign convention in `eigenvalues` or `elliptic_rhs`).
- Provenance on every cited number: SHA, backend, resolution, measured cost (not estimated), plus the platform caveat: the signs and the order of magnitude are stable, the last digits vary with the BLAS library and the summation order. Cite the real numbers from the run (the `gamma_num_mesure` from `provenance.json`, not invented values).

## 7. Anti-bullshit: hard rules

- Deletion test: if removing a sentence loses neither a fact, a number, a symbol, a sign, nor a reason, it goes.
- Zero recap. No "In conclusion" section that re-states the intro. The current `diocotron` exemplar has sections 8 (Architecture) and 12 (Limits) that re-state sections 1 through 7; that budget is reclaimed for the derivation and the figure analysis, at near-constant length. One fact, at a single altitude.
- A theoretical claim without the line of code that implements it is cut.
- A tolerance is a clause justified by an order of magnitude, never a posited constant. `TOL_DE=1e-5` sits between machine noise (dE = 0 exactly at $\epsilon=0$) and the expected physical magnitude ~6e-4 (`TOL_DE` in run.py): write that ratio. `TOL_MASS=1e-9` because the scheme is conservative and the drift comes from floating-point arithmetic. Each tolerance has its "why".
- Always distinguish Proves (by an assert) from Suggests (made plausible by a figure). Anti-over-interpretation guardrail, to apply to all 15 cases.
- For DSL child cases: link, do not copy the parent's physics.

Blacklist (forbidden words/phrasings), with correction:

- "powerful / seamless / leverage / robust (decorative) / elegant" -> remove or replace with the fact. Before: "adc composes powerfully and elegantly." After: "adc.System composes one block per `add_block`, each block freezes its scheme in C++ at insertion."
- Decorative rule of three -> cut the empty triad. Before: "fast, reliable, and extensible." After: "~60 s on one CPU core (3 modes x 900 steps at $192^2$)."
- Empty hedging ("it should be noted that", "in some sense", "overall") -> remove.
- Promotional tone ("this case highlights the richness of the solver") -> replace with the statement of what is tested. After: "this case verifies by assert: mass conserved to 1e-9, net momentum < 1e-8, opposite dE signs."
- Category over-selling. Before: "reproduction of the Hoffart benchmark." After (if `reproduction-candidate`): "targets arXiv:2510.11808; quantitative reproduction pending (validation table not established)."
- Empty emphasis adverb ("clearly", "obviously", "notably" at the opening) -> remove.

## 8. Length and density

Target per type (text outside code blocks):

- Physical reproduction: 350 to 550 lines. It is the longest; the derivation and the analysis of the 3 to 4 figures justify it.
- Invariant validation: 180 to 320 lines. Centered on "why the invariant holds" and the $\epsilon^2$ prediction.
- DSL-vs-native equivalence: 120 to 220 lines. Short by construction: the physics is linked to the parent, the heart is the conventions table and the bit-exact equality.
- Timing: 150 to 250 lines. The measurement methodology and the platform caveats dominate.
- Prototype: 100 to 180 lines. You state what it is, what is missing, you do not over-build.

Density test: no sentence is removable without losing a fact, a number, a symbol, a sign, or a reason. If a section exceeds the target, it is almost always a recap to cut, not substance to add.

## 9. README validation checklist

Binary, to tick before acceptance:

1. [ ] The Contract block is at the top, with the exact category from the manifest.
2. [ ] The Does not prove clause is as detailed as Proves and its tone follows the category (pending if `reproduction-candidate`, "not a published repro" if `validation`, prototype if `experimental`).
3. [ ] Each substantive section names the contract clause it justifies; no orphan clause.
4. [ ] A falsifiable prediction is stated in the contract and an artifact (figure/assert/table) confronts it.
5. [ ] Each theoretical claim points to a real stable symbol (`SYMBOL in run.py`); no trivial line is paraphrased.
6. [ ] The 3-layer table (Python composes / brick freezes / per-cell kernel) is present for a brick-based case, each row pinned to a real symbol.
7. [ ] The physical signs are verified by the asserted behavior, not by a pasted textbook convention.
8. [ ] Each tolerance is justified by an order of magnitude (noise / physical magnitude ratio).
9. [ ] Each figure is embedded and followed by 2 to 4 sentences of analysis, partitioned Proves / Suggests / Not shown.
10. [ ] Transient diagnostics go into `out/<case>/` (via `case_output_dir`); only versioned reproduction figures are in `<case>/figures/` with `provenance.json`.
11. [ ] Each cited number has its provenance (SHA, backend, resolution, measured cost) and the platform caveat (signs/order of magnitude stable, last digits variable).
12. [ ] No recap section; no fact stated twice at two altitudes.
13. [ ] No word from the blacklist; the deletion test passes on every sentence.
14. [ ] The exact launch command, the prerequisites, and the measured cost are given.
15. [ ] For a DSL child case: the parent's physics is linked, not copied; the heart is the core conventions table (`include/adc/physics/*.hpp`) and the bit-exact equality (`np.array_equal`).
