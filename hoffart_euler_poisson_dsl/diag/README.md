# diag/: diagnostics

Diagnostic scripts, **outside the manifest** (these are not cases): they shed light on the
normalization, the analytic spectrum, the convergence, and they generate the figures. Run them by
hand from the case root.

| File | Role |
|---|---|
| `petri_eigenvalue.py` (+ `petri_eigenvalue.md`) | Analytic Davidson/Petri eigenvalue: derives the targets `0.772 / 0.911 / 0.683` and the origin of the 2pi factor (numpy only, no dependency on the `adc` engine). |
| `diag_normalization_audit.py` | Runnable dimensional audit (scales, normalization candidates, window decomposition). Companion to [`../docs/T2_NORMALIZATION_AUDIT.md`](../docs/T2_NORMALIZATION_AUDIT.md). |
| `diag_polar_omega.py` | **Reduced** scalar ExB polar path: validates the `2pi/rhobar` normalization (recovers l=4 exactly). |
| `convergence_reduced.py` | Resolution convergence: the relative error to the paper tends to 0 as `n` grows. |
| `make_paper_figures.py` | Generator for the paper-style figures and GIFs (schlieren snapshots, growth rate, rollup animations). |

See the [case README](../README.md).
