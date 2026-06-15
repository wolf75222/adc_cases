# docs/: analysis and audit notes

In-depth documentation for the `hoffart_euler_poisson_dsl` case. These files are not
executable: they record the reasoning, the normalization audits, and the detailed results.
The readable summary lives in the [case README](../README.md).

| File | Contents |
|---|---|
| `NORMALIZATION.md` | The `2pi/rhobar` normalization of the reduced ExB polar path: where the 2pi factor comes from, and the exact l=4 validation of the reduced polar path. |
| `T2_NORMALIZATION_AUDIT.md` | Detailed dimensional audit: scales, normalization candidates, and the `window x 2pi x grid residual` decomposition. Settles the "geometry vs metrology" question (the deficit was metrology, not geometry). |
| `RESULTS_SYSTEM_SCHUR.md` | Full log: the `system-schur` growth-rate table, the T2 audit, the T3 code, convergence, and the history of the reversals (the "-95%" deficit -> reproduced to within 10%). |
