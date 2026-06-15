# slurm/: ROMEO campaigns

SLURM batch scripts for the measurement campaigns on ROMEO (URCA, x64cpu partition). They drive
`run.py` / `run_polar.py` (at the case root); their internal paths are **absolute**
(`${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl/...`), so this move leaves them unchanged.

| File | Runs | Purpose |
|---|---|---|
| `campaign_geometry.sbatch` | `run.py --geometry {square,staircase}` | Geometry discriminant: is the Cartesian ring boundary the bottleneck of the measured rate? |
| `campaign_polar.sbatch` | `run_polar.py` (frozen-equilibrium) | Full path on a polar grid (resolved ring), l=3,4,5, nr=ntheta=256, t_end=10. |

See the [case README](../README.md), section "Performance and scaling".
