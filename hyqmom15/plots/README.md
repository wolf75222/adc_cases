# hyqmom15 plots (RieMOM2D_Electrostatic_periodic figures, ADC-377)

Matlab-like figures for the five cases (`dicotron`, `fluid_wave`,
`electrostatic_wave`, `magnetic_wave`, `constant`), rendered from ADC simulation
snapshots without re-running any simulation, so the figures are versionable
report artefacts.

## Snapshot format (reused, not invented)

A snapshot is exactly what `adc.System.write(format="npz")` writes, so the ROMEO
campaign (ADC-376) just calls `sim.write` per saved step. Each `.npz` has:

| key | shape | meaning |
|-----|-------|---------|
| `t` | scalar | simulation time |
| `macro_step` | scalar | step counter |
| `nx`, `ny` | scalar | grid resolution |
| `blocks` | `(nblocks,)` | block names; hyqmom15 uses `mom` |
| `state_mom` | `(15, ny, nx)` | the 15 moments `M00..M04` |
| `names_mom` | `(15,)` | moment names |
| `phi` | `(ny, nx)` | Poisson potential (zeros when no field is active) |

The domain is `[-0.5, 0.5]^2` periodic (`init_domain.m`). A campaign directory
holds one sub-directory per case, each with its snapshots and an optional
`run_meta.json` provenance sidecar (case, `Np`, params, commits, backend):

```
<campaign>/
  dicotron/        step_000000.npz ... run_meta.json
  electrostatic_wave/ ...
  figures/         <- written here
```

## Render

```bash
# all cases in a campaign directory, with density animations
python3 hyqmom15/plots/plot_rie_mom2d_case.py <campaign_dir> --gif

# one case, custom output directory
python3 hyqmom15/plots/plot_rie_mom2d_case.py <campaign_dir> --case dicotron --out /tmp/figs
```

Per case it writes `<case>_density.png` (M00 at key times, shared colour
limits), `<case>_phi.png` (potential, skipped when no Poisson field),
`<case>_diagnostics.png` (relative mass drift, M00 min/max positivity, dt over
time), and, with `--gif`, a `<case>_density.gif` animation (PillowWriter; no
ffmpeg needed).

`matplotlib` is required for rendering but is not a CI dependency, so the
renderer is a manual tool like the other `make_figures.py` scripts. The
pure-NumPy loader and diagnostics (`snapshots.py`) and their guard
`check_plots.py` run build-free.

## Vocabulary (ADC-378)

Figures annotate differences precisely and never call the `D` / `Dmax` split a
"Matlab bug": `D` is the physical wave initialization, `Dmax` is the CFL
max-speed bound. That is a clarified convention. Genuine open items (the
`init_magnetic_wave` wiring, the legacy diocotron drift) are labelled true
divergences.
