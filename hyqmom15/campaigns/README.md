# hyqmom15 ROMEO campaign (RieMOM2D_Electrostatic_periodic, ADC-376)

Run the five Matlab reference cases at full resolution, capture exploitable
artefacts (snapshots + provenance + realizability monitoring), and compare to
the Matlab reference (correctness + speedup). The ROMEO run is launched only on
Romain's go; the code here is build-free where it can be, validated in CI via
`check_campaign.py` (the orchestrator's `--dry-run` plus the table/meta helpers).

## Cases

| case | Np | scheme | sources | notes |
|---|---|---|---|---|
| `dicotron` | 128 | HLL, Euler | E + B (oc=-20) | Poisson periodic, `compute_dt` |
| `fluid_wave` | 32 | ROE, Euler | none | no Poisson (ADC-371) |
| `electrostatic_wave` | 128 | HLL, Euler | E (Poisson) | `compute_dt` source cap |
| `magnetic_wave` | 256 | HLL/MUSCL, Euler | E + B (Poisson) | |
| `constant` | 64 | HLL/MUSCL, Euler | none | uniform non-regression |

`D` / `Dmax` is a clarified convention (D = physical init, Dmax = CFL), not a
Matlab bug (ADC-378); the campaign reports divergences with that vocabulary.

## Run (ADC, on ROMEO)

PREREQUISITE for the full run: `_adc` rebuilt from the current `adc_cpp` master.
The stock ROMEO build may predate the ADC-368 ROE hook that `fluid_wave` needs;
rebuild via the adc_cpp ROMEO recipe (`cmake -S adc_cpp -B build-mpik && make _adc
-j32`) before submitting.

```bash
# smoke (reduced Np, capped steps) for a quick local/CI-style check
python3 hyqmom15/campaigns/romeo_rie_mom2d.py --smoke --out out/campaign --threads 8

# structure only, no adc (what CI exercises)
python3 hyqmom15/campaigns/romeo_rie_mom2d.py --dry-run --out /tmp/campaign

# full Matlab resolution on ROMEO -- runs the 5 cases, then export_h5 + make_rapport
sbatch hyqmom15/campaigns/romeo_rie_mom2d.sbatch
```

Each case directory gets `adc.System.write(format="npz")` snapshots, a
`run_meta.json` provenance sidecar (case, params, solver config, backend,
threads, wall-clock, dt range, AMR flag, commits, host), and feeds a
`synthesis.md` table (mass conservation, M00 positivity, realizability, dt,
runtime, status). Post-process into exploitable artefacts:

```bash
python3 hyqmom15/campaigns/export_h5.py     out/campaign  # -> out/campaign/h5/<case>.h5
python3 hyqmom15/campaigns/make_rapport.py  out/campaign  # -> out/campaign/rapport.md
python3 hyqmom15/campaigns/to_paraview.py   out/campaign  # -> out/campaign/paraview/<case>.pvd
python3 hyqmom15/plots/diagnostics_plots.py out/campaign  # -> out/campaign/figures/ (matplotlib)
```

`export_h5.py` packs each case into one HDF5 (time axis, moment fields, potential,
per-snapshot realizability series, full provenance as attributes; skipped if h5py
is absent). `make_rapport.py` writes the per-case analysis report (config,
realizability verdict, conservation, positivity, symmetry, figure list, HDF5 ref)
plus a synthesis table and the optional speedup. Figures come from `hyqmom15/plots`
(ADC-377/384).

**ParaView** comes two ways. The run already writes the **native adc_cpp VTK output**
(`adc.System.write(format="vtk")`): one `step_NNNNNN.vti` (ImageData, CellData
`mom_<moment>` + `phi`) per snapshot in each case dir, opened directly by ParaView /
VisIt -- open the `step_*.vti` series for the animation. `to_paraview.py` is an
optional **enriched** export: from the npz it adds physics-ready cell fields (density,
ux/uy + speed, the realizability margin `lam_min`) and a `<case>.pvd` time collection
(real `t` values) under `paraview/`. The runs are single-rank, so each snapshot is one
grid; parallel pieces (`.pvti`) would only apply to an MPI-distributed run. The enriched
exporter has no VTK/pyvista dependency (hand-written VTK XML).

## Matlab speedup baseline (Octave, run where the Matlab source lives)

```bash
python3 hyqmom15/campaigns/octave_matlab.py <RieMOM2D_src_dir> --out matlab_times.json
# then fold the timings into the report:
python3 hyqmom15/campaigns/make_rapport.py out/campaign --matlab-times matlab_times.json
```

The reference is a single `main.m` parameterized by an internal `case_name`;
`octave_matlab.py` times each case by rewriting that line in a temp copy and
running it under Octave (`speedup = matlab / adc`, >1 means ADC is faster).
Caveat: under Octave the reference crashes in `eigenvalues15_2D` ("matrix contains
Inf or NaN") -- the D7 corner artifact, since Octave's `eig` is stricter than
Matlab's. The speedup baseline therefore needs Matlab proper (or a corner-state
guard); crashing cases report `n/a`.
