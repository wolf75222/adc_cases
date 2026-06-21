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

```bash
# smoke (reduced Np, capped steps) for a quick local/CI-style check
python3 hyqmom15/campaigns/romeo_rie_mom2d.py --smoke --out out/campaign --threads 8

# full Matlab resolution on ROMEO
sbatch hyqmom15/campaigns/romeo_rie_mom2d.sbatch

# structure only, no adc (what CI exercises)
python3 hyqmom15/campaigns/romeo_rie_mom2d.py --dry-run --out /tmp/campaign
```

Each case directory gets `adc.System.write(format="npz")` snapshots, a
`run_meta.json` provenance sidecar (case, params, solver config, backend,
threads, wall-clock, dt range, AMR flag, commits, host), and feeds a
`synthesis.md` table (mass conservation, M00 positivity, realizability, dt,
runtime, status). The figures come from `hyqmom15/plots` (ADC-377/384).

## Matlab speedup baseline (Octave, run where the Matlab source lives)

```bash
python3 hyqmom15/campaigns/octave_matlab.py <RieMOM2D_src_dir> --out matlab_times.json
# then fold the timings into the campaign synthesis:
python3 hyqmom15/campaigns/romeo_rie_mom2d.py --full --matlab-times matlab_times.json ...
```

`CASE_SCRIPTS` in `octave_matlab.py` maps each case to its Matlab entry script;
adjust it to the local Matlab layout. `speedup = matlab / adc` (>1 means ADC is
faster).
