# matlab_ref goldens (ADC-350)

Octave-generated reference data that locks the linear-algebra and initial-condition
layer of `hyqmom15/matlab_ref/` against the new periodic Matlab reference
`RieMOM2D_Electrostatic_periodic`. The Python layer is compared to these files by
[../check_goldens.py](../check_goldens.py) (manifest `ci=true`, build-free).

## Provenance

- Source: `RieMOM2D_Electrostatic_periodic` (maintainer-side, not vendored, see
  [../REFERENCE.md](../REFERENCE.md)). On the reference machine:
  `/Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic`. The tree is
  not under version control; the maintainer comparison and Sacha's clarifications
  (2026-06-19) in `RIEMOM2D_vs_RieMOM2D_Electrostatic_periodic_comparaison_detaillee.md`
  are the reference anchor.
- Generator: GNU Octave 11.3.0.
- Case parameters mirror [../params.py](../params.py) (= the live `init_*.m`).

## Regenerate

From the adc_cases repo root, with the `-p` path pointing at the Matlab source:

```bash
MAT=/Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic
octave --no-gui -p "$MAT" hyqmom15/matlab_ref/golden_linearized_gen.m
octave --no-gui -p "$MAT" hyqmom15/matlab_ref/golden_init_gen.m
octave --no-gui -p "$MAT" hyqmom15/matlab_ref/golden_dt_gen.m
python3 hyqmom15/matlab_ref/check_goldens.py   # 0 = layer reproduces the goldens
```

All CSVs are `%.17g`, no header.

## Files

| File | From | Content |
|---|---|---|
| `lin_<case>_jac.csv` | `golden_linearized_gen.m` | 15x15 Jacobian, columns `[real | imag]` |
| `lin_<case>_eigvals.csv` | same | Matlab-sorted eigenvalues, `[real, imag]` |
| `lin_<case>_eigvec.csv` | same | phase-pinned mode-15 eigenvector, `[real, imag]` |
| `maxspeed.csv` | same | rows: es intended (diag Dmax), es as_written (diag D), diocotron (real-part sort), `[real, imag]` |
| `init_<tag>.csv` | `golden_init_gen.m` | IC field at `Np=16`, block-stacked `(15*Np) x Np` (block `k` = moment `k`) |
| `dt.csv` | `golden_dt_gen.m` | rows `[case_index, vmax, t, dt]` from the actual `compute_dt.m` |

`<case>` in {`fluid_wave`, `electrostatic_wave`, `magnetic_wave`}. `<tag>` adds
`magnetic_wave_aswritten`, `dicotron_standard`, `dicotron_matlab_bug`, `constant`.
`dt.csv` case order: fluid_wave, electrostatic_wave, magnetic_wave, dicotron, constant.

## as_written vs intended

Per REFERENCE.md D3/D4 the intended path is the ADC default; the legacy bug is kept
under a named variant so the divergence is documented, not silently transcribed:

- `maxspeed.csv` row 1 (`intended`, `diag(Dmax)` at `(kmin,kmin)`) vs row 2
  (`as_written`, the `diag(D)` bug at the mode wavenumber).
- `init_magnetic_wave.csv` (intended, magnetostatic Jacobian) vs
  `init_magnetic_wave_aswritten.csv` (the `init_magnetic_wave.m` oversight wiring
  the electrostatic field).
- `init_dicotron_standard.csv` (corrected incompressible ExB) vs
  `init_dicotron_matlab_bug.csv` (the transposed meshgrid drift).

## Scope and resolution

- `Np=16`: the IC recipe is local for the waves/constant and the diocotron pipeline
  (ring, periodic FFT Poisson, ExB drift, Maxwellian) is `Np`-parametrized, so this
  small grid validates the implementation while keeping the goldens tiny. Per-driver
  full-`Np` fields are produced in the driver PRs (ADC-351+).
- Tolerances (`check_goldens.py`): Jacobians bit-identical (`atol 1e-12`),
  eigenvalues / eigenvectors / max_speed `atol 1e-9`, IC fields `atol 1e-11`
  (FFT Poisson round-off for the diocotron), dt `atol 1e-15`.
- Source-term and one-step goldens validate the native adc solver, not this Python
  layer; they are produced and consumed in the wave-case PRs (ADC-352/353/354)
  against the compiled model, to avoid re-coding the solver in Python.
