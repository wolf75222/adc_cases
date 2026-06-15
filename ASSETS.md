# Asset manifest (adc_cases)

Provenance and reproducibility of every **versioned** figure/GIF in the repo. Rule: every committed
asset must carry a `provenance.json` next to it (`adc_cpp` SHA + `adc_cases` SHA, backend,
resolution, command, parameters) and be regenerable **in place** by its command.

## Versioned assets (committed)

| Asset | Producer | Provenance | Regenerate |
|---|---|---|---|
| `diocotron/figures/dispersion.png` | `diocotron/run.py` (analytic Petri + adc measurement) | `diocotron/figures/provenance.json` | `python diocotron/run.py` |
| `diocotron/figures/amplitude.png` | `diocotron/run.py` (\|c_l\|(t), modes 3/4/5) | same | same |
| `diocotron/figures/snapshots.png` | `diocotron/run.py` (4 snapshots, mode l=4) | same | same |
| `diocotron/figures/diocotron.gif` | `diocotron/run.py` (`run_evolution(l=4)`) | same | same |

`diocotron/run.py` now writes its figures directly into `diocotron/figures/` (tracked) and drops a
`provenance.json` next to them: a re-run **refreshes the assets in place** (no more manual copy from
`out/`, which was the source of drift). Cost ~60 s (n=192, modes 3/4/5, serial CPU). The current
`provenance.json` records in particular: `adc_cpp_sha`, `adc_cases_sha`, `backend = native serial`,
`resolution = 192x192`, and the measured rates `gamma_num` (l=3 ~0.599, l=4 ~0.662, l=5 ~0.652,
that is -22/-27/-5 % vs the analytic oracle, see `diocotron/README.md`, "Limitations" section).

## Ephemeral assets (not committed, written under `out/`, gitignored)

- **`hoffart_euler_poisson_dsl/run.py`** writes its figures (amplitude, snapshots, growth_rates, gif)
  under `out/<engine>/...`. They are **not committed** and must not be: this case is
  `reproduction-candidate` **pending** (the quantitative reproduction of arXiv:2510.11808 is not
  established, see `hoffart_euler_poisson_dsl/README.md` and `adc_cpp/docs/HOFFART_FIDELITY.md`).
  Committing these figures would suggest a validated reproduction. The `amr-imex` variant also
  requires an MPI / multi-GPU build (ROMEO/GH200), out of reach for a local machine (Kokkos itself,
  now mandatory, stays accessible locally through a Kokkos Serial install).
- The DSL and validation cases (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`,
  `two_fluid_ap`, `schur_magnetized_cartesian`, ...) write their `.so`/`.csv` under `out/`
  (gitignored): build/measurement artifacts, not versioned.

## Cases without assets

`composition`, `custom_scheme`, `diocotron_amr`, `dsl_euler`, `euler_poisson`, `multispecies`,
`plasma`, `two_euler` produce **textual diagnostics** (invariants via `assert`), no figure. See their
`README.md` ("Expected outputs" section).

## On the `adc_cpp` side

The `adc_cpp` tutorial (`docs/sphinx/tutorials/diocotron_tutorial.py`) produces its own assets
(`docs/sphinx/tutorials/_assets/`) with their `provenance.json`; see `adc_cpp/docs/ASSETS.md`.
