# Manifeste des assets (adc_cases)

Provenance et reproductibilite de chaque figure/GIF **versionne** du depot. Regle : tout asset
committe doit porter un `provenance.json` a cote (SHA `adc_cpp` + SHA `adc_cases`, backend,
resolution, commande, parametres) et etre regenerable **en place** par sa commande.

## Assets versionnes (committes)

| Asset | Producteur | Provenance | Regenerer |
|---|---|---|---|
| `diocotron/figures/dispersion.png` | `diocotron/run.py` (analytique Petri + mesure adc) | `diocotron/figures/provenance.json` | `python diocotron/run.py` |
| `diocotron/figures/amplitude.png` | `diocotron/run.py` (\|c_l\|(t), modes 3/4/5) | idem | idem |
| `diocotron/figures/snapshots.png` | `diocotron/run.py` (4 instantanes, mode l=4) | idem | idem |
| `diocotron/figures/diocotron.gif` | `diocotron/run.py` (`run_evolution(l=4)`) | idem | idem |

`diocotron/run.py` ecrit desormais ses figures directement dans `diocotron/figures/` (tracke) et
depose `provenance.json` a cote : une re-execution **rafraichit les assets en place** (plus de copie
manuelle depuis `out/`, qui etait la source de derive). Cout ~60 s (n=192, modes 3/4/5, CPU serie).
Le `provenance.json` courant enregistre notamment : `adc_cpp_sha`, `adc_cases_sha`, `backend = natif
serie`, `resolution = 192x192`, et les taux mesures `gamma_num` (l=3 ~0.599, l=4 ~0.662, l=5 ~0.652,
soit -22/-27/-5 % vs l'oracle analytique -- cf. `diocotron/README.md`, section « Limites »).

## Assets ephemeres (non committes, ecrits sous `out/`, gitignore)

- **`hoffart_euler_poisson_dsl/run.py`** ecrit ses figures (amplitude, snapshots, growth_rates, gif)
  sous `out/<engine>/...`. Elles ne sont **pas committees** et ne doivent pas l'etre : ce cas est
  `reproduction-candidate` **pending** (la reproduction quantitative d'arXiv:2510.11808 n'est pas
  etablie, cf. `hoffart_euler_poisson_dsl/README.md` et `adc_cpp/docs/HOFFART_FIDELITY.md`).
  Committer ces figures laisserait croire a une reproduction validee. La variante `amr-imex` exige
  en plus un build Kokkos/MPI (ROMEO/GH200) -- hors de portee d'un poste local.
- Les cas DSL et de validation (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`,
  `two_fluid_ap`, `schur_magnetized_cartesian`, ...) ecrivent leurs `.so`/`.csv` sous `out/`
  (gitignore) : artefacts de build/mesure, non versionnes.

## Cas sans asset

`composition`, `custom_scheme`, `diocotron_amr`, `dsl_euler`, `euler_poisson`, `multispecies`,
`plasma`, `two_euler` produisent des **diagnostics textuels** (invariants par `assert`), pas de
figure. Voir leur `README.md` (section « Sorties attendues »).

## Cote `adc_cpp`

Le tutoriel `adc_cpp` (`docs/sphinx/tutorials/diocotron_tutorial.py`) produit ses propres assets
(`docs/sphinx/tutorials/_assets/`) avec leur `provenance.json` ; voir `adc_cpp/docs/ASSETS.md`.
