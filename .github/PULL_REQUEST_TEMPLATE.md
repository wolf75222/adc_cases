<!-- Title: ADC-NN short imperative description. One Linear issue = one PR. -->

Fixes ADC-NN

## What / why

<!-- What the case or facade changes and, above all, WHY. State the assumptions. -->

## How

<!-- Key points: which case(s) or shared module, what was preserved. -->

## Validation

<!-- Tick what was run; paste the commands, not "it runs". CI clones adc_cpp,
     builds the `adc` module (Kokkos Serial) and runs the cases. -->

- [ ] Ran the relevant case(s) locally (e.g. `python <case>/run.py`, reduced/quick mode)
- [ ] CI green (module build + cases)
- [ ] No regression on the touched case(s)

## Numerical validation (physics changes)

<!-- Remove this block for non-numerical PRs. Without it a reviewer cannot tell a
     normal difference from a real change. -->

- Case:
- Observed quantity:
- Expected value:
- Tolerance (and reason):
- Measured difference:

## Risks / attention

<!-- Tolerances, expected drift, safeguards, follow-up. -->

<!-- Before squash-merge: no AI author/committer/co-author in master..branch
     (no-ai-authors guard); delete the source branch on merge. -->
