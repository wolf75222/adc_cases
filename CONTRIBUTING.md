# Contributing to adc_cases

`adc_cases` holds the Python use cases of the `adc` solver. The generic physics and the
cell-by-cell compute live in [adc_cpp](https://github.com/wolf75222/adc_cpp) and are exposed by
its `adc` module (pybind11). Here, Python composes and drives: one folder per case, each
importing `adc`, writing its initial conditions in numpy, and running the simulation.

## Running a case

The `adc` module is built from `adc_cpp` (Kokkos-only; see the README for the build). CI clones
adc_cpp, builds the module and runs the cases. Locally, build the module, then run a case:

```bash
python <case>/run.py        # most cases take a reduced/quick mode for a fast smoke run
```

## Workflow

- **Linear** is the source of truth for tasks: one `ADC-NN` issue = one PR.
- Branch: `adc-<n>-short-description`. PR title: `ADC-<n> Description`. PR body: `Fixes ADC-<n>`.
- `master` is the default branch; never commit directly to it. Deliver through a branch or an
  isolated `git worktree` off `master`.
- Keep PRs focused on one logical change. For a physics change, include numerical validation
  (case, observed quantity, expected value, tolerance, reason) so a reviewer can tell a normal
  difference from a real change.

## Guardrails

- **No AI author, committer or co-author** (Claude, Copilot, Anthropic, ...) anywhere in the
  history: the `no-ai-authors.yml` workflow rejects such commits at the source (the GitHub squash
  hoists `Co-authored-by` trailers). Use your default git identity.
