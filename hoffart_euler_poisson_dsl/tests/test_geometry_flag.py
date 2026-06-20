#!/usr/bin/env python3
"""Test for the --geometry {square,staircase,cutcell} flag of run.py, against the REAL adc.

No fake/mock adc: this compiles the real magnetic Euler-Poisson model once, drives the real
``run.build_uniform`` for each geometry, and asserts on the real System's queryable disc mask
(``sim.disc_mask()``):

  - 'square' (default)      -> the mask stays FULL (n*n active): historical bit-identical
                               full-square Cartesian transport, no set_disc_domain.
  - 'staircase' / 'cutcell' -> the mask is RESTRICTED to the disc centered at (L/2, L/2) with
                               radius R (the same circle as the Poisson wall).
  - an unknown geometry     -> ValueError.
  - --geometry staircase with --engine amr-imex -> rejected at the argument layer (SystemExit).

Needs the real ``adc`` importable AND compile-capable (PYTHONPATH=<adc_cpp>/build/python,
ADC_INCLUDE=<adc_cpp>/include ; KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 for Kokkos-OpenMP).

Run standalone (`python3 test_geometry_flag.py`) or under pytest.
"""

import os
import sys

import numpy as np

import adc  # noqa: F401  (real extension)

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:        # case root: model.py, run*.py
    sys.path.insert(0, os.path.dirname(HERE))
import run  # noqa: E402
from model import (  # noqa: E402
    PaperParameters,
    magnetic_euler_poisson_model,
    paper_initial_density,
)

N = 16
_COMPILED = None


def _compiled():
    """Compile the real model once (DSL-cached); reused across geometries."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = magnetic_euler_poisson_model(PaperParameters(), source="schur").compile(
            backend="production", target="system", name="hoffart_schur")
    return _COMPILED


def _build(geometry):
    p = PaperParameters()
    sim = run.build_uniform(_compiled(), paper_initial_density(N, 3, p), p, geometry=geometry)
    return sim, p


def test_square_keeps_full_cartesian_mask():
    sim, _ = _build("square")
    mask = np.asarray(sim.disc_mask())
    assert float(mask.sum()) == N * N, (
        "square must keep the FULL Cartesian transport (no disc mask), got %g active of %d"
        % (float(mask.sum()), N * N))


def _assert_disc_restricted(geometry):
    sim, p = _build(geometry)
    mask = np.asarray(sim.disc_mask()).reshape(N, N)
    active = float(mask.sum())
    assert active < N * N, "%s must restrict transport to the disc (active < n^2)" % geometry
    # the active cells must lie inside the disc centered at (L/2, L/2), radius R = the wall radius
    h = p.length / N
    xc = (np.arange(N) + 0.5) * h - 0.5 * p.length
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    inside = np.hypot(X, Y) <= p.radius + h          # disc + one cell (staircase over-approximates)
    stray = np.logical_and(mask > 0.5, ~inside)
    assert not stray.any(), (
        "%s active cells must lie within the disc R=%g centered at L/2 (%d stray)"
        % (geometry, p.radius, int(stray.sum())))
    # count is close to the disc area pi R^2 / h^2 (staircase/cutcell over-approximate slightly)
    assert active <= 1.3 * np.pi * p.radius ** 2 / h ** 2, "%s mask too large vs disc area" % geometry


def test_staircase_restricts_transport_to_disc():
    _assert_disc_restricted("staircase")


def test_cutcell_restricts_transport_to_disc():
    _assert_disc_restricted("cutcell")


def test_unknown_geometry_raises():
    p = PaperParameters()
    raised = False
    try:
        run.build_uniform(_compiled(), paper_initial_density(N, 3, p), p, geometry="hexagon")
    except ValueError as exc:
        raised = True
        assert "square" in str(exc) or "staircase" in str(exc), str(exc)
    assert raised, "an unknown geometry must raise ValueError"


def test_staircase_rejected_for_amr_engine():
    argv = sys.argv
    sys.argv = ["run.py", "--engine", "amr-imex", "--geometry", "staircase",
                "--acknowledge-amr-approximation"]
    try:
        raised = False
        try:
            run.main()
        except SystemExit as exc:
            raised = True
            # main() raises SystemExit with the explanatory message before any build/compile.
            assert "staircase" in str(exc), "expected staircase rejection, got %r" % (exc,)
        assert raised, "staircase + amr-imex must be rejected at the argument layer"
    finally:
        sys.argv = argv


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("all %d geometry-flag tests passed (real adc)" % len(tests))


if __name__ == "__main__":
    _run_all()
