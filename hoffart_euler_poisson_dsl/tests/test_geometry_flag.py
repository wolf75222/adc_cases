#!/usr/bin/env python3
"""Smoke test for the --geometry {square,staircase,cutcell} plumbing of the hoffart case.

This test does not need the heavy Kokkos/AMReX `adc` extension: it installs a tiny
fake `adc` module that records every method call on the fake `System`, then drives
`build_uniform` for all three geometries. The assertions are real:

  - 'square' (default) never calls set_disc_domain    -> bit-identical historical path
  - 'staircase' calls set_disc_domain(L/2, L/2, R, mode='staircase')  -> T2 disc mask
  - 'cutcell'  calls set_disc_domain(L/2, L/2, R, mode='cutcell')     -> EB cut-cell mask
  - an unknown geometry raises ValueError
  - the argparse layer rejects --geometry staircase with --engine amr-imex

Run standalone (`python3 test_geometry_flag.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _install_fake_adc() -> types.ModuleType:
    """Register a fake `adc` module that records System calls. Returns the module."""
    adc = types.ModuleType("adc")

    class FakeSystem:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []  # ordered list of (method, args, kwargs)
            self._n = kwargs["n"]

        def _record(self, name, *a, **k):
            self.calls.append((name, a, k))

        def set_poisson(self, *a, **k):
            self._record("set_poisson", *a, **k)

        def set_disc_domain(self, cx, cy, R, mode="staircase"):
            self._record("set_disc_domain", cx, cy, R, mode=mode)

        def set_magnetic_field(self, *a, **k):
            self._record("set_magnetic_field", *a, **k)

        def add_equation(self, *a, **k):
            self._record("add_equation", *a, **k)

        def set_primitive_state(self, *a, **k):
            self._record("set_primitive_state", *a, **k)

        def solve_fields(self, *a, **k):
            self._record("solve_fields", *a, **k)

        def potential(self):
            self._record("potential")
            return np.zeros((self._n, self._n), dtype=np.float64)

    adc.System = FakeSystem
    # Recipe stand-ins: only need to be callable and return a marker.
    adc.FiniteVolume = lambda **k: ("FiniteVolume", k)
    adc.Split = lambda **k: ("Split", k)
    adc.Strang = lambda **k: ("Strang", k)
    adc.Explicit = lambda **k: ("Explicit", k)
    adc.CondensedSchur = lambda **k: ("CondensedSchur", k)
    # model.py does `from adc import dsl` at import time. We only build a model when
    # actually running the engine, which this smoke test never does, so a placeholder
    # submodule is enough to satisfy the import.
    dsl = types.ModuleType("adc.dsl")
    adc.dsl = dsl
    sys.modules["adc"] = adc
    sys.modules["adc.dsl"] = dsl
    return adc


def _import_run():
    case_root = os.path.dirname(
        HERE
    )  # tests/ -> la racine du cas (model.py, run*.py)
    if case_root not in sys.path:
        sys.path.insert(0, case_root)
    # Make `from adc_cases.common.io import case_output_dir` importable without
    # the installed package (the case appends the repo root itself on ImportError).
    import importlib

    return importlib.import_module("run")


def _params():
    from model import PaperParameters

    return PaperParameters()


def test_square_does_not_call_set_disc_domain() -> None:
    _install_fake_adc()
    run = _import_run()
    params = _params()
    rho = np.full((16, 16), params.rho_min)
    sim = run.build_uniform(object(), rho, params, geometry="square")
    names = [c[0] for c in sim.calls]
    assert "set_disc_domain" not in names, (
        "square geometry must stay bit-identical (no set_disc_domain): %r"
        % names
    )


def test_staircase_calls_set_disc_domain_with_center_and_radius() -> None:
    _install_fake_adc()
    run = _import_run()
    params = _params()
    rho = np.full((16, 16), params.rho_min)
    sim = run.build_uniform(object(), rho, params, geometry="staircase")
    disc_calls = [c for c in sim.calls if c[0] == "set_disc_domain"]
    assert (
        len(disc_calls) == 1
    ), "staircase must call set_disc_domain exactly once: %r" % (
        [c[0] for c in sim.calls],
    )
    _, args, kw = disc_calls[0]
    cx, cy, R = args
    assert cx == 0.5 * params.length, "cx must be L/2 (%g), got %g" % (
        0.5 * params.length,
        cx,
    )
    assert cy == 0.5 * params.length, "cy must be L/2 (%g), got %g" % (
        0.5 * params.length,
        cy,
    )
    assert R == params.radius, "R must equal params.radius (%g), got %g" % (
        params.radius,
        R,
    )
    # The disc center must coincide with the circular Poisson wall center and the
    # disc radius with the wall radius, so the FV mask and the elliptic wall agree.
    assert R == params.radius
    assert (
        kw.get("mode") == "staircase"
    ), "mode must be 'staircase', got %r" % kw.get("mode")


def test_cutcell_calls_set_disc_domain_with_mode_cutcell() -> None:
    _install_fake_adc()
    run = _import_run()
    params = _params()
    rho = np.full((16, 16), params.rho_min)
    sim = run.build_uniform(object(), rho, params, geometry="cutcell")
    disc_calls = [c for c in sim.calls if c[0] == "set_disc_domain"]
    assert (
        len(disc_calls) == 1
    ), "cutcell must call set_disc_domain exactly once: %r" % (
        [c[0] for c in sim.calls],
    )
    _, args, kw = disc_calls[0]
    cx, cy, R = args
    assert cx == 0.5 * params.length, "cx must be L/2 (%g), got %g" % (
        0.5 * params.length,
        cx,
    )
    assert cy == 0.5 * params.length, "cy must be L/2 (%g), got %g" % (
        0.5 * params.length,
        cy,
    )
    assert R == params.radius, "R must equal params.radius (%g), got %g" % (
        params.radius,
        R,
    )
    assert (
        kw.get("mode") == "cutcell"
    ), "mode must be 'cutcell', got %r" % kw.get("mode")


def test_unknown_geometry_raises() -> None:
    _install_fake_adc()
    run = _import_run()
    params = _params()
    rho = np.full((16, 16), params.rho_min)
    raised = False
    try:
        run.build_uniform(object(), rho, params, geometry="hexagon")
    except ValueError as exc:
        raised = True
        assert "staircase" in str(exc) or "square" in str(exc)
    assert raised, "an unknown geometry must raise ValueError"


def test_staircase_rejected_for_amr_engine() -> None:
    _install_fake_adc()
    run = _import_run()
    argv = sys.argv
    sys.argv = [
        "run.py",
        "--engine",
        "amr-imex",
        "--geometry",
        "staircase",
        "--acknowledge-amr-approximation",
    ]
    try:
        raised = False
        try:
            run.main()
        except SystemExit as exc:
            raised = True
            # main() raises SystemExit with the explanatory message before any build.
            assert "staircase" in str(
                exc
            ), "expected staircase rejection, got %r" % (exc,)
        assert (
            raised
        ), "staircase + amr-imex must be rejected at the argument layer"
    finally:
        sys.argv = argv


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("all %d geometry-flag smoke tests passed" % len(tests))


if __name__ == "__main__":
    _run_all()
