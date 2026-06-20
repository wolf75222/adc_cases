#!/usr/bin/env python3
"""Test for the --dump-npz raw-state dump of run.py, against the REAL adc engine.

No fake/mock adc: this builds a real ``adc.System`` (built-in isothermal model, like
``adc_cpp/python/tests/test_io_checkpoint.py``) and exercises the real ``sim.write`` path.
It needs the real ``adc`` extension importable (run with the conda env ``adc`` or
``PYTHONPATH=<adc_cpp>/build/python``); set ``KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1``
for the Kokkos-OpenMP build.

  - parse_args        : `--dump-npz` defaults off, is accepted when passed
  - resolve_dump_npz  : pure policy -- enabled single-rank, disabled under MPI (np>1),
                        disabled when not requested
  - dump_state_npz    : a real System writes <mode_dir>/state_<NNNNNN>.npz via
                        sim.write(format='npz', step=idx); the file carries the real
                        per-block state, phi and clock

Run standalone (`python3 test_dump_npz.py`) or under pytest.
"""

import os
import sys
import tempfile

import numpy as np

import adc  # real extension (no fake) -- import fails loudly if adc is not built

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:        # case root: model.py, run*.py
    sys.path.insert(0, os.path.dirname(HERE))
import run  # noqa: E402  (imported after the path tweak; pulls the real adc)


def _real_system(n=16):
    """A minimal REAL adc.System (built-in isothermal model, no DSL compile) whose
    ``write(format='npz')`` path is the same one run.py drives for --dump-npz."""
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block(
        "ions",
        adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                  transport=adc.IsothermalFlux(),
                  source=adc.PotentialForce(charge=1.0),
                  elliptic=adc.ChargeDensity(charge=1.0)),
        spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


def _parse(*extra):
    argv = sys.argv
    sys.argv = ["run.py", "--engine", "system-schur", "--modes", "3", *extra]
    try:
        return run.parse_args()
    finally:
        sys.argv = argv


def test_dump_npz_defaults_off():
    assert _parse().dump_npz is False, "--dump-npz must default to False"


def test_dump_npz_flag_parsed():
    assert _parse("--dump-npz").dump_npz is True, "--dump-npz must set args.dump_npz True"


def test_resolve_disabled_when_not_requested():
    assert run.resolve_dump_npz(False, 1) == (False, None)


def test_resolve_enabled_single_rank():
    assert run.resolve_dump_npz(True, 1) == (True, None)


def test_resolve_disabled_under_mpi():
    enabled, reason = run.resolve_dump_npz(True, 4)
    assert enabled is False, "the raw dump is single-rank only -> disabled under MPI"
    assert reason is not None and "MPI" in reason, reason


def test_dump_state_npz_writes_real_indexed_file():
    sim = _real_system()
    with tempfile.TemporaryDirectory() as out:
        path = run.dump_state_npz(sim, out, mode=4, index=0)
        assert path is not None, "dump_state_npz must return the written path"
        assert path.endswith(os.path.join("mode_4", "state_000000.npz")), path
        assert os.path.exists(path), "the npz file must exist on disk"
        with np.load(path) as d:
            # real fields written by the real System.write(format='npz')
            assert "phi" in d.files, d.files
            assert "macro_step" in d.files, d.files
            assert any(k.startswith("state_") for k in d.files), d.files
            phi = d["phi"]
            assert phi.shape == (16, 16) and np.isfinite(phi).all(), phi.shape


def test_dump_state_npz_uses_snapshot_index_in_name():
    sim = _real_system()
    with tempfile.TemporaryDirectory() as out:
        p0 = run.dump_state_npz(sim, out, mode=3, index=0)
        p7 = run.dump_state_npz(sim, out, mode=3, index=7)
    assert p0 is not None and p7 is not None, (p0, p7)
    assert p0.endswith("state_000000.npz") and p7.endswith("state_000007.npz"), (p0, p7)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("all %d dump-npz tests passed (real adc)" % len(tests))


if __name__ == "__main__":
    _run_all()
