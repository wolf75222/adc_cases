#!/usr/bin/env python3
"""Smoke-test the hyqmom15 plotting layer (ADC-377).

Builds a tiny synthetic campaign (``adc.System.write(format="npz")``-shaped
snapshots, no simulation, no adc build) and checks that the pure-NumPy loader
and diagnostics read it back correctly. If matplotlib is importable it also
renders the figures and asserts the files appear; otherwise that part is skipped
(matplotlib is not a CI dependency, like the other ``make_figures.py`` tools).

Run: ``python3 hyqmom15/plots/check_plots.py`` (0 = OK, 1 = mismatch).
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import snapshots  # noqa: E402

MOMENTS = ["M00", "M10", "M20", "M30", "M40", "M01", "M11", "M21", "M31",
           "M02", "M12", "M22", "M03", "M13", "M04"]


def _synth_case(case_dir, nsteps=4, n=8, with_phi=False):
    """Write ``nsteps`` synthetic snapshots in the adc.System.write npz layout."""
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(nsteps):
        state = np.abs(rng.standard_normal((15, n, n))) + 0.1   # M00 > 0
        kw = dict(
            t=np.float64(0.01 * k),
            macro_step=np.int64(k),
            nx=np.int64(n),
            ny=np.int64(n),
            blocks=np.array(["mom"]),
            state_mom=state,
            names_mom=np.array(MOMENTS),
            roles_mom=np.array(["custom"] * 15),
            phi=(rng.standard_normal((n, n)) if with_phi else np.zeros((n, n))),
        )
        np.savez(case_dir / ("step_%06d.npz" % k), **kw)


def check_loader():
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d) / "electrostatic_wave"
        _synth_case(cd, nsteps=5, n=8, with_phi=True)
        snaps = snapshots.load_case(cd)
        assert len(snaps) == 5, "expected 5 snapshots, got %d" % len(snaps)
        assert [s.step for s in snaps] == [0, 1, 2, 3, 4], "snapshots not sorted by step"
        s = snaps[0]
        assert s.density.shape == (8, 8), "density shape %s" % (s.density.shape,)
        assert s.phi is not None, "phi should be present (non-zero)"
        assert s.mass > 0, "mass must be positive"
        assert np.allclose(s.moment("M00"), s.density), "M00 must equal density"
    return "loader OK (5 snapshots, sorted, density/phi/mass/moment-by-name)"


def check_no_phi():
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d) / "fluid_wave"
        _synth_case(cd, nsteps=3, n=8, with_phi=False)
        snaps = snapshots.load_case(cd)
        assert all(s.phi is None for s in snaps), "all-zero phi must load as None"
    return "no-phi OK (zero phi -> None, so the renderer skips the potential panel)"


def check_diagnostics():
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d) / "constant"
        _synth_case(cd, nsteps=4, n=8, with_phi=False)
        snaps = snapshots.load_case(cd)
        ts = snapshots.time_series(snaps)
        for key in ("t", "dt", "mass", "mass_rel_drift", "m00_min", "m00_max"):
            assert key in ts and len(ts[key]) == 4, "time_series missing/short %r" % key
        assert ts["dt"][0] == 0.0, "first dt should be 0 (prepend convention)"
        assert abs(ts["mass_rel_drift"][0]) < 1e-15, "drift at t0 must be 0"
    return "diagnostics OK (t, dt, mass drift, M00 min/max)"


def check_render():
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return "render SKIPPED (matplotlib not installed; CI runs loader-only)"
    import plot_rie_mom2d_case as plot
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        _synth_case(root / "electrostatic_wave", nsteps=4, n=8, with_phi=True)
        _synth_case(root / "fluid_wave", nsteps=4, n=8, with_phi=False)
        out = root / "figures"
        plot.render_case(root / "electrostatic_wave", out, make_gif=True)
        plot.render_case(root / "fluid_wave", out, make_gif=False)
        must = ["electrostatic_wave_density.png", "electrostatic_wave_phi.png",
                "electrostatic_wave_diagnostics.png", "electrostatic_wave_density.gif",
                "fluid_wave_density.png", "fluid_wave_diagnostics.png"]
        for f in must:
            assert (out / f).exists(), "missing figure %s" % f
        assert not (out / "fluid_wave_phi.png").exists(), "no-phi case must not write a phi panel"
    return "render OK (density/phi/diagnostics PNG + density GIF; phi panel skipped without a field)"


CHECKS = [check_loader, check_no_phi, check_diagnostics, check_render]


def main() -> int:
    failures = []
    for fn in CHECKS:
        try:
            print("  OK   %-18s %s" % (fn.__name__, fn()))
        except Exception as exc:  # noqa: BLE001
            failures.append(fn.__name__)
            print("  FAIL %-18s %s" % (fn.__name__, exc))
    if failures:
        print("CHECK-PLOTS: %d/%d FAILED" % (len(failures), len(CHECKS)), file=sys.stderr)
        return 1
    print("CHECK-PLOTS: OK (%d checks)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
