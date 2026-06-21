#!/usr/bin/env python3
"""Guard for the hyqmom15 ROMEO campaign infrastructure (ADC-376).

Pure Python, no adc build: checks the synthesis and speedup tables, the
run_meta provenance schema, the Octave command construction, and a full
``--dry-run`` of the campaign orchestrator (which writes run_meta.json per case
and the synthesis report without running any simulation).

Run: ``python3 hyqmom15/campaigns/check_campaign.py`` (0 = OK, 1 = mismatch).
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import export_h5  # noqa: E402
import make_rapport  # noqa: E402
import octave_matlab as om  # noqa: E402
import romeo_rie_mom2d as campaign  # noqa: E402
import synthesis  # noqa: E402

_MOMENTS = ["M00", "M10", "M20", "M30", "M40", "M01", "M11", "M21", "M31",
            "M02", "M12", "M22", "M03", "M13", "M04"]


def _synth_campaign(root, cases=("constant", "fluid_wave"), nsteps=4, n=8):
    """Write a tiny synthetic campaign (snapshots + run_meta.json) for the smoke."""
    rng = np.random.default_rng(0)
    for case in cases:
        cdir = root / case
        cdir.mkdir(parents=True, exist_ok=True)
        for k in range(nsteps):
            np.savez(
                cdir / ("step_%06d.npz" % k),
                t=np.float64(0.01 * k), macro_step=np.int64(k),
                nx=np.int64(n), ny=np.int64(n), blocks=np.array(["mom"]),
                state_mom=np.abs(rng.standard_normal((15, n, n))) + 0.1,
                names_mom=np.array(_MOMENTS), roles_mom=np.array(["custom"] * 15),
                phi=np.zeros((n, n)))
        (cdir / "run_meta.json").write_text(json.dumps({
            "case": case, "Np": n,
            "params": {"electrostatic": False, "magnetostatic": False, "time_scheme": "Euler",
                       "tmax": 1.0, "cfl": 0.5, "omega_p": 30, "omega_c": -90},
            "solver": {"riemann": "hll", "reconstruction": "muscl", "limiter": "minmod",
                       "backend": "production"},
            "amr": False, "threads": 8, "wall_clock_s": 5.0, "n_steps": 40,
            "dt_min": 1e-5, "dt_max": 2e-4, "commit_adc_cases": "abc12345",
            "commit_adc_cpp": "def67890", "host": "romeo01", "timestamp": "2026-06-21T00:00:00+00:00"}))


def check_tables():
    rows = [synthesis.synthesis_row("fluid_wave", {
        "Np": 32, "n_steps": 100, "mass_rel_drift": 1e-14, "M00_min": 0.5,
        "realizable": True, "dt_min": 1e-5, "dt_max": 2e-4, "wall_clock_s": 3.1,
        "status": "ok"})]
    tbl = synthesis.synthesis_table(rows)
    assert "fluid_wave" in tbl and "yes" in tbl and "status" in tbl, tbl
    sp = synthesis.speedup_table({"a": 2.0, "b": 4.0}, {"a": 6.0})
    assert "3.00x" in sp and "n/a" in sp, sp  # a: 6/2=3x; b: matlab missing -> n/a
    return "tables OK (synthesis row/table, speedup ratio + missing -> n/a)"


def check_run_meta():
    meta = synthesis.make_run_meta(
        "dicotron", {"Np": 128, "cfl": 0.5}, {"riemann": "hll", "backend": "production"},
        threads=8, wall_clock_s=12.3, n_steps=420, dt_min=1e-5, dt_max=3e-4,
        commit_adc_cases="abc123", commit_adc_cpp="def456", host="romeo01")
    for k in ("case", "Np", "params", "solver", "amr", "threads", "wall_clock_s",
              "n_steps", "dt_min", "dt_max", "commit_adc_cases", "commit_adc_cpp", "host"):
        assert k in meta, "run_meta missing %r" % k
    assert meta["case"] == "dicotron" and meta["Np"] == 128 and meta["amr"] is False
    assert isinstance(synthesis.git_commit(HERE), str), "git_commit must return a string"
    return "run_meta OK (provenance schema complete, git_commit returns str)"


def check_octave_source():
    main = 'clear;\ncase_name = "dicotron";\nparams = init_case(case_name);\n'
    out = om.case_main_source(main, "magnetic_wave")
    assert 'case_name = "magnetic_wave";' in out and 'dicotron' not in out, out
    try:
        om.case_main_source('no case here', "x")
    except ValueError:
        pass
    else:
        raise AssertionError("case_main_source must raise when no case_name line")
    return "octave source OK (case_name rewrite + raises when absent)"


def check_make_rapport():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        _synth_campaign(root, cases=("constant", "fluid_wave"))
        make_rapport.main([str(root)])
        rap = (root / "rapport.md").read_text()
        assert "## constant" in rap and "## fluid_wave" in rap and "## Synthesis" in rap
        assert "Realizability" in rap and "config" in rap.lower()
    return "make_rapport OK (per-case sections + realizability + synthesis)"


def check_export_h5():
    try:
        import h5py  # noqa: F401
    except Exception:
        return "export_h5 SKIPPED (h5py not installed)"
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        _synth_campaign(root, cases=("constant",))
        export_h5.main([str(root)])
        h5 = root / "h5" / "constant.h5"
        assert h5.exists(), "export_h5 did not write constant.h5"
        with h5py.File(h5) as f:
            assert "moments" in f and "t" in f and "realizability" in f
            assert f.attrs.get("case") == "constant"
    return "export_h5 OK (moments + t + realizability + provenance attrs)"


def check_dry_run():
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "campaign"
        cases = ["constant", "fluid_wave", "dicotron"]
        campaign.run_campaign(out, cases, smoke=True, dry_run=True, threads=1)
        for c in cases:
            meta_path = out / c / "run_meta.json"
            assert meta_path.exists(), "dry-run missing run_meta for %s" % c
            meta = json.loads(meta_path.read_text())
            assert meta["case"] == c and "solver" in meta and "params" in meta
        assert (out / "synthesis.md").exists(), "dry-run missing synthesis.md"
        assert "synthesis" in (out / "synthesis.md").read_text()
    return "dry-run OK (3 cases -> run_meta.json each + synthesis.md, no adc)"


CHECKS = [check_tables, check_run_meta, check_octave_source, check_dry_run,
          check_make_rapport, check_export_h5]


def main() -> int:
    failures = []
    for fn in CHECKS:
        try:
            print("  OK   %-20s %s" % (fn.__name__, fn()))
        except Exception as exc:  # noqa: BLE001
            failures.append(fn.__name__)
            print("  FAIL %-20s %s" % (fn.__name__, exc))
    if failures:
        print("CHECK-CAMPAIGN: %d/%d FAILED" % (len(failures), len(CHECKS)), file=sys.stderr)
        return 1
    print("CHECK-CAMPAIGN: OK (%d checks)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
