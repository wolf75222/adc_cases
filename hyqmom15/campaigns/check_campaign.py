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

import octave_matlab as om  # noqa: E402
import romeo_rie_mom2d as campaign  # noqa: E402
import synthesis  # noqa: E402


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


def check_octave_command():
    cmd = om.octave_command("/tmp/matlab_src", "main_dicotron.m")
    assert cmd[0] == "octave" and "--no-gui" in cmd, cmd
    assert cmd[-1].endswith("main_dicotron") and "cd('/tmp/matlab_src')" in cmd[-1], cmd
    return "octave command OK (headless, cd + script name without .m)"


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


CHECKS = [check_tables, check_run_meta, check_octave_command, check_dry_run]


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
