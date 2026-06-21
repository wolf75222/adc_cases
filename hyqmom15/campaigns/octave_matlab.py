#!/usr/bin/env python3
"""Time the Matlab RieMOM2D reference cases under Octave (ADC-376 speedup baseline).

The reference is a single ``main.m`` that picks the case via an internal
``case_name = "..."`` line (then ``init_case``, the time loop, ``tic/toc``). To
time each of the five cases we make a temp copy of ``main.m`` with that line
rewritten to the target case and run it under Octave, capturing wall-clock. We
never edit the Matlab source in place. The per-case wall-clock feeds
``synthesis.speedup_table`` against the ADC wall-clock measured on ROMEO.

Run wherever Octave and the (non-vendored) Matlab source live (local or another
machine), per the campaign decision.

Caveat (measured 2026-06-21): under Octave the reference crashes in
``eigenvalues15_2D`` ("matrix contains Inf or NaN") -- the D7 corner artifact
(zero-state ghost corners; Octave's ``eig`` is stricter than Matlab's about
Inf/NaN). So the speedup baseline needs Matlab proper, or a corner-state guard in
the reference; cases that crash return ``None`` (n/a in the speedup table).

Usage:
    python3 hyqmom15/campaigns/octave_matlab.py <matlab_src_dir> [--out matlab_times.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

CASES = ["dicotron", "fluid_wave", "electrostatic_wave", "magnetic_wave", "constant"]
_CASE_RE = re.compile(r'case_name\s*=\s*"[^"]*"\s*;')


def case_main_source(main_src, case_name):
    """Return ``main.m`` text with its ``case_name = "..."`` line set to ``case_name``.

    Raises ValueError if no ``case_name = "..."`` assignment is found.
    """
    new, n = _CASE_RE.subn('case_name = "%s";' % case_name, main_src, count=1)
    if n == 0:
        raise ValueError('no `case_name = "..."` line found in main.m')
    return new


def run_matlab_case(matlab_src, case_name, timeout=7200):
    """Run one case under Octave (temp main.m copy); return wall-clock s, or None."""
    matlab_src = pathlib.Path(matlab_src)
    main_src = (matlab_src / "main.m").read_text(encoding="utf-8")
    tmp = matlab_src / (".octave_main_%s.m" % case_name)
    tmp.write_text(case_main_source(main_src, case_name), encoding="utf-8")
    cmd = ["octave", "--no-gui", "--norc", tmp.name]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(matlab_src), check=True,
                       capture_output=True, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        tail = (getattr(exc, "stderr", "") or "").strip().splitlines()
        print("  %s: Octave run failed (%s) %s"
              % (case_name, type(exc).__name__, tail[-1] if tail else ""),
              file=sys.stderr)
        wall = None
    finally:
        tmp.unlink(missing_ok=True)
    return wall


def run_all(matlab_src, cases, out_path):
    """Time every case and write ``{case: wall_seconds}`` to ``out_path``."""
    times = {}
    for case in cases:
        wall = run_matlab_case(matlab_src, case)
        times[case] = wall
        print("  %-20s -> %s" % (case, "%.2fs" % wall if wall is not None else "FAILED"))
    pathlib.Path(out_path).write_text(json.dumps(times, indent=2), encoding="utf-8")
    return times


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Time the Matlab RieMOM2D cases under Octave.")
    p.add_argument("matlab_src", help="dir with RieMOM2D_Electrostatic_periodic (main.m)")
    p.add_argument("--out", default="matlab_times.json", help="output timings JSON")
    args = p.parse_args(argv)
    src = pathlib.Path(args.matlab_src)
    if not (src / "main.m").is_file():
        print("main.m not found under %s" % src, file=sys.stderr)
        return 1
    run_all(src, CASES, args.out)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
