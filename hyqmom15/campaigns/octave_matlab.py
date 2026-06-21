#!/usr/bin/env python3
"""Time the Matlab RieMOM2D reference cases under Octave (ADC-376 speedup baseline).

The Octave run happens wherever Octave and the (non-vendored) Matlab source live
-- local or another machine, per the campaign decision -- and its per-case
wall-clock feeds ``synthesis.speedup_table`` against the ADC wall-clock measured
on ROMEO. The per-case entry script is configurable for the local Matlab layout.

Usage:
    python3 hyqmom15/campaigns/octave_matlab.py <matlab_src_dir> [--out matlab_times.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

# Default guess for each case's Matlab entry script; override for the local tree.
CASE_SCRIPTS = {
    "dicotron": "main_dicotron.m",
    "fluid_wave": "main_fluid_wave.m",
    "electrostatic_wave": "main_electrostatic_wave.m",
    "magnetic_wave": "main_magnetic_wave.m",
    "constant": "main_constant.m",
}


def octave_command(matlab_src, script):
    """Headless Octave command (list) that runs ``script`` from ``matlab_src``."""
    name = script[:-2] if script.endswith(".m") else script
    return ["octave", "--no-gui", "--norc", "--eval",
            "cd('%s'); %s" % (str(matlab_src), name)]


def run_matlab_case(matlab_src, script, timeout=7200):
    """Run one case under Octave; return wall-clock seconds, or ``None`` on failure."""
    cmd = octave_command(matlab_src, script)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    return time.perf_counter() - t0


def run_all(matlab_src, scripts, out_path):
    """Time every case and write ``{case: wall_seconds}`` to ``out_path``."""
    times = {}
    for case, script in scripts.items():
        wall = run_matlab_case(matlab_src, script)
        times[case] = wall
        print("  %-20s %s -> %s" % (case, script, "%.2fs" % wall if wall else "FAILED"))
    pathlib.Path(out_path).write_text(json.dumps(times, indent=2), encoding="utf-8")
    return times


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Time the Matlab RieMOM2D cases under Octave.")
    p.add_argument("matlab_src", help="dir with the RieMOM2D_Electrostatic_periodic Matlab source")
    p.add_argument("--out", default="matlab_times.json", help="output timings JSON")
    args = p.parse_args(argv)
    src = pathlib.Path(args.matlab_src)
    if not src.is_dir():
        print("matlab source dir not found: %s" % src)
        return 1
    run_all(src, CASE_SCRIPTS, args.out)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
