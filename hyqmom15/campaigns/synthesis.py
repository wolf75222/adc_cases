#!/usr/bin/env python3
"""Run metadata and synthesis/speedup tables for the hyqmom15 ROMEO campaign (ADC-376).

Pure helpers (no adc build, no matplotlib): assemble the per-case ``run_meta.json``
that the figures read (the ADC-384 provenance schema), and render the campaign's
synthesis and Matlab-vs-ADC speedup tables as Markdown. The actual case runs
(adc) and the Octave Matlab runs live in the orchestrator and octave_matlab.py.
"""
from __future__ import annotations

import subprocess


def git_commit(repo_dir):
    """Short HEAD hash of ``repo_dir`` (``"unknown"`` if not a git repo)."""
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=15)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_run_meta(case, params, solver, *, threads, wall_clock_s, n_steps,
                  dt_min, dt_max, amr=False, commit_adc_cases="unknown",
                  commit_adc_cpp="unknown", host="unknown", timestamp=""):
    """Assemble the provenance ``run_meta`` dict (ADC-384 schema) for one case run."""
    return {
        "case": case,
        "Np": params.get("Np"),
        "params": params,
        "solver": solver,
        "amr": amr,
        "threads": threads,
        "wall_clock_s": wall_clock_s,
        "n_steps": n_steps,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "commit_adc_cases": commit_adc_cases,
        "commit_adc_cpp": commit_adc_cpp,
        "host": host,
        "timestamp": timestamp,
    }


def synthesis_row(case, record):
    """One synthesis-table row from a case run ``record`` dict."""
    return {
        "case": case,
        "Np": record.get("Np", "?"),
        "n_steps": record.get("n_steps", "?"),
        "mass_drift": record.get("mass_rel_drift"),
        "M00_min": record.get("M00_min"),
        "realizable": record.get("realizable"),
        "dt_min": record.get("dt_min"),
        "dt_max": record.get("dt_max"),
        "wall_s": record.get("wall_clock_s"),
        "status": record.get("status", "?"),
    }


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return "%.3e" % v if (v and abs(v) < 1e-2) else "%.4g" % v
    return str(v)


def synthesis_table(rows):
    """Markdown synthesis table (mass drift, positivity, realizability, dt, runtime)."""
    head = ["case", "Np", "steps", "mass_drift", "M00_min", "realizable",
            "dt_min", "dt_max", "wall_s", "status"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k)) for k in head) + " |")
    return "\n".join(lines)


def speedup_table(adc_times, matlab_times):
    """Markdown Matlab-vs-ADC speedup table from two ``{case: wall_seconds}`` maps.

    ``speedup = matlab / adc`` (>1 means ADC is faster). Cases missing from either
    side are reported with ``n/a``.
    """
    cases = sorted(set(adc_times) | set(matlab_times))
    lines = ["| case | ADC wall (s) | Matlab/Octave wall (s) | speedup (mat/adc) |",
             "|---|---|---|---|"]
    for c in cases:
        a, m = adc_times.get(c), matlab_times.get(c)
        sp = "%.2fx" % (m / a) if (a and m and a > 0) else "n/a"
        lines.append("| %s | %s | %s | %s |" % (c, _fmt(a), _fmt(m), sp))
    return "\n".join(lines)
