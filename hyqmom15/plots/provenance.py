#!/usr/bin/env python3
"""Run provenance for hyqmom15 figures (ADC-384).

The campaign (ADC-376) writes a ``run_meta.json`` next to each case's snapshots
so every figure can state exactly which run produced it. This module defines the
expected keys and renders a compact one-line footer; missing keys are skipped, so
partial metadata still produces a useful footer.

Expected ``run_meta.json`` keys (all optional)::

    case, Np,
    params: {cfl, tmax, omega_p, omega_c, kx, ky, mode, eps, sources},
    solver: {riemann, reconstruction, limiter, space_scheme,
             exact_speeds, projection, backend},
    amr (bool), threads, wall_clock_s, n_steps, dt_min, dt_max,
    commit_adc_cases, commit_adc_cpp, host, timestamp
"""
from __future__ import annotations


def provenance_footer(meta):
    """Compact one-line provenance string for a figure footer (``""`` if empty)."""
    if not meta:
        return ""
    solver = meta.get("solver", {})
    parts = []
    case = meta.get("case", "?")
    parts.append("%s%s" % (case, " Np=%s" % meta["Np"] if meta.get("Np") else ""))
    scheme = "/".join(str(solver[k]) for k in ("riemann", "reconstruction", "limiter")
                      if solver.get(k))
    if scheme:
        parts.append(scheme)
    if solver.get("backend"):
        parts.append("backend=%s" % solver["backend"])
    parts.append("AMR=%s" % ("yes" if meta.get("amr") else "no"))
    if meta.get("threads"):
        parts.append("threads=%s" % meta["threads"])
    if meta.get("wall_clock_s") is not None:
        parts.append("wall=%.1fs" % meta["wall_clock_s"])
    if meta.get("n_steps"):
        parts.append("steps=%s" % meta["n_steps"])
    if meta.get("dt_min") is not None and meta.get("dt_max") is not None:
        parts.append("dt=[%.1e,%.1e]" % (meta["dt_min"], meta["dt_max"]))
    commits = []
    if meta.get("commit_adc_cases"):
        commits.append("cases=%s" % str(meta["commit_adc_cases"])[:8])
    if meta.get("commit_adc_cpp"):
        commits.append("cpp=%s" % str(meta["commit_adc_cpp"])[:8])
    if commits:
        parts.append(" ".join(commits))
    if meta.get("host"):
        parts.append("host=%s" % meta["host"])
    return " | ".join(parts)


def add_footer(fig, meta, extra=""):
    """Stamp the provenance footer (and optional ``extra`` note) onto a figure."""
    text = provenance_footer(meta)
    if extra:
        text = (text + "  --  " + extra) if text else extra
    if text:
        fig.text(0.5, 0.003, text, ha="center", fontsize=6.5, color="0.4", wrap=True)
