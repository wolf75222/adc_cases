#!/usr/bin/env python3
"""Generate the per-case analysis report for a hyqmom15 campaign (ADC-376).

Reads a campaign directory (snapshots + run_meta.json per case, the figures from
hyqmom15/plots, the HDF5 from export_h5.py) and emits a single ``rapport.md``: for
each case a config table, a realizability verdict (the ADC-383 checks + recovery),
mass conservation / positivity, the symmetry residual, the figure list, and the
HDF5 export reference; then a global synthesis table and, if given, the
Matlab-vs-ADC speedup. Pure Python, build-free.

Usage:
    python3 hyqmom15/campaigns/make_rapport.py <campaign_dir>
        [--matlab-times matlab_times.json] [--out rapport.md]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "plots"))

import synthesis  # noqa: E402
from diagnostics import evaluate, field_realizability, summarize, symmetry_residual  # noqa: E402
from snapshots import load_case, load_meta, time_series  # noqa: E402


def _config_lines(meta):
    s = meta.get("solver", {})
    p = meta.get("params", {})
    return [
        "| field | value |", "|---|---|",
        "| case | %s |" % meta.get("case", "?"),
        "| Np | %s |" % meta.get("Np", "?"),
        "| scheme | riemann=%s recon=%s limiter=%s |"
        % (s.get("riemann"), s.get("reconstruction"), s.get("limiter")),
        "| sources | es=%s ms=%s (omega_p=%s omega_c=%s) |"
        % (p.get("electrostatic"), p.get("magnetostatic"), p.get("omega_p"), p.get("omega_c")),
        "| time | scheme=%s tmax=%s cfl=%s |"
        % (p.get("time_scheme"), p.get("tmax"), p.get("cfl")),
        "| backend / AMR | %s / %s |"
        % (s.get("backend"), "yes" if meta.get("amr") else "no"),
        "| threads / wall | %s / %ss |" % (meta.get("threads"), meta.get("wall_clock_s")),
        "| dt range / steps | [%s, %s] / %s |"
        % (meta.get("dt_min"), meta.get("dt_max"), meta.get("n_steps")),
        "| commits | cases=%s cpp=%s |"
        % (str(meta.get("commit_adc_cases"))[:8], str(meta.get("commit_adc_cpp"))[:8]),
        "| host / time | %s / %s |" % (meta.get("host"), meta.get("timestamp")),
    ]


def _case_section(case, case_dir, figures_dir, h5_dir):
    snaps = load_case(case_dir)
    if not snaps:
        return ["## %s" % case, "", "_no snapshots_", ""]
    meta = load_meta(case_dir)
    ts = time_series(snaps)
    series = [summarize(*field_realizability(s.moments)) for s in snaps]
    checks = evaluate(snaps)
    lam_min = min(r["lam_min"] for r in series)
    frac_nonreal = max(r["frac_nonrealizable"] for r in series)
    m00_min = min(r["M00_min"] for r in series)
    try:
        sym0 = symmetry_residual(snaps[0].moments, case)
        symf = symmetry_residual(snaps[-1].moments, case)
        sym = "%.2e -> %.2e" % (sym0, symf)
    except KeyError:
        sym = "n/a"
    figs = sorted(f.name for f in figures_dir.glob("%s_*" % case)) if figures_dir.is_dir() else []
    h5 = h5_dir / ("%s.h5" % case)
    out = ["## %s" % case, ""]
    out += _config_lines(meta)
    out += [
        "",
        "**Realizability** (snapshot-interval, non-fatal):",
        "",
        "| check | passed | recovered | failed snapshots |",
        "|---|---|---|---|",
    ]
    for c in checks:
        out.append("| %s | %s | %s | %s |"
                   % (c["name"], "yes" if c["passed"] else "NO",
                      "yes" if c["recovered"] else "no", c["failed_steps"] or "-"))
    out += [
        "",
        "- min lam_min(p2p2) over time: **%.3e** (>=0 realizable)" % lam_min,
        "- max fraction non-realizable cells: **%.3g**" % frac_nonreal,
        "- mass relative drift (t0->tmax): **%.2e**" % float(ts["mass_rel_drift"][-1]),
        "- min M00 (positivity): **%.3e**" % m00_min,
        "- symmetry residual (t0 -> tmax): **%s**" % sym,
        "- figures: %s" % (", ".join("`%s`" % f for f in figs) if figs else "_none_"),
        "- HDF5 export: %s" % ("`%s`" % h5.name if h5.exists() else "_not generated_"),
        "",
    ]
    return out


def build_rapport(campaign_dir, matlab_times=None):
    root = pathlib.Path(campaign_dir)
    figures_dir, h5_dir = root / "figures", root / "h5"
    case_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and d.name not in ("figures", "h5"))
    lines = ["# hyqmom15 ROMEO campaign report", "",
             "Per-case detailed analysis (config, realizability, conservation, "
             "symmetry, figures, HDF5 export). D/Dmax is a clarified convention, "
             "not a Matlab bug (ADC-378).", ""]
    rows, adc_times = [], {}
    for cd in case_dirs:
        lines += _case_section(cd.name, cd, figures_dir, h5_dir)
        snaps = load_case(cd)
        if snaps:
            rec = summarize(*field_realizability(snaps[-1].moments))
            meta = load_meta(cd)
            rows.append(synthesis.synthesis_row(cd.name, {
                "Np": meta.get("Np"), "n_steps": meta.get("n_steps"),
                "mass_rel_drift": float(time_series(snaps)["mass_rel_drift"][-1]),
                "M00_min": rec["M00_min"], "realizable": rec["frac_nonrealizable"] == 0.0,
                "dt_min": meta.get("dt_min"), "dt_max": meta.get("dt_max"),
                "wall_clock_s": meta.get("wall_clock_s"), "status": "ok"}))
            adc_times[cd.name] = meta.get("wall_clock_s")
    lines += ["## Synthesis", "", synthesis.synthesis_table(rows), ""]
    if matlab_times:
        mt = json.loads(pathlib.Path(matlab_times).read_text(encoding="utf-8"))
        lines += ["## Matlab vs ADC speedup", "", synthesis.speedup_table(adc_times, mt), ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the hyqmom15 campaign report.")
    p.add_argument("campaign_dir")
    p.add_argument("--matlab-times")
    p.add_argument("--out")
    args = p.parse_args(argv)
    root = pathlib.Path(args.campaign_dir)
    if not root.is_dir():
        print("campaign dir not found: %s" % root, file=sys.stderr)
        return 1
    out = pathlib.Path(args.out) if args.out else root / "rapport.md"
    out.write_text(build_rapport(root, args.matlab_times), encoding="utf-8")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
