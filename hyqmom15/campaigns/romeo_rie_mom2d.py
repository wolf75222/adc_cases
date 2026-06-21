#!/usr/bin/env python3
"""ROMEO campaign over the five RieMOM2D_Electrostatic_periodic cases (ADC-376).

Runs all five cases (dicotron, fluid_wave, electrostatic_wave, magnetic_wave,
constant) reusing their M8 drivers, writing ``adc.System.write(format="npz")``
snapshots plus a ``run_meta.json`` provenance sidecar per case, monitoring
realizability and symmetry at the snapshot interval (non-fatal), and emitting a
synthesis table. The Matlab-vs-ADC speedup uses ``octave_matlab.py`` timings.

This module needs the ``adc`` build to run the cases (the ROMEO path); the
``--dry-run`` mode and the synthesis/meta helpers are build-free, so the campaign
structure is checked in CI without a build (``check_campaign.py``).

Usage:
    python3 hyqmom15/campaigns/romeo_rie_mom2d.py --smoke|--full [--cases a,b]
        [--out DIR] [--threads N] [--dry-run] [--matlab-times matlab_times.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
HYQMOM = HERE.parent
REPO = HYQMOM.parent
sys.path.insert(0, str(HERE))            # synthesis, octave_matlab
sys.path.insert(0, str(HYQMOM))          # drivers, diagnostics, matlab_ref

import synthesis  # noqa: E402

# Per-case resolution: (full Np, smoke Np). fluid_wave/constant ICs are fixed at
# their Matlab Np, so the smoke value is ignored for those two.
CASE_NP = {
    "dicotron": (128, 64),
    "fluid_wave": (32, 32),
    "electrostatic_wave": (128, 64),
    "magnetic_wave": (256, 64),
    "constant": (64, 64),
}
SMOKE_STEPS = 10  # cap steps in --smoke so a local/CI smoke is quick
DEFAULT_SNAPSHOTS = 12


def _solver_config(case, riemann):
    return {
        "riemann": riemann,
        "reconstruction": case.reconstruction or "first",
        "limiter": case.limiter or "none",
        "space_scheme": case.space_scheme,
        "exact_speeds": True,
        "projection": False,
        "backend": "production",
    }


def _params(case):
    return {
        "Np": case.Np, "cfl": case.cfl, "tmax": case.tmax,
        "omega_p": case.omega_p, "omega_c": case.omega_c,
        "kx": case.kx, "ky": case.ky, "mode": case.mode, "eps": case.eps,
        "electrostatic": case.electrostatic, "magnetostatic": case.magnetostatic,
        "source": case.source, "time_scheme": case.time_scheme,
    }


def _run_and_record(sim, case, case_dir, n_snapshots, dt_fn, smoke):
    """Common loop: step to tmax, write snapshots + monitor realizability/symmetry."""
    import time

    import numpy as np

    from diagnostics import field_realizability, summarize, symmetry_residual

    case_dir.mkdir(parents=True, exist_ok=True)
    tmax, cfl = case.tmax, case.cfl
    snap_dt = tmax / max(1, n_snapshots)
    mass0 = float(np.array(sim.get_state("mom"))[0].sum())
    t, step, dts = 0.0, 0, []
    next_snap_t, snap_k = 0.0, 0
    realiz, sym = [], []
    t0 = time.perf_counter()
    while True:
        if t >= next_snap_t - 1e-15 or t >= tmax:
            sim.write(str(case_dir / "step"), format="npz", step=snap_k)
            u = np.array(sim.get_state("mom"))
            realiz.append(summarize(*field_realizability(u)))
            try:
                sym.append(symmetry_residual(u, case.name))
            except KeyError:
                pass
            snap_k += 1
            next_snap_t += snap_dt
        if t >= tmax or (smoke and step >= SMOKE_STEPS):
            break
        if dt_fn is not None:
            dt = min(dt_fn(t), tmax - t)
            sim.step(dt)
        else:
            dt = sim.step_cfl(cfl)
        t += dt
        step += 1
        dts.append(dt)
    wall = time.perf_counter() - t0
    u = np.array(sim.get_state("mom"))
    final = realiz[-1] if realiz else {}
    return {
        "Np": int(u.shape[1]), "n_steps": step, "n_snapshots": snap_k,
        "mass_rel_drift": (abs(float(u[0].sum()) - mass0) / mass0) if mass0 else 0.0,
        "M00_min": float(u[0].min()),
        "realizable": bool(final.get("frac_nonrealizable", 1.0) == 0.0) if realiz else None,
        "dt_min": min(dts) if dts else None, "dt_max": max(dts) if dts else None,
        "wall_clock_s": wall,
        "status": "ok" if (np.all(np.isfinite(u)) and np.all(u[0] > 0)) else "FAIL",
        "realiz_series": realiz, "sym_series": sym,
    }


def _run_diocotron(n, case_dir, n_snapshots, smoke):
    import run_diocotron_periodic as drv
    from matlab_ref import compute_dt
    case = drv.periodic_case(n)
    u0 = drv.diocotron_ic(n, orientation="standard")
    rho_bg = float(u0[0].mean())
    probe = drv.build_periodic_sim(n, rho_bg=rho_bg)
    probe.set_state("mom", u0)
    probe.solve_fields()
    vmax = case.cfl * case.dx / probe.step_cfl(case.cfl)
    sim = drv.build_periodic_sim(n, rho_bg=rho_bg)
    sim.set_state("mom", u0)
    sim.solve_fields()
    rec = _run_and_record(sim, case, case_dir, n_snapshots, lambda t: compute_dt(vmax, case, t), smoke)
    return case, _solver_config(case, "hll"), rec


def _run_wave(driver, case_builder, ic, build, n, case_dir, n_snapshots, smoke):
    import importlib

    import numpy as np

    from matlab_ref import compute_dt
    drv = importlib.import_module(driver)
    case = getattr(drv, case_builder)(n)
    # The wave ICs return Matlab layer layout (k, x, y); ADC set_state wants
    # (k, ny, nx), so swap as the drivers' check_smoke does before set_state.
    u0 = np.swapaxes(getattr(drv, ic)(n), 1, 2)
    rho_bg = float(u0[0].mean())
    builder = getattr(drv, build)
    probe = builder(n, rho_bg=rho_bg)
    probe.set_state("mom", u0)
    probe.solve_fields()
    vmax = case.cfl * case.dx / probe.step_cfl(case.cfl)
    sim = builder(n, rho_bg=rho_bg)
    sim.set_state("mom", u0)
    sim.solve_fields()
    rec = _run_and_record(sim, case, case_dir, n_snapshots,
                          lambda t: compute_dt(vmax, case, t), smoke)
    return case, _solver_config(case, "hll"), rec


def _run_fluid(n, case_dir, n_snapshots, smoke):
    del n  # fluid_wave IC (fluid_ic) is fixed at the Matlab Np=32
    import numpy as np

    import run_fluid_wave as drv
    case = drv.CASE
    # fluid_ic returns Matlab layer layout (k, x, y); swap to ADC (k, ny, nx)
    # as _evolve does before set_state.
    u0 = np.swapaxes(drv.fluid_ic(), 1, 2)
    compiled = drv.build_fluid_model()
    sim = drv.build_fluid_sim(case.Np, compiled, "roe")
    sim.set_state("mom", u0)
    rec = _run_and_record(sim, case, case_dir, n_snapshots, None, smoke)
    return case, _solver_config(case, "roe"), rec


def _run_constant(n, case_dir, n_snapshots, smoke):
    del n  # constant IC (constant_ic) is fixed at the Matlab Np=64
    import run_constant as drv
    case = drv.CASE
    u0 = drv.constant_ic()
    sim = drv.build_constant_sim(case.Np)
    sim.set_state("mom", u0)
    rec = _run_and_record(sim, case, case_dir, n_snapshots, None, smoke)
    return case, _solver_config(case, "hll"), rec


def run_one(case_name, n, case_dir, n_snapshots, smoke):
    """Run a single case end-to-end; returns (case, solver_config, record)."""
    if case_name == "dicotron":
        return _run_diocotron(n, case_dir, n_snapshots, smoke)
    if case_name == "fluid_wave":
        return _run_fluid(n, case_dir, n_snapshots, smoke)
    if case_name == "constant":
        return _run_constant(n, case_dir, n_snapshots, smoke)
    if case_name == "electrostatic_wave":
        return _run_wave("run_electrostatic_wave", "es_case", "es_ic", "build_es_sim",
                         n, case_dir, n_snapshots, smoke)
    if case_name == "magnetic_wave":
        return _run_wave("run_magnetic_wave", "mag_case", "mag_ic", "build_mag_sim",
                         n, case_dir, n_snapshots, smoke)
    raise ValueError("unknown case %r" % case_name)


def _provenance(case_name, params, solver, rec, threads, host, ts):
    return synthesis.make_run_meta(
        case_name, params, solver, threads=threads,
        wall_clock_s=rec.get("wall_clock_s"), n_steps=rec.get("n_steps"),
        dt_min=rec.get("dt_min"), dt_max=rec.get("dt_max"), amr=False,
        commit_adc_cases=synthesis.git_commit(REPO),
        commit_adc_cpp=synthesis.git_commit(REPO.parent / "adc_cpp"),
        host=host, timestamp=ts)


def run_campaign(out_dir, cases, smoke, dry_run, threads, matlab_times=None):
    """Run the campaign; write per-case snapshots + run_meta.json and a synthesis table."""
    import datetime
    import socket

    out_dir = pathlib.Path(out_dir)
    host = socket.gethostname()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if not dry_run:
        import adc
        adc.set_threads(threads)
        threads = adc.parallel_info().get("omp_num_threads", threads)
    rows, adc_times = [], {}
    for name in cases:
        full_n, smoke_n = CASE_NP[name]
        n = smoke_n if smoke else full_n
        case_dir = out_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            from matlab_ref import get_case
            case = get_case(name)
            params, solver, rec = _params(case), _solver_config(case, "hll"), {
                "Np": n, "n_steps": 0, "status": "dry-run", "wall_clock_s": None}
        else:
            case, solver, rec = run_one(name, n, case_dir, DEFAULT_SNAPSHOTS, smoke)
            params = _params(case)
            adc_times[name] = rec.get("wall_clock_s")
        meta = _provenance(name, params, solver, rec, threads, host, ts)
        (case_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(synthesis.synthesis_row(name, {**rec, "Np": rec.get("Np", n)}))
        print("  %-20s -> %s (%s steps, %s)"
              % (name, rec.get("status"), rec.get("n_steps"), case_dir))
    report = ["# hyqmom15 ROMEO campaign synthesis", "", synthesis.synthesis_table(rows)]
    if matlab_times:
        mt = json.loads(pathlib.Path(matlab_times).read_text(encoding="utf-8"))
        report += ["", "## Matlab vs ADC speedup", "", synthesis.speedup_table(adc_times, mt)]
    (out_dir / "synthesis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("wrote synthesis to %s" % (out_dir / "synthesis.md"))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="hyqmom15 ROMEO campaign over the 5 RieMOM2D cases.")
    p.add_argument("--out", default="out/hyqmom15_campaign", help="campaign output directory")
    p.add_argument("--cases", help="comma-separated subset (default: all 5)")
    p.add_argument("--threads", type=int, default=1, help="adc.set_threads value")
    p.add_argument("--matlab-times", help="octave_matlab.py timings JSON for the speedup table")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--smoke", action="store_true", help="reduced Np + capped steps")
    grp.add_argument("--full", action="store_true", help="full Matlab resolution")
    p.add_argument("--dry-run", action="store_true", help="no adc run; structure + meta only")
    args = p.parse_args(argv)

    cases = args.cases.split(",") if args.cases else list(CASE_NP)
    unknown = [c for c in cases if c not in CASE_NP]
    if unknown:
        print("unknown cases: %s" % ", ".join(unknown), file=sys.stderr)
        return 1
    if not (args.smoke or args.full or args.dry_run):
        p.error("specify --smoke, --full, or --dry-run (no silent default)")
    smoke = args.smoke  # --full -> full resolution; --dry-run ignores it (no sim)
    run_campaign(args.out, cases, smoke=smoke, dry_run=args.dry_run,
                 threads=args.threads, matlab_times=args.matlab_times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
