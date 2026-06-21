#!/usr/bin/env python3
"""Code-anchored golden non-regression of HyQMOM 15-moment transport x native relaxation15 (ADC-203).

Freezes a small deterministic trajectory of the CURRENT adc_cpp code in the regime where the
realizability projector actually fires: the Ma=20 crossing flow (two beams) stresses the moments to
~50% non-realizable per step (run_relaxation.py), and HLL transport is run with the NATIVE relaxation15
projector (build_projection via m.projection / ADC-275), applied post-step by System per ADC-177. The
trajectory is re-run on every CI to catch silent drift of OUR transport x projector path.

This fills a real gap, distinct from existing coverage:
  - run_relaxation.py (3)/(4) apply the Python ORACLE relax_field manually each step, and (4) is
    MATLAB/RIEMOM2D-anchored (fidelity, tol 5e-8);
  - validate_native_projector.py checks the NATIVE projector but in ISOLATION (single states, no
    transport).
THIS golden is the only one that freezes the NATIVE projector run THROUGH a transport trajectory,
self-consistent (code-anchored: produced by this very code, so it pins our own trajectory, not MATLAB
fidelity).

Frozen configuration: crossing IC at Ma=20 (the projector is materially active there), n=32,
riemann="hll", exact_speeds=True, projection=True (relaxation15 ACTIVE), backend="production" (the
native zero-copy path the projector targets), no Poisson (pure transport x collisional projection, no
electric field). The time step is HARDCODED from the recorded meta, NOT re-derived with step_cfl, so a
future eigenvalue change cannot silently re-pick dt and pass as "no drift". Serial only (the projector
ordering is not bitwise across MPI ranks). Auto-skips (exit 0) when no C++ compiler is present.

Tolerance: committed CI gate atol=1e-4, rtol=0. The Ma=20 crossing is a stiff flow that Lyapunov-
amplifies FP differences, and the NATIVE compiled (Kokkos) projector + HLL diverge macOS<->Linux far
more than the numpy oracle the sister golden_crossing_relax uses: a macOS-generated golden vs a Linux
CI run drifts ~7.6e-6 over 3 steps (measured). So the golden is SAME-platform bit-exact (max|dU|=0) but
only CROSS-platform coarse: atol=1e-4 (~13x the measured drift) is a non-regression SMOKE -- it catches
the projector firing (the assertion below) and gross scheme/projector regressions, not subtle bit-level
drift (a bit-tight committed gate is not achievable for a compiled stiff flow checked on a different
platform than it was generated on). The check also asserts the projector is MATERIALLY active (a
projection=False replay must differ), otherwise the golden would gate only the transport.

Usage:
    python3 hyqmom15/runs/run_golden_transport_relax.py       # CHECK against the committed golden (CI)
    python3 hyqmom15/runs/run_golden_transport_relax.py --regen  # rewrite the golden + meta (manual)
"""

import argparse
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # hyqmom15/ : model, relaxation, gen_states
try:
    import adc_cases  # noqa: F401  (ensure the package root is importable)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import adc  # noqa: E402
from model import build_moment_model, crossing_state  # noqa: E402  (path set above)

_GOLD = os.path.join(os.path.dirname(_HERE), "golden")
_STATE_CSV = os.path.join(_GOLD, "golden_transport_relax_state.csv")
_META_CSV = os.path.join(_GOLD, "golden_transport_relax_meta.csv")
_N = 32
_MA = 20.0
_NSTEPS = 3
_DT = 2e-4  # frozen, conservative: the Ma=20 transport overshoots a CFL-sized dt before the post-step
            # projector can correct it (NaN); 2e-4 is the value run_relaxation.py proves stays finite.
# The Ma=20 crossing is a stiff flow that Lyapunov-amplifies FP differences, and this golden runs the
# NATIVE compiled (Kokkos) projector + HLL, whose FP (FMA, -O, libm) diverges macOS<->Linux far more
# than the numpy oracle the sister golden_crossing_relax uses (3.6e-9). Measured: a macOS-generated
# golden vs a Linux CI run drifts ~7.6e-6 over 3 steps (and ~1.2e-5 over 20). The drift is deterministic
# per platform (same Linux toolchain -> same binary -> stable), so a fixed gate is safe. This golden is
# therefore SAME-platform bit-exact (max|dU|=0) but only CROSS-platform coarse: atol=1e-4 (~13x the
# measured drift) is a non-regression SMOKE -- it catches the projector firing (the on/off assertion
# below) and gross scheme/projector regressions, not subtle bit-level drift. A bit-tight committed gate
# is not achievable for a compiled stiff flow checked on a different platform than it was generated on.
_ATOL = 1e-4  # committed CI gate; rtol = 0 (cross-platform coarse smoke, not bit-tight -- see above)


def _have_cxx():
  """Returns True when a C++ compiler is on PATH (the native model brick needs one)."""
  return bool(shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))


def _build_sim(projection):
  """Builds the frozen crossing sim (HLL transport, no Poisson), optionally with the native projector.

  with_sources is left False (no electric field): this isolates transport x collisional projection,
  the regime run_relaxation.py exercises for the projector.
  """
  from adc_cases.common.io import case_output_dir
  from adc_cases.common.native import adc_include

  tag = "_proj" if projection else "_nu"
  m = build_moment_model(name="hyqmom15_golden_relax" + tag, robust=False, exact_speeds=True,
                         projection=projection, Ma=_MA)
  compiled = m.compile(os.path.join(case_output_dir("hyqmom15"), "hyqmom15_golden_relax%s.so" % tag),
                       adc_include(), backend="production")
  sim = adc.System(n=_N, L=1.0, periodic=True)
  sim.add_equation("mom", model=compiled,
                   spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                   time=adc.Explicit())
  sim.set_state("mom", crossing_state(_N, _MA))
  return sim


def _replay(sim, dt):
  """Runs _NSTEPS at the frozen dt and returns the (_NSTEPS, 15, _N, _N) post-step trajectory.

  The native projector has already fired post-step inside each step(), so each recorded slice is the
  state the next step would consume.
  """
  traj = np.empty((_NSTEPS, 15, _N, _N))
  with np.errstate(all="ignore"):
    for k in range(_NSTEPS):
      sim.step(dt)
      traj[k] = np.asarray(sim.get_state("mom")).reshape(15, _N, _N)
  return traj


def regen():
  """Rewrites the committed golden trajectory and its meta (frozen dt) from the current code."""
  traj = _replay(_build_sim(projection=True), _DT)
  assert np.all(np.isfinite(traj)), "golden trajectory is not finite (dt too large or projector inert)"
  os.makedirs(_GOLD, exist_ok=True)
  np.savetxt(_STATE_CSV, traj.reshape(_NSTEPS, -1), delimiter=",", fmt="%.17g")
  np.savetxt(_META_CSV, np.array([[_N, _NSTEPS, _DT]]), delimiter=",", fmt="%.17g",
             header="n,nsteps,dt")  # default comment prefix "# " so np.loadtxt skips the header
  print("wrote %s (%d steps) and %s (dt=%.17g)" % (_STATE_CSV, _NSTEPS, _META_CSV, _DT))


def check():
  """Replays the frozen trajectory and asserts no drift and that the projector is materially active."""
  n, nsteps, dt = np.loadtxt(_META_CSV, delimiter=",")
  assert int(n) == _N and int(nsteps) == _NSTEPS, "golden meta n/nsteps mismatch (regenerate)"
  gold = np.loadtxt(_STATE_CSV, delimiter=",").reshape(_NSTEPS, 15, _N, _N)

  traj = _replay(_build_sim(projection=True), float(dt))
  dmax = float(np.max(np.abs(traj - gold)))
  ok = np.allclose(traj, gold, rtol=0.0, atol=_ATOL)
  print("(golden transport+relaxation) n=%d, %d steps, dt=%.6e, max|dU|=%.2e (atol=%.0e) -- %s"
        % (_N, _NSTEPS, float(dt), dmax, _ATOL, "OK" if ok else "DRIFT"))
  assert ok, ("transport+relaxation trajectory drifted from golden (max abs %.3e > atol %.0e); if "
              "intentional, regenerate with --regen and document the change" % (dmax, _ATOL))

  # The golden gates the PROJECTOR only if relaxation15 actually fires on this config: a
  # projection=False replay must differ, otherwise the freeze silently covers transport alone.
  traj_off = _replay(_build_sim(projection=False), float(dt))
  dact = float(np.nanmax(np.abs(traj - traj_off)))
  assert dact > 1e-9, ("relaxation15 is inert on this config (max %.2e); the golden would gate the "
                       "transport scheme only -- pick a config where the projector fires" % dact)
  print("  projector active: projection on/off differ by %.2e -- OK" % dact)


def main(argv=None):
  ap = argparse.ArgumentParser(description="ADC-203 golden transport + relaxation15 non-regression")
  ap.add_argument("--regen", action="store_true", help="rewrite the golden (manual, never in CI)")
  args = ap.parse_args(argv)

  if not _have_cxx():
    print("run_golden_transport_relax: skip (no C++ compiler) -- OK")
    return
  if int(os.environ.get("ADC_MPI_SIZE", "1")) > 1:
    print("run_golden_transport_relax: serial only (the golden is not bitwise across MPI ranks) -- skip")
    return

  if args.regen:
    regen()
  else:
    assert os.path.exists(_STATE_CSV) and os.path.exists(_META_CSV), (
        "golden missing: run with --regen first (%s)" % _STATE_CSV)
    check()
  print("hyqmom15/run_golden_transport_relax: OK")


if __name__ == "__main__":
  main()
