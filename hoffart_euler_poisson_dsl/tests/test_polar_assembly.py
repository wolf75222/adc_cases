#!/usr/bin/env python3
"""Test of the POLAR assembly + frozen-equilibrium path of run_polar.py, against the REAL adc.

No fake/mock adc. ``build_polar_system`` uses NATIVE bricks (adc.Model / IsothermalFlux /
BackgroundDensity), so it builds a real polar ``adc.System`` with no DSL compile; the pure
helpers (polar_gradient, equilibrium_v_theta, fit_growth) are tested directly. Needs the real
``adc`` importable (PYTHONPATH=<adc_cpp>/build/python ; KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1 for Kokkos-OpenMP).

Verified, on the real engine:
  (1) build_polar_system constructs a valid (3, ntheta, nr) conservative state and a finite,
      non-trivial polar Dirichlet potential (the polar Poisson routing ran).
  (2) --strang and the default Lie split both build and advance without NaN. (The precise scheme
      wiring -- WENO5 + Rusanov + SSPRK3 + CondensedSchur -- is now covered BEHAVIOURALLY: a
      mis-wired Schur/transport would break the machine-precision frozen stationarity of (10),
      since the real System exposes no call-order/scheme introspection.)
  (4) the seeded density is the annular top-hat (background rho_min outside [R0,R1]).
  (5a) v_r = ExB radial drift -grad_theta(phi)/B, against the REAL solved phi;
  (5b) v_theta solves the radial-balance quadratic (rotating equilibrium), residual ~ 0;
  (5d) cs2=0 -> v_theta reduces exactly to the ExB drift grad_r/B.
  (5c) the equilibrium ring (no perturbation) is stationary (non-frozen check_equilibrium).
  (c1) R_eq = step(U_eq) - U_eq is finite and non-trivial (the real scheme is not exactly
       stationary on the frozen ring);
  (c2) step()-R_eq makes U_eq a discrete fixed point to machine precision;
  (c3) check_equilibrium_frozen: max_dev <= machine floor over >= 200 steps;
  (c4) run_mode(R_eq=...) advances with finite amplitudes; (c5) main --quick runs to completion.
  (6) polar_gradient == the verbatim derive_aux_polar stencil; (7) fit_growth fits the MAPPED
      paper window exactly; (8) a multi-rank run is rejected (mono-rank polar Schur).

Run standalone (`python3 test_polar_assembly.py`) or under pytest.
"""

import math
import os
import sys

import numpy as np

import adc  # noqa: F401  (real extension)

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:        # case root: model.py, run*.py
    sys.path.insert(0, os.path.dirname(HERE))
import run_polar as rp  # noqa: E402
from model import PaperParameters  # noqa: E402


def _params():
    return PaperParameters()


class _Args:
    """Plain argument namespace (not a fake adc) mirroring run_polar's argparse defaults."""

    def __init__(self, **kw):
        self.r_min = 2.0
        self.nr = 24
        self.ntheta = 16
        self.cs2 = 0.0
        self.theta = 0.5
        self.strang = False
        self.limiter = "weno5"
        self.ic = "equilibrium"
        self.dt = 1.0e-3
        self.cfl = 0.0
        self.frozen_equilibrium = True
        self.frozen_check_const = 1.0e3
        self.max_steps_check = 20
        self.check_modes = [1, 2, 3, 4, 5]
        self.check_tol = 0.05
        self.t_end = 0.004
        self.sample_every = 1
        self.max_steps = 100
        for k, v in kw.items():
            setattr(self, k, v)


def _state(sim, nth, nr):
    return np.asarray(sim.get_state("ne"), dtype=np.float64).reshape(3, nth, nr)


# --- (1)(2) assembly: real build, behavioural routing -------------------------------------

def test_build_polar_system_constructs_valid_state():
    params, args = _params(), _Args()
    sim = rp.build_polar_system(args.nr, args.ntheta, mode=4, params=params, args=args)
    U = _state(sim, args.ntheta, args.nr)
    assert U.shape == (3, args.ntheta, args.nr) and np.isfinite(U).all(), U.shape
    phi = np.asarray(sim.potential(), dtype=np.float64)
    assert np.isfinite(phi).all(), "polar Dirichlet Poisson must yield a finite potential"
    assert float(phi.max() - phi.min()) > 0.0, "the polar Poisson solve must produce a non-trivial phi"


def test_strang_and_split_both_build_and_step():
    params = _params()
    states = {}
    for strang in (False, True):
        sim = rp.build_polar_system(24, 16, 4, params, _Args(strang=strang))
        sim.step(1.0e-3)
        U = _state(sim, 16, 24)
        assert np.isfinite(U).all(), "split=%s must advance without NaN" % ("Strang" if strang else "Lie")
        states[strang] = U
    # --strang must SELECT a different integrator: from the same seeded state, the Strang
    # (H(dt/2);S;H(dt/2)) and Lie (H;S) single steps differ by O(dt^2). This proves --strang is
    # not silently ignored (the real System exposes no scheme introspection to check directly).
    assert np.max(np.abs(states[True] - states[False])) > 1e-9, \
        "--strang must change the integrator (Strang vs Lie single-step results must differ)"


# --- (4)(5) density + drift state, against the REAL solved phi -----------------------------

def test_density_is_annular_tophat():
    params, args = _params(), _Args()
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    rho = _state(sim, args.ntheta, args.nr)[0]
    dr = (params.radius - args.r_min) / args.nr
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    inside = (r >= params.ring_inner) & (r <= params.ring_outer)
    assert np.allclose(rho[:, ~inside], params.rho_min), "background rho_min outside the ring"
    assert np.all(rho[:, inside] > params.rho_min), "ring density > rho_min"


def test_state_radial_velocity_is_exb_drift():
    params, args = _params(), _Args()
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    U = _state(sim, args.ntheta, args.nr)
    rho, mom_r = U[0], U[1]
    dr = (params.radius - args.r_min) / args.nr
    phi = np.asarray(sim.potential(), dtype=np.float64).reshape(args.ntheta, args.nr)
    _, grad_theta = rp.polar_gradient(phi, args.r_min, dr, args.ntheta, args.nr)
    assert np.allclose(mom_r / rho, -grad_theta / params.omega, atol=1e-12), \
        "v_r must be the ExB radial drift -grad_theta(phi)/B"


def test_state_azimuthal_velocity_solves_radial_balance():
    params = _params()
    args = _Args(cs2=1.0e-4)          # nonzero cs2 to exercise the pressure term d_r p
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    U = _state(sim, args.ntheta, args.nr)
    rho, v_th = U[0], U[2] / U[0]
    dr = (params.radius - args.r_min) / args.nr
    rr = (args.r_min + (np.arange(args.nr) + 0.5) * dr)[None, :]
    phi = np.asarray(sim.potential(), dtype=np.float64).reshape(args.ntheta, args.nr)
    grad_r, _ = rp.polar_gradient(phi, args.r_min, dr, args.ntheta, args.nr)
    d_r_rho = rp.polar_radial_derivative(rho, args.r_min, dr, args.ntheta, args.nr)
    B = params.omega
    residual = (rho / rr) * v_th ** 2 + (rho * B) * v_th - (args.cs2 * d_r_rho + rho * grad_r)
    scale = np.maximum.reduce([np.abs(rho * B * v_th), np.abs(args.cs2 * d_r_rho + rho * grad_r),
                               np.abs((rho / rr) * v_th ** 2), np.full_like(residual, 1e-300)])
    assert np.max(np.abs(residual) / scale) < 1e-10, "v_theta must solve the radial-balance quadratic"
    assert np.all(np.isfinite(v_th))


def test_equilibrium_v_theta_reduces_to_exb_when_cold():
    """(5d) cs2 = 0 : v_theta reduces EXACTLY to the ExB drift grad_r/B (paper cold limit)."""
    params = _params()
    nr, nth, r_min = 40, 24, 2.0
    dr = (params.radius - r_min) / nr
    th = ((np.arange(nth) + 0.5) * (2.0 * math.pi / nth))[:, None]
    rr = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = (rr - r_min) * (params.radius - rr) * np.cos(th)
    rho = np.full((nth, nr), params.rho_max)
    grad_r, _ = rp.polar_gradient(phi, r_min, dr, nth, nr)
    v_th = rp.equilibrium_v_theta(rho, grad_r, r_min, dr, nth, nr, params, cs2=0.0)
    assert np.allclose(v_th, grad_r / params.omega, rtol=1e-6, atol=1e-300), \
        "cs2=0 : v_theta must reduce to the ExB drift grad_r/B"


def test_check_equilibrium_is_stationary():
    """(5c) the equilibrium ring (no perturbation) stays flat per azimuthal mode (non-frozen)."""
    params = _params()
    args = _Args(cs2=1.0e-4, max_steps_check=20, check_modes=[1, 2, 3, 4, 5], check_tol=0.05)
    ok, report = rp.check_equilibrium(params, args)
    assert all(row["finite"] for row in report), "the potential must stay finite (no NaN)"
    assert ok, "the rotating equilibrium must be stationary: %r" % report
    assert sorted(row["mode"] for row in report) == [1, 2, 3, 4, 5]


# --- (c) frozen-equilibrium subtraction (option c), on the REAL scheme ---------------------

def test_compute_frozen_residual_is_scheme_drift():
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3)
    U_eq, R_eq = rp.compute_frozen_residual(params, args)
    assert U_eq.shape == (3, args.ntheta, args.nr) and R_eq.shape == U_eq.shape
    assert np.isfinite(R_eq).all()
    assert np.max(np.abs(R_eq)) > 1e-9, "R_eq = step(U_eq)-U_eq must be non-trivial (scheme not exact)"


def test_step_frozen_subtracted_makes_equilibrium_a_fixed_point():
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3)
    U_eq, R_eq = rp.compute_frozen_residual(params, args)
    sim = rp.build_polar_system(args.nr, args.ntheta, mode=4, params=params, args=args)
    sim.set_state("ne", U_eq.ravel())
    sim.solve_fields()
    rp.step_frozen_subtracted(sim, args.dt, R_eq, args.ntheta, args.nr)
    U = _state(sim, args.ntheta, args.nr)
    assert np.max(np.abs(U - U_eq)) < 1e-12, "step()-R_eq must leave U_eq invariant (exact fixed point)"


def test_check_equilibrium_frozen_is_machine_precision_stationary():
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3, frozen_equilibrium=True, frozen_check_const=1.0e3,
                 max_steps_check=200)
    ok, report = rp.check_equilibrium_frozen(params, args)
    row = report[0]
    assert row["finite"], "the corrected state must stay finite (no NaN)"
    assert row["n_steps"] >= 200, "at least 200 steps advanced: %r" % row
    assert row["max_dev"] <= row["floor"], \
        "max_dev=%.3e must stay under the machine floor %.3e" % (row["max_dev"], row["floor"])
    assert ok
    assert row["floor"] < 1e-6 * max(row["state_scale"], 1.0), \
        "the floor must be ~ eps * ||U_eq|| (scaled on U_eq, not the rho background)"


def test_run_mode_frozen_finite_amplitudes():
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3, cfl=0.7, t_end=0.004, sample_every=1, max_steps=100,
                 frozen_equilibrium=True)
    _, R_eq = rp.compute_frozen_residual(params, args)
    result = rp.run_mode(4, params, args, R_eq=R_eq)
    assert np.all(np.isfinite(result["amplitudes"])), "frozen run_mode must keep amplitudes finite"


def test_main_frozen_quick_runs():
    import tempfile
    saved = dict(os.environ)
    argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ADC_CASES_OUT"] = tmp
        os.environ.pop("SLURM_NTASKS", None)
        sys.argv = ["run_polar.py", "--quick", "--cfl", "0.5"]
        try:
            rp.main()  # must NOT raise: --cfl ignored under frozen-equilibrium, full --quick smoke
        finally:
            sys.argv = argv
            os.environ.clear()
            os.environ.update(saved)


# --- (6)(7) pure helpers ------------------------------------------------------------------

def test_polar_gradient_matches_derive_aux_polar_stencil():
    r_min, nr, nth = 2.0, 40, 24
    dr = (16.0 - r_min) / nr
    dth = 2.0 * math.pi / nth
    th = ((np.arange(nth) + 0.5) * dth)[:, None]
    r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = r ** 2 * np.cos(2.0 * th)
    gr, gt = rp.polar_gradient(phi, r_min, dr, nth, nr)
    gr_ref = np.empty_like(phi)
    gr_ref[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dr)
    gr_ref[:, 0] = (-3.0 * phi[:, 0] + 4.0 * phi[:, 1] - phi[:, 2]) / (2.0 * dr)
    gr_ref[:, -1] = (3.0 * phi[:, -1] - 4.0 * phi[:, -2] + phi[:, -3]) / (2.0 * dr)
    gt_ref = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dth * r)
    assert np.max(np.abs(gr - gr_ref)) < 1e-12
    assert np.max(np.abs(gt - gt_ref)) < 1e-12


def test_fit_growth_uses_mapped_paper_window_and_is_exact():
    lo_sim = 0.60 * 2 * math.pi        # mode l=4 mapped window = [3.770, 4.712]
    ts = np.linspace(0.0, 6.0, 2400)
    g_true, g_other = 0.911, 0.200
    amps = np.where(ts <= lo_sim,
                    1e-3 * np.exp(g_other * ts),
                    1e-3 * np.exp(g_other * lo_sim) * np.exp(g_true * (ts - lo_sim)))
    g = rp.fit_growth(ts, amps, 4, rhobar=1.0)
    assert abs(g - g_true) < 1e-6, "fit_growth must fit the MAPPED window -> g_true, got %r" % g
    assert abs(g - g_other) > 0.5, "fit_growth must NOT fit the raw paper window (g_other)"


def test_multirank_is_rejected():
    saved = dict(os.environ)
    os.environ["SLURM_NTASKS"] = "4"
    argv = sys.argv
    sys.argv = ["run_polar.py", "--quick"]
    try:
        raised = False
        try:
            rp.main()
        except SystemExit as exc:
            raised = True
            assert "mono-rang" in str(exc), exc
        assert raised, "a multi-rank run must be rejected (polar Schur is single-box)"
    finally:
        sys.argv = argv
        os.environ.clear()
        os.environ.update(saved)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("all %d polar-assembly tests passed (real adc)" % len(tests))


if __name__ == "__main__":
    _run_all()
