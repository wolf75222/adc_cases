#!/usr/bin/env python3
"""Self-asserting guard for the matlab_ref layer (ADC-349).

Pure NumPy, no build, no goldens (golden comparison is ADC-350). Checks internal
consistency against the locked REFERENCE.md facts: moment order, equilibrium
Maxwellian, Jacobian structure, the eigenmode phase-pin, the per-case
initializers, the compute_dt policy, and the wave L2 oracle. Cross-checks the
Maxwellian against hyqmom15/model.py + adc.moments when importable (skipped
otherwise).

Run: ``python3 hyqmom15/matlab_ref/check_matlab_ref.py`` (0 = OK, 1 = failure).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Run-by-path support: put hyqmom15/ on sys.path so the package imports.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from matlab_ref import (  # noqa: E402
    CASES,
    MOMENT_NAMES,
    compute_L2_error,
    compute_dt,
    exact_field,
    get_case,
    initial_field,
    init_diocotron_field,
    init_electrostatic_wave_field,
    init_magnetic_wave_field,
    linearized_jacobian_electrostatic,
    linearized_jacobian_fluid,
    linearized_jacobian_magnetostatic,
    maxwellian_moments,
    phase_pin,
)

EXPECTED_MI = np.array([1, 0, 1, 0, 3, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3], dtype=float)
WAVE = ("fluid_wave", "electrostatic_wave", "magnetic_wave")

# Reference moment vectors from Octave InitializeM4_15 (the canonical Maxwellian
# builder), %.17g, generated under RieMOM2D_Electrostatic_periodic. This is the
# always-on external oracle for maxwellian_moments (independent of adc); the
# per-case field goldens are ADC-350.
MAXWELLIAN_REF = [
    ((1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
     [1, 0, 1, 0, 3, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3]),
    ((2.0, 0.5, -0.3, 1.0, 0.0, 2.0),
     [2, 1, 2.5, 3.25, 9.125, -0.59999999999999998, -0.29999999999999999, -0.75,
      -0.97499999999999987, 4.1799999999999997, 2.0899999999999999, 5.2249999999999996,
      -3.6539999999999995, -1.8269999999999997, 26.176200000000001]),
    ((1.5, -0.2, 0.4, 1.0, 0.45, 0.5),
     [1.5, -0.30000000000000004, 1.5600000000000001, -0.91200000000000014, 4.8624000000000009,
      0.60000000000000009, 0.55499999999999994, 0.35400000000000015, 1.7412000000000003,
      0.98999999999999999, 0.34199999999999986, 1.4211, 0.99600000000000022, 1.1373000000000002,
      1.8834000000000004]),
]


def check_moment_order():
    assert MOMENT_NAMES == [
        "M00", "M10", "M20", "M30", "M40", "M01", "M11", "M21", "M31",
        "M02", "M12", "M22", "M03", "M13", "M04",
    ], "MOMENT_NAMES drifted from the canonical order"
    try:
        import adc.moments as gm  # noqa: PLC0415
    except Exception:
        return "moment order OK (adc not importable; skipped adc.moment_names cross-check)"
    assert MOMENT_NAMES == list(gm.moment_names(4)), "MOMENT_NAMES != adc.moments.moment_names(4)"
    return "moment order OK (matches adc.moments.moment_names(4))"


def check_equilibrium():
    for name, case in CASES.items():
        Mi = case.equilibrium_moments()
        assert Mi.shape == (15,), "%s Mi shape %s" % (name, Mi.shape)
        np.testing.assert_allclose(Mi, EXPECTED_MI, rtol=0, atol=1e-13,
                                   err_msg="%s equilibrium Mi != Maxwellian" % name)
    # Always-on external oracle: maxwellian_moments vs frozen Octave InitializeM4_15.
    for (rho, ux, uy, c20, c11, c02), ref in MAXWELLIAN_REF:
        got = maxwellian_moments(rho, ux, uy, c20, c11, c02)
        np.testing.assert_allclose(
            got, ref, rtol=0, atol=1e-12,
            err_msg="maxwellian_moments != Octave InitializeM4_15 for %r" % ((rho, ux, uy, c20, c11, c02),))
    # Bonus cross-check vs model.gaussian_state when adc/_adc is built (the CI path).
    try:
        import model  # noqa: PLC0415
    except Exception as exc:  # adc absent: the frozen Octave oracle above already ran
        return ("equilibrium OK (Maxwellian vs frozen Octave InitializeM4_15; "
                "model.gaussian_state skipped: %s)" % type(exc).__name__)
    for (rho, ux, uy, c20, c11, c02), _ in MAXWELLIAN_REF:
        np.testing.assert_allclose(
            maxwellian_moments(rho, ux, uy, c20, c11, c02),
            np.asarray(model.gaussian_state(rho, ux, uy, c20, c11, c02)),
            rtol=1e-13, atol=0, err_msg="maxwellian_moments != model.gaussian_state")
    return "equilibrium OK (Maxwellian vs frozen Octave InitializeM4_15 and model.gaussian_state)"


def check_jacobian_structure():
    kx, ky, lam, oc = 1.7, 2.3, 0.05, -20.0
    Jf = linearized_jacobian_fluid(kx, ky)
    Je = linearized_jacobian_electrostatic(kx, ky, lam)
    Jm = linearized_jacobian_magnetostatic(kx, ky, lam, oc)
    assert Jf.shape == Je.shape == Jm.shape == (15, 15)
    assert np.isrealobj(Jf) and np.isrealobj(Je) and np.iscomplexobj(Jm)
    # Electrostatic = fluid + exactly the six column-1 Poisson entries.
    kl2 = (kx ** 2 + ky ** 2) * lam ** 2
    diff = Je - Jf
    expected_col1 = {
        (1, 0): kx / kl2, (3, 0): 3 * kx / kl2, (5, 0): ky / kl2,
        (7, 0): ky / kl2, (10, 0): kx / kl2, (12, 0): 3 * ky / kl2,
    }
    nz = list(zip(*np.nonzero(diff)))
    assert set(nz) == set(expected_col1), "electrostatic-minus-fluid pattern wrong: %s" % nz
    for (r, c), v in expected_col1.items():
        assert abs(diff[r, c] - v) < 1e-15, "es col1 entry (%d,%d)" % (r, c)
    # Magnetostatic real part == electrostatic; imag only at the 20 cyclotron entries.
    np.testing.assert_allclose(Jm.real, Je, rtol=0, atol=1e-15,
                               err_msg="magnetostatic real part != electrostatic")
    upper = {(1, 5): 1, (2, 6): 2, (3, 7): 3, (4, 8): 4, (6, 9): 1, (7, 10): 2,
             (8, 11): 3, (10, 12): 1, (11, 13): 2, (13, 14): 1}
    lower = {(5, 1): -1, (6, 2): -1, (7, 3): -1, (8, 4): -1, (9, 6): -2, (10, 7): -2,
             (11, 8): -2, (12, 10): -3, (13, 11): -3, (14, 13): -4}
    expected_imag = {rc: n * oc for rc, n in {**upper, **lower}.items()}
    nz_im = list(zip(*np.nonzero(Jm.imag)))
    assert set(nz_im) == set(expected_imag), "cyclotron imag pattern wrong: %s" % nz_im
    for (r, c), v in expected_imag.items():
        assert abs(Jm.imag[r, c] - v) < 1e-12, "cyclotron entry (%d,%d)=%g" % (r, c, Jm.imag[r, c])
    return "jacobian structure OK (es=fluid+col1; ms.real=es; 20 cyclotron imag entries exact)"


def check_phase_pin():
    rng = np.random.default_rng(20260620)
    v = rng.standard_normal(15) + 1j * rng.standard_normal(15)
    p = phase_pin(v / np.linalg.norm(v))
    k = int(np.argmax(np.abs(p)))
    assert abs(p[k].imag) < 1e-14 and p[k].real > 0, "phase pin did not make max-|.| comp real-positive"
    # Idempotent and magnitude-preserving.
    np.testing.assert_allclose(np.abs(p), np.abs(v / np.linalg.norm(v)), rtol=1e-13, atol=0)
    np.testing.assert_allclose(phase_pin(p), p, rtol=1e-13, atol=1e-15)
    return "phase pin OK (max-|.| component real-positive, idempotent)"


def check_initializers():
    for name, case in CASES.items():
        res = initial_field(case)
        M = res.M
        assert M.shape == (15, case.Np, case.Np), "%s M shape %s" % (name, M.shape)
        assert np.isrealobj(M) and np.all(np.isfinite(M)), "%s M not finite/real" % name
        assert np.all(M[0] > 0), "%s density M00 must stay positive" % name
    # constant: uniform field, mass mean == rho0.
    cst = initial_field(get_case("constant")).M
    assert np.allclose(cst, cst[:, :1, :1], rtol=0, atol=1e-14), "constant field not uniform"
    assert abs(cst[0].mean() - 1.0) < 1e-14, "constant mass mean != rho0"
    # diocotron: density bounded by the ring; the two drift orientations differ in
    # the velocity moments but share the density.
    case = get_case("dicotron")
    std = init_diocotron_field(case, orientation="standard").M
    bug = init_diocotron_field(case, orientation="matlab_bug").M
    np.testing.assert_allclose(std[0], bug[0], rtol=0, atol=1e-15,
                               err_msg="diocotron density should not depend on drift orientation")
    assert not np.allclose(std, bug), "standard and matlab_bug drifts should differ"
    assert std[0].max() <= case.rho_max + 1e-12 and std[0].min() >= case.rho_min - 1e-12
    # wave cases: IC equals the t=0 exact solution -> L2 == 0.
    for name in WAVE:
        case = get_case(name)
        ic = initial_field(case).M
        l2 = compute_L2_error(ic, 0.0, case)
        assert l2 < 1e-12, "%s L2(IC, t=0) = %g, expected ~0" % (name, l2)
    return "initializers OK (shapes/finite/positive; constant uniform; diocotron bounded; wave L2(IC,0)~0)"


def check_dmax_policy():
    case = get_case("electrostatic_wave")
    intended = init_electrostatic_wave_field(case, dmax_policy="intended").max_speed
    as_written = init_electrostatic_wave_field(case, dmax_policy="as_written").max_speed
    assert intended is not None and as_written is not None
    # The two policies use different Jacobians (kmin,kmin vs kx,ky) -> different speeds.
    assert abs(intended - as_written) > 1e-6, "Dmax intended vs as_written should differ"
    return "Dmax policy OK (intended=diag(Dmax) at kmin differs from as_written=diag(D) at mode)"


def check_compute_dt():
    vmax = 7.0
    # fluid_wave: no source caps.
    f = get_case("fluid_wave")
    base = f.cfl * f.dx / vmax
    assert abs(compute_dt(vmax, f, 0.0) - base) < 1e-15, "fluid dt should be the bare CFL bound"
    # electrostatic_wave: es cap applies.
    e = get_case("electrostatic_wave")
    expect = min(e.cfl * e.dx / vmax, e.cfl * e.dx * vmax / e.omega_p ** 2)
    assert abs(compute_dt(vmax, e, 0.0) - expect) < 1e-15, "es dt cap wrong"
    # both-on (magnetic_wave, omega_p=20 != |omega_c|=40 so the caps differ): only the
    # omega_p^2 cap fires (elseif). vmax < omega_p makes a cap bind; the omega_c^2 cap
    # would bind harder (smaller) if it were wrongly applied, so confirm it is inert.
    d = get_case("magnetic_wave")
    vsmall = 10.0
    op_cap = min(d.cfl * d.dx / vsmall, d.cfl * d.dx * vsmall / d.omega_p ** 2)
    oc_cap = d.cfl * d.dx * vsmall / d.omega_c ** 2
    got = compute_dt(vsmall, d, 0.0)
    assert abs(got - op_cap) < 1e-15, "both-on dt must use omega_p^2 cap"
    assert oc_cap < op_cap and got > oc_cap, "omega_c^2 cap must be inert for both-on (elseif)"
    # final-time clamp.
    near = f.tmax - 0.5 * base
    assert abs(compute_dt(vmax, f, near) - (f.tmax - near)) < 1e-15, "final-time clamp wrong"
    return "compute_dt OK (CFL bound, es cap, both-on uses omega_p^2 only, final clamp)"


def check_branch_coverage():
    # (a) magnetic as_written wiring (D4 oversight) == the electrostatic-field IC, finite.
    mc = get_case("magnetic_wave")
    aw = init_magnetic_wave_field(mc, wiring="as_written").M
    es = init_electrostatic_wave_field(mc).M
    assert np.all(np.isfinite(aw)), "magnetic as_written IC not finite"
    np.testing.assert_allclose(aw, es, rtol=0, atol=1e-15,
                               err_msg="magnetic as_written should equal the electrostatic-field IC")
    # (b) diocotron matlab_bug orientation: finite field, positive density.
    bug = init_diocotron_field(get_case("dicotron"), orientation="matlab_bug").M
    assert np.all(np.isfinite(bug)) and np.all(bug[0] > 0), "matlab_bug diocotron field invalid"
    # (c) L2 round-trips the t>0 propagating phase (exercises the complex-eigenmode realification).
    for name in WAVE:
        case = get_case(name)
        t = 0.01
        l2 = compute_L2_error(exact_field(case, t), t, case)
        assert l2 < 1e-12, "%s L2(exact_field, t=%g) = %g, expected ~0" % (name, t, l2)
    # (d) diocotron max_speed uses the real-part rule (real, finite).
    ms = init_diocotron_field(get_case("dicotron")).max_speed
    assert ms is not None and np.isreal(ms) and np.isfinite(ms), "diocotron max_speed invalid"
    return "branch coverage OK (magnetic as_written, matlab_bug diocotron, L2 t>0, diocotron max_speed)"


def check_l2_scope():
    for name in ("dicotron", "constant"):
        try:
            compute_L2_error(np.zeros((15, 4, 4)), 0.0, get_case(name))
        except ValueError:
            pass
        else:
            raise AssertionError("compute_L2_error must raise for %r" % name)
    return "L2 scope OK (raises on dicotron/constant; wave L2(IC,0)~0 checked above)"


CHECKS = [
    check_moment_order,
    check_equilibrium,
    check_jacobian_structure,
    check_phase_pin,
    check_initializers,
    check_dmax_policy,
    check_compute_dt,
    check_branch_coverage,
    check_l2_scope,
]


def main() -> int:
    failures = []
    for fn in CHECKS:
        try:
            msg = fn()
            print("  OK   %-20s %s" % (fn.__name__, msg or ""))
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print("  FAIL %-20s %s" % (fn.__name__, exc))
    if failures:
        print("CHECK-MATLAB-REF: %d/%d checks FAILED" % (len(failures), len(CHECKS)), file=sys.stderr)
        return 1
    print("CHECK-MATLAB-REF: OK (%d checks)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
