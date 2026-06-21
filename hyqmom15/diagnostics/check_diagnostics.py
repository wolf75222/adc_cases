#!/usr/bin/env python3
"""Guard for the hyqmom15 diagnostics layer (ADC-383).

Pure Python, no adc build: checks that the vectorized realizability maps match
the scalar relaxation.py oracle cell-by-cell, that a realizable (Maxwellian)
field passes while an unrealizable one is flagged, that the per-case symmetry
residuals are ~0 for the expected structure and large when it is broken, and
that the non-fatal recovery policy accepts a transient violation that recovers.

Run: ``python3 hyqmom15/diagnostics/check_diagnostics.py`` (0 = OK, 1 = mismatch).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # diagnostics modules
sys.path.insert(0, str(HERE.parent))          # hyqmom15/ (relaxation, matlab_ref)

import realizability as Z  # noqa: E402
import relaxation as R  # noqa: E402
import symmetry as Y  # noqa: E402
from matlab_ref import maxwellian_moments  # noqa: E402


def _realizable_field(ny=6, nx=7, seed=0):
    rng = np.random.default_rng(seed)
    field = np.empty((15, ny, nx))
    for a in range(ny):
        for b in range(nx):
            rho = 0.5 + rng.random()
            u, v = rng.normal() * 0.3, rng.normal() * 0.3
            c20, c02 = 0.5 + rng.random(), 0.5 + rng.random()
            r = (rng.random() - 0.5) * 0.8  # |r| < 1 -> covariance PD
            field[:, a, b] = maxwellian_moments(rho, u, v, c20, r, c02)
    return field


def _cell_centers(n):
    i = np.arange(n)
    return -0.5 + (i + 0.5) / n


def check_oracle_parity():
    field = _realizable_field()
    _, s_vec = Z.standardized_moments(field)
    lmin, lmid, lmax = Z._p2p2_eigs(s_vec)
    ny, nx = field.shape[1:]
    ds = dl = 0.0
    for a in range(ny):
        for b in range(nx):
            _, so = R.m2cs4(field[:, a, b])
            ds = max(ds, max(abs(so[pq] - s_vec[pq][a, b]) for pq in so))
            p = R.p2p2_2d(so[(0, 3)], so[(0, 4)], so[(1, 1)], so[(1, 2)], so[(1, 3)],
                          so[(2, 1)], so[(2, 2)], so[(3, 0)], so[(3, 1)], so[(4, 0)])
            lo = np.sort(np.real(np.linalg.eigvals(p)))
            dl = max(dl, abs(lo[0] - lmin[a, b]), abs(lo[1] - lmid[a, b]), abs(lo[2] - lmax[a, b]))
    assert ds < 1e-12, "standardized moments differ from relaxation.m2cs4: %.2e" % ds
    assert dl < 1e-12, "p2p2 eigenvalues differ from relaxation.p2p2_2d: %.2e" % dl
    return "oracle parity OK (S diff %.1e, lam diff %.1e vs relaxation.py)" % (ds, dl)


def check_realizable_and_not():
    good, real = Z.field_realizability(_realizable_field())
    assert np.all(real["all"]), "a Maxwellian field must be fully realizable"
    assert Z.summarize(good, real)["frac_nonrealizable"] == 0.0
    bad = _realizable_field(seed=1)
    bad[0, 0, 0] = -1.0  # negative density in one cell
    _, realbad = Z.field_realizability(bad)
    assert not realbad["positivity"][0, 0], "negative M00 must fail positivity"
    assert not np.all(realbad["all"]), "the unrealizable field must be flagged"
    return "realizable/unrealizable OK (Maxwellian passes, negative M00 flagged)"


def _field_with_m00(m00):
    field = np.ones((15,) + m00.shape)
    field[0] = m00
    return field


def check_symmetry():
    n = 16
    x = _cell_centers(n)[None, :]
    y = _cell_centers(n)[:, None]
    # constant: uniform -> ~0; a gradient -> large.
    uni = Y.uniformity_residual(_field_with_m00(np.ones((n, n))))
    grad = Y.uniformity_residual(_field_with_m00(1.0 + 0.5 * x * np.ones((n, n))))
    assert uni < 1e-12 and grad > 1e-3, "uniformity residual: %.2e vs %.2e" % (uni, grad)
    # fluid_wave: uniform in y -> axis 0 residual ~0; add y-variation -> large.
    m_y = 1.0 + 0.1 * np.sin(2 * np.pi * 2 * x) * np.ones((n, n))
    res_y = Y.axis_uniformity_residual(_field_with_m00(m_y), axis=0)
    broke_y = Y.axis_uniformity_residual(_field_with_m00(m_y + 0.1 * y), axis=0)
    assert res_y < 1e-12 and broke_y > 1e-3, "y-uniformity: %.2e vs %.2e" % (res_y, broke_y)
    # dicotron: a 4-fold (rotation-invariant) M00 -> ~0; asymmetric -> large.
    r2 = x ** 2 + y ** 2
    rot = Y.rotational_residual(_field_with_m00(np.exp(-20 * r2)))
    asym = Y.rotational_residual(_field_with_m00(np.exp(-20 * r2) + 0.3 * x * np.ones((n, n))))
    assert rot < 1e-9 and asym > 1e-2, "rotational: %.2e vs %.2e" % (rot, asym)
    # magnetic_wave: a clean oblique mode -> purity ~0; add other modes -> large.
    kx, ky = 2 * np.pi * 2, 2 * np.pi * 4
    clean = 1.0 + 0.1 * np.sin(kx * x + ky * y)
    pure = Y.mode_purity_residual(_field_with_m00(clean * np.ones((n, n))), kx, ky)
    noisy = Y.mode_purity_residual(_field_with_m00(clean + 0.1 * np.sin(2 * np.pi * x)), kx, ky)
    assert pure < 1e-9 and noisy > 1e-2, "mode purity: %.2e vs %.2e" % (pure, noisy)
    return "symmetry OK (uniform, y-uniform, rotational, mode-purity ~0 vs broken)"


def check_recovery():
    clean = _realizable_field()
    dip = clean.copy()
    dip[0, 0, 0] = -1.0  # transient positivity violation in the middle snapshot
    seq = [clean, dip, clean]  # violation recovers by the last snapshot
    res = {r["name"]: r for r in Z.evaluate(seq)}
    pos = res["positivity"]
    assert pos["failed_steps"] == [1], "positivity should fail only at step 1: %s" % pos
    assert pos["recovered"] and pos["passed"], "transient + recovery must pass (must_recover)"
    strict = Z.RealizabilityCheck("positivity_strict", Z._ok_positivity, must_recover=False)
    sres = Z.evaluate(seq, [strict])[0]
    assert not sres["passed"], "a strict check must fail on any violation"
    return "recovery OK (transient violation recovers -> pass; strict -> fail)"


CHECKS = [check_oracle_parity, check_realizable_and_not, check_symmetry, check_recovery]


def main() -> int:
    failures = []
    for fn in CHECKS:
        try:
            print("  OK   %-22s %s" % (fn.__name__, fn()))
        except Exception as exc:  # noqa: BLE001
            failures.append(fn.__name__)
            print("  FAIL %-22s %s" % (fn.__name__, exc))
    if failures:
        print("CHECK-DIAGNOSTICS: %d/%d FAILED" % (len(failures), len(CHECKS)), file=sys.stderr)
        return 1
    print("CHECK-DIAGNOSTICS: OK (%d checks)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
