#!/usr/bin/env python3
"""Per-cell realizability diagnostics for hyqmom15 moment fields (ADC-383).

Given a moment field ``(15, ny, nx)`` this reports, per cell, the HyQMOM
realizability quantities the Matlab RieMOM2D reference plots: the standardized
moments ``S_pq``, the realizability Hankel matrix ``p2p2`` and its eigenvalues
``lam_min/mid/max``, the univariate bounds ``H20``/``H02``/``H2``, the
correlation bound ``Del1 = 1 - S11^2``, the covariance determinant
``C20*C02 - C11^2``, and a per-condition realizable mask.

The formulas are the pure-NumPy oracle of ``hyqmom15/relaxation.py`` (``m2cs4``,
``p2p2_2d``), reused here vectorized over the whole field; the build-free guard
``check_diagnostics.py`` asserts these vectorized maps match the scalar oracle
cell-by-cell. Monitoring is snapshot-interval and non-fatal: a transient
violation that recovers is acceptable (see :class:`RealizabilityCheck` and
:func:`evaluate`), so the simulation is never stopped.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from matlab_ref import MOMENT_PQ

# Matlab realizability clamp constants (relaxation.py / realizability_plots.m).
SMALL = 1.0e-6        # univariate border floor (H20, H02)
LAMIN = 1.0e-12       # p2p2 minimal-eigenvalue gate
S11_EPS = 1.0e-6      # realizable correlation: |S11| < 1 - S11_EPS

_IDX = {pq: k for k, pq in enumerate(MOMENT_PQ)}


def standardized_moments(field):
    """Vectorized ``M_pq`` field ``(15, ny, nx)`` -> central ``C`` and standardized ``S``.

    Same transform as ``relaxation.m2cs4``, applied over the whole grid. Returns
    the pair ``(C, S)`` of dicts of ``(ny, nx)`` arrays keyed by ``(p, q)``; ``S``
    only holds the orders ``p + q >= 2``.
    """
    field = np.asarray(field, dtype=float)
    m00 = field[0]
    mn = {pq: field[_IDX[pq]] / m00 for pq in MOMENT_PQ}
    u, v = mn[(1, 0)], mn[(0, 1)]
    central = {}
    for p, q in MOMENT_PQ:
        acc = np.zeros_like(m00)
        for i in range(p + 1):
            for j in range(q + 1):
                acc = acc + (math.comb(p, i) * math.comb(q, j)
                             * (-u) ** (p - i) * (-v) ** (q - j) * mn[(i, j)])
        central[(p, q)] = acc
    with np.errstate(invalid="ignore"):  # nan for unrealizable (negative-variance) cells
        sx, sy = np.sqrt(central[(2, 0)]), np.sqrt(central[(0, 2)])
    std = {pq: central[pq] / (sx ** pq[0] * sy ** pq[1])
           for pq in MOMENT_PQ if pq[0] + pq[1] >= 2}
    return central, std


def _p2p2_eigs(std):
    """``lam_min/mid/max`` maps of the ``p2p2`` matrix from standardized-moment maps.

    Vectorized transcription of ``relaxation.p2p2_2d`` (identical ``t``-terms),
    assembled into ``(ny, nx, 3, 3)`` with the Matlab column-major layout
    ``M[i, j] = flat[i + 3*j]`` and reduced with ``np.linalg.eigvals``.
    """
    a03, a04, a11 = std[(0, 3)], std[(0, 4)], std[(1, 1)]
    a12, a13, a21 = std[(1, 2)], std[(1, 3)], std[(2, 1)]
    a22, a30, a31, a40 = std[(2, 2)], std[(3, 0)], std[(3, 1)], std[(4, 0)]
    t2 = a03 * a12
    t3 = a03 * a21
    t4 = a12 * a21
    t5 = a12 * a30
    t6 = a21 * a30
    t7 = a11 ** 2
    t8 = a11 ** 3
    t9 = a12 ** 2
    t10 = a21 ** 2
    t12 = a03 * a11 * a30
    t15 = -a13
    t16 = -a31
    t11 = a11 * t3
    t13 = a11 * t4
    t14 = a11 * t5
    t17 = -t3
    t18 = -t5
    t19 = a11 * t9
    t20 = a13 * t7
    t21 = a11 * t10
    t22 = a22 * t7
    t23 = a31 * t7
    t24 = -t7
    t25 = -t8
    t26 = t7 - 1.0
    t27 = -t11
    t28 = -t14
    t29 = -t19
    t30 = -t21
    t31 = -t22
    with np.errstate(divide="ignore", invalid="ignore"):
        t32 = 1.0 / t26
        t33 = a22 + t12 + t13 + t17 + t18 + t26 + t31
        t34 = a11 + t2 + t4 + t15 + t20 + t25 + t27 + t29
        t35 = a11 + t4 + t6 + t16 + t23 + t25 + t28 + t30
        t36 = t32 * t33
        t38 = t32 * t34
        t39 = t32 * t35
        t37 = -t36
        f0 = t32 * (-a40 + t10 + t24 - a11 * t6 * 2.0 + a40 * t7 + a30 ** 2 + 1.0)
        f4 = t32 * (-a22 + t7 + t9 + t10 - t13 * 2.0 + t22 + t7 * t24)
        f8 = t32 * (-a04 + t9 + t24 + a04 * t7 - a11 * t2 * 2.0 + a03 ** 2 + 1.0)
    # flat = [f0, t39, t37, t39, f4, t38, t37, t38, f8]; M[i, j] = flat[i + 3*j].
    row0 = np.stack([f0, t39, t37], axis=-1)
    row1 = np.stack([t39, f4, t38], axis=-1)
    row2 = np.stack([t37, t38, f8], axis=-1)
    mats = np.stack([row0, row1, row2], axis=-2)  # (ny, nx, 3, 3)
    # np.linalg.eigvals raises on non-finite input; unrealizable cells (|S11| -> 1
    # or negative variance) produce nan/inf entries. Compute eigenvalues on a
    # sanitized copy, then mark those cells nan (they fail the p2p2_psd mask).
    finite = np.isfinite(mats).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], mats, np.eye(3))
    with np.errstate(invalid="ignore"):
        lam = np.sort(np.real(np.linalg.eigvals(safe)), axis=-1)  # ascending
    lam = np.where(finite[..., None], lam, np.nan)
    return lam[..., 0], lam[..., 1], lam[..., 2]


def field_realizability(field):
    """Per-cell realizability maps and a per-condition realizable mask.

    Returns ``(maps, realizable)``. ``maps`` holds ``(ny, nx)`` arrays
    (``M00``, ``C20``, ``C02``, ``det_cov``, ``lam_min/mid/max``, ``H20``,
    ``H02``, ``H2``, ``abs_S11``); ``realizable`` holds boolean masks per
    condition plus the conjunction under key ``"all"``.
    """
    field = np.asarray(field, dtype=float)
    central, std = standardized_moments(field)
    m00 = field[0]
    c20, c02, c11 = central[(2, 0)], central[(0, 2)], central[(1, 1)]
    lam_min, lam_mid, lam_max = _p2p2_eigs(std)
    s11 = std[(1, 1)]
    h20 = std[(4, 0)] - std[(3, 0)] ** 2 - 1.0
    h02 = std[(0, 4)] - std[(0, 3)] ** 2 - 1.0
    with np.errstate(invalid="ignore"):
        h2 = np.sqrt(np.abs(std[(4, 0)] * std[(0, 4)])) - np.abs(std[(0, 3)] * std[(3, 0)]) - 1.0
    det_cov = c20 * c02 - c11 ** 2
    maps = {
        "M00": m00, "C20": c20, "C02": c02, "det_cov": det_cov,
        "lam_min": lam_min, "lam_mid": lam_mid, "lam_max": lam_max,
        "H20": h20, "H02": h02, "H2": h2, "abs_S11": np.abs(s11),
    }
    realizable = {
        "positivity": m00 > 0.0,
        "variance_x": c20 > 0.0,
        "variance_y": c02 > 0.0,
        "covariance_pd": det_cov >= 0.0,
        "p2p2_psd": lam_min >= -LAMIN,
        "s11_bound": np.abs(s11) < 1.0 - S11_EPS,
        "h20": h20 > 0.0,
        "h02": h02 > 0.0,
    }
    realizable["all"] = np.logical_and.reduce([np.asarray(v) for v in realizable.values()])
    return maps, realizable


def summarize(maps, realizable):
    """Scalar reductions over a field's realizability maps (one row per snapshot)."""
    return {
        "M00_min": float(np.nanmin(maps["M00"])),
        "M00_max": float(np.nanmax(maps["M00"])),
        "frac_negative_M00": float(np.mean(maps["M00"] <= 0.0)),
        "lam_min": float(np.nanmin(maps["lam_min"])),
        "frac_nonrealizable": float(np.mean(~realizable["all"])),
        "H20_min": float(np.nanmin(maps["H20"])),
        "H02_min": float(np.nanmin(maps["H02"])),
        "max_abs_S11": float(np.nanmax(maps["abs_S11"])),
    }


@dataclasses.dataclass(frozen=True)
class RealizabilityCheck:
    """A named, non-fatal realizability assertion over a moment field.

    Attributes:
        name: Identifier reported in the campaign table.
        fn: ``fn(maps, realizable) -> (ny, nx) boolean "ok" mask`` (True passes).
        must_recover: If True the check passes overall as long as any transient
            failure has recovered (the last checked snapshot is clean); if False
            a single failing snapshot fails the check.
        interval: Check only every ``interval``-th snapshot.
        description: Human-readable summary.
    """

    name: str
    fn: Callable[[dict, dict], np.ndarray]
    must_recover: bool = True
    interval: int = 1
    description: str = ""


def _ok_positivity(maps, realizable):
    return realizable["positivity"]


def _ok_variance_pd(maps, realizable):
    return realizable["variance_x"] & realizable["variance_y"] & realizable["covariance_pd"]


def _ok_p2p2(maps, realizable):
    return realizable["p2p2_psd"]


def _ok_s11(maps, realizable):
    return realizable["s11_bound"]


def _ok_hankel(maps, realizable):
    return realizable["h20"] & realizable["h02"]


DEFAULT_CHECKS = [
    RealizabilityCheck("positivity", _ok_positivity, description="M00 > 0 (density positive)"),
    RealizabilityCheck("variance_pd", _ok_variance_pd,
                       description="C20 > 0, C02 > 0, C20*C02 - C11^2 >= 0"),
    RealizabilityCheck("p2p2_psd", _ok_p2p2, description="lam_min(p2p2) >= -1e-12"),
    RealizabilityCheck("s11_bound", _ok_s11, description="|S11| < 1 - 1e-6"),
    RealizabilityCheck("hankel", _ok_hankel, description="H20 > 0 and H02 > 0"),
]


def _as_field(snap):
    """Accept a raw ``(15, ny, nx)`` array or a snapshot with a ``.moments`` attribute."""
    return np.asarray(getattr(snap, "moments", snap), dtype=float)


def evaluate(snaps, checks=None):
    """Run realizability checks over a time-ordered snapshot sequence (non-fatal).

    Args:
        snaps: Time-ordered moment fields ``(15, ny, nx)`` or objects with a
            ``.moments`` attribute (e.g. ``plots.snapshots.Snapshot``).
        checks: Checks to run; defaults to :data:`DEFAULT_CHECKS`.

    Returns:
        One result dict per check: ``name``, ``n_checked``, ``failed_steps``
        (snapshot indices with any failing cell), ``recovered`` (the last
        checked snapshot is clean), and ``passed`` (``recovered`` when
        ``must_recover`` else no failure at all). Never raises on a violation.
    """
    checks = list(checks) if checks is not None else DEFAULT_CHECKS
    diags = [field_realizability(_as_field(s)) for s in snaps]
    results = []
    for chk in checks:
        steps = list(range(0, len(diags), max(1, chk.interval)))
        failed = [k for k in steps if not np.all(chk.fn(*diags[k]))]
        recovered = (steps[-1] not in failed) if (failed and steps) else True
        passed = recovered if chk.must_recover else not failed
        results.append({
            "name": chk.name,
            "n_checked": len(steps),
            "failed_steps": failed,
            "recovered": recovered,
            "passed": passed,
        })
    return results
