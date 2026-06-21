#!/usr/bin/env python3
"""Per-case spatial-symmetry residuals for hyqmom15 snapshots (ADC-383).

Each RieMOM2D case has a known structure the ADC solution should preserve; the
residuals below are ~0 when the structure holds and grow when the solver breaks
it. They are computed on the density M00 (which transforms simply under the
relevant symmetry):

  * ``constant``           : fully uniform (no spatial variation);
  * ``fluid_wave``         : uniform in y (ky = 0), a sin(kx x) wave in x;
  * ``electrostatic_wave`` : uniform in x (kx = 0), a sin(ky y) wave in y;
  * ``magnetic_wave``      : a single oblique Fourier mode sin(kx x + ky y);
  * ``dicotron``           : 4-fold rotational (the sin(4 theta) ring is
                             invariant under a 90 degree rotation).

Domain is ``[-0.5, 0.5]^2`` periodic with ``L = 1`` (so a wavenumber ``k`` maps
to the integer Fourier index ``k / (2 pi)``).
"""
from __future__ import annotations

import numpy as np

L = 1.0  # domain length (XMAX - XMIN); System(L=1.0)
_TINY = 1.0e-300


def _m00(field):
    return np.asarray(field, dtype=float)[0]


def _scale(arr):
    return float(np.abs(arr).mean()) + _TINY


def uniformity_residual(field):
    """Relative spatial spread of M00 (``~0`` for the uniform ``constant`` case)."""
    m = _m00(field)
    return float(np.std(m) / _scale(m))


def axis_uniformity_residual(field, axis):
    """Relative variation of M00 along ``axis`` (0 = y/rows, 1 = x/cols).

    ``~0`` when the field is constant along that axis: ``axis=0`` for
    ``fluid_wave`` (uniform in y), ``axis=1`` for ``electrostatic_wave``
    (uniform in x).
    """
    m = _m00(field)
    return float(np.std(m, axis=axis).max() / _scale(m))


def rotational_residual(field):
    """Relative L2 of ``M00 - rot90(M00)`` (``~0`` for a 4-fold symmetric ring).

    A 90 degree rotation of the density on a centered square grid is
    ``np.rot90``; the diocotron ``sin(4 theta)`` ring is invariant under it.
    """
    m = _m00(field)
    if m.shape[0] != m.shape[1]:
        return float("nan")  # rotation symmetry only defined on a square grid
    diff = m - np.rot90(m)
    denom = float(np.linalg.norm(m)) + _TINY
    return float(np.linalg.norm(diff) / denom)


def mode_purity_residual(field, kx, ky):
    """Fraction of the M00 perturbation energy NOT in the ``(kx, ky)`` Fourier mode.

    ``~0`` for a clean eigenmode ``sin(kx x + ky y)``; grows when the solution
    leaks energy into other modes.
    """
    m = _m00(field)
    ny, nx = m.shape
    power = np.abs(np.fft.fft2(m - m.mean())) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    nkx = int(round(kx * L / (2.0 * np.pi)))
    nky = int(round(ky * L / (2.0 * np.pi)))
    peak = power[nky % ny, nkx % nx] + power[(-nky) % ny, (-nkx) % nx]
    return float(1.0 - peak / total)


# case name -> (residual function taking the field, human-readable description).
# magnetic_wave/wave cases that need (kx, ky) are dispatched in symmetry_residual.
SYMMETRY_BY_CASE = {
    "constant": (uniformity_residual, "uniform field (no spatial variation)"),
    "fluid_wave": (lambda f: axis_uniformity_residual(f, axis=0), "uniform in y (ky = 0)"),
    "electrostatic_wave": (lambda f: axis_uniformity_residual(f, axis=1), "uniform in x (kx = 0)"),
    "magnetic_wave": (None, "single oblique Fourier mode sin(kx x + ky y)"),
    "dicotron": (rotational_residual, "4-fold rotational (90 degree invariant ring)"),
}


def symmetry_residual(field, case):
    """Symmetry residual for ``field`` under the expected symmetry of ``case``.

    ``case`` is the Matlab case name (e.g. ``"fluid_wave"``). For the oblique
    ``magnetic_wave`` the wavenumbers are read from ``matlab_ref.get_case``.
    Returns ``~0`` when the expected structure holds.
    """
    if case == "magnetic_wave":
        from matlab_ref import get_case
        c = get_case(case)
        return mode_purity_residual(field, c.kx, c.ky)
    try:
        fn, _ = SYMMETRY_BY_CASE[case]
    except KeyError:
        raise KeyError("no symmetry residual for case %r; have %s"
                       % (case, ", ".join(sorted(SYMMETRY_BY_CASE)))) from None
    if fn is None:
        raise KeyError("case %r needs wavenumbers; handled above" % case)
    return float(fn(field))
