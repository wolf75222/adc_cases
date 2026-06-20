"""Linearized 15-moment Jacobians of the new periodic Matlab reference.

Bit-faithful NumPy ports of ``linearized_Jacobian_fluid.m``,
``linearized_Jacobian_electrostatic.m`` and ``linearized_Jacobian_magnetostatic.m``
from ``RieMOM2D_Electrostatic_periodic``, plus the shared eigenmode helper the
``init_*_wave_field.m`` routines use to seed the sinusoidal perturbation.

The entry tables below are written with the Matlab 1-based ``(row, col)`` indices
so they read against the source; :func:`_fill` subtracts 1. The fluid and
electrostatic matrices are real; the magnetostatic matrix is complex (the
``1i*omega_c`` cyclotron coupling). See ``REFERENCE.md`` D3/D4 and ADC-349.
"""
from __future__ import annotations

import numpy as np

NMOM = 15


def _fill(entries, dtype=float) -> np.ndarray:
    """Build a 15x15 matrix from 1-based ``(row, col, value)`` triples."""
    J = np.zeros((NMOM, NMOM), dtype=dtype)
    for r, c, v in entries:
        J[r - 1, c - 1] = v
    return J


def linearized_jacobian_fluid(kx: float, ky: float) -> np.ndarray:
    """Real advection-moment Jacobian (``linearized_Jacobian_fluid.m``)."""
    return _fill([
        (1, 2, kx), (1, 6, ky),
        (2, 3, kx), (2, 7, ky),
        (3, 4, kx), (3, 8, ky),
        (4, 5, kx), (4, 9, ky),
        (5, 2, -6 * kx), (5, 4, 7 * kx), (5, 6, -3 * ky), (5, 8, 6 * ky),
        (6, 7, kx), (6, 10, ky),
        (7, 8, kx), (7, 11, ky),
        (8, 9, kx), (8, 12, ky),
        (9, 2, -3 * ky), (9, 4, ky), (9, 6, -3 * kx), (9, 8, 6 * kx), (9, 11, 3 * ky),
        (10, 11, kx), (10, 13, ky),
        (11, 12, kx), (11, 14, ky),
        (12, 2, -3 * kx), (12, 4, kx), (12, 6, -3 * ky), (12, 8, 3 * ky), (12, 11, 3 * kx), (12, 13, ky),
        (13, 14, kx), (13, 15, ky),
        (14, 2, -3 * ky), (14, 6, -3 * kx), (14, 8, 3 * kx), (14, 11, 6 * ky), (14, 13, kx),
        (15, 2, -3 * kx), (15, 6, -6 * ky), (15, 11, 6 * kx), (15, 13, 7 * ky),
    ])


def linearized_jacobian_electrostatic(kx: float, ky: float, lam: float) -> np.ndarray:
    """Electrostatic Jacobian: fluid plus the column-1 Poisson coupling.

    ``lam`` is ``adim_debye_length``; the coupling denominator is
    ``kl2 = (kx^2 + ky^2) * lam^2``. Note J(6,1) and J(8,1) are both ``ky/kl2``
    (no factor 3) while J(4,1) and J(13,1) carry the factor 3 (copied literally).
    """
    J = linearized_jacobian_fluid(kx, ky)
    kl2 = (kx ** 2 + ky ** 2) * lam ** 2
    if kl2 == 0:
        raise ValueError(
            "electrostatic Jacobian undefined at kx=ky=0 (zero Poisson wavenumber)"
        )
    for r, c, v in [
        (2, 1, kx / kl2),
        (4, 1, 3 * kx / kl2),
        (6, 1, ky / kl2),
        (8, 1, ky / kl2),
        (11, 1, kx / kl2),
        (13, 1, 3 * ky / kl2),
    ]:
        J[r - 1, c - 1] = v
    return J


def linearized_jacobian_magnetostatic(kx: float, ky: float, lam: float, omega_c: float) -> np.ndarray:
    """Complex magnetostatic Jacobian: electrostatic plus ``1i*omega_c`` coupling.

    The cyclotron terms are NOT symmetric (e.g. J(3,7)=+2i but J(7,3)=-1i);
    every entry is transcribed literally. None collide with the real fluid /
    electrostatic positions, so they are pure additions.
    """
    J = linearized_jacobian_electrostatic(kx, ky, lam).astype(complex)
    oc = omega_c
    for r, c, v in [
        (2, 6, 1j * oc), (3, 7, 2j * oc), (4, 8, 3j * oc), (5, 9, 4j * oc),
        (6, 2, -1j * oc),
        (7, 3, -1j * oc), (7, 10, 1j * oc),
        (8, 4, -1j * oc), (8, 11, 2j * oc),
        (9, 5, -1j * oc), (9, 12, 3j * oc),
        (10, 7, -2j * oc),
        (11, 8, -2j * oc), (11, 13, 1j * oc),
        (12, 9, -2j * oc), (12, 14, 2j * oc),
        (13, 11, -3j * oc),
        (14, 12, -3j * oc), (14, 15, 1j * oc),
        (15, 14, -4j * oc),
    ]:
        J[r - 1, c - 1] = v
    return J


def matlab_sort_indices(w: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Deterministic, BLAS-robust eigenvalue ordering for mode selection.

    Octave's ``sort`` orders a real spectrum ascending by value and a complex
    spectrum ascending by magnitude. Two cases make that raw rule non-portable
    between Octave and NumPy, so this canonical version reproduces the *selection*
    while being robust to the ~1e-13 differences between their LAPACK builds (the
    identical rule is applied in the ADC-350 Octave generators so the chosen mode
    matches on any BLAS):

    * a numerically real spectrum that NumPy may return as complex dtype
      (``imag == +0``) is classified by ``max|imag| <= tol*scale`` and sorted
      ascending by real value, as Octave's ``isreal`` array would be (otherwise
      ``iscomplexobj`` flips the rule per-build);
    * a genuinely complex spectrum is sorted ascending by magnitude, but
      near-magnitude ties (equal to ``tol`` relative) are broken by ``(real,
      imag)``. The magnetic_wave mode-15 is such a tie: the +/-168.16 modes have
      magnitudes equal to ~1e-13 but real parts 336 apart, so a raw magnitude sort
      picks one or the other depending on rounding, while the real-part tie-break
      is stable.
    """
    w = np.asarray(w)
    scale = max(1.0, float(np.max(np.abs(w)))) if w.size else 1.0
    if float(np.max(np.abs(w.imag))) <= tol * scale:
        return np.argsort(w.real, kind="stable")
    q = np.round(np.abs(w) / scale, 9)
    return np.lexsort((w.imag, w.real, q))


def phase_pin(vec: np.ndarray) -> np.ndarray:
    """Fix the eigenvector phase/sign gauge deterministically.

    Rotates ``vec`` so its largest-magnitude component is real and positive. The
    eigenvector is only defined up to a complex unit scalar; LAPACK leaves that
    free and Octave / NumPy disagree on it. The identical pin is applied in the
    ADC-350 Octave generators so the seeded IC matches bit-for-bit (the locked
    "phase-pin both sides" decision).
    """
    k = int(np.argmax(np.abs(vec)))
    pivot = vec[k]
    if pivot != 0:
        vec = vec * (np.conj(pivot) / np.abs(pivot))
    return vec


def eigenmode(J: np.ndarray, mode: int):
    """Select the ``mode``-th eigenpair the way ``init_*_wave_field.m`` does.

    Returns ``(lam, vec)`` where ``mode`` is the 1-based Matlab column index into
    the eigenvalue-sorted eigenvectors, ``vec`` is L2-normalized and phase-pinned.
    ``vec`` stays complex; callers take the real part for the physical state.

    The phase-pin fixes the gauge uniquely only when the selected eigenvalue is
    simple. All committed wave cases select ``mode=15`` (the largest eigenvalue),
    which is simple, so Octave/NumPy parity holds. A future case selecting a mode
    inside a degenerate cluster would have an ambiguous eigenspace basis that the
    phase-pin alone cannot reconcile.
    """
    w, V = np.linalg.eig(J)
    order = matlab_sort_indices(w)
    w_sorted = w[order]
    V_sorted = V[:, order]
    lam = w_sorted[mode - 1]
    vec = V_sorted[:, mode - 1]
    vec = vec / np.linalg.norm(vec)
    vec = phase_pin(vec)
    return lam, vec
