"""L2 error against the linear-eigenmode exact solution (``compute_L2_error.m``).

Covers the three wave cases only; ``dicotron`` and ``constant`` have no analytic
L2 oracle in the Matlab (REFERENCE.md D9) and raise here, exactly as
``compute_L2_error.m`` errors on them.
"""
from __future__ import annotations

import numpy as np

from . import params as P
from .linearized import (
    eigenmode,
    linearized_jacobian_electrostatic,
    linearized_jacobian_fluid,
    linearized_jacobian_magnetostatic,
)

_WAVE = {"fluid_wave", "electrostatic_wave", "magnetic_wave"}


def _case_jacobian(case: P.Case) -> np.ndarray:
    """Jacobian used by ``compute_L2_error.m`` for ``case`` (magnetic uses the
    magnetostatic one, matching the source even though the as-shipped IC wires the
    electrostatic field; the intended IC also uses magnetostatic so they agree)."""
    lam = case.adim_debye_length
    if case.name == "fluid_wave":
        return linearized_jacobian_fluid(case.kx, case.ky)
    if case.name == "electrostatic_wave":
        return linearized_jacobian_electrostatic(case.kx, case.ky, lam)
    return linearized_jacobian_magnetostatic(case.kx, case.ky, lam, case.omega_c)


def exact_field(case: P.Case, t: float) -> np.ndarray:
    """Linear-eigenmode exact solution at time ``t``, real part.

    ``U_exact = real(Mi + eps*eigvec*sin(kx*x + ky*y - lambda_mode*t))`` with node
    coordinates ``x=(i-1)*dx`` and the phase-pinned eigenmode. At ``t=0`` this is
    the IC, so ``compute_L2_error(IC, 0, case) == 0``. The Matlab keeps the field
    complex; we take the physical real part (the locked realification decision).
    """
    if case.name not in _WAVE:
        raise ValueError("compute_L2_error only handles wave cases, not %r" % case.name)
    lam, vec = eigenmode(_case_jacobian(case), case.mode)
    Mi = case.equilibrium_moments()
    xnode = np.arange(case.Np) * case.dx
    ynode = np.arange(case.Np) * case.dy
    X, Y = np.meshgrid(xnode, ynode, indexing="ij")
    phase = case.kx * X + case.ky * Y - lam * t
    pert = case.eps * vec[:, None, None] * np.sin(phase)[None, :, :]
    return np.real(Mi[:, None, None] + pert)


def compute_L2_error(U: np.ndarray, t: float, case: P.Case) -> float:
    """``L2 = sqrt(sum((U - U_exact)^2) * dx * dy)`` (``compute_L2_error.m``).

    The oracle assumes the intended IC. For magnetic_wave it grades against the
    magnetostatic eigenmode (as compute_L2_error.m does, matching the intended
    IC); pairing it with the ``as_written`` magnetic IC (electrostatic Jacobian)
    is internally inconsistent and only meaningful under a named bug-for-bug test.
    """
    err = U - exact_field(case, t)
    return float(np.sqrt(np.sum(err ** 2) * case.dx * case.dy))
