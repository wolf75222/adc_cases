"""Initial-condition routines of the new periodic Matlab reference.

Ports of the ``init_*_field.m`` routines from ``RieMOM2D_Electrostatic_periodic``.
Each returns an :class:`InitField` carrying the conservative moment array ``M``
of shape ``(15, Np, Np)``.

Array layout is kept FAITHFUL to the Matlab source, which is not uniform across
cases:

* wave cases (fluid / electrostatic / magnetic): ``M[k, i, j]`` with ``i`` the
  first Matlab index used in ``phase = kx*(i-1)*dx + ky*(j-1)*dy`` (so ``i`` runs
  with ``kx``, ``j`` with ``ky``);
* diocotron: ``M[k, i, j]`` from ``meshgrid(xm, ym)``, so ``i`` is the row (``y``)
  and ``j`` the column (``x``).

Mapping to the ADC ``System`` grid (k, ny, nx) is therefore per-case and is the
drivers' job (ADC-351+), not this layer's. See ``REFERENCE.md`` D2/D3/D4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import params as P
from .linearized import (
    eigenmode,
    linearized_jacobian_electrostatic,
    linearized_jacobian_fluid,
    linearized_jacobian_magnetostatic,
    matlab_sort_indices,
)


@dataclass
class InitField:
    """Result of an initializer.

    ``M`` is the ``(15, Np, Np)`` conservative state (Matlab layout, see module
    docstring). ``eigenvalue`` / ``eigen_vect`` are the selected mode (wave cases
    only), the eigenvector phase-pinned and L2-normalized. ``max_speed`` is a CFL
    probe speed (NOT used by the committed time loop, which derives dt from
    compute_speeds; see ADC-356): electrostatic_wave reports the D3 ``max`` of a
    possibly-complex spectrum, while diocotron reports the real-part rule of
    :func:`diocotron_max_speed`.
    """

    M: np.ndarray
    eigenvalue: Optional[complex] = None
    eigen_vect: Optional[np.ndarray] = None
    max_speed: Optional[complex] = None


def _poisson_fft(rho: np.ndarray, lam: float, dx: float, dy: float) -> np.ndarray:
    """Periodic FFT Poisson solve, faithful to ``poisson_fft.m``.

    Solves ``-K^2 * lam^2 * phi_hat = -rho_hat`` (electron sign) with the zero
    mode set to zero, after removing the mean of ``rho``. Reproduces the Matlab
    ``meshgrid(ky, kx)`` axis convention.
    """
    nx, ny = rho.shape
    rho = rho - rho.mean()
    rho_hat = np.fft.fft2(rho)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    # Matlab [KX,KY] = meshgrid(ky,kx): KX[i,j]=ky[j], KY[i,j]=kx[i].
    kxx, kyy = np.meshgrid(ky, kx)
    k2 = kxx ** 2 + kyy ** 2
    phi_hat = np.zeros_like(rho_hat)
    mask = k2 != 0
    phi_hat[mask] = -rho_hat[mask] / (lam ** 2 * k2[mask])
    return np.real(np.fft.ifft2(phi_hat))


def _wave_state(case: P.Case, J: np.ndarray):
    """Seed ``M = Mi + eps*real(eigvec)*sin(phase)`` like ``init_*_wave_field.m``.

    Uses node coordinates ``x = (i-1)*dx`` (no cell-center offset, matching the
    Matlab wave loops and ``compute_L2_error.m``). The physical IC is the real
    part of the complex perturbation; at ``t=0`` ``sin(phase)`` is real so this is
    exact and only the eigenvector's real part survives.
    """
    lam, vec = eigenmode(J, case.mode)
    Mi = case.equilibrium_moments()
    Np, dx, dy = case.Np, case.dx, case.dy
    xnode = np.arange(Np) * dx
    ynode = np.arange(Np) * dy
    X, Y = np.meshgrid(xnode, ynode, indexing="ij")  # X[i,j]=xnode[i]
    sin_phase = np.sin(case.kx * X + case.ky * Y)
    M = Mi[:, None, None] + case.eps * np.real(vec)[:, None, None] * sin_phase[None, :, :]
    return M, lam, vec


def _es_max_speed(case: P.Case, policy: str) -> complex:
    """Electrostatic CFL speed, ``intended`` (diag(Dmax)) or ``as_written`` (diag(D)).

    ``intended`` (default per D3) takes the spectrum of the Jacobian at
    ``(kmin, kmin)``; ``as_written`` reproduces the Matlab bug that reuses the
    mode Jacobian at ``(kx, ky)``. ``max`` follows Matlab semantics (last element
    after the eigenvalue sort).
    """
    lam = case.adim_debye_length
    if policy == "intended":
        J = linearized_jacobian_electrostatic(case.kmin, case.kmin, lam)
    elif policy == "as_written":
        J = linearized_jacobian_electrostatic(case.kx, case.ky, lam)
    else:
        raise ValueError("dmax_policy must be 'intended' or 'as_written', got %r" % policy)
    w, _ = np.linalg.eig(J)
    return w[matlab_sort_indices(w)][-1]


def diocotron_max_speed(case: P.Case) -> float:
    """Diocotron CFL probe speed (``init_diocotron.m`` lines 84-90).

    ``sort(real(eig(magnetostatic(kmin, kmin, debye, omega_c))))[-1]`` -- the
    LARGEST REAL PART. This rule differs from the wave :func:`_es_max_speed`, which
    takes the largest magnitude of the complex spectrum; reusing the wave helper
    here would return the wrong number. The value is vestigial in the committed
    Matlab time loop (dt comes from compute_speeds, not this), kept for parity and
    to pin the convention for ADC-351.
    """
    J = linearized_jacobian_magnetostatic(
        case.kmin, case.kmin, case.adim_debye_length, case.omega_c
    )
    w, _ = np.linalg.eig(J)
    return float(np.sort(w.real)[-1])


def init_constant_field(case: P.Case) -> InitField:
    """Uniform equilibrium Maxwellian everywhere (``init_constant_field.m``)."""
    Mi = case.equilibrium_moments()
    M = np.broadcast_to(Mi[:, None, None], (P.NMOM, case.Np, case.Np)).copy()
    return InitField(M=M)


def init_fluid_wave_field(case: P.Case) -> InitField:
    """Fluid eigenmode IC (``init_fluid_wave_field.m``), real Jacobian."""
    J = linearized_jacobian_fluid(case.kx, case.ky)
    M, lam, vec = _wave_state(case, J)
    return InitField(M=M, eigenvalue=lam, eigen_vect=vec)


def init_electrostatic_wave_field(case: P.Case, *, dmax_policy: str = "intended") -> InitField:
    """Electrostatic eigenmode IC (``init_electrostatic_wave_field.m``).

    The IC uses the mode Jacobian at ``(kx, ky)``; ``dmax_policy`` selects the D3
    CFL-speed convention (default ``intended`` = diag(Dmax)).
    """
    J = linearized_jacobian_electrostatic(case.kx, case.ky, case.adim_debye_length)
    M, lam, vec = _wave_state(case, J)
    return InitField(M=M, eigenvalue=lam, eigen_vect=vec, max_speed=_es_max_speed(case, dmax_policy))


def init_magnetic_wave_field(
    case: P.Case, *, wiring: str = "intended", dmax_policy: str = "intended"
) -> InitField:
    """Magnetic eigenmode IC.

    ``wiring='intended'`` (default per D4) builds the IC from the complex
    magnetostatic Jacobian (``init_magnetic_wave_field.m``). ``wiring='as_written'``
    reproduces the source oversight where ``init_magnetic_wave.m`` wires
    ``init_electrostatic_wave_field`` instead, dropping the cyclotron coupling.
    """
    if wiring == "intended":
        J = linearized_jacobian_magnetostatic(
            case.kx, case.ky, case.adim_debye_length, case.omega_c
        )
        M, lam, vec = _wave_state(case, J)
        return InitField(M=M, eigenvalue=lam, eigen_vect=vec)
    if wiring == "as_written":
        return init_electrostatic_wave_field(case, dmax_policy=dmax_policy)
    raise ValueError("wiring must be 'intended' or 'as_written', got %r" % wiring)


def init_diocotron_field(case: P.Case, *, orientation: str = "standard") -> InitField:
    """Diocotron ring + ExB drift IC (``init_diocotron_field.m``).

    ``orientation='standard'`` (default per D2) is the corrected incompressible
    ExB drift; ``orientation='matlab_bug'`` reproduces the transposed/divergent
    meshgrid drift of the Matlab source (for strict golden parity under the named
    ``--ic-matlab-bug`` path). Moments use the unit-temperature Maxwellian raw
    moments of ``(rho, vx, vy)`` (the active Matlab block).
    """
    Np, dx, dy = case.Np, case.dx, case.dy
    xm, ym = case.cell_centers()
    X, Y = np.meshgrid(xm, ym)  # Matlab meshgrid(xm,ym): X[i,j]=xm[j], Y[i,j]=ym[i]
    R = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    rho = np.full((Np, Np), case.rho_min, dtype=float)
    mask = (R >= case.r0) & (R <= case.r1)
    delta = 1.0 - case.eps + case.eps * np.sin(case.mode * theta)
    rho[mask] = case.rho_max * delta[mask]

    phi = _poisson_fft(rho, case.adim_debye_length, dx, dy)
    # Periodic centered gradient; grad component 1 = d/d(first index), 2 = d/d(second).
    grad1 = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dx)
    grad2 = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dy)
    oc = case.omega_c
    if orientation == "matlab_bug":
        # Faithful to init_diocotron_field.m: vx=-grad_phi(:,:,2)/oc, vy=+grad_phi(:,:,1)/oc.
        vx = -grad2 / oc
        vy = grad1 / oc
    elif orientation == "standard":
        # Corrected incompressible ExB (first index = y, second = x).
        vx = -grad1 / oc
        vy = grad2 / oc
    else:
        raise ValueError("orientation must be 'standard' or 'matlab_bug', got %r" % orientation)

    M = P.maxwellian_moments(rho, vx, vy, 1.0, 0.0, 1.0)
    return InitField(M=M, max_speed=diocotron_max_speed(case))


def initial_field(case: P.Case, **opts) -> InitField:
    """Dispatch to the initializer for ``case.name`` (Matlab ``init_case`` spelling)."""
    dispatch = {
        "dicotron": init_diocotron_field,
        "fluid_wave": init_fluid_wave_field,
        "electrostatic_wave": init_electrostatic_wave_field,
        "magnetic_wave": init_magnetic_wave_field,
        "constant": init_constant_field,
    }
    try:
        fn = dispatch[case.name]
    except KeyError:
        raise KeyError("Unknown case %r" % case.name) from None
    return fn(case, **opts)
