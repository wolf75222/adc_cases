"""Static per-case data of the new periodic Matlab reference.

Mirrors ``RieMOM2D_Electrostatic_periodic`` (``init_case.m`` and the five
``init_*.m`` routines): the shared 15-moment hierarchy, the equilibrium
Maxwellian builder, and one frozen :class:`Case` per scenario. Pure NumPy, no
``adc`` dependency, so the layer imports in any environment that has NumPy.

See ``hyqmom15/matlab_ref/REFERENCE.md`` (ADC-348) for the locked values and the
D1-D9 divergence decisions. All values here are transcribed from the Matlab
source, which is the canonical reference (the Linear issue text is secondary).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# Domain and moment hierarchy shared by every case (init_*.m, identical).
XMIN, XMAX = -0.5, 0.5
NMOM = 15
FLAGSYM = 1

# State-vector order, identical to hyqmom15/model.py MOMENT_NAMES and to
# adc.moments.moment_names(4) (q-outer / p-inner canonical order).
MOMENT_NAMES = [
    "M00", "M10", "M20", "M30", "M40",
    "M01", "M11", "M21", "M31",
    "M02", "M12", "M22",
    "M03", "M13",
    "M04",
]
MOMENT_PQ = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2),
    (0, 3), (1, 3),
    (0, 4),
]
IDX = {name: k for k, name in enumerate(MOMENT_NAMES)}


def _gaussian_central(c20: float, c11: float, c02: float, p: int, q: int) -> float:
    """Central moment of order (p, q) of a 2D Gaussian (Isserlis).

    Mirrors ``_gaussian_central`` in hyqmom15/model.py: odd total orders vanish,
    even orders follow Isserlis' theorem in the covariance (c20, c11, c02).
    """
    if (p + q) % 2 == 1:
        return 0.0
    table = {
        (0, 0): 1.0,
        (2, 0): c20, (1, 1): c11, (0, 2): c02,
        (4, 0): 3.0 * c20 ** 2,
        (3, 1): 3.0 * c20 * c11,
        (2, 2): c20 * c02 + 2.0 * c11 ** 2,
        (1, 3): 3.0 * c02 * c11,
        (0, 4): 3.0 * c02 ** 2,
    }
    return table[(p, q)]


def maxwellian_moments(
    rho: ArrayLike,
    vx: ArrayLike = 0.0,
    vy: ArrayLike = 0.0,
    c20: float = 1.0,
    c11: float = 0.0,
    c02: float = 1.0,
) -> np.ndarray:
    """Raw 15-moment vector of a Gaussian, broadcasting over array inputs.

    Equivalent to ``InitializeM4_15(rho, vx, vy, c20, c11, c02)`` in the Matlab
    tree and to ``hyqmom15.model.gaussian_state(rho, vx, vy, c20, c11, c02)``.
    ``rho``, ``vx``, ``vy`` may be scalars or arrays of a common shape; the
    result has shape ``(15,) + broadcast_shape``. With ``c20=c02=1, c11=0`` this
    is the unit-temperature Maxwellian used by every case equilibrium and by the
    diocotron field (e.g. ``M20 = rho*(vx**2 + 1)``).
    """
    rho, vx, vy = np.broadcast_arrays(
        np.asarray(rho, dtype=float),
        np.asarray(vx, dtype=float),
        np.asarray(vy, dtype=float),
    )
    out = np.empty((NMOM,) + rho.shape, dtype=float)
    for k, (p, q) in enumerate(MOMENT_PQ):
        acc = np.zeros(rho.shape, dtype=float)
        for i in range(p + 1):
            for j in range(q + 1):
                c = _gaussian_central(c20, c11, c02, i, j)
                if c == 0.0:
                    continue
                acc += math.comb(p, i) * math.comb(q, j) * vx ** (p - i) * vy ** (q - j) * c
        out[k] = rho * acc
    return out


@dataclass(frozen=True)
class Case:
    """Parameters of one Matlab case, transcribed from ``init_<name>.m``.

    Field names use the Matlab spelling; ``name`` keeps the source spelling
    ``dicotron`` (the second ``o`` is missing in the new tree). ``reconstruction``
    and ``limiter`` are ``None`` when the source leaves them unset (the
    electrostatic_wave case does, see REFERENCE.md and ADC-353).
    """

    name: str
    Np: int
    cfl: float
    tmax: float
    omega_p: float
    omega_c: float
    electrostatic: bool
    magnetostatic: bool
    source: bool
    space_scheme: str
    time_scheme: str
    bc: str
    reconstruction: str = ""          # "" when the source leaves it unset
    limiter: str = ""
    # Eigenmode wave cases (left at 0 / unused for diocotron / constant).
    eps: float = 0.0
    mode: int = 0                     # 1-based Matlab column index
    kx: float = 0.0
    ky: float = 0.0
    # Diocotron ring (0 / unused elsewhere).
    r0: float = 0.0
    r1: float = 0.0
    rho_min: float = 0.0
    rho_max: float = 0.0
    # Equilibrium (rho0=1, U0=V0=0, T=1, r=0 for every committed case).
    rho0: float = 1.0
    U0: float = 0.0
    V0: float = 0.0
    T: float = 1.0
    r: float = 0.0

    @property
    def dx(self) -> float:
        return (XMAX - XMIN) / self.Np

    @property
    def dy(self) -> float:
        return (XMAX - XMIN) / self.Np

    @property
    def adim_debye_length(self) -> float:
        """``1 / omega_p`` (init_*.m: ``adim_debye_length = 1./omega_p``)."""
        return 1.0 / self.omega_p

    @property
    def C20(self) -> float:
        return self.T

    @property
    def C02(self) -> float:
        return self.T

    @property
    def C11(self) -> float:
        return self.r * math.sqrt(self.C20 * self.C02)

    @property
    def kmin(self) -> float:
        """CFL probe wavenumber.

        electrostatic_wave uses ``2*pi/dx`` (the ``sqrt(2)*pi/(xmax-xmin)`` line
        is commented out in init_electrostatic_wave.m); every other case uses
        ``sqrt(2)*pi/(xmax-xmin)``.
        """
        if self.name == "electrostatic_wave":
            return 2.0 * math.pi / self.dx
        return math.sqrt(2.0) * math.pi / (XMAX - XMIN)

    def equilibrium_moments(self) -> np.ndarray:
        """Maxwellian ``Mi = InitializeM4_15(rho0, U0, V0, C20, C11, C02)``."""
        return maxwellian_moments(self.rho0, self.U0, self.V0, self.C20, self.C11, self.C02)

    def cell_centers(self):
        """``(xm, ym)`` cell centers, matching init_domain.m.

        ``x = linspace(xmin, xmax, Np+1); xm = x(1:Np) + dx/2`` so
        ``xm[i] = xmin + (i + 0.5)*dx`` for 0-based ``i``.
        """
        i = np.arange(self.Np)
        xm = XMIN + (i + 0.5) * self.dx
        ym = XMIN + (i + 0.5) * self.dy
        return xm, ym


# One frozen Case per scenario. Values transcribed verbatim from the live
# init_<name>.m of RieMOM2D_Electrostatic_periodic (NOT init_diocotron.asv,
# which is a stale autosave: Np=512, omega_c=-200, RK2).
CASES = {
    "dicotron": Case(
        name="dicotron", Np=128, cfl=0.5, tmax=1.0,
        omega_p=20.0, omega_c=-20.0,
        electrostatic=True, magnetostatic=True, source=True,
        space_scheme="HLL", reconstruction="first", limiter="minmod",
        time_scheme="Euler", bc="periodic",
        eps=0.1, mode=4,
        r0=0.35, r1=0.4, rho_min=1e-4, rho_max=1.0,
    ),
    "fluid_wave": Case(
        name="fluid_wave", Np=32, cfl=0.4, tmax=0.05,
        omega_p=30.0, omega_c=-90.0,
        electrostatic=False, magnetostatic=False, source=False,
        space_scheme="ROE", reconstruction="first", limiter="none",
        time_scheme="Euler", bc="periodic",
        eps=0.01, mode=15,
        kx=4.0 * math.pi / (XMAX - XMIN), ky=0.0 * math.pi / (XMAX - XMIN),
    ),
    "electrostatic_wave": Case(
        name="electrostatic_wave", Np=128, cfl=0.5, tmax=1.0,
        omega_p=30.0, omega_c=-90.0,
        electrostatic=True, magnetostatic=False, source=True,
        space_scheme="HLL", reconstruction="", limiter="",
        time_scheme="Euler", bc="periodic",
        eps=0.01, mode=15,
        kx=0.0 * math.pi / (XMAX - XMIN), ky=4.0 * math.pi / (XMAX - XMIN),
    ),
    "magnetic_wave": Case(
        name="magnetic_wave", Np=256, cfl=0.5, tmax=1.0,
        omega_p=20.0, omega_c=-40.0,
        electrostatic=True, magnetostatic=True, source=True,
        space_scheme="HLL", reconstruction="muscl", limiter="minmod",
        time_scheme="Euler", bc="periodic",
        eps=0.01, mode=15,
        kx=2.0 * math.pi / (XMAX - XMIN), ky=4.0 * math.pi / (XMAX - XMIN),
    ),
    "constant": Case(
        name="constant", Np=64, cfl=0.5, tmax=1.0,
        omega_p=30.0, omega_c=-90.0,
        electrostatic=False, magnetostatic=False, source=False,
        space_scheme="HLL", reconstruction="muscl", limiter="minmod",
        time_scheme="Euler", bc="periodic",
    ),
}


def get_case(name: str) -> Case:
    """Return the :class:`Case` for ``name`` (Matlab spelling, e.g. ``dicotron``)."""
    try:
        return CASES[name]
    except KeyError:
        raise KeyError(
            "Unknown case %r; known: %s" % (name, ", ".join(sorted(CASES)))
        ) from None
