"""Time-step policy of the new periodic Matlab reference (``compute_dt.m``).

Kept as an explicit per-case function (REFERENCE.md D6) instead of folded into
the model ``source_frequency``: the Matlab source caps are tighter and must not
be hidden. The drivers (ADC-351+) impose this dt for faithful replays.
"""
from __future__ import annotations


def compute_dt(vmax: float, case, t: float) -> float:
    """Return ``dt`` exactly as ``compute_dt.m``.

    ``dt = CFL*dx/vmax``, then a source cap that MULTIPLIES ``vmax``:
    electrostatic -> ``CFL*dx*vmax/omega_p^2``, else magnetostatic ->
    ``CFL*dx*vmax/omega_c^2``. The ``elseif`` means a both-source case (diocotron,
    magnetic_wave) applies ONLY the ``omega_p^2`` cap; the ``omega_c^2`` cap is
    dead there. A final-time clamp caps ``dt`` to ``tmax - t``.
    """
    dt = case.cfl * case.dx / vmax
    if case.electrostatic:
        dt = min(dt, case.cfl * case.dx * vmax / case.omega_p ** 2)
    elif case.magnetostatic:
        dt = min(dt, case.cfl * case.dx * vmax / case.omega_c ** 2)
    if t + dt > case.tmax:
        dt = case.tmax - t
    return dt
