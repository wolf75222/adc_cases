#!/usr/bin/env python3
"""Map the Matlab ``params.time_scheme`` to the ADC explicit integrator (ADC-379).

The refactored Matlab ``RieMOM2D_Electrostatic_periodic/time_step.m`` dispatches
on ``params.time_scheme``:

  * ``"Euler"`` -> ``euler_step.m`` : forward Euler
  * ``"RK2"``   -> ``rk2_step.m``   : Heun / SSPRK2
  * ``"RK3"``   -> ``rk3_step.m``   : SSPRK3 (Shu-Osher)

All five committed cases declare ``time_scheme = "Euler"``, so the M8 drivers
reproduce them with forward Euler. This module keeps the Matlab -> ADC name
mapping in one place so a future case that selects ``RK2`` or ``RK3`` is honoured
by the drivers without any adc_cpp change: the cartesian production ``System``
already exposes the three integrators through ``adc.Explicit`` (ForwardEuler,
SSPRK2Step, SSPRK3Step).

Parity note. ``System.step(dt)`` solves the fields once at the start of the
macro-step and keeps ``phi`` frozen across the RK stages, matching the Matlab
``time_step.m`` (which builds ``rhs = @(U) spatial_operator(U, phi, params)``
with a fixed ``phi``). The ``adc.integrate.ssprk*_step`` Python helpers re-solve
Poisson every stage; that is a more tightly coupled mode, NOT strict Matlab
parity, and is out of scope for the reference port.
"""
from __future__ import annotations

# Matlab time_scheme spelling -> adc.Explicit(method=...) name. The keys use the
# exact Matlab spelling found in params.time_scheme (init_<case>.m / time_step.m).
_ADC_METHOD = {
    "Euler": "euler",   # euler_step.m : forward Euler
    "RK2": "ssprk2",    # rk2_step.m   : Heun / SSPRK2
    "RK3": "ssprk3",    # rk3_step.m   : SSPRK3 (Shu-Osher)
}


def adc_method(time_scheme: str) -> str:
    """Return the ``adc.Explicit`` method name for a Matlab ``time_scheme``.

    Case-sensitive on the Matlab spelling (``"Euler"``, ``"RK2"``, ``"RK3"``);
    any other value, including a wrong-case variant, raises ``ValueError`` so a
    typo fails loudly instead of silently selecting a default integrator.
    """
    try:
        return _ADC_METHOD[time_scheme]
    except KeyError:
        raise ValueError(
            "unknown Matlab time_scheme %r; expected one of %s"
            % (time_scheme, ", ".join(repr(k) for k in _ADC_METHOD))
        ) from None


def explicit_for(time_scheme: str, adc_module=None, **kwargs):
    """Build the ``adc.Explicit`` integrator for a Matlab ``time_scheme``.

    ``adc_module`` defaults to the real ``adc`` package (imported lazily so the
    pure mapping in :func:`adc_method` needs no adc build); tests inject a stub.
    Extra keyword arguments are forwarded to ``adc.Explicit``.
    """
    if adc_module is None:
        import adc as adc_module
    return adc_module.Explicit(method=adc_method(time_scheme), **kwargs)
