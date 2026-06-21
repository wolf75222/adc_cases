"""Shared layer mirroring the new periodic Matlab reference.

``RieMOM2D_Electrostatic_periodic`` parameters, linearized Jacobians, eigenmode
and diocotron initializers, the ``compute_dt`` policy, and the wave ``L2`` oracle,
ported to pure NumPy. The drivers (run_diocotron / run_*_wave, ADC-351+) build on
this layer; the goldens (ADC-350) validate it. See ``REFERENCE.md`` (ADC-348) for
the locked values and the D1-D9 decisions.
"""
from __future__ import annotations

from .params import (
    CASES,
    MOMENT_NAMES,
    MOMENT_PQ,
    NMOM,
    XMAX,
    XMIN,
    Case,
    get_case,
    maxwellian_moments,
)
from .linearized import (
    eigenmode,
    linearized_jacobian_electrostatic,
    linearized_jacobian_fluid,
    linearized_jacobian_magnetostatic,
    matlab_sort_indices,
    phase_pin,
)
from .initializers import (
    InitField,
    diocotron_max_speed,
    init_constant_field,
    init_diocotron_field,
    init_electrostatic_wave_field,
    init_fluid_wave_field,
    init_magnetic_wave_field,
    initial_field,
)
from .dt_policy import compute_dt
from .l2 import compute_L2_error, exact_field
from .time_policy import adc_method, explicit_for

__all__ = [
    "CASES", "Case", "get_case",
    "MOMENT_NAMES", "MOMENT_PQ", "NMOM", "XMIN", "XMAX", "maxwellian_moments",
    "linearized_jacobian_fluid", "linearized_jacobian_electrostatic",
    "linearized_jacobian_magnetostatic", "eigenmode", "phase_pin",
    "matlab_sort_indices",
    "InitField", "initial_field", "init_constant_field", "init_diocotron_field",
    "init_fluid_wave_field", "init_electrostatic_wave_field", "init_magnetic_wave_field",
    "diocotron_max_speed",
    "compute_dt", "compute_L2_error", "exact_field",
    "adc_method", "explicit_for",
]
