"""Snapshot-interval realizability and symmetry diagnostics for hyqmom15 (ADC-383).

Pure-NumPy post-processing of moment snapshots (no adc build, no in-solver hook):
realizability of the HyQMOM moment state (positivity, variance/covariance
positive-definiteness, the p2p2 Hankel eigenvalues, the univariate bounds), the
per-case spatial symmetry residuals, and a declarative, non-fatal assert layer
so a transient violation that recovers does not stop the simulation.
"""
from __future__ import annotations

from .realizability import (
    DEFAULT_CHECKS,
    RealizabilityCheck,
    evaluate,
    field_realizability,
    standardized_moments,
    summarize,
)
from .symmetry import SYMMETRY_BY_CASE, symmetry_residual

__all__ = [
    "DEFAULT_CHECKS",
    "RealizabilityCheck",
    "evaluate",
    "field_realizability",
    "standardized_moments",
    "summarize",
    "SYMMETRY_BY_CASE",
    "symmetry_residual",
]
