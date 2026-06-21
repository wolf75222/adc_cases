#!/usr/bin/env python3
"""Load and diagnose RieMOM2D_Electrostatic_periodic campaign snapshots (ADC-377).

A snapshot is exactly what ``adc.System.write(format="npz")`` produces, so the
ROMEO campaign (ADC-376) does not invent a format: each saved step is one
``.npz`` with keys ``t``, ``macro_step``, ``nx``, ``ny``, ``blocks``,
``state_<block>`` (shape ``(nvar, ny, nx)``), ``names_<block>``,
``roles_<block>``, and ``phi`` (shape ``(ny, nx)``) when a Poisson field is
active. The hyqmom15 block is ``mom`` with the 15 moments ``M00..M04``.

This module is pure NumPy (no ``adc`` build, no matplotlib) so the diagnostics
and the renderer (``plot_rie_mom2d_case.py``) share one loader and the format
contract can be checked in CI (``check_plots.py``).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

# RieMOM2D_Electrostatic_periodic domain (init_domain.m): [-0.5, 0.5]^2 periodic.
XMIN, XMAX = -0.5, 0.5
# Moment block written by the hyqmom15 drivers (add_equation name "mom").
MOMENT_BLOCK = "mom"


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One saved step: moments, optional potential, time, and grid."""

    t: float
    step: int
    nx: int
    ny: int
    moments: np.ndarray            # (nvar, ny, nx); moments[0] == M00 density
    names: tuple[str, ...]
    phi: np.ndarray | None         # (ny, nx) or None when no Poisson field

    @property
    def density(self) -> np.ndarray:
        """M00 (the zeroth moment), shape (ny, nx)."""
        return self.moments[0]

    @property
    def dx(self) -> float:
        return (XMAX - XMIN) / self.nx

    @property
    def dy(self) -> float:
        return (XMAX - XMIN) / self.ny

    @property
    def mass(self) -> float:
        """Integral of M00 over the cell-area measure (conserved under transport)."""
        return float(self.density.sum() * self.dx * self.dy)

    def moment(self, name: str) -> np.ndarray:
        """Return the named moment field (e.g. ``"M00"``), shape (ny, nx)."""
        try:
            return self.moments[self.names.index(name)]
        except ValueError:
            raise KeyError("unknown moment %r; have %s" % (name, ", ".join(self.names))) from None


def _block_name(data) -> str:
    blocks = [str(b) for b in np.atleast_1d(data["blocks"])]
    if MOMENT_BLOCK in blocks:
        return MOMENT_BLOCK
    if len(blocks) == 1:
        return blocks[0]
    raise KeyError("snapshot has blocks %s; expected %r" % (blocks, MOMENT_BLOCK))


def load_snapshot(path) -> Snapshot:
    """Load one ``adc.System.write(format="npz")`` file into a :class:`Snapshot`."""
    data = np.load(path, allow_pickle=False)
    block = _block_name(data)
    moments = np.asarray(data["state_%s" % block], dtype=float)
    names = tuple(str(n) for n in data["names_%s" % block])
    phi = None
    if "phi" in data.files:
        # Sentinel convention (not a physics test): the drivers/campaign write an
        # all-zero phi when no Poisson field is solved, so the renderer skips the
        # potential panel for the source-free cases (fluid_wave, constant).
        phi_arr = np.asarray(data["phi"], dtype=float)
        if np.any(phi_arr):
            phi = phi_arr
    return Snapshot(
        t=float(data["t"]),
        step=int(data["macro_step"]),
        nx=int(data["nx"]),
        ny=int(data["ny"]),
        moments=moments,
        names=names,
        phi=phi,
    )


def load_case(case_dir) -> list[Snapshot]:
    """Load every ``*.npz`` snapshot under ``case_dir``, sorted by (t, step)."""
    case_dir = pathlib.Path(case_dir)
    snaps = [load_snapshot(p) for p in case_dir.glob("*.npz")]
    snaps.sort(key=lambda s: (s.t, s.step))
    return snaps


def load_meta(case_dir) -> dict:
    """Load the optional ``run_meta.json`` provenance sidecar, or ``{}``."""
    meta = pathlib.Path(case_dir) / "run_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def time_series(snaps) -> dict:
    """Scalar diagnostics over time: t, dt, mass drift, and M00 min/max."""
    t = np.array([s.t for s in snaps])
    mass = np.array([s.mass for s in snaps])
    m00_min = np.array([float(s.density.min()) for s in snaps])
    m00_max = np.array([float(s.density.max()) for s in snaps])
    dt = np.diff(t, prepend=t[:1])
    # Normalize the drift by the initial mass; guard the degenerate all-zero
    # initial field (drift then reports the absolute mass, not a ratio).
    mass0 = mass[0] if mass.size and mass[0] != 0 else 1.0
    return {
        "t": t,
        "dt": dt,
        "mass": mass,
        "mass_rel_drift": (mass - mass[0]) / mass0 if mass.size else mass,
        "m00_min": m00_min,
        "m00_max": m00_max,
    }
