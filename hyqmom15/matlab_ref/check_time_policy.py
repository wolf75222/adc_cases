#!/usr/bin/env python3
"""Test the Matlab time_scheme -> ADC integrator mapping (ADC-379).

Pure Python, no build: exercises ``matlab_ref.time_policy`` in isolation (the
real ``adc.Explicit`` is stubbed) and checks that every committed case still
maps to forward Euler, that the mapping is case-exact and rejects unknown
schemes, and that a case switched to ``RK2``/``RK3`` selects the matching ADC
method without touching the physics.

Run: ``python3 hyqmom15/matlab_ref/check_time_policy.py`` (0 = OK, 1 = mismatch).
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from matlab_ref import CASES, get_case  # noqa: E402
from matlab_ref.time_policy import adc_method, explicit_for  # noqa: E402


class _StubExplicit:
    """Records the method a driver-style call would hand to ``adc.Explicit``."""

    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs


class _StubAdc:
    Explicit = _StubExplicit


def check_mapping():
    assert adc_method("Euler") == "euler"
    assert adc_method("RK2") == "ssprk2"
    assert adc_method("RK3") == "ssprk3"
    return "mapping OK (Euler->euler, RK2->ssprk2, RK3->ssprk3)"


def check_case_exact():
    # The Matlab spelling is exact: a wrong-case or unknown scheme must fail loud
    # rather than silently fall back to a default integrator.
    bad = ("euler", "EULER", "rk2", "Rk3", "SSPRK2", "heun", "", "RK4")
    for scheme in bad:
        try:
            adc_method(scheme)
        except ValueError:
            continue
        raise AssertionError("adc_method(%r) should raise ValueError" % scheme)
    return "case-exact OK (%d wrong/unknown schemes rejected)" % len(bad)


def check_committed_cases_stay_euler():
    # The five M8 cases declare time_scheme="Euler" in their Matlab source, so the
    # mapping must keep reproducing forward Euler for all of them.
    for name in CASES:
        scheme = get_case(name).time_scheme
        assert scheme == "Euler", "%s no longer Euler in params.py: %r" % (name, scheme)
        assert adc_method(scheme) == "euler"
    return "committed cases OK (%d cases all Euler -> euler)" % len(CASES)


def check_driver_selection():
    # A driver that follows case.time_scheme must select the right ADC integrator
    # without changing the physics: force RK2/RK3 on the simplest cases and check
    # the adc.Explicit(method=...) the driver would build (via a stubbed adc).
    stub = _StubAdc()
    for case_name in ("constant", "fluid_wave"):
        base = get_case(case_name)
        for scheme, want in (("Euler", "euler"), ("RK2", "ssprk2"), ("RK3", "ssprk3")):
            case = dataclasses.replace(base, time_scheme=scheme)
            integ = explicit_for(case.time_scheme, adc_module=stub)
            assert integ.method == want, "%s/%s -> %s != %s" % (case_name, scheme, integ.method, want)
    return "driver selection OK (constant+fluid_wave x Euler/RK2/RK3 via explicit_for)"


CHECKS = [
    check_mapping,
    check_case_exact,
    check_committed_cases_stay_euler,
    check_driver_selection,
]


def main() -> int:
    failures = []
    for fn in CHECKS:
        try:
            print("  OK   %-32s %s" % (fn.__name__, fn()))
        except Exception as exc:  # noqa: BLE001
            failures.append(fn.__name__)
            print("  FAIL %-32s %s" % (fn.__name__, exc))
    if failures:
        print("CHECK-TIME-POLICY: %d/%d FAILED" % (len(failures), len(CHECKS)), file=sys.stderr)
        return 1
    print("CHECK-TIME-POLICY: OK (%d checks vs time_step.m Euler/RK2/RK3)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
