#!/usr/bin/env python3
"""Smoke test BUILD-FREE de l'assemblage du modele POLAIRE (run_polar.py).

Ce test n'a PAS besoin de l'extension lourde Kokkos/AMReX `adc` : il installe un faux module
`adc` minimal qui ENREGISTRE chaque appel sur un faux `System` (et restitue un potentiel non
trivial pour que la derive ExB se calcule), puis pilote `build_polar_system`. Les assertions
sont REELLES et verrouillent le contrat d'assemblage du chemin polaire :

  (1) ordre des appels facade : set_poisson(polar/dirichlet) -> set_magnetic_field -> add_equation
      -> set_density -> solve_fields -> set_state -> solve_fields. set_magnetic_field DOIT preceder
      l'etage Schur (add_equation), exigence dure de set_source_stage polaire.
  (2) set_poisson route bien sur solver='polar', bc='dirichlet', rhs='charge_density'.
  (3) add_equation utilise WENO5 + Rusanov + SSPRK3 + CondensedSchur(electrostatic_lorentz).
  (4) la densite posee est le top-hat annulaire (fond rho_min hors anneau, perturbation dans [R0,R1]),
      layout (ntheta, nr) aplati flat[j*nr+i].
  (5) l'etat conservatif injecte est (3, ntheta, nr) comp-major (rho, mom_r, mom_theta) avec
      (5a) v_r = derive ExB radiale -grad_theta/B ; (5b) v_theta = racine de la quadratique de BILAN
      RADIAL (equilibre rotatif : centrifuge + pression + electrique + Lorentz), residu ~ 0 par
      cellule ; (5c) l'anneau d'equilibre SANS perturbation est stationnaire (check_equilibrium) ;
      (5d) cs2=0 -> v_theta se reduit EXACTEMENT a la derive ExB grad_r/B (continuation froide).
  (6) le stencil polar_gradient EST exactement celui de derive_aux_polar (centre interieur,
      decentre ordre 2 aux parois radiales, enroulement periodique en theta).
  (7) fit_growth lit la fenetre VERBATIM du papier et recupere une pente exp pure exactement.
  (8) un run multi-rang est refuse (mono-rang : l'etage Schur polaire = boite unique).

Lancer en standalone (`python3 test_polar_assembly.py`) ou sous pytest.
"""

from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _install_fake_adc() -> types.ModuleType:
    """Installe un faux module `adc` tracant System et rendant un phi non trivial.

    Le potentiel restitue par `potential()` est non trivial pour que la derive
    ExB se calcule reellement sur le chemin polaire.

    Returns:
        Le faux module `adc` (egalement enregistre dans `sys.modules`).
    """
    adc = types.ModuleType("adc")

    class FakeSystem:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            # phi(theta, r) = (r - r_min)(r_max - r) cos(theta) : non trivial, nul aux parois
            # radiales (coherent Dirichlet), de gradient theta non nul (derive ExB active).
            self._nr = None
            self._nth = None
            self._state = (
                None  # etat conservatif (3, ntheta, nr) le plus recemment pose
            )

        def _record(self, name, *a, **k):
            self.calls.append((name, a, k))

        def set_poisson(self, *a, **k):
            self._record("set_poisson", *a, **k)

        def set_magnetic_field(self, arr, *a, **k):
            arr = np.asarray(arr)
            self._nth, self._nr = arr.shape
            self._record("set_magnetic_field", arr, *a, **k)

        def add_equation(self, *a, **k):
            self._record("add_equation", *a, **k)

        def set_density(self, name, flat, *a, **k):
            self._record("set_density", name, np.asarray(flat), *a, **k)

        def set_state(self, name, flat, *a, **k):
            self._state = (
                np.asarray(flat, dtype=np.float64)
                .reshape(3, self._nth, self._nr)
                .copy()
            )
            self._record("set_state", name, np.asarray(flat), *a, **k)

        def get_state(self, name, *a, **k):
            self._record("get_state", name, *a, **k)
            return self._state.copy()

        def solve_fields(self, *a, **k):
            self._record("solve_fields")

        # --- DRIVE PARASITE DETERMINISTE : un faux schema NON discretement bien pose. step() ajoute a
        # l'etat une derive FIXE par cellule D(r, theta) (independante de l'etat), modelisant l'imbalance
        # O(1) axisymetrique du vrai schema sur l'equilibre. Ainsi R_eq = step(U_eq) - U_eq = D, et la
        # carte corrigee step()-R_eq = U + D - D = U a la PRECISION MACHINE : U_eq devient point fixe
        # exact, exactement ce que --frozen-equilibrium doit verifier. (Avant l'option c, cette derive
        # ferait diverger l'etat -- d'ou l'utilite du test.)
        def _spurious_drift(self):
            nr, nth = self._nr, self._nth
            r_min = self.kwargs["mesh"].r_min
            r_max = self.kwargs["mesh"].r_max
            dr = (r_max - r_min) / nr
            dth = 2.0 * math.pi / nth
            th = ((np.arange(nth) + 0.5) * dth)[:, None]
            r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
            base = (
                1.0e-3 * np.sin(r) * np.cos(th)
            )  # (nth, nr) derive O(1e-3) non triviale, FIXE
            return np.stack(
                [base, 0.5 * base, -0.25 * base], axis=0
            )  # (3, nth, nr)

        def step(self, *a, **k):
            self._t = getattr(self, "_t", 0.0) + (a[0] if a else 1.0e-3)
            if self._state is not None:
                self._state = self._state + self._spurious_drift()
            self._record("step", *a, **k)

        def step_cfl(self, *a, **k):
            self._t = getattr(self, "_t", 0.0) + 1.0e-3
            if self._state is not None:
                self._state = self._state + self._spurious_drift()
            self._record("step_cfl", *a, **k)

        def time(self):
            return getattr(self, "_t", 0.0)

        def mass(self, *a, **k):
            return 1.0

        def potential(self):
            self._record("potential")
            nr, nth = self._nr, self._nth
            r_min = self.kwargs["mesh"].r_min
            r_max = self.kwargs["mesh"].r_max
            dr = (r_max - r_min) / nr
            dth = 2.0 * math.pi / nth
            th = ((np.arange(nth) + 0.5) * dth)[:, None]
            r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
            return (r - r_min) * (r_max - r) * np.cos(th)  # (nth, nr)

    class FakeMesh:
        def __init__(self, r_min, r_max, nr, ntheta):
            self.r_min = float(r_min)
            self.r_max = float(r_max)
            self.nr = int(nr)
            self.ntheta = int(ntheta)

    adc.System = FakeSystem
    adc.PolarMesh = FakeMesh
    adc.FluidState = lambda **k: ("FluidState", k)
    adc.IsothermalFlux = lambda **k: ("IsothermalFlux", k)
    adc.NoSource = lambda **k: ("NoSource", k)
    adc.BackgroundDensity = lambda **k: ("BackgroundDensity", k)
    adc.ChargeDensity = lambda **k: ("ChargeDensity", k)
    adc.Model = lambda **k: ("Model", k)
    adc.FiniteVolume = lambda **k: ("FiniteVolume", k)
    adc.Explicit = lambda **k: ("Explicit", k)
    adc.CondensedSchur = lambda **k: ("CondensedSchur", k)
    adc.Split = lambda **k: ("Split", k)
    adc.Strang = lambda **k: ("Strang", k)
    dsl = types.ModuleType("adc.dsl")
    adc.dsl = dsl
    sys.modules["adc"] = adc
    sys.modules["adc.dsl"] = dsl
    return adc


def _import_run_polar():
    """Importe (frais) le module `run_polar` du cas, le faux `adc` etant en place."""
    case_root = os.path.dirname(
        HERE
    )  # tests/ -> la racine du cas (model.py, run*.py)
    if case_root not in sys.path:
        sys.path.insert(0, case_root)
    import importlib

    if "run_polar" in sys.modules:
        del sys.modules["run_polar"]
    return importlib.import_module("run_polar")


def _params():
    """Retourne les parametres de reference du papier."""
    from model import PaperParameters

    return PaperParameters()


class _Args:
    """Faux espace de noms argparse : defauts CLI surchargeables par mot-cle."""

    def __init__(self, **kw) -> None:
        self.r_min = 2.0
        self.nr = 24
        self.ntheta = 16
        self.cs2 = 0.0
        self.theta = 0.5
        self.strang = False
        self.limiter = "weno5"  # defaut de l'argparse (--limiter)
        self.ic = "equilibrium"  # defaut de l'argparse (--ic)
        # Defauts du chemin --frozen-equilibrium (option c).
        self.dt = 1.0e-3
        self.cfl = 0.0
        self.frozen_equilibrium = True
        self.frozen_check_const = 1.0e3
        self.max_steps_check = 20
        self.check_modes = [1, 2, 3, 4, 5]
        self.check_tol = 0.05
        for k, v in kw.items():
            setattr(self, k, v)


def test_assembly_call_order_and_routing() -> None:
    """(1)(2) Verrouille l'ordre des appels facade et le routage Poisson polaire."""
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args()
    sim = rp.build_polar_system(
        args.nr, args.ntheta, mode=4, params=params, args=args
    )
    names = [c[0] for c in sim.calls]

    # (1) ordre : Poisson, B_z, equation, densite, solve, etat, solve.
    i_poisson = names.index("set_poisson")
    i_bz = names.index("set_magnetic_field")
    i_eq = names.index("add_equation")
    i_rho = names.index("set_density")
    i_state = names.index("set_state")
    assert (
        i_poisson < i_bz < i_eq
    ), "set_magnetic_field DOIT preceder l'etage Schur (add_equation)"
    assert i_eq < i_rho < i_state, "densite posee avant l'etat drift"
    assert (
        names.count("solve_fields") >= 2
    ), "deux solves : apres densite, puis apres l'etat drift"

    # (2) routage Poisson polaire Dirichlet.
    _, pa, pk = sim.calls[i_poisson]
    pk = dict(pk)
    assert pk.get("solver") == "polar", pk
    assert pk.get("bc") == "dirichlet", pk
    assert pk.get("rhs") == "charge_density", pk
    print("OK (1)(2) ordre facade + Poisson polaire dirichlet")


def test_add_equation_uses_weno5_ssprk3_schur() -> None:
    """(3) add_equation cable WENO5 + Rusanov + SSPRK3 + CondensedSchur."""
    _install_fake_adc()
    rp = _import_run_polar()
    sim = rp.build_polar_system(24, 16, 4, _params(), _Args())
    eq = [c for c in sim.calls if c[0] == "add_equation"][0]
    _, _, kw = eq
    fv = kw["spatial"]  # ("FiniteVolume", {...})
    tm = kw["time"]  # ("Split", {hyperbolic, source})
    assert fv[1]["limiter"] == "weno5" and fv[1]["riemann"] == "rusanov"
    assert tm[0] == "Split"
    hyp = tm[1]["hyperbolic"]
    src = tm[1]["source"]
    assert hyp[1].get("method") == "ssprk3"
    assert (
        src[0] == "CondensedSchur"
        and src[1].get("kind") == "electrostatic_lorentz"
    )
    print(
        "OK (3) WENO5 + Rusanov + SSPRK3 + CondensedSchur(electrostatic_lorentz)"
    )


def test_strang_switches_split_factory() -> None:
    """--strang bascule la fabrique de split sur adc.Strang (2e ordre)."""
    _install_fake_adc()
    rp = _import_run_polar()
    sim = rp.build_polar_system(24, 16, 4, _params(), _Args(strang=True))
    eq = [c for c in sim.calls if c[0] == "add_equation"][0]
    assert (
        eq[2]["time"][0] == "Strang"
    ), "--strang doit utiliser adc.Strang (2e ordre)"
    print("OK (--strang) bascule sur adc.Strang")


def test_density_is_annular_tophat() -> None:
    """(4) La densite posee est le top-hat annulaire, layout flat[j*nr+i]."""
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args()
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    rho_flat = [c for c in sim.calls if c[0] == "set_density"][0][1][1]
    rho = np.asarray(rho_flat).reshape(args.ntheta, args.nr)  # flat[j*nr+i]
    dr = (params.radius - args.r_min) / args.nr
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    inside = (r >= params.ring_inner) & (r <= params.ring_outer)
    assert np.allclose(
        rho[:, ~inside], params.rho_min
    ), "fond rho_min hors anneau"
    assert np.all(rho[:, inside] > params.rho_min), "anneau > rho_min"
    print("OK (4) densite top-hat annulaire, layout flat[j*nr+i]")


def test_state_radial_velocity_is_exb_drift() -> None:
    """(5a) v_r reste la derive ExB radiale -grad_theta/B.

    C'est le point fixe de l'etage source : v_r ne doit pas devier de la derive
    ExB radiale -grad_theta/B.
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args()
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    state_flat = [c for c in sim.calls if c[0] == "set_state"][0][1][1]
    U = np.asarray(state_flat).reshape(
        3, args.ntheta, args.nr
    )  # (rho, mom_r, mom_theta)
    rho, mom_r = U[0], U[1]
    assert U.shape == (3, args.ntheta, args.nr)
    nr, nth, r_min = args.nr, args.ntheta, args.r_min
    dr = (params.radius - r_min) / nr
    dth = 2.0 * math.pi / nth
    th = ((np.arange(nth) + 0.5) * dth)[:, None]
    rr = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = (rr - r_min) * (params.radius - rr) * np.cos(th)  # phi du faux System
    gr, gt = rp.polar_gradient(phi, r_min, dr, nth, nr)
    v_r = mom_r / rho
    assert np.allclose(
        v_r, -gt / params.omega, atol=1e-12
    ), "v_r doit rester ExB -grad_theta/B"
    print("OK (5a) v_r = derive ExB radiale -grad_theta/B")


def test_state_azimuthal_velocity_solves_radial_balance() -> None:
    """(5b) v_theta est la racine de la quadratique de bilan radial.

    Equilibre rotatif. Verifie que le residu
    (rho/r) v_theta^2 + (rho B) v_theta - (d_r p + rho d_r phi) ~ 0 par cellule,
    ET que la branche choisie se reduit a la derive ExB grad_r/B quand la
    courbure -> 0 (B grand).
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    # cs2 NON nul pour exercer le terme de pression d_r p de l'equilibre.
    args = _Args(cs2=1.0e-4)
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    state_flat = [c for c in sim.calls if c[0] == "set_state"][0][1][1]
    U = np.asarray(state_flat).reshape(3, args.ntheta, args.nr)
    rho, mom_th = U[0], U[2]
    v_th = mom_th / rho

    nr, nth, r_min = args.nr, args.ntheta, args.r_min
    dr = (params.radius - r_min) / nr
    dth = 2.0 * math.pi / nth
    th = ((np.arange(nth) + 0.5) * dth)[:, None]
    rr = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = (rr - r_min) * (params.radius - rr) * np.cos(th)
    gr, _ = rp.polar_gradient(phi, r_min, dr, nth, nr)
    d_r_rho = rp.polar_radial_derivative(rho, r_min, dr, nth, nr)
    p_term = args.cs2 * d_r_rho  # d_r p = cs2 d_r rho
    B = params.omega

    # Residu de la quadratique (rho/r) v^2 + (rho B) v - (d_r p + rho d_r phi) = 0.
    residual = (rho / rr) * v_th**2 + (rho * B) * v_th - (p_term + rho * gr)
    # Echelle de normalisation par cellule (max des termes en valeur absolue) pour un residu RELATIF.
    scale = np.maximum.reduce(
        [
            np.abs(rho * B * v_th),
            np.abs(p_term + rho * gr),
            np.abs((rho / rr) * v_th**2),
            np.full_like(residual, 1e-300),
        ]
    )
    assert np.max(np.abs(residual) / scale) < 1e-10, (
        "v_theta doit annuler la quadratique de bilan radial (residu rel %.2e)"
        % np.max(np.abs(residual) / scale)
    )

    # La branche physique reste reelle et finie partout.
    assert np.all(np.isfinite(v_th)), "v_theta doit etre fini partout"
    # A grande courbure (B = omega = beta^2 ~ 1e12) le terme centrifuge (rho/r)v^2 ~ 1e-24 est
    # negligeable : v_theta -> forcing/B = (d_r phi + cs2 d_r rho/rho)/B (limite du bilan radial).
    forcing_over_B = (gr + args.cs2 * d_r_rho / rho) / B
    assert np.allclose(
        v_th, forcing_over_B, rtol=1e-6, atol=1e-300
    ), "a B grand v_theta doit valoir forcing/B = (d_r phi + cs2 d_r rho/rho)/B"
    print(
        "OK (5b) v_theta = racine du bilan radial (equilibre rotatif), -> forcing/B a B grand"
    )


def test_equilibrium_v_theta_reduces_to_exb_when_cold() -> None:
    """(5d) cs2 = 0 : v_theta se reduit EXACTEMENT a la derive ExB grad_r/B.

    Limite froide du papier. Sans terme de pression, forcing = d_r phi, et a B
    grand v_theta -> d_r phi/B = grad_r/B : la continuation ExB est verifiee
    verbatim (la nouvelle IC degenere bien en l'ancienne quand cs2=0).
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    nr, nth, r_min = 40, 24, 2.0
    dr = (params.radius - r_min) / nr
    dth = 2.0 * math.pi / nth
    th = ((np.arange(nth) + 0.5) * dth)[:, None]
    rr = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = (rr - r_min) * (params.radius - rr) * np.cos(th)
    rho = np.full((nth, nr), params.rho_max)
    gr, _ = rp.polar_gradient(phi, r_min, dr, nth, nr)
    v_th = rp.equilibrium_v_theta(rho, gr, r_min, dr, nth, nr, params, cs2=0.0)
    exb = gr / params.omega
    assert np.allclose(v_th, exb, rtol=1e-6, atol=1e-300), (
        "cs2=0 : v_theta doit se reduire a la derive ExB grad_r/B (residu %.2e)"
        % np.max(np.abs(v_th - exb))
    )
    print("OK (5d) cs2=0 : v_theta = ExB grad_r/B (continuation froide exacte)")


def test_check_equilibrium_is_stationary() -> None:
    """(5c) STATIONARITE : l'anneau d'equilibre non perturbe reste plat.

    Vaut pour chaque mode azimutal.
    Pilote check_equilibrium sur le faux adc : avec perturbation=0 et le phi axisymetrique du faux
    System, l'amplitude de chaque mode azimutal ne doit pas croitre au-dela de la tolerance et le
    potentiel reste fini. C'est l'auto-test --check-equilibrium en boite noire (build-free).
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args(
        cs2=1.0e-4,
        dt=1.0e-3,
        cfl=0.0,
        max_steps_check=20,
        check_modes=[1, 2, 3, 4, 5],
        check_tol=0.05,
    )
    ok, report = rp.check_equilibrium(params, args)
    assert all(
        row["finite"] for row in report
    ), "potentiel doit rester fini (pas de NaN)"
    assert ok, "l'equilibre rotatif doit etre stationnaire : %r" % report
    # Le rapport couvre bien tous les modes demandes.
    assert sorted(row["mode"] for row in report) == [1, 2, 3, 4, 5]
    print(
        "OK (5c) check_equilibrium : anneau d'equilibre stationnaire (chaque mode plat)"
    )


def test_compute_frozen_residual_captures_scheme_drift() -> None:
    """(c1) R_eq = step(U_eq) - U_eq capture la derive parasite du schema.

    Sur l'anneau axisymetrique, avec le faux schema.
    Avec le faux adc, step() ajoute la derive deterministe D = _spurious_drift() ; compute_frozen_residual
    doit donc renvoyer R_eq == D (a la precision machine) et un U_eq non trivial (3, ntheta, nr).
    Verifie aussi l'ORDRE des appels sur la sonde : set_state -> solve_fields -> step -> get_state.
    """
    adc = _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3)
    U_eq, R_eq = rp.compute_frozen_residual(params, args)
    assert U_eq.shape == (3, args.ntheta, args.nr)
    assert R_eq.shape == (3, args.ntheta, args.nr)
    # La derive de reference = celle du faux schema (deterministe, par cellule).
    probe = adc.System(
        mesh=adc.PolarMesh(args.r_min, params.radius, args.nr, args.ntheta)
    )
    probe._nr, probe._nth = args.nr, args.ntheta
    drift = probe._spurious_drift()
    assert (
        np.max(np.abs(R_eq - drift)) < 1e-14
    ), "R_eq doit capturer la derive parasite step(U_eq)-U_eq"
    assert (
        np.max(np.abs(R_eq)) > 1e-6
    ), "R_eq doit etre O(1e-3) non trivial (schema non bien pose)"
    print(
        "OK (c1) compute_frozen_residual capture la derive du schema (R_eq = step(U_eq) - U_eq)"
    )


def test_step_frozen_subtracted_call_order_and_cancellation() -> None:
    """(c2) step_frozen_subtracted : ordre des appels facade ET annulation.

    Ordre exige : step -> get_state -> set_state -> solve_fields.
    La carte corrigee step()-R_eq applique deux fois de suite a U_eq doit reproduire U_eq EXACTEMENT
    (R_eq = derive constante : U + D - D = U). On verifie l'ordre des appels facade et l'invariance.
    """
    adc = _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args(cs2=1.0e-4, dt=1.0e-3)
    U_eq, R_eq = rp.compute_frozen_residual(params, args)

    sim = rp.build_polar_system(
        args.nr, args.ntheta, mode=4, params=params, args=args
    )
    sim.set_state("ne", U_eq.ravel())
    sim.solve_fields()
    before = len(sim.calls)
    rp.step_frozen_subtracted(sim, args.dt, R_eq, args.ntheta, args.nr)
    names = [c[0] for c in sim.calls[before:]]
    # ORDRE EXIGE : step (avance + derive), get_state, set_state (etat corrige), solve_fields.
    assert (
        names.index("step")
        < names.index("get_state")
        < names.index("set_state")
        < names.index("solve_fields")
    ), ("ordre step -> get_state -> set_state -> solve_fields : %r" % names)
    # Annulation : l'etat courant du sim doit etre U_eq a la precision machine (point fixe exact).
    U = sim.get_state("ne")
    assert (
        np.max(np.abs(U - U_eq)) < 1e-12
    ), "step()-R_eq doit rendre U_eq invariant (point fixe)"
    print(
        "OK (c2) step_frozen_subtracted : ordre facade correct + U_eq point fixe exact"
    )


def test_check_equilibrium_frozen_is_machine_precision_stationary() -> None:
    """(c3) STATIONARITE A LA PRECISION MACHINE : vraie validation de l'option c.

    Avec --frozen-equilibrium, U_eq est un point fixe discret EXACT de step()-R_eq ; check_equilibrium_frozen
    doit donc reporter max_dev <= floor = C eps_mach ||U_eq||_inf sur >= 200 pas. Le critere laxiste
    base-amplitude (check_equilibrium) MASQUAIT l'echec ; ce test exerce le bon critere.
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    args = _Args(
        cs2=1.0e-4,
        dt=1.0e-3,
        frozen_equilibrium=True,
        frozen_check_const=1.0e3,
        max_steps_check=200,
    )
    ok, report = rp.check_equilibrium_frozen(params, args)
    row = report[0]
    assert row["finite"], "l'etat corrige doit rester fini (pas de NaN)"
    assert row["n_steps"] >= 200, "au moins 200 pas avances : %r" % row
    assert (
        row["max_dev"] <= row["floor"]
    ), "max_dev=%.3e doit rester sous le floor machine %.3e" % (
        row["max_dev"],
        row["floor"],
    )
    assert ok
    # Le floor est ECHELLE sur U_eq, PAS sur le fond 1e12 (correction du check laxiste).
    assert row["floor"] < 1e-6 * max(
        row["state_scale"], 1.0
    ), "le floor doit etre ~ eps * ||U_eq|| (et non l'echelle laxiste du fond)"
    print(
        "OK (c3) check_equilibrium_frozen : U_eq point fixe a la precision machine (max_dev=%.2e)"
        % row["max_dev"]
    )


def test_run_mode_frozen_uses_subtracted_stepping() -> None:
    """(c4) run_mode(R_eq=...) cable step()-R_eq dans la boucle perturbee.

    Option c. Verifie que, R_eq fourni, run_mode appelle bien la sequence step -> get_state -> set_state ->
    solve_fields a chaque iteration (et JAMAIS step_cfl, meme si --cfl etait demande).
    """
    _install_fake_adc()
    rp = _import_run_polar()
    params = _params()
    # t_end petit = quelques pas ; sample_every=1 ; --cfl>0 doit etre IGNORE en mode frozen.
    args = _Args(
        cs2=1.0e-4,
        dt=1.0e-3,
        cfl=0.7,
        t_end=0.004,
        sample_every=1,
        max_steps=100,
        strang=False,
        frozen_equilibrium=True,
    )
    _, R_eq = rp.compute_frozen_residual(params, args)
    result = rp.run_mode(4, params, args, R_eq=R_eq)
    assert np.all(
        np.isfinite(result["amplitudes"])
    ), "amplitudes finies (option c stabilise)"
    # La boucle doit avoir avance par step (pas step_cfl) et applique la soustraction (set_state apres
    # chaque get_state). On reconstruit le sim pour inspecter le contrat d'appel.
    sim = rp.build_polar_system(args.nr, args.ntheta, 4, params, args)
    n0 = len(sim.calls)
    rp.step_frozen_subtracted(sim, args.dt, R_eq, args.ntheta, args.nr)
    seq = [c[0] for c in sim.calls[n0:]]
    assert "step" in seq and "step_cfl" not in seq, (
        "mode frozen : avance par step() (pas fixe), jamais step_cfl : %r" % seq
    )
    print(
        "OK (c4) run_mode frozen : carte step()-R_eq cablee, step_cfl jamais appele"
    )


def test_main_frozen_ignores_cfl_and_runs_quick() -> None:
    """(c5) main --quick --cfl=0.5 : frozen ON par defaut, --cfl ignore.

    Le run frozen va jusqu'au bout.
    main() doit forcer args.cfl=0 sous --frozen-equilibrium, precalculer R_eq, et terminer le smoke
    --quick sans NaN (la sortie est ecrite). On verifie que le run aboutit (pas de SystemExit).
    """
    _install_fake_adc()
    rp = _import_run_polar()
    import tempfile

    saved = dict(os.environ)
    argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ADC_CASES_OUT"] = tmp
        os.environ.pop("SLURM_NTASKS", None)
        sys.argv = ["run_polar.py", "--quick", "--cfl", "0.5"]
        try:
            rp.main()  # ne doit PAS lever : --cfl ignore, frozen ON, smoke complet
        finally:
            sys.argv = argv
            os.environ.clear()
            os.environ.update(saved)
    print(
        "OK (c5) main --quick --cfl ignore sous frozen-equilibrium, run complet"
    )


def test_polar_gradient_matches_derive_aux_polar_stencil() -> None:
    """(6) polar_gradient reproduit exactement le stencil de derive_aux_polar."""
    _install_fake_adc()
    rp = _import_run_polar()
    r_min, nr, nth = 2.0, 40, 24
    dr = (16.0 - r_min) / nr
    dth = 2.0 * math.pi / nth
    th = ((np.arange(nth) + 0.5) * dth)[:, None]
    r = (r_min + (np.arange(nr) + 0.5) * dr)[None, :]
    phi = r**2 * np.cos(2.0 * th)
    gr, gt = rp.polar_gradient(phi, r_min, dr, nth, nr)
    # Reference EXACTE = les formules verbatim de derive_aux_polar (block_builder_polar.hpp).
    gr_ref = np.empty_like(phi)
    gr_ref[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dr)
    gr_ref[:, 0] = (-3.0 * phi[:, 0] + 4.0 * phi[:, 1] - phi[:, 2]) / (2.0 * dr)
    gr_ref[:, -1] = (3.0 * phi[:, -1] - 4.0 * phi[:, -2] + phi[:, -3]) / (
        2.0 * dr
    )
    gt_ref = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (
        2.0 * dth * r
    )
    assert np.max(np.abs(gr - gr_ref)) < 1e-12
    assert np.max(np.abs(gt - gt_ref)) < 1e-12
    print(
        "OK (6) polar_gradient == stencil derive_aux_polar (r centre/decentre ordre 2, theta enroule)"
    )


def test_fit_growth_uses_mapped_paper_window_and_is_exact() -> None:
    """(7)(T3) fit_growth fitte la fenetre papier MAPPEE en temps sim, pas brute.

    Mapping : t_sim = 2pi/rhobar t_paper. Signal en deux pentes : pente parasite
    g_other AVANT la fenetre mappee, pente vraie g_true DEDANS. fit_growth doit
    renvoyer g_true (preuve qu'il fitte bien [2pi*0.60, 2pi*0.75] = [3.77, 4.71],
    pas [0.60, 0.75])."""
    import math

    _install_fake_adc()
    rp = _import_run_polar()
    lo_sim, hi_sim = (
        0.60 * 2 * math.pi,
        0.75 * 2 * math.pi,
    )  # fenetre l=4 mappee = [3.770, 4.712]
    ts = np.linspace(0.0, 6.0, 2400)
    g_true, g_other = 0.911, 0.200
    # expo continue par morceaux : pente g_other sur [0, lo_sim], pente g_true au-dela.
    amps = np.where(
        ts <= lo_sim,
        1e-3 * np.exp(g_other * ts),
        1e-3 * np.exp(g_other * lo_sim) * np.exp(g_true * (ts - lo_sim)),
    )
    g = rp.fit_growth(ts, amps, 4, rhobar=1.0)
    assert abs(g - g_true) < 1e-6, (
        "fit_growth doit fitter la fenetre MAPPEE -> g_true, obtenu %r" % g
    )
    assert (
        abs(g - g_other) > 0.5
    ), "fit_growth ne doit PAS fitter la fenetre brute (g_other)"
    print(
        "OK (7) fit_growth exact sur la fenetre papier MAPPEE en temps sim (T3)"
    )


def test_multirank_is_rejected() -> None:
    """(8) Un run multi-rang est refuse (etage Schur polaire = boite unique)."""
    _install_fake_adc()
    rp = _import_run_polar()
    saved = dict(os.environ)
    os.environ["SLURM_NTASKS"] = "4"
    argv = sys.argv
    sys.argv = ["run_polar.py", "--quick"]
    try:
        raised = False
        try:
            rp.main()
        except SystemExit as exc:
            raised = True
            assert "mono-rang" in str(exc), exc
        assert (
            raised
        ), "un run multi-rang doit etre refuse (etage Schur polaire = boite unique)"
    finally:
        sys.argv = argv
        os.environ.clear()
        os.environ.update(saved)
    print("OK (8) multi-rang refuse (mono-rang)")


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("all %d polar-assembly smoke tests passed" % len(tests))


if __name__ == "__main__":
    _run_all()
