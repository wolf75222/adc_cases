#!/usr/bin/env python3
"""Cas hyqmom15/run_fluid_wave : onde fluide eigenmode du nouveau Matlab.

Porte `fluid_wave` de RieMOM2D_Electrostatic_periodic (init_fluid_wave.m +
init_fluid_wave_field.m) : le cas wave le plus simple (pas de Poisson, pas de
source electrique ni magnetique). Perturbation eigenmode lineaire d'amplitude
eps=0.01 sur la Maxwellienne au repos, mode 15, kx=4*pi, ky=0. Voir
matlab_ref/REFERENCE.md et la couche matlab_ref (ADC-349).

Schema : le Matlab utilise space_scheme="ROE". Le hook Roe generique adc_cpp
(ADC-368, build_moment_model(roe=True) -> m.roe_from_jacobian) rend riemann="roe"
disponible pour ce modele de moments DSL sans primitive 'p', donc on suit le Matlab
a la lettre (ADC-371). reconstruction="first" -> limiter="none" ; time_scheme="Euler"
-> adc.Explicit(method="euler") sur backend="production".

Validation
----------
  (1) IC : matlab_ref.init_fluid_wave_field (verrouille vs golden Octave en
      ADC-350) finie ; L2(IC, t=0) == 0 contre la solution eigenmode analytique ;
  (2) smoke natif : ROE + exact_speeds + Euler (production), pas de Poisson ni de
      source : etat fini, M00 > 0, masse conservee (transport pur), et ROE suit
      l'eigenmode strictement mieux que le HLL qu'il remplace (L2_roe < L2_hll sur
      la meme IC et le meme nombre de pas, seul le solveur de Riemann change).

Ne prouve pas : le golden un-pas natif vs flux_ROE Matlab (suivi optionnel ADC-371),
la fidelite trajectoire longue.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import build_moment_model  # noqa: E402

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

import adc  # noqa: E402

from matlab_ref import compute_L2_error, explicit_for, get_case, init_fluid_wave_field  # noqa: E402

CASE = get_case("fluid_wave")  # Np=32, eps=0.01, mode=15, kx=4*pi, ky=0, no sources


def fluid_ic() -> np.ndarray:
    """IC eigenmode (15, Np, Np) via matlab_ref (couche verrouillee ADC-350)."""
    return init_fluid_wave_field(CASE).M


def build_fluid_model() -> object:
    """Compile le modele 15 moments ROE (transport pur, Euler production).

    Le hook Roe generique adc_cpp (ADC-368, build_moment_model(roe=True) ->
    m.roe_from_jacobian) emet la dissipation matrice-signe et rend riemann="roe"
    disponible (space_scheme="ROE" du Matlab). exact_speeds=True pour les vitesses
    d'onde signees, qui servent a la CFL et au HLL de comparaison. backend="production"
    pour Euler fidele (le backend "aot" figerait SSPRK2, cf. ADC-356).
    reconstruction="first" du Matlab -> limiter="none".
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name="hyqmom15_fluid",
        with_sources=False,       # fluid_wave : source=0, electrostatic=0, magnetostatic=0
        exact_speeds=True,
        roe=True,                 # hook Roe generique (ADC-368) -> riemann="roe"
    )
    return m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_fluid.so"),
        adc_include(),
        backend="production",
    )


def build_fluid_sim(n: int, compiled: object, riemann: str, name: str = "mom") -> "adc.System":
    """System periodique transport pur : un solveur de Riemann, Euler (production).

    Le meme modele compile (roe + exact_speeds) sert aux deux solveurs ; seul
    riemann change (ROE fidele, ou HLL de comparaison). Pas de source ni de Poisson.
    """
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        name,
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
        time=explicit_for(CASE.time_scheme),  # "Euler" -> adc.Explicit(method="euler")
    )
    return sim


def check_ic() -> None:
    """(1) IC matlab_ref : finie, M00 > 0, L2(IC, t=0) == 0 (eigenmode analytique)."""
    U = fluid_ic()
    assert U.shape == (15, CASE.Np, CASE.Np), "forme IC %s" % (U.shape,)
    assert np.all(np.isfinite(U)), "IC fluid_wave non finie"
    assert np.all(U[0] > 0), "IC fluid_wave : M00 non positif"
    l2 = compute_L2_error(U, 0.0, CASE)
    assert l2 < 1e-12, "L2(IC, t=0) = %g, attendu ~0" % l2
    print(
        "(1) IC matlab_ref (mode=%d, kx=4*pi, ky=0, eps=%g) : (15,%d,%d) finie, M00 > 0, "
        "L2(IC,0)=%.2e -- OK" % (CASE.mode, CASE.eps, CASE.Np, CASE.Np, l2)
    )


def _evolve(compiled: object, riemann: str, U0: np.ndarray, nsteps: int, cfl: float):
    """nsteps de transport pur (un solveur de Riemann), retourne (U_final, t)."""
    sim = build_fluid_sim(CASE.Np, compiled, riemann)
    # Orientation grille : l'IC eigenmode layer est (k, x, y) (i selon kx) ; ADC
    # attend (k, ny, nx) (x en dernier axe, cf. run_crossing) -> swapaxes(1,2) a
    # l'entree, retranspose a la sortie pour l'oracle L2 (convention layer).
    sim.set_state("mom", np.swapaxes(U0, 1, 2))
    t = 0.0
    for _ in range(nsteps):
        # fluid_wave n'a pas de source : compute_dt se reduit au CFL de flux, donc
        # step_cfl == dt Matlab (identique pour ROE et HLL : meme spectre exact).
        t += sim.step_cfl(cfl)
    U = np.swapaxes(np.array(sim.get_state("mom")), 1, 2)
    return U, t


def check_smoke(nsteps: int = 10) -> None:
    """(2) smoke natif ROE : transport pur fini + masse conservee + L2_roe < L2_hll.

    Critere relatif ADC-371 (pas de nouveau golden ni de tolerance absolue) : ROE
    (fidele a space_scheme="ROE") suit l'eigenmode analytique strictement mieux que le
    HLL qu'il remplace. Meme IC, meme modele compile (roe + exact_speeds), meme nombre
    de pas et meme CFL ; seul le solveur de Riemann change.
    """
    cfl = CASE.cfl
    U0 = fluid_ic()
    mass0 = float(U0[0].sum())
    compiled = build_fluid_model()

    # ROE : le chemin fidele (space_scheme="ROE" du Matlab), hook generique ADC-368.
    U_roe, t = _evolve(compiled, "roe", U0, nsteps, cfl)
    assert np.all(np.isfinite(U_roe)), "smoke ROE : etat non fini"
    assert np.all(U_roe[0] > 0), "smoke ROE : M00 non positif"
    drift = abs(float(U_roe[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "smoke ROE : masse non conservee (transport pur) (%.2e)" % drift
    l2_roe = compute_L2_error(U_roe, t, CASE)

    # HLL : le chemin non-strict que ROE remplace (meme modele, seul riemann change).
    U_hll, t_hll = _evolve(compiled, "hll", U0, nsteps, cfl)
    l2_hll = compute_L2_error(U_hll, t_hll, CASE)

    # ROE est moins diffusif que HLL sur cet eigenmode lisse : il doit le suivre mieux.
    assert l2_roe < l2_hll, (
        "smoke : ROE ne suit pas l'eigenmode mieux que HLL "
        "(L2_roe=%.3e >= L2_hll=%.3e)" % (l2_roe, l2_hll)
    )
    print(
        "(2) smoke natif ROE vs HLL + Euler(production), transport pur : %d pas "
        "(t=%.3e), M00 > 0, masse %.1e, L2_roe=%.4e < L2_hll=%.4e (-%.2f%% de "
        "diffusion, ADC-368/371) -- OK"
        % (nsteps, t, drift, l2_roe, l2_hll, 100.0 * (l2_hll - l2_roe) / l2_hll)
    )


def main() -> None:
    print("=== hyqmom15/run_fluid_wave : onde fluide eigenmode (ADC-352, ROE ADC-371) ===")
    check_ic()
    check_smoke()


if __name__ == "__main__":
    main()
