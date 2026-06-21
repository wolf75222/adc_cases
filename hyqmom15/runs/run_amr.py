#!/usr/bin/env python3
"""Cas hyqmom15/run_amr : le modele 15 moments HyQMOM porte sur adc.AmrSystem.

Premier run du bloc compile (build_moment_model, exact_speeds=True) sur une
hierarchie AMR.

Pourquoi ce cas
---------------
Quatrieme brique de l'integration HyQMOM : apres le System uniforme (run_diocotron), le meme
.so (backend='production', target='amr_system') est porte sur la hierarchie adaptative --
raffinement sur M00 (l'anneau diocotron est taggue et couvert par des patchs fins), Poisson
composite geometric_mg, riemann='hll' avec les vitesses exactes.

Schema temporel : sur AMR, adc.Explicit() par defaut = EULER AVANT + reflux conservatif par
pas -- c'est le schema le PLUS fidele au MATLAB de reference (split additif + Euler == Euler
non-splite, prouve par le replay de run_crossing). ssprk3 est cable sur AMR mais pour les
blocs NATIFS add_block seulement : l'ABI plate du loader .so ne transporte pas la methode
temporelle, la demande est rejetee explicitement (verifie en (4)). Les runs System de
comparaison utilisent donc time=adc.Explicit(method='euler') (meme schema, parite stricte).

relaxation15 (projection de realisabilite, boucle numpy par champ) n'est PAS disponible sur
AMR : pas d'acces in-place a l'etat des niveaux entre les pas. Smoke a horizon court
uniquement ; les runs longs realisables restent sur le System uniforme.

Validation
----------
  (1) smoke diocotron AMR : anneau + derive ExB (IC de run_diocotron), n=48, regrid_every=4,
      12 pas step_cfl(0.4) ; etat fini sur TOUS les niveaux, M00 > 0 partout, phi fini,
      la zone M00 > seuil est couverte par les patchs fins ; masse conservee : derive de
      sim.mass() < 1e-9 ET la somme ponderee par niveau (dx_k^2 sur les cellules valides non
      recouvertes, calculee ici depuis level_state/patch_boxes) reste egale a sim.mass() ;
  (2) coherence grossier/fin : MEME IC (prolongation constante, exactement le seed AMR),
      MEME dt, MEME schema euler -> le run AMR (1 niveau fin, hierarchie figee) est compare
      au System uniforme a la resolution fine (96^2), restreint au grossier. Dans la zone
      raffinee l'ecart L2 relatif sur M00 est petit ET strictement plus petit que celui du
      System uniforme grossier (48^2) : le raffinement rapproche bien de la reference fine ;
  (3) garde-fous de composition : backend='aot' + target='amr_system' rejete a la
      compilation ; un .so target='system' est refuse par AmrSystem.add_equation ;
      riemann='hll' sans vitesses signees (exact_speeds=False) est refuse AVANT le C++ ;
  (4) ssprk3 + loader .so : rejet explicite (jamais un repli Euler silencieux).

Ne prouve pas : le taux de croissance diocotron sur AMR (horizon court, smoke) ; la
realisabilite long-terme (relaxation15 absente du chemin AMR) ; le MPI (mono-rang).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # hyqmom15/ : model, relaxation, gen_states

from model import build_moment_model  # noqa: E402
from run_diocotron import DEBYE, OMEGA_P, diocotron_state  # noqa: E402

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import adc  # noqa: E402

N = 48  # grossier AMR (et System de controle) ; fin = 2N
NSTEPS_SMOKE = 12  # (1) : 10-20 pas, smoke
NSTEPS_REF = 15  # (2) : meme horizon pour les trois runs
DT_REF = 8.0e-4  # (2) : dt fixe, CFL fine ~ 0.26 (vitesses ~ 3.3, h_f = 1/96)
REFINE_THRESHOLD = (
    0.5  # M00 de l'anneau dans [0.8, 1.0], fond 1e-4 : seuil discriminant
)
# (2) : seuil L2 relatif dans la zone raffinee, choisi APRES mesure : ecart mesure 3.4e-3
# pour l'AMR (bord grossier-fin + Poisson composite vs uniforme sur la fenetre) contre
# 1.1e-1 pour le grossier uniforme (l'AMR rapproche d'un facteur ~30). Marge ~x3 sur la
# mesure AMR ; l'inegalite stricte AMR < grossier (assertee a part) prouve que le
# raffinement rapproche de la reference fine. Smoke de coherence, pas une etude de
# convergence.
TOL_COHERENCE = 1.0e-2


def compile_model(
    name: str, rho_bg: float, target: str, exact_speeds: bool = True
):
    """Compile le modele Vlasov-Poisson 15 moments de run_diocotron en production.

    Modele a sources electriques + Poisson, compile en backend='production' (seul
    chemin .so branchable sur AmrSystem ; il marshale aussi method='euler' cote
    System, requis pour la parite de schema en (2)).

    Args:
        name: nom du modele et base du fichier .so genere.
        rho_bg: fond neutralisant du Poisson (moyenne du scenario).
        target: cible du loader ('amr_system' ou 'system').
        exact_speeds: vitesses d'onde exactes (sinon borne bring-up robuste).

    Returns:
        Le CompiledModel charge depuis le .so.
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    m = build_moment_model(
        name=name,
        robust=not exact_speeds,
        with_sources=True,
        q_over_m=1.0,
        omega_c=0.0,
        debye=DEBYE,
        rho_background=rho_bg,
        omega_p=OMEGA_P,
        exact_speeds=exact_speeds,
    )
    return m.compile(
        os.path.join(case_output_dir("hyqmom15"), name + ".so"),
        adc_include(),
        backend="production",
        target=target,
    )


def build_amr(
    compiled,
    U0: np.ndarray,
    regrid_every: int,
    riemann: str = "hll",
    time=None,
) -> adc.AmrSystem:
    """Construit un AmrSystem periodique mono-bloc seede sur U0.

    Raffinement sur M00, Poisson composite geometric_mg, etat conservatif complet
    (15 composantes) seede sur le grossier puis prolonge.
    """
    sim = adc.AmrSystem(n=N, L=1.0, periodic=True, regrid_every=regrid_every)
    sim.add_equation(
        "mom",
        compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
        time=time if time is not None else adc.Explicit(),
    )
    sim.set_refinement(threshold=REFINE_THRESHOLD)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_conservative_state("mom", U0)
    return sim


def build_system(compiled, U0: np.ndarray, n: int) -> adc.System:
    """System uniforme de reference, MEME schema temporel que l'AMR (euler avant)."""
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        "mom",
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        time=adc.Explicit(method="euler"),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_state("mom", U0)
    sim.solve_fields()
    return sim


def fine_masks(sim) -> tuple[np.ndarray, np.ndarray]:
    """Masques de la hierarchie depuis patch_boxes().

    Convention [j, i] (j = y, i = x), boites a coins inclusifs dans l'espace
    d'indices du niveau.

    Returns:
        (covered, valid) : cellules grossieres recouvertes (N x N) et cellules
        fines valides (2N x 2N).
    """
    covered = np.zeros((N, N), dtype=bool)
    valid = np.zeros((2 * N, 2 * N), dtype=bool)
    for lev, ilo, jlo, ihi, jhi in sim.patch_boxes():
        if lev != 1:
            continue
        valid[jlo : jhi + 1, ilo : ihi + 1] = True
        covered[jlo // 2 : jhi // 2 + 1, ilo // 2 : ihi // 2 + 1] = True
    return covered, valid


def composite_mass(sim) -> float:
    """Masse par somme ponderee par niveau (dx0^2 grossier + dx1^2 fin).

    dx0^2 sur les cellules grossieres NON recouvertes + dx1^2 sur les cellules
    fines valides. Doit coincider avec sim.mass() (masse du grossier apres
    sync_down : les cellules recouvertes y valent la moyenne de leurs 4 filles).
    """
    covered, valid = fine_masks(sim)
    dx0 = 1.0 / N
    m00_c = np.asarray(sim.level_state(0)).reshape(-1, N, N)[0]
    total = float(m00_c[~covered].sum()) * dx0 * dx0
    if sim.n_levels() >= 2 and valid.any():
        m00_f = np.asarray(sim.level_state(1)).reshape(-1, 2 * N, 2 * N)[0]
        total += float(m00_f[valid].sum()) * (dx0 / 2.0) ** 2
    return total


def check_smoke(compiled_amr, U0: np.ndarray) -> None:
    """(1) construction + pas finis, positivite, couverture tagging, masse conservee."""
    sim = build_amr(compiled_amr, U0, regrid_every=4)
    mass0 = sim.mass()
    assert np.isfinite(mass0) and mass0 > 0.0, "masse initiale invalide"

    # le tagging couvre l'anneau : cellules M00 > seuil sous les patchs fins (>= 95 %)
    covered0, _ = fine_masks(sim)
    ring = U0[0] > REFINE_THRESHOLD
    frac = float((ring & covered0).sum()) / float(ring.sum())
    assert frac > 0.95, "anneau non couvert par les patchs fins (%.1f %%)" % (
        100 * frac
    )
    assert sim.n_patches() >= 2, (
        "anneau couvert par moins de 2 patchs (%d)" % sim.n_patches()
    )

    for _ in range(NSTEPS_SMOKE):
        sim.step_cfl(0.4)
    nlev = sim.n_levels()
    st0 = np.asarray(sim.level_state(0)).reshape(-1, N, N)
    assert np.all(np.isfinite(st0)), "etat grossier non fini"
    assert np.all(st0[0] > 0), "M00 grossier non positif"
    if nlev >= 2:
        _, valid = fine_masks(sim)
        st1 = np.asarray(sim.level_state(1)).reshape(-1, 2 * N, 2 * N)
        assert np.all(np.isfinite(st1)), "etat fin non fini"
        assert np.all(st1[0][valid] > 0), "M00 fin non positif sur les patchs"
    assert np.all(np.isfinite(np.asarray(sim.potential()))), "phi non fini"

    drift = abs(sim.mass() - mass0) / mass0
    assert drift < 1e-9, "masse (reflux) non conservee : drel=%.3e" % drift
    dcomp = abs(composite_mass(sim) - sim.mass()) / mass0
    assert dcomp < 1e-12, (
        "somme ponderee par niveau != mass() (drel=%.3e) : sync_down "
        "incoherent" % dcomp
    )
    print(
        "(1) smoke AMR : %d pas (step_cfl 0.4, euler+reflux), %d niveaux, %d patchs, anneau "
        "couvert a %.0f %%, M00 > 0, phi fini ; masse : derive %.1e, somme par niveau == "
        "mass() a %.1e -- OK"
        % (NSTEPS_SMOKE, nlev, sim.n_patches(), 100 * frac, drift, dcomp)
    )


def check_coherence(compiled_amr, compiled_sys, U0: np.ndarray) -> None:
    """(2) AMR (hierarchie figee) vs System fin, meme IC prolongee, meme dt/schema."""
    U0f = np.repeat(
        np.repeat(U0, 2, axis=1), 2, axis=2
    )  # prolongation constante = seed AMR

    sim_amr = build_amr(
        compiled_amr, U0, regrid_every=0
    )  # figee : regrid jamais apres init
    sim_fine = build_system(compiled_sys, U0f, n=2 * N)
    sim_coarse = build_system(compiled_sys, U0, n=N)
    covered, _ = fine_masks(sim_amr)
    assert (
        covered.any()
    ), "aucune cellule grossiere recouverte (pas de niveau fin ?)"

    for _ in range(NSTEPS_REF):
        sim_amr.step(DT_REF)
        sim_fine.step(DT_REF)
        sim_coarse.step(DT_REF)

    rho_amr = np.asarray(sim_amr.density("mom"))  # grossier post sync_down
    rho_fine = np.array(sim_fine.get_state("mom"))[0]
    rho_ref = rho_fine.reshape(N, 2, N, 2).mean(
        axis=(1, 3)
    )  # restriction 2x2 -> N x N
    rho_coarse = np.array(sim_coarse.get_state("mom"))[0]
    assert np.all(np.isfinite(rho_amr)) and np.all(np.isfinite(rho_ref))

    den = float(np.sqrt(np.sum(rho_ref[covered] ** 2)))
    gap_amr = float(np.sqrt(np.sum((rho_amr - rho_ref)[covered] ** 2))) / den
    gap_coarse = (
        float(np.sqrt(np.sum((rho_coarse - rho_ref)[covered] ** 2))) / den
    )
    # mesures TOUJOURS imprimees avant les asserts : un echec reste diagnosticable.
    print(
        "(2) coherence grossier/fin (%d pas, dt=%.1e, euler partout) : ecart L2 relatif M00 "
        "zone raffinee AMR=%.2e (seuil %.1e) ; grossier uniforme=%.2e"
        % (NSTEPS_REF, DT_REF, gap_amr, TOL_COHERENCE, gap_coarse)
    )
    assert gap_amr < TOL_COHERENCE, (
        "coherence grossier/fin : ecart L2 relatif %.3e >= %.1e dans la zone raffinee"
        % (gap_amr, TOL_COHERENCE)
    )
    assert gap_amr < gap_coarse, (
        "l'AMR (%.3e) devrait etre plus proche de la reference fine que le grossier uniforme "
        "(%.3e) dans la zone raffinee" % (gap_amr, gap_coarse)
    )
    print(
        "    l'AMR rapproche de la reference fine (x%.1f vs grossier uniforme) -- OK"
        % (gap_coarse / gap_amr)
    )


def check_rejections(compiled_sys, rho_bg: float):
    """(3) rejets propres : aot+amr, target system sur AMR, hll sans vitesses signees."""
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    # Modele bring-up (borne k*sqrt(C), PAS de vitesses signees) : sert au rejet (3a) a la
    # compilation puis, compile, au rejet (3c) et au contrepoint rusanov de (4).
    m_noex = build_moment_model(
        name="hyqmom15_noex_amr",
        robust=True,
        exact_speeds=False,
        debye=DEBYE,
        rho_background=rho_bg,
    )

    # (3a) backend='aot' ne cible pas l'AMR : rejete a la compilation, AVANT tout codegen.
    try:
        m_noex.compile(
            os.path.join(case_output_dir("hyqmom15"), "hyqmom15_aot_amr.so"),
            adc_include(),
            backend="aot",
            target="amr_system",
        )
        raise AssertionError(
            "compile(backend='aot', target='amr_system') aurait du lever"
        )
    except ValueError as e:
        assert "amr_system" in str(e)

    # (3b) un .so compile pour target='system' est refuse par AmrSystem.add_equation.
    sim = adc.AmrSystem(n=16, L=1.0, periodic=True)
    try:
        sim.add_equation(
            "mom",
            compiled_sys,
            spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        )
        raise AssertionError(
            "AmrSystem a accepte un CompiledModel target='system'"
        )
    except ValueError as e:
        assert "amr_system" in str(e) or "target" in str(e)

    # (3c) riemann='hll' sans vitesses signees (exact_speeds=False : borne bring-up seule,
    # pas de wave_speeds ni de primitive 'p') : refuse par la facade, avant la frontiere C++.
    cm_noex = m_noex.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_noex_amr.so"),
        adc_include(),
        backend="production",
        target="amr_system",
    )
    sim = adc.AmrSystem(n=16, L=1.0, periodic=True)
    try:
        sim.add_equation(
            "mom",
            cm_noex,
            spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        )
        raise AssertionError(
            "AmrSystem a accepte riemann='hll' sans vitesses signees"
        )
    except ValueError as e:
        assert "hll" in str(e).lower()
    print(
        "(3) rejets propres : aot+amr_system (compile), .so target='system' (add_equation), "
        "hll sans vitesses signees -- OK"
    )
    return cm_noex


def check_ssprk3_rejected(compiled_amr, cm_noex, U0: np.ndarray) -> None:
    """(4) ssprk3 + loader .so : rejet explicite, jamais un repli Euler silencieux.

    L'ABI plate ne transporte pas la methode temporelle. En contrepoint, le meme
    .so bring-up tourne en rusanov + Explicit() (euler avant) : le rejet vise la
    METHODE, pas le bloc.
    """
    try:
        build_amr(
            compiled_amr, U0, regrid_every=0, time=adc.Explicit(ssprk3=True)
        )
        raise AssertionError("AmrSystem a accepte ssprk3 sur un loader .so")
    except ValueError as e:
        assert "ssprk3" in str(e).lower(), "message inattendu : %s" % e

    sim = build_amr(cm_noex, U0, regrid_every=0, riemann="rusanov")
    for _ in range(2):
        sim.step_cfl(0.4)
    assert np.all(
        np.isfinite(np.asarray(sim.density("mom")))
    ), "bring-up rusanov non fini"
    print(
        "(4) ssprk3 sur loader .so rejete explicitement ; le meme bloc en rusanov + "
        "Explicit() (euler avant) tourne -- OK"
    )


def main() -> None:
    print(
        "=== hyqmom15/run_amr : 15 moments HyQMOM sur AmrSystem (hll exact, MG composite) ==="
    )
    U0 = diocotron_state(N)
    rho_bg = float(
        U0[0].mean()
    )  # fond neutralisant : rhs periodique a moyenne nulle
    compiled_amr = compile_model("hyqmom15_vp_amr", rho_bg, target="amr_system")
    compiled_sys = compile_model("hyqmom15_vp_fine", rho_bg, target="system")
    check_smoke(compiled_amr, U0)
    check_coherence(compiled_amr, compiled_sys, U0)
    cm_noex = check_rejections(compiled_sys, rho_bg)
    check_ssprk3_rejected(compiled_amr, cm_noex, U0)
    print("hyqmom15/run_amr : OK")


if __name__ == "__main__":
    main()
