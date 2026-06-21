#!/usr/bin/env python3
"""Sources de la hierarchie de moments + croisement de jets 2D (Rusanov).

Premiere evolution temporelle du modele HyQMOM, sur la borne bring-up.

Pourquoi ce cas
---------------
Suite de run.py (qui valide le flux en
statique) : les termes sources electrique/magnetique de la hierarchie (document maths eq. 1.2)
sont generes programmatiquement (model.moment_sources) et le modele avance en temps dans un
adc.System sur le scenario de reference du depot MATLAB (croisement de jets,
main_pb_2Dcrossing_2DHyQMOM15.m), a Mach modere et sur Rusanov.

Validation
----------
  (1) oracle PDF : la table generee par moment_sources == les 15 equations EXPLICITES 1.3-1.7
      du document maths, ecrites A LA MAIN ici (jamais copiees du generateur), evaluees sur des
      etats et champs aleatoires -- rtol 1e-14 ;
  (2) structure : S[M00] = 0 (masse sans source) ; le terme magnetique conserve M20 + M02
      (rotation dans l'espace des vitesses : d(M20+M02)/dt = 2 oc M11 - 2 oc M11 = 0) ;
  (3) rotation de Larmor via System : etat gaussien UNIFORME derivant (ux0, 0), omega_c != 0,
      E = 0 -> l'advection ne contribue pas (periodique uniforme), la source fait tourner la
      vitesse moyenne : M10(t) = rho ux0 cos(oc t), M01(t) = -rho ux0 sin(oc t) ; compare a
      l'analytique apres ~1/8 de tour (rtol 1e-3, erreur d'integration temporelle dominante) ;
  (4) crossing smoke : IC de reference (carre central coupe par l'anti-diagonale, fond a
      rho = 1e-3) a Ma = 2 (modere : le Ma = 20 du MATLAB exige la projection de realisabilite
      relaxation15, non portee -- question ouverte), robust=True, rusanov, 10 pas CFL 0.4 :
      etat fini, M00 > 0, C20/C02 >= 0 partout, masse conservee a 1e-12 ;
  (5) snapshot npz des 15 moments ecrit et relisible (System.write) ;
  (6) fidelite HLL : adc rejoue le golden HLL de reference (golden_hll_gen.m, Octave sur
      RIEMOM2D, crossing Ma = 2, Np = 64, 20 pas) avec exact_speeds en riemann='hll' ssprk2 ;
      l'ecart L2 mesure la difference de schema seule, et adc-HLL est plus proche du golden
      que adc-Rusanov (check_hll_fidelity).

Ne prouve pas : la fidelite du crossing a Ma = 20 (qui exige la projection relaxation15, validee
isolement par run_relaxation, pas en ligne ici) ; le couplage Poisson (E reste nul dans ce smoke,
valide par run_diocotron) ; toute convergence en maillage.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # hyqmom15/ : model, relaxation, gen_states

from model import (  # noqa: E402
    MOMENT_NAMES,
    build_moment_model,
    crossing_state,
    gaussian_state,
    moment_sources,
)

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import adc  # noqa: E402


def pdf_source_table(
    M: dict, ex: float, ey: float, qm: float, oc: float
) -> list:
    """Les 15 equations sources EXPLICITES du document maths (eq. 1.3 a 1.7).

    Transcrites a la main dans l'ordre de MOMENT_NAMES. Oracle de l'invariant
    (1) : toute faute du generateur programmatique (indices, coefficients,
    signes) casse la comparaison.
    """
    return [
        0.0,  # M00 (1.3)
        qm * M["M00"] * ex + oc * M["M01"],  # M10 (1.4)
        qm * 2 * M["M10"] * ex + 2 * oc * M["M11"],  # M20 (1.5)
        qm * 3 * M["M20"] * ex + 3 * oc * M["M21"],  # M30 (1.6)
        qm * 4 * M["M30"] * ex + 4 * oc * M["M31"],  # M40 (1.7)
        qm * M["M00"] * ey - oc * M["M10"],  # M01 (1.4)
        qm * (M["M01"] * ex + M["M10"] * ey)
        + oc * (M["M02"] - M["M20"]),  # M11 (1.5)
        qm * (2 * M["M11"] * ex + M["M20"] * ey)
        + oc * (2 * M["M12"] - M["M30"]),  # M21 (1.6)
        qm * (3 * M["M21"] * ex + M["M30"] * ey)
        + oc * (3 * M["M22"] - M["M40"]),  # M31 (1.7)
        qm * 2 * M["M01"] * ey - 2 * oc * M["M11"],  # M02 (1.5)
        qm * (M["M02"] * ex + 2 * M["M11"] * ey)
        + oc * (M["M03"] - 2 * M["M21"]),  # M12 (1.6)
        qm * (2 * M["M12"] * ex + 2 * M["M21"] * ey)
        + 2 * oc * (M["M13"] - M["M31"]),  # M22 (1.7)
        qm * 3 * M["M02"] * ey - 3 * oc * M["M12"],  # M03 (1.6)
        qm * (M["M03"] * ex + 3 * M["M12"] * ey)
        + oc * (M["M04"] - 3 * M["M22"]),  # M13 (1.7)
        qm * 4 * M["M03"] * ey - 4 * oc * M["M13"],  # M04 (1.7)
    ]


def check_sources_vs_pdf() -> None:
    """(1) + (2) : generateur == equations explicites, sur etats/champs aleatoires."""
    rng = np.random.default_rng(7)
    for trial in range(20):
        M = {nm: float(rng.normal()) for nm in MOMENT_NAMES}
        ex, ey = float(rng.normal()), float(rng.normal())
        qm, oc = float(rng.normal()), float(rng.normal())
        gen = moment_sources(M, ex, ey, qm, oc)
        ref = pdf_source_table(M, ex, ey, qm, oc)
        for k, nm in enumerate(MOMENT_NAMES):
            g = float(gen[k]) if not isinstance(gen[k], float) else gen[k]
            np.testing.assert_allclose(
                g,
                ref[k],
                rtol=1e-14,
                atol=1e-14,
                err_msg="source de %s (tirage %d)" % (nm, trial),
            )
        assert (
            gen[0] == 0.0
        ), "S[M00] doit etre exactement 0 (masse sans source)"
        # rotation pure (E = 0) : d(M20 + M02)/dt = 0
        gb = moment_sources(M, 0.0, 0.0, 0.0, oc)
        trace = float(gb[MOMENT_NAMES.index("M20")]) + float(
            gb[MOMENT_NAMES.index("M02")]
        )
        assert (
            abs(trace) < 1e-14
        ), "le terme magnetique doit conserver M20 + M02"
    print(
        "(1)(2) sources == eq. 1.3-1.7 du document (20 tirages), S[M00] = 0, "
        "trace M20+M02 conservee par B -- OK"
    )


def check_larmor_rotation() -> None:
    """(3) rotation de Larmor de la vitesse moyenne a travers le System complet.

    Source compilee dans la brique, integration ssprk2 : etat uniforme
    periodique, E = 0, omega_c = 2.
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    oc, ux0, rho = 2.0, 0.7, 1.3
    m = build_moment_model(
        name="hyqmom15_larmor", with_sources=True, q_over_m=1.0, omega_c=oc
    )
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_larmor.so"),
        adc_include(),
        backend="aot",
    )
    n = 8
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        "mom",
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
        time=adc.Explicit(),
    )
    U0 = np.empty((15, n, n))
    U0[:] = gaussian_state(rho, ux0, 0.0, 1.0, 0.0, 1.0)[:, None, None]
    sim.set_state("mom", U0)
    # ~1/8 de tour en 200 pas : erreur ssprk2 ~ (oc dt)^2 par pas, negligeable devant 1e-3
    t_end = (np.pi / 4.0) / oc
    nsteps = 200
    dt = t_end / nsteps
    for _ in range(nsteps):
        sim.step(dt)
    U = np.array(sim.get_state("mom"))
    m10 = float(np.mean(U[1]))
    m01 = float(np.mean(U[5]))
    ref10 = rho * ux0 * np.cos(oc * t_end)
    ref01 = -rho * ux0 * np.sin(oc * t_end)
    scale = rho * ux0
    assert (
        abs(m10 - ref10) < 1e-3 * scale
    ), "M10 : %r != %r (rotation de Larmor)" % (m10, ref10)
    assert (
        abs(m01 - ref01) < 1e-3 * scale
    ), "M01 : %r != %r (rotation de Larmor)" % (m01, ref01)
    # l'energie agitee + dirigee tourne sans se dissiper : M20 + M02 conserve
    tr0 = float(np.mean(U0[2] + U0[9]))
    tr1 = float(np.mean(U[2] + U[9]))
    assert abs(tr1 - tr0) < 1e-10 * abs(
        tr0
    ), "M20 + M02 doit etre conserve par la rotation"
    print(
        "(3) rotation de Larmor via System : M10/M01 == analytique a 1e-3 apres 1/8 de tour, "
        "M20 + M02 conserve -- OK"
    )


def check_crossing_smoke() -> None:
    """(4) + (5) : croisement de jets a Ma = 2, robust, rusanov, 10 pas ; snapshot npz."""
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    n, ma, nsteps = 64, 2.0, 10
    m = build_moment_model(name="hyqmom15_robust", robust=True)
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_robust.so"),
        adc_include(),
        backend="aot",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        "mom",
        model=compiled,
        spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
        time=adc.Explicit(),
    )
    U0 = crossing_state(n, ma)
    sim.set_state("mom", U0)
    mass0 = float(U0[0].sum())
    for _ in range(nsteps):
        sim.step_cfl(0.4)
    U = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(U)), "etat non fini apres %d pas" % nsteps
    assert np.all(U[0] > 0), "M00 non strictement positif"
    ux, uy = U[1] / U[0], U[5] / U[0]
    c20 = U[2] / U[0] - ux * ux
    c02 = U[9] / U[0] - uy * uy
    assert np.all(c20 >= 0) and np.all(c02 >= 0), "variances C20/C02 negatives"
    drift = abs(float(U[0].sum()) - mass0) / mass0
    assert drift < 1e-12, "masse non conservee (derive %.3e)" % drift
    out = sim.write(
        os.path.join(case_output_dir("hyqmom15"), "crossing"), format="npz"
    )
    snap = np.load(out)
    state_keys = [k for k in snap.files if "state" in k or "mom" in k]
    assert state_keys, "snapshot npz sans etat (%r)" % snap.files
    print(
        "(4) crossing Ma = %g : %d pas finis, M00 > 0, C20/C02 >= 0, derive de masse %.1e ; "
        "(5) snapshot %s -- OK" % (ma, nsteps, drift, os.path.basename(out))
    )


def check_hll_fidelity() -> None:
    """(6) fidelite HLL au schema MATLAB de reference.

    golden_hll_state.csv est produit par golden/gen/golden_hll_gen.m (Octave sur RIEMOM2D)
    avec le SCHEMA du depot -- vitesses eigenvalues15_2D(M, 1), flux
    Flux_closure15_2D, HLL de Davis pas_HLL, split dimensionnel ADDITIF + Euler
    explicite -- sur le crossing Ma = 2, Np = 64, 20 pas. adc rejoue la MEME
    sequence de dt (golden_hll_dts.csv) avec le modele exact_speeds (memes
    vitesses) en riemann='hll' NON-splite ssprk2 : l'ecart residuel mesure la
    difference de SCHEMA seule. Verifie aussi que adc-HLL est PLUS PROCHE du
    golden HLL que adc-Rusanov (HLL moins diffusif : critere qualitatif de
    l'issue, rendu quantitatif).
    """
    from adc_cases.common.io import case_output_dir
    from adc_cases.common.native import adc_include

    g = os.path.join(os.path.dirname(HERE), "golden")
    meta = np.loadtxt(os.path.join(g, "golden_hll_meta.csv"), delimiter=",")
    n, ma, nsteps = int(meta[0]), float(meta[1]), int(meta[2])
    dts = np.atleast_1d(
        np.loadtxt(os.path.join(g, "golden_hll_dts.csv"), delimiter=",")
    )
    raw = np.loadtxt(os.path.join(g, "golden_hll_state.csv"), delimiter=",")
    gold = np.stack(
        [raw[k * n : (k + 1) * n, :].T for k in range(15)], axis=0
    )  # (15, ny, nx)

    m = build_moment_model(name="hyqmom15_exact", exact_speeds=True)
    compiled = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_exact.so"),
        adc_include(),
        backend="aot",
    )
    U0 = crossing_state(n, ma)
    dist = {}
    for riemann in ("hll", "rusanov"):
        sim = adc.System(n=n, L=1.0, periodic=True)
        sim.add_equation(
            "mom",
            model=compiled,
            spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
            time=adc.Explicit(),
        )
        sim.set_state("mom", U0)
        for dt in dts:
            sim.step(float(dt))
        U = np.array(sim.get_state("mom"))
        assert np.all(np.isfinite(U)), "etat %s non fini" % riemann
        num = float(np.sqrt(np.sum((U - gold) ** 2)))
        den = float(np.sqrt(np.sum(gold**2)))
        dist[riemann] = num / den
    assert dist["hll"] < 0.05, (
        "fidelite HLL : ecart L2 relatif %.3f au golden MATLAB-HLL "
        "(attendu : niveau discretisation, schema split-Euler vs "
        "non-splite ssprk2)" % dist["hll"]
    )
    assert dist["hll"] < dist["rusanov"], (
        "adc-HLL (%.4f) devrait etre plus proche du golden HLL que adc-Rusanov (%.4f)"
        % (dist["hll"], dist["rusanov"])
    )

    # replay en time='euler' : le split dimensionnel ADDITIF + Euler du
    # MATLAB est ALGEBRIQUEMENT le Euler non-splite (Mx + My - M = M + dt(Lx + Ly)), donc en
    # rejouant les dt golden avec le meme flux/vitesses, l'ecart de SCHEMA disparait : il ne
    # reste que l'arrondi (et la convention degeneree |sL-sR| < 1e-10, de mesure nulle).
    # la jambe euler passe par backend='production' : l'ABI du .so AOT fige SSPRK2 (le
    # coeur rejette explicitement euler sur ce chemin), le loader production marshale method.
    compiled_prod = m.compile(
        os.path.join(case_output_dir("hyqmom15"), "hyqmom15_exact_prod.so"),
        adc_include(),
        backend="production",
    )
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation(
        "mom",
        model=compiled_prod,
        spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
        time=adc.Explicit(method="euler"),
    )
    sim.set_state("mom", U0)
    for dt in dts:
        sim.step(float(dt))
    Ue = np.array(sim.get_state("mom"))
    assert np.all(np.isfinite(Ue)), "etat euler non fini"
    dist["euler"] = float(
        np.sqrt(np.sum((Ue - gold) ** 2)) / np.sqrt(np.sum(gold**2))
    )
    assert dist["euler"] < 1e-9, (
        "replay euler : ecart L2 relatif %.3e au golden MATLAB-HLL (attendu ~arrondi : "
        "memes flux, memes vitesses, meme schema -- le 4.4 %% de ssprk2 etait bien le "
        "2e etage)" % dist["euler"]
    )
    print(
        "(6) fidelite HLL (%d pas rejoues, Ma = %g) : ssprk2 = %.4f (difference de schema "
        "seule) ; Rusanov = %.4f > HLL ; time='euler' = %.2e (== schema MATLAB a "
        "l'arrondi : split additif + Euler == Euler non-splite) -- OK"
        % (nsteps, ma, dist["hll"], dist["rusanov"], dist["euler"])
    )


def main() -> None:
    print(
        "=== hyqmom15/run_crossing : sources de moments + croisement de jets (Rusanov) ==="
    )
    check_sources_vs_pdf()
    check_larmor_rotation()
    check_crossing_smoke()
    check_hll_fidelity()
    print("hyqmom15/run_crossing : OK")


if __name__ == "__main__":
    main()
