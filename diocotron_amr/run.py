"""Cas "diocotron_amr" : instabilite diocotron sur grille AMR multi-patch.

Compose GENERIQUEMENT depuis Python via `adc.AmrSystem` (le pendant raffine de
`adc.System`), sans solveur dedie : un bloc `diocotron` porte sur une hierarchie
adaptative (grossier + un niveau fin suivi par regrid Berger-Rigoutsos, reflux
conservatif). Toutes les `regrid_every` iterations, les patchs fins sont re-decoupes
pour suivre la bande de charge. Python compose la config et la CI (numpy), le calcul
reste en C++.

Capacites verifiees (asserts) :
  - VRAI raffinement adaptatif : la bande de charge est taggee et couverte par PLUSIEURS
    patchs fins (n_patches() >= 2 a chaque pas). Le seuil discrimine reellement : un run de
    CONTROLE avec un seuil inatteignable (1e30) ne produit qu'un patch degenere (1), donc
    strictement moins. Le raffinement vient bien du tagging, pas du build de la hierarchie ;
  - EFFET sur la solution : la solution raffinee (projetee sur la grille de base) differe
    mesurablement de celle du run de controle non raffine (ecart sup > seuil) ;
  - conservation de masse sur AMR (reflux) a l'arrondi machine : drel < 1e-9 ;
  - integrite numerique : densite finie partout.

Schema spatial : NoSlope + Rusanov (ordre 1, robuste) pour la bande qui s'enroule sur
grille AMR grossiere. Le couplage Poisson periodique exige une CI a moyenne nulle, d'ou
le fond neutralisant n_i0 = <n_e>.
"""

import numpy as np

import adc

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402
from adc_cases.common.checks import assert_finite, relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402

PI = np.pi

N, L = 64, 1.0
MODE, REFINE_FRAC = 4, 0.15
NSTEPS = 40
TOL_MASS = 1e-9
# Seuil inatteignable : aucune maille ne le depasse, le critere ne tagge donc jamais. Sert de
# run de CONTROLE pour montrer que les patchs fins du run nominal viennent bien du tagging.
NO_REFINE = 1e30
# Le run raffine doit changer la solution projetee d'au moins cet ecart sup ; bien au-dessus du
# bruit (l'ecart mesure vaut ~6e-2), assez bas pour rester robuste aux variations de schema.
MIN_SOLUTION_GAP = 1e-3


def build_sim(ne, n_i0, threshold):
    """Construit un AmrSystem diocotron identique au nominal, au seuil de raffinement pres."""
    sim = adc.AmrSystem(n=N, L=L, regrid_every=10, periodic=True)
    sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0, n_i0=n_i0),
                  spatial=adc.Spatial(none=True))
    sim.set_refinement(threshold=threshold)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)
    return sim


def main():
    amp, width, disp = 1.0, 0.05, 0.02
    ne = band_density(N, L, amp=amp, width=width, mode=MODE, disp=disp)
    n_i0 = float(ne.mean())  # fond neutralisant : Poisson periodique a moyenne nulle

    sim = build_sim(ne, n_i0, threshold=n_i0 + REFINE_FRAC)

    mass0 = sim.mass()
    assert np.isfinite(mass0) and mass0 > 0.0, "masse initiale invalide"

    print("# cas diocotron_amr : instabilite diocotron sur AMR multi-patch (adc.AmrSystem)")
    print("# n_base=%d regrid_every=10 band_mode=%d  n_i0=%.4f" % (N, MODE, n_i0))
    print("# %-5s %-8s %-8s %-13s %-11s" % ("step", "t", "patches", "mass", "drel"))

    patches_seen = set()
    for k in range(NSTEPS):
        sim.step_cfl(0.4)
        mass = sim.mass()
        npatch = sim.n_patches()
        drel = relative_drift(mass, mass0)
        patches_seen.add(npatch)
        # VRAI raffinement : la bande taggee est couverte par PLUSIEURS patchs fins (>= 2),
        # pas seulement un niveau fin present (n_patches() >= 1 etait verifie trivialement).
        assert npatch >= 2, "raffinement insuffisant au pas %d : n_patches()=%d (< 2)" % (k, npatch)
        assert drel < TOL_MASS, "masse non conservee au pas %d : drel=%.3e" % (k, drel)
        assert_finite(sim.density(), "densite au pas %d" % k)
        if k % 10 == 0 or k == NSTEPS - 1:
            print("  %-5d %-8.4f %-8d %-13.8e %-11.3e" % (k, sim.time(), npatch, mass, drel))

    dens = sim.density()
    print("# patchs observes : %s" % sorted(patches_seen))
    print("# masse : init=%.12e final=%.12e drel=%.3e"
          % (mass0, sim.mass(), relative_drift(sim.mass(), mass0)))
    print("# densite : min=%.6e max=%.6e" % (float(dens.min()), float(dens.max())))
    assert min(patches_seen) >= 2, "AMR n'a jamais couvert la bande (< 2 patchs)"

    # --- Run de CONTROLE : meme CI, mais seuil inatteignable -> aucun tagging ---
    # Prouve que les patchs fins du run nominal viennent du critere de raffinement (pas du
    # build de la hierarchie) et que le raffinement CHANGE la solution la ou il agit.
    ctrl = build_sim(ne, n_i0, threshold=NO_REFINE)
    for _ in range(NSTEPS):
        ctrl.step_cfl(0.4)
    npatch_ctrl = ctrl.n_patches()
    dens_ctrl = ctrl.density()
    gap = float(np.abs(dens - dens_ctrl).max())
    print("# controle (seuil %.0e) : patches=%d  ecart_sup solution=%.6e"
          % (NO_REFINE, npatch_ctrl, gap))
    assert min(patches_seen) > npatch_ctrl, (
        "le seuil ne discrimine pas : nominal=%d patchs, controle=%d (devrait etre <)"
        % (min(patches_seen), npatch_ctrl))
    assert gap > MIN_SOLUTION_GAP, (
        "le raffinement ne change pas la solution : ecart sup %.3e <= %.1e"
        % (gap, MIN_SOLUTION_GAP))

    print("OK diocotron_amr")


if __name__ == "__main__":
    main()
