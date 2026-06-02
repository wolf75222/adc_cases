"""Cas "diocotron_amr" : instabilite diocotron sur grille AMR multi-patch.

Compose GENERIQUEMENT depuis Python via `adc.AmrSystem` (le pendant raffine de
`adc.System`), sans solveur dedie : un bloc `diocotron` porte sur une hierarchie
adaptative (grossier + un niveau fin suivi par regrid Berger-Rigoutsos, reflux
conservatif). Toutes les `regrid_every` iterations, les patchs fins sont re-decoupes
pour suivre la bande de charge. Python compose la config et la CI (numpy), le calcul
reste en C++.

Capacites verifiees :
  - raffinement adaptatif actif : n_patches() >= 1 a chaque pas ;
  - conservation de masse sur AMR (reflux) a l'arrondi machine : drel < 1e-9 ;
  - integrite numerique : densite finie partout.

Schema spatial : NoSlope + Rusanov (ordre 1, robuste) pour la bande qui s'enroule sur
grille AMR grossiere. Le couplage Poisson periodique exige une CI a moyenne nulle, d'ou
le fond neutralisant n_i0 = <n_e>.
"""

import numpy as np

import adc

PI = np.pi


def band_density(n, L, amp, width, mode, disp):
    """CI en bande : un ruban gaussien horizontal perturbe en x (mode azimutal `mode`)."""
    coord = (np.arange(n) + 0.5) / n * L
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    y0 = 0.5 * L + disp * np.cos(2 * PI * mode * xx / L)
    return 1.0 + amp * np.exp(-((yy - y0) ** 2) / (width ** 2))


def main():
    n, L = 64, 1.0
    amp, width, mode, disp, refine_frac = 1.0, 0.05, 4, 0.02, 0.15
    ne = band_density(n, L, amp, width, mode, disp)
    n_i0 = float(ne.mean())  # fond neutralisant : Poisson periodique a moyenne nulle

    sim = adc.AmrSystem(n=n, L=L, B0=1.0, alpha=1.0, n_i0=n_i0, regrid_every=10,
                        periodic=True)
    sim.add_block("ne", model="diocotron", charge=1.0, spatial=adc.Spatial(none=True))
    sim.set_refinement(threshold=n_i0 + refine_frac)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", ne)

    mass0 = sim.mass()
    assert np.isfinite(mass0) and mass0 > 0.0, "masse initiale invalide"

    print("# cas diocotron_amr : instabilite diocotron sur AMR multi-patch (adc.AmrSystem)")
    print("# n_base=%d regrid_every=10 band_mode=%d  n_i0=%.4f" % (n, mode, n_i0))
    print("# %-5s %-8s %-8s %-13s %-11s" % ("step", "t", "patches", "mass", "drel"))

    tol_mass = 1e-9
    patches_seen = set()
    for k in range(40):
        sim.step_cfl(0.4)
        mass = sim.mass()
        npatch = sim.n_patches()
        drel = abs(mass - mass0) / abs(mass0)
        patches_seen.add(npatch)
        assert npatch >= 1, "AMR inactif (n_patches() < 1)"
        assert drel < tol_mass, "masse non conservee au pas %d : drel=%.3e" % (k, drel)
        assert np.isfinite(sim.density()).all(), "densite non finie au pas %d" % k
        if k % 10 == 0 or k == 39:
            print("  %-5d %-8.4f %-8d %-13.8e %-11.3e" % (k, sim.time(), npatch, mass, drel))

    dens = sim.density()
    print("# patchs observes : %s" % sorted(patches_seen))
    print("# masse : init=%.12e final=%.12e drel=%.3e"
          % (mass0, sim.mass(), abs(sim.mass() - mass0) / abs(mass0)))
    print("# densite : min=%.6e max=%.6e" % (float(dens.min()), float(dens.max())))
    assert min(patches_seen) >= 1, "AMR n'a jamais ete actif"
    print("OK diocotron_amr")


if __name__ == "__main__":
    main()
