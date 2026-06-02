"""Demo "diocotron_amr" : instabilite diocotron sur grille AMR multi-patch.

Cette demo pilote depuis Python la facade compilee `adc.DiocotronAmr` (toute la physique
est en C++), solveur SPECIALISE sur AMR. On instancie le solveur AMR qui simule la meme instabilite
diocotron qu'en mono-grille, mais sur une hierarchie de patchs raffines dynamiquement :
toutes les `regrid_every` iterations, le maillage est re-decoupe pour suivre les zones
ou la structure (gradients de densite) est la plus fine. Python ne fait que composer la
configuration, avancer en temps via `step_cfl` (pas adaptatif sous contrainte CFL) et
diagnostiquer.

Capacites demontrees :
  - Le raffinement adaptatif est actif et suit la structure : `n_patches() >= 1` a chaque
    instant (le nombre de patchs peut varier au cours du regrid dynamique).
  - La conservation de masse sur AMR : meme avec un maillage qui change, le reflux
    (flux correction) inter-patchs est conservatif a l'arrondi machine. On verifie
    |mass(t) - mass(0)| / |mass(0)| < 1e-9 a chaque pas.
  - L'integrite numerique : la densite reste finie partout (np.isfinite(...).all()).
"""

import numpy as np
import adc


def main():
    # --- Configuration AMR : petite grille de base + regrid frequent ---------
    cfg = adc.DiocotronAmrConfig()
    cfg.n = 64                # grille de base 64x64 (petit -> rapide)
    cfg.L = 1.0               # domaine carre [0,L]^2
    cfg.B0 = 1.0              # champ magnetique de fond (derive E x B)
    cfg.alpha = 1.0           # facteur de couplage Poisson
    cfg.band_mode = 4         # nombre azimutal de la perturbation initiale
    cfg.band_amp = 1.0        # amplitude de la bande de charge
    cfg.band_width = 0.05     # epaisseur de la bande
    cfg.band_disp = 0.02      # perturbation initiale qui amorce l'instabilite
    cfg.refine_frac = 0.15    # fraction de cellules marquees pour raffinement
    cfg.regrid_every = 10     # re-decoupage de la hierarchie tous les 10 pas

    solver = adc.DiocotronAmr(cfg)

    # Masse de reference a t=0 : invariant a conserver sur tout le run.
    mass0 = solver.mass()
    assert np.isfinite(mass0) and mass0 > 0.0, "masse initiale invalide"

    n_steps = 40              # ~40 pas -> couvre plusieurs cycles de regrid
    cfl = 0.4                 # pas de temps adaptatif sous contrainte CFL

    # Tolerance machine : le reflux AMR est conservatif a l'arrondi pres.
    tol_mass = 1e-9

    print("# demo diocotron_amr : instabilite diocotron sur AMR multi-patch")
    print("# n_base=%d  regrid_every=%d  band_mode=%d  cfl=%.2f"
          % (cfg.n, cfg.regrid_every, cfg.band_mode, cfl))
    print("# %-6s %-8s %-9s %-14s %-12s" % ("step", "t", "patches", "mass", "drel"))

    patches_seen = set()
    for k in range(n_steps):
        solver.step_cfl(cfl)

        mass = solver.mass()
        npatch = solver.n_patches()
        drel = abs(mass - mass0) / abs(mass0)
        patches_seen.add(npatch)

        # --- Invariants physiques verifies a CHAQUE pas ----------------------
        # 1) Le raffinement adaptatif est toujours actif (au moins un patch).
        assert npatch >= 1, "n_patches() doit valoir au moins 1 (AMR inactif ?)"
        # 2) Conservation de masse sur AMR a l'arrondi machine.
        assert drel < tol_mass, (
            "masse non conservee au pas %d : drel=%.3e >= %.1e"
            % (k, drel, tol_mass))
        # 3) Integrite : densite finie partout (pas de NaN/Inf).
        assert np.isfinite(solver.density()).all(), \
            "densite non finie au pas %d" % k

        # Affichage periodique (tous les regrid_every pas + dernier pas).
        if k % cfg.regrid_every == 0 or k == n_steps - 1:
            print("  %-6d %-8.4f %-9d %-14.8e %-12.3e"
                  % (k, solver.time(), npatch, mass, drel))

    # --- Bilan final ---------------------------------------------------------
    mass_f = solver.mass()
    drel_f = abs(mass_f - mass0) / abs(mass0)
    dens = solver.density()

    print("# patchs observes au cours du run : %s"
          % sorted(patches_seen))
    print("# masse  : init=%.12e  final=%.12e  drel=%.3e"
          % (mass0, mass_f, drel_f))
    print("# densite: min=%.6e  max=%.6e  finite=%s"
          % (float(dens.min()), float(dens.max()),
             bool(np.isfinite(dens).all())))

    # Garde-fous finaux (redondants mais explicites pour la demo).
    assert drel_f < tol_mass, "masse finale non conservee"
    assert np.isfinite(dens).all(), "densite finale non finie"
    assert min(patches_seen) >= 1, "AMR n'a jamais ete actif"

    print("OK diocotron_amr")


if __name__ == "__main__":
    main()
