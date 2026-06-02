#!/usr/bin/env python3
"""Demo "two_fluid_ap" : modele bi-fluide isotherme en regime RAIDE (asymptotic-preserving).

On pilote depuis Python la facade compilee `adc.TwoFluidAP` (toute la physique est en
C++), un solveur SPECIALISE (integrateur sur mesure, non composable bloc-a-bloc comme
`adc.System`). Il integre deux fluides charges (electrons + ions) couples au champ
electrique par la contrainte de quasi-neutralite. La frequence plasma omega_pe fixe
l'echelle de temps RAIDE du systeme : un schema explicite serait limite par
dt * omega_pe < O(1), donc exploserait des qu'on prend un grand pas de temps.

Le solveur emploie un traitement IMEX / asymptotic-preserving (AP) : le terme raide
est integre de maniere implicite, ce qui rend le schema STABLE et CONSISTANT meme
quand dt * omega_pe >> 1. On le demontre sur deux scenarios.

  Run 1 "raide" (non magnetise) :
      omega_pe = 1e3, omega_pi = 20, advance(5.0/1e3, 200)
      => dt * omega_pe = 5 (un schema explicite EXPLOSERAIT).
      Invariants verifies :
        - l'ecart a la quasi-neutralite max_dev() reste PETIT  (< 0.1) ;
        - la charge nette locale max_charge() reste PETITE      (< 0.1) ;
        - la masse electronique mass_e est conservee            (erreur relative < 1e-7).

  Run 2 "magnetise" :
      omega_ce = 4, omega_ci = 0.2, advance(0.01, 100)
      => terme de rotation cyclotron actif ; on verifie la conservation de la masse
         electronique (erreur relative < 1e-7).

Conclusion : le schema IMEX / AP reste stable et conservatif pour un plasma raide,
magnetise ou non, la ou un schema explicite serait inutilisable.

Sortie : diagnostics numeriques imprimes, puis la ligne finale "OK two_fluid_ap".
Dependances : numpy + adc uniquement.
"""

import numpy as np

import adc


def _rel_err(a, b):
    """Erreur relative robuste (denominateur protege contre le zero)."""
    return abs(a - b) / max(abs(b), 1e-30)


def run_stiff():
    """Run 1 : regime raide non magnetise, dt * omega_pe = 5."""
    cfg = adc.TwoFluidAPConfig()
    cfg.n = 64
    cfg.omega_pe = 1.0e3   # frequence plasma electronique : echelle de temps RAIDE
    cfg.omega_pi = 20.0    # frequence plasma ionique

    solver = adc.TwoFluidAP(cfg)

    dt = 5.0 / 1.0e3                 # dt choisi tel que dt * omega_pe = 5
    stiffness = dt * cfg.omega_pe    # nombre de raideur (>> 1 ici)
    mass_e0 = solver.mass_e()

    solver.advance(dt, 200)

    max_dev = solver.max_dev()       # ecart max a la quasi-neutralite
    max_charge = solver.max_charge() # charge nette locale max
    mass_e = solver.mass_e()
    mass_rel = _rel_err(mass_e, mass_e0)

    print("[run 1 - raide, non magnetise]")
    print("  n=%d  omega_pe=%.3e  omega_pi=%.3e" % (cfg.n, cfg.omega_pe, cfg.omega_pi))
    print("  dt=%.3e  nsteps=200  dt*omega_pe=%.1f  (explicite EXPLOSERAIT)" % (dt, stiffness))
    print("  max_dev()    = %.6e   (ecart a la quasi-neutralite)" % max_dev)
    print("  max_charge() = %.6e   (charge nette locale)" % max_charge)
    print("  mass_e: %.6e -> %.6e   (err. relative %.3e)" % (mass_e0, mass_e, mass_rel))

    # --- Invariants physiques (propriete AP) ---
    # Le grand pas de temps a bien ete fait sans exploser : valeurs finies.
    assert np.isfinite(max_dev), "max_dev non fini : le schema a explose"
    assert np.isfinite(max_charge), "max_charge non fini : le schema a explose"
    assert np.isfinite(mass_e), "mass_e non finie : le schema a explose"
    # Propriete AP : la quasi-neutralite est maintenue malgre dt*omega_pe = 5.
    assert max_dev < 0.1, "max_dev trop grand (%.3e) : quasi-neutralite non maintenue" % max_dev
    assert max_charge < 0.1, "max_charge trop grand (%.3e)" % max_charge
    # Conservation de la masse electronique.
    assert mass_rel < 1e-7, "masse electronique non conservee (err. rel. %.3e)" % mass_rel

    return max_dev, max_charge, mass_rel


def run_magnetized():
    """Run 2 : plasma raide magnetise (rotation cyclotron active)."""
    cfg = adc.TwoFluidAPConfig()
    cfg.n = 64
    cfg.omega_ce = 4.0   # frequence cyclotron electronique
    cfg.omega_ci = 0.2   # frequence cyclotron ionique

    solver = adc.TwoFluidAP(cfg)

    dt = 0.01
    mass_e0 = solver.mass_e()

    solver.advance(dt, 100)

    max_dev = solver.max_dev()
    max_charge = solver.max_charge()
    mass_e = solver.mass_e()
    mass_rel = _rel_err(mass_e, mass_e0)

    print("[run 2 - raide magnetise]")
    print("  n=%d  omega_ce=%.3e  omega_ci=%.3e" % (cfg.n, cfg.omega_ce, cfg.omega_ci))
    print("  dt=%.3e  nsteps=100" % dt)
    print("  max_dev()    = %.6e" % max_dev)
    print("  max_charge() = %.6e" % max_charge)
    print("  mass_e: %.6e -> %.6e   (err. relative %.3e)" % (mass_e0, mass_e, mass_rel))

    # --- Invariants physiques ---
    assert np.isfinite(max_dev), "max_dev non fini : le schema a explose"
    assert np.isfinite(mass_e), "mass_e non finie : le schema a explose"
    # Conservation de la masse electronique sous rotation cyclotron.
    assert mass_rel < 1e-7, "masse electronique non conservee (err. rel. %.3e)" % mass_rel

    return max_dev, max_charge, mass_rel


def main():
    print("=== Demo two_fluid_ap : bi-fluide isotherme raide (asymptotic-preserving) ===")
    run_stiff()
    run_magnetized()
    print("Conclusion : schema IMEX / asymptotic-preserving stable et conservatif")
    print("pour un plasma raide, magnetise ou non (un schema explicite echouerait).")
    print("OK two_fluid_ap")


if __name__ == "__main__":
    main()
