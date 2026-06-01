"""Demo "euler_poisson" : fluide d'Euler couple a Poisson, attractif vs repulsif.

Capacite demontree
------------------
Le solveur compile `adc.EulerPoissonSolver` integre les equations d'Euler
compressibles (densite, quantite de mouvement, energie) couplees a une equation
de Poisson pour le champ de force auto-consistant. Toute la physique est en C++ ;
Python ne fait que composer la config, avancer en temps et diagnostiquer.

On effectue DEUX runs identiques (n=64) ne differant que par le signe de
l'interaction :
  1) adc.InteractionKind.Gravity  -> force ATTRACTIVE (auto-gravite)
  2) adc.InteractionKind.Plasma   -> force REPULSIVE (charge d'espace)

Invariants physiques verifies (par assert)
-------------------------------------------
  * Conservation de la masse : la masse totale est un invariant exact du schema
    conservatif. On exige une derive relative < 1e-9.
  * Quantite de mouvement totale nulle : sur un domaine periodique et homogene,
    la force de Poisson derive d'un potentiel et sa somme spatiale est nulle ;
    elle ne peut donc creer aucune impulsion nette. On exige |p_x|, |p_y| < 1e-8.

Le contraste gravite vs plasma se lit sur l'energie : la force attractive et la
force repulsive agissent en sens opposes, ce qui se traduit par une derive
d'energie de signes opposes entre les deux runs (diagnostic, non asserte).
"""

import numpy as np
import adc

# Tolerances physiques.
TOL_MASS = 1e-9   # derive relative de masse admissible
TOL_MOM = 1e-8    # impulsion nette admissible (doit rester nulle)

# Parametres d'integration : petites tailles et peu de pas pour rester rapide.
N = 64
DT = 0.004
NSTEPS = 20


def run_case(kind, label):
    """Avance un run Euler-Poisson et renvoie un dictionnaire de diagnostics."""
    cfg = adc.EulerPoissonConfig()
    cfg.n = N
    cfg.interaction = kind
    solver = adc.EulerPoissonSolver(cfg)

    # Etat initial : on memorise la masse de reference pour la conservation.
    mass0 = solver.mass()
    energy0 = solver.energy()

    print(f"[{label}] etat initial : "
          f"mass={mass0:.12e}  energy={energy0:.12e}")
    print(f"[{label}] pas  {'mass':>20s} {'energy':>20s} "
          f"{'p_x':>14s} {'p_y':>14s}")

    max_rel_mass = 0.0   # plus grande derive relative de masse rencontree
    max_mom = 0.0        # plus grande impulsion (toutes directions, tous pas)

    for step in range(1, NSTEPS + 1):
        solver.step(DT)
        m = solver.mass()
        e = solver.energy()
        px = solver.total_momentum(0)
        py = solver.total_momentum(1)

        rel_mass = abs(m - mass0) / abs(mass0)
        max_rel_mass = max(max_rel_mass, rel_mass)
        max_mom = max(max_mom, abs(px), abs(py))

        # On imprime quelques pas representatifs pour garder la sortie lisible.
        if step % 5 == 0 or step == 1:
            print(f"[{label}] {step:3d}  {m:20.12e} {e:20.12e} "
                  f"{px:14.3e} {py:14.3e}")

    return {
        "label": label,
        "mass0": mass0,
        "energy0": energy0,
        "energy_final": solver.energy(),
        "max_rel_mass": max_rel_mass,
        "max_mom": max_mom,
        "time": solver.time(),
    }


def main():
    # Deux runs : seul le signe de l'interaction change.
    grav = run_case(adc.InteractionKind.Gravity, "GRAVITE")
    print()
    plas = run_case(adc.InteractionKind.Plasma, "PLASMA ")
    print()

    # --- Verification des invariants physiques ---
    for res in (grav, plas):
        # Conservation de la masse (schema conservatif).
        assert res["max_rel_mass"] < TOL_MASS, (
            f"{res['label']} : derive de masse {res['max_rel_mass']:.3e} "
            f">= {TOL_MASS:.1e}")
        # Quantite de mouvement totale nulle : la force de Poisson, derivant
        # d'un potentiel sur domaine periodique, n'injecte aucune impulsion.
        assert res["max_mom"] < TOL_MOM, (
            f"{res['label']} : impulsion nette {res['max_mom']:.3e} "
            f">= {TOL_MOM:.1e}")

    # --- Contraste gravite (attractif) vs plasma (repulsif) ---
    dE_grav = grav["energy_final"] - grav["energy0"]
    dE_plas = plas["energy_final"] - plas["energy0"]
    print("Bilan des invariants (sur les 20 pas) :")
    print(f"  GRAVITE : max derive masse relative = {grav['max_rel_mass']:.3e} "
          f"(< {TOL_MASS:.0e})   max |p| = {grav['max_mom']:.3e} "
          f"(< {TOL_MOM:.0e})")
    print(f"  PLASMA  : max derive masse relative = {plas['max_rel_mass']:.3e} "
          f"(< {TOL_MASS:.0e})   max |p| = {plas['max_mom']:.3e} "
          f"(< {TOL_MOM:.0e})")
    print("Contraste energetique (attractif vs repulsif) :")
    print(f"  dE GRAVITE = {dE_grav:+.6e}   dE PLASMA = {dE_plas:+.6e}")
    print(f"  -> les deux forces agissent en sens opposes : "
          f"signes de dE {'opposes' if dE_grav * dE_plas < 0 else 'identiques'}")

    print("OK euler_poisson")


if __name__ == "__main__":
    main()
