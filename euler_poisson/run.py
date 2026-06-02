"""Demo "euler_poisson" : fluide d'Euler couple a Poisson, attractif vs repulsif.

Capacite demontree
------------------
On compose un systeme generique `adc.System` avec UN bloc de modele
"euler_poisson" : il integre les equations d'Euler compressibles (densite,
quantite de mouvement, energie) couplees a une equation de Poisson pour le champ
de force auto-consistant. Toute la physique est en C++ ; Python ne fait que
composer le systeme, fixer la condition initiale, avancer en temps et
diagnostiquer.

On effectue DEUX runs identiques (n=64) ne differant que par le SIGNE du
couplage, passe au modele compose ``models.euler_poisson(sign=...)`` :
  1) sign = +1.0  -> force ATTRACTIVE (auto-gravite)
  2) sign = -1.0  -> force REPULSIVE (charge d'espace, plasma)

Condition initiale commune : densite au repos faiblement perturbee par un cosinus
  rho = rho0 * (1 + eps*cos(2*pi*x/L)),  eps = 0.01
set_density fixe rho sur la composante 0 et l'energie E = rho/(gamma-1) au repos
(quantite de mouvement nulle).

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

Etat : `adc.System.get_state("gas")` renvoie un tableau numpy de forme
(4, n, n) = [rho, rho*u, rho*v, E]. Il n'y a PAS de energy()/total_momentum()
sur System : on lit ces diagnostics directement sur cet etat
(p_x = U[1].sum(), p_y = U[2].sum(), E_tot = U[3].sum()).
"""

import os
import sys

import numpy as np
import adc

# La composition de modeles nommee vit cote application (adc_cases/models.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models

# Tolerances physiques.
TOL_MASS = 1e-9   # derive relative de masse admissible
TOL_MOM = 1e-8    # impulsion nette admissible (doit rester nulle)

# Parametres d'integration : petites tailles et peu de pas pour rester rapide.
N = 64
L = 1.0
GAMMA = 1.4
EPS = 0.01        # amplitude de la perturbation cosinus de densite
RHO0 = 1.0
DT = 0.004
NSTEPS = 20


def initial_density():
    """Densite initiale : repos faiblement perturbe par un cosinus selon x."""
    x = (np.arange(N) + 0.5) * L / N
    xx, _ = np.meshgrid(x, x, indexing="ij")
    return RHO0 * (1.0 + EPS * np.cos(2.0 * np.pi * xx / L))


def energy_and_momentum(sim):
    """Diagnostics globaux lus sur l'etat complet (4, n, n) = [rho, rho*u, rho*v, E]."""
    U = sim.get_state("gas")
    return U[3].sum(), U[1].sum(), U[2].sum()  # E_tot, p_x, p_y


def run_case(sign, label):
    """Avance un run Euler-Poisson et renvoie un dictionnaire de diagnostics.

    sign = +1.0 -> GRAVITE (attractif) ; sign = -1.0 -> PLASMA (repulsif).
    """
    sim = adc.System(n=N, L=L, periodic=True)
    sim.add_block("gas",
                  model=models.euler_poisson(sign=sign, gamma=GAMMA,
                                              four_pi_G=1.0, rho0=RHO0),
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("gas", initial_density())

    # Etat initial : on memorise la masse de reference pour la conservation.
    mass0 = sim.mass("gas")
    energy0, _, _ = energy_and_momentum(sim)

    print(f"[{label}] etat initial : "
          f"mass={mass0:.12e}  energy={energy0:.12e}")
    print(f"[{label}] pas  {'mass':>20s} {'energy':>20s} "
          f"{'p_x':>14s} {'p_y':>14s}")

    max_rel_mass = 0.0   # plus grande derive relative de masse rencontree
    max_mom = 0.0        # plus grande impulsion (toutes directions, tous pas)

    for step in range(1, NSTEPS + 1):
        sim.advance(DT, 1)
        m = sim.mass("gas")
        e, px, py = energy_and_momentum(sim)

        rel_mass = abs(m - mass0) / abs(mass0)
        max_rel_mass = max(max_rel_mass, rel_mass)
        max_mom = max(max_mom, abs(px), abs(py))

        # On imprime quelques pas representatifs pour garder la sortie lisible.
        if step % 5 == 0 or step == 1:
            print(f"[{label}] {step:3d}  {m:20.12e} {e:20.12e} "
                  f"{px:14.3e} {py:14.3e}")

    energy_final, _, _ = energy_and_momentum(sim)
    return {
        "label": label,
        "mass0": mass0,
        "energy0": energy0,
        "energy_final": energy_final,
        "max_rel_mass": max_rel_mass,
        "max_mom": max_mom,
        "time": sim.time(),
    }


def main():
    # Deux runs : seul le signe du couplage (sign du modele) change.
    grav = run_case(+1.0, "GRAVITE")
    print()
    plas = run_case(-1.0, "PLASMA ")
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
