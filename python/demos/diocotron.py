"""Demo "diocotron" : instabilite diocotron (derive E x B).

Capacite demontree : transport hyperbolique d'une densite de charge couple a la
resolution de Poisson sur une grille uniforme, le tout calcule en C++ compile.
Python ne fait que composer la configuration et diagnostiquer le resultat.

Physique : un plasma non neutre confine par un champ magnetique B0 derive a la
vitesse E x B. Une bande de densite perturbee (cfg.ic = Band) developpe
l'instabilite diocotron : une petite perturbation periodique le long de x croit
exponentiellement, signature de l'instabilite de cisaillement de la derive.

Le solveur (DiocotronSolver) avance la densite par transport hyperbolique et
recalcule le potentiel via Poisson a chaque sous-pas. Comme le transport est
purement advectif (champ de vitesse a divergence nulle), la masse totale doit
etre conservee exactement.

Diagnostics imprimes : a chaque k pas, le temps t, l'amplitude de la perturbation
(norme L2 de la deviation de la densite par rapport a sa moyenne en x, c.-a-d. la
partie non axisymetrique du mode) et la masse totale.

Invariants verifies par assert :
  - conservation de la masse : |mass - mass0| / mass0 < 1e-6 a chaque pas ;
  - croissance de l'instabilite : amplitude finale > amplitude initiale.
"""

import numpy as np
import adc


def perturbation_amplitude(density):
    """Amplitude L2 de la perturbation = deviation par rapport a la moyenne en x.

    La bande non perturbee est uniforme le long de x (axis=1) et structuree le
    long de y (axis=0). La moyenne par ligne (sur x) reconstruit donc le profil
    de base ; ce qui reste est la perturbation portant l'instabilite.
    """
    base = density.mean(axis=1, keepdims=True)  # profil moyen en x, par ligne
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


def main():
    # --- Configuration : condition initiale en bande, grille 96 x 96 ---
    cfg = adc.DiocotronConfig()
    cfg.n = 96
    cfg.ic = adc.DiocotronIC.Band
    # On garde les autres parametres par defaut (B0, alpha, band_*, eps, ...).

    solver = adc.DiocotronSolver(cfg)

    # Masse de reference et vitesse de derive initiale.
    mass0 = solver.mass()
    amp0 = perturbation_amplitude(solver.density())
    drift0 = solver.max_drift_speed()

    print("=== Demo diocotron : instabilite de derive E x B ===")
    print(f"grille n = {solver.nx()} x {solver.nx()}, dx = {solver.dx():.6e}")
    print(f"max_drift_speed initial = {drift0:.6e}")
    print(f"masse initiale          = {mass0:.6e}")
    print(f"amplitude initiale      = {amp0:.6e}")
    print()
    print(f"{'pas':>4} {'t':>12} {'amplitude':>14} {'mass':>16}")

    # --- Boucle d'integration : ~120 pas, CFL = 0.4 ---
    n_steps = 120
    k = 10            # frequence d'impression des diagnostics
    rel_mass_tol = 1e-6

    amp_last = amp0
    for step in range(1, n_steps + 1):
        solver.step_cfl(0.4)

        # Invariant physique : la masse totale est conservee (transport advectif).
        mass = solver.mass()
        rel_mass = abs(mass - mass0) / abs(mass0)
        assert rel_mass < rel_mass_tol, (
            f"masse non conservee au pas {step} : ecart relatif {rel_mass:.3e}"
        )

        amp_last = perturbation_amplitude(solver.density())

        if step % k == 0 or step == n_steps:
            print(f"{step:4d} {solver.time():12.6f} {amp_last:14.6e} {mass:16.8e}")

    print()
    print(f"amplitude finale = {amp_last:.6e}  (initiale {amp0:.6e})")
    print(f"facteur de croissance = {amp_last / amp0:.4f}")

    # --- Invariant : l'instabilite a fait croitre la perturbation ---
    assert amp_last > amp0, (
        "l'amplitude n'a pas cru : instabilite diocotron non observee"
    )

    print("OK diocotron")


if __name__ == "__main__":
    main()
