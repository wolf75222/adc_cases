#!/usr/bin/env python3
r"""Balayage ordre x resolution x mode du cas diocotron (mesure, pas de nouvelle physique).

But (Hoffart "PR-0") : quantifier quelle part de l'ecart de taux de croissance diocotron
(numerique vs analytique de Petri) se referme en montant en resolution / en ordre de
reconstruction (= diffusion numerique), et quelle part reste a peu pres plate en resolution
(= verrou structurel du bord d'anneau cartesien). C'est l'entree quantitative pour decider la
PR-A "transport-wall".

Ce script ne redefinit aucune physique ni aucun observable : il reutilise tel quel le pipeline
de `diocotron/run.py` (CI anneau partagee, FFT azimutale du mode l de phi, ajustement de la
phase lineaire `exp(gamma t)`, normalisation par omega_D, cible analytique de Petri en numpy).
Il ne fait que balayer (n, limiteur, l) et reporter gamma_num + %ecart.

Axe ordre : la facade `adc.System.add_block` (chemin du cas diocotron) expose les
reconstructions de la dispatch runtime `make_block` : `none` (ordre 1), `minmod` (ordre 2 TVD),
`vanleer` (ordre 2, moins dissipatif), et depuis adc_cpp #88 (master `ca803dc`) `weno5`
(WENO5-Z, ordre 5, stencil 5 points, 3 ghosts) appariee a SSPRK3 (`adc.Explicit(method="ssprk3")`,
3 etages ordre 3). L'axe ordre balaye ici est donc {O1 none, O2 minmod, O2 vanleer, O5 weno5}.
La cle d'ordre `weno5` aiguille a la fois la reconstruction (`adc.Spatial(limiter="weno5")`) et
l'integrateur temporel (SSPRK3) : O5 = WENO5-Z + SSPRK3, pour borner la diffusion residuelle et
isoler le plancher structurel. Voir SWEEP_RESULTS.md.

Comparabilite des runs : a `nsteps` fixe le pas `dt ~ CFL dx` decroit avec n, donc le temps
physique final decroit avec n. A `nsteps=900` (calage de run.py) n=256 n'atteint que t ~ 35
(phase encore non saturee) et la fenetre d'ajustement (ancree sur 1.3 a0 -> 0.85 pic) sur-lit
gamma. On tient donc le temps physique final ~ constant (`T_END`) en avancant jusqu'a t_end
plutot qu'a un nsteps fixe. T_END = 48 est l'horizon du calage valide de run.py (n=192, 900 pas
-> t ~ 48) : a cet horizon le balayage reproduit le README a n=192 (l=3 -22 %, l=4 -27 %,
l=5 -5 %), ce qui ancre la mesure. C'est un reglage de boucle, pas un nouvel observable.

Usage :
    PYTHONPATH=<adc_cpp>/build-py/python python3 diocotron/sweep.py
    # options : --modes 3,4,5  --ns 128,192,256  --orders minmod,vanleer,weno5  --t-end 48  --quick
"""
import argparse
import math
import os
import sys
import time

import numpy as np

import adc  # notre solveur (facade compilee)

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On reutilise le pipeline du cas (CI, diagnostic, analytique) ; aucun observable reinvente.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as dioc  # noqa: E402  (diocotron/run.py : meme dossier)
from adc_cases import models  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402

# Ordres de reconstruction atteignables via add_block (make_block runtime). La cle 'weno5'
# represente O5 = WENO5-Z (reconstruction) + SSPRK3 (temps) : elle aiguille les deux briques
# (cf. make_system : limiter='weno5' -> time=Explicit(method='ssprk3')).
ORDER_LABEL = {"none": "O1", "minmod": "O2-minmod", "vanleer": "O2-vanleer", "weno5": "O5-weno5"}

# Horizon physique commun. T_END = 48 = horizon du calage valide de run.py (n=192, 900 pas
# -> t ~ 48) ; a cet horizon le balayage reproduit le README a n=192. On vise ce temps physique
# (et non un nsteps fixe) pour que la fenetre lineaire couvre la meme phase de croissance partout.
T_END = 48.0


def make_system(n, l, limiter, delta):
    """Compose le meme systeme que run.py.make_ring_system, mais avec un limiteur parametrable.

    Pour la cle d'ordre 'weno5' (O5), on apparie la reconstruction WENO5-Z a SSPRK3 (3 etages,
    ordre 3) : appairer un ordre eleve en espace a un integrateur d'ordre 1-2 en temps brides
    l'ordre effectif. Les ordres O1/O2 gardent l'integrateur historique SSPRK2 (adc.Explicit()
    par defaut), bit-identique a PR-0.
    """
    time = adc.Explicit(method="ssprk3") if limiter == "weno5" else adc.Explicit()
    sim = adc.System(n=n, L=dioc.L, periodic=False)
    sim.add_block("ne", model=models.diocotron(B0=dioc.B0, alpha=dioc.ALPHA, n_i0=0.0),
                  spatial=adc.Spatial(limiter=limiter, flux="rusanov"), time=time)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                    wall="circle", wall_radius=dioc.RWALL)
    sim.set_density("ne", dioc.ring_density(n, l, delta))
    return sim


def measure_growth(l, n, limiter, t_end=T_END, delta=0.01, cfl=0.4, max_steps=4000):
    """gamma_num normalise (= gamma * 2 pi / rhobar) pour (n, limiteur, l), temps physique ~ t_end.

    Reutilise dioc.mode_l_amplitude (FFT azimutale du mode l de phi) et dioc.fit_linear_phase
    (pente de log(amp) sur la phase lineaire). Marche jusqu'a t_end (au lieu d'un nsteps fixe)
    pour rendre la fenetre d'ajustement comparable entre resolutions.

    Retourne (gamma_norm, win, t_final, nsteps, fit_i0, fit_i1, fit_t0, fit_t1) : les bornes
    de la fenetre de fit (indices et temps) sont remontees pour tracabilite dans le CSV.
    """
    sim = make_system(n, l, limiter, delta)
    rm = 0.5 * (dioc.R0 + dioc.R1)
    ts, amp = [], []
    step = 0
    while True:
        phi = sim.potential()
        if not np.isfinite(phi).all():
            print(f"  [n={n} {limiter} l={l}] NaN au pas {step} -> arret", file=sys.stderr)
            break
        ts.append(sim.time())
        amp.append(dioc.mode_l_amplitude(phi, n, rm, l))
        if sim.time() >= t_end or step >= max_steps:
            break
        sim.step_cfl(cfl)
        step += 1
    ts = np.asarray(ts)
    amp = np.asarray(amp)
    gamma_raw, win = dioc.fit_linear_phase(ts, amp)
    gamma_norm = gamma_raw * 2.0 * math.pi / dioc.RHOBAR
    # win = (t0, t1) = bornes temporelles de la fenetre de fit (dioc.fit_linear_phase). On
    # retrouve les indices correspondants dans ts (premier echantillon a chaque borne) pour
    # tracer la fenetre dans le CSV (i0, i1, t0, t1) : c'est ce qui rend le point n=192 O5 l=4
    # verifiable (fenetre ouverte trop tot -> transitoire). Aucun observable nouveau.
    fit_t0, fit_t1 = float(win[0]), float(win[1])
    fit_i0 = int(np.searchsorted(ts, fit_t0)) if len(ts) else 0
    fit_i1 = int(np.searchsorted(ts, fit_t1)) if len(ts) else 0
    fit_i1 = min(fit_i1, len(ts) - 1) if len(ts) else 0
    return (gamma_norm, win, (ts[-1] if len(ts) else 0.0), len(ts),
            fit_i0, fit_i1, fit_t0, fit_t1)


def parse_args():
    p = argparse.ArgumentParser(description="Balayage diocotron ordre x resolution x mode (mesure).")
    p.add_argument("--modes", default="3,4,5", help="modes azimutaux l (defaut 3,4,5).")
    p.add_argument("--ns", default="128,192,256", help="resolutions n (defaut 128,192,256).")
    p.add_argument("--orders", default="minmod,vanleer,weno5",
                   help="ordres atteignables : none,minmod,vanleer,weno5 (defaut minmod,vanleer,weno5 ; "
                        "weno5 = WENO5-Z + SSPRK3).")
    p.add_argument("--t-end", type=float, default=T_END, help="temps physique final commun.")
    p.add_argument("--quick", action="store_true",
                   help="grille reduite (n=128,192 ; minmod ; modes 3,4,5) pour fumee rapide.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        ns_list = [128, 192]
        orders = ["minmod"]
        modes = [3, 4, 5]
    else:
        ns_list = [int(x) for x in args.ns.split(",") if x.strip()]
        orders = [x.strip() for x in args.orders.split(",") if x.strip()]
        modes = [int(x) for x in args.modes.split(",") if x.strip()]

    for o in orders:
        if o not in ORDER_LABEL:
            raise SystemExit(f"ordre '{o}' non atteignable via add_block ; choisir parmi "
                             f"{list(ORDER_LABEL)} (weno5 = WENO5-Z + SSPRK3, cf. SWEEP_RESULTS.md).")

    out = case_output_dir("diocotron")
    csv_path = os.path.join(out, "sweep_results.csv")

    # Cible analytique de Petri (numpy, cote run.py) : invariante en n, par mode.
    gamma_ana = {l: dioc.diocotron_eigenvalue(l).imag for l in modes}

    print("=" * 78)
    print("Balayage diocotron : ordre x resolution x mode (mesure, PR-0 Hoffart)")
    print(f"axe ordre atteignable via add_block : {[ORDER_LABEL[o] for o in orders]}")
    print(f"  (O5 = WENO5-Z + SSPRK3, atteignable depuis adc_cpp #88 / master ca803dc)")
    print("=" * 78)
    print(f"{'n':>5} {'ordre':>11} {'l':>3} {'gamma_num':>10} {'gamma_ana':>10} "
          f"{'%err':>7} {'t_final':>8} {'pas':>6}  {'[fenetre de fit]'}")

    rows = []
    t0 = time.time()
    for o in orders:
        for n in ns_list:
            for l in modes:
                (g, win, tf, nstep,
                 fit_i0, fit_i1, fit_t0, fit_t1) = measure_growth(l, n, o, t_end=args.t_end)
                ga = gamma_ana[l]
                err = 100.0 * (g - ga) / ga
                print(f"{n:>5} {ORDER_LABEL[o]:>11} {l:>3} {g:>10.4f} {ga:>10.4f} "
                      f"{err:>+6.1f}% {tf:>8.1f} {nstep:>6}  "
                      f"[fit i{fit_i0}..{fit_i1} t{fit_t0:.1f}..{fit_t1:.1f}]")
                rows.append((n, o, ORDER_LABEL[o], l, g, ga, err, tf, nstep,
                             fit_i0, fit_i1, fit_t0, fit_t1))

    with open(csv_path, "w") as f:
        # Colonnes de fenetre de fit (fit_i0/i1 = indices, fit_t0/t1 = temps) ecrites pour
        # chaque ligne : c'est la tracabilite demandee a la revue (le point n=192 O5 l=4
        # s'explique par une fenetre ouverte trop tot, lisible directement dans le CSV).
        f.write("n,order,order_label,l,gamma_num,gamma_ana,err_pct,t_final,nsteps,"
                "fit_i0,fit_i1,fit_t0,fit_t1\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f},{r[6]:.3f},"
                    f"{r[7]:.3f},{r[8]},{r[9]},{r[10]},{r[11]:.3f},{r[12]:.3f}\n")

    print("-" * 78)
    print(f"{len(rows)} runs en {time.time() - t0:.0f}s -> {csv_path}")
    print("OK sweep diocotron")


if __name__ == "__main__":
    main()
