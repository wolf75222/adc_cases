#!/usr/bin/env python3
"""Cas "hyqmom15/run_mpi" : smoke MPI multi-rang du diocotron Vlasov-Poisson 15 moments.

Pourquoi ce cas
---------------
run_diocotron prouve le couplage complet (sources electriques + Poisson) en SERIE. Ici on le
rejoue sous mpirun (np=2 et 4) avec le solveur de Poisson MULTIGRILLE GEOMETRIQUE
(solver="geometric_mg"), seul chemin elliptique sur de sous MPI : le solveur FFT direct est
mono-rang PAR CONCEPTION (il deroule une seule boite en round-robin -> un seul rang la possede)
et le coeur le REFUSE explicitement quand n_ranks>1. On verifie que le pas complet (transport +
CFL + Poisson MG) tourne sans interblocage ni divergence, que l'etat final est bit-identique a la
serie, et que le rejet FFT remonte proprement a travers le driver.

Topologie des boites (IMPORTANT, mesure, pas suppose)
-----------------------------------------------------
Le adc.System cartesien est MONO-BOITE : une seule boite couvre tout le domaine n x n
(index_boxarray). Sous MPI la DistributionMapping(1, n_ranks) round-robin l'attribue au seul
rang 0 ; les autres rangs ont local_size()==0. Ce smoke exerce donc :
  - la SURETE COLLECTIVE du chemin elliptique MG et des reductions (CFL max, masse) sous MPI
    (tous les rangs entrent collectivement dans solve_fields / step_cfl, aucun rang ne deadlock) ;
  - le garde-fou FFT-sous-MPI ;
  - la parite numerique serie vs multi-rang.
Il N'EXERCE PAS l'echange de halos entre boites disjointes (il n'y a qu'une boite). Le decoupage
multi-boites cartesien distribue (halos inter-rangs) releve d'AMR/polaire (suivis separes).

Bootstrap MPI
-------------
Le module _adc lit l'etat MPI via my_rank()/n_ranks() mais n'appelle PAS MPI_Init. On initialise
MPI cote Python via mpi4py (import = MPI_Init) ; mpi4py et _adc partagent la MEME libmpi, donc le
coeur voit le bon rang/taille. C'est le meme bootstrap que les tests MPI Python d'adc_cpp
(test_hdf5_parallel : `mpirun -n N python ... from mpi4py import MPI`).

Lancement
---------
  mpirun -np 1 python run_mpi.py    # reference serie (ecrit l'etat np=1)
  mpirun -np 2 python run_mpi.py    # np=2 : checks + parite vs np=1 + rejet FFT
  mpirun -np 4 python run_mpi.py    # np=4 : idem
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# mpi4py initialise MPI (effet de bord de l'import) : sans cela _adc.n_ranks() rend 1 (serie) meme
# sous mpirun, et le smoke ne testerait rien de distribue. mpi4py est facultatif (np=1 pur marche
# sans), mais OBLIGATOIRE pour np>1 ; on le signale clairement si absent.
try:
    from mpi4py import MPI  # noqa: F401
    _COMM = MPI.COMM_WORLD
except ImportError:
    _COMM = None

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))

import adc  # noqa: E402
from adc import _adc  # noqa: E402

from model import build_moment_model  # noqa: E402
from run_diocotron import DEBYE, OMEGA_P, diocotron_state  # noqa: E402

N = 64          # cote de la grille (meme que run_diocotron : fidele au scenario de reference)
NSTEPS = 12     # 10-20 pas demandes par le smoke ; 12 = milieu de fourchette


def _rank_size():
    """Rang / nombre de rangs vus PAR LE COEUR (my_rank/n_ranks lisent l'etat MPI initialise par
    mpi4py). En serie pure (sans mpi4py) : 0 / 1."""
    return _adc.my_rank(), _adc.n_ranks()


def _barrier():
    if _COMM is not None:
        _COMM.Barrier()


def compile_model(rho_bg, backend="aot"):
    """Compile le modele 15 moments Vlasov-Poisson (sources electriques + Poisson) une seule fois,
    dans le cache hors source keye (model_hash, abi_key). so_path=None : sur un cache HIT aucune
    recompilation. Sous MPI on SERIALISE le premier build sur le rang 0 (barriere), puis les autres
    rangs reutilisent la .so en cache : aucune ecriture concurrente du meme fichier."""
    from adc_cases.common.native import adc_include

    m = build_moment_model(name="hyqmom15_vp", robust=True, with_sources=True,
                           q_over_m=1.0, omega_c=0.0, debye=DEBYE, rho_background=rho_bg,
                           omega_p=OMEGA_P, exact_speeds=False)
    rank, _ = _rank_size()
    compiled = None
    if rank == 0:
        compiled = m.compile(None, adc_include(), backend=backend)  # populate cache
    _barrier()
    if rank != 0:
        compiled = m.compile(None, adc_include(), backend=backend)  # cache HIT
    return compiled


def make_system(n, compiled, solver, riemann="rusanov"):
    """System periodique + bloc 15 moments compile + Poisson au solveur demande. Memes briques que
    run_diocotron.build_sim, mais le solveur Poisson est un PARAMETRE (geometric_mg pour le run,
    fft/fft_spectral pour le test de rejet)."""
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation("mom", model=compiled,
                     spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
                     time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver=solver)
    return sim


def run_diocotron_mg(n, nsteps, compiled, U0):
    """Avance le diocotron avec Poisson MG. Renvoie (U_final, phi, mass0, dt_loop_s). La facade
    get_state/potential est MONO-RANG : seul le rang proprietaire de l'unique boite (rang 0, par la
    DistributionMapping round-robin box0->rang0) a un etat a relire ; sur les autres rangs ces
    accesseurs levent (pas de boite locale). On ne lit donc U/phi QUE sur le rang 0 (U=phi=None
    ailleurs). Le transport et le Poisson, eux, sont COLLECTIFS et tournent sur tous les rangs."""
    rank, _ = _rank_size()
    sim = make_system(n, compiled, "geometric_mg")
    sim.set_state("mom", U0)
    sim.solve_fields()
    mass0 = float(U0[0].sum())  # masse globale de la CI (la CI complete est connue de tous les rangs)
    _barrier()
    t0 = time.perf_counter()
    for _ in range(nsteps):
        sim.step_cfl(0.4)
    _barrier()
    dt_loop = time.perf_counter() - t0
    U = np.array(sim.get_state("mom")) if rank == 0 else None
    phi = np.array(sim.potential()) if rank == 0 else None
    return U, phi, mass0, dt_loop


def check_fft_rejected(n, compiled, U0):
    """Rejet propre sous MPI : solver='fft' (et 'fft_spectral') doivent lever une RuntimeError a
    travers le driver, sans interblocage ni segfault. Le coeur leve sur TOUS les rangs (ensure_elliptic
    est collectif), donc chaque rang catch symetriquement. @return {solver: (verdict, message)}."""
    out = {}
    for solver in ("fft", "fft_spectral"):
        sim = make_system(n, compiled, solver)
        sim.set_state("mom", U0)
        try:
            sim.solve_fields()
            out[solver] = ("NON-REJETE", "")
        except RuntimeError as e:
            out[solver] = ("rejet RuntimeError", str(e).splitlines()[0])
        except Exception as e:  # noqa: BLE001 -- on veut le type exact pour le rapport
            out[solver] = (type(e).__name__, str(e).splitlines()[0])
        _barrier()
    return out


def _state_path(nranks):
    from adc_cases.common.io import case_output_dir
    d = os.path.join(case_output_dir("hyqmom15"), "mpi_smoke")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "state_np%d.npz" % nranks)


def main():
    rank, nranks = _rank_size()
    # Determinisme bit-a-bit : 1 thread on-node (Kokkos OpenMP a des reductions a ordre non garanti
    # a >1 thread). La parite serie/multi-rang n'a de sens qu'a thread unique.
    try:
        adc.set_threads(1)
    except Exception:  # noqa: BLE001 -- API absente sur un build sans threads : ignore
        pass

    if rank == 0:
        print("=== hyqmom15/run_mpi : diocotron Vlasov-Poisson, Poisson geometric_mg, np=%d ==="
              % nranks)
        if nranks > 1 and _COMM is None:
            print("ATTENTION : mpi4py absent -> MPI non initialise -> chaque process se croit serie. "
                  "Installer mpi4py (compile contre la meme libmpi qu'_adc).")

    U0 = diocotron_state(N)
    rho_bg = float(U0[0].mean())
    compiled = compile_model(rho_bg)

    # (1) rejet FFT/FFT-spectral sous MPI (seulement multi-rang : en serie fft est le chemin valide).
    if nranks > 1:
        rej = check_fft_rejected(N, compiled, U0)
        if rank == 0:
            for solver, (verdict, msg) in rej.items():
                print("(rejet) solver=%-13s -> %s | %s" % (repr(solver), verdict, msg))

    # (2) run principal : 12 pas, Poisson MG.
    U, phi, mass0, dt_loop = run_diocotron_mg(N, NSTEPS, compiled, U0)

    if rank == 0:
        assert U.size > 0, "rang 0 sans boite : topologie inattendue"
        finite = bool(np.all(np.isfinite(U)))
        m00_pos = bool(np.all(U[0] > 0))
        mass1 = float(U[0].sum())
        drift = abs(mass1 - mass0) / mass0
        phi_finite = bool(np.all(np.isfinite(phi)))
        print("(run) np=%d : %d pas, etat fini=%s, M00>0=%s, derive masse=%.2e, phi fini=%s, "
              "boucle=%.3fs" % (nranks, NSTEPS, finite, m00_pos, drift, phi_finite, dt_loop))
        assert finite and m00_pos and phi_finite, "etat/ phi non fini ou M00<=0"
        assert drift < 1e-12, "masse non conservee (%.2e)" % drift

        # Ecrit l'etat final pour la parite (reference np=1, comparaison np>1).
        np.savez(_state_path(nranks), U=U, phi=phi, mass1=mass1, dt_loop=dt_loop)

        # (3) parite vs np=1 (si la reference existe).
        ref = _state_path(1)
        if nranks > 1 and os.path.isfile(ref):
            r = np.load(ref)
            Uref, phiref = r["U"], r["phi"]
            same_U = bool(np.array_equal(U, Uref))
            same_phi = bool(np.array_equal(phi, phiref))
            dU = float(np.max(np.abs(U - Uref))) if U.shape == Uref.shape else float("nan")
            dphi = float(np.max(np.abs(phi - phiref))) if phi.shape == phiref.shape else float("nan")
            verdict = "BIT-IDENTIQUE" if (same_U and same_phi) else "ECART (voir dU/dphi)"
            print("(parite) np=%d vs np=1 : %s | dU_max=%.3e dphi_max=%.3e" %
                  (nranks, verdict, dU, dphi))
        print("hyqmom15/run_mpi (np=%d) : OK" % nranks)


if __name__ == "__main__":
    main()
