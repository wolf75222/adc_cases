#!/usr/bin/env python3
"""Run REEL system-schur -> dumps npz SEULS (sans matplotlib) -- runner de campagne ADC-79 (ROMEO).

Avance le modele complet (build_real de make_paper_figures, minmod par defaut) jusqu'a t_f et dumpe
l'etat brut (densite + phi) via sim.write(format="npz") aux 9 fractions de snapshot. Le RENDU
(schlieren/GIF) se fait en local sur les npz rapatries : aucun import matplotlib ici.

    python3 run_real_npz.py --mode 3 --n 512 [--limiter minmod] [--out DIR] [--precompile-only]

--precompile-only : compile le .so DSL (cache) et sort -- a lancer UNE fois sur le login AVANT les
jobs (les noeuds reutilisent le cache, pas de g++ concurrent par job).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_DIAG = os.path.dirname(os.path.abspath(__file__))
_CASE = os.path.dirname(_DIAG)
_REPO = os.path.dirname(_CASE)
for _p in (_CASE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import (
    PaperParameters,
    paper_initial_density,
    drift_velocity_from_potential,
)  # noqa: E402
from run import compile_model  # noqa: E402
import adc  # noqa: E402

SNAP_FRAC = (0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def build_real(compiled, rho0: np.ndarray, params, limiter: str) -> adc.System:
    """Copie de make_paper_figures.build_real (sans dependre du module qui importe matplotlib)."""
    sim = adc.System(n=rho0.shape[0], L=params.length, periodic=False)
    sim.set_poisson(
        rhs="composite",
        solver="geometric_mg",
        bc="dirichlet",
        wall="circle",
        wall_radius=params.radius,
    )
    sim.set_magnetic_field(params.omega * np.ones_like(rho0))
    sim.add_equation(
        "electrons",
        model=compiled,
        spatial=adc.FiniteVolume(
            limiter=limiter, riemann="rusanov", variables="conservative"
        ),
        time=adc.Strang(
            hyperbolic=adc.Explicit(method="ssprk3"),
            source=adc.CondensedSchur(theta=0.5, alpha=params.alpha),
        ),
    )
    zeros = np.zeros_like(rho0)
    sim.set_primitive_state("electrons", rho=rho0, u=zeros, v=zeros)
    sim.solve_fields()
    u0, v0 = drift_velocity_from_potential(np.asarray(sim.potential()), params)
    sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0)
    sim.solve_fields()
    return sim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--limiter", choices=["minmod", "weno5"], default="minmod")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument(
        "--out", default=None, help="dossier npz (defaut: ./npz_l<mode>_n<n>)"
    )
    ap.add_argument("--precompile-only", action="store_true")
    args = ap.parse_args()

    params = PaperParameters()
    workdir = os.environ.get("ADC_DSL_CACHE", os.path.join(_CASE, "out_dsl"))
    os.makedirs(workdir, exist_ok=True)
    compiled = compile_model(params, "system-schur", workdir)
    if args.precompile_only:
        print(f"[precompile] .so DSL pret dans {workdir}")
        return 0

    out = args.out or f"npz_l{args.mode}_n{args.n}"
    os.makedirs(out, exist_ok=True)
    rho0 = paper_initial_density(args.n, args.mode, params)
    sim = build_real(compiled, rho0, params, args.limiter)

    t_end = 2.0 * math.pi * 10.0
    snaps = [round(f * t_end, 6) for f in SNAP_FRAC]
    si = step = 0
    t0 = time.time()
    while True:
        t = float(sim.time())
        while si < len(snaps) and t >= snaps[si] - 1e-9:
            dens = np.asarray(sim.density("electrons"), float)
            if not np.isfinite(dens).all():
                print(
                    f"FAIL: densite non finie a t={t:.3f} ({100*t/t_end:.0f}%)"
                )
                return 1
            sim.write(os.path.join(out, "state"), format="npz", step=si)
            print(
                f"  snap {si} t={t:7.2f} ({100*t/t_end:5.1f}%) min_rho={dens.min():+.3e} "
                f"wall={time.time()-t0:7.0f}s",
                flush=True,
            )
            si += 1
        if t >= t_end:
            break
        sim.step_cfl(args.cfl)
        step += 1
        if step > 2_000_000:
            print("FAIL: budget de pas epuise")
            return 1
    print(
        f"OK l={args.mode} n={args.n} {args.limiter} : {si} npz dans {out}, "
        f"{step} pas, {time.time()-t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
