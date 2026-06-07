#!/usr/bin/env python3
"""Reproduce the magnetic Euler-Poisson diocotron experiment of arXiv:2510.11808.

Two execution paths are intentionally separated:

``system-schur``
    Uniform finite volumes, WENO5-Z + SSPRK3 and the global condensed Schur
    electrostatic/Lorentz source stage.  This is the closest current ADC path to
    the paper and includes the paper initial drift velocity.

``amr-imex``
    Finite volumes on ``adc.AmrSystem`` with dynamic AMR, Kokkos and MPI.  It
    advances the exact same PDE formulas, but uses the cell-local backward-Euler
    source step because CondensedSchur is not yet implemented on AMR.  The current
    AMR facade also initializes only density, so momentum starts at zero and
    relaxes toward the drift state.  Results from this path are experimental and
    are never labelled as a quantitative reproduction of the paper.
"""

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

import adc

try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adc_cases.common.io import case_output_dir  # noqa: E402
from model import (  # noqa: E402
    PAPER_FIT_WINDOWS,
    PAPER_GROWTH_RATES,
    PAPER_SNAPSHOT_FRACTIONS,
    PaperParameters,
    drift_velocity_from_potential,
    magnetic_euler_poisson_model,
    paper_initial_density,
)
from results import (  # noqa: E402
    adc_cases_sha,
    adc_cpp_sha,
    build_record,
    engine_label,
    verify_paper_windows,
    write_records,
)


@dataclass
class Result:
    mode: int
    times: np.ndarray
    amplitudes: np.ndarray
    growth_rate: float
    snapshots: list
    frames: list
    frame_times: list


def mpi_rank():
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def mpi_size():
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS"):
        if key in os.environ:
            return int(os.environ[key])
    return 1


def compile_model(params, engine, out):
    """Compile once across MPI ranks; subsequent ranks reuse the DSL cache."""
    source = "schur" if engine == "system-schur" else "local"
    target = "system" if engine == "system-schur" else "amr_system"
    model = magnetic_euler_poisson_model(params, source=source)
    lock_path = os.path.join(out, "compile-%s.lock" % engine)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        compiled = model.compile(
            backend="production",
            target=target,
            name="hoffart_%s" % source,
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return compiled


def sample_circle(field, radius, length, ntheta=2048):
    n = field.shape[0]
    h = length / n
    theta = np.linspace(0.0, 2.0 * math.pi, ntheta, endpoint=False)
    x = 0.5 * length + radius * np.cos(theta)
    y = 0.5 * length + radius * np.sin(theta)
    fi = x / h - 0.5
    fj = y / h - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    tx, ty = fi - i0, fj - j0
    values = (
        field[j0, i0] * (1.0 - tx) * (1.0 - ty)
        + field[j0, i0 + 1] * tx * (1.0 - ty)
        + field[j0 + 1, i0] * (1.0 - tx) * ty
        + field[j0 + 1, i0 + 1] * tx * ty
    )
    return values


def mode_amplitude(phi, mode, params):
    values = sample_circle(phi, params.ring_inner, params.length)
    coeffs = np.fft.rfft(values) / values.size
    return 2.0 * abs(coeffs[mode])


def fit_growth(times, amplitudes, mode):
    lo, hi = PAPER_FIT_WINDOWS[mode]
    mask = (times >= lo) & (times <= hi) & (amplitudes > 0.0)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    return float(np.polyfit(times[mask], np.log(amplitudes[mask]), 1)[0])


def build_uniform(compiled, rho, params):
    n = rho.shape[0]
    sim = adc.System(n=n, L=params.length, periodic=False)
    sim.set_poisson(
        rhs="composite",
        solver="geometric_mg",
        bc="dirichlet",
        wall="circle",
        wall_radius=params.radius,
    )
    # CondensedSchur requires B_z before set_source_stage is installed.
    sim.set_magnetic_field(params.omega * np.ones_like(rho))
    sim.add_equation(
        "electrons",
        model=compiled,
        spatial=adc.FiniteVolume(
            limiter="weno5",
            riemann="rusanov",
            variables="conservative",
        ),
        time=adc.Split(
            hyperbolic=adc.Explicit(ssprk3=True),
            source=adc.CondensedSchur(theta=0.5, alpha=params.alpha),
        ),
    )

    zeros = np.zeros_like(rho)
    sim.set_primitive_state("electrons", rho=rho, u=zeros, v=zeros)
    sim.solve_fields()
    u0, v0 = drift_velocity_from_potential(np.asarray(sim.potential()), params)
    sim.set_primitive_state("electrons", rho=rho, u=u0, v=v0)
    sim.solve_fields()
    return sim


def build_amr(compiled, rho, params, args):
    sim = adc.AmrSystem(
        n=rho.shape[0],
        L=params.length,
        periodic=False,
        regrid_every=args.regrid_every,
        distribute_coarse=args.distribute_coarse,
        coarse_max_grid=args.coarse_max_grid,
    )
    sim.add_equation(
        "electrons",
        model=compiled,
        spatial=adc.FiniteVolume(
            limiter="weno5",
            riemann="rusanov",
            variables="conservative",
        ),
        time=adc.IMEX(substeps=args.substeps),
    )
    sim.set_poisson(
        rhs="composite",
        solver="geometric_mg",
        bc="dirichlet",
        wall="circle",
        wall_radius=params.radius,
    )
    sim.set_refinement(args.refine_threshold)
    sim.set_density("electrons", rho)
    return sim


def density(sim):
    return np.asarray(sim.density("electrons"))


def potential(sim):
    return np.asarray(sim.potential())


def run_mode(mode, compiled, params, args):
    rho0 = paper_initial_density(args.n, mode, params)
    if args.engine == "system-schur":
        sim = build_uniform(compiled, rho0, params)
    else:
        sim = build_amr(compiled, rho0, params, args)

    snapshot_targets = [f * args.t_end for f in PAPER_SNAPSHOT_FRACTIONS]
    frame_targets = np.linspace(0.0, args.t_end, args.gif_frames)
    snapshots, frames, frame_times = [], [], []
    times, amplitudes = [], []
    next_snapshot = next_frame = 0
    step = 0

    while True:
        t = float(sim.time())
        if step % args.sample_every == 0 or t >= args.t_end:
            phi = potential(sim)
            amp = mode_amplitude(phi, mode, params)
            times.append(t)
            amplitudes.append(amp)
            current_density = None

            while next_snapshot < len(snapshot_targets) and t >= snapshot_targets[next_snapshot] - 0.5 * args.dt:
                if current_density is None:
                    current_density = density(sim)
                snapshots.append((t, current_density.copy()))
                next_snapshot += 1
            while next_frame < len(frame_targets) and t >= frame_targets[next_frame] - 0.5 * args.dt:
                if current_density is None:
                    current_density = density(sim)
                frames.append(current_density.copy())
                frame_times.append(t)
                next_frame += 1

            if not np.isfinite(phi).all() or not np.isfinite(amp):
                raise FloatingPointError("non-finite potential/amplitude at t=%g" % t)

        if t >= args.t_end - 0.5 * args.dt:
            break
        sim.step(min(args.dt, args.t_end - t))
        step += 1
        if step > args.max_steps:
            raise RuntimeError("max_steps reached before t_end")

    times = np.asarray(times)
    amplitudes = np.asarray(amplitudes)
    return Result(
        mode=mode,
        times=times,
        amplitudes=amplitudes,
        growth_rate=fit_growth(times, amplitudes, mode),
        snapshots=snapshots,
        frames=frames,
        frame_times=frame_times,
    )


def schlieren(rho, params):
    h = params.length / rho.shape[0]
    gy, gx = np.gradient(rho, h, h, edge_order=2)
    grad = np.hypot(gx, gy)
    image = np.log1p(20.0 * grad / max(float(grad.max()), 1.0e-30))
    x = (np.arange(rho.shape[0]) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.ma.array(image, mask=np.hypot(X, Y) > params.radius)


def write_mode_outputs(result, out, params, engine, make_gif):
    mode_dir = os.path.join(out, "mode_%d" % result.mode)
    os.makedirs(mode_dir, exist_ok=True)

    with open(os.path.join(mode_dir, "amplitude.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "amplitude", "amplitude_over_initial"])
        a0 = max(float(result.amplitudes[0]), 1.0e-300)
        for t, a in zip(result.times, result.amplitudes):
            writer.writerow([t, a, a / a0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    normalized = result.amplitudes / max(float(result.amplitudes[0]), 1.0e-300)
    ax.semilogy(result.times, normalized, color="black", lw=1.5)
    lo, hi = PAPER_FIT_WINDOWS[result.mode]
    fit = (result.times >= lo) & (result.times <= hi) & np.isfinite(normalized)
    if np.any(fit):
        anchor_index = np.flatnonzero(fit)[len(np.flatnonzero(fit)) // 2]
        anchor_time = result.times[anchor_index]
        anchor_value = normalized[anchor_index]
        theory = anchor_value * np.exp(
            PAPER_GROWTH_RATES[result.mode] * (result.times - anchor_time)
        )
        ax.semilogy(
            result.times,
            theory,
            color="tab:red",
            ls="--",
            lw=1.2,
            label=r"$\exp(\gamma_%d t)$, paper" % result.mode,
        )
    ax.axvspan(lo, hi, color="tab:blue", alpha=0.12, label="paper fit window")
    ax.set(xlabel="time", ylabel=r"$|c_l(t)|/|c_l(0)|$",
           title="Mode l=%d, gamma=%s" % (
               result.mode,
               "n/a" if not np.isfinite(result.growth_rate) else "%.4f" % result.growth_rate,
           ))
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(mode_dir, "amplitude.png"), dpi=180)
    plt.close(fig)

    if result.snapshots:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
        for ax, (t, rho) in zip(axes.flat, result.snapshots):
            ax.imshow(
                schlieren(rho, params),
                origin="lower",
                extent=(-params.radius, params.radius, -params.radius, params.radius),
                cmap="inferno",
            )
            ax.set_title("t = %.3f" % t)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[len(result.snapshots):]:
            ax.axis("off")
        fig.suptitle("%s: density schlieren, mode l=%d" % (engine, result.mode))
        fig.savefig(os.path.join(mode_dir, "snapshots.png"), dpi=180)
        plt.close(fig)

    if make_gif and result.frames:
        from matplotlib import animation
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        image = ax.imshow(
            schlieren(result.frames[0], params),
            origin="lower",
            extent=(-params.radius, params.radius, -params.radius, params.radius),
            cmap="inferno",
        )
        title = ax.set_title("l=%d, t=%.3f" % (result.mode, result.frame_times[0]))
        ax.set_aspect("equal")

        def update(k):
            image.set_data(schlieren(result.frames[k], params))
            title.set_text("l=%d, t=%.3f" % (result.mode, result.frame_times[k]))
            return image, title

        movie = animation.FuncAnimation(fig, update, frames=len(result.frames), interval=80)
        movie.save(
            os.path.join(mode_dir, "diocotron_l%d.gif" % result.mode),
            writer=animation.PillowWriter(fps=12),
        )
        plt.close(fig)


def write_summary(results, out, params, args):
    rows = []
    for result in results:
        target = PAPER_GROWTH_RATES[result.mode]
        error = 100.0 * (result.growth_rate - target) / target if np.isfinite(result.growth_rate) else float("nan")
        rows.append((result.mode, result.growth_rate, target, error))

    with open(os.path.join(out, "growth_rates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "gamma_numeric", "gamma_paper", "relative_error_percent"])
        writer.writerows(rows)

    # Enregistrement de mesure PRE-ENREGISTRE (graine de la table de validation
    # Phase 2). Un enregistrement par mode, avec SHA des deux depots, backend, n,
    # dt, splitting, theta du Schur, fenetre de fit verbatim, gamma_numeric (BRUT,
    # aucun facteur 2 pi / rhobar) et err_pct. Le moteur est etiquete explicitement
    # (full-system-schur vs amr-imex-experimental) pour que la pente BRUTE du
    # modele complet ne soit jamais melangee aux nombres reduits porteurs du 2 pi
    # (ceux-la vivent dans diag/diag_polar_omega.py, engine='reduced-ExB').
    splitting = "Lie"  # les deux chemins sont en Lie (Godunov), pas Strang ; cf. docs
    if args.engine == "system-schur":
        schur_theta = 0.5
        backend = "kokkos-serial" if mpi_size() == 1 else "mpi-%d" % mpi_size()
    else:
        schur_theta = None  # amr-imex : source IMEX cell-local, pas de CondensedSchur
        backend = "kokkos-mpi-%d" % mpi_size() if mpi_size() > 1 else "kokkos-serial"
    cpp_sha = adc_cpp_sha(adc)
    cases_sha = adc_cases_sha()
    records = [
        build_record(
            engine=args.engine,
            mode=result.mode,
            gamma_numeric=result.growth_rate,
            gamma_paper=PAPER_GROWTH_RATES[result.mode],
            fit_window=PAPER_FIT_WINDOWS[result.mode],
            n=args.n,
            dt=args.dt,
            splitting=splitting,
            schur_theta=schur_theta,
            backend=backend,
            mpi_size=mpi_size(),
            adc_cpp_sha_value=cpp_sha,
            adc_cases_sha_value=cases_sha,
        )
        for result in results
    ]
    write_records(records, out)

    metadata = {
        "paper": "https://arxiv.org/abs/2510.11808",
        "engine": args.engine,
        "engine_label": engine_label(args.engine),
        "normalization": "raw (no 2pi, no rhobar): full-model slope vs paper directly; "
                         "the 2pi/rhobar factor belongs ONLY to the reduced ExB-scalar "
                         "path (diag/diag_polar_omega.py, engine=reduced-ExB)",
        "adc_cpp_sha": cpp_sha,
        "adc_cases_sha": cases_sha,
        "parameters": params.to_dict(),
        "numerics": {
            "finite_volume": "WENO5-Z + Rusanov",
            "time": "SSPRK3 + CondensedSchur(theta=0.5)" if args.engine == "system-schur"
                    else "AMR transport + cell-local backward-Euler source",
            "dt": args.dt,
            "n": args.n,
            "mpi_size": mpi_size(),
        },
        "fidelity": {
            "same_pde": True,
            "paper_initial_drift": args.engine == "system-schur",
            "paper_schur_source": args.engine == "system-schur",
            "amr": args.engine == "amr-imex",
            "quantitative_comparison_enabled": args.engine == "system-schur",
            "quantitative_paper_claim": False,
            "known_differences": (
                ["finite-volume spatial discretisation", "Lie rather than Strang splitting"]
                if args.engine == "system-schur"
                else [
                    "finite-volume AMR spatial discretisation",
                    "cell-local IMEX rather than condensed Schur",
                    "zero initial momentum rather than the drift state",
                    "Cartesian transport outside the circular Poisson wall",
                ]
            ),
        },
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [r[0] for r in rows]
    numeric = [r[1] for r in rows]
    target = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(modes, target, "s-", color="tab:red", label="paper")
    ax.plot(modes, numeric, "o-", color="black", label=args.engine)
    ax.set(xlabel="azimuthal mode l", ylabel="growth rate gamma",
           xticks=modes, title="Diocotron growth rates")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "growth_rates.png"), dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("system-schur", "amr-imex"), default="system-schur")
    parser.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--gif-frames", type=int, default=80)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="paper does not state a numerical theta; default is the cold limit")
    parser.add_argument("--beta", type=float, default=1.0e6)
    parser.add_argument("--regrid-every", type=int, default=20)
    parser.add_argument("--refine-threshold", type=float, default=0.05)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--distribute-coarse", action="store_true")
    parser.add_argument("--coarse-max-grid", type=int, default=0)
    parser.add_argument("--acknowledge-amr-approximation", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(mode not in PAPER_GROWTH_RATES for mode in args.modes):
        raise SystemExit("--modes must be selected from 3, 4, 5")
    if args.engine == "system-schur" and mpi_size() > 1:
        raise SystemExit("system-schur is a single-rank reference; use amr-imex under MPI")
    if args.engine == "amr-imex" and not args.acknowledge_amr_approximation:
        raise SystemExit(
            "amr-imex uses the same PDE but not the paper Schur stage or initial drift. "
            "Re-run with --acknowledge-amr-approximation."
        )
    # Pre-enregistrement : la comparaison du modele complet DOIT utiliser les
    # fenetres de fit verbatim du papier (Fig. 5.4). Cet assert verrouille
    # PAPER_FIT_WINDOWS contre toute fenetre adaptative qui se glisserait dans le
    # chemin complet (fit_growth lit directement PAPER_FIT_WINDOWS). Leve avant
    # toute mesure ; n'introduit AUCUNE nouvelle fenetre.
    verify_paper_windows(PAPER_FIT_WINDOWS)
    if args.quick:
        args.n = 48
        args.t_end = 0.02
        args.dt = 1.0e-3
        args.sample_every = 1
        args.gif_frames = 8
        args.modes = [3]

    params = PaperParameters(
        beta=args.beta,
        final_time=args.t_end,
        temperature=args.temperature,
    )
    out = case_output_dir("hoffart_euler_poisson_dsl_%s" % args.engine.replace("-", "_"))
    compiled = compile_model(params, args.engine, out)
    if args.compile_only:
        if mpi_rank() == 0:
            print("compiled model:", compiled.so_path)
        return

    results = []
    for mode in args.modes:
        if mpi_rank() == 0:
            print("[%s] mode l=%d, n=%d, t_end=%g, dt=%g" % (
                args.engine, mode, args.n, args.t_end, args.dt))
        result = run_mode(mode, compiled, params, args)
        results.append(result)
        if mpi_rank() == 0:
            target = PAPER_GROWTH_RATES[mode]
            print("  gamma = %s (paper %.3f)" % (
                "n/a" if not np.isfinite(result.growth_rate) else "%.6f" % result.growth_rate,
                target,
            ))
            write_mode_outputs(result, out, params, args.engine, not args.no_gif)

    if mpi_rank() == 0:
        write_summary(results, out, params, args)
        print("outputs:", out)


if __name__ == "__main__":
    main()
