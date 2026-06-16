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

from __future__ import annotations

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
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

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
    gamma_to_paper_units,
    paper_to_sim_time_window,
    verify_paper_windows,
    write_records,
)


@dataclass
class Result:
    """Time history and diagnostics produced by a single-mode diocotron run.

    Attributes:
        mode: Azimuthal mode number l of the perturbation.
        times: Sampled simulation times.
        amplitudes: Mode amplitude |c_l(t)| on the inner ring at each time.
        growth_rate: Raw slope of log|c_l| in the (mapped) paper fit window.
        snapshots: (time, density) pairs at the paper snapshot fractions.
        frames: Density fields sampled on the GIF frame schedule.
        frame_times: Simulation times matching ``frames``.
    """

    mode: int
    times: np.ndarray
    amplitudes: np.ndarray
    growth_rate: float
    snapshots: list
    frames: list
    frame_times: list


def mpi_rank() -> int:
    """Return this process' MPI rank from the environment (0 if serial)."""
    for key in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "SLURM_PROCID",
    ):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def mpi_size() -> int:
    """Return the MPI world size from the environment (1 if serial)."""
    for key in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "SLURM_NTASKS",
    ):
        if key in os.environ:
            return int(os.environ[key])
    return 1


def compile_model(params, engine: str, out: str):
    """Compile the model once across MPI ranks; later ranks hit the cache.

    A file lock serialises the DSL compilation so a single rank builds the
    shared object and the rest reuse the on-disk cache.

    Args:
        params: Paper parameters fed to the model builder.
        engine: ``"system-schur"`` or ``"amr-imex"``; selects source/target.
        out: Output directory holding the per-engine compile lock file.

    Returns:
        The compiled model block.
    """
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


def sample_circle(
    field: np.ndarray,
    radius: float,
    length: float,
    ntheta: int = 2048,
) -> np.ndarray:
    """Bilinearly sample a Cartesian field along a centred circle.

    Args:
        field: 2D field on a uniform ``n x n`` grid spanning ``[0, length]``.
        radius: Radius of the sampling circle, centred at the box centre.
        length: Physical side length of the square domain.
        ntheta: Number of equispaced angular samples around the circle.

    Returns:
        The ``ntheta`` interpolated field values, ordered by angle.
    """
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


def mode_amplitude(phi: np.ndarray, mode: int, params) -> float:
    """Return the amplitude of azimuthal mode ``mode`` on the inner ring.

    Samples ``phi`` along the inner ring and reads off the FFT coefficient of
    the requested mode.

    Args:
        phi: Electrostatic potential field on the Cartesian grid.
        mode: Azimuthal mode number l to extract.
        params: Paper parameters providing the ring radius and box length.

    Returns:
        The (two-sided) modal amplitude |c_l|.
    """
    values = sample_circle(phi, params.ring_inner, params.length)
    coeffs = np.fft.rfft(values) / values.size
    return 2.0 * abs(coeffs[mode])


def fit_growth(
    times: np.ndarray,
    amplitudes: np.ndarray,
    mode: int,
    rhobar: float = 1.0,
) -> float:
    """Pente BRUTE de log|c_l| dans la fenetre papier mappee en temps sim.

    T3 : la fenetre papier (model.PAPER_FIT_WINDOWS, en temps T_d) est convertie en temps
    sim par ``t_sim = (2pi/rhobar) t_paper`` (le solveur tourne en horloge ExB-naturelle,
    le papier en horloge omega_d cyclique). Fitter la fenetre papier BRUTE sur ``times``
    (sim) tomberait dans le transitoire -- c'etait l'artefact -95 %. La conversion en
    unites papier (x2pi/rhobar) est faite a l'enregistrement.

    Args:
        times: Temps de simulation echantillonnes.
        amplitudes: Amplitude |c_l| du mode a chaque temps.
        mode: Numero de mode azimutal l (indexe PAPER_FIT_WINDOWS).
        rhobar: Densite de reference (rho_max) liant horloge sim et papier.

    Returns:
        gamma_raw_sim, la pente brute ; NaN si la fenetre a moins de 4 points.
    """
    lo, hi = paper_to_sim_time_window(PAPER_FIT_WINDOWS[mode], rhobar)
    mask = (times >= lo) & (times <= hi) & (amplitudes > 0.0)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    return float(np.polyfit(times[mask], np.log(amplitudes[mask]), 1)[0])


def build_uniform(
    compiled,
    rho: np.ndarray,
    params,
    geometry: str = "square",
    gauss_policy: str = "restart",
) -> adc.System:
    """Build the uniform System for the paper-faithful system-schur path.

    Sets up the circular-wall Poisson solver, magnetic field, WENO5/Strang
    transport with the condensed Schur source, then seeds the paper drift state
    via a two-pass Poisson -> drift -> Poisson relaxation.

    Args:
        compiled: Compiled model block targeting ``System``.
        rho: Initial density on the ``n x n`` grid.
        params: Paper parameters (length, radius, omega, alpha, ...).
        geometry: ``"square"`` (full Cartesian transport, default), or
            ``"staircase"``/``"cutcell"`` to confine transport to the disc.
        gauss_policy: ``"restart"`` (re-solve Poisson each ``solve_fields``) or
            ``"evolve"`` (solve only at t=0, then carry phi in the Schur stage).

    Returns:
        The initialised ``adc.System`` ready to step.

    Raises:
        ValueError: If ``geometry`` is not one of the accepted values.
    """
    n = rho.shape[0]
    sim = adc.System(n=n, L=params.length, periodic=False)
    sim.set_poisson(
        rhs="composite",
        solver="geometric_mg",
        bc="dirichlet",
        wall="circle",
        wall_radius=params.radius,
    )
    # GEOMETRIE DU TRANSPORT FV.
    # 'square' (defaut) : transport sur le carre cartesien complet (cote L = 2R),
    # comportement historique BIT-IDENTIQUE, aucun appel a set_disc_domain.
    # 'staircase'/'cutcell' : set_disc_domain(L/2, L/2, R, mode=...) materialise le
    # masque disque (meme level set que le mur Poisson circulaire) ET est branche dans
    # System::step depuis adc_cpp #224 -- le transport FV est confine au disque.
    # NB : cut-cell n'a pas d'effet mesurable sur le taux. Le "deficit -95%" historique
    # etait un artefact de metrologie (fenetre + horloge), pas la geometrie ni le schema :
    # mesure paper-faithful, le full reproduit a <10% et converge (RESULTS sections 9-11, T3).
    if geometry in ("staircase", "cutcell"):
        sim.set_disc_domain(
            0.5 * params.length,
            0.5 * params.length,
            params.radius,
            mode=geometry,
        )
    elif geometry != "square":
        raise ValueError(
            "geometry must be 'square', 'staircase' or 'cutcell', got %r"
            % geometry
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
        time=adc.Strang(
            # Strang donne le splitting symetrique d'ordre 2 du papier :
            # H(dt/2) ; S(dt) ; H(dt/2). ssprk3 est le schema RK faible
            # dissipation du papier et tourne sur le chemin production via
            # adc_cpp feat/ssprk3-aot-gauss-policy.
            hyperbolic=adc.Explicit(method="ssprk3"),
            source=adc.CondensedSchur(theta=0.5, alpha=params.alpha),
        ),
    )

    # Loi de Gauss : 'restart' (defaut) re-resout le Poisson a chaque solve_fields
    # (bit-identique a l'historique) ; 'evolve' ne resout qu'a t=0 puis laisse l'etage
    # source Schur porter phi in-place. Pose AVANT le premier solve_fields (le premier
    # solve resout dans les deux cas ; le verrou gauss_solved_once_ est remis a zero ici).
    sim.set_gauss_policy(gauss_policy)

    zeros = np.zeros_like(rho)
    sim.set_primitive_state("electrons", rho=rho, u=zeros, v=zeros)
    sim.solve_fields()
    u0, v0 = drift_velocity_from_potential(np.asarray(sim.potential()), params)
    sim.set_primitive_state("electrons", rho=rho, u=u0, v=v0)
    sim.solve_fields()
    return sim


def amr_initial_drift(params, rho: np.ndarray):
    """Vitesse de derive initiale du papier pour semer l'etat AMR (Phase B).

    Calcule ``v0 = -(grad phi0 x Omega)/|Omega|^2`` en resolvant le Poisson initial
    ``-Delta phi = alpha rho`` (meme paroi circulaire, resolution = niveau grossier AMR)
    sur un System uniforme JETABLE, via un compile target='system' du meme modele (le
    chemin AMR ne resout pas le Poisson au build, et son modele compile cible
    'amr_system' n'est pas chargeable dans un System).

    SOLVE UNIQUE : contrairement au chemin system-schur (relaxation a deux passes
    Poisson->derive->Poisson, cf. build_uniform), on ne fait qu'un solve -> fidelite
    SINGLE-PASS, signalee distinctement dans les metadonnees.

    Args:
        params: Parametres papier (alpha, paroi circulaire, omega, ...).
        rho: Densite initiale definissant la resolution du System sonde.

    Returns:
        Le couple (u0, v0) de la vitesse de derive sur la grille grossiere.

    Raises:
        Exception: Si le solve Poisson echoue (l'appelant retombe sur set_density).
    """
    n = rho.shape[0]
    probe_model = magnetic_euler_poisson_model(
        params, source="schur"
    )  # source nulle : seul le Poisson importe
    compiled_sys = probe_model.compile(
        backend="production", target="system", name="hoffart_amr_drift_probe"
    )
    probe = adc.System(n=n, L=params.length, periodic=False)
    probe.set_poisson(
        rhs="composite",
        solver="geometric_mg",
        bc="dirichlet",
        wall="circle",
        wall_radius=params.radius,
    )
    probe.add_equation(
        "electrons",
        model=compiled_sys,
        spatial=adc.FiniteVolume(
            limiter="weno5", riemann="rusanov", variables="conservative"
        ),
        time=adc.Explicit(method="ssprk2"),
    )
    zeros = np.zeros_like(rho)
    probe.set_primitive_state("electrons", rho=rho, u=zeros, v=zeros)
    probe.solve_fields()
    return drift_velocity_from_potential(np.asarray(probe.potential()), params)


def build_amr(compiled, rho: np.ndarray, params, args) -> adc.AmrSystem:
    """Build the dynamic-AMR System for the experimental amr-imex path.

    Configures the circular-wall Poisson solver, refinement threshold and
    IMEX transport, then seeds the paper drift state when possible (Phase B);
    on any failure it falls back to the m=0 density-only state.

    Args:
        compiled: Compiled model block targeting ``AmrSystem``.
        rho: Initial coarse-level density.
        params: Paper parameters (length, radius, ...).
        args: Parsed CLI namespace (regrid/refine/substep/coarse options).

    Returns:
        The initialised ``adc.AmrSystem`` ready to step.
    """
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
    # Phase B (Probleme 2) : demarrer l'AMR depuis l'etat de DERIVE du papier (rho, rho*u0, rho*v0)
    # au lieu de m=0. set_conservative_state pose l'etat conservatif COMPLET sur le grossier (prolonge
    # aux niveaux fins). Tout echec -- solve degenere, OU un adc anterieur a set_conservative_state --
    # RETOMBE proprement sur set_density (comportement historique m=0) : aucune regression de robustesse.
    try:
        u0, v0 = amr_initial_drift(params, rho)
        # Modele isotherme 3-var : conservatif = [rho, rho*u, rho*v] (pas d'energie). On valide le
        # NOMBRE de composantes contre le modele compile (l'ordre suit conservative_from de model.py).
        state = np.stack([rho, rho * u0, rho * v0])
        if state.shape[0] != compiled.n_vars:
            raise ValueError(
                "etat de derive a %d composantes mais le bloc compile en a %d"
                % (state.shape[0], compiled.n_vars)
            )
        sim.set_conservative_state("electrons", state)
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- fallback robuste, jamais de regression du build
        if mpi_rank() == 0:
            print(
                "[amr-imex] seed de derive indisponible (%s) -> fallback set_density (m=0)"
                % exc
            )
        sim.set_density("electrons", rho)
    return sim


def density(sim) -> np.ndarray:
    """Return the electron density field of ``sim`` as a NumPy array."""
    return np.asarray(sim.density("electrons"))


def potential(sim) -> np.ndarray:
    """Return the electrostatic potential of ``sim`` as a NumPy array."""
    return np.asarray(sim.potential())


def run_mode(mode: int, compiled, params, args) -> Result:
    """Run one azimuthal mode to ``t_end`` and collect its diagnostics.

    Builds the engine-appropriate System, time-steps it while sampling the
    modal amplitude, density snapshots and GIF frames, then fits the growth
    rate over the mapped paper window.

    Args:
        mode: Azimuthal mode number l to perturb.
        compiled: Compiled model block for the active engine.
        params: Paper parameters.
        args: Parsed CLI namespace (engine, n, dt, t_end, sampling, ...).

    Returns:
        A :class:`Result` holding the time history and diagnostics.

    Raises:
        FloatingPointError: If the potential or amplitude becomes non-finite.
        RuntimeError: If ``max_steps`` is reached before ``t_end``.
    """
    rho0 = paper_initial_density(args.n, mode, params)
    if args.engine == "system-schur":
        sim = build_uniform(
            compiled,
            rho0,
            params,
            geometry=args.geometry,
            gauss_policy=args.gauss_policy,
        )
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

            while (
                next_snapshot < len(snapshot_targets)
                and t >= snapshot_targets[next_snapshot] - 0.5 * args.dt
            ):
                if current_density is None:
                    current_density = density(sim)
                snapshots.append((t, current_density.copy()))
                next_snapshot += 1
            while (
                next_frame < len(frame_targets)
                and t >= frame_targets[next_frame] - 0.5 * args.dt
            ):
                if current_density is None:
                    current_density = density(sim)
                frames.append(current_density.copy())
                frame_times.append(t)
                next_frame += 1

            if not np.isfinite(phi).all() or not np.isfinite(amp):
                raise FloatingPointError(
                    "non-finite potential/amplitude at t=%g" % t
                )

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
        growth_rate=fit_growth(times, amplitudes, mode, rhobar=params.rho_max),
        snapshots=snapshots,
        frames=frames,
        frame_times=frame_times,
    )


def schlieren(rho: np.ndarray, params) -> np.ndarray:
    """Return a disc-masked schlieren image of the density gradient.

    Args:
        rho: Density field on the Cartesian grid.
        params: Paper parameters (length and disc radius).

    Returns:
        A masked array of the log-compressed gradient magnitude, with cells
        outside the disc masked out.
    """
    h = params.length / rho.shape[0]
    gy, gx = np.gradient(rho, h, h, edge_order=2)
    grad = np.hypot(gx, gy)
    image = np.log1p(20.0 * grad / max(float(grad.max()), 1.0e-30))
    x = (np.arange(rho.shape[0]) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.ma.array(image, mask=np.hypot(X, Y) > params.radius)


def write_mode_outputs(
    result: Result,
    out: str,
    params,
    engine: str,
    make_gif: bool,
) -> None:
    """Write the per-mode CSV and (if matplotlib is present) figures/GIF.

    Always writes ``amplitude.csv``; the amplitude plot, schlieren snapshot
    grid and optional diocotron GIF are skipped when matplotlib is missing.

    Args:
        result: The mode's time history and diagnostics.
        out: Run output directory; a ``mode_<l>`` subdir is created.
        params: Paper parameters used for plotting and normalisation.
        engine: Engine label used in figure titles.
        make_gif: Whether to render the animated GIF.
    """
    mode_dir = os.path.join(out, "mode_%d" % result.mode)
    os.makedirs(mode_dir, exist_ok=True)

    with open(os.path.join(mode_dir, "amplitude.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "amplitude", "amplitude_over_initial"])
        a0 = max(float(result.amplitudes[0]), 1.0e-300)
        for t, a in zip(result.times, result.amplitudes):
            writer.writerow([t, a, a / a0])

    try:
        import matplotlib
    except ImportError:
        return  # figures optionnelles : sans matplotlib on garde les CSV (amplitude.csv ci-dessus + growth_rates.csv en aval)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    normalized = result.amplitudes / max(float(result.amplitudes[0]), 1.0e-300)
    ax.semilogy(result.times, normalized, color="black", lw=1.5)
    # T3 : la fenetre papier (T_d) est mappee en temps SIM ; le taux papier (unites omega_d)
    # se trace en horloge sim avec le facteur rhobar/(2pi) (l'inverse de gamma_to_paper_units).
    rhobar = params.rho_max
    lo, hi = paper_to_sim_time_window(PAPER_FIT_WINDOWS[result.mode], rhobar)
    paper_rate_sim = PAPER_GROWTH_RATES[result.mode] * rhobar / (2.0 * math.pi)
    gamma_paper_units = gamma_to_paper_units(result.growth_rate, rhobar)
    fit = (result.times >= lo) & (result.times <= hi) & np.isfinite(normalized)
    if np.any(fit):
        anchor_index = np.flatnonzero(fit)[len(np.flatnonzero(fit)) // 2]
        anchor_time = result.times[anchor_index]
        anchor_value = normalized[anchor_index]
        theory = anchor_value * np.exp(
            paper_rate_sim * (result.times - anchor_time)
        )
        ax.semilogy(
            result.times,
            theory,
            color="tab:red",
            ls="--",
            lw=1.2,
            label=r"$\exp(\gamma_%d^{paper} t_{sim})$" % result.mode,
        )
    ax.axvspan(
        lo,
        hi,
        color="tab:blue",
        alpha=0.12,
        label="paper fit window (mapped to sim time)",
    )
    ax.set(
        xlabel="sim time",
        ylabel=r"$|c_l(t)|/|c_l(0)|$",
        title="l=%d  gamma_raw=%s  gamma_paper=%s (x2pi/rhobar)  target %.3f"
        % (
            result.mode,
            (
                "n/a"
                if not np.isfinite(result.growth_rate)
                else "%.4f" % result.growth_rate
            ),
            "n/a" if gamma_paper_units is None else "%.3f" % gamma_paper_units,
            PAPER_GROWTH_RATES[result.mode],
        ),
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(mode_dir, "amplitude.png"), dpi=180)
    plt.close(fig)

    if result.snapshots:
        fig, axes = plt.subplots(
            3, 3, figsize=(10, 10), constrained_layout=True
        )
        for ax, (t, rho) in zip(axes.flat, result.snapshots):
            ax.imshow(
                schlieren(rho, params),
                origin="lower",
                extent=(
                    -params.radius,
                    params.radius,
                    -params.radius,
                    params.radius,
                ),
                cmap="inferno",
            )
            ax.set_title("t = %.3f" % t)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[len(result.snapshots) :]:
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
            extent=(
                -params.radius,
                params.radius,
                -params.radius,
                params.radius,
            ),
            cmap="inferno",
        )
        title = ax.set_title(
            "l=%d, t=%.3f" % (result.mode, result.frame_times[0])
        )
        ax.set_aspect("equal")

        def update(k):
            image.set_data(schlieren(result.frames[k], params))
            title.set_text(
                "l=%d, t=%.3f" % (result.mode, result.frame_times[k])
            )
            return image, title

        movie = animation.FuncAnimation(
            fig, update, frames=len(result.frames), interval=80
        )
        movie.save(
            os.path.join(mode_dir, "diocotron_l%d.gif" % result.mode),
            writer=animation.PillowWriter(fps=12),
        )
        plt.close(fig)


def write_summary(results: list, out: str, params, args) -> None:
    """Write the cross-mode summary: growth_rates CSV, records, metadata, plot.

    Aggregates all modes into ``growth_rates.csv``, the pre-registered
    measurement records, ``metadata.json`` (provenance, normalisation and
    fidelity notes) and, when matplotlib is present, ``growth_rates.png``.

    Args:
        results: One :class:`Result` per mode.
        out: Run output directory.
        params: Paper parameters used for normalisation and metadata.
        args: Parsed CLI namespace (engine, n, dt, geometry, ...).
    """
    # T3 : on reporte gamma_raw_sim (pente brute, fenetre MAPPEE) ET gamma_paper_units
    # = gamma_raw_sim * 2pi/rhobar ; l'erreur compare gamma_paper_units a la cible.
    rhobar = params.rho_max
    rows = []
    for result in results:
        target = PAPER_GROWTH_RATES[result.mode]
        g_paper = gamma_to_paper_units(result.growth_rate, rhobar)
        error = (
            (100.0 * (g_paper - target) / target)
            if g_paper is not None
            else float("nan")
        )
        rows.append(
            (
                result.mode,
                result.growth_rate,
                ("" if g_paper is None else g_paper),
                target,
                error,
            )
        )

    with open(os.path.join(out, "growth_rates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "gamma_raw_sim",
                "gamma_paper_units",
                "gamma_paper",
                "relative_error_percent",
            ]
        )
        writer.writerows(rows)

    # Enregistrement de mesure PRE-ENREGISTRE (graine de la table de validation Phase 2).
    # Un enregistrement par mode, avec SHA des deux depots, backend, n, dt, splitting, theta
    # du Schur, fenetre papier (T_d) + fenetre sim mappee, gamma_raw_sim ET gamma_paper_units
    # (= raw * 2pi/rhobar -- T3 : le facteur s'applique au modele complet AUSSI), err_pct.
    splitting = "Strang" if args.engine == "system-schur" else "Lie"
    # system-schur : Strang H(dt/2);S(dt);H(dt/2) via adc.Strang + ssprk3.
    # amr-imex    : Lie/Godunov (CondensedSchur absent sur AmrSystem).
    if args.engine == "system-schur":
        schur_theta = 0.5
        backend = "kokkos-serial" if mpi_size() == 1 else "mpi-%d" % mpi_size()
    else:
        schur_theta = (
            None  # amr-imex : source IMEX cell-local, pas de CondensedSchur
        )
        backend = (
            "kokkos-mpi-%d" % mpi_size() if mpi_size() > 1 else "kokkos-serial"
        )
    cpp_sha = adc_cpp_sha(adc)
    cases_sha = adc_cases_sha()
    records = [
        build_record(
            engine=args.engine,
            mode=result.mode,
            gamma_raw_sim=result.growth_rate,
            gamma_paper=PAPER_GROWTH_RATES[result.mode],
            fit_window=PAPER_FIT_WINDOWS[result.mode],
            n=args.n,
            dt=args.dt,
            splitting=splitting,
            schur_theta=schur_theta,
            backend=backend,
            rhobar=rhobar,
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
        "geometry": args.geometry,
        "geometry_note": (
            "square: full Cartesian square transport (historical default, bit-identical). "
            "staircase/cutcell: set_disc_domain(L/2, L/2, R, mode=...) materialises the "
            "disc mask AND is routed into System::step transport (adc_cpp #224). "
            "Cut-cell has no measurable effect on the rate; the residual cart-vs-polar gap is "
            "~10-20% (NOT a structural deficit -- see T2/T3 normalization audit)."
        ),
        "normalization": (
            "T3: gamma_paper_units = gamma_raw_sim * 2pi/rhobar (rhobar=rho_max=%g). The 2pi is "
            "the cyclic->angular conversion of the diocotron drift clock and applies to the FULL "
            "model AND the reduced ExB (alpha/|Omega|=1/rho_max=1 -> same drift field); the prior "
            "'no 2pi for full' premise was incorrect. Fit windows are the paper windows MAPPED to "
            "sim time (t_sim=2pi/rhobar * t_paper). Residual after 2pi (~8-14%%) is cart ring-edge + "
            "resolution + window roll-off = metrologie PARTIELLE. See docs/T2_NORMALIZATION_AUDIT.md."
            % params.rho_max
        ),
        "adc_cpp_sha": cpp_sha,
        "adc_cases_sha": cases_sha,
        "parameters": params.to_dict(),
        "numerics": {
            "finite_volume": "WENO5-Z + Rusanov",
            "time": (
                "Strang(SSPRK3 + CondensedSchur(theta=0.5))"
                if args.engine == "system-schur"
                else "AMR transport + cell-local backward-Euler source"
            ),
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
                [
                    "finite-volume spatial discretisation",
                    # Gauss re-imposee (solve_fields) en tete de chaque macro-pas et entre les
                    # etages Strang, au lieu de l'evolution sans restart de -Delta phi du papier
                    # (section 5.3). Leve par GaussPolicy.InitialOnly (chantier A5, non encore cable).
                    "Poisson re-solved each macro-step (Gauss reimposed) rather than the paper "
                    "restart-free -Delta-phi evolution",
                ]
                if args.engine == "system-schur"
                else [
                    "finite-volume AMR spatial discretisation",
                    "cell-local IMEX rather than condensed Schur",
                    # Phase B : l'AMR demarre desormais a l'etat de derive (set_conservative_state),
                    # mais via UN SEUL solve Poisson (probe uniforme), pas la relaxation a deux passes
                    # du chemin system-schur. Sur un adc anterieur a set_conservative_state, le seed
                    # retombe sur m=0 (avertissement imprime au run).
                    "single-pass drift initialization (one Poisson solve, not the two-pass relaxation "
                    "of system-schur)",
                    "Cartesian transport outside the circular Poisson wall",
                ]
            ),
        },
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    try:
        import matplotlib
    except ImportError:
        return  # growth_rates.csv + metadata.json deja ecrits ci-dessus ; la figure est optionnelle
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [r[0] for r in rows]
    numeric = [r[1] for r in rows]
    target = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(modes, target, "s-", color="tab:red", label="paper")
    ax.plot(modes, numeric, "o-", color="black", label=args.engine)
    ax.set(
        xlabel="azimuthal mode l",
        ylabel="growth rate gamma",
        xticks=modes,
        title="Diocotron growth rates",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "growth_rates.png"), dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the diocotron driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("system-schur", "amr-imex"), default="system-schur"
    )
    parser.add_argument(
        "--geometry",
        choices=("square", "staircase", "cutcell"),
        default="square",
        help="FV transport sub-domain (system-schur only). 'square' (default) keeps the "
        "historical full-square Cartesian transport, bit-identical. 'staircase'/'cutcell' "
        "call set_disc_domain(L/2, L/2, R, mode=...) which materialises the disc mask AND "
        "is routed into System::step transport (adc_cpp #224). Cut-cell has no measurable "
        "effect on the growth rate (the residual gap is ~10-20% cart-vs-polar, not structural).",
    )
    parser.add_argument(
        "--gauss-policy",
        choices=("restart", "evolve"),
        default="restart",
        help="loi de Gauss du chemin system-schur (System.set_gauss_policy, adc_cpp). "
        "'restart' (defaut) : solve_fields re-resout -Delta phi = rho a chaque appel "
        "(bit-identique a l'historique ; 3 solves Poisson par macro-pas Strang). "
        "'evolve' : seul le premier pas resout le Poisson (phi^0) ; ensuite phi est "
        "porte in-place par l'etage source Schur (Gauss imposee a t=0 seulement), ce "
        "qui supprime les solves elliptiques repetes. Sans effet sur --engine amr-imex.",
    )
    parser.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--gif-frames", type=int, default=80)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="paper does not state a numerical theta; default is the cold limit",
    )
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


def main() -> None:
    """Parse CLI, compile the model, run the requested modes and write output.

    Validates the engine/geometry/MPI combination, applies ``--quick`` presets,
    runs each azimuthal mode and emits per-mode and summary artefacts on rank 0.

    Raises:
        SystemExit: On an invalid engine/geometry/MPI/mode combination or when
            ``--engine amr-imex`` is used without acknowledging the approximation.
    """
    args = parse_args()
    if any(mode not in PAPER_GROWTH_RATES for mode in args.modes):
        raise SystemExit("--modes must be selected from 3, 4, 5")
    if args.engine == "system-schur" and mpi_size() > 1:
        raise SystemExit(
            "system-schur is a single-rank reference; use amr-imex under MPI"
        )
    if (
        args.geometry in ("staircase", "cutcell")
        and args.engine != "system-schur"
    ):
        # set_disc_domain est expose sur System (system-schur), pas sur AmrSystem :
        # le masque disque n'a pas de point d'entree dans le chemin amr-imex.
        raise SystemExit(
            "--geometry staircase/cutcell is only available with --engine system-schur"
        )
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
    # Le suffixe geometrie separe les sorties square vs staircase (chaque geometrie
    # ecrit son propre growth_rates.csv), ce qui permet de mesurer le gain du seul
    # masque escalier. 'square' garde le nom historique pour ne rien casser des
    # sorties existantes ; seul 'staircase' ajoute un suffixe.
    case_name = "hoffart_euler_poisson_dsl_%s" % args.engine.replace("-", "_")
    if args.geometry != "square":
        case_name += "_%s" % args.geometry
    out = case_output_dir(case_name)
    compiled = compile_model(params, args.engine, out)
    if args.compile_only:
        if mpi_rank() == 0:
            print("compiled model:", compiled.so_path)
        return

    results = []
    for mode in args.modes:
        if mpi_rank() == 0:
            print(
                "[%s] mode l=%d, n=%d, t_end=%g, dt=%g"
                % (args.engine, mode, args.n, args.t_end, args.dt)
            )
        result = run_mode(mode, compiled, params, args)
        results.append(result)
        if mpi_rank() == 0:
            target = PAPER_GROWTH_RATES[mode]
            g_paper = gamma_to_paper_units(result.growth_rate, params.rho_max)
            print(
                "  gamma_raw_sim = %s | gamma_paper (x2pi/rhobar) = %s | paper %.3f"
                % (
                    (
                        "n/a"
                        if not np.isfinite(result.growth_rate)
                        else "%.6f" % result.growth_rate
                    ),
                    "n/a" if g_paper is None else "%.4f" % g_paper,
                    target,
                )
            )
            write_mode_outputs(
                result, out, params, args.engine, not args.no_gif
            )

    if mpi_rank() == 0:
        write_summary(results, out, params, args)
        print("outputs:", out)


if __name__ == "__main__":
    main()
