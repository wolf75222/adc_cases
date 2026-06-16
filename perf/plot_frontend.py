#!/usr/bin/env python3
"""Trace les figures de la campagne de perf a partir des JSONL (frontend + scaling). AUCUNE physique.

Figures FRONTEND (depuis frontend_compare.jsonl) :
  - barres empilees du TEMPS cold-cache par etage et par front (import / model_build / dsl_compile /
    addblock / state_init / first_step / warmup / run_loop / diag) ;
  - hot ms/pas par front avec barres d'erreur p10-p90 ;
  - `step`-loop (hot median) vs `advance` par front (isole le crossing Python par pas) ;
  - ratio du hot ms/pas, C++ = 1.0 (briques / DSL froid / DSL chaud).

Figures SCALING (depuis un scaling.jsonl optionnel, plusieurs lignes variant ranks/threads/n) :
  - strong : speedup et efficacite vs ressources ; weak : efficacite ; debit cells/s.

GARDE-FOU d'acceptation : une figure ne MELANGE JAMAIS deux (adc_cpp_sha, adc_cpp_branch) ; sinon on
leve. Chaque figure porte un pied de page de provenance (SHA des deux depots, machine).

Lancement : python3 perf/plot_frontend.py [--frontend out/safe_euler_periodic/frontend_compare.jsonl]
            [--scaling out/safe_euler_periodic/scaling.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _bootstrap() -> None:
    try:
        import adc_cases  # noqa: F401
    except ImportError:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )


def read_jsonl(path: str) -> list:
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def assert_single_build(recs: list, what: str) -> None:
    """GARDE-FOU : tous les enregistrements doivent partager (adc_cpp_sha, adc_cpp_branch)."""
    keys = {(r.get("adc_cpp_sha"), r.get("adc_cpp_branch")) for r in recs}
    if len(keys) > 1:
        raise SystemExit(
            "REFUS : %s melange plusieurs builds adc_cpp (master vs PR ?) : %s. "
            "Un graphe ne doit jamais melanger deux SHA/branches."
            % (what, sorted(keys))
        )


def _prov_footer(recs: list) -> str:
    r = recs[0]
    return "adc_cpp %s@%s | adc_cases %s@%s | %s" % (
        str(r.get("adc_cpp_branch")),
        str(r.get("adc_cpp_sha"))[:10],
        str(r.get("adc_cases_branch")),
        str(r.get("adc_cases_sha"))[:10],
        str(r.get("machine")),
    )


STAGE_ORDER = [
    "import",
    "model_build",
    "dsl_compile",
    "addblock",
    "state_init",
    "first_step",
    "warmup",
    "run_loop",
    "diag",
]


def _front_label(r: dict) -> str:
    f = r["front"]
    if f == "python_dsl":
        return "DSL/%s" % r.get("dsl_cache", "?")
    return {"cpp": "C++", "python_bricks": "briques"}.get(f, f)


def plot_frontend(path: str, figdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    recs = read_jsonl(path)
    recs = [
        r
        for r in recs
        if r.get("schema") == "adc_perf_v1" and "hot_ms_per_step" in r
    ]
    if not recs:
        raise SystemExit(
            "aucun enregistrement frontend exploitable dans %s" % path
        )
    assert_single_build(recs, "frontend_compare")
    os.makedirs(figdir, exist_ok=True)
    labels = [_front_label(r) for r in recs]
    footer = _prov_footer(recs)
    poisson = recs[0].get("poisson", "?")

    # (a) barres empilees du temps cold-cache par etage.
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bottoms = np.zeros(len(recs))
    cmap = plt.get_cmap("tab10")
    for k, stage in enumerate(STAGE_ORDER):
        vals = np.array([r.get("stages", {}).get(stage, 0.0) for r in recs])
        if vals.sum() <= 0:
            continue
        ax.bar(labels, vals, bottom=bottoms, label=stage, color=cmap(k % 10))
        bottoms += vals
    ax.set_ylabel("temps utilisateur cold-cache (s)")
    ax.set_title(
        "Decomposition du temps cold-cache par front (Euler sur, poisson=%s)"
        % poisson
    )
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.text(0.01, 0.005, footer, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(figdir, "frontend_cold_stages.png"), dpi=130)
    plt.close(fig)

    # (b) hot ms/pas avec barres d'erreur p10-p90.
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    med = np.array([r["hot_ms_per_step"]["median"] for r in recs])
    lo = np.array(
        [
            r["hot_ms_per_step"]["median"] - r["hot_ms_per_step"]["p10"]
            for r in recs
        ]
    )
    hi = np.array(
        [
            r["hot_ms_per_step"]["p90"] - r["hot_ms_per_step"]["median"]
            for r in recs
        ]
    )
    ax.bar(labels, med, yerr=[lo, hi], capsize=5, color="steelblue")
    for i, r in enumerate(recs):
        ax.text(
            i,
            med[i],
            "cv=%.1f%%" % (100 * r["hot_ms_per_step"]["cv"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("hot ms / pas (median, p10-p90)")
    ax.set_title(
        "Cout du pas chaud par front (Euler sur, poisson=%s)" % poisson
    )
    fig.text(0.01, 0.005, footer, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(figdir, "frontend_hot_ms.png"), dpi=130)
    plt.close(fig)

    # (c) step-loop vs advance (isole le crossing Python par pas).
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(recs))
    w = 0.38
    step_loop = np.array([r["hot_ms_per_step"]["median"] for r in recs])
    advance = np.array(
        [r.get("advance_ms_per_step", float("nan")) for r in recs]
    )
    ax.bar(x - w / 2, step_loop, w, label="step(dt) en boucle Python")
    ax.bar(x + w / 2, advance, w, label="advance(dt, nsteps) un appel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms / pas")
    ax.set_title("step-loop vs advance : cout du crossing Python par pas")
    ax.legend()
    fig.text(0.01, 0.005, footer, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(figdir, "frontend_step_vs_advance.png"), dpi=130)
    plt.close(fig)

    # (d) ratio du hot ms/pas (C++ = 1.0).
    cpp = next((r for r in recs if r["front"] == "cpp"), None)
    if cpp:
        base = cpp["hot_ms_per_step"]["median"]
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ratios = [r["hot_ms_per_step"]["median"] / base for r in recs]
        ax.bar(labels, ratios, color="indianred")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        for i, v in enumerate(ratios):
            ax.text(i, v, "%.2fx" % v, ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("ratio hot ms/pas (C++ = 1.0)")
        ax.set_title(
            "Surcout du front vs C++ direct (Euler sur, poisson=%s)" % poisson
        )
        fig.text(0.01, 0.005, footer, fontsize=7, color="gray")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(figdir, "frontend_ratio.png"), dpi=130)
        plt.close(fig)
    print("figures frontend ecrites dans %s" % figdir)


def plot_scaling(path: str, figdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    recs = [
        r
        for r in read_jsonl(path)
        if r.get("front") == "cpp_scaling"
        and r.get("status") != "not_implemented"
    ]
    if not recs:
        print("aucun enregistrement scaling exploitable dans %s (saute)" % path)
        return
    assert_single_build(recs, "scaling")
    os.makedirs(figdir, exist_ok=True)
    footer = _prov_footer(recs)

    # ressources = ranks * threads (CPU) ou gpus (si > 0).
    def units(r):
        g = r.get("gpus", 0)
        return g if g and g > 0 else r.get("ranks", 1) * r.get("threads", 1)

    # On SEPARE par backend (kokkos-omp / kokkos-cuda / mpi-* ...) : ne JAMAIS melanger CPU et GPU
    # dans une meme courbe de scaling. Une courbe = un (backend, workload, scaling) avec >=2 points
    # a unites DISTINCTES (sinon balayage degenere, p.ex. mono-GPU a unites=1 : montre en table).
    combos = sorted(
        {
            (r.get("backend", "?"), r["workload"], r.get("scaling", "strong"))
            for r in recs
        }
    )
    for backend, workload, scaling in combos:
        sub = sorted(
            [
                r
                for r in recs
                if r.get("backend") == backend
                and r["workload"] == workload
                and r.get("scaling", "strong") == scaling
            ],
            key=units,
        )
        if len(sub) < 2 or len({units(r) for r in sub}) < 2:
            continue
        if True:
            u = np.array([units(r) for r in sub], dtype=float)
            t = np.array([r["hot_ms_per_step"]["median"] for r in sub])
            cps = np.array([r.get("cells_per_s", 0.0) for r in sub])
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
            if scaling == "strong":
                speedup = t[0] / t
                axes[0].plot(u, speedup, "o-", label="mesure")
                axes[0].plot(u, u / u[0], "k--", lw=0.8, label="ideal")
                axes[0].set_title("strong speedup")
                axes[0].set_xlabel("unites (ranks x threads / GPUs)")
                axes[0].legend()
                axes[1].plot(u, speedup / (u / u[0]), "o-")
                axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
                axes[1].set_title("strong efficiency")
                axes[1].set_xlabel("unites")
            else:
                eff = cps / cps[0]
                axes[0].plot(u, cps, "o-")
                axes[0].set_title("weak : debit cells/s")
                axes[0].set_xlabel("unites")
                axes[1].plot(u, eff, "o-")
                axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
                axes[1].set_title("weak efficiency")
                axes[1].set_xlabel("unites")
            axes[2].plot(u, cps, "o-", color="seagreen")
            axes[2].set_title("debit cells/s")
            axes[2].set_xlabel("unites")
            for a in axes:
                a.grid(True, ls=":", alpha=0.5)
            fig.suptitle("Scaling %s -- %s (%s)" % (scaling, workload, backend))
            fig.text(0.01, 0.005, footer, fontsize=7, color="gray")
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            out = os.path.join(
                figdir, "scaling_%s_%s_%s.png" % (scaling, workload, backend)
            )
            fig.savefig(out, dpi=130)
            plt.close(fig)
            print("figure scaling ecrite : %s" % out)


def main() -> None:
    _bootstrap()
    from adc_cases.common.io import case_output_dir

    default_dir = case_output_dir("safe_euler_periodic")
    ap = argparse.ArgumentParser(
        description="Figures de la campagne de perf (frontend + scaling)"
    )
    ap.add_argument(
        "--frontend",
        default=os.path.join(default_dir, "frontend_compare.jsonl"),
    )
    ap.add_argument("--scaling", default=None)
    ap.add_argument("--figdir", default=os.path.join(default_dir, "figures"))
    args = ap.parse_args()
    if os.path.exists(args.frontend):
        plot_frontend(args.frontend, args.figdir)
    else:
        print("pas de JSONL frontend (%s) -- saute" % args.frontend)
    if args.scaling and os.path.exists(args.scaling):
        plot_scaling(args.scaling, args.figdir)


if __name__ == "__main__":
    main()
