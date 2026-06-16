#!/usr/bin/env python3
"""Campagne de perf : C++ direct vs Python briques vs Python DSL (cas sur).

AXE FRONTENDS sur le CAS SUR. Le cas sur = Euler compressible PUR, periodique,
bulle de pression lisse de faible amplitude
(rho>0, p>0 garantis). Les TROIS fronts jouent la MEME physique avec les MEMES reglages
numeriques (minmod / rusanov / reconstruction conservative / SSPRK2 / dt FIXE), de sorte que
le seul ecart mesure est le COUT du front, a calcul identique.

  - C++ direct   : binaire bench/frontend_cpp (sous-processus) ; import/model_build/dsl_compile = 0.
  - Python briques : adc.System + add_block(models.euler) + step(dt).
  - Python DSL     : adc.dsl.Model(...).compile(backend="production") + add_equation + step(dt).

Methodologie cold-cache (cf. plan) : CHAQUE front Python tourne dans un SOUS-PROCESSUS FRAIS, pour
que `import adc` soit reellement froid et que le cache DSL soit maitrise. Le DSL est mesure DEUX
fois : froid (so_dir vide -> compilation g++ reelle) et chaud (meme so_dir -> cache touche). On
chronometre par etage : import / model_build / dsl_compile / addblock / state_init / first_step /
warmup / run_loop / diag, plus la boucle chaude (median/p10/p90/cv), `advance(dt,nsteps)` (un seul
appel Python, isole le crossing par pas) et, si Poisson actif, `solve_fields` isole.

Sortie : une ligne JSONL (schema adc_perf_v1) par (front x cache) dans out/safe_euler_periodic/
frontend_compare.jsonl, consommee par perf/plot_frontend.py. Verifie : identite numerique
briques<->DSL (np.allclose serre), invariants (masse, rho>0, p>0, pas de NaN) sur les trois fronts.

Lancement (depuis adc_cases, avec le build sur le PYTHONPATH) :
  PYTHONPATH=<adc_cpp>/build-master/python:. python3 perf/frontend_compare.py \
      --n 256 --steps 50 --warmup 5 --poisson off \
      --cpp-bin <adc_cpp>/build-bench-serie/bin/frontend_cpp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def _bootstrap() -> None:
    """Rend adc_cases importable quand le script est lance directement (sans installation)."""
    try:
        import adc_cases  # noqa: F401
    except ImportError:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )


def _pct(samples_ms) -> tuple[float, float, float, float]:
    """Resume des echantillons en (mediane, p10, p90, coefficient de variation)."""
    a = np.sort(np.asarray(samples_ms, dtype=float))
    med, p10, p90 = (float(np.percentile(a, q)) for q in (50, 10, 90))
    mean = float(a.mean())
    cv = float(a.std() / mean) if mean > 0 else 0.0
    return med, p10, p90, cv


def _build_sim(adc, sc, front: str, n: int, poisson: str, model, compiled):
    """Construit un System et y branche le bloc (briques ou DSL), Poisson optionnel.

    Branche `model` via add_block (front "bricks") ou `compiled` via
    add_equation (front "dsl"), puis active Poisson si poisson == "on".
    """
    sim = adc.System(n=n, L=sc.L, periodic=True)
    if front == "bricks":
        sim.add_block(
            "gas", model=model, spatial=sc.spatial_bricks(), time=adc.Explicit()
        )
    else:
        sim.add_equation(
            "gas", model=compiled, spatial=sc.spatial_dsl(), time=adc.Explicit()
        )
    if poisson == "on":
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    return sim


# --------------------------------------------------------------------------------------------------
# WORKER : un seul front, dans un sous-processus FRAIS (import froid, cache DSL maitrise).
# --------------------------------------------------------------------------------------------------
def run_front(
    front: str,
    n: int,
    steps: int,
    warmup: int,
    poisson: str,
    dsl_cache: str,
    state_out: str | None,
) -> dict:
    """Mesure un seul front dans le processus courant et renvoie son record perf.

    A appeler dans un sous-processus FRAIS pour que `import adc` soit froid et
    que le cache DSL soit maitrise. Chronometre chaque etage (import, model_build,
    dsl_compile, addblock, state_init, first_step, warmup, run_loop, diag), la
    boucle chaude, `advance` et, si Poisson actif, `solve_fields` ; verifie les
    invariants finals.

    Args:
        front: Front mesure, "bricks" ou "dsl".
        n: Taille de grille (n x n).
        steps: Nombre de pas chronometres dans la boucle chaude.
        warmup: Nombre de pas de chauffe, hors mesure.
        poisson: "on" pour activer le solveur de Poisson, "off" sinon.
        dsl_cache: Etiquette du cache DSL ("cold" / "warm" / "n/a"), reportee.
        state_out: Chemin .npy ou sauver l'etat final, ou None.

    Returns:
        Record JSON-serialisable (schema adc_perf_v1), enrichi de provenance.

    Raises:
        RuntimeError: Si aucun backend DSL ne compile le modele (front "dsl").
    """
    t = time.perf_counter()
    import adc  # IMPORT FROID (premier import du processus) -> mesure reelle du chargement de _adc
    from adc_cases.common import provenance as prov_mod
    from adc_cases.common import safe_euler as sc

    t_import = time.perf_counter() - t

    dt = sc.dt(n)
    U = sc.ic(n)
    stages = {"import": t_import, "dsl_compile": 0.0}
    dsl_backend = None
    compiled = None
    model = None

    # model_build
    t = time.perf_counter()
    model = sc.bricks_model() if front == "bricks" else sc.dsl_model()
    stages["model_build"] = time.perf_counter() - t

    # dsl_compile (DSL only) : prefere "production" (natif zero-copie), repli "aot". so_path=None
    # ENGAGE le cache hors source (adc_cache_dir(), keye model_hash+abi_key) : COLD = cache vide
    # ($ADC_CACHE_DIR frais) -> recompilation g++ ; WARM = meme $ADC_CACHE_DIR peuple -> HIT, pas de
    # recompilation. (Passer un so_path explicite forcerait une recompilation a chaque fois.)
    if front == "dsl":
        from adc_cases.common.native import adc_include

        include = adc_include()
        t = time.perf_counter()
        last = None
        for cand in ("production", "aot"):
            try:
                compiled = model.compile(None, include, backend=cand)
                dsl_backend = cand
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                compiled = None
        stages["dsl_compile"] = time.perf_counter() - t
        if compiled is None:
            raise RuntimeError(
                "aucun backend DSL n'a compile le modele Euler sur (%s)" % last
            )

    # addblock / add_equation (+ Poisson)
    t = time.perf_counter()
    sim = _build_sim(adc, sc, front, n, poisson, model, compiled)
    stages["addblock"] = time.perf_counter() - t

    # state_init
    t = time.perf_counter()
    sim.set_state("gas", U.reshape(-1).tolist())
    stages["state_init"] = time.perf_counter() - t

    # first_step (inclut premier kernel / premier V-cycle)
    t = time.perf_counter()
    sim.step(dt)
    stages["first_step"] = time.perf_counter() - t

    # warmup (hors mesure chaude)
    t = time.perf_counter()
    for _ in range(warmup):
        sim.step(dt)
    stages["warmup"] = time.perf_counter() - t

    # run_loop : step(dt) x steps, echantillon ms/pas (crossing Python par pas)
    samples = []
    t0 = time.perf_counter()
    for _ in range(steps):
        s0 = time.perf_counter()
        sim.step(dt)
        samples.append((time.perf_counter() - s0) * 1e3)
    stages["run_loop"] = time.perf_counter() - t0

    # diag : invariants finals
    t = time.perf_counter()
    final = np.asarray(sim.get_state("gas"), dtype=float).reshape(4, n, n)
    rho = final[0]
    p = sc.pressure(final)
    mass, rho_min, p_min = float(rho.sum()), float(rho.min()), float(p.min())
    has_nan = not bool(np.isfinite(final).all())
    stages["diag"] = time.perf_counter() - t

    med, p10, p90, cv = _pct(samples)

    # SECONDAIRE : advance(dt, steps) en UN appel Python -> isole le cout du crossing par pas.
    sim2 = _build_sim(adc, sc, front, n, poisson, model, compiled)
    sim2.set_state("gas", U.reshape(-1).tolist())
    for _ in range(warmup):
        sim2.step(dt)
    ta = time.perf_counter()
    sim2.advance(dt, steps)
    advance_ms = (time.perf_counter() - ta) * 1e3 / steps

    # SECONDAIRE : solve_fields isole (seulement significatif si Poisson actif).
    solve_fields_ms = 0.0
    if poisson == "on":
        sf = []
        for _ in range(min(steps, 10)):
            s0 = time.perf_counter()
            sim.solve_fields()
            sf.append((time.perf_counter() - s0) * 1e3)
        solve_fields_ms = float(np.median(sf))

    total_cold = float(sum(stages.values()))
    cells_per_s = (n * n) / (med / 1e3) if med > 0 else 0.0

    rec = prov_mod.provenance(
        {
            "schema": "adc_perf_v1",
            "front": "python_%s" % front,
            "backend": "serial",
            "ranks": 1,
            "gpus": 0,
            "nx": n,
            "ny": n,
            "boxes": 1,
            "max_grid": n,
            "workload": sc.WORKLOAD,
            "limiter": sc.LIMITER,
            "flux": sc.FLUX,
            "recon": sc.RECON,
            "time": sc.TIMEINT,
            "poisson": poisson,
            "dt": dt,
            "warmup": warmup,
            "steps": steps,
            "dsl_backend": dsl_backend,
            "dsl_cache": dsl_cache,
            "stages": stages,
            "total_cold_user_s": total_cold,
            "hot_ms_per_step": {
                "median": med,
                "p10": p10,
                "p90": p90,
                "cv": cv,
            },
            "advance_ms_per_step": advance_ms,
            "phases_ms_per_step": {"solve_fields": solve_fields_ms},
            "cells_per_s": cells_per_s,
            "invariants": {
                "mass": mass,
                "rho_min": rho_min,
                "p_min": p_min,
                "nan": has_nan,
            },
        }
    )
    if state_out:
        np.save(state_out, final)
    return rec


# --------------------------------------------------------------------------------------------------
# ORCHESTRATEUR : lance le binaire C++ + un sous-processus FRAIS par front Python ; agrege & verifie.
# --------------------------------------------------------------------------------------------------
def _spawn_worker(
    front: str,
    n: int,
    steps: int,
    warmup: int,
    poisson: str,
    dsl_cache: str,
    json_out: str,
    state_out: str | None,
    cache_dir: str | None = None,
) -> dict:
    """Relance ce script en mode worker (--front) dans un sous-processus frais.

    `cache_dir` est exporte en $ADC_CACHE_DIR (cache DSL : repertoire frais =
    compile froid, repertoire reutilise = chaud). Le worker ecrit son record
    dans `json_out`, qu'on relit et renvoie.

    Returns:
        Le record JSON ecrit par le worker.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--front",
        front,
        "--n",
        str(n),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--poisson",
        poisson,
        "--dsl-cache",
        dsl_cache,
        "--json-out",
        json_out,
    ]
    if state_out:
        cmd += ["--state-out", state_out]
    env = dict(os.environ)
    if cache_dir:
        env["ADC_CACHE_DIR"] = cache_dir
    subprocess.run(
        cmd, check=True, env=env
    )  # diagnostics -> stdout/stderr ; record -> json_out
    with open(json_out) as fh:
        return json.load(fh)


def orchestrate(args: argparse.Namespace) -> None:
    """Lance les quatre fronts, verifie les invariants et ecrit le JSONL.

    Fronts : C++ direct (binaire), briques, DSL froid puis DSL chaud (chacun
    Python dans un sous-processus frais). Verifie l'identite numerique
    briques <-> DSL, les invariants par front et la coherence de masse C++ vs
    Python, puis ecrit out/safe_euler_periodic/frontend_compare.jsonl et un
    resume.
    """
    _bootstrap()
    from adc_cases.common.io import case_output_dir
    from adc_cases.common import provenance as prov_mod

    out_dir = case_output_dir("safe_euler_periodic")
    jsonl_path = os.path.join(out_dir, "frontend_compare.jsonl")
    tmp = tempfile.mkdtemp(prefix="safe_euler_perf_")
    cache_dir = os.path.join(
        tmp, "dsl_cache"
    )  # $ADC_CACHE_DIR : vide au COLD, peuple au WARM
    records = []

    # 1. Front C++ direct (binaire). Sa ligne ne porte que adc_cpp_sha : on complete avec adc_cases.
    if args.cpp_bin and os.path.exists(args.cpp_bin):
        line = (
            subprocess.check_output(
                [
                    args.cpp_bin,
                    "--n",
                    str(args.n),
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                    "--poisson",
                    args.poisson,
                    "--backend",
                    "serial",
                    "--machine",
                    prov_mod.provenance()["machine"],
                ],
                text=True,
            )
            .strip()
            .splitlines()[-1]
        )
        rec_cpp = json.loads(line)
        cprov = prov_mod.provenance()
        rec_cpp.setdefault("adc_cases_sha", cprov["adc_cases_sha"])
        rec_cpp.setdefault("adc_cases_branch", cprov["adc_cases_branch"])
        records.append(rec_cpp)
    else:
        print(
            "[frontend_compare] front C++ saute (binaire absent : %r)"
            % args.cpp_bin
        )

    # 2. Briques (sous-processus frais ; cache DSL non concerne)
    sb = os.path.join(tmp, "state_bricks.npy")
    rec_b = _spawn_worker(
        "bricks",
        args.n,
        args.steps,
        args.warmup,
        args.poisson,
        "n/a",
        os.path.join(tmp, "b.json"),
        sb,
    )
    records.append(rec_b)

    # 3. DSL froid ($ADC_CACHE_DIR vide -> recompile) puis chaud (meme cache_dir, peuple -> HIT)
    sd = os.path.join(tmp, "state_dsl.npy")
    rec_dc = _spawn_worker(
        "dsl",
        args.n,
        args.steps,
        args.warmup,
        args.poisson,
        "cold",
        os.path.join(tmp, "dc.json"),
        sd,
        cache_dir=cache_dir,
    )
    rec_dw = _spawn_worker(
        "dsl",
        args.n,
        args.steps,
        args.warmup,
        args.poisson,
        "warm",
        os.path.join(tmp, "dw.json"),
        None,
        cache_dir=cache_dir,
    )
    records += [rec_dc, rec_dw]

    # --- Verifications d'acceptation -------------------------------------------------------------
    # (a) identite numerique briques <-> DSL (etat complet), comme diocotron_dsl / two_species_dsl.
    state_b, state_d = np.load(sb), np.load(sd)
    max_abs = float(np.max(np.abs(state_b - state_d)))
    identical = bool(np.array_equal(state_b, state_d))
    print("\n=== identite numerique briques <-> DSL ===")
    print(
        "max|briques - DSL| = %.3e   bit-identique = %s" % (max_abs, identical)
    )
    assert max_abs < args.atol, (
        "briques et DSL divergent (max|d|=%.3e >= atol=%.1e) : reglages numeriques desynchronises"
        % (max_abs, args.atol)
    )

    # (b) invariants sur tous les fronts ; (c) cross-check masse C++ vs Python.
    print("\n=== invariants par front ===")
    py_mass = None
    for r in records:
        inv = r.get("invariants", {})
        if "nan" in inv:
            assert not inv["nan"], "%s : etat non fini (NaN/Inf)" % r["front"]
            assert inv["rho_min"] > 0, "%s : rho <= 0 (min=%.3e)" % (
                r["front"],
                inv["rho_min"],
            )
            assert inv["p_min"] > 0, "%s : p <= 0 (min=%.3e)" % (
                r["front"],
                inv["p_min"],
            )
            print(
                "  %-16s mass=%.10e rho_min=%.4e p_min=%.4e cv=%.3f"
                % (
                    r["front"],
                    inv["mass"],
                    inv["rho_min"],
                    inv["p_min"],
                    r["hot_ms_per_step"]["cv"],
                )
            )
            if r["front"].startswith("python_") and py_mass is None:
                py_mass = inv["mass"]
    # cross-check : masse C++ ~ masse Python (memes IC, meme schema).
    for r in records:
        if r["front"] == "cpp" and py_mass is not None:
            dm = abs(r["invariants"]["mass"] - py_mass) / max(
                abs(py_mass), 1e-30
            )
            print(
                "  cross-check masse C++ vs Python : derive relative = %.3e"
                % dm
            )
            assert dm < 1e-6, (
                "masse C++ vs Python incoherente (derive %.3e)" % dm
            )

    # --- ecriture JSONL + resume -----------------------------------------------------------------
    with open(jsonl_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(
        "\n=== resume (cas sur Euler, n=%d, %d pas, poisson=%s) ==="
        % (args.n, args.steps, args.poisson)
    )
    print(
        "%-18s %12s %12s %12s %12s"
        % ("front", "cold_user_s", "hot_ms/pas", "advance_ms", "cells/s")
    )
    cpp_hot = None
    for r in records:
        tag = r["front"] + (
            "/" + r.get("dsl_cache", "") if r["front"] == "python_dsl" else ""
        )
        hot = r["hot_ms_per_step"]["median"]
        print(
            "%-18s %12.4f %12.4f %12.4f %12.3e"
            % (
                tag,
                r.get("total_cold_user_s", 0.0),
                hot,
                r.get("advance_ms_per_step", float("nan")),
                r.get("cells_per_s", 0.0),
            )
        )
        if r["front"] == "cpp":
            cpp_hot = hot
    if cpp_hot:
        print("\nratio hot ms/pas (C++ = 1.0) :")
        for r in records:
            if r["front"].startswith("python"):
                tag = r["front"] + (
                    "/" + r.get("dsl_cache", "")
                    if r["front"] == "python_dsl"
                    else ""
                )
                print(
                    "  %-18s %.2fx"
                    % (tag, r["hot_ms_per_step"]["median"] / cpp_hot)
                )
    print("\nJSONL ecrit : %s" % jsonl_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Comparaison de fronts C++ / briques / DSL (cas Euler sur)"
    )
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--poisson", choices=["off", "on"], default="off")
    ap.add_argument(
        "--cpp-bin",
        default=None,
        help="chemin du binaire bench/frontend_cpp (front C++)",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="tolerance identite numerique briques/DSL",
    )
    # mode worker (interne) : le cache DSL cold/warm est pilote par $ADC_CACHE_DIR (pose par l'orchestrateur).
    ap.add_argument("--front", choices=["bricks", "dsl"], default=None)
    ap.add_argument("--dsl-cache", default="n/a")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--state-out", default=None)
    args = ap.parse_args()

    if args.front:  # mode WORKER : un seul front, record -> --json-out
        _bootstrap()
        rec = run_front(
            args.front,
            args.n,
            args.steps,
            args.warmup,
            args.poisson,
            args.dsl_cache,
            args.state_out,
        )
        with open(args.json_out, "w") as fh:
            json.dump(rec, fh)
        print(
            "[worker %s/%s] cold=%.4fs hot_med=%.4fms"
            % (
                args.front,
                args.dsl_cache,
                rec["total_cold_user_s"],
                rec["hot_ms_per_step"]["median"],
            )
        )
    else:  # mode ORCHESTRATEUR
        orchestrate(args)


if __name__ == "__main__":
    main()
