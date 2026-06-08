#!/usr/bin/env python3
"""Cas "tutorial" : la meme physique diocotron ecrite de trois facons equivalentes, mirroir cote
adc_cases du tutoriel Sphinx adc_cpp (docs/sphinx/getting_started/tutorial.md).

Ce cas est un tutoriel executable. Il montre, de bout en bout et sur une seule physique (le
diocotron : une densite electronique scalaire transportee par derive E x B, avec un fond ionique
neutralisant), l'API generale d'adc, sans jamais dependre d'une "classe specialisee" qui cacherait
la composition. On construit le meme modele de trois manieres :

  (1) helper specialise  : adc_cases.models.diocotron(...)  , l'oracle "tout fait" ;
  (2) briques natives    : adc.Model(state, transport, source, elliptic) reconstruit a la main ;
  (3) formules (DSL)     : adc.dsl.Model(...) ou la physique est ecrite en expressions symboliques.

Le helper (1) n'est rien d'autre que la composition (2) (son corps est adc.Model(Scalar, ExB,
NoSource, BackgroundDensity)) : on le prouve par np.array_equal sur la sortie. Et (3) reproduit
exactement les conventions des briques du coeur (ExBVelocity, BackgroundDensity), donc il est lui
aussi bit-identique. La lecon : un "modele nomme" est une composition de briques generiques ; on
peut l'ecrire en briques ou en formules, c'est interchangeable et numeriquement identique.

Sorties (figures/), reproductibles :
  - tutorial_growth.png        : amplitude de la perturbation vs temps (semilog) + carte finale ;
  - tutorial.gif               : evolution de la densite (enroulement en oeil-de-chat) ;
  - tutorial_bricks_vs_dsl.png : etats finals briques | DSL cote a cote + max|ecart| (= 0) ;
  - provenance.json            : SHA, backend retenu, resolution, commande.

Lancement : PYTHONPATH=<build>/python:. python3 tutorial/run.py
La verification CI legere de l'equivalence (sans figures) vit dans tutorial/equivalence.py.
"""

import json
import os
import subprocess
import sys

import numpy as np

import adc
from adc import dsl

# Paquet partage adc_cases : installe (voie nominale, CI), sinon depot mis sur le chemin d'import.
try:
    import adc_cases  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc_cases import models  # noqa: E402  (helper specialise : l'oracle de reference)
from adc_cases.common.checks import relative_drift  # noqa: E402
from adc_cases.common.initial_conditions import band_density  # noqa: E402
from adc_cases.common.io import case_output_dir  # noqa: E402
from adc_cases.common.native import adc_include  # noqa: E402

# Parametres physiques partages par les trois constructions : ils doivent coincider pour que
# l'equivalence soit testable (memes conventions de briques que le helper et le coeur).
B0 = 1.0       # champ magnetique de fond (la derive vaut E x B / B0)
ALPHA = 1.0    # facteur du second membre elliptique : rhs = alpha (n - n_i0)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


# --------------------------------------------------------------------------------------------------
# (2) briques natives : on reconstruit models.diocotron a la main, brique par brique.
# --------------------------------------------------------------------------------------------------
def diocotron_from_bricks(n_i0):
    """Le modele diocotron compose a partir des quatre briques de role du coeur. C'est, mot pour mot,
    le corps de adc_cases.models.diocotron (adc_cases/models.py) : un "modele nomme" n'est qu'une
    composition de briques generiques.

        state     = Scalar()                     -> une seule variable conservative (la densite n) ;
        transport = ExB(B0)                      -> flux d'advection f = n * v, v = (-dy phi, dx phi)/B0 ;
        source    = NoSource()                   -> aucune source par cellule (scalaire pur) ;
        elliptic  = BackgroundDensity(alpha, n0) -> rhs de Poisson = alpha (n - n0), fond neutralisant.
    """
    return adc.Model(
        state=adc.Scalar(),
        transport=adc.ExB(B0=B0),
        source=adc.NoSource(),
        elliptic=adc.BackgroundDensity(alpha=ALPHA, n0=n_i0),
    )


# --------------------------------------------------------------------------------------------------
# (3) formules (DSL) : la meme physique ecrite en expressions symboliques (aucune brique nommee).
# --------------------------------------------------------------------------------------------------
def diocotron_from_dsl(n_i0):
    """Le modele diocotron ecrit en formules (adc.dsl.Model). Chaque ligne reproduit la convention
    exacte de la brique native correspondante, donc la sortie est bit-identique (cf. equivalence).

    On declare : la variable conservative n ; les champs auxiliaires phi / grad phi fournis par le
    solveur ; le flux d'advection E x B ; les valeurs propres (vitesses de derive, pour Rusanov/CFL) ;
    le layout primitif (= conservatif, scalaire) ; et le second membre elliptique."""
    m = dsl.Model("tutorial_diocotron")
    (n,) = m.conservative_vars("n")            # 1 variable conservative, role canonique Density
    m.aux("phi")                               # potentiel (contrat aux ; non lu par le flux)
    grad_x = m.aux("grad_x")                   # composantes du gradient de phi, fournies par le solveur
    grad_y = m.aux("grad_y")
    vx = (-grad_y) / B0                        # derive E x B : v = (-dy phi, dx phi) / B0 (div v = 0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])             # flux physique d'advection f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])              # 1 onde : la vitesse de derive (borne Rusanov / CFL)
    m.primitive_vars(n=n)                      # scalaire : primitif = conservatif (layout [n])
    m.conservative_from([n])                   # inversion triviale prim -> cons
    m.elliptic_rhs(ALPHA * (n - n_i0))         # couplage Poisson : alpha (n - n_i0)
    m.check()                                  # toute variable referencee est-elle declaree ?
    return m


def compile_dsl(model):
    """Compile le modele DSL en preferant le backend "production" (chemin natif zero-copie, cible du
    plan) ; en cas d'echec (p.ex. cle ABI du module incompatible avec include/ en local), on retombe
    sur "aot" (numeriquement identique, host-marshale). Renvoie (CompiledModel, backend_retenu)."""
    include = adc_include()
    so_dir = case_output_dir("tutorial")
    for cand in ("production", "aot"):
        try:
            compiled = model.compile(os.path.join(so_dir, "tutorial_%s.so" % cand),
                                     include, backend=cand)
            return compiled, cand
        except Exception as exc:  # noqa: BLE001 (diagnostic : on essaie le backend suivant)
            print("backend DSL %r indisponible (%s), essai suivant" % (cand, type(exc).__name__))
    raise RuntimeError("aucun backend DSL n'a compile le modele diocotron")


# --------------------------------------------------------------------------------------------------
# Construction du System (commune) + integration avec capture des trames et de l'amplitude.
# --------------------------------------------------------------------------------------------------
def make_system(ne0):
    """System diocotron periodique vide : grille, Poisson et densite identiques pour les trois
    constructions ; seul le bloc (briques via add_block, ou DSL via add_equation) differe."""
    return adc.System(n=ne0.shape[0], L=1.0, periodic=True)


def add_bricks_block(sim, model):
    """Bloc natif : add_block prend un ModelSpec (briques), schema minmod + Rusanov, SSPRK2."""
    sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())


def add_dsl_block(sim, compiled):
    """Bloc DSL : add_equation aiguille le CompiledModel vers le bon adder selon son backend. meme
    schema (minmod + Rusanov) et integrateur que le bloc natif, pour que l'equivalence tienne."""
    sim.add_equation("ne", model=compiled,
                     spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                     time=adc.Explicit())


def perturbation_amplitude(density):
    """Amplitude L2 de la perturbation = ecart-type par rapport a la moyenne le long de x. La bande
    non perturbee est uniforme en x (axis=1) ; ce qui reste mesure la croissance de l'instabilite."""
    base = density.mean(axis=1, keepdims=True)
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


def run_capture(sim, n_steps, every=2):
    """Avance n_steps pas CFL en capturant la densite tous les `every` pas (pour le GIF) et
    l'amplitude de la perturbation a chaque trame. Renvoie (frames, times, amps, final, t, mass)."""
    frames, times, amps = [], [], []
    for k in range(n_steps + 1):
        d = np.asarray(sim.density("ne")).copy()
        if k % every == 0:
            frames.append(d)
            times.append(sim.time())
            amps.append(perturbation_amplitude(d))
        if k < n_steps:
            sim.step_cfl(0.4)
    final = np.asarray(sim.density("ne"))
    return frames, times, amps, final, sim.time(), sim.mass("ne")


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------------------
# Figures du tutoriel (mirroir des assets Sphinx).
# --------------------------------------------------------------------------------------------------
def make_figures(frames, times, amps, final_bricks, final_dsl, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    os.makedirs(FIGDIR, exist_ok=True)

    # (a) croissance : amplitude vs temps (semilog) + carte finale de densite.
    fig, (axc, axm) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axc.semilogy(times, amps, "o-", ms=3)
    axc.set_xlabel("temps"); axc.set_ylabel("amplitude L2 de la perturbation")
    axc.set_title("croissance de l'instabilite diocotron (mode $l=2$)")
    axc.grid(True, which="both", ls=":", alpha=0.5)
    im = axm.imshow(final_bricks, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmax=vmax)
    axm.set_title("densite finale $n_e$"); axm.set_xticks([]); axm.set_yticks([])
    fig.colorbar(im, ax=axm, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "tutorial_growth.png"), dpi=120)
    plt.close(fig)

    # (b) GIF : evolution de la densite (enroulement en oeil-de-chat).
    figg, axg = plt.subplots(figsize=(4.6, 4.6))
    img = axg.imshow(frames[0], origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmax=vmax)
    axg.set_xticks([]); axg.set_yticks([])
    ttl = axg.set_title("")

    def upd(k):
        img.set_data(frames[k])
        ttl.set_text("diocotron : densite $n_e$  (t = %.3f)" % times[k])
        return img, ttl

    anim = animation.FuncAnimation(figg, upd, frames=len(frames), interval=120, blit=False)
    anim.save(os.path.join(FIGDIR, "tutorial.gif"), writer=animation.PillowWriter(fps=8))
    plt.close(figg)

    # (c) briques vs DSL : etats finals cote a cote + max|ecart| (doit etre 0, bit-identique).
    figd, (a0, a1) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    a0.imshow(final_bricks, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmax=vmax)
    a1.imshow(final_dsl, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmax=vmax)
    a0.set_title("briques natives"); a1.set_title("formules (DSL)")
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
    figd.suptitle("meme physique, deux fronts : max|briques - DSL| = %.0e"
                  % float(np.max(np.abs(final_bricks - final_dsl))))
    figd.tight_layout(); figd.savefig(os.path.join(FIGDIR, "tutorial_bricks_vs_dsl.png"), dpi=120)
    plt.close(figd)


def main():
    n, L = 96, 1.0
    n_steps = 60
    ne0 = band_density(n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())   # fond neutralisant = moyenne (solubilite de Poisson periodique)

    print("=== tutorial : diocotron en helper / briques / formules (meme physique) ===")
    print("grille %d x %d, %d pas, CFL 0.4, n_i0 = %.6e" % (n, n, n_steps, n_i0))

    # (1) helper specialise, l'oracle.
    s_helper = make_system(ne0)
    add_bricks_block(s_helper, models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0))
    s_helper.set_poisson(rhs="charge_density", solver="geometric_mg")
    s_helper.set_density("ne", ne0)
    _, _, _, final_helper, _, _ = run_capture(s_helper, n_steps)

    # (2) briques reconstruites a la main, on capture les trames et l'amplitude ici.
    s_bricks = make_system(ne0)
    add_bricks_block(s_bricks, diocotron_from_bricks(n_i0))
    s_bricks.set_poisson(rhs="charge_density", solver="geometric_mg")
    s_bricks.set_density("ne", ne0)
    frames, times, amps, final_bricks, t_b, m_b = run_capture(s_bricks, n_steps)

    # (3) formules (DSL), compile (production -> aot) puis meme systeme.
    compiled, backend = compile_dsl(diocotron_from_dsl(n_i0))
    s_dsl = make_system(ne0)
    add_dsl_block(s_dsl, compiled)
    s_dsl.set_poisson(rhs="charge_density", solver="geometric_mg")
    s_dsl.set_density("ne", ne0)
    _, _, _, final_dsl, t_d, m_d = run_capture(s_dsl, n_steps)

    # --- equivalence : les trois constructions donnent un etat bit-identique ---
    eq_hb = bool(np.array_equal(final_helper, final_bricks))
    eq_bd = bool(np.array_equal(final_bricks, final_dsl))
    print("backend DSL retenu : %r" % backend)
    print("helper == briques : %s   briques == DSL : %s" % (eq_hb, eq_bd))
    print("max|briques - DSL| = %.3e" % float(np.max(np.abs(final_bricks - final_dsl))))
    assert eq_hb, "le helper models.diocotron diverge de la composition de briques (pourtant identiques)"
    assert eq_bd, "le modele DSL n'est pas bit-identique aux briques (une formule diverge du coeur)"

    # --- Invariants physiques (sur les briques ; l'oracle est deja valide ailleurs) ---
    amp0, ampf = amps[0], amps[-1]
    mass_drift = relative_drift(m_b, float(ne0.sum()))
    print("amplitude : %.6e -> %.6e (facteur %.3f)" % (amp0, ampf, ampf / amp0))
    print("derive de masse relative = %.3e" % mass_drift)
    assert ampf > amp0, "l'instabilite n'a pas cru (amp finale <= initiale)"
    assert mass_drift < 1e-6, "masse non conservee (derive %.3e)" % mass_drift

    # --- Figures ---
    vmax = float(max(f.max() for f in frames))
    make_figures(frames, times, amps, final_bricks, final_dsl, vmax)

    adc_cpp_root = os.path.abspath(os.path.join(os.path.dirname(adc.__file__), "..", "..", ".."))
    prov = {
        "script": "tutorial/run.py",
        "command": "python tutorial/run.py",
        "produces": ["tutorial_growth.png", "tutorial.gif", "tutorial_bricks_vs_dsl.png"],
        "demontre": "diocotron construit 3 facons (helper / briques / formules DSL) ; sorties "
                    "bit-identiques (np.array_equal) ; figures de croissance + enroulement KH",
        "backend_dsl": backend,
        "equivalence": {"helper_eq_bricks": eq_hb, "bricks_eq_dsl": eq_bd},
        "adc_cpp_sha": git_sha(adc_cpp_root),
        "adc_cases_sha": git_sha(os.path.dirname(HERE)),
        "resolution": "%dx%d" % (n, n), "n_steps": n_steps, "cfl": 0.4, "mode": 2,
        "python": "%d.%d.%d" % sys.version_info[:3], "adc_module": adc.__file__,
    }
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("OK tutorial (3 fronts equivalents, backend %r, figures dans figures/)" % backend)


if __name__ == "__main__":
    main()
