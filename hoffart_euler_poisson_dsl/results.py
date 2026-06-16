"""Emetteur d'enregistrements de mesure pour le modele COMPLET system-schur (T3).

But
---
Rendre la mesure du modele complet HONNETE et PRE-ENREGISTREE :

1. Pre-enregistrer et VERIFIER les fenetres de fit verbatim du papier
   (Fig. 5.4, arXiv:2510.11808) : l=3 [0.40,0.70], l=4 [0.60,0.75],
   l=5 [1.15,1.35]. Ces fenetres sont EN UNITES PAPIER (temps diocotron T_d).
   Aucune fenetre adaptative n'est introduite (cf. ``verify_paper_windows``) ;
   mais le fit doit se faire sur le temps de SIMULATION MAPPE (voir point 2).

2. CORRECTION T3 (audit de normalisation, juin 2026 ; cf. ``docs/T2_NORMALIZATION_AUDIT.md``,
   ``docs/RESULTS_SYSTEM_SCHUR.md`` section 8, workflow de verif adversariale 4 lentilles).
   La PREMISSE PRECEDENTE de ce module -- "la pente brute du modele complet est
   DIRECTEMENT comparable a 0.772/0.911/0.683, SANS facteur 2 pi ; le 2 pi
   appartient UNIQUEMENT au chemin reduit" -- est INCORRECTE. Mesure :
     * le run complet (alpha=omega=1e12) advecte rho avec EXACTEMENT le champ de
       vitesse du reduit ExB normalise, car alpha/|Omega| = 1/rho_max = 1 (le
       1e12 se simplifie) ; full == reduit a ~2 % en fenetre etablie sur l=3/4/5 ;
     * le solveur NUMERIQUE (complet OU reduit) mesure dans l'horloge ExB-NATURELLE ;
       le papier rapporte en horloge omega_d CYCLIQUE (T_d = 1 = un tour = 2 pi rad) ;
     * la conversion est donc gamma_paper = gamma_raw_sim * (2 pi / rhobar), avec
       rhobar = rho_max = 1. Le 2 pi est la conversion CYCLIQUE -> ANGULAIRE de
       l'horloge de derive (verifiee MODE-INDEPENDANTE a < 0.5 % contre la valeur
       propre analytique Petri : diag/petri_eigenvalue.py reproduit les cibles avec
       Wd = 2 pi omega_d, et donne cibles/6.2832 avec Wd = omega_d = 1). Il
       s'applique IDENTIQUEMENT au complet ET au reduit (PAS "reduit seulement").

3. On REPORTE LES DEUX, cote a cote, sans en ecraser une :
       gamma_raw_sim     = pente BRUTE dans l'horloge sim (reproductibilite) ;
       gamma_paper_units = gamma_raw_sim * (2 pi / rhobar)  (comparaison papier).
   Le moteur reste distingue (full-system-schur vs reduced-ExB) MAIS les deux
   portent le meme 2 pi. ``err_pct`` compare gamma_paper_units a la cible papier.

   METROLOGIE PARTIELLE (PAS pure) : apres le 2 pi, un residu REEL subsiste
   (full l=3 -14 %, l=4 -8 %, l=5 +13 % ; reduit mappe ~3.5-9.5 %) = bord d'anneau
   cartesien + resolution finie + roll-off de fenetre (le slope local n'a PAS de
   plateau scale-free : il culmine a t~3.2-4.2 puis decline ~13 % -- lissage WENO5,
   PAS une saturation non-lineaire). l=5 est sensible a la fenetre (+/-27-29 %) :
   ne PAS le citer comme support propre. Le 2 pi est exact ; le residu est de la
   metrologie grille/fenetre, pas un facteur d'horloge.

4. Emettre un enregistrement par run (CSV + JSON) capturant : SHA adc_cpp, SHA
   adc_cases, backend, n, dt, splitting (Lie/Strang), schur(theta), fenetre papier,
   fenetre sim mappee, rhobar, gamma_raw_sim, gamma_paper_units, gamma_paper,
   err_pct. C'est la graine de la table de validation de la Phase 2.

Ce module NE fabrique AUCUN nombre : il mesure et enregistre ce qu'un run
produit (gamma_raw_sim vient du fit ; PENDING si le run n'a pas tourne).

Ce module est PUR PYTHON (aucune dependance au binding ``adc`` ni a un build) :
son auto-test ``python results.py`` tourne en CI sans compiler le coeur.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess

# Fenetres de fit verbatim du papier (Fig. 5.4, arXiv:2510.11808), EN UNITES PAPIER
# (temps diocotron T_d). Dupliquees ici en tant que VERITE PRE-ENREGISTREE :
# ``verify_paper_windows`` confronte les fenetres papier utilisees par run.py
# (model.PAPER_FIT_WINDOWS) a celles-ci et leve si elles divergent. Verrou contre toute
# fenetre adaptative. ATTENTION (T3) : ces fenetres sont en temps PAPIER ; le fit se fait
# sur le temps SIMULATION MAPPE par ``paper_to_sim_time_window`` (horloge ExB-naturelle).
PAPER_FIT_WINDOWS_VERBATIM = {3: (0.40, 0.70), 4: (0.60, 0.75), 5: (1.15, 1.35)}

# Etiquettes de moteur. La cle est l'``--engine`` du runner ; la valeur est le label
# explicite porte par l'enregistrement.
ENGINE_LABELS = {
    "system-schur": "full-system-schur",
    "amr-imex": "amr-imex-experimental",
    # Modele COMPLET porte sur la GRILLE POLAIRE (anneau resolu). NB (T3) : son docstring
    # "pente brute, aucun facteur" est de la MEME classe logique que la premisse corrigee
    # ici (le 2 pi s'applique a tout solveur en horloge ExB-naturelle) ; mais la repro
    # quantitative du polaire complet n'est PAS etablie (VOIE 1 #236 diverge), a traiter
    # separement.
    "polar-schur": "full-polar-schur",
}

REDUCED_EXB_LABEL = "reduced-ExB"


def paper_to_sim_time_window(
    window_paper: tuple[float, float], rhobar: float = 1.0
) -> tuple[float, float]:
    """Mappe une fenetre de fit du temps PAPIER (T_d) vers le temps de SIMULATION.

    Le solveur tourne dans l'horloge ExB-NATURELLE ; le papier rapporte dans l'horloge
    omega_d CYCLIQUE (T_d = 1 = un tour = 2 pi radians). La conversion est
    ``t_sim = (2 pi / rhobar) * t_paper`` (rhobar = rho_max = 1). Appliquer la fenetre
    papier BRUTE au temps sim tomberait dans le transitoire (l'artefact -95 % d'origine) ;
    il faut fitter sur ``t_sim in [2 pi lo, 2 pi hi]``. Voir ``docs/T2_NORMALIZATION_AUDIT.md``.
    """
    scale = 2.0 * math.pi / rhobar
    lo, hi = window_paper
    return (float(lo) * scale, float(hi) * scale)


def gamma_to_paper_units(
    gamma_raw_sim: float | None, rhobar: float = 1.0
) -> float | None:
    """gamma_paper = gamma_raw_sim * (2 pi / rhobar) : conversion horloge sim -> papier.

    Le 2 pi est la conversion CYCLIQUE -> ANGULAIRE de l'horloge de derive diocotron,
    EXACTE et MODE-INDEPENDANTE (verifiee < 0.5 % contre la valeur propre analytique
    Petri : diag/petri_eigenvalue.py reproduit les cibles avec Wd = 2 pi omega_d). Il
    s'applique IDENTIQUEMENT au modele complet ET au reduit ExB (alpha/|Omega| =
    1/rho_max = 1 -> meme champ de derive). renvoie None si gamma_raw_sim non fini.
    """
    if gamma_raw_sim is None:
        return None
    try:
        g = float(gamma_raw_sim)
    except (TypeError, ValueError):
        return None
    if g != g or g in (float("inf"), float("-inf")):
        return None
    return g * (2.0 * math.pi / rhobar)


def engine_label(engine: str) -> str:
    """Label explicite du moteur ; refuse tout moteur inconnu (pas de melange muet)."""
    try:
        return ENGINE_LABELS[engine]
    except KeyError:
        raise ValueError(
            "moteur inconnu %r : labels connus %s (le label reduit %r vient du chemin "
            "diag/diag_polar_omega.py, jamais de run.py)"
            % (engine, sorted(ENGINE_LABELS), REDUCED_EXB_LABEL)
        )


def verify_paper_windows(windows: dict) -> bool:
    """Verifie que ``windows`` EST exactement les fenetres verbatim du papier (unites T_d).

    Appele au demarrage du run complet (run.py) AVANT toute mesure : c'est le
    pre-enregistrement. Toute fenetre manquante, en trop, ou differente leve une
    ``AssertionError`` explicite. Empeche d'introduire des fenetres adaptatives.

    NB CLOCK (T3) : ces fenetres sont en temps PAPIER (T_d). Ce gate verrouille les
    ENDPOINTS ; l'horloge est enforced par ``run.py:fit_growth`` qui fitte sur le temps
    sim MAPPE ``paper_to_sim_time_window(window, rhobar)`` (= 2pi/rhobar * window). Fitter
    la fenetre papier BRUTE sur le temps sim (l'ancien comportement) tombait dans le
    transitoire -- c'etait l'origine de l'artefact -95 % (cf. docs/T2_NORMALIZATION_AUDIT.md).
    """
    got = {int(k): (float(v[0]), float(v[1])) for k, v in windows.items()}
    want = {
        k: (float(v[0]), float(v[1]))
        for k, v in PAPER_FIT_WINDOWS_VERBATIM.items()
    }
    if got != want:
        raise AssertionError(
            "les fenetres de fit du modele complet doivent etre les fenetres verbatim "
            "du papier (Fig. 5.4) %s, obtenu %s. Aucune fenetre adaptative n'est "
            "autorisee pour la comparaison du modele complet." % (want, got)
        )
    return True


def _git_sha(path: str) -> str:
    """SHA court du depot git contenant ``path`` (ou 'unknown' hors git/sans binaire)."""
    if not path:
        return "unknown"
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if not directory:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def adc_cases_sha() -> str:
    """SHA court du depot adc_cases (celui qui contient ce fichier)."""
    return _git_sha(os.path.dirname(os.path.abspath(__file__)))


def adc_cpp_sha(adc_module=None) -> str:
    """SHA court du depot adc_cpp.

    Localise le depot via le module ``adc`` importe (son ``__file__`` vit dans
    ``<adc_cpp>/build-*/python/adc/``) ; ``$ADC_CPP_SHA`` surcharge si defini ;
    'unknown' si rien n'est resolvable (pas de build importe).
    """
    env = os.environ.get("ADC_CPP_SHA")
    if env:
        return env
    if adc_module is not None and getattr(adc_module, "__file__", None):
        return _git_sha(os.path.dirname(os.path.abspath(adc_module.__file__)))
    return "unknown"


def err_pct(gamma_numeric, gamma_paper: float) -> float | None:
    """err_pct = 100*(gamma_numeric - gamma_paper)/gamma_paper, ou None si non mesure.

    Ne fabrique rien : renvoie None des que gamma_numeric n'est pas un nombre fini
    (run non joue, fit echoue) ; l'enregistrement portera alors 'PENDING'.
    """
    if gamma_numeric is None:
        return None
    try:
        g = float(gamma_numeric)
    except (TypeError, ValueError):
        return None
    if g != g or g in (float("inf"), float("-inf")):  # NaN ou inf
        return None
    return 100.0 * (g - gamma_paper) / gamma_paper


def _fmt(value):
    """Cellule texte : 'PENDING' si None/non-fini, sinon le nombre tel quel."""
    if value is None:
        return "PENDING"
    if isinstance(value, float) and value != value:
        return "PENDING"
    return value


# Colonnes de l'enregistrement, dans l'ordre. C'est la graine de la table de
# validation Phase 2 ; ne PAS reordonner sans mettre a jour les consommateurs.
# T3 : on porte LES DEUX pentes (brute sim + papier) cote a cote, jamais l'une ecrase
# l'autre, + les deux fenetres (papier T_d et sim mappee) + rhobar + le facteur.
RECORD_FIELDS = (
    "engine",  # full-system-schur / reduced-ExB / amr-imex-experimental
    "mode",  # l
    "gamma_raw_sim",  # pente BRUTE dans l'horloge sim, fenetre MAPPEE (reproductibilite)
    "gamma_paper_units",  # = gamma_raw_sim * 2pi/rhobar (comparaison papier)
    "gamma_paper",  # cible papier (0.772 / 0.911 / 0.683)
    "err_pct",  # 100*(gamma_paper_units - paper)/paper (PENDING si non mesure)
    "normalization",  # formule explicite du facteur applique
    "fit_window_paper",  # 'lo,hi' verbatim papier (unites T_d)
    "fit_window_sim",  # 'lo,hi' MAPPEE en temps sim (= 2pi/rhobar * papier)
    "rhobar",  # rho_max (= 1) ; facteur = 2pi/rhobar
    "time_scale_sim_per_paper",  # 2pi/rhobar : nb d'unites de temps sim par unite papier
    "n",  # resolution
    "dt",  # pas de temps
    "splitting",  # 'Lie' / 'Strang'
    "schur_theta",  # theta du CondensedSchur (None si pas de Schur)
    "backend",  # ex. 'kokkos-serial', 'mpi-4'
    "mpi_size",  # nb de rangs
    "adc_cpp_sha",  # SHA court adc_cpp
    "adc_cases_sha",  # SHA court adc_cases
)


def build_record(
    *,
    engine: str,
    mode: int,
    gamma_raw_sim,
    gamma_paper: float,
    fit_window: tuple[float, float],
    n: int,
    dt: float,
    splitting: str,
    schur_theta,
    backend: str,
    rhobar: float = 1.0,
    mpi_size: int = 1,
    adc_cpp_sha_value=None,
    adc_cases_sha_value=None,
) -> dict:
    """Construit un enregistrement de mesure pour un (engine, mode).

    ``gamma_raw_sim`` est la pente BRUTE mesuree dans l'horloge sim (fenetre MAPPEE).
    On en derive ``gamma_paper_units = gamma_raw_sim * 2pi/rhobar`` (T3) et ``err_pct``
    compare CE dernier a la cible. ``gamma_raw_sim=None`` (ou NaN) => 'PENDING' pour les
    deux gammas et err_pct (aucune valeur inventee). ``fit_window`` est la fenetre PAPIER
    (T_d) ; la fenetre sim mappee est derivee. ``engine`` inconnu leve.
    """
    label = engine_label(engine)
    gamma_paper_units = gamma_to_paper_units(gamma_raw_sim, rhobar)
    scale = 2.0 * math.pi / rhobar
    normalization = (
        "gamma_paper = gamma_raw_sim * 2pi/rhobar (rhobar=%g, factor=%.6f); fit sur temps sim MAPPE"
        % (rhobar, scale)
    )
    lo, hi = fit_window
    slo, shi = paper_to_sim_time_window(fit_window, rhobar)
    return {
        "engine": label,
        "mode": int(mode),
        "gamma_raw_sim": _fmt(gamma_raw_sim),
        "gamma_paper_units": _fmt(gamma_paper_units),
        "gamma_paper": float(gamma_paper),
        "err_pct": _fmt(err_pct(gamma_paper_units, gamma_paper)),
        "normalization": normalization,
        "fit_window_paper": "%g,%g" % (float(lo), float(hi)),
        "fit_window_sim": "%g,%g" % (slo, shi),
        "rhobar": float(rhobar),
        "time_scale_sim_per_paper": scale,
        "n": int(n),
        "dt": float(dt),
        "splitting": splitting,
        "schur_theta": (None if schur_theta is None else float(schur_theta)),
        "backend": backend,
        "mpi_size": int(mpi_size),
        "adc_cpp_sha": adc_cpp_sha_value or "unknown",
        "adc_cases_sha": adc_cases_sha_value or "unknown",
    }


def write_records(
    records, out_dir: str, basename: str = "measurement_record"
) -> tuple[str, str]:
    """Ecrit les enregistrements en CSV ET JSON sous ``out_dir``.

    Renvoie (csv_path, json_path). Le CSV porte l'entete RECORD_FIELDS ; le JSON
    est une liste d'objets (None -> null, gere par json).
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, basename + ".csv")
    json_path = os.path.join(out_dir, basename + ".json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in RECORD_FIELDS})

    with open(json_path, "w") as f:
        json.dump(list(records), f, indent=2, sort_keys=True)

    return csv_path, json_path


def _selftest() -> None:
    """Auto-test pur Python (CI) : assertions REELLES, aucun build requis."""
    import math
    import tempfile

    # 1. Les fenetres verbatim sont bien celles du papier ; une fenetre modifiee leve.
    assert verify_paper_windows(PAPER_FIT_WINDOWS_VERBATIM) is True
    bad = dict(PAPER_FIT_WINDOWS_VERBATIM)
    bad[3] = (0.30, 0.70)  # fenetre adaptative interdite
    try:
        verify_paper_windows(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "verify_paper_windows aurait du rejeter une fenetre modifiee"
        )

    # 2. Labels de moteur : mapping explicite, refus d'un moteur inconnu, jamais
    #    le label reduit pour le chemin complet.
    assert engine_label("system-schur") == "full-system-schur"
    assert engine_label("amr-imex") == "amr-imex-experimental"
    assert (
        engine_label("polar-schur") == "full-polar-schur"
    )  # chemin polaire (run_polar.py)
    assert REDUCED_EXB_LABEL not in ENGINE_LABELS.values()
    try:
        engine_label("reduced-ExB")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "engine_label aurait du refuser le label reduit pour run.py"
        )

    # 3. err_pct exact, et None (=> PENDING) pour les entrees non mesurees.
    assert abs(err_pct(0.911, 0.911) - 0.0) < 1e-12
    assert abs(err_pct(0.772 * 1.1, 0.772) - 10.0) < 1e-9
    assert err_pct(None, 0.772) is None
    assert err_pct(float("nan"), 0.772) is None

    # 3bis. T3 : helpers de mapping fenetre + conversion gamma.
    assert paper_to_sim_time_window((0.40, 0.70), 1.0) == (
        0.40 * 2 * math.pi,
        0.70 * 2 * math.pi,
    )
    assert abs(gamma_to_paper_units(0.10, 1.0) - 0.10 * 2 * math.pi) < 1e-12
    assert (
        gamma_to_paper_units(None) is None
        and gamma_to_paper_units(float("nan")) is None
    )
    # rhobar=2 : facteur 2pi/2 = pi (le rhobar divise)
    assert abs(gamma_to_paper_units(1.0, 2.0) - math.pi) < 1e-12

    # 4. Enregistrement mesure (T3) : on porte gamma_raw_sim ET gamma_paper_units = raw*2pi,
    #    err_pct compare gamma_paper_units a la cible, normalisation = formule explicite.
    rec = build_record(
        engine="system-schur",
        mode=4,
        gamma_raw_sim=0.145,
        gamma_paper=0.911,
        fit_window=(0.60, 0.75),
        n=384,
        dt=1e-3,
        splitting="Strang",
        schur_theta=0.5,
        backend="kokkos-serial",
        rhobar=1.0,
        mpi_size=1,
        adc_cpp_sha_value="abc1234",
        adc_cases_sha_value="def5678",
    )
    assert rec["engine"] == "full-system-schur"
    assert rec["gamma_raw_sim"] == 0.145
    assert (
        abs(rec["gamma_paper_units"] - 0.145 * 2 * math.pi) < 1e-12
    )  # le 2pi EST applique au full
    assert "2pi/rhobar" in rec["normalization"]
    assert (
        abs(rec["err_pct"] - 100.0 * (0.145 * 2 * math.pi - 0.911) / 0.911)
        < 1e-9
    )
    assert rec["fit_window_paper"] == "0.6,0.75"
    assert rec["fit_window_sim"] == "%g,%g" % (
        0.60 * 2 * math.pi,
        0.75 * 2 * math.pi,
    )
    assert abs(rec["time_scale_sim_per_paper"] - 2 * math.pi) < 1e-12
    assert rec["schur_theta"] == 0.5
    assert rec["adc_cpp_sha"] == "abc1234"

    # 5. Enregistrement PENDING : gamma_raw_sim non mesure => 'PENDING' pour les 2 gammas + err.
    pend = build_record(
        engine="system-schur",
        mode=3,
        gamma_raw_sim=float("nan"),
        gamma_paper=0.772,
        fit_window=(0.40, 0.70),
        n=512,
        dt=1e-3,
        splitting="Strang",
        schur_theta=0.5,
        backend="kokkos-serial",
    )
    assert pend["gamma_raw_sim"] == "PENDING"
    assert pend["gamma_paper_units"] == "PENDING"
    assert pend["err_pct"] == "PENDING"

    # 6. Ecriture CSV + JSON : round-trip JSON, entete CSV exacte.
    with tempfile.TemporaryDirectory() as d:
        csv_path, json_path = write_records([rec, pend], d)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded[0]["engine"] == "full-system-schur"
        assert loaded[1]["gamma_raw_sim"] == "PENDING"
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        assert header == list(RECORD_FIELDS)

    # 7. Le 2 pi est BIEN applique au modele complet (T3 : plus de "reduced-only").
    assert math.isclose(2.0 * math.pi, 6.283185307, rel_tol=1e-6)
    assert (
        rec["gamma_paper_units"] != rec["gamma_raw_sim"]
    )  # le facteur n'est pas 1

    print(
        "OK results.py: fenetres verbatim+clock, labels moteur, mapping T3, record raw+paper, PENDING, IO"
    )


if __name__ == "__main__":
    _selftest()
