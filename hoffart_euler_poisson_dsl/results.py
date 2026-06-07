"""Emetteur d'enregistrements de mesure pour le modele COMPLET system-schur (T3).

But
---
Rendre la mesure du modele complet HONNETE et PRE-ENREGISTREE :

1. Pre-enregistrer et VERIFIER les fenetres de fit verbatim du papier
   (Fig. 5.4, arXiv:2510.11808) : l=3 [0.40,0.70], l=4 [0.60,0.75],
   l=5 [1.15,1.35]. Aucune fenetre adaptative n'est introduite pour la
   comparaison du modele complet (cf. ``verify_paper_windows``).

2. Reporter la pente exp BRUTE du modele complet system-schur DIRECTEMENT
   contre 0.772/0.911/0.683, SANS facteur 2 pi et SANS facteur rhobar. C'est
   la normalisation resolue (cf. docs/HOFFART_FIDELITY.md, ligne |Omega|) : le
   2 pi / rhobar appartient UNIQUEMENT au chemin reduit ExB scalaire
   (diag/diag_polar_omega.py), jamais au chemin complet.

3. Distinguer sans ambiguite les moteurs :
       engine = 'full-system-schur'  -> pente brute, AUCUN facteur
       engine = 'reduced-ExB'        -> 2 pi / rhobar (NORMALIZATION.md)
   pour que les nombres reduits porteurs du 2 pi ne soient JAMAIS melanges avec
   les nombres bruts du modele complet.

4. Emettre un enregistrement par run (CSV + JSON) capturant : SHA adc_cpp, SHA
   adc_cases, backend, n, dt, splitting (Lie/Strang), schur(theta), fenetre de
   fit, gamma_numeric, gamma_paper, err_pct. C'est la graine de la table de
   validation de la Phase 2.

Ce module NE fabrique AUCUN nombre : il mesure et enregistre ce qu'un run
produit (gamma_numeric vient du fit ; PENDING si le run n'a pas tourne).

Ce module est PUR PYTHON (aucune dependance au binding ``adc`` ni a un build) :
son auto-test ``python results.py`` tourne en CI sans compiler le coeur.
"""

import csv
import json
import os
import subprocess

# Fenetres de fit verbatim du papier (Fig. 5.4, arXiv:2510.11808). Dupliquees ici
# en tant que VERITE PRE-ENREGISTREE : ``verify_paper_windows`` confronte les
# fenetres effectivement utilisees par run.py (model.PAPER_FIT_WINDOWS) a celles-ci
# et leve si elles divergent. C'est le verrou contre toute fenetre adaptative qui
# se glisserait dans la comparaison du modele complet.
PAPER_FIT_WINDOWS_VERBATIM = {3: (0.40, 0.70), 4: (0.60, 0.75), 5: (1.15, 1.35)}

# Etiquettes de moteur. La cle est l'``--engine`` du runner ; la valeur est le label
# explicite porte par l'enregistrement. 'full-system-schur' = pente BRUTE (aucun
# facteur) ; 'reduced-ExB' (hors run.py, cf. diag) = 2 pi / rhobar.
ENGINE_LABELS = {
    "system-schur": "full-system-schur",
    "amr-imex": "amr-imex-experimental",
}

# Le facteur 2 pi / rhobar n'appartient QU'au chemin reduit. Conserve ici comme
# constante nommee pour rendre explicite, dans le code et les tests, que le chemin
# complet ne l'applique PAS (normalization_factor == 1.0 pour full-system-schur).
REDUCED_EXB_LABEL = "reduced-ExB"


def engine_label(engine):
    """Label explicite du moteur ; refuse tout moteur inconnu (pas de melange muet)."""
    try:
        return ENGINE_LABELS[engine]
    except KeyError:
        raise ValueError(
            "moteur inconnu %r : labels connus %s (le label reduit %r vient du chemin "
            "diag/diag_polar_omega.py, jamais de run.py)"
            % (engine, sorted(ENGINE_LABELS), REDUCED_EXB_LABEL)
        )


def verify_paper_windows(windows):
    """Verifie que ``windows`` EST exactement les fenetres verbatim du papier.

    Appele au demarrage du run complet (run.py) AVANT toute mesure : c'est le
    pre-enregistrement. Toute fenetre manquante, en trop, ou differente leve une
    ``AssertionError`` explicite. Empeche d'introduire des fenetres adaptatives
    pour la comparaison du modele complet.
    """
    got = {int(k): (float(v[0]), float(v[1])) for k, v in windows.items()}
    want = {k: (float(v[0]), float(v[1])) for k, v in PAPER_FIT_WINDOWS_VERBATIM.items()}
    if got != want:
        raise AssertionError(
            "les fenetres de fit du modele complet doivent etre les fenetres verbatim "
            "du papier (Fig. 5.4) %s, obtenu %s. Aucune fenetre adaptative n'est "
            "autorisee pour la comparaison du modele complet." % (want, got)
        )
    return True


def _git_sha(path):
    """SHA court du depot git contenant ``path`` (ou 'unknown' hors git/sans binaire)."""
    if not path:
        return "unknown"
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if not directory:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def adc_cases_sha():
    """SHA court du depot adc_cases (celui qui contient ce fichier)."""
    return _git_sha(os.path.dirname(os.path.abspath(__file__)))


def adc_cpp_sha(adc_module=None):
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


def err_pct(gamma_numeric, gamma_paper):
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
RECORD_FIELDS = (
    "engine",          # label explicite : full-system-schur / reduced-ExB / amr-imex-experimental
    "mode",            # l
    "gamma_numeric",   # pente exp BRUTE mesuree (PENDING si non mesuree)
    "gamma_paper",     # cible papier (0.772 / 0.911 / 0.683)
    "err_pct",         # 100*(num - paper)/paper (PENDING si gamma_numeric absent)
    "normalization",   # 'raw (no 2pi, no rhobar)' pour full ; documente le facteur
    "fit_window",      # 'lo,hi' verbatim papier
    "n",               # resolution
    "dt",              # pas de temps
    "splitting",       # 'Lie' / 'Strang'
    "schur_theta",     # theta du CondensedSchur (None si pas de Schur)
    "backend",         # ex. 'kokkos-serial', 'mpi-4'
    "mpi_size",        # nb de rangs
    "adc_cpp_sha",     # SHA court adc_cpp
    "adc_cases_sha",   # SHA court adc_cases
)


def build_record(
    *,
    engine,
    mode,
    gamma_numeric,
    gamma_paper,
    fit_window,
    n,
    dt,
    splitting,
    schur_theta,
    backend,
    mpi_size=1,
    adc_cpp_sha_value=None,
    adc_cases_sha_value=None,
):
    """Construit un enregistrement de mesure pour un (engine, mode).

    ``gamma_numeric=None`` (ou NaN) => l'enregistrement porte 'PENDING' pour
    gamma_numeric et err_pct : aucune valeur n'est inventee. ``engine`` est
    converti en son label explicite ; un moteur inconnu leve.
    """
    label = engine_label(engine)
    # Le facteur de normalisation est explicite et code en dur par moteur : le
    # chemin complet est BRUT (1.0), jamais 2 pi / rhobar.
    normalization = "raw (no 2pi, no rhobar)"
    lo, hi = fit_window
    return {
        "engine": label,
        "mode": int(mode),
        "gamma_numeric": _fmt(gamma_numeric),
        "gamma_paper": float(gamma_paper),
        "err_pct": _fmt(err_pct(gamma_numeric, gamma_paper)),
        "normalization": normalization,
        "fit_window": "%g,%g" % (float(lo), float(hi)),
        "n": int(n),
        "dt": float(dt),
        "splitting": splitting,
        "schur_theta": (None if schur_theta is None else float(schur_theta)),
        "backend": backend,
        "mpi_size": int(mpi_size),
        "adc_cpp_sha": adc_cpp_sha_value or "unknown",
        "adc_cases_sha": adc_cases_sha_value or "unknown",
    }


def write_records(records, out_dir, basename="measurement_record"):
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


def _selftest():
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
        raise AssertionError("verify_paper_windows aurait du rejeter une fenetre modifiee")

    # 2. Labels de moteur : mapping explicite, refus d'un moteur inconnu, jamais
    #    le label reduit pour le chemin complet.
    assert engine_label("system-schur") == "full-system-schur"
    assert engine_label("amr-imex") == "amr-imex-experimental"
    assert REDUCED_EXB_LABEL not in ENGINE_LABELS.values()
    try:
        engine_label("reduced-ExB")
    except ValueError:
        pass
    else:
        raise AssertionError("engine_label aurait du refuser le label reduit pour run.py")

    # 3. err_pct exact, et None (=> PENDING) pour les entrees non mesurees.
    assert abs(err_pct(0.911, 0.911) - 0.0) < 1e-12
    assert abs(err_pct(0.772 * 1.1, 0.772) - 10.0) < 1e-9
    assert err_pct(None, 0.772) is None
    assert err_pct(float("nan"), 0.772) is None

    # 4. Enregistrement mesure : pente brute reportee telle quelle, err_pct calcule,
    #    facteur de normalisation BRUT (jamais 2 pi).
    rec = build_record(
        engine="system-schur", mode=4, gamma_numeric=0.9, gamma_paper=0.911,
        fit_window=(0.60, 0.75), n=384, dt=1e-3, splitting="Lie", schur_theta=0.5,
        backend="kokkos-serial", mpi_size=1,
        adc_cpp_sha_value="abc1234", adc_cases_sha_value="def5678",
    )
    assert rec["engine"] == "full-system-schur"
    assert rec["gamma_numeric"] == 0.9
    assert rec["normalization"] == "raw (no 2pi, no rhobar)"
    assert abs(rec["err_pct"] - 100.0 * (0.9 - 0.911) / 0.911) < 1e-9
    assert rec["fit_window"] == "0.6,0.75"
    assert rec["schur_theta"] == 0.5
    assert rec["adc_cpp_sha"] == "abc1234"
    assert "2pi" not in str(rec["normalization"]).replace("no 2pi", "")  # pas de 2pi residuel

    # 5. Enregistrement PENDING : gamma_numeric non mesure => 'PENDING', err_pct 'PENDING'.
    pend = build_record(
        engine="system-schur", mode=3, gamma_numeric=float("nan"), gamma_paper=0.772,
        fit_window=(0.40, 0.70), n=512, dt=1e-3, splitting="Lie", schur_theta=0.5,
        backend="kokkos-serial",
    )
    assert pend["gamma_numeric"] == "PENDING"
    assert pend["err_pct"] == "PENDING"

    # 6. Ecriture CSV + JSON : round-trip JSON, entete CSV exacte.
    with tempfile.TemporaryDirectory() as d:
        csv_path, json_path = write_records([rec, pend], d)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded[0]["engine"] == "full-system-schur"
        assert loaded[1]["gamma_numeric"] == "PENDING"
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        assert header == list(RECORD_FIELDS)

    # 7. La constante de normalisation reduite (2 pi / rhobar) ne fuit pas dans le
    #    chemin complet : aucun enregistrement complet ne porte un facteur != brut.
    assert math.isclose(2.0 * math.pi, 6.283185307, rel_tol=1e-6)  # sanity du facteur reduit
    assert rec["normalization"] != "%g" % (2.0 * math.pi)

    print("OK results.py: fenetres verbatim, labels moteur, err_pct, record brut, PENDING, IO")


if __name__ == "__main__":
    _selftest()
