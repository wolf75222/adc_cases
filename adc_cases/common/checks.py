"""Invariants physiques verifies par plusieurs cas.

Ces controles (conservation de la masse, finitude, positivite) reviennent dans presque tous
les cas. Ils levent `AssertionError` avec un message clair : un cas qui echoue sort en erreur,
ce qui rend la CV/CI rouge. On garde le COMPORTEMENT historique (memes tolerances par defaut).
"""

import numpy as np


def relative_drift(value, reference):
    """Ecart relatif |value - reference| / |reference| (denominateur protege contre zero)."""
    return abs(value - reference) / max(abs(reference), 1e-30)


def assert_mass_conserved(mass, mass0, tol=1e-9, label="", relative=True):
    """Verifie la conservation de la masse : derive (relative par defaut) sous `tol`.

    `relative=False` compare l'ecart ABSOLU |mass - mass0| (utilise par les cas qui historiquement
    asseraient en absolu). Renvoie la derive mesuree.
    """
    drift = relative_drift(mass, mass0) if relative else abs(mass - mass0)
    kind = "relative" if relative else "absolue"
    tag = f"{label} : " if label else ""
    assert drift < tol, f"{tag}masse non conservee (derive {kind} {drift:.3e} >= {tol:.1e})"
    return drift


def assert_finite(array, label="champ"):
    """Verifie qu'un tableau ne contient ni NaN ni Inf."""
    arr = np.asarray(array)
    assert np.isfinite(arr).all(), f"{label} non fini (NaN/Inf)"


def assert_positive(array, label="densite"):
    """Verifie la positivite stricte (un fluide physique reste > 0). Renvoie le minimum."""
    arr = np.asarray(array)
    mn = float(arr.min())
    assert mn > 0.0, f"{label} negative ou nulle (min = {mn:.3e})"
    return mn


def assert_opposite_sign(a, b, min_mag=0.0, label=""):
    """Verifie que `a` et `b` ont des signes STRICTEMENT opposes (a*b < 0).

    `min_mag` exige en plus que chaque grandeur depasse ce seuil en valeur absolue : on evite
    ainsi de valider trivialement deux quasi-zeros (bruit machine) dont le produit serait negatif
    par accident. Utilise pour le contraste physique attractif vs repulsif (euler_poisson).
    """
    tag = f"{label} : " if label else ""
    assert abs(a) > min_mag and abs(b) > min_mag, (
        f"{tag}magnitude trop faible (|a|={abs(a):.3e}, |b|={abs(b):.3e} <= {min_mag:.1e}) : "
        "signe non significatif")
    assert a * b < 0.0, f"{tag}signes non opposes (a={a:+.3e}, b={b:+.3e})"
