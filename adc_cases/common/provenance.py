"""Provenance d'une mesure : SHA + branche des DEUX depots, machine, python, threads.

Centralise la capture jusqu'ici eparpillee (cf. `tutorial/run.py:git_sha`,
`hoffart_euler_poisson_dsl/results.py`). Toute sortie de la campagne de perf (JSONL) DOIT
porter ces champs : l'acceptation exige des SHA exacts et qu'aucun graphe ne melange master et PR.

Le SHA d'adc_cpp est lu depuis le depot qui CONTIENT le module `adc` importe (le build du
worktree fige) ; celui d'adc_cases depuis ce paquet. Sur un noeud de calcul (ROMEO) ou git peut
manquer / differer, les variables d'environnement `ADC_CPP_SHA`, `ADC_CPP_BRANCH`, `ADC_CASES_SHA`,
`ADC_CASES_BRANCH`, `ADC_MACHINE` ont la PRIORITE (injectees par le script de lancement).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import adc

from .. import REPO_ROOT


def _git(path: str, *args: str) -> str:
    """`git -C path args` ; renvoie la sortie ou "unknown" (git absent, hors depot, etc.)."""
    try:
        return subprocess.check_output(
            ["git", "-C", path, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def adc_cpp_root() -> str:
    """Racine du depot adc_cpp qui fournit le module `adc` importe.

    `adc.__file__` = <root>/<build>/python/adc/__init__.py -> on remonte de trois niveaux
    (adc -> python -> build -> root), comme tutorial/run.py.
    """
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(adc.__file__)), "..", "..", ".."
        )
    )


def provenance(extra: dict | None = None) -> dict[str, object]:
    """Dictionnaire de provenance (SHA/branche des deux depots, machine, python, threads).

    Les variables d'environnement `ADC_*` surchargent les valeurs lues via git (chemin ROMEO).
    `extra` fusionne des champs additionnels (backend, nx, ...) dans le resultat.
    """
    cpp_root = adc_cpp_root()
    prov = {
        "adc_cpp_sha": os.environ.get("ADC_CPP_SHA")
        or _git(cpp_root, "rev-parse", "HEAD"),
        "adc_cpp_branch": (
            os.environ.get("ADC_CPP_BRANCH")
            or _git(cpp_root, "rev-parse", "--abbrev-ref", "HEAD")
        ),
        "adc_cases_sha": os.environ.get("ADC_CASES_SHA")
        or _git(REPO_ROOT, "rev-parse", "HEAD"),
        "adc_cases_branch": (
            os.environ.get("ADC_CASES_BRANCH")
            or _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
        ),
        "machine": os.environ.get("ADC_MACHINE") or platform.node(),
        "python": "%d.%d.%d" % sys.version_info[:3],
        "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "adc_module": adc.__file__,
    }
    if extra:
        prov.update(extra)
    return prov
