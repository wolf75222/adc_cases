"""Package `adc_cases` : utilitaires partages des cas d'utilisation de la lib `adc`.

Chaque cas (un dossier par cas a la racine du depot) importe ce paquet pour reutiliser
ce qui est commun a plusieurs scenarios :

  - `adc_cases.models`               : modeles nommes = compositions de briques generiques
                                       de `adc` (electron_euler, ion_isothermal, diocotron,
                                       euler_poisson, recettes systeme two_fluid / plasma) ;
  - `adc_cases.common.grid`          : grilles a centres de cellules (meshgrid) ;
  - `adc_cases.common.initial_conditions` : conditions initiales reutilisees (bande, anneau,
                                       bulle de pression Euler) ;
  - `adc_cases.common.checks`        : invariants verifies par plusieurs cas (masse, finitude,
                                       positivite) ;
  - `adc_cases.common.io`            : repertoire de sortie `out/` (hors source, gitignore).

Importer ce paquet, mode d'emploi
---------------------------------
Le paquet est installable (``pip install -e .`` a la racine du depot) ; les scripts de cas
font alors un simple ``import adc_cases`` sans toucher a ``sys.path``. C'est ce que fait la CI.

Pour lancer un cas SANS installer le paquet (``python3 diocotron/run.py`` depuis le depot),
chaque script appelle ``adc_cases.ensure_importable()`` une fois en tete : cette fonction
place la racine du depot sur le chemin d'import si elle n'y est pas deja (idempotent), de sorte
que les sous-modules ``adc_cases.models`` / ``adc_cases.common.*`` soient resolus. Une fois le
paquet installe, l'appel est sans effet.

Le module C++ ``adc`` (bindings pybind11 d'adc_cpp) reste fourni par le PYTHONPATH du build
(cf. README) ; ce paquet ne le construit ni ne le localise.
"""

import os
import sys

#: Racine du depot adc_cases (le dossier qui CONTIENT ce paquet).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_importable():
    """Rend le depot importable pour un cas lance directement (sans installation du paquet).

    Insere la racine du depot sur ``sys.path`` si absente (idempotent). Inutile lorsque le
    paquet est installe (``pip install -e .``), ou ``import adc_cases`` resout deja tout seul.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def bootstrap():
    """Compat : ``ensure_importable()`` puis renvoie le sous-module ``models``."""
    ensure_importable()
    from adc_cases import models
    return models
