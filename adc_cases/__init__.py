"""Package `adc_cases` : utilitaires partages des cas d'utilisation de la lib `adc`.

Le paquet separe les responsabilites en sous-modules dedies :

  - `adc_cases.models`               : modeles d'espece = compositions de briques generiques de
                                       `adc` (electron_euler, ion_isothermal, diocotron, euler) ;
  - `adc_cases.recipes`              : recettes systeme = configurations multi-especes pretes a
                                       l'emploi (two_fluid, plasma : blocs + Poisson + couplages) ;
  - `adc_cases.common.grid`          : grilles a centres de cellules (meshgrid) ;
  - `adc_cases.common.initial_conditions` : conditions initiales reutilisees (bande, anneau,
                                       bulle de pression Euler) ;
  - `adc_cases.common.checks`        : invariants verifies par plusieurs cas (masse, finitude,
                                       positivite) ;
  - `adc_cases.common.io`            : repertoire de sortie `out/` (hors source, gitignore) ;
  - `adc_cases.common.native`        : compilation a la volee + chargement ctypes des scenarios
                                       sur mesure (cache hors source, controle d'ABI explicite).

Importer ce paquet, mode d'emploi
---------------------------------
Le paquet est installable (`pip install -e .` a la racine du depot) ; les scripts de cas
font alors un simple `import adc_cases.models` sans toucher a `sys.path`. C'est la voie
nominale, celle de la CI.

Pour lancer un cas sans installer le paquet (`python3 diocotron/run.py` depuis le depot),
chaque script appelle `adc_cases.ensure_importable()` une fois en tete : cette fonction place
la racine du depot sur le chemin d'import si elle n'y est pas deja (idempotent). Une fois le
paquet installe, l'appel est sans effet. C'est l'unique manipulation de `sys.path` du depot :
les cas ne reconstruisent plus de chemin a la main.

Le module C++ `adc` (bindings pybind11 d'adc_cpp) reste fourni par le PYTHONPATH du build
(cf. README) ; ce paquet ne le construit ni ne le localise.
"""

from __future__ import annotations

import os
import sys

#: Racine du depot adc_cases (le dossier qui contient ce paquet).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_importable() -> None:
    """Rend le depot importable pour un cas lance directement (sans installation).

    Insere la racine du depot sur `sys.path` si absente (idempotent). Inutile
    lorsque le paquet est installe (`pip install -e .`), ou `import adc_cases`
    resout deja tout seul.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
