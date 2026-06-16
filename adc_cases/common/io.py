"""Repertoire de sortie des cas : `out/` a la racine du depot, hors des dossiers source.

Les cas qui produisent des fichiers (figures, gif) ecrivent sous `out/<cas>/` plutot que dans
leur dossier source. `out/` est ignore par git (cf. .gitignore), ce qui evite de polluer le
depot avec des artefacts. On peut surcharger la racine via la variable d'environnement
`ADC_CASES_OUT`.
"""

from __future__ import annotations

import os

from .. import REPO_ROOT


def out_root() -> str:
    """Racine des sorties : $ADC_CASES_OUT si defini, sinon `<depot>/out`."""
    return os.environ.get("ADC_CASES_OUT", os.path.join(REPO_ROOT, "out"))


def case_output_dir(case_name: str) -> str:
    """Cree et renvoie `<out_root>/<case_name>` (sorties d'un cas donne)."""
    path = os.path.join(out_root(), case_name)
    os.makedirs(path, exist_ok=True)
    return path
