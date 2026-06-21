#!/usr/bin/env python3
"""Garde-fous adc_cases : executes en CI (ci.yml), purs Python, rapides.

Echoue (exit 1) sur :
  1. un dossier contenant `run.py` sans `README.md` (chaque cas doit etre documente) ;
  2. une entree du manifeste `cases_manifest.toml` dont le `path` n'existe pas ;
  3. un `README.md` de cas qui contient un em-dash U+2014 (convention adc_cases : accents OK, mais
     pas d'em-dash, le depot existant en a 0).
  4. une image `![](figures/...)` referencee par un README mais introuvable sur le disque.

Lancement : python3 check_cases.py   (depuis la racine du depot). 0 = OK, 1 = violations.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
EM_DASH = "—"


def manifest_paths() -> list[str]:
    """Extrait les `path = "..."` de cases_manifest.toml (sans dependance tomllib pour py<3.11)."""
    f = ROOT / "cases_manifest.toml"
    if not f.exists():
        return []
    return re.findall(r'^\s*path\s*=\s*"([^"]+)"', f.read_text(encoding="utf-8"), flags=re.M)


def check() -> int:
    violations: list[str] = []
    # 1. chaque dossier avec run.py a un README.md
    for run in sorted(ROOT.glob("*/run.py")):
        readme = run.parent / "README.md"
        if not readme.exists():
            violations.append(f"{run.parent.name}/ : run.py present mais README.md MANQUANT")
    for run in sorted(ROOT.glob("*/runs/run.py")):
        case = run.parent.parent
        if not (case / "README.md").exists():
            violations.append(f"{case.name}/ : runs/run.py present mais README.md MANQUANT")
    # 2. chaque path du manifeste existe
    for p in manifest_paths():
        if not (ROOT / p).exists():
            violations.append(f"cases_manifest.toml : path inexistant '{p}'")
    # 3. + 4. em-dash et images des README
    img_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for readme in sorted(ROOT.glob("*/README.md")) + ([ROOT / "README.md"] if (ROOT / "README.md").exists() else []):
        text = readme.read_text(encoding="utf-8")
        if EM_DASH in text:
            violations.append(f"{readme.relative_to(ROOT)} : {text.count(EM_DASH)} em-dash (U+2014) interdits")
        for m in img_re.finditer(text):
            t = m.group(1).split()[0].split("#")[0]
            if t and not t.startswith(("http://", "https://", "data:")) and "..." not in t:
                if not (readme.parent / t).resolve().exists():
                    line = text[: m.start()].count("\n") + 1
                    violations.append(f"{readme.relative_to(ROOT)}:{line} : image introuvable : {t}")
    if violations:
        print(f"CHECK-CASES : {len(violations)} violation(s)", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1
    n_cases = len(list(ROOT.glob("*/run.py"))) + len(list(ROOT.glob("*/runs/run.py")))
    print(f"CHECK-CASES : OK ({n_cases} cas, tous documentes, manifeste coherent, 0 em-dash)")
    return 0


if __name__ == "__main__":
    sys.exit(check())
