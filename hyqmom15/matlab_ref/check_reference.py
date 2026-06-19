#!/usr/bin/env python3
"""Garde-fou matlab_ref : verifie que REFERENCE.md verrouille bien la reference.

Pur Python (aucune dependance, aucun build), pense pour tourner en CI via le
manifeste. Echoue (exit 1) si la note de reference HyQMOM15 (ADC-348) :

  1. est absente ou trop courte (verrouillage vide) ;
  2. contient un em-dash U+2014 ou un caractere non-ASCII (note en-US) ;
  3. n'explicite plus un des points d'acceptation d'ADC-348 : reference
     canonique vs legacy, parametres diocotron, decisions Dmax /
     init_magnetic_wave_field / source magnetique / abandon HLLC-WENO, politique
     bug-for-bug, et le relais vers ADC-349/350/351/356.

Lancement : python3 hyqmom15/matlab_ref/check_reference.py (0 = OK, 1 = manque).
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REFERENCE = HERE / "REFERENCE.md"
EM_DASH = "—"
MIN_CHARS = 1500

# Chaque jeton encode un critere d'acceptation d'ADC-348 ; la note doit tous les
# couvrir. Compares en minuscules (insensible a la casse). Ranges par theme.
REQUIRED_TOKENS = [
    # Reference canonique vs heritage
    "riemom2d_electrostatic_periodic",
    "legacy",
    "init_diocotron_field",
    # Parametres diocotron canoniques (init_diocotron.m)
    "np=128",
    "omega_p=20",
    "omega_c=-20",
    "cfl=0.5",
    "hll",
    "euler",
    "electrostatic",
    "magnetostatic",
    # Decisions explicites exigees par les criteres d'acceptation
    "dmax",
    "diag(d)",
    "init_magnetic_wave_field",
    "meshgrid",
    "transposed",
    "--ic-matlab-bug",
    "hllc",
    "weno",
    "bug-for-bug",
    "intent",
    # Relais vers la suite du milestone M8
    "adc-349",
    "adc-350",
    "adc-351",
    "adc-356",
]


def check() -> int:
    """Retourne 0 si REFERENCE.md est present, propre et complet, sinon 1."""
    violations: list[str] = []

    if not REFERENCE.exists():
        print("CHECK-REFERENCE : REFERENCE.md MANQUANT", file=sys.stderr)
        return 1

    text = REFERENCE.read_text(encoding="utf-8")
    if len(text) < MIN_CHARS:
        violations.append(
            "REFERENCE.md trop court (%d < %d caracteres) : verrouillage vide"
            % (len(text), MIN_CHARS)
        )
    if EM_DASH in text:
        violations.append(
            "REFERENCE.md : %d em-dash (U+2014) interdits" % text.count(EM_DASH)
        )
    non_ascii = sorted({c for c in text if ord(c) > 0x7F})
    if non_ascii:
        violations.append(
            "REFERENCE.md : caracteres non-ASCII interdits (note en-US) : "
            + " ".join("U+%04X" % ord(c) for c in non_ascii)
        )

    low = text.lower()
    missing = [tok for tok in REQUIRED_TOKENS if tok not in low]
    if missing:
        violations.append(
            "REFERENCE.md : points d'ADC-348 non documentes : " + ", ".join(missing)
        )

    if violations:
        print(
            "CHECK-REFERENCE : %d violation(s)" % len(violations), file=sys.stderr
        )
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1

    print(
        "CHECK-REFERENCE : OK (REFERENCE.md present, sans em-dash, %d criteres "
        "ADC-348 couverts)" % len(REQUIRED_TOKENS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(check())
