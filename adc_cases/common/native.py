"""Compilation a la volee + chargement ctypes des scenarios C++ sur mesure.

Certains cas portent leur propre C++ (un scenario qui n'est PAS une brique generique du coeur,
p.ex. l'integrateur deux-fluides AP). Ce module factorise la mecanique commune :

  - localiser les en-tetes du coeur adc_cpp (`adc_include`) ;
  - compiler le source en bibliotheque partagee, AVEC CACHE HORS SOURCE (`build_shared`) ;
  - charger la lib et verifier la presence des symboles attendus (`load_symbols`).

Deux exigences fortes, par rapport a un simple `c++ -shared` ad hoc :

1. Le cache de build ne pollue JAMAIS l'arborescence source : la .so et sa cle d'ABI vont dans
   `out/<cas>/build/` (cf. `adc_cases.common.io`), ignore par git. Aucun artefact compile ne
   reste a cote du .cpp.

2. Une incompatibilite d'ABI est EXPLICITE, jamais silencieuse. La lib en cache est indexee par
   une CLE d'ABI (hash du compilateur, des flags, des sources, ET de la signature de l'arbre
   d'en-tetes du coeur). Si la cle change (en-tetes du coeur modifies = ABI potentiellement
   differente), la lib est recompilee ; on ne recharge jamais une lib perimee. Au chargement, on
   verifie que tous les symboles attendus existent : un symbole manquant leve une erreur claire
   au lieu d'un `AttributeError` opaque au premier appel.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys

import adc

from .io import case_output_dir


def adc_include():
    """Renvoie le dossier include/ d'adc_cpp (en-tetes header-only du coeur).

    Priorite a $ADC_INCLUDE (override explicite) ; sinon on remonte depuis le paquet `adc`
    installe (build-py/python/adc/ -> ../../../include) ; en dernier recours, le depot voisin
    ../adc_cpp/include depuis adc_cases. On exige que adc/mesh/multifab.hpp existe.
    """
    here = os.path.dirname(os.path.abspath(__file__))           # adc_cases/common
    repo = os.path.dirname(os.path.dirname(here))               # racine du depot
    candidates = []
    env = os.environ.get("ADC_INCLUDE")
    if env:
        candidates.append(env)
    pkg = os.path.dirname(os.path.abspath(adc.__file__))        # .../adc
    candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    candidates.append(os.path.normpath(os.path.join(repo, "..", "adc_cpp", "include")))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "adc", "mesh", "multifab.hpp")):
            return c
    raise RuntimeError(
        "en-tetes adc_cpp introuvables (cherche adc/mesh/multifab.hpp). "
        "Definir ADC_INCLUDE=<adc_cpp>/include. Candidats essayes : " + ", ".join(candidates))


def _compiler():
    """Compilateur C++ : $CXX, sinon c++ / g++ / clang++. Leve si aucun n'est trouve."""
    cxx = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
           or shutil.which("clang++"))
    if not cxx:
        raise RuntimeError("aucun compilateur C++ trouve (definir CXX, ou installer c++/g++/clang++)")
    return cxx


def _include_signature(include):
    """Signature de l'arbre d'en-tetes du coeur (chemin relatif, taille, mtime de chaque .hpp).

    Sert de proxy d'ABI : si un en-tete du coeur change, la signature change, donc la cle de
    cache change et la lib est recompilee. Evite de recharger une .so liee a une ABI perimee.
    """
    parts = []
    for root, _dirs, files in os.walk(include):
        for name in sorted(files):
            if name.endswith((".hpp", ".h")):
                p = os.path.join(root, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                parts.append("%s:%d:%d" % (os.path.relpath(p, include), st.st_size,
                                           int(st.st_mtime)))
    return "\n".join(sorted(parts))


def _abi_key(cxx, flags, sources, include):
    """Cle d'ABI : hash stable du compilateur, des flags, du contenu des sources et de la
    signature de l'arbre d'en-tetes du coeur. Deux builds avec la meme cle sont interchangeables."""
    h = hashlib.sha256()
    h.update(("cxx=" + cxx + "\n").encode())
    h.update(("flags=" + " ".join(flags) + "\n").encode())
    h.update(("py=" + sys.version + "\n").encode())
    for src in sources:
        with open(src, "rb") as f:
            h.update(("src=" + os.path.basename(src) + "\n").encode())
            h.update(f.read())
    h.update(("include_sig=\n" + _include_signature(include)).encode())
    return h.hexdigest()


def build_shared(case_name, sources, include=None, flags=("-O2",), std="c++20"):
    """Compile `sources` en bibliotheque partagee, avec CACHE HORS SOURCE indexe par cle d'ABI.

    - `case_name` : nom du cas (sous-dossier de `out/`) ; le cache vit dans `out/<cas>/build/`.
    - `sources`   : liste de chemins .cpp/.hpp (le premier .cpp est compile, les autres servent
                    de dependances pour la cle de cache).
    - `include`   : dossier include/ du coeur (defaut : `adc_include()`).
    - `flags`/`std` : flags de compilation et standard C++.

    Recompile UNIQUEMENT si la lib en cache manque ou si sa cle d'ABI differe de la cle courante
    (compilateur, flags, sources, en-tetes du coeur). On ne recharge jamais une lib perimee en
    silence. Renvoie le chemin de la .so/.dylib.
    """
    if include is None:
        include = adc_include()
    cxx = _compiler()
    cpp = next((s for s in sources if s.endswith(".cpp")), None)
    if cpp is None:
        raise ValueError("build_shared : aucun source .cpp dans " + repr(sources))
    full_flags = ["-shared", "-fPIC", "-std=" + std, *flags]

    cache = os.path.join(case_output_dir(case_name), "build")
    os.makedirs(cache, exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    base = os.path.splitext(os.path.basename(cpp))[0]
    lib = os.path.join(cache, base + suffix)
    keyfile = lib + ".abikey"

    want = _abi_key(cxx, full_flags, sources, include)
    have = None
    if os.path.exists(lib) and os.path.exists(keyfile):
        with open(keyfile) as f:
            have = f.read().strip()

    if have != want:
        cmd = [cxx, *full_flags, "-I", include, cpp, "-o", lib]
        print("%s : (re)compilation du solveur natif\n  %s" % (case_name, " ".join(cmd)))
        subprocess.run(cmd, check=True)
        with open(keyfile, "w") as f:
            f.write(want)
    return lib


def load_symbols(lib_path, symbols):
    """Charge `lib_path` (ctypes) et verifie que tous les `symbols` attendus existent.

    Un symbole manquant signe une ABI incompatible (lib obsolete ou source divergent) : on leve
    une RuntimeError EXPLICITE ici, au chargement, plutot qu'un AttributeError opaque au premier
    appel. Renvoie l'objet CDLL charge.
    """
    lib = ctypes.CDLL(lib_path)
    missing = []
    for name in symbols:
        if not hasattr(lib, name):
            missing.append(name)
    if missing:
        raise RuntimeError(
            "ABI incompatible pour %s : symboles attendus absents : %s. "
            "Le cache est probablement perime ; supprimer %s pour forcer une recompilation."
            % (lib_path, ", ".join(missing), os.path.dirname(lib_path)))
    return lib
