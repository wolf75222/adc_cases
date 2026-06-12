"""Compilation a la volee + chargement ctypes des scenarios C++ sur mesure.

Certains cas portent leur propre C++ (un scenario qui n'est pas une brique generique du coeur,
p.ex. l'integrateur deux-fluides AP). Ce module factorise la mecanique commune :

  - localiser les en-tetes du coeur adc_cpp (`adc_include`) ;
  - compiler le source en bibliotheque partagee, avec cache hors source (`build_shared`) ;
  - charger la lib et verifier la presence des symboles attendus (`load_symbols`).

Deux exigences fortes, par rapport a un simple `c++ -shared` ad hoc :

1. Le cache de build ne pollue jamais l'arborescence source : la .so et sa cle d'ABI vont dans
   `out/<cas>/build/` (cf. `adc_cases.common.io`), ignore par git. Aucun artefact compile ne
   reste a cote du .cpp.

2. Une incompatibilite d'ABI est explicite, jamais silencieuse. La lib en cache est indexee par
   une cle d'ABI (hash du compilateur, des flags, des sources, et de la signature de l'arbre
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
    """Compilateur C++ : $CXX, sinon (Windows) cl/clang-cl, (Unix) c++/g++/clang++. Leve si aucun."""
    if sys.platform == "win32":
        cxx = (os.environ.get("CXX") or shutil.which("cl") or shutil.which("clang-cl")
               or shutil.which("clang-cl", path=r"C:\Program Files\LLVM\bin"))
        if not cxx:
            raise RuntimeError("aucun compilateur C++ trouve (definir CXX, ou lancer depuis un "
                               "invite vcvars / installer MSVC ou clang-cl)")
        return cxx
    cxx = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
           or shutil.which("clang++"))
    if not cxx:
        raise RuntimeError("aucun compilateur C++ trouve (definir CXX, ou installer c++/g++/clang++)")
    return cxx


def _kokkos_root():
    """Racine d'une install Kokkos (ADC_KOKKOS_ROOT / Kokkos_ROOT / KOKKOS_ROOT), ou None."""
    for key in ("ADC_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        root = os.environ.get(key)
        if root and os.path.isfile(os.path.join(root, "include", "Kokkos_Core.hpp")):
            return root
    return None


def _libomp_prefix():
    """Prefixe Homebrew libomp sur macOS (pour -Xpreprocessor -fopenmp), ou None."""
    if sys.platform != "darwin":
        return None
    try:
        p = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        prefix = p.stdout.strip()
        if prefix and os.path.isdir(os.path.join(prefix, "lib")):
            return prefix
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _kokkos_build_flags():
    """adc_cpp est KOKKOS-ONLY : un solveur natif qui inclut les en-tetes adc (mesh/for_each) ne
    compile PAS sans Kokkos (for_each.hpp #error). Comme la .so est STANDALONE (chargee via ctypes,
    sans partager le runtime Kokkos de `_adc`), elle compile ET LINKE Kokkos elle-meme.

    Renvoie (compile_flags, link_flags). Compile_flags entre dans la cle d'ABI (recompilation si le
    backend Kokkos change). Leve si aucun Kokkos installe n'est visible (Serial suffit sur CPU)."""
    import glob
    root = _kokkos_root()
    if root is None:
        raise RuntimeError(
            "adc_cpp est Kokkos-only : un Kokkos installe est requis pour compiler le solveur natif. "
            "Pointe-le via ADC_KOKKOS_ROOT (ou Kokkos_ROOT), p.ex. `export ADC_KOKKOS_ROOT=/chemin/kokkos` "
            "(Serial suffit sur un poste CPU).")
    inc = os.path.join(root, "include")
    lib = os.path.join(root, "lib")
    if not os.path.isdir(lib) and os.path.isdir(os.path.join(root, "lib64")):
        lib = os.path.join(root, "lib64")
    if sys.platform == "win32":
        # Windows (cl/clang-cl) : Kokkos en DLL partagee -> linker les import libs (.lib) par chemin.
        # cl accepte -D/-I. Pas de -L/-Wl/-ldl/-fopenmp (Unix). La .dll standalone (ctypes) initialise
        # son propre runtime Kokkos contre kokkos*.dll (a cote de la .dll au chargement).
        cflags = ["-DADC_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc]
        lflags = []
        for comp in ("kokkoscore", "kokkoscontainers", "kokkossimd", "kokkosalgorithms"):
            cand = os.path.join(lib, comp + ".lib")
            if os.path.exists(cand):
                lflags.append(cand)
        return cflags, lflags
    cflags = ["-DADC_HAS_KOKKOS", "-I", inc]
    lflags = ["-L", lib, "-Wl,-rpath," + lib, "-ldl"]
    for comp in ("kokkoscore", "kokkoscontainers", "kokkossimd", "kokkosalgorithms"):
        if glob.glob(os.path.join(lib, "lib" + comp + ".*")):
            lflags.append("-l" + comp)
    # OpenMP (espace Kokkos OpenMP ; inoffensif sous Serial). macOS / AppleClang : -Xpreprocessor.
    # Standalone -> on LIE libomp (seul runtime OpenMP du process, pas de _adc partage ici).
    libomp = _libomp_prefix()
    if libomp is not None:
        cflags += ["-Xpreprocessor", "-fopenmp", "-I", os.path.join(libomp, "include")]
        lflags += ["-L", os.path.join(libomp, "lib"), "-lomp"]
    elif sys.platform == "darwin":
        cflags += ["-Xpreprocessor", "-fopenmp"]
        lflags.append("-lomp")
    else:
        cflags.append("-fopenmp")
        lflags.append("-fopenmp")
    return cflags, lflags


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
    """Compile `sources` en bibliotheque partagee, avec cache hors source indexe par cle d'ABI.

    - `case_name` : nom du cas (sous-dossier de `out/`) ; le cache vit dans `out/<cas>/build/`.
    - `sources`   : liste de chemins .cpp/.hpp (le premier .cpp est compile, les autres servent
                    de dependances pour la cle de cache).
    - `include`   : dossier include/ du coeur (defaut : `adc_include()`).
    - `flags`/`std` : flags de compilation et standard C++.

    Recompile uniquement si la lib en cache manque ou si sa cle d'ABI differe de la cle courante
    (compilateur, flags, sources, en-tetes du coeur). On ne recharge jamais une lib perimee en
    silence. Renvoie le chemin de la .so/.dylib.
    """
    if include is None:
        include = adc_include()
    cxx = _compiler()
    cpp = next((s for s in sources if s.endswith(".cpp")), None)
    if cpp is None:
        raise ValueError("build_shared : aucun source .cpp dans " + repr(sources))
    # adc_cpp est Kokkos-only : le solveur natif inclut les en-tetes adc (mesh/for_each) et exige donc
    # Kokkos. Les flags de COMPILATION entrent dans la cle d'ABI (recompilation si le backend change) ;
    # les flags de LIEN (libs Kokkos + OpenMP) s'ajoutent a la commande (la .so est standalone, ctypes).
    kokkos_cflags, kokkos_lflags = _kokkos_build_flags()
    if sys.platform == "win32":
        # cl/clang-cl : .dll standalone (ctypes). /O2 remplace les flags -O* Unix. /permissive- +
        # /Zc:preprocessor requis par la STL/Kokkos modernes ; /DNOMINMAX (windows.h via adc).
        full_flags = ["/nologo", "/LD", "/std:" + std, "/O2", "/EHsc",
                      "/permissive-", "/Zc:preprocessor", "/DNOMINMAX", "/bigobj", *kokkos_cflags]
    else:
        full_flags = ["-shared", "-fPIC", "-std=" + std, *flags, *kokkos_cflags]

    # Dossier des en-tetes communs aux cas (common/), pour que les sources trouvent case_export.h
    # (macro ADC_CASE_EXPORT : __declspec(dllexport) portable sur les entry points C charges par ctypes).
    # Dans full_flags -> capture par la cle d'ABI (recompilation si le contrat d'export change).
    full_flags = [*full_flags, "-I", os.path.dirname(os.path.abspath(__file__))]

    cache = os.path.join(case_output_dir(case_name), "build")
    os.makedirs(cache, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    base = os.path.splitext(os.path.basename(cpp))[0]
    lib = os.path.join(cache, base + suffix)
    keyfile = lib + ".abikey"

    want = _abi_key(cxx, full_flags, sources, include)
    have = None
    if os.path.exists(lib) and os.path.exists(keyfile):
        with open(keyfile) as f:
            have = f.read().strip()

    if have != want:
        if sys.platform == "win32":
            cmd = [cxx, *full_flags, "-I", include, cpp, "/Fe:" + lib,
                   "/Fo" + cache + os.sep, "/link", *kokkos_lflags]
        else:
            cmd = [cxx, *full_flags, "-I", include, cpp, "-o", lib, *kokkos_lflags]
        print("%s : (re)compilation du solveur natif\n  %s" % (case_name, " ".join(cmd)))
        subprocess.run(cmd, check=True)
        with open(keyfile, "w") as f:
            f.write(want)
    return lib


def load_symbols(lib_path, symbols):
    """Charge `lib_path` (ctypes) et verifie que tous les `symbols` attendus existent.

    Un symbole manquant signe une ABI incompatible (lib obsolete ou source divergent) : on leve
    une RuntimeError explicite ici, au chargement, plutot qu'un AttributeError opaque au premier
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
