#!/usr/bin/env python3
"""Demo "two_fluid_ap" : modele bi-fluide isotherme en regime RAIDE (asymptotic-preserving).

L'integrateur AP-IMEX deux-fluides a QUITTE le coeur adc_cpp : c'est un SCENARIO sur mesure,
pas une brique generique composable bloc-a-bloc comme `adc.System`. La stabilisation AP couple
la raideur (frequence plasma) au pas de temps DANS l'elliptique (Poisson reformule
lap(phi) = (ne* - ni*)/(1 + dt^2 (wpe^2 + wpi^2))), ce que la composition System ne sait pas
reproduire. Le solveur vit donc ICI, dans adc_cases : la physique est en C++ (two_fluid_ap.hpp +
_two_fluid_ap.cpp) et n'utilise du coeur que des BRIQUES GENERIQUES (maillage, elliptique,
parallele) incluses depuis adc_cpp/include. Ce module .cpp est compile A LA VOLEE en une
bibliotheque partagee et charge dans CE process via ctypes (meme principe que le JIT du DSL),
puis pilote depuis Python : aucun binding C++ cote cas, et le coeur ne nomme aucun scenario.

Il integre deux fluides charges (electrons + ions) couples au champ electrique par la
contrainte de quasi-neutralite. La frequence plasma omega_pe fixe l'echelle de temps RAIDE
du systeme : un schema explicite serait limite par dt * omega_pe < O(1), donc exploserait
des qu'on prend un grand pas de temps.

Le solveur emploie un traitement IMEX / asymptotic-preserving (AP) : le terme raide
est integre de maniere implicite, ce qui rend le schema STABLE et CONSISTANT meme
quand dt * omega_pe >> 1. On le demontre sur deux scenarios.

  Run 1 "raide" (non magnetise) :
      omega_pe = 1e3, omega_pi = 20, advance(5.0/1e3, 200)
      => dt * omega_pe = 5 (un schema explicite EXPLOSERAIT).
      Invariants verifies :
        - l'ecart a la quasi-neutralite max_dev() reste PETIT  (< 0.1) ;
        - la charge nette locale max_charge() reste PETITE      (< 0.1) ;
        - la masse electronique mass_e est conservee            (erreur relative < 1e-7).

  Run 2 "magnetise" :
      omega_ce = 4, omega_ci = 0.2, advance(0.01, 100)
      => terme de rotation cyclotron actif ; on verifie la conservation de la masse
         electronique (erreur relative < 1e-7).

Conclusion : le schema IMEX / AP reste stable et conservatif pour un plasma raide,
magnetise ou non, la ou un schema explicite serait inutilisable.

Sortie : diagnostics numeriques imprimes, puis la ligne finale "OK two_fluid_ap".
Dependances : numpy + un compilateur C++20 (le solveur AP est compile a la volee).
"""

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

import adc  # le coeur : on s'en sert pour localiser ses en-tetes (adc_cpp/include)


HERE = os.path.dirname(os.path.abspath(__file__))


# --- Localisation des en-tetes du coeur adc_cpp -------------------------------------------
def _adc_include():
    """Renvoie le dossier include/ d'adc_cpp (en-tetes header-only du coeur).

    Priorite a $ADC_INCLUDE (override explicite) ; sinon on remonte depuis le paquet `adc`
    installe (build-py/python/adc/ -> ../../../include) ; en dernier recours, le depot
    voisin ../adc_cpp/include depuis adc_cases. On exige que adc/mesh/multifab.hpp existe.
    """
    candidates = []
    env = os.environ.get("ADC_INCLUDE")
    if env:
        candidates.append(env)
    pkg = os.path.dirname(os.path.abspath(adc.__file__))  # .../adc
    candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    candidates.append(os.path.normpath(os.path.join(HERE, "..", "..", "adc_cpp", "include")))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "adc", "mesh", "multifab.hpp")):
            return c
    raise RuntimeError(
        "two_fluid_ap : en-tetes adc_cpp introuvables (cherche adc/mesh/multifab.hpp). "
        "Definir ADC_INCLUDE=<adc_cpp>/include. Candidats essayes : " + ", ".join(candidates))


# --- Compilation a la volee du solveur AP -------------------------------------------------
def _build_lib(include):
    """Compile _two_fluid_ap.cpp en bibliotheque partagee et la charge (ctypes).

    Recompile seulement si la .so manque ou est plus vieille que les sources (cache local).
    """
    cxx = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
           or shutil.which("clang++"))
    if not cxx:
        print("two_fluid_ap : aucun compilateur C++ trouve (definir CXX) -> abandon")
        sys.exit(1)
    cpp = os.path.join(HERE, "_two_fluid_ap.cpp")
    hpp = os.path.join(HERE, "two_fluid_ap.hpp")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = os.path.join(HERE, "_two_fluid_ap" + suffix)
    stale = (not os.path.exists(lib) or
             os.path.getmtime(lib) < max(os.path.getmtime(cpp), os.path.getmtime(hpp)))
    if stale:
        cmd = [cxx, "-shared", "-fPIC", "-std=c++20", "-O2", "-I", include, cpp, "-o", lib]
        print("two_fluid_ap : compilation du solveur AP\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    return ctypes.CDLL(lib)


def _bind(lib):
    """Declare les signatures ctypes des fonctions extern "C" du solveur AP."""
    dptr = ctypes.POINTER(ctypes.c_double)
    lib.tfap_create.restype = ctypes.c_void_p
    lib.tfap_create.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                                ctypes.c_int, ctypes.c_double, ctypes.c_double]
    lib.tfap_destroy.restype = None
    lib.tfap_destroy.argtypes = [ctypes.c_void_p]
    lib.tfap_step.restype = None
    lib.tfap_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.tfap_advance.restype = None
    lib.tfap_advance.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    lib.tfap_nx.restype = ctypes.c_int
    lib.tfap_nx.argtypes = [ctypes.c_void_p]
    for name in ("tfap_mass_e", "tfap_mass_i", "tfap_max_charge", "tfap_max_dev"):
        f = getattr(lib, name)
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_void_p]
    for name in ("tfap_density_e", "tfap_density_i"):
        f = getattr(lib, name)
        f.restype = None
        f.argtypes = [ctypes.c_void_p, dptr]
    return lib


class TwoFluidAP:
    """Pilote Python du solveur deux-fluides AP (C++ charge via ctypes).

    Remplace l'ancien echappatoire interne `adc._adc._TwoFluidAP` (retire du coeur) :
    meme API (step / advance / mass_e / max_charge / ...), mais le solveur vit dans adc_cases.
    """

    def __init__(self, lib, n=64, L=2.0 * np.pi, cse2=1.0, csi2=0.04, omega_pe=5.0,
                 omega_pi=1.0, stabilize=True, eps=1e-3, upwind_continuity=False,
                 omega_ce=0.0, omega_ci=0.0):
        self._lib = lib
        self._h = lib.tfap_create(n, L, cse2, csi2, omega_pe, omega_pi, int(stabilize), eps,
                                  int(upwind_continuity), omega_ce, omega_ci)
        if not self._h:
            raise RuntimeError("two_fluid_ap : tfap_create a echoue")

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.tfap_destroy(h)
            self._h = None

    def step(self, dt):
        self._lib.tfap_step(self._h, dt)

    def advance(self, dt, nsteps):
        self._lib.tfap_advance(self._h, dt, int(nsteps))

    def nx(self):
        return self._lib.tfap_nx(self._h)

    def mass_e(self):
        return self._lib.tfap_mass_e(self._h)

    def mass_i(self):
        return self._lib.tfap_mass_i(self._h)

    def max_charge(self):
        return self._lib.tfap_max_charge(self._h)

    def max_dev(self):
        return self._lib.tfap_max_dev(self._h)

    def _density(self, fn):
        n = self.nx()
        out = np.empty(n * n, dtype=np.float64)
        fn(self._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out.reshape(n, n)

    def density_e(self):
        return self._density(self._lib.tfap_density_e)

    def density_i(self):
        return self._density(self._lib.tfap_density_i)


def _rel_err(a, b):
    """Erreur relative robuste (denominateur protege contre le zero)."""
    return abs(a - b) / max(abs(b), 1e-30)


def run_stiff(lib):
    """Run 1 : regime raide non magnetise, dt * omega_pe = 5."""
    omega_pe = 1.0e3   # frequence plasma electronique : echelle de temps RAIDE
    omega_pi = 20.0    # frequence plasma ionique
    n = 64
    solver = TwoFluidAP(lib, n=n, omega_pe=omega_pe, omega_pi=omega_pi)

    dt = 5.0 / 1.0e3                 # dt choisi tel que dt * omega_pe = 5
    stiffness = dt * omega_pe        # nombre de raideur (>> 1 ici)
    mass_e0 = solver.mass_e()

    solver.advance(dt, 200)

    max_dev = solver.max_dev()       # ecart max a la quasi-neutralite
    max_charge = solver.max_charge() # charge nette locale max
    mass_e = solver.mass_e()
    mass_rel = _rel_err(mass_e, mass_e0)

    print("[run 1 - raide, non magnetise]")
    print("  n=%d  omega_pe=%.3e  omega_pi=%.3e" % (n, omega_pe, omega_pi))
    print("  dt=%.3e  nsteps=200  dt*omega_pe=%.1f  (explicite EXPLOSERAIT)" % (dt, stiffness))
    print("  max_dev()    = %.6e   (ecart a la quasi-neutralite)" % max_dev)
    print("  max_charge() = %.6e   (charge nette locale)" % max_charge)
    print("  mass_e: %.6e -> %.6e   (err. relative %.3e)" % (mass_e0, mass_e, mass_rel))

    # --- Invariants physiques (propriete AP) ---
    # Le grand pas de temps a bien ete fait sans exploser : valeurs finies.
    assert np.isfinite(max_dev), "max_dev non fini : le schema a explose"
    assert np.isfinite(max_charge), "max_charge non fini : le schema a explose"
    assert np.isfinite(mass_e), "mass_e non finie : le schema a explose"
    # Propriete AP : la quasi-neutralite est maintenue malgre dt*omega_pe = 5.
    assert max_dev < 0.1, "max_dev trop grand (%.3e) : quasi-neutralite non maintenue" % max_dev
    assert max_charge < 0.1, "max_charge trop grand (%.3e)" % max_charge
    # Conservation de la masse electronique.
    assert mass_rel < 1e-7, "masse electronique non conservee (err. rel. %.3e)" % mass_rel

    return max_dev, max_charge, mass_rel


def run_magnetized(lib):
    """Run 2 : plasma raide magnetise (rotation cyclotron active)."""
    omega_ce = 4.0   # frequence cyclotron electronique
    omega_ci = 0.2   # frequence cyclotron ionique
    n = 64
    solver = TwoFluidAP(lib, n=n, omega_ce=omega_ce, omega_ci=omega_ci)

    dt = 0.01
    mass_e0 = solver.mass_e()

    solver.advance(dt, 100)

    max_dev = solver.max_dev()
    max_charge = solver.max_charge()
    mass_e = solver.mass_e()
    mass_rel = _rel_err(mass_e, mass_e0)

    print("[run 2 - raide magnetise]")
    print("  n=%d  omega_ce=%.3e  omega_ci=%.3e" % (n, omega_ce, omega_ci))
    print("  dt=%.3e  nsteps=100" % dt)
    print("  max_dev()    = %.6e" % max_dev)
    print("  max_charge() = %.6e" % max_charge)
    print("  mass_e: %.6e -> %.6e   (err. relative %.3e)" % (mass_e0, mass_e, mass_rel))

    # --- Invariants physiques ---
    assert np.isfinite(max_dev), "max_dev non fini : le schema a explose"
    assert np.isfinite(mass_e), "mass_e non finie : le schema a explose"
    # Conservation de la masse electronique sous rotation cyclotron.
    assert mass_rel < 1e-7, "masse electronique non conservee (err. rel. %.3e)" % mass_rel

    return max_dev, max_charge, mass_rel


def main():
    print("=== Demo two_fluid_ap : bi-fluide isotherme raide (asymptotic-preserving) ===")
    lib = _bind(_build_lib(_adc_include()))
    run_stiff(lib)
    run_magnetized(lib)
    print("Conclusion : schema IMEX / asymptotic-preserving stable et conservatif")
    print("pour un plasma raide, magnetise ou non (un schema explicite echouerait).")
    print("OK two_fluid_ap")


if __name__ == "__main__":
    main()
