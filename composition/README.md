# composition : composer un systeme multi-blocs heterogene bloc par bloc

Cas **tutoriel** (`category = "tutoriel"`, `ci = true`, `needs = []` dans
[`cases_manifest.toml`](../cases_manifest.toml)). Il **demontre une CAPACITE de l'API**, il
ne reproduit AUCUN resultat publie et ne porte AUCUN claim physique : les CI sont des
profils simples (cosinus), les temps d'integration sont courts, et le seul but est de
montrer que l'on peut assembler un systeme **un bloc d'equation a la fois**, choisir un
**schema numerique different par bloc**, et meme **ecrire son propre integrateur temporel en
Python**. Aucune figure, aucun GIF, aucun fichier de sortie : tout est imprime (`print`) et
verifie par `assert`.

Le script vit dans [`run.py`](run.py). Il s'appuie sur les **modeles d'espece** de
[`adc_cases/models.py`](../adc_cases/models.py) (compositions de briques) et la grille de
[`adc_cases/common/grid.py`](../adc_cases/common/grid.py).

---

## 1. Objectif du cas

Montrer le niveau d'abstraction vise par l'API `adc` : **Python compose, le C++ calcule**.
Le script enchaine quatre demonstrations independantes (`partie_A` a `partie_D`) :

- **(A) Composition heterogene** : deux especes coexistent dans un meme `adc.System`, chacune
  avec son modele physique, sa reconstruction spatiale, son flux de Riemann, son traitement
  temporel et son nombre de sous-pas. Electrons (Euler compressible, VanLeer + HLLC, IMEX,
  10 sous-pas) et ions (Euler isotherme, Minmod + Rusanov, explicite, 1 sous-pas). On verifie
  que chaque espece conserve sa masse, que le Poisson couple est actif (potentiel non nul),
  et que les electrons evoluent.
- **(B) Determinisme bit a bit** : un MEME modele diocotron, recompose deux fois a partir des
  memes briques avec la meme CI, donne deux densites **identiques au bit pres** (ecart
  exactement 0.0). La composition de briques est reproductible.
- **(C) Garde-fous** : trois combinaisons invalides (flux HLLC sur un transport scalaire ;
  source fluide sur un transport scalaire ; etat scalaire avec flux compressible) doivent
  lever une **erreur claire**, et non planter ou s'executer en silence.
- **(D) Integrateur temporel ecrit en Python** : on avance un bloc diocotron avec un SSPRK2
  ecrit en Python (`adc.integrate.ssprk2_step`), sans appeler `sim.advance(...)`. Le schema en
  temps est en Python (par pas), le residu `-div F + S` et le Poisson restent en C++ (par
  cellule).

Le script imprime `OK composition_api` en cas de succes. Ce label est purement diagnostique.

## 2. Equations

Le cas n'introduit pas de nouvelle physique : il **reutilise** les briques generiques d'`adc`,
exposees par trois modeles nommes (cf. [`adc_cases/models.py`](../adc_cases/models.py)). Sur
le domaine periodique `[0, L]^2` :

**Electrons (Euler compressible)** -- `models.electron_euler(charge=-1.0, gamma=1.4)` :
etat conservatif `U_e = (rho, rho u, rho v, E)`,

```
d_t rho       + div(rho v)                 = 0
d_t (rho v)   + div(rho v⊗v + p I)         = (q/m) rho E
d_t E         + div((E + p) v)             = (q/m) rho v · E
p = (gamma - 1) (E - 1/2 rho |v|^2),   q = -1
```

**Ions (Euler isotherme)** -- `models.ion_isothermal(charge=1.0, cs2=0.5)` :
etat `U_i = (rho, rho u, rho v)`,

```
d_t rho       + div(rho v)                 = 0
d_t (rho v)   + div(rho v⊗v + cs2 rho I)   = (q/m) rho E,   q = +1
```

**Diocotron (transport scalaire ExB)** -- `models.diocotron(B0, alpha, n_i0)` :
etat scalaire `n`,

```
d_t n + div(v_E n) = 0,   v_E = (E × B) / B0^2 = (-d_y phi, d_x phi) / B0
```

**Couplage elliptique (Poisson)** -- le champ `E = -grad phi` est self-consistant. Le second
membre est la SOMME des briques elliptiques portees par les blocs :

- electrons/ions portent `ChargeDensity(charge=q)` -> contribuent `q n` ;
- diocotron porte `BackgroundDensity(alpha, n0=n_i0)` -> contribue `alpha (n - n_i0)` (fond
  neutralisant qui rend le Poisson periodique solvable, charge nette nulle).

```
lap phi = f,   f = somme_blocs (contribution elliptique du bloc),   E = -grad phi
```

> Honnetete : ce sont des modeles JOUETS pour le tutoriel. La partie (A) n'est PAS un plasma
> physique calibre ; la partie (B)/(D) n'est PAS une reproduction du benchmark diocotron
> (arXiv:2510.11808) -- ce dernier vit dans [`../diocotron/`](../diocotron/) (categorie
> `reproduction`) avec un anneau de charge et des figures. Ici, on ne mesure aucun taux de
> croissance.

## 3. Modele physique

Un modele d'espece est une **composition de quatre briques generiques** assemblees par
`adc.Model(state, transport, source, elliptic)` (defini dans le module `adc` fourni par
`adc_cpp`, fichier `python/adc/__init__.py`, fonction `Model`, qui valide la coherence
etat <-> transport et renvoie une `ModelSpec`) :

| Modele (`models.py`)         | state                                  | transport          | source                     | elliptic                            |
|------------------------------|----------------------------------------|--------------------|----------------------------|-------------------------------------|
| `electron_euler()`           | `FluidState(kind="compressible", gamma=1.4)` | `CompressibleFlux()` | `PotentialForce(charge=-1.0)` | `ChargeDensity(charge=-1.0)`        |
| `ion_isothermal()`           | `FluidState(kind="isothermal", cs2=0.5)`     | `IsothermalFlux()`   | `PotentialForce(charge=1.0)`  | `ChargeDensity(charge=1.0)`         |
| `diocotron(B0, alpha, n_i0)` | `Scalar()`                             | `ExB(B0)`          | `NoSource()`               | `BackgroundDensity(alpha, n0=n_i0)` |

La parametrisation physique (gamma, cs2, B0, charge, fond n_i0) vit **dans les briques**, pas
dans la config du systeme. `adc.System(n, L, periodic)` ne porte que le **maillage**.

## 4. Methode numerique

Le choix numerique est fait **par bloc**, independamment du modele physique :

- **Reconstruction spatiale** (`adc.Spatial(limiter=...)`) : `none` / `minmod` / `vanleer`
  (et `weno5`, non utilise ici). Limiteur MUSCL applique sur les variables `conservative` ou
  `primitive`.
- **Flux numerique de Riemann** (`adc.Spatial(flux=...)`) : `rusanov` (robuste, tout
  transport) ou `hllc` (onde de contact, **exige** un transport compressible 4 variables +
  pression). `roe` existe aussi.
- **Traitement temporel** :
  - `adc.Explicit(substeps=1, method="ssprk2")` : SSPRK2 (Shu-Osher 2 etages, ordre 2) sur tout
    le bloc ;
  - `adc.IMEX(substeps=1)` : transport explicite (SSPRK) + **source raide implicite**
    (backward-Euler, Newton local a la cellule). Ce n'est PAS un solveur implicite global PDE :
    seule la source est implicite (cf. docstring `IMEX` dans la facade).
- **Sous-pas** (`substeps=N`) : le bloc avance N fois par macro-pas, chaque sous-pas de
  longueur `dt/N` (electrons rapides -> `substeps=10`).
- **Poisson** : operateur `div(eps grad)` a `eps = 1` (Poisson), second membre `charge_density`
  (somme des briques elliptiques), solveur `geometric_mg` (multigrille geometrique), BC `auto`
  (periodique ici).

En **partie (D)**, le schema en temps est **ecrit en Python** : `adc.integrate.ssprk2_step`
(module `adc`, fichier `python/adc/integrate.py`) assemble les
etages RK en Python en s'appuyant sur les primitives `solve_fields` / `eval_rhs` /
`get_state` / `set_state`. Le residu et le Poisson restent calcules en C++. L'integrateur
re-resout Poisson **a chaque etage RK** (couplage per-stage, plus precis que le couplage fige
par pas du `advance` compile).

## 5. Architecture ADC utilisee

```
Python (run.py)                         C++ compile (module adc / adc_cpp)
----------------------------------      -------------------------------------------
adc.System(n, L, periodic)          ->  contexte maillage (carre periodique)
sim.add_block(name, model, spatial, ->  fige une fermeture d'avancee compilee par bloc
              time)                      (assemble_rhs<Limiter, Flux>, Newton source IMEX)
sim.set_poisson(rhs, solver, bc)    ->  EllipticPhysicalModel (Poisson) de systeme
sim.set_density(name, array)        ->  ecrit l'etat du bloc
sim.solve_fields()                  ->  Poisson + E = -grad phi
sim.advance(dt, n)                  ->  boucle en temps COMPILEE (partie A, B)
adc.integrate.ssprk2_step(sim, dt)  ->  primitives eval_rhs / get_state / set_state
                                        (boucle en temps en PYTHON, partie D)
```

Point cle : **aucun callback Python dans le hot path**. Chaque bloc embarque une fermeture
d'avancee compilee, type-erased seulement au niveau de la liste de blocs. Le residu
(`-div F + S`), le Newton de la source implicite et le Poisson de systeme (`somme_s`
contribution elliptique) restent 100 % en C++, par cellule. Seul l'assemblage des etages RK
de la partie (D) est en Python, par pas.

Pour les details d'API (module `adc` d'`adc_cpp`) : `Model`, `Spatial`, `Explicit`, `IMEX`,
`System.add_block`, `System.set_poisson` dans `python/adc/__init__.py` ; `ssprk2_step` dans
`python/adc/integrate.py`.

## 6. Carte des fichiers

| Fichier                                                             | Role |
|--------------------------------------------------------------------|------|
| [`composition/run.py`](run.py)                                     | le cas : 4 parties (A/B/C/D), `print` + `assert`, pas d'assets. |
| [`adc_cases/models.py`](../adc_cases/models.py)                    | `electron_euler()`, `ion_isothermal()`, `diocotron()` (compositions de briques). |
| [`adc_cases/common/grid.py`](../adc_cases/common/grid.py)          | `meshgrid_xy(n, L)` : grille a centres de cellules, convention `field[j, i]`. |
| [`adc_cases/__init__.py`](../adc_cases/__init__.py)                | `ensure_importable()` (le script utilise un fallback `sys.path` direct equivalent). |
| [`cases_manifest.toml`](../cases_manifest.toml)                    | declare le cas : `tutoriel`, `ci = true`, `needs = []`. |
| Module `adc` (hors depot)                                          | bindings pybind11 d'`adc_cpp` ; fourni par `PYTHONPATH` (voir Prerequis). |
| `adc.integrate` (dans le module `adc`)                             | `euler_step`, `ssprk2_step` ecrits en Python (partie D). |

Le cas n'utilise PAS `recipes.py`, ni `initial_conditions.py`, ni `checks.py`, ni `io.py`,
ni `native.py` : il pose ses CI a la main (cosinus) et fait ses asserts en ligne.

## 7. Prerequis

- **Python 3.12** avec **numpy**.
- Le **module `adc`** (bindings d'`adc_cpp`) construit et disponible sur le `PYTHONPATH`. Sur
  cette machine, le build est dans
  `/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python`.
- Le **paquet `adc_cases`** importable. Soit installe (`pip install -e .` a la racine du
  depot, voie nominale CI), soit la racine du depot ajoutee au `PYTHONPATH` (le script tente
  `import adc_cases` et, a defaut, insere `os.path.dirname(os.path.dirname(__file__))` dans
  `sys.path`).
- **Aucun compilateur C++** requis (`needs = []`) : ce cas ne compile rien a la volee,
  contrairement aux cas `needs = ["cxx"]` (DSL / two_fluid_ap).
- Lancer avec le **meme interpreteur** que celui ayant compile le module (suffixe ABI
  `cpython-312`).

Construction du module (si absent), depuis `adc_cpp` :

```bash
cmake -B build-py -DADC_BUILD_PYTHON=ON
cmake --build build-py --target _adc -j
export PYTHONPATH=$PWD/build-py/python
```

## 8. Commande exacte

Commande reellement executee pour ce README (worktree + build local) :

```bash
cd /private/tmp/adc_cases-readmes/composition && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
/opt/homebrew/anaconda3/bin/python3.12 run.py
```

Forme generique (paquet `adc_cases` installe en editable, module `adc` sur le `PYTHONPATH`) :

```bash
export PYTHONPATH=<adc_cpp>/build-py/python
python3 composition/run.py
```

## 9. Explication du code par etapes

**En-tete** (`run.py` lignes 68-86) : import de `numpy` et `adc` ; import de
`adc_cases.models` (fallback `sys.path` si le paquet n'est pas installe) ; `MASS_TOL = 1e-10` ;
`_meshgrid_centres(n, L)` delegue a `meshgrid_xy`.

**`partie_A()` (composition heterogene, lignes 89-137)** :
1. `sim = adc.System(n=48, L=1.0, periodic=True)` -- config = maillage seul.
2. `sim.add_block("electrons", model=electron_euler(), spatial=Spatial(vanleer=True,
   flux="hllc"), time=IMEX(substeps=10))` -- bloc Euler, VanLeer + HLLC, IMEX, 10 sous-pas.
3. `sim.add_block("ions", model=ion_isothermal(), spatial=Spatial(minmod=True,
   flux="rusanov"), time=Explicit())` -- bloc isotherme, Minmod + Rusanov, explicite, 1 sous-pas.
4. `assert sim.n_species() == 2` ; impression des noms de blocs.
5. `sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")`.
6. CI : electrons perturbes `1 + 0.02 cos(2 pi X / L)`, ions uniformes `1.0`.
7. `sim.solve_fields()` puis `assert |phi|_max > 1e-8` (Poisson couple actif).
8. `sim.advance(0.001, 8)` (8 macro-pas), puis verifie : derive de masse electrons/ions
   `< MASS_TOL = 1e-10` (absolue) et evolution electrons `> 1e-9` (dynamique non triviale).

**`partie_B()` (determinisme, lignes 140-166)** : grille `n=32` ; CI cosinus
`1 + 0.1 cos(2 pi X / L)` ; fond `n_i0 = rho0.mean()`. La fonction `construire_et_avancer()`
recompose un `System` + un bloc `diocotron(B0=1.0, alpha=1.0, n_i0)` (Minmod + Rusanov,
explicite), `set_poisson()`, `set_density`, `advance(0.002, 12)`, et renvoie la densite. Deux
appels strictement identiques -> `assert max|da - db| == 0.0` (egalite EXACTE, bit a bit).

**`partie_C()` (garde-fous, lignes 169-209)** : helper `doit_lever(fn, why)` qui exige qu'une
exception soit levee. Trois cas invalides :
1. `Spatial(flux="hllc")` sur `diocotron()` (transport scalaire) -> rejet a `add_block`.
2. Modele cree avec `Scalar()` + `ExB()` + `PotentialForce()` (source fluide sur transport
   scalaire) -> rejet a `add_block`.
3. `adc.Model(state=Scalar(), transport=CompressibleFlux(), ...)` -> rejet **a la composition**
   (`adc.Model` valide etat <-> transport, leve `ValueError` immediatement).
   Enfin `assert sim.n_species() == 0` : aucun bloc invalide n'a ete ajoute.

**`partie_D()` (integrateur Python, lignes 212-244)** : grille `n=32` ; CI
`1 + 0.1 cos(2 pi X / L) sin(2 pi Y / L)` ; fond `n_i0 = rho0.mean()`. Un seul bloc
`diocotron` (Minmod, explicite), `set_poisson()`, `set_density`. **On n'appelle PAS
`sim.advance(...)`** : on boucle 20 fois sur `adc.integrate.ssprk2_step(sim, dt=0.001)`
(SSPRK2 ecrit en Python, Poisson re-resolu per-stage). Verifie : derive de masse `< 1e-9` et
etat fini (`np.isfinite(rho).all()`).

**`main()` (lignes 247-256)** : enchaine A, B, C, D, imprime un resume et `OK composition_api`.

## 10. Conditions initiales

Toutes les CI sont posees a la main en numpy (le cas n'utilise pas
`common.initial_conditions`). Convention de grille `field[j, i]` avec
`X, Y = meshgrid_xy(n, L)` (centres de cellules).

| Partie | Champ initial |
|--------|---------------|
| (A) electrons | `1.0 + 0.02 cos(2 pi X / L)` (perturbation cosinus) |
| (A) ions      | `np.ones((n, n))` (uniforme) |
| (B) diocotron | `1.0 + 0.1 cos(2 pi X / L)`, fond `n_i0 = rho0.mean()` |
| (D) diocotron | `1.0 + 0.1 cos(2 pi X / L) sin(2 pi Y / L)`, fond `n_i0 = rho0.mean()` |

Le fond neutralisant `n_i0 = rho0.mean()` (parties B/D) annule la charge nette : le Poisson
periodique est alors solvable (condition de compatibilite `integrale du second membre = 0`).
En partie (A), les electrons perturbes + les ions uniformes donnent une charge nette ~ 0
(electrons charge `-1`, ions charge `+1`, densites moyennes proches de 1).

## 11. Invariants et assertions

Toutes les verifications sont des `assert` en ligne dans `run.py`. Valeurs reelles capturees
lors de l'execution (voir section 12 et section 15) :

| Assert | Condition | Valeur mesuree |
|--------|-----------|----------------|
| (A) Poisson actif | `|phi|_max > 1e-8` | `5.062437e-04` |
| (A) masse electrons | `|m_e - m_e0| < 1e-10` | `2.728e-12` |
| (A) masse ions | `|m_i - m_i0| < 1e-10` | `1.819e-12` |
| (A) evolution electrons | `max|rho_e - rho_e0| > 1e-9` | `3.506e-05` |
| (B) determinisme | `max|da - db| == 0.0` (EXACT) | `0.000e+00` |
| (C) HLLC sur scalaire | `add_block` leve | `RuntimeError` (voir ci-dessous) |
| (C) source fluide sur scalaire | `add_block` leve | `RuntimeError` |
| (C) modele incoherent | `adc.Model` leve | `ValueError` |
| (C) aucun bloc invalide | `sim.n_species() == 0` | `0` |
| (D) masse (SSPRK2 Python) | `|m - m0| < 1e-9` | `2.274e-13` |
| (D) etat fini | `np.isfinite(rho).all()` | `True` |

Messages d'erreur EXACTS de la partie (C), capturees a l'execution (le `print` du script les
tronque a 70 caracteres ; voici la version complete) :

```
hllc/diocotron       -> RuntimeError : System : flux 'hllc' exige un transport compressible
                        (4 variables + pression) ; ce transport -> 'rusanov'
potential/scalar     -> RuntimeError : source 'potential' invalide ici (exige un transport
                        fluide >= 3 variables, ou 'none')
Scalar+CompressibleFlux -> ValueError : Scalar exige transport=ExB(...)
```

Les deux premieres erreurs (`RuntimeError`) viennent du C++ (`System::add_block`) ; la
troisieme (`ValueError`) vient de la facade Python `adc.Model` qui valide la composition avant
toute frontiere C++.

## 12. Sorties attendues

Aucun fichier produit : tout est imprime. Sortie complete reelle (execution du
2026-06-07 sur cette machine) :

```
== Partie (A) : un schema (modele/spatial/temps/sous-pas) par bloc ==
  n_species              = 2
  blocs                  = ['electrons', 'ions']
  electrons : electron_euler() | Spatial(vanleer, hllc) | IMEX(substeps=10)
  ions      : ion_isothermal() | Spatial(minmod, rusanov) | Explicit()
  |phi|_max (initial)    = 5.062437e-04
  derive masse electrons = 2.728e-12  (Euler/HLLC/IMEX, 10 sous-pas)
  derive masse ions      = 1.819e-12  (isotherme/Rusanov/explicite)
  evolution electrons    = 3.506e-05  (dynamique non triviale)
== Partie (B) : determinisme de la composition de briques (bit pour bit) ==
  ecart max (deux compositions independantes) = 0.000e+00
== Partie (C) : garde-fous des combinaisons invalides ==
  rejete (hllc sur diocotron (transport scalaire)) : System : flux 'hllc' exige un transport compressible (4 variables + pr
  rejete (source PotentialForce sur transport scalaire) : source 'potential' invalide ici (exige un transport fluide >= 3 variab
  rejete (modele incoherent (Scalar + CompressibleFlux)) : Scalar exige transport=ExB(...)
== Partie (D) : integrateur temporel custom en Python (SSPRK2) ==
  pas Python (SSPRK2)    = 20  (Poisson re-resolu per-stage)
  derive masse           = 2.274e-13  (integrateur ecrit en Python)
  etat fini              = True
Systeme compose bloc par bloc depuis Python (modeles = compositions de briques) ;
calcul 100 % C++ compile. Le schema en temps lui-meme peut etre ecrit en Python
(partie D), le calcul par cellule restant en C++.
OK composition_api
```

Les valeurs en virgule flottante (`5.062437e-04`, `2.728e-12`, etc.) peuvent varier au dernier
chiffre selon la plateforme et l'ordre de reduction, mais restent tres en dessous des seuils
d'assert. La partie (B) (`0.000e+00`) doit, elle, etre **exactement** zero sur une plateforme
donnee : c'est la garantie de determinisme du cas.

## 13. Generation figures/GIF

**Aucune.** Ce cas ne produit ni figure ni GIF ni fichier de sortie. Il n'importe pas
`matplotlib` (`needs = []`), n'utilise pas `adc_cases.common.io` et n'ecrit rien dans `out/`.
Toute la verification passe par `print` (diagnostics lisibles) et `assert` (invariants). Si
l'on cherche des figures de l'instabilite diocotron, voir le cas
[`../diocotron/`](../diocotron/) (categorie `reproduction`, `needs = ["matplotlib"]`).

## 14. Backends reellement supportes

- **CPU mono-rang uniquement** (chemin `System.add_block` natif). C'est le chemin de
  composition de briques natives (`ModelSpec`), sans `.so` compile a la volee, sans MPI, sans
  AMR, sans GPU dans ce cas.
- Le **solveur Poisson** utilise est `geometric_mg` (multigrille geometrique), valable en
  periodique et en paroi. L'alternative `fft` (periodique, `n = 2^k`) n'est pas demandee ici.
- **Limiteurs** disponibles sur ce chemin : `none` / `minmod` / `vanleer` / `weno5` (seuls
  `vanleer` et `minmod` sont exerces). **Flux** : `rusanov` / `hllc` / `roe` (`hllc` et
  `rusanov` exerces). **Temporel** : `Explicit` (SSPRK2) et `IMEX` (source implicite) exerces.
- La machine de test est **macOS arm64 (Darwin)**, module compile avec
  `_adc.cpython-312-darwin.so`. La CI tourne sur **ubuntu-latest** (build Release du module,
  `ADC_USE_EIGEN=OFF`). Le cas n'a aucune dependance materielle : il s'execute partout ou le
  module `adc` se construit.

## 15. Cout approximatif

Mesure reelle (temps mur) sur cette machine (macOS arm64, build `build-master`, Python 3.12 d'anaconda) :

```
PYTHONPATH=... python3.12 run.py
1.82s user 0.40s system 342% cpu 0.649 total
```

- **Temps mur total : ~0.65 s** (dont ~1.8 s CPU cumule, le multigrille exploitant plusieurs
  threads -> 342 % CPU). Negligeable, conforme a un cas `ci = true`.
- **Memoire** : quelques Mo (grilles `48x48` en partie A, `32x32` en B/D ; 4 a 1 composantes
  par bloc). Aucune allocation notable.
- **Pas de compilation** a la volee (`needs = []`), donc pas de surcout `c++` au premier
  lancement, contrairement aux cas DSL/`two_fluid_ap`.

## 16. Limites et differences avec les references

- **Cas tutoriel, AUCUN claim physique.** Les CI sont des cosinus jouets ; les temps
  d'integration sont volontairement courts (8 a 20 pas). Le cas ne mesure aucun taux de
  croissance, aucune quantite physique calibree. Il **demontre l'API**, pas un resultat.
- **Ce n'est PAS la reproduction du diocotron.** Le diocotron ici est un transport scalaire
  ExB avec un fond neutralisant `BackgroundDensity`, sur grille periodique carree, sans
  anneau de charge ni etude de mode. La reproduction du benchmark arXiv:2510.11808 (anneau,
  modes azimutaux, figures, GIF) vit dans [`../diocotron/`](../diocotron/) (categorie
  `reproduction`, hors CI) -- ne pas confondre.
- **IMEX = source-only.** Le `adc.IMEX(substeps=10)` de la partie (A) traite la SOURCE de
  maniere implicite (backward-Euler, Newton local), pas le transport ni le Poisson. Ce n'est
  PAS un solveur implicite global PDE (Newton-Krylov / Schur), qui est un chantier distinct.
- **Determinisme bit a bit (B)** : garanti pour une **plateforme et un build donnes** (meme
  ordre de reduction, meme compilateur). L'egalite exacte `== 0.0` est attendue a chaque
  re-execution sur la meme machine, mais les valeurs de la partie (A) peuvent varier au
  dernier ULP entre plateformes (sans franchir les seuils d'assert).
- **Couplage per-stage (D) different de `advance` (A/B).** L'integrateur Python re-resout
  Poisson a chaque etage RK (couplage hyperbolique/elliptique per-stage), alors que le
  `sim.advance(...)` compile peut figer le couplage par pas. Les deux chemins sont corrects
  mais ne sont pas censes etre bit-identiques entre eux.
- **Mono-rang, pas de GPU/MPI/AMR** sur ce cas (cf. section 14).

## 17. Tests/CI associes

- **Manifeste** : [`cases_manifest.toml`](../cases_manifest.toml) declare ce cas
  `path = "composition/run.py"`, `category = "tutoriel"`, `ci = true`, `needs = []`.
- **CI** : [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) clone `adc_cpp`, construit
  le module `_adc` (CMake, Release, `ADC_BUILD_PYTHON=ON`, `ADC_USE_EIGEN=OFF`), installe
  `adc_cases` en editable, puis lit le manifeste et lance **uniquement** les cas `ci = true`
  avec `python3 <path>`. Ce cas en fait partie : `python3 composition/run.py` doit sortir 0 et
  imprimer `OK composition_api`.
- **Mecanisme de test** : il n'y a pas de framework de test externe (pas de pytest). Le cas
  EST son propre test : chaque invariant est un `assert`, et un `assert` qui echoue provoque
  une `AssertionError` -> code de sortie non nul -> CI rouge. Les garde-fous de la partie (C)
  testent par construction que les combinaisons invalides levent bien.
- Pour relancer en local exactement comme la CI : installer le paquet (`pip install -e .` a la
  racine), exporter `PYTHONPATH=<adc_cpp>/build-py/python`, puis `python3 composition/run.py`.
