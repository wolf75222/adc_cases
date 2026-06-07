# euler_poisson — Euler compressible couple a Poisson : auto-gravite (attractif) vs plasma (repulsif)

Cas de **validation** (manifeste `cases_manifest.toml`, categorie `validation`, `ci = true`,
`needs = []`). Ce n'est PAS une reproduction d'un resultat publie : c'est un test numerique leger
qui verifie par `assert` trois invariants physiques (masse conservee, impulsion nette nulle,
contraste de signe de la derive d'energie entre force attractive et force repulsive).

---

## 1. Objectif du cas

Demontrer, sur un systeme generique `adc.System` compose d'**un seul bloc** de modele
`euler_poisson`, que :

1. le solveur natif (C++) integre les equations d'Euler compressibles 2D couplees a une equation
   de Poisson pour le champ de force **auto-consistant** ;
2. ce couplage est **piloté par un signe** passe a la brique elliptique : `sign = +1` donne une
   force **attractive** (auto-gravite), `sign = -1` une force **repulsive** (charge d'espace,
   plasma) ;
3. les deux runs (memes CI, meme grille, meme schema, seul le signe change) produisent des derives
   d'energie de **signes opposes** et physiquement attendus : auto-gravite -> `dE < 0`, plasma ->
   `dE > 0`, tandis que la **masse** et l'**impulsion totale** restent invariantes.

Python ne fait que **composer** le systeme, poser la condition initiale, avancer en temps et lire
des diagnostics globaux. Toute la physique (flux d'Euler, force, second membre et solveur de
Poisson) est en C++.

---

## 2. Equations

Etat conservatif par cellule, 4 composantes (transport compressible 2D) :

```
U = (rho, rho u, rho v, E)
```

Euler compressible 2D :

```
d_t rho        + div(rho v)                 = 0
d_t (rho v)    + div(rho v (x) v + p I)     = rho g        (source : force par unite de volume)
d_t E          + div((E + p) v)             = rho v . g    (travail de la force)
```

avec la pression issue de l'EOS gaz parfait (etat `compressible`, `gamma = 1.4`) :

```
p = (gamma - 1) (E - 1/2 rho |v|^2)
```

Le champ de force `g = -grad(phi)` derive d'un potentiel `phi` solution de l'equation de Poisson
de **systeme**, dont le second membre est fourni par la brique elliptique du bloc :

```
lap(phi) = f,     f = sign * 4 pi G * (rho - rho0)
```

- `sign = +1` (auto-gravite) : `f = +4 pi G (rho - rho0)`, force attractive vers les surdensites ;
- `sign = -1` (plasma) : `f = -4 pi G (rho - rho0)`, force repulsive (la surdensite se relaxe).

Le terme de fond `rho0` rend le second membre de **moyenne nulle** (`rho` perturbe autour de
`rho0`), ce qui est la condition de compatibilite du Poisson **periodique** (un Laplacien
periodique n'est inversible que pour un second membre de somme nulle, a une constante pres).

> Convention `4 pi G` : ici `four_pi_G = 1.0`. Ce n'est PAS un calcul gravitationnel quantitatif
> calibre ; le but est le **contraste de signe**, pas une valeur de G physique.

---

## 3. Modele physique

Le modele nomme `models.euler_poisson(sign, gamma, four_pi_G, rho0)`
(`/private/tmp/adc_cases-readmes/adc_cases/models.py`, l.48-55) compose **quatre briques
generiques natives** de `adc` :

| Role            | Brique `adc`                                   | Effet physique |
|-----------------|------------------------------------------------|----------------|
| etat            | `adc.FluidState(kind="compressible", gamma)`   | EOS gaz parfait, 4 variables (rho, rho u, rho v, E) |
| transport       | `adc.CompressibleFlux()`                        | flux d'Euler compressible (le `gamma` vient de l'etat) |
| source          | `adc.GravityForce()`                            | force `rho g` sur la quantite de mouvement + travail `rho v.g` (car 4 variables) |
| elliptique      | `adc.GravityCoupling(sign, four_pi_G, rho0)`    | second membre de Poisson `f = sign * 4 pi G * (rho - rho0)` |

La docstring native de `adc.GravityForce` (`adc/__init__.py` l.153) : *"Force gravitationnelle
rho g (+ travail si 4 var)"*. Celle de `adc.GravityCoupling` (l.173) : *"Couplage self-consistant
f = sign 4piG (rho - rho0). sign = +1 gravite, -1 plasma"*.

**Difference natif vs charge electrostatique** : ce modele utilise `GravityForce` +
`GravityCoupling` (signe explicite porte par la brique elliptique), et NON `PotentialForce` +
`ChargeDensity` (utilises par `electron_euler` / `ion_isothermal`, ou le signe vient de la charge
`q`). Les deux familles passent par le **meme** chemin numerique cote C++ (somme generique des
briques elliptiques de chaque bloc), seul le second membre differe.

---

## 4. Methode numerique

Discretisation **volumes finis 2D** sur grille cartesienne uniforme a centres de cellules,
domaine periodique `[0, L]^2`, `n x n` cellules.

- **Reconstruction** : limiteur **van Leer** (`adc.Spatial(vanleer=True)`) -> ordre 2 avec
  capture sans oscillation pres des gradients.
- **Flux de Riemann** : **HLLC** (`flux="hllc"`) — solveur approche restituant l'onde de contact,
  adapte au transport compressible (HLLC exige une pression, fournie par l'etat compressible).
- **Variables reconstruites** : `conservative` (defaut de `adc.Spatial`, non surcharge ici).
- **Integration en temps** : explicite **SSPRK2** (Shu-Osher 2 etages, ordre 2) — c'est le defaut
  de `adc.Explicit()` (`method="ssprk2"`, `substeps=1`, `stride=1`). Pas de temps **fixe**
  `dt = 0.004`, **20 pas** (`NSTEPS`).
- **Poisson** : second membre `charge_density` (alias generique : somme des briques elliptiques par
  bloc — ici l'unique `GravityCoupling`), solveur **`geometric_mg`** (multigrille geometrique),
  conditions aux limites **periodiques** (heritees de `adc.System(periodic=True)`).

A chaque pas de temps : transport explicite SSPRK2 ; a chaque etage, le Poisson de systeme est
resolu pour le `phi` courant, d'ou `g = -grad(phi)` injecte par `GravityForce` dans la source.

---

## 5. Architecture ADC utilisee

```
Python (application)                         C++ (adc_cpp, via pybind11)
--------------------                         ---------------------------
adc.System(n, L, periodic=True)  ---------->  facade System (registre de blocs + Poisson de systeme)
   |
   +-- sim.add_block("gas",
   |        model = models.euler_poisson(...)  -> adc.Model(FluidState, CompressibleFlux,
   |        spatial = adc.Spatial(vanleer,hllc)      GravityForce, GravityCoupling) = ModelSpec
   |        time = adc.Explicit())               -> compose le bloc natif (transport+source+elliptique)
   |
   +-- sim.set_poisson(rhs="charge_density",     -> configure le Poisson de systeme
   |        solver="geometric_mg")                  (multigrille geometrique, BC periodiques)
   |
   +-- sim.set_density("gas", rho_init)          -> ecrit rho, met v=0 et E=rho/(gamma-1) (repos)
   |
   +-- sim.advance(dt, 1)  (x 20)                -> 20 pas SSPRK2 + Poisson a chaque etage
   |
   +-- sim.mass / sim.get_state / sim.time       -> diagnostics globaux (lus sur l'etat host)
```

Couches concernees :

- **`adc_cases.models.euler_poisson`** : nomme la composition des 4 briques (cote application). Le
  C++ ne connait que des briques ; le **nom** d'espece vit en Python.
- **`adc.Model(...)`** (`adc/__init__.py` l.182-235) : valide la coherence etat<->transport et
  reporte les parametres (gamma, sign, four_pi_G, rho0) dans un `ModelSpec`.
- **`adc.System.add_block`** : transmet limiteur/flux/recon/kind temporel/substeps/stride au C++.
- **`set_poisson` / `set_density` / `mass` / `get_state` / `advance` / `time`** : NE sont PAS
  redefinis cote Python ; ils sont **delegues** a la facade compilee `_System` via
  `System.__getattr__` (`adc/__init__.py` l.1116-1117).

> Important : `System` n'expose PAS de `energy()` ni de `total_momentum()`. Le cas lit ces
> diagnostics directement sur l'etat conservatif renvoye par `get_state` (cf. section 9).

---

## 6. Carte des fichiers

| Chemin | Role |
|--------|------|
| `/private/tmp/adc_cases-readmes/euler_poisson/run.py` | le cas : 2 runs (signe +/-), invariants par `assert` |
| `/private/tmp/adc_cases-readmes/adc_cases/models.py` | `euler_poisson(sign,...)` = composition des 4 briques natives (l.48-55) |
| `/private/tmp/adc_cases-readmes/adc_cases/common/checks.py` | `relative_drift`, `assert_opposite_sign` (utilises par le cas) |
| `/private/tmp/adc_cases-readmes/adc_cases/__init__.py` | `ensure_importable` (pas appele ici : voir section 9, fallback `sys.path` inline) |
| `/private/tmp/adc_cases-readmes/cases_manifest.toml` | classe le cas : `validation`, `ci = true` |
| `/private/tmp/adc_cases-readmes/.github/workflows/ci.yml` | CI qui lance les cas `ci = true` du manifeste |
| `adc/__init__.py` (build) | facade Python : briques, `Model`, `System` (delegue le reste au C++) |

> Le cas n'ecrit AUCUN fichier de sortie (pas de figure, pas de gif) : tout passe par `print` et
> les `assert`. `common/io.py`, `common/native.py`, `common/grid.py`,
> `common/initial_conditions.py`, `recipes.py` ne sont PAS utilises par ce cas (sa CI initiale est
> auto-portee).

---

## 7. Prerequis

- **Module `adc`** : les bindings pybind11 d'adc_cpp, deja construits. Dans cet environnement :
  `/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python/adc/`. Le `.so`
  present : `_adc.cpython-312-darwin.so` (Python 3.12). En CI, le module est construit a la volee
  (cf. section 17).
- **Python 3.12** + **numpy** (seule dependance Python du cas). Ici :
  `/opt/homebrew/anaconda3/bin/python3.12`.
- **Paquet `adc_cases`** importable : soit installe (`pip install -e .`, voie CI), soit la racine
  du depot sur `PYTHONPATH` / `sys.path`. Le cas gere lui-meme le fallback (try/except `ImportError`
  -> insertion du parent dans `sys.path`, `run.py` l.48-53).
- **Aucun compilateur C++** requis a l'execution (`needs = []` dans le manifeste) : il n'y a pas de
  compilation a la volee ; le natif est deja dans le module `adc`.

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/euler_poisson && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le premier chemin du `PYTHONPATH` fournit le module C++ `adc` ; le second rend le paquet
`adc_cases` importable sans installation.

---

## 9. Explication du code par etapes

`run.py` (190 lignes) se lit du haut vers le bas :

1. **Imports et fallback** (l.44-55) : `import adc` (C++) ; `import adc_cases` ou, a defaut,
   insertion du dossier parent dans `sys.path` ; puis `from adc_cases import models` et
   `from adc_cases.common.checks import assert_opposite_sign, relative_drift`.
2. **Tolerances** (l.57-63) : `TOL_MASS = 1e-9` (derive relative de masse), `TOL_MOM = 1e-8`
   (impulsion nette), `TOL_DE = 1e-5` (magnitude minimale du contraste energetique pour que le
   signe soit significatif — `dE = 0` exactement a perturbation nulle).
3. **Parametres** (l.65-72) : `N=64`, `L=1.0`, `GAMMA=1.4`, `EPS=0.01`, `RHO0=1.0`, `DT=0.004`,
   `NSTEPS=20`.
4. **`initial_density()`** (l.75-79) : densite au repos faiblement perturbee par un cosinus selon x
   (cf. section 10).
5. **`energy_and_momentum(sim)`** (l.82-85) : lit `U = sim.get_state("gas")` de forme `(4, n, n)`
   et renvoie `(E_tot, p_x, p_y) = (U[3].sum(), U[1].sum(), U[2].sum())`. Ce sont des **sommes sur
   cellules** des composantes conservatives (PAS d'integrale avec poids `dx^2` ; voir section 11).
6. **`run_case(sign, label)`** (l.88-137) :
   - `sim = adc.System(n=N, L=L, periodic=True)` ;
   - `sim.add_block("gas", model=models.euler_poisson(sign=sign, gamma=GAMMA, four_pi_G=1.0,
     rho0=RHO0), spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.Explicit())` ;
   - `sim.set_poisson(rhs="charge_density", solver="geometric_mg")` ;
   - `sim.set_density("gas", initial_density())` -> pose rho, met v=0 et `E = rho/(gamma-1)` ;
   - memorise `mass0 = sim.mass("gas")` et `energy0` ;
   - boucle `for step in 1..20` : `sim.advance(DT, 1)`, relit masse/energie/impulsion, accumule
     `max_rel_mass` (via `relative_drift`) et `max_mom` ; imprime les pas 1, 5, 10, 15, 20 ;
   - renvoie un dict de diagnostics (mass0, energy0, energy_final, max_rel_mass, max_mom, time).
7. **`main()`** (l.140-183) :
   - lance `run_case(+1.0, "GRAVITE")` puis `run_case(-1.0, "PLASMA ")` ;
   - **assertions par run** : `max_rel_mass < TOL_MASS` (masse conservee) et `max_mom < TOL_MOM`
     (impulsion nette nulle) ;
   - calcule `dE_grav` et `dE_plas` (energie finale - energie initiale) ;
   - `assert_opposite_sign(dE_grav, dE_plas, min_mag=TOL_DE, ...)` : signes opposes ET magnitudes
     franches ;
   - `assert dE_grav < 0` (attractif) et `assert dE_plas > 0` (repulsif) ;
   - imprime le bilan et `OK euler_poisson`.

---

## 10. Conditions initiales

Densite (commune aux deux runs), `initial_density()` (`run.py` l.75-79) :

```
x_i   = (i + 0.5) * L / N                          (centres de cellules, i = 0..N-1)
rho(x) = RHO0 * (1 + EPS * cos(2 pi x / L))         RHO0 = 1.0, EPS = 0.01
```

Une perturbation cosinus de mode 1 selon x, amplitude 1 %, autour de la densite uniforme `RHO0`.
`set_density("gas", rho)` ecrit cette densite sur la composante 0, met la **quantite de mouvement
a zero** (v = 0) et fixe l'energie au repos `E = rho / (gamma - 1)` (le bloc compressible convertit
en variables conservatives cote C++).

> A perturbation nulle (`EPS = 0`), le second membre de Poisson `f = sign*4piG*(rho - rho0)` est
> identiquement nul, la force est nulle, et `dE = 0` exactement. La perturbation de 1 % est ce qui
> rend le travail de la force mesurable (et donc le signe de `dE` significatif).

---

## 11. Invariants et assertions

Quatre controles, tous **passes** a l'execution (valeurs reelles capturees, section 12) :

1. **Conservation de la masse** (`assert res["max_rel_mass"] < TOL_MASS`, TOL = `1e-9`) : le schema
   volumes finis est conservatif sur domaine periodique. La masse de reference est
   `mass0 = sim.mass("gas") = 4096` (somme de `rho` sur les `64x64` cellules, `RHO0=1`). Derive
   relative max **mesuree** : `2.598e-14` (GRAVITE), `2.098e-14` (PLASMA) — du bruit machine, tres
   en dessous de `1e-9`.
2. **Impulsion totale nulle** (`assert res["max_mom"] < TOL_MOM`, TOL = `1e-8`) : la force de
   Poisson derive d'un potentiel ; sur domaine periodique homogene, sa somme spatiale est nulle,
   donc elle n'injecte aucune impulsion nette. `max |p|` **mesure** : `8.882e-16` (les deux runs).
3. **Contraste de signe** (`assert_opposite_sign(dE_grav, dE_plas, min_mag=TOL_DE)`, TOL = `1e-5`) :
   les deux derives d'energie doivent etre de signes **strictement opposes** ET de magnitude
   superieure a `1e-5` (sinon le signe serait du bruit). Mesure : `dE_grav = -5.857667e-04`,
   `dE_plas = +6.137105e-04` -> produit < 0, magnitudes ~6e-4 (>> 1e-5).
4. **Signe physique de chaque run** : `assert dE_grav < 0` (auto-gravite, attractif : l'energie
   totale diagnostiquee baisse) et `assert dE_plas > 0` (charge d'espace, repulsif : elle monte).

> **Honnetete sur les diagnostics** : `mass`, `E_tot`, `p_x`, `p_y` sont des **sommes sur
> cellules** des composantes conservatives, sans facteur de volume `dx^2`. Ce sont des proxys de
> masse/energie/impulsion totales, suffisants pour les invariants relatifs (conservation) et de
> signe (contraste), ce que le cas teste. Ce ne sont PAS des integrales physiques calibrees, et la
> valeur absolue `dE ~ 6e-4` n'est pas comparee a une reference theorique : seuls comptent sa
> **conservation** (masse, impulsion) et son **signe** (gravite vs plasma).

---

## 12. Sorties attendues

Execution reelle (machine de developpement, sortie integrale capturee) :

```
[GRAVITE] etat initial : mass=4.096000000000e+03  energy=1.024000000000e+04
[GRAVITE] pas                  mass               energy            p_x            p_y
[GRAVITE]   1    4.096000000000e+03   1.023999999840e+04      1.456e-31      0.000e+00
[GRAVITE]   5    4.096000000000e+03   1.023999996026e+04      1.270e-30      2.220e-16
[GRAVITE]  10    4.096000000000e+03   1.023999984355e+04     -9.738e-31      0.000e+00
[GRAVITE]  15    4.096000000000e+03   1.023999965748e+04     -1.233e-32     -8.882e-16
[GRAVITE]  20    4.096000000000e+03   1.023999941423e+04      2.712e-31      0.000e+00

[PLASMA ] etat initial : mass=4.096000000000e+03  energy=1.024000000000e+04
[PLASMA ] pas                  mass               energy            p_x            p_y
[PLASMA ]   1    4.096000000000e+03   1.024000000168e+04      1.791e-31      0.000e+00
[PLASMA ]   5    4.096000000000e+03   1.024000004180e+04     -8.073e-31      0.000e+00
[PLASMA ]  10    4.096000000000e+03   1.024000016442e+04     -1.676e-30      0.000e+00
[PLASMA ]  15    4.096000000000e+03   1.024000035951e+04     -4.117e-30      0.000e+00
[PLASMA ]  20    4.096000000000e+03   1.024000061371e+04     -1.689e-30      0.000e+00

Bilan des invariants (sur les 20 pas) :
  GRAVITE : max derive masse relative = 2.598e-14 (< 1e-09)   max |p| = 8.882e-16 (< 1e-08)
  PLASMA  : max derive masse relative = 2.098e-14 (< 1e-09)   max |p| = 8.882e-16 (< 1e-08)
Contraste energetique (attractif vs repulsif) :
  dE GRAVITE = -5.857667e-04   dE PLASMA = +6.137105e-04
  -> signes opposes (gravite dE<0, plasma dE>0), magnitudes > 1e-05 : OK
OK euler_poisson
```

Lecture : la masse est strictement constante (`4.096e+03`), l'energie **decroit** lentement pour
GRAVITE (de `1.0240000e+04` a `1.0239999e+04`) et **croit** pour PLASMA — exactement le contraste
de signe attendu. `p_x`, `p_y` restent au niveau du bruit machine (`~1e-16` ou moins). Le code se
termine sur `OK euler_poisson` (tous les `assert` passent) et un code de retour 0.

> Les chiffres `dE` au-dela de quelques chiffres significatifs peuvent varier legerement selon la
> plateforme (BLAS, ordre de sommation, version du module). Les **signes**, l'ordre de grandeur
> (`~6e-4`) et le verdict `OK` sont robustes.

---

## 13. Generation figures/GIF

**Aucune.** Ce cas ne produit ni figure ni gif : il imprime un tableau de diagnostics et verifie
des `assert`. Il n'importe pas `matplotlib` (cf. `needs = []` dans le manifeste, contrairement au
cas `diocotron` qui demande `matplotlib`). Il n'ecrit aucun fichier sur disque.

---

## 14. Backends reellement supportes

- **CPU natif (C++ via pybind11)** : le SEUL backend exerce. Le module `adc` est construit pour
  CPU ; le solveur de Poisson est `geometric_mg` (multigrille geometrique). Pas de GPU, pas de MPI
  dans ce cas (un seul rang, une seule grille `64x64`).
- **Mono-bloc, mono-rang** : `adc.System` avec un unique `add_block("gas", ...)`. Pas d'AMR (ce
  serait `adc.AmrSystem`), pas de multi-especes.
- **Pas de compilation a la volee** : aucun chemin `ctypes`/`.so` JIT, aucun DSL interprete. Tout
  le natif est deja dans le module `adc` (`needs = []`).
- **Reconstruction/flux** : van Leer + HLLC sont disponibles sur le chemin natif `add_block`.
  HLLC exige un transport compressible (pression) — satisfait ici par `FluidState(compressible)`.

---

## 15. Cout approximatif

Mesure reelle (`/usr/bin/time -p`, `/opt/homebrew/anaconda3/bin/python3.12`, machine de dev macOS
arm64), **3 executions consecutives** apres une premiere chauffe :

```
real 0.27   user 1.05   sys 0.74     (1re mesure : real 0.29)
real 0.28
real 0.28
```

- **Temps mur total ~0.27-0.29 s**, import du module/numpy inclus. Le `user > real` reflete des
  threads internes (BLAS / multigrille).
- Cout de calcul pur negligeable : 2 runs x 20 pas x une grille `64x64` (4096 cellules, 4
  variables) + un Poisson multigrille par etage. C'est un cas **leger**, concu pour la CI.
- Aucune memoire notable (quelques tableaux `(4, 64, 64)` en double).

---

## 16. Limites et differences avec les references

- **Pas une reproduction publiee.** Categorie `validation` : le cas teste des invariants
  (conservation, signes), pas un resultat quantitatif d'un article. Il ne doit PAS etre presente
  comme une reproduction d'instabilite de Jeans, d'effondrement gravitationnel ou d'un benchmark
  plasma. Le contraste gravite/plasma est un test de **signe**, pas une simulation calibree.
- **`4 pi G = 1`, `rho0 = 1`, pas d'unites physiques.** Les constantes sont choisies pour la
  lisibilite du test, pas pour modeliser un systeme reel.
- **Diagnostics = sommes sur cellules**, sans poids de volume `dx^2` (section 11). `mass`, `E_tot`,
  `p_x/p_y` sont des proxys ; les invariants verifies (relatifs / de signe) ne dependent pas du
  facteur d'echelle, mais les valeurs absolues ne sont pas des integrales physiques.
- **Regime quasi-lineaire, horizon court.** `EPS = 0.01`, 20 pas, `dt = 0.004` : la perturbation
  reste petite ; on observe la tendance energetique (travail de la force), pas une dynamique non
  lineaire (effondrement, formation de structure).
- **Domaine periodique homogene.** C'est ce qui garantit l'impulsion nette nulle et la
  compatibilite du Poisson (second membre de moyenne nulle grace au fond `rho0`). Un domaine a
  parois ou un second membre non centre changerait ces invariants.
- **Pas de comparaison a une solution analytique** du taux d'energie : seul le **signe** de `dE`
  est asserte, pas sa valeur.

---

## 17. Tests/CI associes

- **Manifeste** (`cases_manifest.toml`) : `path = "euler_poisson/run.py"`, `category =
  "validation"`, `ci = true`, `needs = []`,
  `desc = "Euler couple a Poisson : auto-gravite (attractif) vs plasma (repulsif)."`.
- **CI GitHub Actions** (`.github/workflows/ci.yml`) : le workflow construit le module `adc`
  (clone `wolf75222/adc_cpp`, `cmake -DADC_BUILD_PYTHON=ON -DADC_USE_EIGEN=OFF`, cible `_adc`),
  installe `adc_cases` en editable (`pip install -e .`), lit le manifeste et lance **chaque cas
  `ci = true`** par `python3 <path>` avec `PYTHONPATH` pointant le build. Un `assert` qui echoue
  fait sortir `run.py` en erreur -> **CI rouge**. Ce cas est l'un des cas legers executes a chaque
  push/PR.
- **Le test EST le cas** : les `assert` de `main()` (masse, impulsion, contraste de signe) sont la
  validation. Il n'y a pas de fichier de test separe ; `run.py` se suffit a lui-meme et imprime
  `OK euler_poisson` en cas de succes.
