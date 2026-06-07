# custom_scheme : ecrire son propre schema numerique autour du solveur Poisson de adc

Cas **tutoriel** (`category = "tutoriel"`, `ci = true` dans `cases_manifest.toml`).

Le schema numerique est ecrit **entierement en Python/numpy** : reconstruction,
flux upwind conservatif et integrateur temporel SSPRK2 vivent cote application.
La bibliotheque `adc` n'est appelee **qu'une seule fois par evaluation du second
membre**, comme **oracle de Poisson** : on lui donne la densite, elle rend le
potentiel self-consistent. C'est le pendant *spatial* de l'integrateur temporel
Python de `adc.integrate` : la densite vit cote Python, la lib ne joue que le role
de solveur elliptique.

But pedagogique : montrer comment on peut greffer son propre transport (numpy)
sur les briques couteuses de `adc` (ici, le seul Poisson geometrique multigrille),
sans rien deleguer du transport au C++.

---

## 1. Objectif du cas

Implementer un solveur diocotron periodique dont la partie hyperbolique
(advection ExB) est ecrite a la main en numpy, et ne demander a `adc` **que** la
resolution de l'equation de Poisson a chaque sous-pas. On verifie trois proprietes,
par `assert`, ce qui fait un test leger de CI :

1. **Couplage actif** : le potentiel resolu par `adc` est non nul
   (`|phi|_max > 1e-8`) ; le transport est donc reellement pilote par le Poisson de
   la lib, pas par un champ trivial.
2. **Conservation de la masse** : la forme flux upwind sur domaine periodique
   conserve la masse a la precision machine (derive relative `< 1e-12`).
3. **Dynamique non triviale** : la bande de charge evolue
   (`max|n - n0| > 1e-3`), donc le schema avance bien quelque chose.

Ce cas n'est PAS une reproduction d'un resultat publie : il ne mesure aucun taux
de croissance, n'utilise aucune figure. C'est un tutoriel d'API.

---

## 2. Equations

Modele diocotron scalaire (derive ExB d'une densite, fond neutralisant) :

```
d_t n + div(n v) = 0                 (transport, ecrit en Python)
v = (-d_y phi, d_x phi) / B0          (vitesse ExB, ecrite en Python)
lap phi = alpha (n - n_i0)            (Poisson, resolu par adc)
```

- `n(x, y, t)` : densite scalaire (charge), vit dans un tableau numpy `n[j, i]`.
- `phi(x, y, t)` : potentiel electrostatique, sortie du solveur Poisson de `adc`.
- `v` : vitesse de derive ExB, incompressible (`div v = 0` analytiquement, car
  `v = rot(phi e_z)/B0`).
- `B0` : intensite du champ magnetique de fond (ici `B0 = 1`).
- `alpha` : coefficient du couplage Poisson (ici `alpha = 1`).
- `n_i0` : densite du fond neutralisant. Domaine **periodique** : le Laplacien
  periodique n'a de solution que si le second membre est de moyenne nulle, d'ou le
  choix `n_i0 = mean(n)` (impose dans `run.py`).

---

## 3. Modele physique

Modele d'espece nomme `diocotron`, defini cote application dans
`adc_cases/models.py` :

```python
def diocotron(B0=1.0, alpha=1.0, n_i0=0.0):
    return adc.Model(
        state=adc.Scalar(),
        transport=adc.ExB(B0=B0),
        source=adc.NoSource(),
        elliptic=adc.BackgroundDensity(alpha=alpha, n0=n_i0),
    )
```

Point clef du cas : **la brique `transport=adc.ExB(B0)` n'est jamais executee**.
Le modele est ajoute au `System` uniquement pour porter la brique elliptique
`BackgroundDensity(alpha, n0)`, qui configure le second membre `alpha (n - n_i0)`
du Poisson de systeme. Tout le transport ExB est re-ecrit en numpy dans `run.py`
(fonctions `drift` et `divergence_upwind`). On n'appelle jamais `sim.step()`.

---

## 4. Methode numerique

Tout le schema est en Python. Trois ingredients :

- **Vitesse ExB** (`drift`) : differences centrees periodiques d'ordre 2 sur le
  potentiel, via `np.roll` :
  `vx = -d_y phi / B0`, `vy = +d_x phi / B0`, avec
  `d_x phi = (phi[i+1] - phi[i-1]) / (2 dx)` (idem en y). Pas est uniforme,
  `dx = L / nx`.

- **Divergence du flux** (`divergence_upwind`) : schema **upwind du premier ordre
  en forme flux**, donc **conservatif**. A l'interface `i+1/2` selon x :
  - vitesse d'interface = moyenne `vxr = 0.5 (vx_i + vx_{i+1})` ;
  - etat amont selon le signe : `fxr = (vxr > 0 ? n_i : n_{i+1}) * vxr` ;
  - `fxl = roll(fxr, +1)` (le flux en `i-1/2` est celui en `i+1/2` de la cellule
    de gauche) ;
  - residu `-div(n v) = -((fxr - fxl) + (fyr - fyl)) / dx`.
  Comme chaque flux d'interface est partage entre deux cellules voisines (forme
  telescopique), la somme des residus sur un domaine periodique est nulle a la
  precision machine : la masse est conservee exactement.

- **Integration temporelle** (boucle `main`) : **SSPRK2** (Heun / Runge-Kutta
  fort-stable d'ordre 2), ecrit a la main :
  ```
  n1 = n + dt * R(n)
  n  = 0.5 * n + 0.5 * (n1 + dt * R(n1))
  ```
  ou `R = -div(n v)`. Le pas de temps est recalcule a chaque iteration par une
  **condition CFL** : `dt = cfl * dx / max(|v|)`, avec `cfl = 0.4` et `max(|v|)`
  evalue au debut de l'etage 1 (`speed = max(hypot(vx, vy))`).

Chaque etage SSPRK2 appelle `rhs`, qui re-resout Poisson : il y a donc **deux
resolutions de Poisson par pas de temps**.

---

## 5. Architecture ADC utilisee

Le **seul** point de contact avec la bibliotheque est l'oracle de Poisson
(fonction `poisson_oracle`) :

```python
def poisson_oracle(sim, n):
    sim.set_density("ne", n)   # ecrit la densite du bloc "ne"
    sim.solve_fields()         # resout lap phi = alpha (n - n_i0)
    return sim.potential()     # rend phi (tableau n*n)
```

Mise en place de l'oracle (une fois, dans `main`) :

```python
sim = adc.System(n=nx, L=L, periodic=True)
sim.add_block("ne", model=models.diocotron(B0=B0, alpha=1.0, n_i0=n_i0))
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

- `adc.System(n, L, periodic=True)` : domaine carre `[0, L]^2`, conditions aux
  limites periodiques, grille a centres de cellules (convention `field[j, i]`).
- `add_block("ne", model=...)` : ajoute le bloc nomme `"ne"` qui porte la brique
  elliptique. La facade `System.add_block` (cf. `adc/__init__.py`) cree le bloc
  cote C++ ; on n'appelle jamais l'integrateur du systeme dessus.
- `set_poisson(rhs="charge_density", solver="geometric_mg")` : assemble le Poisson
  de systeme (somme des briques elliptiques des blocs) et selectionne le solveur
  multigrille geometrique. `set_poisson`, `set_density`, `solve_fields` et
  `potential` sont des methodes de la facade compilee `_System`, exposees a travers
  `System.__getattr__` (cf. `adc/__init__.py`, ligne ~1116).

Tout le reste (`drift`, `divergence_upwind`, SSPRK2, CFL) est du numpy pur.

---

## 6. Carte des fichiers

| Fichier | Role |
| --- | --- |
| `custom_scheme/run.py` | Le cas : schema spatial + temporel en numpy, oracle Poisson `adc`. |
| `custom_scheme/README.md` | Ce document. |
| `adc_cases/models.py` | `models.diocotron(...)` : le modele d'espece (brique elliptique utile). |
| `adc_cases/common/initial_conditions.py` | `band_density(...)` : la CI (bande gaussienne perturbee). |
| `adc_cases/common/grid.py` | `meshgrid_xy`, `cell_centers` : grille a centres de cellules (utilise par la CI). |
| `adc_cases/common/checks.py` | `assert_finite`, `relative_drift` : invariants verifies. |
| `cases_manifest.toml` | Manifeste : ce cas y est `category = "tutoriel"`, `ci = true`. |

Le paquet `adc_cases` est importe soit comme paquet installe (voie nominale CI),
soit, a defaut, en ajoutant la racine du depot au `sys.path` (bloc `try/except
ImportError` en tete de `run.py`).

---

## 7. Prerequis

- Python 3.12, **numpy** (seule dependance tierce du cas).
- Le module `adc` (bindings de `adc_cpp`) doit etre **importable** : soit installe,
  soit accessible via `PYTHONPATH` pointant sur le repertoire `python/` d'un build
  de `adc_cpp` (qui contient le paquet `adc` et le `.so` `_adc`).
- Le paquet `adc_cases` doit etre importable (installe en editable, ou racine du
  depot sur le `sys.path`, ce que `run.py` fait automatiquement).
- **Aucun compilateur C++** n'est requis a l'execution du cas (`needs = []` dans le
  manifeste). Le `.so` `_adc` doit deja exister (il est construit au prealable, hors
  de ce cas).

Build de reference utilise pour ce README :
`/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python`.

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/custom_scheme
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le premier element du `PYTHONPATH` rend le module `adc` importable ; le second rend
le paquet `adc_cases` importable (le `try/except` de `run.py` couvre aussi le cas ou
`adc_cases` n'est pas installe).

En CI, le cas est lance par `.github/workflows/ci.yml` apres construction du module
`adc` et installation editable de `adc_cases` : `python3 custom_scheme/run.py`, avec
`PYTHONPATH` pointant sur le build `adc_cpp`.

---

## 9. Explication du code par etapes

1. **Imports et fallback de chemin** : `import adc`, puis tentative
   `import adc_cases` ; en cas d'echec, la racine du depot est ajoutee au
   `sys.path`. On importe ensuite `models`, `assert_finite`, `relative_drift`,
   `band_density`.

2. **`drift(phi, dx, B0)`** : calcule `(vx, vy)` par differences centrees
   periodiques (`np.roll`) du potentiel. Retourne `(-d_y phi / B0, d_x phi / B0)`.

3. **`divergence_upwind(n, vx, vy, dx)`** : calcule `-div(n v)` par flux upwind
   conservatif (cf. section 4). N'introduit aucune fuite de masse sur domaine
   periodique.

4. **`poisson_oracle(sim, n)`** : ecrit la densite (`set_density("ne", n)`), resout
   (`solve_fields()`), rend le potentiel (`potential()`). C'est l'unique appel a la
   lib.

5. **`rhs(sim, n, dx, B0)`** : compose tout cela — Poisson par `adc`, vitesse +
   divergence en numpy — et renvoie `(residu, speed)`, ou `speed = max(|v|)` sert au
   CFL.

6. **`main()`** :
   - parametres : `nx = 96`, `L = 1.0`, `B0 = 1.0`, `dx = L / nx`.
   - CI : `n = band_density(nx, L, amp=1.0, width=0.05, mode=4, disp=0.02)` (bande
     gaussienne perturbee mode 4, meme profil que le cas diocotron).
   - fond neutralisant : `n_i0 = mean(n)` (Poisson periodique a moyenne nulle).
   - construction de l'oracle (`System` + `add_block` + `set_poisson`).
   - **verification de couplage** : `phi0 = poisson_oracle(sim, n)` puis
     `assert |phi0|_max > 1e-8`.
   - `mass0 = sum(n) * dx*dx`, `n0 = n.copy()`.
   - **boucle SSPRK2** sur `nsteps = 200` pas, `cfl = 0.4` :
     deux appels `rhs` par pas, `dt = cfl * dx / max(speed, 1e-12)`,
     `assert_finite(n, ...)` a chaque pas.
   - **bilans finaux** : `mass1`, `drel = relative_drift(mass1, mass0)`,
     `moved = max|n - n0|`, puis les deux `assert` (masse conservee, bande evoluee).

---

## 10. Conditions initiales

CI generee par `band_density` (`adc_cases/common/initial_conditions.py`) :

```
ne(x, y) = floor + amp * exp(-(y - y0)^2 / width^2)
y0       = 0.5 L + disp * cos(2 pi mode x / L)
```

avec, ici, `floor = 1.0` (defaut), `amp = 1.0`, `width = 0.05`, `mode = 4`,
`disp = 0.02`, `L = 1.0`, `nx = 96`. C'est une **bande horizontale** de charge,
centree en `y = 0.5 L`, dont la position est perturbee sinusoidalement le long de
`x` (mode azimutal 4). Tableau `(96, 96)` contigu, convention `ne[j, i]`.

Le fond neutralisant `n_i0 = mean(ne)` rend le second membre du Poisson periodique
de moyenne nulle (condition de compatibilite).

---

## 11. Invariants et assertions

Verifies par `assert` (un echec sort en erreur -> CI rouge) :

| Invariant | Code | Seuil | Valeur mesuree (run de reference) |
| --- | --- | --- | --- |
| Couplage actif | `\|phi0\|_max > 1e-8` | `1e-8` | `6.124932e-03` (PASS) |
| Conservation de la masse | `relative_drift(mass1, mass0) < 1e-12` | `1e-12` | `2.040e-16` (PASS) |
| Dynamique non triviale | `max\|n - n0\| > 1e-3` | `1e-3` | `3.280e-01` (PASS) |
| Finitude | `assert_finite(n, ...)` a chaque pas | NaN/Inf | aucun NaN/Inf (PASS) |

- `relative_drift(value, reference)` (cf. `checks.py`) =
  `|value - reference| / max(|reference|, 1e-30)`.
- `assert_finite` (cf. `checks.py`) leve si un NaN/Inf apparait.

La conservation de la masse a `2.04e-16` (de l'ordre du `eps` machine) est attendue :
la forme flux upwind est telescopique et le domaine est periodique.

---

## 12. Sorties attendues

Sortie texte du run de reference (aucun fichier produit) :

```
== custom_scheme : transport diocotron 100 % Python, Poisson par adc ==
  |phi|_max initial = 6.124932e-03  (Poisson de adc actif)
  derive de masse relative = 2.040e-16  (flux upwind conservatif)
  evolution max|dn|        = 3.280e-01  (dynamique non triviale)
Schema spatial + temporel ecrit en Python ; adc ne fait que Poisson.
OK custom_scheme
```

Le cas se termine par `OK custom_scheme` et un code de sortie 0 si tous les
`assert` passent.

---

## 13. Generation figures/GIF

**Aucune.** Ce cas ne produit ni figure ni GIF ni fichier de sortie. Il
n'importe pas `matplotlib` (`needs = []` dans le manifeste). C'est un test/tutoriel
purement textuel.

Pour la version qui produit figures + GIF (et vise une reproduction
d'`arXiv:2510.11808`), voir le cas `diocotron/` (`category = "reproduction"`,
`ci = false`, `needs = ["matplotlib"]`), qui n'est PAS ce cas-ci.

---

## 14. Backends reellement supportes

- **Schema (transport + temps)** : **CPU Python/numpy uniquement**. Aucun backend
  GPU ni MPI cote transport — c'est du numpy host, mono-processus.
- **Solveur Poisson** : `solver="geometric_mg"` (multigrille geometrique), execute
  sur le backend du module `adc` charge (CPU dans le build de reference). Le cas ne
  selectionne pas d'autre solveur ; il ne tente pas `fft` (qui serait pourtant
  legitime ici, periodique avec `nx = 96` non puissance de 2 -> `fft` exigerait
  `n = 2^k`, donc `geometric_mg` est le bon choix).
- Conditions aux limites : **periodiques** (`periodic=True`).

Le cas etant un tutoriel host-numpy, il n'a pas vocation a explorer les backends
device/MPI de `adc`.

---

## 15. Cout approximatif

Mesure sur la machine de reference (Apple Silicon, `/usr/bin/time -p`,
`build-master` CPU) :

```
real 1.07 s    user 2.45 s    sys 0.32 s    (1er run)
real 1.08 s    user 2.39 s    sys 0.37 s    (2e run)
```

Soit **~1.1 s de temps mur**, dominees par le demarrage de l'interpreteur,
l'import du module `adc` et les `2 * 200 = 400` resolutions Poisson `geometric_mg`
sur grille `96 x 96`. Le transport numpy lui-meme est negligeable a cette taille.
`user > real` reflete le multithreading interne (BLAS/OpenMP) lors des resolutions.

Cout tres faible : adapte a la CI.

---

## 16. Limites et differences avec les references

- **Ce n'est pas une reproduction d'un papier.** Categorie `tutoriel`. Le cas ne
  mesure aucun taux de croissance diocotron, ne compare a aucune table de
  reference. Ne pas le presenter comme une reproduction de l'instabilite diocotron.
- **Schema volontairement simple** : upwind du **premier ordre** en espace (pas de
  limiteur de pente, pas de reconstruction MUSCL/WENO) et SSPRK2 a l'ordre 2 en
  temps. La precision n'est pas l'objectif ; la lisibilite et la conservation le
  sont. Une bande fine sur grille `96 x 96` diffuse numeriquement de facon visible
  (l'upwind d'ordre 1 est tres dissipatif).
- **La brique `adc.ExB` du modele n'est pas utilisee** : elle est presente
  uniquement pour porter la brique elliptique `BackgroundDensity`. Le transport ExB
  du cas est une re-implementation numpy independante, qui n'est PAS garantie
  bit-identique au transport natif de `adc` (schemas, ordre et flux differents).
- **Domaine periodique, fond impose** : `n_i0 = mean(n)` est requis pour que le
  Poisson periodique soit compatible ; ce choix est physique (fond neutralisant)
  mais specifique au cadre periodique.
- L'invariant de masse `~1e-16` valide la **conservation discrete du schema
  numpy**, pas une quelconque fidelite a une solution analytique.

---

## 17. Tests/CI associes

- Manifeste `cases_manifest.toml` : ce cas est
  ```
  [[case]]
  path = "custom_scheme/run.py"
  category = "tutoriel"
  ci = true
  needs = []
  ```
- CI (`.github/workflows/ci.yml`) : le workflow construit le module `adc` (cmake
  `-DADC_BUILD_PYTHON=ON`), installe `adc_cases` en editable, lit le manifeste et
  lance **uniquement** les cas `ci = true` via `python3 custom_scheme/run.py`. Le
  cas etant `needs = []`, il tourne sur la matrice de base (`ubuntu-latest`,
  Python 3.12), sans dependance `matplotlib` ni compilateur supplementaire.
- Le test lui-meme EST le `run.py` : ses quatre `assert` (couplage, conservation,
  dynamique, finitude) sont les criteres de reussite. Aucun fichier de test
  separe (`pytest`) n'est associe a ce cas.
