# dsl_euler — Euler 2D ecrit en formules (mini-DSL adc.dsl, interprete CPU)

Cas `dsl_euler` du depot `adc_cases`. Categorie **experimental**, `ci = false`, `needs = []`
(cf. `cases_manifest.toml`).

Ce cas demontre le principe « Python ecrit les equations, le coeur execute les boucles » dans sa
version PROTOTYPE la plus depouillee : on declare le systeme d'Euler compressible 2D entierement en
EXPRESSIONS SYMBOLIQUES (variables, primitives, flux, valeurs propres), `adc.dsl` interprete cet
arbre en numpy, et le branche sur le backend hote `adc.PythonFlux` qui assemble `-div(F*)` par
volumes finis (Rusanov, ordre 1, periodique). Avance par Euler avant explicite.

IMPORTANT — honnetete sur le statut. Ce cas est un **prototype declaratif interprete sur CPU**, PAS
le chemin de production. Il se distingue des autres cas `*_dsl` du manifeste
(`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`, `hoffart_euler_poisson_dsl`) qui,
eux, COMPILENT le meme genre d'arbre en C++/Kokkos (`emit_cpp*` -> `add_compiled_model`, etat
bit-identique au natif). Ici, AUCUNE compilation : tout tourne en numpy via `Expr.eval`. C'est ce
qui le rend `needs = []` (pas de compilateur C++ requis) mais aussi `experimental`.

---

## 1. Objectif du cas

Montrer le bout « declaratif cote utilisateur » du DSL `adc` :

- ecrire un modele hyperbolique (Euler 2D) sans appeler AUCUNE brique nommee
  (`adc.CompressibleFlux`, `adc.FluidState`...), uniquement des formules symboliques ;
- faire TOURNER ce modele immediatement (interprete numpy) via `adc.PythonFlux`, sans etape de
  compilation, pour iterer vite sur un systeme inedit ;
- verifier par assertions que la physique est correcte : masse conservee (domaine periodique),
  positivite (`rho > 0`, `p > 0`), finitude, et dynamique non triviale (une bulle de pression
  centrale doit generer des ondes acoustiques).

Le `run.py` cite explicitement la cible : « Le MEME arbre alimenterait plus tard un codegen
C++/Kokkos pour la production (cf. `adc_cpp/docs/ARCHITECTURE_CIBLE.md` sect. 3). Le chemin de
production reste les briques compilees ; ce cas montre le bout declaratif. »

## 2. Equations

Euler compressible 2D, forme conservative, sans terme source ni couplage Poisson :

```
d_t U + d_x F_x(U) + d_y F_y(U) = 0,   U = (rho, rho u, rho v, E).
```

Avec p la pression, H = (E + p) / rho l'enthalpie totale et gamma l'indice adiabatique :

```
F_x(U) = ( rho u,  rho u^2 + p,  rho u v,      rho H u )
F_y(U) = ( rho v,  rho u v,      rho v^2 + p,  rho H v )

p = (gamma - 1) * ( E - 0.5 * rho * (u^2 + v^2) )
```

Valeurs propres (vitesses caracteristiques) par direction, avec c = sqrt(gamma p / rho) la vitesse
du son :

```
lambda_x = { u - c,  u,  u + c },   lambda_y = { v - c,  v,  v + c }.
```

Ces sept lignes sont exactement ce qui est ecrit dans `make_euler()` (cf. section 9) : il n'y a pas
de code C++ derriere, l'arbre symbolique EST la specification.

## 3. Modele physique

Gaz parfait monoespece, gamma = 1.4 (`GAMMA = 1.4` dans `run.py`). Pas de gravite, pas de champ
electrostatique, pas de magnetisme : le terme source et le second membre elliptique du modele DSL
sont LAISSES indefinis (`set_source` / `set_elliptic_rhs` jamais appeles), donc `source_value`
renvoie des zeros et le systeme est purement hyperbolique.

Condition initiale : gaz uniforme au repos (`rho = 1`, `u = v = 0`) avec une bulle de surpression
gaussienne au centre du domaine. La detente de cette bulle produit une onde acoustique radiale
(cf. section 10). Domaine carre periodique `[0, 1]^2`.

## 4. Methode numerique

- **Discretisation spatiale** : volumes finis ordre 1, flux numerique de **Rusanov (Lax-Friedrichs
  local)**, assemble par `adc.PythonFlux.residual`. Pour chaque axe (x = axe 2, y = axe 1 du
  tableau numpy) :

  ```
  F_face = 0.5 * (F_i + F_{i+1}) - 0.5 * a * (U_{i+1} - U_i),   a = max_wave_speed(U)
  res    = -(F_{i+1/2} - F_{i-1/2}) / h                          (-div F*)
  ```

  La diffusion de Rusanov utilise UNE seule vitesse globale `a = max_k max_cellules |lambda_k|`
  (max sur les deux directions), recalculee a chaque appel. C'est volontairement le schema le plus
  simple : pas de reconstruction MUSCL, pas de limiteur, ordre 1 en espace.

- **Periodicite** : implementee par `numpy.roll` (decalage circulaire) sur chaque axe ; il n'y a pas
  de cellules fantomes explicites, le domaine est periodique par construction.

- **Avance en temps** : **Euler avant** (explicite, ordre 1), ecrit a la main dans `run.py` :

  ```
  U = U + pf.cfl_dt(U, h, 0.4) * pf.residual(U, h)
  ```

  avec `cfl_dt = cfl * h / max(max_wave_speed(U), 1e-30)`, cfl = 0.4. Le pas est donc reevalue a
  chaque iteration selon la CFL courante (pas fixe a priori). 120 pas.

- **Backend** : interprete CPU pur. `HyperbolicModel.flux` evalue l'arbre symbolique en numpy
  (`Expr.eval(env)` ou env = cons depuis U, primitives derivees dans l'ordre de dependance) sur
  TOUT le tableau d'un coup. Aucun kernel Kokkos, aucun GPU, aucun MPI.

## 5. Architecture ADC utilisee

Le cas n'utilise du module `adc` que le sous-systeme DSL/prototype, PAS les briques compilees ni
`adc.System` :

- `adc.dsl.HyperbolicModel` — modele hyperbolique declaratif (arbre d'expressions). Methodes
  employees : `conservative_vars`, `primitive`, `set_flux`, `set_eigenvalues`, `check`,
  proprietes `n_vars` / `cons_names`, et `to_python_flux`.
- `adc.dsl.sqrt` — racine carree symbolique (noeud `Sqrt`), pour la vitesse du son.
- `adc.dsl.Expr` et ses sous-classes (`Const`, `Var`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`,
  `Sqrt`) — l'arbre construit par surcharge d'operateurs Python (`__add__`, `__mul__`...).
- `adc.PythonFlux` — backend HOTE numpy (defini dans `adc/__init__.py`). C'est lui qui contient le
  flux de Rusanov, la periodicite par `np.roll`, `residual` et `cfl_dt`. `HyperbolicModel`
  l'instancie via `to_python_flux()` en lui passant deux closures (le flux et la vitesse d'onde
  max).

Chaine exacte (cf. docstring `to_python_flux`) :

```
make_euler() -> HyperbolicModel (arbre symbolique)
            .to_python_flux()  -> adc.PythonFlux(flux=lambda U,d: model.flux(U,{},d),
                                                  max_wave_speed=lambda U: max(.. dir 0, dir 1))
            pf.residual / pf.cfl_dt  -> assemblage Rusanov numpy + CFL
```

Distinction avec les autres `*_dsl` (a retenir) : ce cas appelle `to_python_flux()` (chemin
INTERPRETE). Les cas `diocotron_dsl` / `two_species_dsl` / `magnetic_isothermal_dsl` appellent
`emit_cpp_brick` / `emit_cpp_source` -> `add_compiled_model` (chemin COMPILE, `needs = ["cxx"]`),
absent ici.

## 6. Carte des fichiers

| Chemin | Role |
|---|---|
| `dsl_euler/run.py` | LE cas. Declare le modele Euler en formules, construit la CI, integre 120 pas, asserte les invariants, imprime les diagnostics. Seul fichier propre au cas (~99 lignes). |
| `adc_cases/common/grid.py` | `meshgrid_xy(n, L)` : grilles `(X, Y)` a centres de cellules, convention `field[j, i]`. |
| `adc_cases/common/initial_conditions.py` | `euler_pressure(U, gamma)` : pression d'un etat conservatif Euler (utilisee pour les diagnostics et l'assertion `p > 0`). |
| `adc_cases/common/checks.py` | `assert_finite` (pas de NaN/Inf), `relative_drift` (derive relative de la masse). |
| `adc_cases/__init__.py` | `ensure_importable()` ; ici le `run.py` gere lui-meme le `sys.path` (try/except sur `import adc_cases`). |
| `<build>/python/adc/dsl.py` | mini-DSL : `HyperbolicModel`, `sqrt`, l'arbre `Expr` et l'interprete numpy (`flux`, `max_wave_speed`, `to_python_flux`). FOURNI PAR LE BUILD adc_cpp, hors depot adc_cases. |
| `<build>/python/adc/__init__.py` | facade `adc` ; contient `PythonFlux` (Rusanov + periodicite + `residual`/`cfl_dt`). |

NB : le cas n'utilise PAS `adc_cases.models.euler` (qui, lui, compose des briques natives
`adc.CompressibleFlux`). Ici tout est ecrit en formules dans `run.py`.

## 7. Prerequis

- Python 3.12 + numpy (teste : Python 3.12.2, numpy 1.26.4).
- Le module `adc` (bindings pybind11 d'adc_cpp) accessible via `PYTHONPATH`. Le binaire compile
  `_adc.cpython-312-darwin.so` n'est PAS exerce par le hot path (tout passe par `dsl.py` /
  `PythonFlux` en numpy), mais l'import `from adc import dsl` charge quand meme le package `adc`.
- AUCUN compilateur C++ requis (`needs = []`). C'est la difference clef avec les autres `*_dsl`.
- Le paquet `adc_cases` importable : soit installe (`pip install -e .`), soit son depot sur le
  `PYTHONPATH` (le `run.py` insere automatiquement la racine du depot en cas d'`ImportError`).

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/dsl_euler && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le premier chemin du `PYTHONPATH` fournit le module `adc` (build), le second fournit le paquet
`adc_cases`. Adapter les chemins a votre arborescence (build adc_cpp et racine du depot adc_cases).

## 9. Explication du code par etapes

`run.py` :

1. **Imports** : `numpy`, `from adc import dsl`, et les helpers `assert_finite`, `relative_drift`,
   `meshgrid_xy`, `euler_pressure`. `GAMMA = 1.4`.

2. **`make_euler()` — declaration du modele en formules** :
   - `e = dsl.HyperbolicModel("euler")` cree un modele vide.
   - `rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")` declare 4 variables
     conservatives (retourne des noeuds `Var`).
   - primitives definies par leurs expressions : `u = e.primitive("u", rhou/rho)`,
     `v = e.primitive("v", rhov/rho)`, `p = e.primitive("p", (GAMMA-1)*(E - 0.5*rho*(u*u+v*v)))`.
     Chaque `/`, `*`, `-` construit un noeud d'arbre (`Div`, `Mul`, `Sub`...).
   - quantites derivees PURE-Python (pas enregistrees comme primitives, juste reutilisees dans les
     formules) : `H = (E + p)/rho`, `c = dsl.sqrt(GAMMA*p/rho)`.
   - `e.set_flux(x=[...], y=[...])` : les 4 composantes du flux par direction (cf. section 2).
   - `e.set_eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])`.
   - `e.check()` : verifie que toute variable referencee (dans primitives, flux, valeurs propres) a
     bien ete declaree comme cons/prim/aux ; leve `ValueError` sinon. Ici tout est defini, donc OK.

3. **Affichage du modele** : `print("modele declare en formules : %d variables %s" %
   (euler.n_vars, euler.cons_names))` -> `4 variables ['rho', 'rho_u', 'rho_v', 'E']`.

4. **Grille et CI** : `n = 64`, `L = 1.0`, `h = L/n`. `gx, gy = meshgrid_xy(n, L)`,
   `r2 = (gx-0.5)^2 + (gy-0.5)^2`. Bulle de pression `p0 = 1 + 0.4*exp(-r2/0.01)`. Etat conservatif
   `U` de forme `(4, 64, 64)` : `U[0] = 1` (rho), `U[1] = U[2] = 0` (quantites de mouvement nulles),
   `U[3] = p0/(GAMMA-1)` (energie au repos = p/(gamma-1)).

5. **References pour les invariants** : `mass0 = U[0].sum()`, `p_init = pressure(U).copy()`.

6. **Construction du flux interprete** : `pf = euler.to_python_flux()`. Renvoie un `adc.PythonFlux`
   dont les closures appellent l'interprete numpy du modele.

7. **Boucle d'integration** (120 pas) : a chaque pas
   `U = U + pf.cfl_dt(U, h, 0.4) * pf.residual(U, h)` (Euler avant, CFL 0.4, dt reevalue a chaque
   pas).

8. **Diagnostics et assertions** (cf. sections 11-12) : `relative_drift` de la masse,
   `max|p - p_init|` (dynamique), affichage de `drho_max`, `|v|_max`, puis les 4 assertions.

## 10. Conditions initiales

Definies directement dans `run.py` (et NON via `euler_pressure_blob` du module commun, bien que le
profil soit le meme esprit) :

```
rho(x, y) = 1
u = v = 0
p(x, y)   = 1 + 0.4 * exp( -r^2 / 0.01 ),   r^2 = (x - 0.5)^2 + (y - 0.5)^2
E(x, y)   = p / (gamma - 1)          (energie au repos : 0.5 rho |v|^2 = 0)
```

Gaz au repos avec une surpression gaussienne (+40 %) au centre d'ecart-type ~0.07 (variance 0.01).
La detente de cette bulle est ce qui « met le systeme en mouvement » et genere l'onde acoustique
mesuree par l'assertion de dynamique.

## 11. Invariants et assertions

Les 4 assertions de `run.py` (toutes verifiees lors de l'execution, cf. section 12) :

| Assertion | Code | Resultat mesure |
|---|---|---|
| Etat fini | `assert_finite(U, "etat")` | OK (aucun NaN/Inf) |
| Positivite | `assert U[0].min() > 0 and pressure(U).min() > 0` | OK (rho > 0 et p > 0) |
| Masse conservee | `assert drel < 1e-9` | `drel = 0.00e+00` (bien < 1e-9) |
| Dynamique non triviale | `assert moved > 1e-3` | `moved = 0.394` (>> 1e-3) |

- `drel = relative_drift(U[0].sum(), mass0)` : derive relative de la masse totale. Le flux de
  masse est en forme conservative et le domaine periodique, donc la masse est conservee a l'arrondi
  pres — ici exactement `0.00e+00` (les flux de bord s'annulent par `np.roll`).
- `moved = max|pressure(U) - p_init|` : ecart maximal de pression vs l'instant initial, preuve que
  la bulle s'est detendue (sinon le cas validerait un etat fige). Mesure `0.394`.

A noter : le cas asserte une masse strictement conservee, mais PAS la conservation de la quantite de
mouvement ni de l'energie totale (Euler avant ordre 1 + diffusion de Rusanov dissipent l'energie ;
c'est attendu pour un prototype).

## 12. Sorties attendues

Sortie console reelle (capturee, reproductible bit-a-bit sur 3 executions) :

```
modele declare en formules : 4 variables ['rho', 'rho_u', 'rho_v', 'E']
apres 120 pas : drho_max=0.123  |v|_max=0.027
masse : drel=0.00e+00   dynamique : max|dp|=0.394
OK dsl_euler
```

Interpretation :
- `drho_max = 0.123` : amplitude des variations de densite (max - min) creees par l'onde.
- `|v|_max = 0.027` : vitesse maximale atteinte (la bulle est de faible amplitude, le regime reste
  presque incompressible).
- `drel = 0.00e+00` : masse conservee exactement.
- `max|dp| = 0.394` : la pression a bouge de pres de 0.4 (la bulle s'est detendue).
- `OK dsl_euler` : les 4 assertions sont passees.

Code de retour 0. Le cas n'ecrit AUCUN fichier (pas de figure, pas de GIF, pas de dump). Seule
sortie : la console.

## 13. Generation figures/GIF

Aucune. Ce cas ne produit aucun artefact graphique ni fichier de sortie : il n'importe pas
`matplotlib` (cf. `needs = []`), n'utilise pas `adc_cases.common.io`, et ne sauvegarde rien. C'est
un cas de validation de l'API declarative, pas de reproduction figuree. Pour visualiser l'onde, il
faudrait ajouter soi-meme un `imshow` sur `pressure(U)` ; ce n'est pas dans le scope du cas.

## 14. Backends reellement supportes

- **CPU / numpy interprete UNIQUEMENT.** Tout le calcul passe par `HyperbolicModel.flux` /
  `max_wave_speed` (evaluation de l'arbre `Expr` en numpy) et par `adc.PythonFlux` (assemblage
  Rusanov numpy). Aucun kernel Kokkos n'est appele.
- **PAS de GPU, PAS de MPI, PAS de multi-box / AMR.** `PythonFlux` est documente comme « HORS hot
  path GPU/MPI : chemin HOTE pur ». Le tableau est un unique `(4, 64, 64)` numpy mono-bloc.
- **PAS de chemin compile.** Contrairement aux autres `*_dsl`, ce cas n'appelle ni `emit_cpp*` ni
  `add_compiled_model` ; il n'y a donc rien a compiler et `needs = []`. Le binaire `_adc.*.so` est
  charge a l'import du package mais n'execute aucune boucle de ce cas.

En clair : c'est un PROTOTYPE hote. La « production » (GPU/MPI/bit-identique au natif) passerait par
le codegen C++ du meme arbre (`emit_cpp_brick`), demontre par les cas `*_dsl` compiles, pas par
celui-ci.

## 15. Cout approximatif

Mesure reelle sur cette machine (Apple Silicon, `/usr/bin/time`), 3 executions :

| Run | Temps mur (real) |
|---|---|
| 1 | 0.29 s |
| 2 | 0.25 s |
| 3 | 0.22 s |

- Temps mur typique : **~0.2 a 0.3 s** (dont une part dominante = import du package `adc` /
  chargement du `.so` ; le calcul pur 120 pas sur 64x64 est negligeable).
- Pic memoire resident : **~44 Mo** (`maximum resident set size` 44 417 024 octets) ;
  `peak memory footprint` ~33 Mo.
- Mono-thread cote calcul (numpy vectorise sur 64x64). Aucun acces disque, aucune compilation.

C'est le cas le moins couteux de la famille DSL (aucune compilation C++ a amortir). A `ci = false`
uniquement par prudence (statut `experimental`), pas pour des raisons de cout.

## 16. Limites et differences avec les references

- **Prototype, pas production.** L'interprete numpy n'est PAS le chemin de production d'adc. Il sert
  a iterer sur des equations inedites sans recompiler. La voie de production reste les briques
  compilees (et, pour le DSL, le codegen C++ des autres cas `*_dsl`). Ne pas presenter ce cas comme
  une demonstration de performance ou de fidelite GPU.
- **Schema d'ordre 1, dissipatif.** Rusanov ordre 1 + Euler avant : forte diffusion numerique,
  energie non conservee, fronts etales. Adapte a une demo qualitative (« ca bouge, c'est stable,
  la masse est conservee »), pas a une etude quantitative de l'acoustique.
- **Pas de reference publiee.** Ce cas ne vise aucun resultat d'article (categorie `experimental`,
  pas `reproduction` ni `reproduction-candidate`). Les seuls criteres sont les invariants internes
  (masse, positivite, finitude, dynamique non nulle), pas une comparaison a une solution de
  reference.
- **Source / Poisson absents.** Le modele declare ne pose ni terme source ni second membre
  elliptique : c'est de l'Euler PUR. Pour le couplage electrostatique/magnetique en DSL, voir les
  cas dedies (`two_species_dsl`, `magnetic_isothermal_dsl`, `hoffart_euler_poisson_dsl`).
- **Geometrie fixe.** 64x64 periodique en dur dans `run.py` ; pas de parametre de ligne de commande.

## 17. Tests/CI associes

- **Hors CI** (`ci = false` dans `cases_manifest.toml`, categorie `experimental`). La CI
  (`.github/workflows/ci.yml`) ne lance QUE les cas legers marques `ci = true` ; ce prototype
  interprete reste hors CI par prudence, malgre son cout negligeable et `needs = []`.
- **Auto-validation interne.** Le cas se teste lui-meme : ses 4 assertions (masse, positivite,
  finitude, dynamique) font echouer le process en `AssertionError` si la physique deraille. Il se
  lance donc a la main avec la commande de la section 8 et renvoie 0 si tout passe (`OK dsl_euler`).
- **Cas compiles equivalents (couverts en CI).** Les cas `*_dsl` qui COMPILENT le DSL en C++
  (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl` ; `needs = ["cxx"]`, `ci = true`)
  valident, eux, l'equivalence bit-a-bit DSL-compile vs brique native. Ce cas-ci valide le bout
  amont (declaration + interprete), pas l'equivalence au natif.
