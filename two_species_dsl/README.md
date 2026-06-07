# two_species_dsl : deux especes ecrites en formules (DSL), equivalentes au natif

Cas de **validation** (`ci = true`, `needs = ["cxx"]`). Deux fluides heterogenes --
electrons (Euler compressible, 4 variables) et ions (Euler isotherme, 3 variables) --
sont ecrits **entierement en formules symboliques** via `adc.dsl.Model`, couples par
**un seul Poisson** dont le second membre agrege les charges des deux especes, puis
**compares bit-a-bit** a la composition native equivalente (briques nommees de
`adc_cases.models`). Le cas prouve qu'un modele DSL **a source electrostatique** et a
**contribution elliptique** reproduit le chemin natif a l'epsilon machine pres.

Label du manifeste : `category = "validation"`. Ce n'est pas une reproduction d'une
reference physique externe : c'est un test d'**equivalence DSL <-> natif** interne a la
bibliotheque.

---

## 1. Objectif du cas

Demontrer, sur un systeme **multi-especes couple par Poisson**, que la voie declarative
(modeles ecrits en formules avec `adc.dsl.Model`, compiles en `.so`) est **numeriquement
equivalente** a la voie native (composition de briques C++ nommees). Concretement :

- deux especes DSL de **tailles d'etat differentes** (4 et 3 variables) dans le meme `System` ;
- un modele DSL **a source** (la force electrostatique lit `grad phi` par le canal `aux`) ;
- un **Poisson couple** dont le RHS additionne les densites de charge des deux blocs DSL.

Critere de reussite (asserte par le script) : par espece, l'etat final DSL coincide avec
l'etat natif (electrons a `< 1e-24`, ions bit-identiques), la masse est conservee par
espece, les densites restent finies et strictement positives.

L'interet va au-dela du diocotron mono-espece deja couvert ailleurs : ici le DSL doit
gerer **source + couplage elliptique + heterogeneite des especes** simultanement.

---

## 2. Equations

Chaque espece `s` est un systeme hyperbolique 2D `dU_s/dt + d_x F_s(U_s) + d_y G_s(U_s) = S_s(U_s, E)`,
les deux especes etant reliees par un champ electrostatique commun `E = -grad phi`,
`phi` solution d'**un seul** Poisson.

### Electrons -- Euler compressible (4 variables)

Etat conservatif `U_e = (rho, rho u, rho v, E)` ; pression `p = (gamma - 1)(E - 1/2 rho |v|^2)`,
`gamma = 5/3`. Flux (convention `include/adc/physics/euler.hpp`) :

```
F_x = [ rho u,  rho u^2 + p,  rho u v,        (E + p) u ]
F_y = [ rho v,  rho u v,      rho v^2 + p,    (E + p) v ]
```

Valeurs propres `(u - c, u, u, u + c)` en x et `(v - c, v, v, v + c)` en y, avec
`c = sqrt(gamma p / rho)`.

### Ions -- Euler isotherme (3 variables)

Etat conservatif `U_i = (rho, rho u, rho v)` ; fermeture isotherme `p = cs2 rho`, `cs2 = 1`.
Flux (convention `IsothermalFlux`, `include/adc/physics/hyperbolic.hpp`) :

```
F_x = [ rho u,  rho u^2 + p,  rho u v ]
F_y = [ rho v,  rho u v,      rho v^2 + p ]
```

Valeurs propres `(u - c, u, u + c)` / `(v - c, v, v + c)`, `c = sqrt(cs2)` constant.

### Source electrostatique (force et travail)

Convention `PotentialForce` (`include/adc/physics/source.hpp`), `E = (-grad_x phi, -grad_y phi)` :

```
S_e = [ 0,  q_e rho E_x,  q_e rho E_y,  q_e (rho u E_x + rho v E_y) ]   (electrons : 4 var, avec travail)
S_i = [ 0,  q_i rho E_x,  q_i rho E_y ]                                  (ions : 3 var, sans energie)
```

avec `q_e = -1`, `q_i = +1` (la charge joue le role de `q/m` dans le coeur, champ `qom`).

### Poisson couple

`phi` resout `lap phi = f` avec un **second membre unique** agregeant les densites de
charge des deux especes (convention `ChargeDensity`, `include/adc/physics/elliptic.hpp`,
`rhs = q n`, `n = U[0]`) :

```
f = q_e n_e + q_i n_i
```

Conditions aux limites periodiques.

---

## 3. Modele physique

Plasma a deux fluides electrostatique, non magnetise (pas de `B_z`), avec separation de
charge initiale. Les electrons sont compressibles adiabatiques (`gamma = 5/3`), les ions
isothermes (`cs2 = 1`). Le couplage est **electrostatique pur** : chaque espece subit la
force `q rho E` (et son travail pour les electrons), et chaque espece **alimente** le
Poisson par sa densite de charge. Aucune collision ni echange thermique inter-especes
n'est active (pas de `CoupledSource`).

Le but n'est pas un regime physique de reference : c'est de **mettre en regard** deux
ecritures du **meme** modele (formules vs briques) sur un systeme suffisamment riche
(deux especes heterogenes + source + Poisson partage) pour exercer tout le chemin DSL.

---

## 4. Methode numerique

Identique pour la voie native et la voie DSL (c'est la condition de l'equivalence
bit-a-bit) :

- **Volumes finis** 2D, reconstruction limitee **minmod**, flux de Riemann **rusanov** ;
- integration temporelle **explicite SSPRK2** (`adc.Explicit()` ; le DSL emis tourne le
  meme `assemble_rhs<Limiter, Flux>` + SSPRK2 que le natif) ;
- **pas de sous-cyclage** : les deux blocs avancent au meme `dt` (`step_cfl(0.4)`), sinon
  la comparaison bit-a-bit avec le natif (qui avance les deux blocs au meme pas) serait
  rompue ; `add_equation` accepte un `time`/`substeps` par bloc, mais ce cas ne s'en sert
  volontairement pas ;
- pas de temps par **CFL global** : `sim.step_cfl(0.4)` (CFL = 0.4) ;
- **Poisson** : RHS `charge_density`, solveur `geometric_mg` (multigrille geometrique),
  domaine periodique.

Grille `48 x 48`, domaine `L = 1.0`, `15` pas de temps.

---

## 5. Architecture ADC utilisee

Deux voies construites cote a cote dans `run.py`, puis comparees.

### Voie native (oracle de reference) -- `run_native`

```python
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("electrons", model=models.electron_euler(charge=Q_E, gamma=GAMMA_E),
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.add_block("ions",      model=models.ion_isothermal(charge=Q_I, cs2=CS2_I),
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

`models.electron_euler` / `models.ion_isothermal` (dans `adc_cases/models.py`) sont des
`adc.Model(state, transport, source, elliptic)` composant les briques natives
`FluidState` + `CompressibleFlux`/`IsothermalFlux` + `PotentialForce` + `ChargeDensity`.

### Voie DSL (sujet du cas) -- `run_dsl`

Chaque espece est un `adc.dsl.Model` decrit **en formules** (cf. section 9), compile en
`.so`, puis branche via `add_equation` (qui aiguille sur le **type** du modele : un
`CompiledModel` part vers l'adder du backend) :

```python
ce, be = _compile(electron_dsl_model(), "electron")   # CompiledModel
ci, bi = _compile(ion_dsl_model(),      "ion")
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_equation("electrons", model=ce,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
sim.add_equation("ions", model=ci,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

`adc.Spatial(minmod=True)` (natif) et `adc.FiniteVolume(limiter="minmod", riemann="rusanov")`
(DSL) decrivent le **meme** schema spatial : c'est ce qui garantit la comparaison a
schema identique.

### Compilation DSL et choix de backend -- `_compile`

`_compile(model, tag)` (dans `run.py`) tente d'abord le backend `"production"` (loader
natif zero-copie, `add_native_block`), puis bascule sur `"aot"` (`.so` autosuffisant,
chemin de production host-marshale, `add_compiled_block`) **si la compilation production
echoue**. Les deux backends sont numeriquement identiques ; seul l'AOT marshale des
tableaux au lieu d'installer le modele comme bloc natif.

```python
def _compile(model, tag):
    include = adc_include()
    so_dir = case_output_dir("two_species_dsl")
    for cand in ("production", "aot"):
        try:
            c = model.compile(os.path.join(so_dir, "%s_%s.so" % (tag, cand)), include, backend=cand)
            return c, cand
        except Exception as exc:
            print("backend %r indisponible pour %s (%s)" % (cand, tag, type(exc).__name__))
    raise RuntimeError("aucun backend DSL n'a compile le modele %s" % tag)
```

**Limite importante du fallback** (cf. section 14) : ce `try/except` ne couvre que l'etape
de **compilation**. Le backend `"production"` compile sans erreur ; l'incompatibilite
d'ABI eventuelle n'est detectee que plus tard, au **branchement** (`add_native_block`,
dans `run_dsl`), donc hors de ce `try/except` -- le fallback AOT ne se declenche alors pas
automatiquement.

---

## 6. Carte des fichiers

| Fichier | Role |
| --- | --- |
| `two_species_dsl/run.py` | Le cas : modeles DSL, oracle natif, run couple, asserts d'equivalence et d'invariants. |
| `adc_cases/models.py` | Modeles d'espece natifs `electron_euler` / `ion_isothermal` (compositions de briques), oracle de reference. |
| `adc_cases/common/native.py` | `adc_include()` : localise les en-tetes `adc_cpp` (`$ADC_INCLUDE`, paquet installe, depot voisin). |
| `adc_cases/common/io.py` | `case_output_dir("two_species_dsl")` : cree/retourne `<out_root>/two_species_dsl` (les `.so` y atterrissent). |
| `adc_cases/common/checks.py` | `assert_finite`, `assert_positive`, `relative_drift` (invariants physiques). |
| `adc/dsl.py` (paquet `adc`) | Le mini-DSL : `Model`, arbre d'expressions, codegen C++, backends `prototype`/`aot`/`production`. |
| `adc/__init__.py` (paquet `adc`) | `System`, `Spatial`, `FiniteVolume`, `Explicit`, `add_block`, `add_equation`, `set_poisson`, `step_cfl`. |

Ce cas ne porte **pas** de C++ sur mesure ni de `check_model.py`/`band_instability.py` : tout
le C++ est **genere** par `adc.dsl` a partir des formules de `run.py`.

---

## 7. Prerequis

- **Module `adc`** (bindings pybind11 d'`adc_cpp`) accessible par `PYTHONPATH` (le build
  fournit `build-master/python/adc`).
- **Paquet `adc_cases`** importable (installe via `pip install -e .`, ou place sur le
  `PYTHONPATH` ; `run.py` retombe sinon sur le dossier parent du script).
- **Un compilateur C++** (`needs = ["cxx"]`) : `$CXX`, sinon `c++` / `g++` / `clang++`.
  Le DSL compile chaque modele en `.so` (`-std=c++23` pour `production`, `-std=c++20` pour
  `aot`).
- **Les en-tetes `adc_cpp`** (`include/`), localises par `adc_include()` ; surchargeables
  par `ADC_INCLUDE=<adc_cpp>/include`.
- **Contrainte d'ABI pour le backend `production`** : les en-tetes utilises pour compiler
  le loader natif doivent etre **exactement** ceux contre lesquels le module `_adc` a ete
  bati (meme signature de headers, compilateur, `std`). Sinon `add_native_block` rejette le
  loader (cf. sections 14 et 16). Le backend `aot` n'a pas cette contrainte.

NumPy est requis (importe par `run.py`).

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/two_species_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Pour forcer un dossier de sortie hors source : `export ADC_CASES_OUT=/chemin/out`.
Pour imposer les en-tetes : `export ADC_INCLUDE=/.../adc_cpp/include`.

---

## 9. Explication du code par etapes

1. **Parametres physiques partages** (`GAMMA_E = 5/3`, `CS2_I = 1`, `Q_E = -1`, `Q_I = +1`).
   Ils doivent etre identiques entre DSL et natif, sinon l'equivalence ne tient pas.

2. **`electron_dsl_model()`** -- electrons en formules :
   - `conservative_vars("rho", "rho_u", "rho_v", "E")` ;
   - `aux("grad_x")`, `aux("grad_y")` : composantes canoniques du canal `aux` (lues comme
     `a.grad_x` / `a.grad_y` dans le C++ genere) ;
   - `param("gamma", GAMMA_E)` : constante **nommee**, inlinee au codegen ET enregistree
     comme metadonnee ABI (`set_gamma`) pour la coherence des couplages cote `System` ;
   - primitives `u = rho_u/rho`, `v = rho_v/rho`, `p = (g-1)(E - 1/2 rho(u^2+v^2))`, son
     `c = sqrt(g p / rho)` ;
   - `flux(x=[...], y=[...])` et `eigenvalues(x=[...], y=[...])` (conventions `euler.hpp`) ;
   - `primitive_vars(rho, u, v, p)` (forme **positionnelle** : fixe le layout de `Prim`
     sans redefinir les primitives) puis `conservative_from([...])` (inverse prim -> cons,
     que le DSL ne sait pas deriver seul) ;
   - `source([0.0, Q_E rho E_x, Q_E rho E_y, Q_E (rho_u E_x + rho_v E_y)])` avec
     `E = -grad phi` ;
   - `elliptic_rhs(Q_E rho)` : contribution au RHS de Poisson ;
   - `check()` verifie que toute variable referencee est declaree.

3. **`ion_dsl_model()`** -- ions en formules : meme structure, 3 variables, `p = cs2 rho`,
   `c = sqrt(cs2)` constant, source sans composante energie, `elliptic_rhs(Q_I rho)`.

4. **`initial_conditions(n)`** -- separation de charge (section 10).

5. **`run_native(...)`** -- construit le `System` natif (deux `add_block`), pose le Poisson
   couple, fixe les densites, avance `n_steps` fois (`step_cfl(0.4)`), retourne les etats
   et masses par espece. C'est **l'oracle**.

6. **`_compile(model, tag)`** -- compile un modele DSL en `.so` (backend `production`,
   sinon `aot`) ; retourne `(CompiledModel, backend)`.

7. **`run_dsl(...)`** -- compile les deux modeles DSL, construit le **meme** `System`
   (deux `add_equation`), pose le **meme** Poisson, fixe les **memes** densites, avance
   `n_steps` fois, retourne etats + masses + le backend retenu.

8. **`main()`** -- lance les deux voies, calcule `max|DSL - natif|` par espece, **asserte**
   l'equivalence (electrons `< 1e-24`, ions bit-identiques) puis les invariants physiques
   (masse, finitude, positivite), et imprime un resume.

---

## 10. Conditions initiales

`initial_conditions(n)` impose une **separation de charge** : electrons perturbes par un
cosinus le long de `x`, ions uniformes.

```python
x = (np.arange(n) + 0.5) / n          # centres de cellules le long des colonnes
ne = 1.0 + 0.02 * np.cos(2.0 * np.pi * x)
ne2d = np.broadcast_to(ne, (n, n)).copy()   # densite electronique 2D
ni2d = np.ones((n, n))                       # densite ionique uniforme
```

Les densites different localement, donc le RHS de Poisson `f = q_e n_e + q_i n_i` est non
nul et le couplage est reellement exerce. Les vitesses initiales sont nulles (seules les
densites sont posees via `set_density` ; le reste de l'etat est initialise par defaut a
partir de la densite par le coeur).

---

## 11. Invariants et assertions

Tous verifies par `assert` dans `main()` ; un echec sort en erreur (CI rouge). Valeurs
**reellement mesurees** lors de l'execution capturee (grille 48x48, 15 pas, backend `aot`,
cf. sections 12 et 14) :

| Invariant | Assertion | Valeur mesuree |
| --- | --- | --- |
| Equivalence electrons | `max|DSL - natif| < 1e-24` | `4.930e-32` (PASSE) |
| Equivalence ions | bit-identique ou `< 1e-24` | `0.000e+00` (bit-identique) |
| Masse electrons conservee | derive relative `< 1e-9` | `1.204e-14` |
| Masse ions conservee | derive relative `< 1e-9` | `1.165e-14` |
| Densites finies | `assert_finite` (pas de NaN/Inf) | PASSE (electrons + ions) |
| Densites positives | `assert_positive` (min > 0) | PASSE (electrons + ions) |

**Pourquoi l'ecart electrons est non nul mais < 1e-24.** A **un** pas, le residu et le
flux de chaque espece DSL sont bit-identiques au natif. Sur plusieurs pas **couples**, la
seule difference est une **reassociation flottante** dans l'accumulation du **second
membre du Poisson partage** (les deux blocs y contribuent ; l'ordre d'addition differe
legerement). Ce n'est pas un ecart de physique mais un epsilon machine : d'ou la tolerance
serree `1e-24` (et non un simple "proche"). Les ions, qui contribuent differemment a cette
accumulation, restent **exactement** bit-identiques (`0.0`).

---

## 12. Sorties attendues

Sortie console reelle (execution capturee, backend **AOT**, cf. section 14 pour la
nuance production/AOT) :

```
=== two_species_dsl : electrons + ions ecrits en formules vs briques natives ===
grille 48 x 48, 15 pas, CFL = 0.4 ; q_e = -1, q_i = 1
backend DSL retenu : 'aot'
electrons : max|DSL - natif| = 4.930e-32 (bit-identique = False)
ions      : max|DSL - natif| = 0.000e+00 (bit-identique = True)
masse electrons : derive relative 1.204e-14 ; ions : 1.165e-14
OK two_species_dsl (equivalence DSL <-> natif par espece, backend 'aot')
```

**Artefacts produits** : les `.so` des modeles DSL, dans
`<out_root>/two_species_dsl/` (par defaut `<racine_paquet>/out/two_species_dsl/`, ou
`$ADC_CASES_OUT/two_species_dsl/`). Observe apres execution :

```
electron_aot.so   ion_aot.so          # backend AOT (chemin emprunte ici)
electron_production.so  ion_production.so   # backend production (compiles, mais rejetes au branchement, cf. 14/16)
```

Le cas ne produit ni figure ni GIF.

---

## 13. Generation figures/GIF

**Aucune.** Ce cas est un test d'equivalence numerique : il n'ecrit aucune image. Sa seule
sortie disque est constituee des bibliotheques partagees `.so` generees par le DSL (cf.
section 12). La preuve du cas est la sortie console + le code de retour (0 si tous les
`assert` passent).

---

## 14. Backends reellement supportes

| Backend DSL | Adder `System` | Statut dans CE cas / cet environnement |
| --- | --- | --- |
| `production` (loader natif zero-copie) | `add_native_block` | **Tente en premier**, **compile**, mais **REJETE au branchement** dans l'environnement teste (ABI de headers divergente, cf. ci-dessous et section 16). |
| `aot` (`.so` autosuffisant, host-marshale) | `add_compiled_block` | **Fonctionne** ; numerique identique au natif. C'est le backend reellement emprunte ici. |
| `prototype` (JIT, dispatch virtuel hote) | `add_dynamic_block` | Non utilise par ce cas (`_compile` n'essaie que `production` puis `aot`). |

**Etat reel observe a l'execution.** Avec la commande de la section 8 **sans modification**,
le backend `production` compile les deux `.so` puis `add_native_block` echoue avec :

```
RuntimeError: add_native_block : ABI incompatible -- cle du loader
'...;headers=408168b4...' != cle du module '...;headers=f8273719...'.
Recompiler le loader avec le MEME compilateur, standard C++ et en-tetes adc que le module _adc.
```

Le compilateur (`Apple LLVM 21.0.0`) et le standard (`std=202302L`) **concordent** ; seule
la **signature des en-tetes** differe : les en-tetes `include/` du depot ont evolue depuis
le build du module `_adc` (le superprojet montre `M adc_cpp` -- sous-module modifie). Comme
ce rejet survient **hors** du `try/except` de `_compile` (au branchement, pas a la
compilation), le **fallback AOT automatique ne se declenche pas** et le cas s'arrete en
erreur **dans cet environnement precis**.

**Le cas lui-meme est correct.** En forcant le backend `aot` (ABI tolerante par
construction : `.so` autosuffisant, pas de partage d'ABI avec `_adc`), le cas passe
**integralement** avec les valeurs de la section 11. La cause racine est purement un
**desynchronisation build/headers** de l'environnement (le module `_adc` de `build-master`
a ete bati contre un instantane anterieur des en-tetes), pas un defaut du cas. Pour faire
passer la voie nominale `production` : **rebatir le module `_adc`** contre les en-tetes
courants (ou utiliser des en-tetes correspondant exactement au module).

---

## 15. Cout approximatif

Mesure (temps mur, `/usr/bin/time -p`), machine de developpement (Apple Silicon, macOS,
Python 3.12) :

- **Execution forcee en AOT** (cas complet : compilation des deux `.so` + run natif +
  run DSL + asserts) : **`real ~6.3 s`** (`user ~5.8 s`, `sys ~0.8 s`).
- **Execution nominale (tente production puis echoue au branchement)** : **`real ~5.4 s`**
  (arret apres compilation des deux `.so` production et l'echec `add_native_block`).

L'essentiel du cout est la **compilation C++** des modeles DSL (deux invocations du
compilateur). Sur un re-run, le cache hors source du DSL (`adc_cache_dir`, keye par
`model_hash + abi_key`) evite la recompilation **uniquement** si `so_path` n'est pas force ;
or `run.py` passe un `so_path` explicite (`out/.../*_<backend>.so`), donc **chaque run
recompile** (retro-compatibilite stricte du chemin force). Le calcul numerique proprement
dit (48x48, 15 pas, deux fois) est negligeable devant la compilation.

---

## 16. Limites et differences avec les references

- **Pas une reproduction d'une reference physique externe.** C'est un cas de
  **validation interne** (equivalence DSL <-> natif). Aucune comparaison a une solution
  analytique ou a un papier ; l'oracle est la composition native `adc_cases.models`.
- **Equivalence "a epsilon machine", pas exactement bit-a-bit pour les electrons.** L'ecart
  electrons (`~4.9e-32`) vient d'une **reassociation flottante** dans l'accumulation du RHS
  de Poisson **partage** par les deux blocs, asserte sous `1e-24`. Les ions sont
  bit-identiques. Ne pas presenter ce cas comme "DSL == natif bit-a-bit en toutes
  circonstances" : c'est vrai **par espece a 1 pas** et pour les ions, **a epsilon machine**
  pour les electrons sur plusieurs pas couples.
- **Backend `production` non operationnel dans l'environnement teste.** Voir section 14 :
  l'echec est une **divergence d'ABI de headers** (module `_adc` bati contre un instantane
  anterieur des en-tetes), pas un bug du cas. La voie reellement validee ici est l'**AOT**.
- **Fallback AOT non automatique sur l'echec production.** Le `try/except` de `_compile`
  ne couvre que la **compilation** ; le rejet ABI de `production` survient au **branchement**
  (`add_native_block`) et n'est donc pas rattrape. C'est une limite du script telle qu'ecrite.
- **Pas de sous-cyclage, pas d'IMEX, pas de magnetisme.** Le cas fige explicitement un
  schema strictement identique (minmod + rusanov + SSPRK2, meme `dt`) pour rendre la
  comparaison possible ; toute variation (substeps, IMEX, `B_z`) romprait l'equivalence
  bit-a-bit et n'est pas l'objet du cas.
- **Petite grille, peu de pas.** `48x48`, `15` pas : taille de **test de validation**, pas
  une simulation de production.

---

## 17. Tests/CI associes

- Manifeste (`cases_manifest.toml`) :

  ```toml
  path = "two_species_dsl/run.py"
  category = "validation"
  ci = true
  needs = ["cxx"]
  desc = "Electrons + ions en formules (adc.dsl.Model), Poisson couple ; equivalence au natif par espece."
  ```

- `ci = true` : le cas est execute en integration continue. Il **est** son propre test : la
  reussite = tous les `assert` de `main()` passent (equivalence par espece + invariants
  physiques) et code de retour 0.
- `needs = ["cxx"]` : la CI doit fournir un compilateur C++ (le DSL compile les modeles en
  `.so`). En CI, la voie nominale `production` exige en plus que les en-tetes correspondent
  au module `_adc` (cf. section 14) ; a defaut, seul l'`aot` est exploitable et le script,
  tel qu'ecrit, ne bascule pas automatiquement (limite a connaitre pour la CI).
- Aucun test unitaire C++ dedie n'accompagne ce cas (pas de C++ sur mesure : tout est
  genere par `adc.dsl`).
