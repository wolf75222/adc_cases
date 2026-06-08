# diocotron_dsl : le modele diocotron ecrit ENTIEREMENT en formules, prouve bit-identique au natif

Ce cas reecrit la physique du diocotron (derive E x B d'une densite scalaire couplee a un
Poisson de fond neutralisant) NON pas en assemblant des briques C++ nommees du coeur, mais en
DECLARANT des formules symboliques avec `adc.dsl.Model`. Le DSL genere le C++, le compile en
bibliotheque partagee (`.so`), et l'installe comme bloc du `System`. Le cas PROUVE ensuite que
l'etat produit est **bit-identique** (`np.array_equal`) a la composition native de briques. C'est
un test de validation (`category = "validation"`, `ci = true`) ; il a besoin d'un compilateur
C++20 (`needs = ["cxx"]`).

---

## 1. Objectif du cas

Demontrer que le mini-DSL declaratif `adc.dsl` n'introduit AUCUN ecart numerique : un modele
ecrit en formules (variable conservative, champs auxiliaires, flux d'advection E x B, valeurs
propres, second membre elliptique) reproduit EXACTEMENT les conventions des briques natives du
coeur. Le critere de reussite (coeur du cas) est l'egalite bit-pour-bit, sans tolerance :

```
max|DSL - natif| == 0   et   np.array_equal(etat_DSL, etat_natif) == True
```

Le cas sert donc de garde-fou de NON-REGRESSION du codegen DSL : si une formule diverge d'une
brique du coeur (`ExBVelocity`, `BackgroundDensity`), l'assertion d'equivalence echoue.

LIMITE DE PORTEE (honnetete) : ce cas N'EST PAS une reproduction d'un resultat publie. C'est une
variante minimale et periodique du diocotron (condition initiale en BANDE, pas l'anneau de
charge du benchmark arXiv:2510.11808). Le cas `diocotron/run.py` (category `reproduction`, hors
CI) vise la reproduction des figures ; ici on ne valide QUE l'equivalence DSL <-> natif et trois
invariants physiques legers. Ne pas presenter ce cas comme une reproduction du diocotron complet.

---

## 2. Equations

Densite de charge scalaire `n(x, y, t)` transportee par la derive E x B d'un potentiel `phi`,
en domaine periodique `[0, L]^2`.

Transport (conservatif, divergence nulle) :

```
d_t n + div( n v ) = 0,        v = ( -d_y phi / B0 ,  d_x phi / B0 )
```

Champ auto-consistant (Poisson de systeme, fond ionique neutralisant `n_i0`) :

```
- laplacien(phi) = rho_charge,   rho_charge = alpha ( n - n_i0 )
```

Le champ `v` est a divergence nulle (`div v = (-d_xy + d_yx) phi / B0 = 0`), donc le transport
conserve la masse `integrale(n)` sur le domaine periodique. Aucune source : `S(U) = 0`.

Parametres du cas : `B0 = 1.0`, `alpha = 1.0`, `n_i0 = moyenne(n_initial)` (assure la solubilite
du Poisson periodique : second membre `alpha(n - n_i0)` a moyenne nulle).

---

## 3. Modele physique

Une seule espece, une seule variable conservative `n` (role canonique `Density`). Le primitif
est trivial (`Prim = [n]`, primitif == conservatif : transport scalaire, pas d'inversion).

Conventions REPRODUITES a l'identique depuis le coeur (ancrees dans les briques natives) :

- Transport E x B (`include/adc/physics/hyperbolic.hpp`, `struct ExBVelocity`) :
  - `v = (-grad_y / B0, grad_x / B0)` (a divergence nulle) ;
  - flux physique `f = n * v(dir)` (une composante) ;
  - valeur propre (1 onde) `lambda = v(dir)`.
- Second membre elliptique (`include/adc/physics/elliptic.hpp`, `struct BackgroundDensity`) :
  - `rhs = alpha * (n - n_i0)`.
- Source : aucune (`adc.NoSource` cote natif ; pas de `m.source(...)` cote DSL).

Les champs auxiliaires `phi`, `grad_x`, `grad_y` sont les composantes fixes du canal `adc::Aux`
(indices canoniques 0/1/2). Le flux DSL lit `grad_x` / `grad_y` ; `phi` est declare pour
completer le contrat de base (3 composantes), il n'apparait dans aucune formule.

---

## 4. Methode numerique

Volumes finis, ordre 2 en espace, explicite en temps. IDENTIQUE entre les deux modeles :

- reconstruction MUSCL avec limiteur **minmod** (`FiniteVolume(limiter="minmod")`,
  equivalent natif `Spatial(minmod=True)`) ;
- flux numerique de Riemann **Rusanov** (Lax-Friedrichs local, `riemann="rusanov"`) ;
- variables reconstruites : conservatives (defaut) ;
- integration temporelle explicite **SSPRK2** (Runge-Kutta SSP a 2 etages ; c'est ce qu'avancent
  `adc.Explicit()` et `step_cfl` sur les chemins natif et aot, cf.
  `include/adc/runtime/compiled_block_abi.hpp` et `include/adc/numerics/time/ssprk.hpp`) ;
- pas de temps choisi par `step_cfl(0.4)` (CFL = 0.4), 60 macro-pas.

Le Poisson de systeme est resolu par multigrille geometrique (`solver="geometric_mg"`), second
membre `rhs="charge_density"` (somme des contributions `elliptic_rhs` des blocs).

Le DSL ne change PAS la numerique : les backends "aot" et "production" inlinent le MEME chemin de
production (`assemble_rhs<Limiter, Flux>`, SSPRK2) sur le modele genere que la composition native.
C'est ce qui rend l'egalite bit-pour-bit possible.

---

## 5. Architecture ADC utilisee

Le cas confronte DEUX chemins de construction du MEME bloc "ne", sur la meme grille / IC / Poisson :

1. Chemin NATIF (oracle de reference) :
   `adc_cases.models.diocotron(B0, alpha, n_i0)` =
   `adc.Model(state=adc.Scalar(), transport=adc.ExB(B0), source=adc.NoSource(),
   elliptic=adc.BackgroundDensity(alpha, n0))`, branche par `sim.add_block(...)`.

2. Chemin DSL : `adc.dsl.Model("diocotron_dsl")` declare les memes formules ; `m.compile(...,
   backend=...)` genere et compile une `.so` ; `sim.add_equation("ne", model=compiled, ...)`
   aiguille sur l'adder du backend (`add_native_block` pour "production",
   `add_compiled_block` pour "aot").

Surface DSL employee (facade `adc.dsl.Model`, cf. `python/adc/dsl.py`) :
`conservative_vars`, `aux`, `flux`, `eigenvalues`, `primitive_vars`, `conservative_from`,
`elliptic_rhs`, `check`, `compile`. Le DSL emet un
`adc::CompositeModel<HyperboliqueGen, NoSource, ElliptiqueGen>` (transport + elliptique generes,
source nulle), avec elimination des sous-expressions communes (CSE) au codegen.

Backend (voir aussi section 14) : le cas PREFERE `backend="production"` (loader natif
zero-copie, `add_native_block`, parite stricte avec `add_block`) ; si la compilation native
echoue, il retombe sur `backend="aot"` (chemin de production host-marshale, numerique identique).
Les deux donnent un etat bit-identique au natif.

---

## 6. Carte des fichiers

| Fichier | Role |
|---|---|
| `diocotron_dsl/run.py` | Le cas : construit les deux modeles, compile le DSL, asserte l'equivalence et les invariants. SEUL fichier propre au cas. |
| `adc_cases/models.py` | `diocotron(...)` = composition native de briques (oracle de reference). |
| `adc_cases/common/initial_conditions.py` | `band_density(...)` = bande gaussienne de charge perturbee (CI partagee). |
| `adc_cases/common/grid.py` | `meshgrid_xy` (convention de grille `field[j, i]`, centres de cellules). |
| `adc_cases/common/checks.py` | `relative_drift` (derive de masse relative). |
| `adc_cases/common/io.py` | `case_output_dir("diocotron_dsl")` -> `out/diocotron_dsl/` (sortie des `.so`). |
| `adc_cases/common/native.py` | `adc_include()` = localisation des en-tetes adc_cpp (pour la compilation DSL). |
| `python/adc/dsl.py` (coeur adc_cpp) | Le mini-DSL : `Model`, codegen C++, backends `prototype`/`aot`/`production`. |
| `cases_manifest.toml` | Source de verite du scope : `diocotron_dsl` = `validation`, `ci=true`, `needs=["cxx"]`. |

Le code DSL ne reside PAS dans le cas : `run.py` importe `adc.dsl` du module compile.

---

## 7. Prerequis

- Module `adc` compile et importable (ici `adc_cpp/build-master/python`).
- Paquet `adc_cases` importable (installe, ou son depot ajoute au `PYTHONPATH`).
- `numpy`.
- **Compilateur C++20** (`needs = ["cxx"]`) : `c++` / `g++` / `clang++` (ou `$CXX`). Le DSL
  compile une `.so` a la volee. Sans compilateur, `m.compile(...)` echoue.
- En-tetes du coeur adc_cpp accessibles (`adc/mesh/multifab.hpp` doit exister sous le dossier
  `include/` ; surchargeable via `$ADC_INCLUDE`).

POINT CRITIQUE (backend "production") : le loader natif partage l'ABI C++ du module `_adc` deja
charge. La cle d'ABI inclut une **signature des en-tetes adc** (`adc_header_signature`). Si les
en-tetes du dossier `include/` ont DIVERGE de ceux contre lesquels `_adc` a ete construit
(p.ex. arbre source modifie depuis le dernier build du module), `add_native_block` REJETTE le
loader avec une erreur explicite (cf. sections 12 et 16). Le backend "aot" n'a pas cette
contrainte (pas de cle d'ABI verifiee) et reste fonctionnel.

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/diocotron_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le `PYTHONPATH` rend importables le module `adc` (premier chemin) et le paquet `adc_cases`
(second chemin). Aucune option en ligne de commande.

---

## 9. Explication du code par etapes

1. **Imports et parametres** : `numpy`, `adc`, `adc.dsl` ; constantes partagees `B0 = 1.0`,
   `ALPHA = 1.0`. Import de `adc_cases.models` (oracle natif) et des helpers communs.

2. **`diocotron_dsl_model(n_i0)`** : construit le `adc.dsl.Model` :
   - `m.conservative_vars("n")` : une variable conservative `n` (role `Density`) ;
   - `m.aux("phi")`, `m.aux("grad_x")`, `m.aux("grad_y")` : champs du canal `adc::Aux` ;
   - `vx = -grad_y / B0`, `vy = grad_x / B0` : vitesse de derive E x B (convention `ExBVelocity`) ;
   - `m.flux(x=[n*vx], y=[n*vy])` : flux physique d'advection ;
   - `m.eigenvalues(x=[vx], y=[vy])` : spectre (une onde, vitesse de derive) ;
   - `m.primitive_vars(n=n)` + `m.conservative_from([n])` : layout `Prim = [n]`, inversion triviale ;
   - `m.elliptic_rhs(ALPHA * (n - n_i0))` : second membre elliptique (convention `BackgroundDensity`) ;
   - `m.check()` : verifie que toute variable referencee est declaree.

3. **`make_system(ne0)`** : `adc.System(n, L=1.0, periodic=True)`, vide (le bloc est ajoute par
   l'appelant). Garantit que grille / Poisson / densite sont identiques entre les deux runs.

4. **`run_native(ne0, n_i0, n_steps)`** (reference) : `sim.add_block("ne",
   model=models.diocotron(B0, ALPHA, n_i0), spatial=adc.Spatial(minmod=True),
   time=adc.Explicit())` ; `sim.set_poisson("charge_density", "geometric_mg")` ;
   `sim.set_density("ne", ne0)` ; 60 fois `sim.step_cfl(0.4)`. Renvoie `(densite, temps, masse)`.

5. **`run_dsl(ne0, n_i0, n_steps)`** : compile le modele DSL puis branche le bloc compile :
   - boucle de backend : essaie `"production"` puis `"aot"` ; `model.compile(out/.../diocotron_dsl_<cand>.so,
     include, backend=cand)`. La 1re compilation reussie est retenue (`break`). **Le try/except
     n'entoure QUE `model.compile()`** (voir section 16) ;
   - `sim.add_equation("ne", model=compiled, spatial=adc.FiniteVolume(limiter="minmod",
     riemann="rusanov"), time=adc.Explicit())` ;
   - meme Poisson, meme densite, memes 60 `step_cfl(0.4)`. Renvoie `(densite, temps, masse, backend)`.

6. **`main()`** : pose la CI en bande (mode 2), `n_i0 = moyenne(ne0)`, lance les deux runs,
   imprime le backend retenu, compare les etats, puis verifie les assertions (section 11).

---

## 10. Conditions initiales

`band_density(n=96, L=1.0, amp=1.0, width=0.05, mode=2, disp=0.02)` (depuis
`adc_cases/common/initial_conditions.py`) : bande horizontale de charge perturbee
sinusoidalement le long de x.

```
ne(x, y) = floor + amp * exp( -(y - y0)^2 / width^2 ),
y0       = 0.5 L + disp * cos( 2 pi * mode * x / L ),   floor = 1.0
```

Grille 96 x 96, convention `ne[j, i]` (centres de cellules). C'est la MEME grille et IC que la
variante CI native `diocotron/band_instability.py` (mode 2). Le fond ionique neutralisant
`n_i0 = moyenne(ne0)` (mesure : `1.088623e+00` pour cette CI) rend le Poisson periodique soluble.

---

## 11. Invariants et assertions

Toutes verifiees par `assert` (un echec sort en erreur, CI rouge). Valeurs MESUREES lors du run
de reference (backend "aot", voir sections 12 et 14) :

| Invariant | Assertion | Valeur mesuree |
|---|---|---|
| **Equivalence (coeur du cas)** | `np.array_equal(etat_DSL, etat_natif)` | `True`, `max|DSL - natif| = 0.000e+00` |
| Temps avance identique | `td == tn` | `t = 6.213869` (les deux) |
| Masse identique DSL/natif | `md == mn` | `1.0032746734e+04` (les deux) |
| Conservation de la masse (DSL) | `relative_drift(md, masse0) < 1e-6` | `1.813e-16` |
| Croissance de l'instabilite | `amp_finale > amp_initiale` | `6.777566e-02 -> 1.031025e-01` (facteur `1.5212`) |

L'amplitude de perturbation est la deviation L2 par rapport a la moyenne en x
(`perturbation_amplitude`, `axis=1`), comme dans le cas diocotron : la bande non perturbee est
uniforme le long de x, ce qui reste porte l'instabilite.

---

## 12. Sorties attendues

ATTENTION : sur la machine de redaction, le run nominal (`run.py` tel quel) ECHOUE avec une
erreur d'ABI au branchement du backend "production" (en-tetes du dossier `include/` diverges de
ceux du module `_adc` de `build-master` ; cf. sections 7 et 16). Sortie reelle observee :

```
=== diocotron_dsl : modele ecrit en formules (adc.dsl.Model) vs briques natives ===
grille n = 96 x 96, 60 pas, CFL = 0.4
fond ionique n_i0 = 1.088623e+00 (moyenne de ne)
Traceback (most recent call last):
  ...
  File ".../adc/__init__.py", line 979, in add_equation
    self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux, ...)
RuntimeError: add_native_block : ABI incompatible -- cle du loader '...;headers=408168b4...'
 != cle du module '...;headers=f8273719...'. Recompiler le loader avec le MEME compilateur,
 standard C++ et en-tetes adc que le module _adc.
```

Le backend "production" COMPILE bien (la `.so` est produite, voir section 13) ; l'echec survient
au branchement, hors du try/except de `run.py`. Le prerequis manquant est un module `_adc`
construit contre les MEMES en-tetes que le dossier `include/` (rebuild du module, ou un
`build-master` a jour). Une fois cette coherence retablie, le run nominal selectionne "production".

Quand le branchement reussit (ici verifie en forcant le backend "aot", numerique identique au
natif), la sortie attendue est :

```
=== diocotron_dsl : modele ecrit en formules (adc.dsl.Model) vs briques natives ===
grille n = 96 x 96, 60 pas, CFL = 0.4
fond ionique n_i0 = 1.088623e+00 (moyenne de ne)
backend DSL retenu : 'aot'        # 'production' si l'ABI du module est coherente
natif : t = 6.213869, masse = 1.0032746734e+04
DSL   : t = 6.213869, masse = 1.0032746734e+04
max|DSL - natif| = 0.000e+00   bit-identique = True
amplitude : initiale 6.777566e-02 -> finale 1.031025e-01 (facteur 1.5212)
derive de masse relative (DSL) = 1.813e-16
OK diocotron_dsl (equivalence DSL <-> natif bit-identique, backend 'aot')
```

Ces chiffres (`t = 6.213869`, masse `1.0032746734e+04`, derive `1.813e-16`, facteur `1.5212`,
`max|DSL - natif| = 0`) sont les valeurs REELLES capturees en executant le cas sur le chemin aot.

---

## 13. Generation figures/GIF

AUCUNE. Ce cas ne produit ni figure ni GIF (il n'importe pas matplotlib ; `needs` ne contient pas
`"matplotlib"`). Son unique artefact est la (ou les) bibliotheque(s) partagee(s) generee(s) par le
DSL, ecrites sous `out/diocotron_dsl/` (chemin `case_output_dir`) :

```
out/diocotron_dsl/diocotron_dsl_production.so   # produite par la tentative "production"
out/diocotron_dsl/diocotron_dsl_aot.so          # produite si on atteint le backend "aot"
```

Fichiers reellement observes apres execution (la `.so` "production" existe meme quand le
branchement echoue, puisque la compilation, elle, a reussi). Les figures du diocotron complet
sont l'affaire du cas `diocotron/run.py` (`reproduction`, hors CI).

---

## 14. Backends reellement supportes

Trois backends DSL existent (`python/adc/dsl.py`) ; ce cas en essaie deux, dans cet ordre :

| Backend | Adder System | Chemin | Statut dans ce cas |
|---|---|---|---|
| `production` | `add_native_block` | loader natif zero-copie, inline `add_compiled_model<ProdModel>`, parite stricte `add_block`, MPI/AMR-ready | PREFERE. Compile, MAIS rejete au branchement sur la machine de redaction (cle d'ABI en-tetes `408168b4...` != module `f8273719...`). |
| `aot` | `add_compiled_block` | chemin de production host-marshale, numerique inlinee, **identique** au natif | FALLBACK. VERIFIE fonctionnel : etat bit-identique au natif (`max|d| = 0`). |
| `prototype` | `add_dynamic_block` | JIT IModel, dispatch virtuel, Rusanov ordre 1 (hote) | NON utilise par ce cas. |

Capacites declarees (`_BACKEND_CAPS`) : `production` cpu/mpi/amr=True, gpu=False (validation
device end-to-end depuis Python = PR dediee) ; `aot` cpu seul. Backend EFFECTIVEMENT valide pour
l'equivalence sur cette plateforme : **aot**. Le backend "production" est la cible nominale mais
exige une coherence d'ABI module <-> en-tetes (cf. sections 7, 12, 16).

---

## 15. Cout approximatif

Cas LEGER. Temps mur mesure (Apple Silicon, Python 3.12, clang Apple 21) :

- run nominal jusqu'a l'echec ABI (compilation "production" + 60 pas natifs) : **~3.3 s** ;
- run sur le chemin aot (compilation "aot" + run natif + run DSL, 60 pas chacun) : **~2.7 s**.

Inclut la compilation de la `.so` DSL (clang `-O2 -std=c++23`/`c++20`). Pic memoire ~190 Mo
(maximum resident set). Grille 96 x 96, 60 macro-pas, 2 runs (natif + DSL). Sans GPU, sans MPI,
mono-process. Le cache de `.so` (`adc_cache_dir`, indexe par `model_hash + abi_key`) accelere les
relances quand `so_path` est omis ; ici `so_path` est explicite (`out/...`), donc la `.so` est
recompilee a chaque run.

---

## 16. Limites et differences avec les references

- **Pas une reproduction publiee.** Variante minimale periodique (CI en bande, mode 2), pas
  l'anneau de charge du benchmark arXiv:2510.11808. Le cas valide l'EQUIVALENCE DSL <-> natif et
  3 invariants legers ; il ne reproduit aucune figure ni taux de croissance publie. La
  reproduction est l'objet de `diocotron/run.py` (`reproduction`, hors CI).

- **Fallback de backend incomplet (defaut reel de `run.py`).** Le `try/except` n'entoure QUE
  `model.compile()`. Le backend "production" COMPILE sans exception, donc `backend="production"`
  est retenu et la boucle s'arrete ; l'echec d'ABI survient ensuite a `sim.add_equation(...)`
  (`add_native_block`), HORS du `try`, et n'est PAS rattrape vers "aot". Sur une plateforme ou
  l'ABI du module et les en-tetes du dossier `include/` ont diverge, le run nominal echoue donc
  malgre la promesse de fallback de la docstring. Deux corrections possibles : (a) rebuild du
  module `_adc` contre les MEMES en-tetes que `include/` (retablit "production") ; (b) elargir le
  `try/except` pour couvrir le branchement `add_equation` et retomber reellement sur "aot".

- **Egalite bit-pour-bit conditionnee a la coherence des conventions.** L'assertion
  `np.array_equal` est SANS tolerance : elle ne tient que parce que les formules DSL reproduisent
  EXACTEMENT `ExBVelocity` et `BackgroundDensity`, et parce que aot/production inlinent le meme
  chemin numerique (meme limiteur minmod, meme flux Rusanov, meme SSPRK2, meme Poisson MG). Toute
  divergence d'une formule, ou un backend a numerique differente (p.ex. "prototype", Rusanov
  ordre 1), casserait l'egalite.

- **GPU non couvert.** Capacite gpu=False pour les deux backends utilises ; ce cas est CPU
  mono-process. La validation device du chemin natif depuis Python est hors scope ici.

---

## 17. Tests/CI associes

- Manifeste `cases_manifest.toml` : `diocotron_dsl/run.py` est classe `category = "validation"`,
  `ci = true`, `needs = ["cxx"]`. Il fait donc partie des cas LEGERS lances par la CI
  (`.github/workflows/ci.yml`), a condition qu'un compilateur C++20 soit disponible sur le
  runner.

- Le cas EST son propre test : `main()` leve `AssertionError` si l'equivalence bit-pour-bit,
  l'egalite de temps/masse, la conservation de la masse ou la croissance de l'instabilite
  echouent. Aucun fichier de test separe.

- Cas frere DSL en CI (memes `needs = ["cxx"]`, meme esprit d'equivalence au natif) :
  `two_species_dsl/run.py` (electrons + ions en formules) et `magnetic_isothermal_dsl/run.py`
  (fluide isotherme magnetise en formules). Le pendant natif non-DSL valide ailleurs :
  `diocotron/band_instability.py` (CI) et `diocotron_amr/run.py` (CI).

- IMPORTANT pour la CI reelle : si le runner fournit un module `_adc` dont la signature d'en-tetes
  diverge du dossier `include/` du depot, le branchement "production" echouera comme decrit en
  section 16 (le `try/except` actuel ne retombe pas sur "aot"). Pour une CI robuste, le module
  doit etre construit contre les en-tetes du depot, OU le fallback doit etre elargi au branchement.
