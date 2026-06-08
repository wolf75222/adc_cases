# multispecies : deux fluides heterogenes couples par UN Poisson de systeme

Cas **validation** (`category = "validation"`, `ci = true`, `needs = []` dans
[`cases_manifest.toml`](../cases_manifest.toml)). Il **demontre et verifie une CAPACITE de
l'API** : faire avancer SIMULTANEMENT deux especes decrites par des modeles physiques
DIFFERENTS (electrons en Euler compressible complet, ions en Euler isotherme), couplees par
UN SEUL probleme elliptique (Poisson de systeme dont le second membre agrege les charges des
deux especes). Il ne reproduit AUCUN resultat publie et ne porte AUCUN claim physique : les CI
sont un cosinus jouet, l'integration est courte (20 pas), et le seul but est de montrer que le
**bilan de masse de chaque espece reste conserve independamment** meme quand les deux fluides
partagent le meme champ. Aucune figure, aucun GIF, aucun fichier de sortie : tout est imprime
(`print`) et verifie par `assert`.

Le script vit dans [`run.py`](run.py). Il s'appuie sur les **modeles d'espece** de
[`adc_cases/models.py`](../adc_cases/models.py) (`electron_euler`, `ion_isothermal`,
compositions de briques) et les invariants partages de
[`adc_cases/common/checks.py`](../adc_cases/common/checks.py).

---

## 1. Objectif du cas

Montrer le niveau d'abstraction vise par l'API `adc` : **Python compose, le C++ calcule**. Le
script :

1. Compose un `adc.System` periodique 48x48 avec **deux blocs heterogenes** :
   - `electrons` : Euler compressible **complet** (4 variables `rho, rho_u, rho_v, E`),
     charge `-1`, gamma = 5/3 ;
   - `ions` : Euler **isotherme** (3 variables `rho, rho_u, rho_v`), charge `+1`, ferme par
     cs2 = 1.0.
2. Les couple par **UN SEUL Poisson de systeme** dont le second membre est la densite de
   charge `f = q_e n_e + q_i n_i` (token `charge_density`).
3. Pose une **separation de charge initiale** (electrons perturbes par un cosinus, ions
   uniformes), avance 20 pas de `dt = 0.001`, et **verifie par assert** :
   - la **conservation de la masse PAR ESPECE** (`|m_e - m_e0| < 1e-9` et `|m_i - m_i0| < 1e-9`,
     en ABSOLU) : c'est le test fort du decouplage des bilans de masse -- meme couplees par le
     champ, les deux especes conservent independamment leur masse ;
   - la **finitude** des densites (pas de NaN/Inf) ;
   - la **positivite** des densites (un fluide physique reste `> 0`) ;
   - la **presence effective** d'une separation de charge initiale (`max|f| > 1e-6`), qui est
     precisement le terme source qui alimente le Poisson couple.

Le script imprime `OK multispecies` en cas de succes. Ce label est purement diagnostique.

## 2. Equations

Le cas n'introduit pas de nouvelle physique : il **reutilise** les briques generiques d'`adc`,
exposees par deux modeles nommes (cf. [`adc_cases/models.py`](../adc_cases/models.py)). Sur le
domaine periodique `[0, L]^2` (L = 1).

**Electrons (Euler compressible)** -- `models.electron_euler(charge=-1.0, gamma=5/3)`, etat
conservatif `U_e = (rho, rho u, rho v, E)` :

```
d_t rho       + div(rho v)                 = 0
d_t (rho v)   + div(rho v(x)v + p I)       = (q/m) rho E
d_t E         + div((E + p) v)             = (q/m) rho v . E
p = (gamma - 1) (E - 1/2 rho |v|^2),   gamma = 5/3,   q = -1
```

**Ions (Euler isotherme)** -- `models.ion_isothermal(charge=+1.0, cs2=1.0)`, etat
`U_i = (rho, rho u, rho v)` (PAS d'equation d'energie : fermeture par la vitesse du son) :

```
d_t rho       + div(rho v)                 = 0
d_t (rho v)   + div(rho v(x)v + cs2 rho I) = (q/m) rho E,   cs2 = 1.0,   q = +1
```

**Couplage elliptique (Poisson de systeme)** -- le champ `E = -grad phi` est self-consistant.
Le second membre est la **somme** des briques elliptiques portees par les blocs ; ici les deux
blocs portent une `ChargeDensity(charge=q)`, donc :

```
lap phi = f,   f = q_e n_e + q_i n_i = (-1) n_e + (+1) n_i,   E = -grad phi
```

C'est le coeur de la demo : un UNIQUE Poisson agrege les charges des DEUX especes (heterogenes)
et leur renvoie a chacune le MEME champ electrique via leur brique `PotentialForce`.

> Honnetete : ce sont des modeles JOUETS de validation. Ce n'est PAS un plasma physique
> calibre, ni une reproduction d'un benchmark publie. On ne mesure aucun taux de croissance,
> aucune quantite physique. Le cas verifie une PROPRIETE NUMERIQUE (conservation de masse par
> espece sous couplage), pas un resultat physique.

## 3. Modele physique

Un modele d'espece est une **composition de quatre briques generiques** assemblees par
`adc.Model(state, transport, source, elliptic)` (defini dans le module `adc` fourni par
`adc_cpp`, fichier `python/adc/__init__.py`, fonction `Model`, qui valide la coherence
etat <-> transport et renvoie une `ModelSpec`) :

| Modele (`models.py`)        | state                                          | transport            | source                        | elliptic                     |
|-----------------------------|------------------------------------------------|----------------------|-------------------------------|------------------------------|
| `electron_euler(charge=-1, gamma=5/3)` | `FluidState(kind="compressible", gamma=5/3)` | `CompressibleFlux()` | `PotentialForce(charge=-1.0)` | `ChargeDensity(charge=-1.0)` |
| `ion_isothermal(charge=+1, cs2=1.0)`   | `FluidState(kind="isothermal", cs2=1.0)`     | `IsothermalFlux()`   | `PotentialForce(charge=+1.0)` | `ChargeDensity(charge=+1.0)` |

Points cles :

- La parametrisation physique (gamma, cs2, charge) vit **dans les briques**, pas dans la config
  du systeme. `adc.System(n, L, periodic)` ne porte que le **maillage**. En particulier `gamma`
  est porte par `electron_euler(...)` (via `FluidState(gamma=...)`) et `cs2` par
  `ion_isothermal(...)`, pas par `adc.System`.
- Les deux blocs ont des **nombres de variables differents** : electrons = 4
  (`rho, rho_u, rho_v, E`), ions = 3 (`rho, rho_u, rho_v`). Le systeme les avance cote a cote
  sans les homogeneiser.
- La brique `source = PotentialForce(charge=q)` applique la force `(q/m) rho E` sur la quantite
  de mouvement (et le travail `(q/m) rho v . E` sur l'energie pour les electrons, qui ont 4
  variables) ; la brique `elliptic = ChargeDensity(charge=q)` contribue `q n` au second membre
  du Poisson de systeme. C'est le MEME `q` des deux cotes : c'est ce qui rend le couplage
  physiquement coherent.

## 4. Methode numerique

Le choix numerique est fait **par bloc**, independamment du modele physique. Ici les deux blocs
partagent le MEME schema (minmod + Rusanov + explicite), mais l'API permettrait de differer
(cf. la recette `two_fluid` de `recipes.py` qui met VanLeer+HLLC sur les electrons et
Minmod sur les ions).

- **Reconstruction spatiale** (`adc.Spatial(minmod=True)`) : limiteur MUSCL **minmod** sur les
  variables `conservative` (defaut), pour les DEUX blocs.
- **Flux numerique de Riemann** : `rusanov` (defaut de `adc.Spatial`), robuste, valable sur tout
  transport -- pas besoin de la pression de HLLC ici.
- **Traitement temporel** : `adc.Explicit()` (defaut SSPRK2, Shu-Osher 2 etages ordre 2),
  `substeps = 1`, `stride = 1`, pour les DEUX blocs. Pas de sous-pas ni de cadence (multirate)
  dans ce cas.
- **Poisson** : operateur `div(eps grad)` a `eps = 1` (Poisson), second membre `charge_density`
  (somme des briques elliptiques = `q_e n_e + q_i n_i`), solveur `geometric_mg` (multigrille
  geometrique), BC `auto` (periodique ici, herite de `System(periodic=True)`).
- **Avancee en temps** : `sim.advance(dt=0.001, nsteps=20)` -- boucle en temps **compilee** en
  C++ (pas d'integrateur Python ici, contrairement au cas `composition` partie D ou
  `custom_scheme`).

> Note de solvabilite : sur grille periodique, le Poisson `lap phi = f` n'est solvable que si
> `integrale(f) = 0` (condition de compatibilite). Ici `f = (-1) n_e + (+1) n_i` avec
> `n_e = 1 + 0.02 cos(...)` (moyenne exactement 1, le cosinus s'integre a 0 sur la periode) et
> `n_i = 1` : la charge nette moyenne est nulle, le Poisson periodique est bien pose.

## 5. Architecture ADC utilisee

```
Python (run.py)                          C++ compile (module adc / adc_cpp)
-----------------------------------      -------------------------------------------
adc.System(n=48, L=1.0, periodic)    ->  contexte maillage (carre periodique)
sim.add_block("electrons", model,    ->  fige une fermeture d'avancee compilee par bloc
              spatial, time)             (assemble_rhs<minmod, rusanov>, force PotentialForce)
sim.add_block("ions", model, ...)    ->  second bloc heterogene (3 var, isotherme)
sim.set_poisson(rhs="charge_density",->  EllipticPhysicalModel (Poisson) de SYSTEME :
                solver="geometric_mg")    f = somme_b (q_b n_b), multigrille geometrique
sim.set_density(name, array)         ->  ecrit la densite du bloc (reste au repos)
sim.mass(name) / sim.density(name)   ->  diagnostics par bloc (somme cellulaire de rho)
sim.advance(dt, nsteps)              ->  boucle en temps COMPILEE (transport + source + Poisson)
sim.time() / sim.nx()                ->  etat de l'horloge / taille de grille
```

Point cle : **aucun callback Python dans le hot path**. Chaque bloc embarque une fermeture
d'avancee compilee, type-erased seulement au niveau de la liste de blocs. Le residu
(`-div F + S`), la force electrostatique et le Poisson de systeme (somme des briques
elliptiques de chaque bloc) restent 100 % en C++, par cellule. Python ne fait que **composer**
(`add_block`), **poser** l'etat initial (`set_density`), **piloter** l'avancee (`advance`) et
**diagnostiquer** (`mass`, `density`, `time`).

Pour les details d'API (module `adc` d'`adc_cpp`, `python/adc/__init__.py`) : `Model`,
`FluidState`, `CompressibleFlux`, `IsothermalFlux`, `PotentialForce`, `ChargeDensity`,
`Spatial`, `Explicit`, `System.add_block`, `System.set_poisson`. Le `rhs="charge_density"` est
l'alias historique (bit-identique) du second membre composite generique
(`ChargeDensitySource` herite de `CompositeRhs`) quand tous les blocs portent une densite de
charge.

## 6. Carte des fichiers

| Fichier                                                                | Role |
|------------------------------------------------------------------------|------|
| [`multispecies/run.py`](run.py)                                        | le cas : compose 2 blocs + Poisson de systeme, pose la CI, avance 20 pas, `print` + `assert`. |
| [`adc_cases/models.py`](../adc_cases/models.py)                        | `electron_euler()`, `ion_isothermal()` (compositions de briques d'espece). |
| [`adc_cases/common/checks.py`](../adc_cases/common/checks.py)          | `assert_mass_conserved`, `assert_finite`, `assert_positive` (invariants partages). |
| [`adc_cases/__init__.py`](../adc_cases/__init__.py)                    | `ensure_importable()` (le script utilise un fallback `sys.path` direct equivalent). |
| [`cases_manifest.toml`](../cases_manifest.toml)                        | declare le cas : `validation`, `ci = true`, `needs = []`. |
| Module `adc` (hors depot)                                              | bindings pybind11 d'`adc_cpp` ; fourni par `PYTHONPATH` (voir Prerequis). |

Le cas n'utilise PAS `recipes.py` (il cable ses deux blocs a la main, plutot que d'appeler la
recette `two_fluid`), ni `initial_conditions.py`, ni `grid.py`, ni `io.py`, ni `native.py` : il
pose sa CI en numpy et fait ses diagnostics en ligne, en deleguant seulement les **asserts** a
`common.checks`.

## 7. Prerequis

- **Python 3.12** avec **numpy**.
- Le **module `adc`** (bindings d'`adc_cpp`) construit et disponible sur le `PYTHONPATH`. Sur
  cette machine, le build est dans
  `/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python`.
- Le **paquet `adc_cases`** importable. Soit installe (`pip install -e .` a la racine du depot,
  voie nominale CI), soit la racine du depot ajoutee au `PYTHONPATH` (le script tente
  `import adc_cases` et, a defaut, insere `os.path.dirname(os.path.dirname(__file__))` dans
  `sys.path`, cf. lignes 36-41 de `run.py`).
- **Aucun compilateur C++** requis (`needs = []`) : ce cas ne compile rien a la volee,
  contrairement aux cas `needs = ["cxx"]` (DSL / `two_fluid_ap`). Tout le calcul passe par des
  briques natives deja compilees dans le module `adc`.
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
cd /private/tmp/adc_cases-readmes/multispecies && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
/opt/homebrew/anaconda3/bin/python3.12 run.py
```

Forme generique (paquet `adc_cases` installe en editable, module `adc` sur le `PYTHONPATH`) :

```bash
export PYTHONPATH=<adc_cpp>/build-py/python
python3 multispecies/run.py
```

## 9. Explication du code par etapes

Tout vit dans la fonction `main()` de [`run.py`](run.py) :

1. **Systeme** (lignes 49-53) : `sim = adc.System(n=48, L=1.0, periodic=True)` -- config =
   maillage seul (grille 48x48, domaine carre `[0, 1]^2`, CL periodiques).
2. **Bloc electrons** (lignes 57-62) : `sim.add_block("electrons",
   model=models.electron_euler(charge=-1.0, gamma=5/3), spatial=adc.Spatial(minmod=True),
   time=adc.Explicit())` -- Euler compressible complet, charge `-1`, minmod, explicite.
3. **Bloc ions** (lignes 64-69) : `sim.add_block("ions",
   model=models.ion_isothermal(charge=+1.0, cs2=1.0), spatial=adc.Spatial(minmod=True),
   time=adc.Explicit())` -- Euler isotherme, charge `+1`, minmod, explicite.
4. **Poisson couple** (ligne 72) : `sim.set_poisson(rhs="charge_density",
   solver="geometric_mg")` -- UN Poisson de systeme, second membre = densite de charge agregee,
   multigrille geometrique.
5. **CI separation de charge** (lignes 78-85) : `x = (i + 0.5)/n` (centres de cellules) ;
   `ne = 1.0 + 0.02 cos(2 pi x)` ; `ne2d` broadcaste `ne` le long de `axis=1` (x varie le long
   des colonnes) ; `ni2d = ones((n, n))` (ions uniformes) ; `set_density("electrons", ne2d)` et
   `set_density("ions", ni2d)`.
6. **Etat initial mesure** (lignes 87-104) : `mass_e0`, `mass_i0` (masses initiales par espece) ;
   separation de charge `charge0 = (-1) n_e + (+1) n_i`, `qmax0 = max|charge0|` ; impression ;
   `assert qmax0 > 1e-6` (la separation de charge doit etre non nulle, sinon le Poisson couple
   est trivial et la demo ne demontre rien).
7. **Avancee** (lignes 107-109) : `dt = 0.001`, `nsteps = 20`, `sim.advance(dt, nsteps)` (boucle
   en temps compilee).
8. **Diagnostics finaux** (lignes 111-120) : `mass_e1`, `mass_i1`, separation de charge
   `qmax1`, impression de l'horloge `sim.time()`.
9. **Verification des invariants** (lignes 125-139), via `adc_cases.common.checks` :
   - `assert_mass_conserved(mass_e1, mass_e0, tol=1e-9, relative=False)` puis idem ions --
     conservation de masse PAR ESPECE en ABSOLU ; renvoie la derive `|dM|` ;
   - `assert_finite(de, ...)` et `assert_finite(di, ...)` -- pas de NaN/Inf ;
   - `assert_positive(de, ...)` et `assert_positive(di, ...)` -- densites strictement `> 0` ;
   - impression des intervalles `[n.min(), n.max()]` de chaque densite.
10. **Succes** (ligne 141) : `print("OK multispecies")`.

## 10. Conditions initiales

La CI est posee a la main en numpy (le cas n'utilise pas `common.initial_conditions`).
Convention : `x = (i + 0.5)/n * L` le long de l'**axe 1** (colonnes), broadcast sur l'axe 0.

| Espece    | Champ initial                          | Vitesse / energie |
|-----------|----------------------------------------|-------------------|
| electrons | `n_e = 1.0 + 0.02 cos(2 pi x)` (cosinus le long de x) | au repos (densite seule posee via `set_density`) |
| ions      | `n_i = 1.0` (uniforme)                 | au repos          |

L'idee : on perturbe les electrons par un petit cosinus le long de x et on laisse les ions
uniformes. Cela cree un **desequilibre local** entre `n_e` et `n_i`, donc un second membre
`f = (-1) n_e + (+1) n_i` non nul qui **pilote le Poisson couple**. L'amplitude mesuree de
cette separation de charge initiale est `max|f| = 1.995718e-02` (voir section 12).

`set_density(name, array)` pose UNIQUEMENT la densite (le fluide reste au repos : quantite de
mouvement nulle, energie a la valeur de fermeture du modele). C'est pourquoi la dynamique
ulterieure (faible : 20 pas seulement) reste de petite amplitude.

## 11. Invariants et assertions

Toutes les verifications sont des `assert` (en ligne dans `run.py`, ou delegues a
`adc_cases.common.checks`). Valeurs **reelles** capturees lors de l'execution du 2026-06-07 sur
cette machine (macOS arm64, build `build-master`, Python 3.12 d'anaconda) :

| Assert | Condition | Source | Valeur mesuree |
|--------|-----------|--------|----------------|
| separation de charge initiale | `max|f| > 1e-6` (`run.py` l.104) | `run.py` | `1.995718e-02` |
| masse electrons (ABSOLU) | `|m_e - m_e0| < 1e-9` (`relative=False`) | `assert_mass_conserved` | `|dM_e| = 5.912e-12` |
| masse ions (ABSOLU) | `|m_i - m_i0| < 1e-9` (`relative=False`) | `assert_mass_conserved` | `|dM_i| = 6.821e-12` |
| densite electrons finie | pas de NaN/Inf | `assert_finite` | OK |
| densite ions finie | pas de NaN/Inf | `assert_finite` | OK |
| densite electrons positive | `min(n_e) > 0` | `assert_positive` | `min = 0.980255` |
| densite ions positive | `min(n_i) > 0` | `assert_positive` | `min = 0.999996` |

Le **coeur de la demo** est la conservation de masse PAR ESPECE : `|dM_e| = 5.912e-12` et
`|dM_i| = 6.821e-12` sont tous deux **trois ordres de grandeur sous** le seuil `1e-9`. C'est la
preuve numerique que les deux especes, bien que couplees par le MEME champ electrique via un
SEUL Poisson, conservent **independamment** leur masse (les bilans de masse sont decouples).

Detail sur `mass` : `sim.mass(name)` renvoie la **somme cellulaire** de la densite du bloc
(`sum(rho)` sur les 48x48 cellules), pas l'integrale `sum(rho) * dx * dy`. Ainsi `mass_e0 =
2304.0` exactement (= `48 * 48`, car `n_e` a une moyenne de exactement 1, le cosinus s'integrant
a 0 sur la periode), et `mass_i0 = 2304.0` (ions uniformes a 1). La comparaison ABSOLUE
`|m - m0| < 1e-9` est donc faite sur cette echelle (~2304), ce qui rend la tolerance d'autant
plus exigeante en relatif (~3e-15).

## 12. Sorties attendues

Aucun fichier produit : tout est imprime. Sortie complete **reelle** (execution du 2026-06-07
sur cette machine) :

```
[init] grille nx = 48 x 48
[init] masse electrons mass_e0 = 2.304000000000e+03
[init] masse ions      mass_i0 = 2.304000000000e+03
[init] separation de charge max|f| = 1.995718e-02
[t=0.0200] masse electrons mass_e = 2.304000000000e+03
[t=0.0200] masse ions      mass_i = 2.304000000000e+03
[t=0.0200] separation de charge max|f| = 1.974902e-02
[diag] derive masse electrons |dM_e| = 5.912e-12
[diag] derive masse ions      |dM_i| = 6.821e-12
[diag] n_e dans [0.980255, 1.019745]
[diag] n_i dans [0.999996, 1.000004]
OK multispecies
```

Lecture :

- `t = 0.0200` = `20 * dt` = `20 * 0.001`, l'horloge finale apres `advance`.
- La separation de charge passe de `1.995718e-02` (initiale) a `1.974902e-02` (apres 20 pas) :
  legere relaxation, le couplage est actif mais la dynamique reste de faible amplitude (20 pas
  courts).
- `n_e` s'est etale de `[0.980, 1.020]` (cosinus initial d'amplitude 0.02) ; `n_i` a developpe
  une **toute petite** modulation `[0.999996, 1.000004]` (~4e-6) : les ions, initialement
  uniformes, ont commence a repondre au champ electrique cree par la perturbation electronique.
  C'est la signature du couplage inter-especes par le Poisson.
- Les masses restent affichees a `2.304000000000e+03` aux 12 decimales : la derive (`~6e-12`)
  est invisible a cette precision d'affichage, d'ou les lignes `[diag]` qui la donnent
  explicitement.

Les valeurs en virgule flottante peuvent varier au dernier chiffre selon la plateforme et
l'ordre de reduction, mais les derives de masse restent tres en dessous du seuil `1e-9`.

## 13. Generation figures/GIF

**Aucune.** Ce cas ne produit ni figure ni GIF ni fichier de sortie. Il n'importe pas
`matplotlib` (`needs = []`), n'utilise pas `adc_cases.common.io` et n'ecrit rien dans `out/`.
Toute la verification passe par `print` (diagnostics lisibles) et `assert` (invariants). Pour
des figures d'instabilite, voir le cas [`../diocotron/`](../diocotron/) (categorie
`reproduction`, `needs = ["matplotlib"]`).

## 14. Backends reellement supportes

- **CPU mono-rang uniquement** (chemin `System.add_block` natif). C'est le chemin de composition
  de briques natives (`ModelSpec`), sans `.so` compile a la volee, sans MPI, sans AMR, sans GPU
  dans ce cas.
- Le **solveur Poisson** utilise est `geometric_mg` (multigrille geometrique), valable en
  periodique et en paroi. L'alternative `fft` (periodique, `n = 2^k`) n'est pas demandee ici (et
  `n = 48` n'est pas une puissance de 2, donc FFT serait inapplicable).
- **Limiteur** exerce : `minmod` (sur les deux blocs). **Flux** : `rusanov` (defaut). **Temporel**
  : `Explicit` (SSPRK2), `substeps = 1`, `stride = 1`. Pas de HLLC/Roe, pas de WENO5, pas d'IMEX,
  pas de multirate dans ce cas (l'API les permettrait -- cf. recette `two_fluid` /
  `plasma` de `recipes.py`).
- La machine de test est **macOS arm64 (Darwin)**, module compile avec
  `_adc.cpython-312-darwin.so`. La CI tourne sur **ubuntu-latest** (build Release du module,
  `ADC_USE_EIGEN=OFF`). Le cas n'a aucune dependance materielle : il s'execute partout ou le
  module `adc` se construit.

## 15. Cout approximatif

Mesure reelle (temps mur) sur cette machine (macOS arm64, build `build-master`, Python 3.12
d'anaconda), via `time` sur la commande de la section 8 :

```
0.66s user 0.26s system 536% cpu 0.172 total
```

- **Temps mur total : ~0.17 s** (dont ~0.66 s CPU cumule : le multigrille geometrique exploite
  plusieurs threads -> 536 % CPU). Negligeable, conforme a un cas `ci = true`.
- **Memoire** : quelques Mo (grille `48x48`, 4 composantes pour les electrons + 3 pour les ions,
  un seul champ de potentiel). Aucune allocation notable.
- **Pas de compilation** a la volee (`needs = []`), donc pas de surcout `c++` au premier
  lancement, contrairement aux cas DSL/`two_fluid_ap`. Le gros du temps mur est l'import du
  module `adc` (~12 Mo de `.so`), pas le calcul (20 pas sur 48x48).

## 16. Limites et differences avec les references

- **Cas validation, AUCUN claim physique.** La CI est un cosinus jouet (amplitude 0.02) ;
  l'integration est volontairement courte (20 pas, `t_final = 0.02`). Le cas ne mesure aucun
  taux de croissance, aucune quantite physique calibree. Il **verifie une propriete numerique**
  (conservation de masse par espece sous couplage Poisson), pas un resultat physique. Ce n'est
  PAS la reproduction d'un benchmark publie.
- **Ce n'est PAS un plasma calibre.** Le couplage est un Poisson electrostatique jouet
  (`eps = 1`, charges `+/-1`, pas d'echelle physique, pas de longueur de Debye, pas de frequence
  plasma reelle). La dynamique observee (ions modules a ~4e-6) est qualitative, pas quantitative.
- **Mono-rang, pas de GPU/MPI/AMR** sur ce cas (cf. section 14). Le couplage multi-especes sur
  hierarchie AMR existe ailleurs (`adc.AmrSystem`, cf. cas [`../diocotron_amr/`](../diocotron_amr/)),
  pas ici.
- **Pas de multirate ni de schema heterogene** : les deux blocs partagent minmod + Rusanov +
  explicite, `substeps = 1`. La capacite a differer le schema PAR bloc (VanLeer+HLLC sur les
  electrons, etc.) ou a faire avancer les blocs a des cadences differentes (`substeps`/`stride`)
  est demontree par d'autres cas (recette `two_fluid` / `plasma` de
  [`recipes.py`](../adc_cases/recipes.py), cas [`../composition/`](../composition/),
  [`../two_euler/`](../two_euler/), [`../plasma/`](../plasma/)).
- **Conservation de masse en ABSOLU.** Le seuil est `|m - m0| < 1e-9` sur une masse de l'ordre
  de 2304 (somme cellulaire, cf. section 11), soit ~3e-15 en relatif. C'est exigeant, et c'est
  intentionnel : la conservation de masse par transport conservatif (flux finis) doit etre au
  niveau du bruit machine, le couplage par le champ ne modifiant QUE la quantite de mouvement /
  l'energie, jamais la densite.
- **Equivalent DSL.** La meme physique ecrite ENTIEREMENT en formules (mini-DSL `adc.dsl.Model`,
  Poisson couple, equivalence au natif PAR ESPECE) vit dans
  [`../two_species_dsl/`](../two_species_dsl/) (`needs = ["cxx"]`, compile un `.so` a la volee).
  Le present cas, lui, compose des briques NATIVES deja compilees (`needs = []`).

## 17. Tests/CI associes

- **Manifeste** : [`cases_manifest.toml`](../cases_manifest.toml) declare ce cas
  `path = "multispecies/run.py"`, `category = "validation"`, `ci = true`, `needs = []`,
  `desc = "Electrons Euler + ions isothermes couples par un Poisson de systeme."`.
- **CI** : [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) clone `adc_cpp`, construit le
  module `_adc` (CMake, Release, `ADC_BUILD_PYTHON=ON`, `ADC_USE_EIGEN=OFF`), installe
  `adc_cases` en editable, puis lit le manifeste et lance **uniquement** les cas `ci = true` avec
  `python3 <path>`. Ce cas en fait partie : `python3 multispecies/run.py` doit sortir 0 et
  imprimer `OK multispecies`.
- **Mecanisme de test** : il n'y a pas de framework de test externe (pas de pytest). Le cas EST
  son propre test : chaque invariant est un `assert` (en ligne ou delegue a
  `adc_cases.common.checks`), et un `assert` qui echoue provoque une `AssertionError` -> code de
  sortie non nul -> CI rouge. Le test FORT est la conservation de masse par espece (`|dM| <
  1e-9`).
- Pour relancer en local exactement comme la CI : installer le paquet (`pip install -e .` a la
  racine), exporter `PYTHONPATH=<adc_cpp>/build-py/python`, puis `python3 multispecies/run.py`.
