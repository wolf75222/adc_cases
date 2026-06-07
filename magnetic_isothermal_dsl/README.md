# Cas `magnetic_isothermal_dsl` : fluide isotherme magnetise ecrit en formules

Troisieme demonstrateur du plan declaratif ADC, apres
[`diocotron_dsl`](../diocotron_dsl/run.py) (mono-espece) et
[`two_species_dsl`](../two_species_dsl/run.py) (multi-espece). Tout le modele physique est ecrit
en expressions symboliques `adc.dsl.Model` ; `adc.dsl` genere le C++, le compile et l'installe
comme bloc via `add_equation(...)`. Script : [`run.py`](run.py).

Categorie manifeste : **validation**, `ci = true`, `needs = ["cxx"]` (compilateur C++20 requis).
Aucune brique nommee, aucun modele natif de reference n'existe pour ce modele : sa correction est
prouvee par equivalence inter-backend (quand les deux backends se lient) et par un oracle de
Lorentz analytique.

---

## 1. Objectif du cas

Exercer ce que les deux premiers demonstrateurs DSL ne couvraient pas : une **source qui lit un
champ auxiliaire ETENDU** (`B_z`, indice 3 du canal `adc::Aux`, au dela du contrat de base
`phi` / `grad phi`). Le modele est un fluide d'Euler **isotherme** (fermeture `p = cs2 rho`)
soumis a :

- une force **electrostatique** `q rho E` (avec `E = -grad phi`), couplee au Poisson de systeme ;
- une force de **Lorentz** `(q rho / c) v x B` projetee en 2D avec `B = B_z e_z`.

Le terme de Lorentz fait TOURNER la quantite de mouvement sans changer la masse ni l'energie
cinetique : c'est la nouveaute demontree ici. `B_z` est pilote a 100 % depuis Python
(`sim.set_magnetic_field`), sans aucune modification du coeur `adc_cpp`.

Le cas valide aussi le chemin DSL -> C++ : compilation `aot` (et tentative `production`), liaison
au `System`, resolution des champs auxiliaires, evaluation du residu et avancement temporel.

## 2. Equations

Variables conservatives : `rho`, `mx = rho u`, `my = rho v`. Fermeture isotherme `p = cs2 rho`,
vitesse du son `cs = sqrt(cs2)` constante.

Flux physiques (convention `IsothermalFlux` du coeur, pas de composante energie) :

```
flux_x = [ mx,  mx u + cs2 rho,  mx v        ]
flux_y = [ my,  my u,           my v + cs2 rho ]
```

Spectre caracteristique (utilise par Rusanov pour la vitesse d'onde) :

```
lambda_x = (u - cs, u, u + cs)
lambda_y = (v - cs, v, v + cs)
```

Source (electrostatique + Lorentz) :

```
S = [ 0,
      q rho (-grad_x) + B_z my,     <- qte de mvt x : electrostatique + Lorentz
      q rho (-grad_y) - B_z mx ]    <- qte de mvt y : electrostatique + Lorentz
```

Le couple `(+B_z my, -B_z mx)` est la projection 2D de `(q rho / c) v x B_z e_z` (constantes
physiques absorbees dans `B_z`). Il ne touche pas la composante densite (`S[0] = 0`).

Second membre elliptique (densite de charge, couple au Poisson du systeme) :

```
rhs_Poisson = q rho        (n = rho ; couplage charge_density)
```

Parametres du run : `cs2 = 1.0`, `q = -1.0` (signe inclus, comme la brique `PotentialForce` du
coeur), `B0 = 2.0` (champ de fond constant ; `!= 0` -> Lorentz actif).

## 3. Modele physique

Plasma fluide isotherme magnetise, electrostatique. La densite est positive partout, la masse est
conservee (source de masse nulle). Le potentiel `phi` provient d'un Poisson sur la densite de
charge `q rho` (regime repulsif, `q < 0` avec la convention du coeur). La force de Lorentz
gyromagnetique avec `B_z` constant uniforme fait precessioner la quantite de mouvement : une
composante transverse `my` apparait alors meme si elle est nulle a l'instant initial.

Le champ `B_z` est ici **constant et uniforme** (`B0` partout). Il n'est pas evolue : c'est un
champ de fond fixe peuple cote Python. La generalisation a un `B_z(x)` variable se ferait par un
autre tableau passe a `set_magnetic_field`, sans changer le modele DSL.

## 4. Methode numerique

- **Discretisation spatiale** : volumes finis, reconstruction MUSCL limiteur **minmod**, flux de
  Riemann **Rusanov** (`adc.FiniteVolume(limiter="minmod", riemann="rusanov")`, variables
  conservatives). Rusanov n'exige pas de pression (pas de primitive `p` requise pour le flux), ce
  qui convient au modele isotherme.
- **Integration temporelle** : explicite **SSPRK2** (Shu-Osher 2 etages, ordre 2),
  `adc.Explicit()` par defaut (`method="ssprk2"`, `kind="explicit"`). Pas de temps pilote par
  `step_cfl(0.4)`.
- **Poisson** : `set_poisson(rhs="charge_density", solver="geometric_mg")`, conditions aux limites
  periodiques (le `System` est `periodic=True`). Resolu avant l'evaluation du residu via
  `solve_fields()` puis a chaque pas.
- **Source** : appliquee dans le residu du bloc (chemin host-marshale du backend `aot`), pas par
  un etage de splitting Schur (ce cas reste explicite ; le splitting condense est l'objet de
  `schur_magnetized_cartesian`).

## 5. Architecture ADC utilisee

Chaine declarative `adc.dsl.Model` -> codegen C++ -> `.so` -> bloc du `System` :

- **`adc.dsl.Model`** decrit le modele en formules : `conservative_vars`, `primitive`, `aux`,
  `param`, `flux`, `eigenvalues`, `source`, `primitive_vars`, `conservative_from`,
  `elliptic_rhs`, `check`. Voir [`adc/dsl.py`](../../adc_cpp/build-master/python/adc/dsl.py).
- **`m.compile(so_path, include, backend=...)`** genere le C++ complet (modele en
  `CompositeModel<...>`) et compile une `.so`. Le cas tente deux backends :
  - `"production"` : loader natif zero-copie (`compile_native`, std C++23) ; cible
    `add_native_block`. Sur macOS arm64, le `.so` compile mais le `dlopen` echoue (espace de noms
    a deux niveaux) -> ce backend est **saute** ici.
  - `"aot"` : chemin de production host-marshale (`compile_aot`, std C++20), numerique identique
    au natif ; cible `add_compiled_block`. **Seul backend lie sur macOS.**
- **`adc.System(n, L, periodic=True)`** : grille cartesienne periodique.
- **`sim.add_equation("plasma", model=compiled, spatial=..., time=...)`** aiguille sur le backend
  du `CompiledModel` (`add_compiled_block` pour `aot`, `add_native_block` pour `production`). Le
  `.so` transporte noms, roles physiques (`Density`/`MomentumX`/`MomentumY`) et `n_aux = 4`.
- **`adc::Aux` etendu** : le modele lit `aux("B_z")` (indice 3) -> la brique generee declare
  `n_aux = 4` ; `add_equation` elargit le canal aux partage ; `sim.set_magnetic_field(tableau)`
  peuple `B_z`. **`set_magnetic_field` existe deja dans le binding** : aucune modification du coeur.
- **Diagnostics** : `eval_rhs("plasma")` (residu sans avancer), `step_cfl(0.4)`,
  `get_state("plasma")`, `mass("plasma")`, `time()` (delegues a la facade C++ via `__getattr__`).

## 6. Carte des fichiers

| Chemin | Role |
| --- | --- |
| [`run.py`](run.py) | Le cas complet : modele DSL, IC, build des backends, validations, run. |
| [`README.md`](README.md) | Ce document. |
| `../adc_cases/common/checks.py` | `assert_finite`, `assert_positive`, `relative_drift`. |
| `../adc_cases/common/io.py` | `case_output_dir` -> repertoire de sortie `out/<cas>/`. |
| `../adc_cases/common/native.py` | `adc_include()` -> dossier `include/` du coeur adc_cpp. |
| `../adc_cases/__init__.py` | `REPO_ROOT`, `ensure_importable()` (chemin d'import du paquet). |
| `../cases_manifest.toml` | Manifeste : categorie `validation`, `ci = true`, `needs = ["cxx"]`. |
| `out/magnetic_isothermal_dsl/*.so` | Artefacts produits a l'execution (voir section 12). |

Le module C++ `adc` (bindings pybind11 d'adc_cpp) est fourni par le `PYTHONPATH` du build ; le
paquet `adc_cases` n'est qu'une couche utilitaire Python.

## 7. Prerequis

- Le module `adc` (bindings adc_cpp) accessible via `PYTHONPATH` (ex.
  `<adc_cpp>/build-master/python`).
- Le paquet `adc_cases` importable (installe via `pip install -e .`, ou son depot sur le
  `PYTHONPATH` ; `run.py` ajoute automatiquement la racine du depot si l'import echoue).
- `numpy`.
- **Un compilateur C++ (C++20)** : `needs = ["cxx"]`. `adc_include()` localise les en-tetes du
  coeur (`$ADC_INCLUDE`, sinon le paquet `adc` installe, sinon le depot voisin `../adc_cpp/include`)
  et exige `adc/mesh/multifab.hpp`.
- **macOS** : seul le backend `aot` se lie (le `production` compile mais ne se charge pas en
  arm64). En CI Ubuntu, les deux backends se lient.

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/magnetic_isothermal_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Forme generique (depuis la racine du depot, paquet installe ou sur le `PYTHONPATH`) :

```bash
PYTHONPATH=<adc_cpp>/build-master/python python magnetic_isothermal_dsl/run.py
```

## 9. Explication du code par etapes

1. **Construction du modele** (`magnetic_isothermal_model`) : declare les conservatives
   `rho/rho_u/rho_v` avec roles `Density/MomentumX/MomentumY` ; les primitives nommees `u = mx/rho`,
   `v = my/rho`, `p = cs2 rho` ; les champs aux `phi`, `grad_x`, `grad_y`, `B_z` ; les params
   nommes `cs2`, `charge` (inlines au codegen). Pose le flux isotherme, le spectre,
   la source (electrostatique + Lorentz), le layout primitif `(rho, u, v, p)`, l'inverse
   `conservative_from([rho, rho*u, rho*v])`, et `elliptic_rhs(q*rho)`. `m.check()` valide la
   coherence du modele.
2. **Etat initial** (`initial_state`) : densite perturbee en cosinus le long de `x`, quantite de
   mouvement **purement longitudinale** (`mx = 0.3 rho`, `my = 0`). Voir section 10.
3. **Construction du `System`** (`_build_sim`) : `System(n, L=1, periodic=True)` ;
   `add_equation("plasma", compiled, FiniteVolume(minmod, rusanov), Explicit())` ;
   `set_poisson("charge_density", "geometric_mg")` ; `set_state` ;
   `set_magnetic_field(bz * ones(n,n))` ; `solve_fields()` (peuple `phi`/`grad`/`B_z`).
4. **Liaison des backends** (`bind_backends`) : compile `production` puis `aot`, tente de lier
   chaque `.so`. Renvoie le dict `{backend: sim}` des backends effectivement lies (au moins `aot`).
   Une exception (compilation OU `dlopen`) est capturee et le backend est saute avec un message.
5. **`main`** : grille 32x32, 40 pas. Lie les backends avec `B_z = B0` (`bound`) puis avec
   `B_z = 0` (`bound0`, controle), et execute les quatre validations (section 11).

## 10. Conditions initiales

```python
x = (np.arange(n) + 0.5) / n
rho0 = 1.0 + 0.05 * np.cos(2*pi*x)[None, :] * np.ones((n, n))   # densite cosinus le long de x
mx0  = 0.3 * rho0          # vitesse longitudinale u = 0.3
my0  = np.zeros((n, n))    # quantite de mouvement transverse NULLE au depart
state0 = np.stack([rho0, mx0, my0], axis=0)
```

Le choix `my = 0` initial est essentiel a la validation (4) : toute composante transverse apparue
en cours de run ne peut venir que du terme de Lorentz. Le `System` est periodique (`L = 1`).

## 11. Invariants et assertions

Quatre validations, dont les sorties reelles (run macOS, backend `aot`) sont reportees en
section 12 :

1. **Parite inter-backend** : si `production` ET `aot` se lient, leurs `eval_rhs` et leurs etats
   apres quelques pas sont BIT-IDENTIQUES (`np.array_equal`, `dmax == 0`). Sur macOS, un seul
   backend (`aot`) se lie -> cette parite est **sautee** (message explicite) ; la correction reste
   prouvee par l'oracle (2).
2. **Oracle Lorentz** : pour chaque backend lie, `eval_rhs(B_z=B0) - eval_rhs(B_z=0)` vaut, sur la
   quantite de mouvement, EXACTEMENT `(B0 my, -B0 mx)` calcule en numpy. Assertions
   `err_x == 0.0 and err_y == 0.0` (egalite stricte), la composante densite `dR[0] == 0`
   (Lorentz ne touche pas la masse), et `max|dR| > 0` (le terme est bien lu, pas partout nul).
   *Resultat reel* : `err_x = 0.000e+00`, `err_y = 0.000e+00`, `max|dR| = 6.299e-01`.
3. **Evolution** (40 pas, `step_cfl(0.4)`) : `assert_finite(state)` (ni NaN ni Inf),
   `assert_positive(state[0])` (densite > 0), `relative_drift(mass, mass0) < 1e-9` (masse
   conservee). *Resultat reel* : derive de masse `2.887e-15`.
4. **Rotation de Lorentz** : `abs(my_mean) > 1e-6` apres le run (la composante transverse,
   initialement nulle, est devenue non nulle). *Resultat reel* : `my_mean` passe de
   `0.000e+00` a `-2.080e-01`.

## 12. Sorties attendues

Sortie console reelle (macOS arm64, Python 3.12, grille 32x32, 40 pas) :

```
=== magnetic_isothermal_dsl : fluide isotherme magnetise ecrit en formules ===
grille 32 x 32, 40 pas, CFL = 0.4 ; cs2 = 1.0, q = -1, B_z = 2.0
backend 'production' indisponible (RuntimeError), essai suivant
backend 'production' indisponible (RuntimeError), essai suivant
backends DSL lies : 'aot'
parite inter-backend SAUTEE (un seul backend lie sur cette plateforme : 'aot') ; correction prouvee par l'oracle analytique de Lorentz
oracle Lorentz ['aot'] : err_x = 0.000e+00, err_y = 0.000e+00, max|dR| = 6.299e-01
apres 40 pas (backend 'aot') : t = 0.382939, derive de masse = 2.887e-15
qte de mvt transverse moyenne : initiale 0.000e+00 -> finale -2.080e-01 (rotation de Lorentz)
OK magnetic_isothermal_dsl (Lorentz exerce, B_z = 2.0 pilote depuis Python, backends 'aot')
```

Note : le message `backend 'production' indisponible (RuntimeError)` apparait DEUX fois (une fois
pour `B_z = B0`, une fois pour `B_z = 0`), ce qui est attendu sur macOS (echec `dlopen` arm64).
En CI Ubuntu, on verrait `production` ET `aot` se lier, la parite inter-backend s'executerait
(`dmax = 0.000e+00`, bit-identique), et l'oracle Lorentz s'evaluerait sur les deux backends.

Artefacts produits dans `out/magnetic_isothermal_dsl/` :

```
magnetic_isothermal_aot.so          (~237 KiB)
magnetic_isothermal_production.so   (~244 KiB)   <- compile, mais non chargeable sur macOS
```

Les deux `.so` sont generees (le `production` est compile meme s'il ne se charge pas ensuite). Le
chemin `out/` est hors source et ignore par git.

## 13. Generation figures/GIF

**Aucune.** Ce cas est un test de validation numerique : il n'ouvre pas `matplotlib`, ne sauve
aucune figure ni GIF. Ses seuls fichiers produits sont les `.so` compilees (section 12). Pour la
reproduction avec figures, voir les cas `reproduction` du manifeste (`diocotron/run.py`,
`hoffart_euler_poisson_dsl/run.py`).

## 14. Backends reellement supportes

| Backend | Compilation | Liaison (`dlopen`) | macOS arm64 | CI Ubuntu |
| --- | --- | --- | --- | --- |
| `aot` (host-marshale) | OK | OK | **utilise** | utilise |
| `production` (natif zero-copie) | OK | echoue (arm64) | saute | utilise |

Le run macOS de reference ne lie que **`aot`** : le `.so` `production` est genere mais son
chargement leve une `RuntimeError` (espace de noms a deux niveaux d'arm64). La parite
inter-backend exige les deux backends ; elle n'est donc evaluee qu'en CI Ubuntu. Sur macOS, la
correction reste integralement prouvee par l'oracle Lorentz analytique (egalite stricte `dmax == 0`).

Le coeur de calcul est numeriquement identique entre les deux backends (la docstring du coeur
garantit `compile_aot` numerique identique au bloc natif) ; seule la frontiere d'integration
(host-marshale vs zero-copie) differe.

## 15. Cout approximatif

Mesure reelle (macOS arm64, Python 3.12 anaconda) : **~8.7 s de temps mur** pour le run complet
(`8.70s real`, ~7.6s user). Un run a cache chaud reste a **~8.0 s** : le cas passe `so_path=`
explicite a `m.compile(...)`, ce qui **force une recompilation a chaque appel** (la docstring de
`compile` precise que passer `so_path=` "compile toujours"). Le cas compile au total **quatre
`.so`** (`production` + `aot`, chacune pour `B_z = B0` et `B_z = 0`), d'ou un cout domine par la
compilation C++.

Le calcul lui-meme est negligeable : grille 32x32, 40 pas explicites + quelques resolutions
Poisson `geometric_mg`. La memoire est minime (quelques tableaux 32x32). Sur une plateforme ou
les deux backends se lient (CI Ubuntu), le cout monte legerement (deux backends a charger et a
diagnostiquer, plus la parite inter-backend a evaluer).

## 16. Limites et differences avec les references

- **Pas de modele natif de reference** : il n'existe aucune brique nommee ni modele natif
  equivalent pour ce modele magnetise. La correction repose sur (a) l'equivalence inter-backend
  DSL<->DSL (uniquement quand les deux backends se lient, donc PAS sur macOS) et (b) l'oracle
  analytique de Lorentz. Ce n'est pas une comparaison a un solveur tiers publie.
- **`B_z` constant et non evolue** : champ de fond uniforme `B0`, peuple cote Python. Pas de
  retroaction (le fluide ne modifie pas `B_z`), pas d'electromagnetisme complet (pas de loi de
  Faraday/Ampere). C'est un modele electrostatique + Lorentz cinematique, pas de la MHD.
- **Schema explicite, source non raide** : la source de Lorentz est integree explicitement, sans
  etage de splitting condense par Schur. Le traitement de la source raide magnetisee (et l'effet
  du Schur sur le pas stable) est l'objet du cas `schur_magnetized_cartesian` (experimental, hors
  CI) et du systeme Euler-Poisson complet `hoffart_euler_poisson_dsl` (reproduction-candidate
  PENDING, hors CI). **Ce cas-ci n'est PAS une reproduction d'un resultat publie** : c'est un
  demonstrateur de validation de la chaine DSL avec aux etendu.
- **Geometrie cartesienne periodique** : pas de bords physiques, pas de geometrie polaire. Run
  court (40 pas) : on demontre que la physique est exercee (rotation visible), pas un regime
  asymptotique.
- **macOS = `aot` only** : sur arm64, le backend `production` (chemin de production zero-copie) ne
  se charge pas ; la couverture inter-backend complete n'existe qu'en CI Ubuntu.

## 17. Tests/CI associes

- **Manifeste** (`cases_manifest.toml`) : categorie `validation`, `ci = true`, `needs = ["cxx"]`.
  La CI (`.github/workflows/ci.yml`) lance ce cas car il est leger et marque `ci = true`.
- **Le cas EST le test** : `run.py` est auto-validant. Toutes les verifications de la section 11
  sont des `assert` qui levent `AssertionError` en cas d'echec, faisant sortir le script en erreur
  (CI rouge). Pas de fichier de test separe : un exit code 0 et la ligne finale
  `OK magnetic_isothermal_dsl ...` signent le succes.
- **Dependance compilateur** : le marqueur `cxx` impose un C++20 dans l'environnement CI ; un
  defaut de toolchain ferait echouer la compilation des `.so` (et donc le cas). En CI Ubuntu, les
  deux backends se lient et la parite inter-backend est effectivement evaluee (etape sautee sur
  macOS).
