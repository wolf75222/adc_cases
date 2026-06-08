# two_euler : deux gaz d'Euler independants, meme schema

Cas de **validation** (manifeste : `category = "validation"`, `ci = true`, `needs = []`).

Etape "deux Euler, meme code" de l'echelle de tests : on instancie DEUX blocs d'Euler
compressible (etiquetes `electrons` et `ions`) qui partagent EXACTEMENT le meme modele natif
et le meme schema spatial/temporel, et qui n'ont AUCUN couplage physique entre eux. Seules les
conditions initiales different (les electrons sont 100x moins denses, donc 10x plus "rapides").
Le but n'est pas une physique de plasma : c'est de montrer que l'API multi-blocs d'`adc` fait
tourner N especes Euler sans une seule ligne de code dediee par espece, et que l'integrateur
multirate (`step_adaptive`) sous-cycle automatiquement le bloc le plus rapide.

---

## 1. Objectif du cas

- Demontrer la composition multi-blocs d'`adc.System` : deux blocs Euler portes par le MEME
  modele (`adc_cases.models.euler`) et le MEME objet `adc.Spatial`, distingues uniquement par
  leur etat initial.
- Verifier que chaque bloc conserve sa masse independamment (schema conservatif sur domaine
  periodique).
- Verifier la positivite (`rho > 0`, `p > 0`) que la reconstruction PRIMITIVE doit preserver
  pour des detentes de pression.
- Verifier que le bloc le plus leger (`electrons`, vitesse du son ~10x plus grande) s'etend plus
  vite : son front de detente couvre plus de cellules que celui des `ions`.
- Exercer le multirate `step_adaptive(cfl)` : un macro-pas cale sur le bloc le plus lent, chaque
  bloc plus rapide etant sous-cycle automatiquement (cf. section 4).

Ce cas ne reproduit AUCUN resultat publie : c'est un test d'architecture et d'invariants, pas
une reproduction.

---

## 2. Equations

Chaque bloc resout les equations d'Euler compressibles 2D sous forme conservative :

```
d_t U + d_x F(U) + d_y G(U) = 0
U = (rho, rho u, rho v, E)
p = (gamma - 1) (E - 1/2 rho |u|^2),   |u|^2 = u^2 + v^2
```

avec `gamma = 1.4` (constante `GAMMA` du `run.py`). Les flux `F`, `G` sont les flux d'Euler
standard (densite de masse, de quantite de mouvement, et flux d'energie `(E + p) u`).

Les deux blocs partagent ces equations a l'identique. Ils ne sont **pas couples** : pas de
source (`NoSource`), et le second membre du Poisson est nul (charge `q = 0`, cf. section 3),
donc le potentiel resolu ne retroagit sur aucun bloc. Le terme source du modele est strictement
zero ; chaque bloc est une loi de conservation pure.

---

## 3. Modele physique

Le modele d'espece est `adc_cases.models.euler(GAMMA)`, soit la composition de briques natives
suivante (cf. `adc_cases/models.py`, fonction `euler`) :

```python
adc.Model(
    state=adc.FluidState(kind="compressible", gamma=gamma),  # (rho, rho u, rho v, E)
    transport=adc.CompressibleFlux(),                        # flux d'Euler compressible
    source=adc.NoSource(),                                   # aucune source
    elliptic=adc.ChargeDensity(charge=0.0),                  # f = q n = 0 -> Poisson trivial
)
```

Points cles, tels qu'ecrits dans le code :

- `source=adc.NoSource()` : le terme source est nul. Aucune force electrostatique, aucune
  gravite. Le bloc est une pure loi de conservation hyperbolique.
- `elliptic=adc.ChargeDensity(charge=0.0)` : la brique elliptique declare une densite de charge
  `f = q n` avec `q = 0`. Le second membre du Poisson de systeme (`f = somme_b q_b n_b`) est donc
  identiquement nul. `set_poisson()` est appele quand meme (section 5) parce que `step_adaptive`
  ouvre chaque macro-pas par `solve_fields()` ; avec `f = 0` le potentiel reste nul et n'agit sur
  personne. C'est la justification exacte du commentaire du `run.py` : "Poisson f=0 (charge
  nulle) : blocs independants, juste pour solve_fields".

Les DEUX blocs utilisent ce meme modele `euler(GAMMA)` (deux appels distincts a
`models.euler(GAMMA)`, un par bloc). Aucune asymetrie n'est portee par le modele : toute la
difference physique vient des conditions initiales (section 10).

---

## 4. Methode numerique

Schema spatial commun aux deux blocs (objet `adc.Spatial` unique, partage) :

```python
spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")
```

- **Reconstruction MUSCL** avec limiteur **van Leer** (`vanleer=True` -> `limiter="vanleer"`).
  2 cellules fantomes.
- **Flux de Riemann HLLC** (`flux="hllc"`), variante a contact restitue ; HLLC/Roe exigent un
  transport compressible (verifie cote facade), ce qui est le cas ici.
- **Variables reconstruites : primitives** (`recon="primitive"`). La reconstruction primitive est
  plus robuste pour Euler : elle aide a preserver la positivite de `rho` et `p` a travers une
  detente, ce que le cas verifie explicitement (section 11).

Traitement temporel : `time=adc.Explicit()` par bloc, soit SSPRK2 (Shu-Osher 2 etages, ordre 2),
`substeps=1`, `stride=1` par defaut.

**Avance multirate.** La boucle appelle `sim.step_adaptive(0.4)` (CFL = 0.4) 20 fois. La
semantique exacte de `step_adaptive`, lue dans
`adc_cpp/include/adc/runtime/system_stepper.hpp` (methode `step_adaptive`) :

1. `solve_fields()` resout le Poisson (ici `f = 0`, potentiel nul).
2. La vitesse d'onde max `w_b` est calculee par bloc ; `wmin` = la plus PETITE (bloc le plus
   lent). Le macro-pas est `macro_dt = cfl * h / wmin` (cale sur le bloc le plus lent).
3. Chaque bloc est sous-cycle `n_b = ceil(stride_b * w_b / wmin)` fois sur `macro_dt`. Le bloc
   le plus lent fait `n = 1` sous-pas ; un bloc deux fois plus rapide en fait 2, etc. Avec
   `c_electrons / c_ions ~ 10`, le bloc `electrons` est sous-cycle environ 10x par macro-pas
   au depart.
4. `apply_couplings(macro_dt)` (aucun couplage ici), puis `t += macro_dt`.

C'est cela que le cas appelle "le multirate sous-cycle automatiquement les electrons (plus
rapides)" : le sous-cyclage decoule du rapport des vitesses d'onde, sans configuration explicite.

---

## 5. Architecture ADC utilisee

Pile de la facade `adc` (build pybind11 d'adc_cpp), telle qu'employee par le `run.py` :

| Appel facade | Role |
|---|---|
| `adc.System(n=64, L=1.0, periodic=True)` | grille cartesienne 64x64, domaine `[0,1]^2` periodique |
| `adc.Spatial(vanleer=True, flux="hllc", recon="primitive")` | discretisation spatiale partagee |
| `adc.Explicit()` | politique temporelle (SSPRK2, substeps=1, stride=1) |
| `sim.add_block("electrons", model=..., spatial=..., time=...)` | ajoute le 1er bloc Euler |
| `sim.add_block("ions", model=..., spatial=..., time=...)` | ajoute le 2e bloc Euler (meme modele) |
| `sim.set_poisson()` | Poisson de systeme, `rhs="charge_density"` (defaut), ici `f = 0` |
| `sim.set_state(name, flat_list)` | initialise l'etat conservatif d'un bloc (liste plate) |
| `sim.mass(name)` | masse totale d'un bloc (diagnostic conservation) |
| `sim.step_adaptive(0.4)` | un macro-pas multirate (cf. section 4) |
| `sim.get_state(name)` | etat conservatif d'un bloc, tableau `(n_vars, ny, nx) = (4, 64, 64)` |

Cote `adc_cases`, le cas n'utilise QUE :

- `adc_cases.models.euler(GAMMA)` (composition de briques natives, cote application) ;
- `adc_cases.common.initial_conditions.euler_pressure_blob` (CI) et `euler_pressure`
  (diagnostic de pression) ;
- `adc_cases.common.checks.assert_finite` et `assert_mass_conserved` (invariants).

Il n'utilise PAS `adc_cases.recipes` (pas de recette systeme : la composition est ecrite a la main
dans le `run.py`), ni `adc_cases.common.native` (aucun C++ compile a la volee : `needs = []`).

Point d'architecture mis en avant : adc_cpp ne connait que des BRIQUES (FluidState,
CompressibleFlux, NoSource, ChargeDensity...). Le nom "electrons" / "ions" vit cote application ;
les deux blocs sont strictement le meme modele compose, instancie deux fois.

---

## 6. Carte des fichiers

| Chemin | Role |
|---|---|
| `two_euler/run.py` | le cas : construit le systeme, pose les CI, avance, verifie les asserts |
| `adc_cases/models.py` (`euler`) | modele d'espece Euler pur (compose de briques natives) |
| `adc_cases/common/initial_conditions.py` (`euler_pressure_blob`, `euler_pressure`) | CI bulle de pression + diagnostic de pression |
| `adc_cases/common/checks.py` (`assert_finite`, `assert_mass_conserved`) | invariants (finitude, conservation de la masse) |
| `adc_cases/common/grid.py` (`meshgrid_xy`) | grille a centres de cellules (utilisee par la CI) |
| `adc_cases/__init__.py` | `ensure_importable()` (chemin d'import du paquet hors installation) |
| `cases_manifest.toml` | declare ce cas : `validation`, `ci = true`, `needs = []` |
| module `adc` (build adc_cpp) | facade pybind11 : `System`, `Spatial`, `Explicit`, `Model`, briques |

Le `run.py` ne lit/n'ecrit AUCUN fichier (pas de figure, pas de sortie disque) : tout passe par
des `print` et des `assert`.

---

## 7. Prerequis

- Python 3.12 (interpreteur utilise ici : `/opt/homebrew/anaconda3/bin/python3.12`).
- `numpy` (seule dependance Python du cas).
- Le module `adc` (bindings pybind11 d'adc_cpp) construit et accessible via `PYTHONPATH`.
  Build de reference utilise ici :
  `/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python`.
- Le paquet `adc_cases` accessible a l'import : soit installe (`pip install -e .`, voie CI),
  soit la racine du depot sur `PYTHONPATH` (voie utilisee ci-dessous). Le `run.py` retombe aussi
  sur un `sys.path.insert` si `import adc_cases` echoue.
- **Aucun compilateur C++ requis a l'execution** (`needs = []`) : le modele est compose de
  briques natives DEJA compilees dans le module `adc`. Rien n'est compile a la volee.

---

## 8. Commande exacte

```bash
cd /private/tmp/adc_cases-readmes/two_euler && \
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
/opt/homebrew/anaconda3/bin/python3.12 run.py
```

Le premier element du `PYTHONPATH` fournit le module `adc` (bindings), le second fournit le
paquet `adc_cases`. Sur une machine ou `adc_cases` est installe en editable et `adc` est deja
sur le chemin, `python3 run.py` suffit.

---

## 9. Explication du code par etapes

Lecture lineaire de `run.py` (fonction `main`) :

1. `n, L = 64, 1.0` ; `sim = adc.System(n=n, L=L, periodic=True)` : grille 64x64, domaine unite
   periodique.
2. `spatial = adc.Spatial(vanleer=True, flux="hllc", recon="primitive")` : un SEUL objet de
   schema, partage par les deux blocs.
3. `sim.add_block("electrons", model=models.euler(GAMMA), spatial=spatial, time=adc.Explicit())`
   puis le bloc `"ions"` a l'identique : deux blocs Euler, meme modele, meme schema, meme
   politique temporelle. C'est le coeur du message "2 Euler, meme code".
4. `sim.set_poisson()` : configure le Poisson de systeme (defaut `rhs="charge_density"`). Comme
   les deux blocs ont `q = 0`, le second membre est nul ; l'appel sert uniquement a ce que
   `step_adaptive -> solve_fields()` ait un solveur valide.
5. Construction des etats initiaux via la fonction locale `blob` (qui delegue a
   `euler_pressure_blob`) : `Ue0` (electrons, `rho0=0.01`) et `Ui0` (ions, `rho0=1.0`),
   memes `p0=1.0` et `dp=0.5` (section 10).
6. `sim.set_state("electrons", Ue0.reshape(-1).tolist())` et idem pour `ions` : l'etat
   conservatif `(4, n, n)` est aplati en liste plate et pose dans chaque bloc.
7. `me0, mi0 = sim.mass("electrons"), sim.mass("ions")` : masses initiales de reference.
8. Deux `print` d'entete, dont `c_electrons/c_ions ~ sqrt(1/0.01) = 10.0` (rapport des vitesses
   du son a pression egale, qui scale en `1/sqrt(rho)`).
9. Boucle `for _ in range(20): sim.step_adaptive(0.4)` : 20 macro-pas multirate a CFL 0.4.
10. Relecture des etats : `Ue = np.array(sim.get_state("electrons")).reshape(4, n, n)`, idem
    `Ui`. (`get_state` renvoie deja `(4, n, n)` ; le `reshape` est inoffensif.)
11. Verifications de masse : `assert_mass_conserved(sim.mass(...), m0, tol=1e-9, label=...)`
    par bloc ; renvoie la derive relative.
12. Diagnostics de pression `pe, pi = pressure(Ue), pressure(Ui)` et fraction de cellules
    perturbees `fe, fi = disturbed(...)` (fraction de cellules ou `|p - p0| > 0.02`).
13. `print` des trois diagnostics (masse, positivite, front), puis les `assert` (section 11),
    et enfin `print("OK two_euler")`.

Fonctions locales utilitaires :

- `blob(n, L, rho0, p0, dp)` -> `euler_pressure_blob(...)` : gaz au repos avec surpression
  gaussienne centrale.
- `pressure(U)` -> `euler_pressure(U, gamma=GAMMA)` : `p = (gamma-1)(E - 1/2 rho|u|^2)`.
- `disturbed(U, U0, thr)` : `mean(|p(U) - p(U0)| > thr)`, l'etendue (fraction de cellules) du
  front de detente.

---

## 10. Conditions initiales

Generees par `euler_pressure_blob` (cf. `adc_cases/common/initial_conditions.py`) :

```
U = (rho, rho u, rho v, E),   u = v = 0  (gaz au repos)
rho(x, y) = rho0                       (densite uniforme)
p(x, y)   = p0 + dp * exp(-r^2 / (sigma2 * L^2)),   sigma2 = 0.02 (defaut)
r^2       = (x - L/2)^2 + (y - L/2)^2
E         = p / (gamma - 1)            (energie interne pure, pas d'energie cinetique)
```

Parametres par bloc (depuis `run.py`) :

| Bloc | rho0 | p0 | dp | c ~ sqrt(gamma p / rho) au repos |
|---|---|---|---|---|
| electrons | 0.01 | 1.0 | 0.5 | ~10x plus grande (rho 100x plus faible) |
| ions | 1.0 | 1.0 | 0.5 | reference |

Les deux gaz partent au repos avec une bulle de surpression gaussienne au centre, identique en
forme. La seule difference est la densite : `rho0 = 0.01` pour les electrons contre `1.0` pour
les ions. A pression egale, la vitesse du son scale en `1/sqrt(rho)`, d'ou le rapport
`c_e / c_i = sqrt(1/0.01) = 10` affiche par le cas. Les electrons, plus "raides", developpent une
detente radiale plus rapide et plus etendue sur la duree fixe de la simulation.

`sigma2` n'est PAS passe par le `run.py` : il prend sa valeur par defaut `0.02`.

---

## 11. Invariants et assertions

Asserts presents dans `run.py` (un echec sort en `AssertionError` et rend la CI rouge) :

1. **Conservation de la masse par bloc** :
   `assert_mass_conserved(sim.mass("electrons"), me0, tol=1e-9, label="electrons")` et idem
   `ions`. Derive RELATIVE < 1e-9. Justifie par schema conservatif sur domaine periodique.
   *Mesure (ce run)* : electrons `drel = 1.02e-14`, ions `drel = 2.22e-16` (au niveau du bruit
   machine, tres en dessous de la tolerance).
2. **Positivite** : `assert Ue[0].min() > 0 and Ui[0].min() > 0` (densite) et
   `assert pe.min() > 0 and pi.min() > 0` (pression). La reconstruction primitive doit
   maintenir `rho > 0` et `p > 0` a travers la detente.
   *Mesure (ce run)* : `rho_min` e=`7.707e-03`, i=`8.033e-01` ; `p_min` e=`9.324e-01`,
   i=`1.000e+00`. Tous strictement positifs.
3. **Front electrons plus etendu que ions** : `assert fe > fi`, avec `fe`, `fi` la fraction de
   cellules ou la pression a change de plus de `0.02`. Verifie que le bloc plus leger s'etend
   plus vite.
   *Mesure (ce run)* : `fe = 0.861`, `fi = 0.287` (les electrons perturbent ~86% des cellules,
   les ions ~29%).
4. **Finitude** : `assert_finite(Ue, "etat electrons")` et `assert_finite(Ui, "etat ions")`
   (ni NaN ni Inf).

Le multirate (sous-cyclage automatique des electrons) n'a PAS d'assert dedie : il est exerce par
la boucle `step_adaptive` et indirectement valide par la stabilite (pas d'explosion, positivite
maintenue) et par le front electrons > ions.

---

## 12. Sorties attendues

Le cas ecrit uniquement sur la sortie standard. Sortie REELLE capturee (run du 2026-06-07,
deterministe et reproduite a l'identique sur 3 executions consecutives) :

```
== two_euler : deux Euler independants (meme schema HLLC + recon primitive) ==
  c_electrons/c_ions ~ 10.0 (electrons 100x plus legers)
  masse      : electrons drel=1.02e-14  ions drel=2.22e-16
  positivite : rho_min e=7.707e-03 i=8.033e-01 ; p_min e=9.324e-01 i=1.000e+00
  front (frac cellules perturbees) : electrons=0.861 ions=0.287
OK two_euler
```

La ligne finale `OK two_euler` n'est imprimee que si TOUS les asserts passent. Code de retour 0.

Aucun fichier produit (pas de figure, pas de gif, pas de sortie disque).

---

## 13. Generation figures/GIF

**Aucune.** Ce cas ne produit ni figure ni GIF : il n'importe pas `matplotlib` (manifeste :
`needs = []`), n'utilise pas `adc_cases.common.io`, et n'ecrit aucun fichier. Toute la sortie est
textuelle (section 12). Pour de la visualisation de detente Euler avec figures, voir d'autres cas
du depot (p.ex. `diocotron`, qui declare `needs = ["matplotlib"]`).

---

## 14. Backends reellement supportes

- **CPU natif uniquement** (chemin par defaut du module `adc`). Le cas utilise `add_block` avec
  un modele compose de briques NATIVES (`adc.Model(...)`), donc le chemin natif compose, pas un
  backend `.so` (DSL prototype/aot/production).
- **Mono-rang** (pas de MPI exerce par ce cas ; `adc.System` sur une seule grille).
- **Pas de GPU exerce** par ce cas (build de reference CPU). Le device n'est ni configure ni
  teste ici.
- HLLC est disponible parce que le transport est compressible (exigence verifiee cote facade).
  La reconstruction primitive et le limiteur van Leer sont des chemins natifs standard.

En clair : ce cas est un test d'architecture CPU natif ; il ne valide aucun backend compile,
MPI, ni GPU.

---

## 15. Cout approximatif

Mesure (wall time, `/usr/bin/time -p`, machine darwin arm64, Python 3.12, build
`build-master`) sur 3 executions :

```
real 0.47   real 0.44   real 0.46
```

Soit **~0.45 s** de temps mur de bout en bout (import du module `adc` + numpy compris). Le calcul
lui-meme (grille 64x64, 2 blocs, 20 macro-pas multirate) est negligeable devant le temps
d'import. Cas tres leger, adapte a la CI (`ci = true`).

---

## 16. Limites et differences avec les references

- **Pas une physique de plasma, pas une reproduction.** Les blocs s'appellent "electrons" et
  "ions" par analogie pedagogique, mais ils ne sont PAS couples : pas de force electrostatique
  (source nulle), Poisson a second membre nul (`q = 0`). Aucun champ ne relie les deux gaz. Un
  vrai modele electrons + ions couples par Poisson est un AUTRE cas (`multispecies`, `plasma`,
  `two_fluid` via `adc_cases.recipes`), pas celui-ci.
- **Aucun resultat publie vise.** `category = "validation"` : on verifie des invariants
  (conservation, positivite, ordre des fronts), pas une figure ou un taux de croissance d'une
  reference.
- **Asymetrie purement par les CI.** Le seul ingredient qui distingue les deux blocs est la
  densite initiale (`rho0 = 0.01` vs `1.0`). Le facteur ~10 sur la vitesse du son et le
  sous-cyclage multirate en decoulent ; ce n'est pas une difference de modele.
- **Domaine periodique fini** : la detente n'est suivie que sur 20 macro-pas. Sur une duree plus
  longue, le front electronique (~86% des cellules au pas final) finirait par interagir avec ses
  images periodiques ; le cas s'arrete avant pour rester un test propre d'invariants.
- **CPU mono-rang** : aucune validation device/MPI/AMR ici (cf. section 14).

---

## 17. Tests/CI associes

- Declaration manifeste (`cases_manifest.toml`) :
  ```toml
  [[case]]
  path = "two_euler/run.py"
  category = "validation"
  ci = true
  needs = []
  desc = "Deux gaz d'Euler independants, meme schema (HLLC + recon primitive), multirate."
  ```
- **Lance en CI** (`.github/workflows/ci.yml`, job "cases (legers)") : la CI clone adc_cpp,
  construit le module `_adc`, installe `adc_cases` en editable, lit le manifeste et lance tous
  les cas `ci = true` via `python3 two_euler/run.py`. Un assert qui echoue -> CI rouge. Aucun
  prerequis `needs` particulier (pas de `matplotlib`, pas de `cxx`).
- Les invariants verifies tiennent lieu de test : conservation de la masse par bloc (tol 1e-9),
  positivite `rho > 0` / `p > 0`, ordre des fronts `fe > fi`, finitude. Pas de fichier de
  reference ni de table de validation a comparer (ce n'est pas une reproduction).
