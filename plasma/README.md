# Cas `plasma` : trois especes (electrons + ions + neutres) couplees par Poisson, ionisation et collision

Cas de **validation** (manifeste : `category = "validation"`, `ci = true`, `needs = []`). Il exerce la
machinerie de couplage multi-especes de `adc.System` : trois blocs fluides partageant un Poisson de
systeme, une source d'ionisation (un neutre devient un ion + un electron) et une friction ion-neutre.
Le cas ne valide pas une physique de reference publiee : il verifie par `assert` que la composition se
cable, tourne, et respecte trois invariants (Poisson actif, transfert de masse neutre -> ion conserve,
densites finies et positives).

> **Honnetete (cf. `run.py` et le coeur C++).** L'ionisation ne transfere que la **densite** (composante 0
> de l'etat) du neutre vers l'ion ; le transfert de **quantite de mouvement et d'energie** des particules
> creees est une **simplification**. La friction ion-neutre **neglige l'echauffement** (la chaleur de
> friction n'est pas rendue a l'energie). Ces deux points sont des limites assumees du modele, pas des
> bugs (voir section 16).

---

## 1. Objectif du cas

Demontrer et verifier qu'on peut composer DEPUIS PYTHON un plasma a trois especes via une **recette
systeme** (`adc_cases.recipes.plasma`), sans ecrire une seule ligne de C++ par cas. Concretement le cas
verifie que la machinerie de couplage se compose et tourne, via trois invariants asseres :

1. **Poisson de systeme actif** : la separation de charge initiale (electrons modules, ions/neutres
   uniformes) produit un potentiel non nul (`|phi|_max > 1e-8`).
2. **Ionisation conservative en masse** : la masse totale `n_i + n_g` est conservee (transfert
   neutre -> ion), les neutres sont consommes et les ions augmentent.
3. **Integrite** : les densites des trois especes restent finies et strictement positives sur tout le run.

C'est un test de la **plomberie de couplage** (Poisson de systeme + sources inter-especes), pas une
reproduction quantitative d'un resultat de plasma publie.

---

## 2. Equations

Trois fluides 2D couples sur le carre periodique `[0, L]^2`. On note `q_e = -1`, `q_i = +1` (les neutres
sont de charge nulle).

**Electrons** (Euler compressible, `gamma = 5/3`), etat conservatif `U_e = (rho_e, rho_e u, rho_e v, E_e)` :

```
d_t rho_e        + div(rho_e v_e)              = S_e^ion
d_t (rho_e v_e)  + div(rho_e v_e (x) v_e + p_e I) = (q_e/m) rho_e E + S_e^col + ...
d_t E_e          + div((E_e + p_e) v_e)        = (q_e/m) rho_e v_e . E + ...
p_e = (gamma - 1)(E_e - 1/2 rho_e |v_e|^2)
```

**Ions** (Euler isotherme, vitesse du son `c_s^2 = cs2`), etat `U_i = (rho_i, rho_i u, rho_i v)` :

```
d_t rho_i       + div(rho_i v_i)              = S_i^ion
d_t (rho_i v_i) + div(rho_i v_i (x) v_i + c_s^2 rho_i I) = (q_i/m) rho_i E + S_i^col
```

**Neutres** (Euler isotherme, meme `cs2`, charge nulle, hors Poisson), etat `U_n = (rho_n, rho_n u, rho_n v)` :

```
d_t rho_n       + div(rho_n v_n)              = S_n^ion
d_t (rho_n v_n) + div(rho_n v_n (x) v_n + c_s^2 rho_n I) = S_n^col
```

**Couplage elliptique (Poisson de systeme)** : le champ `E = -grad phi` est self-consistant,

```
- lap phi = f = q_e n_e + q_i n_i           (les neutres, charge nulle, ne contribuent pas)
```

**Sources inter-especes** (operator-split, appliquees apres le transport) :

- **Ionisation** `n_g -> n_i + n_e`, taux `k_ion n_e n_g`. La masse transferee va du neutre vers l'ion
  (et vers l'electron). Implementee sur la **densite seulement** (composante 0).
- **Collision / friction ion-neutre**, force `k_col (u_a - u_b)` sur la quantite de mouvement, opposee sur
  chaque espece (quantite de mouvement totale conservee). Echauffement neglige.

---

## 3. Modele physique

Chaque espece est un **modele d'espece** nomme cote application (`adc_cases.models`), compose de briques
generiques de `adc` via `adc.Model(state, transport, source, elliptic)`. Pour ce cas :

| Espece     | Modele                       | Etat                          | Transport          | Source           | Brique elliptique        |
|------------|------------------------------|-------------------------------|--------------------|------------------|--------------------------|
| electrons  | `models.electron_euler`      | `FluidState(compressible, gamma=5/3)` | `CompressibleFlux` | `PotentialForce(q=-1)` | `ChargeDensity(q=-1)` |
| ions       | `models.ion_isothermal`      | `FluidState(isothermal, cs2=1)` | `IsothermalFlux`   | `PotentialForce(q=+1)` | `ChargeDensity(q=+1)` |
| neutrals   | `models.neutral_isothermal`  | `FluidState(isothermal, cs2=1)` | `IsothermalFlux`   | `NoSource`       | `ChargeDensity(q=0)`     |

Detail des briques (verifie dans `adc_cpp/build-master/python/adc/__init__.py`) :

- `PotentialForce(charge=q)` applique `(q/m) rho E` sur la quantite de mouvement (+ travail si 4 variables,
  donc les electrons recoivent le terme d'energie, pas les ions isothermes).
- `ChargeDensity(charge=q)` injecte `q n` dans le second membre du Poisson de systeme. La brique des
  neutres porte `charge=0.0` : presente, mais nulle, donc les neutres n'entrent pas dans `f`.
- Les neutres servent d'**espece de fond reactive** : pas de force electrostatique, mais ils alimentent
  l'ionisation et subissent la friction.

Les parametres effectifs de la recette `plasma` (cf. `recipes.py`) : `qe = -1.0`, `qi = +1.0`,
`gamma = 5/3`, `cs2 = 1.0`, `ionization_rate = 0.3`, `collision_rate = 0.5` (les deux derniers surcharges
par `run.py`, qui passe exactement ces memes valeurs).

---

## 4. Methode numerique

Discretisation **volumes finis** sur grille cartesienne `48 x 48`, integration **explicite** par pas CFL.

- **Electrons** : reconstruction limitee **van Leer** + flux de Riemann **HLLC** + reconstruction en
  variables **primitives** (`adc.Spatial(vanleer=True, flux="hllc", recon="primitive")`). Le primitif
  protege la positivite de `rho` et `p` pour Euler compressible (c'est le schema "Phase 1" du projet).
- **Ions** et **neutres** : limiteur **minmod**, flux par defaut **rusanov**, reconstruction conservative
  (`adc.Spatial(minmod=True)`).
- **Integration temporelle** : `Explicit()` par defaut (SSPRK2, Shu-Osher 2 etages ordre 2), `substeps=1`,
  `stride=1` pour les trois blocs.
- **Pas de temps** : `sim.step_cfl(0.3)` choisit un `dt` stable a CFL = 0.3 et avance d'un macro-pas. Le
  cas effectue **20 macro-pas**.
- **Poisson** : `sim.set_poisson()` configure le solveur elliptique de systeme par defaut
  (`rhs="charge_density"`, `solver="geometric_mg"` = multigrille geometrique, `bc="auto"` qui devient
  periodique ici, `epsilon=1`). Le champ `E` est resolu a chaque pas et lu par les `PotentialForce`.
- **Sources inter-especes** : appliquees en **operator-split** apres le transport hyperbolique
  (`add_ionization`, `add_collision`).

---

## 5. Architecture ADC utilisee

Le cas n'utilise que la **facade Python** `adc` (bindings pybind11 de `adc_cpp`) et le paquet partage
`adc_cases`. Aucune compilation a la volee (`needs = []`), contrairement aux cas DSL/AP.

Chaine de composition (du plus generique au plus specifique) :

```
adc (coeur C++ : briques ExB, CompressibleFlux, PotentialForce, ChargeDensity, ...)
  |
  v
adc_cases.models     : modeles d'ESPECE nommes (electron_euler, ion_isothermal, neutral_isothermal)
  |                     = adc.Model(state, transport, source, elliptic)
  v
adc_cases.recipes.plasma : recette SYSTEME = sim COMPLET (3 blocs + Poisson + 2 couplages + densites)
  |
  v
plasma/run.py        : pose les CI numpy, lance 20 pas, verifie les invariants
```

API systeme exercee (toutes deleguees a la facade compilee `adc._System` via `System.__getattr__`) :

- `adc.System(n, L, periodic)` : construit le coupleur cartesien (carre periodique).
- `sim.add_block(name, model, spatial=..., time=...)` : ajoute une espece.
- `sim.set_poisson()` : active le Poisson de systeme `-lap phi = sum_b q_b n_b`.
- `sim.add_ionization(electron, ion, neutral, rate)` : source d'ionisation (sur la densite).
- `sim.add_collision(a, b, rate)` : friction inter-especes (sur la quantite de mouvement).
- `sim.set_density(name, rho)` : pose la densite initiale d'un bloc (reste au repos).
- `sim.solve_fields()` / `sim.potential()` : resout et lit le potentiel (diagnostic initial).
- `sim.step_cfl(cfl)` : un macro-pas a CFL donne.
- `sim.mass(name)` / `sim.density(name)` : diagnostics (masse = **somme** des densites de cellule,
  pas l'integrale ponderee par l'aire ; cf. section 11).

---

## 6. Carte des fichiers

| Fichier                                                | Role                                                                 |
|--------------------------------------------------------|----------------------------------------------------------------------|
| `plasma/run.py`                                        | LE cas : pose les CI, cable la recette, lance 20 pas, asserte 4 invariants. |
| `adc_cases/recipes.py` (`plasma`)                      | Recette systeme : `add_block` x3 + `set_poisson` + `add_ionization` + `add_collision` + `set_density`. |
| `adc_cases/models.py`                                  | Modeles d'espece : `electron_euler`, `ion_isothermal`, `neutral_isothermal`. |
| `adc_cases/common/checks.py` (`relative_drift`)        | Ecart relatif protege contre zero, utilise pour l'invariant de masse. |
| `adc_cases/__init__.py`                                | `ensure_importable()` : place la racine du depot sur `sys.path` si le paquet n'est pas installe. |
| `cases_manifest.toml`                                  | Manifeste : ce cas est `category = "validation"`, `ci = true`, `needs = []`. |
| `.github/workflows/ci.yml`                             | CI : build du module `adc`, install editable, lance les cas `ci = true`. |

Le module `adc` lui-meme vient du build de `adc_cpp` (header-only + bindings), fourni par le PYTHONPATH ;
`adc_cases` ne le construit ni ne le localise. Le cas n'ecrit aucun fichier de sortie (pas de figure,
pas de gif, pas de `out/`).

---

## 7. Prerequis

- **Python 3.12** avec **numpy** (seule dependance Python directe du cas).
- Le module **`adc`** (bindings de `adc_cpp`) accessible sur le `PYTHONPATH`. Dans cet environnement il
  est pre-construit dans `adc_cpp/build-master/python/`.
- Le paquet **`adc_cases`** importable : soit installe (`pip install -e .` a la racine du depot, voie
  nominale CI), soit la racine du depot ajoutee au `PYTHONPATH` (ce que fait la commande ci-dessous), soit
  via le fallback `sys.path` integre au debut de `run.py`.
- **Aucun compilateur C++** requis pour CE cas (`needs = []` dans le manifeste) : il n'y a pas de
  compilation a la volee.

---

## 8. Commande exacte

Depuis le worktree, avec le module `adc` pre-construit et le depot sur le `PYTHONPATH` :

```bash
cd /private/tmp/adc_cases-readmes/plasma
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

En CI (`.github/workflows/ci.yml`), c'est equivalent a (apres build de `adc` et `pip install -e .`) :

```bash
PYTHONPATH=$GITHUB_WORKSPACE/adc_cpp/build-py/python python3 plasma/run.py
```

---

## 9. Explication du code par etapes

Lecture pas a pas de `plasma/run.py` (avec les lignes de `recipes.plasma` qu'il declenche) :

1. **Import et fallback d'import** (`run.py` l.20-32) : `import adc`, puis tentative `import adc_cases` ;
   en cas d'echec (paquet non installe), la racine du depot est inseree sur `sys.path`. On importe ensuite
   `recipes` et `relative_drift`.
2. **Grille et condition initiale** (l.38-40) : `n = 48`, `L = 1.0`. La densite electronique est une
   **faible separation de charge** `ne = 1.0 + 0.05 cos(2 pi x)`, broadcastee en `(48, 48)`. Ions et neutres
   sont uniformes a `1.0`.
3. **Construction du systeme** (l.42) : `adc.System(n=48, L=1.0, periodic=True)` -> coupleur cartesien
   periodique.
4. **Cablage de la recette** (l.43-44 -> `recipes.py` l.37-50) : `recipes.plasma(...)` execute, dans
   l'ordre :
   - `add_block("electrons", electron_euler(charge=-1, gamma=5/3), spatial=Spatial(vanleer, hllc, primitive))` ;
   - `add_block("ions", ion_isothermal(charge=+1, cs2=1), spatial=Spatial(minmod))` ;
   - `add_block("neutrals", neutral_isothermal(cs2=1), spatial=Spatial(minmod))` ;
   - `set_poisson()` (Poisson de systeme) ;
   - `add_ionization(electron="electrons", ion="ions", neutral="neutrals", rate=0.3)` ;
   - `add_collision("ions", "neutrals", rate=0.5)` ;
   - `set_density(...)` pour les trois especes.
5. **Champ initial** (l.46-47) : `solve_fields()` resout le Poisson, `potential()` rend phi, on prend
   `phi0 = max |phi|`.
6. **Masses de reference** (l.48) : `mi0 = mass("ions")`, `mg0 = mass("neutrals")`.
7. **Avance temporelle** (l.53-54) : **20** appels `step_cfl(0.3)`.
8. **Diagnostics finaux** (l.56-64) : masses finales `mi1`, `mg1` ; derive relative de `mi + mg` ; densites
   des trois especes ; test de finitude et positivite.
9. **Assertions** (l.66-70) : voir section 11. En cas de succes, impression de `OK plasma`.

---

## 10. Conditions initiales

Posees en **numpy** dans `run.py` (la physique du scenario vit cote application, jamais en C++ par cas) :

```python
n, L = 48, 1.0
x  = (np.arange(n) + 0.5) / n                                  # centres de cellules le long de x
ne = 1.0 + 0.05 * np.cos(2 * PI * x)[None, :] * np.ones((n, n))  # electrons modules le long de x
ni = np.ones((n, n))                                          # ions uniformes
ng = np.ones((n, n))                                          # neutres uniformes
```

- **Electrons** : densite `1.0 + 0.05 cos(2 pi x)`, soit une **faible separation de charge** (modulation
  5 %) constante le long de y. C'est elle qui rend le Poisson de systeme non trivial (`f = -n_e + n_i`
  n'est pas nul).
- **Ions** et **neutres** : uniformes a `1.0`.
- Toutes les especes demarrent **au repos** (vitesse nulle) : `set_density` ne pose que la densite, l'etat
  conservatif est complete au repos par le modele du bloc.

Convention de grille (cf. `adc_cases.common.grid`) : `field[j, i]`, centre de cellule
`x = (i + 0.5)/n L`. Ici la modulation ne depend que de l'indice colonne `i` (axe x).

---

## 11. Invariants et assertions

Quatre `assert` dans `run.py` (l.66-69). Les valeurs entre crochets sont celles **reellement mesurees**
lors de l'execution capturee (section 12), grille `48 x 48`, 20 pas, CFL 0.3.

1. **Poisson actif** : `assert phi0 > 1e-8`.
   Mesure : `|phi|_max = 1.266e-03` >> `1e-8`. La separation de charge produit bien un champ.

2. **Sens de l'ionisation** : `assert mg1 < mg0 - 1e-6 and mi1 > mi0 + 1e-6`.
   Mesure : neutres `2304.0000 -> 2237.3230` (consommes), ions `2304.0000 -> 2370.6770` (crees). On a bien
   `n_g` qui baisse et `n_i` qui monte.

3. **Conservation de masse de l'ionisation** : `assert drel < 1e-7` ou
   `drel = relative_drift(mi1 + mg1, mi0 + mg0)`.
   Mesure : `drel = 2.37e-15` (precision machine). La masse `n_i + n_g` est conservee : tout neutre ionise
   devient exactement un ion (transfert de la densite, comp 0).

4. **Integrite** : `assert finite_pos`, avec
   `finite_pos = all(isfinite(d).all() and d.min() > 0 for d in densites)`.
   Mesure : `min e = 9.862e-01`, `min i = 1.028e+00`, `min n = 9.698e-01` : toutes finies et `> 0`.

**Semantique de `mass(name)`** : la valeur 2304 = 48 x 48 montre que `mass()` rend la **somme** des
densites de cellule (et non l'integrale `sum * h^2`, qui vaudrait 1.0 ici). C'est sans incidence sur les
invariants, qui sont des rapports relatifs ou des comparaisons de signe.

> **Ce que ces invariants NE prouvent PAS.** Ils ne controlent pas l'energie totale (l'ionisation ne
> transfere pas l'energie cinetique/interne des particules creees ; la friction neglige l'echauffement),
> ni une croissance/decroissance physique de reference. Ce sont des controles de **plomberie** : le
> couplage se compose, transfere la masse correctement, et ne diverge pas.

---

## 12. Sorties attendues

Sortie texte **reellement capturee** (commande de la section 8, environnement local, build
`adc_cpp/build-master`) :

```
== plasma : electrons + ions + neutres (Poisson + ionisation + collision) ==
  |phi|_max = 1.266e-03  (Poisson de systeme actif)
  ionisation : n_i 2304.0000 -> 2370.6770,  n_g 2304.0000 -> 2237.3230,  (n_i+n_g) drel = 2.37e-15
  densites   : min e=9.862e-01 i=1.028e+00 n=9.698e-01 (toutes finies et positives : True)
OK plasma
```

Code de retour `0` (toutes les assertions passent). Ces nombres sont **deterministes** d'un run a l'autre
(re-execute trois fois, sortie identique) ; ils peuvent legerement varier sur une autre plateforme/un
autre build (ordre des operations flottantes), mais les invariants restent largement satisfaits.

---

## 13. Generation figures/GIF

**Aucune.** Ce cas est un test de validation textuel : il n'ecrit aucun fichier, ne trace aucune figure et
ne produit aucun gif (`needs = []`, pas de dependance `matplotlib`). Il imprime un resume sur la sortie
standard et sort en erreur si un invariant casse. Pour de la visualisation, voir les cas marques
`needs = ["matplotlib"]` (p.ex. `diocotron`).

---

## 14. Backends reellement supportes

- **CPU host, mono-rang** : c'est le chemin exerce par ce cas (grille cartesienne `48 x 48`, sans MPI,
  sans AMR, sans GPU). La commande ci-dessus tourne tel quel.
- **Solveur Poisson** : multigrille geometrique (`geometric_mg`, defaut de `set_poisson`), domaine
  periodique.
- **Pas de compilation a la volee** : `needs = []`. Le cas n'emprunte ni le chemin DSL (`adc.dsl`), ni le
  chemin natif compile (`add_native_block`), ni `adc_cases.common.native` (ctypes) ; il n'utilise que les
  briques **natives composees** (`adc.Model` -> `add_block`).
- Les couplages `add_ionization` / `add_collision` sont des **formules nommees figees** cote C++
  (operator-split), pas le chemin `CompiledCoupledSource` du DSL.
- La portabilite GPU/MPI/AMR de la facade existe dans `adc_cpp` mais **n'est pas exercee** par ce cas.

---

## 15. Cout approximatif

Mesure locale (Apple Silicon, `/usr/bin/time -p`, build `adc_cpp/build-master`, trois executions) :

| Mesure                | Valeur                              |
|-----------------------|-------------------------------------|
| Temps mur (`real`)    | **~0.20-0.23 s** (0.20, 0.21, 0.23) |
| Temps CPU user        | ~0.75 s (multi-thread d'import/build interne) |
| Temps CPU sys         | ~0.50 s                             |
| Memoire               | negligeable (grille `48 x 48`, 3 blocs) |

C'est un cas **leger** : il rentre largement dans le budget CI. L'essentiel du temps mur est l'import du
module compile et la construction du systeme ; les 20 pas sur `48 x 48` sont quasi instantanes.

---

## 16. Limites et differences avec les references

Ce cas est **explicitement** un test de plomberie de couplage, **pas** une reproduction d'un plasma de
reference. Limites assumees, verifiees dans le code et le coeur C++ :

- **Ionisation = transfert de densite seulement.** `add_ionization` (cf.
  `adc_cpp/include/adc/runtime/system.hpp` l.297-301) agit sur la **composante 0** (densite). Le transfert
  de **quantite de mouvement** et d'**energie** des particules creees (un electron/ion ne nait pas au repos
  ni a l'energie du fluide local) est une **simplification**. La conservation asseree (`drel < 1e-7`) ne
  porte donc que sur la masse `n_i + n_g`.
- **Friction sans echauffement.** `add_collision` (system.hpp l.303-306) applique `k (u_a - u_b)` sur la
  quantite de mouvement, opposee sur chaque espece (quantite de mouvement totale conservee), mais
  l'**echauffement par friction est neglige** (la chaleur n'est pas rendue a l'energie). C'est un
  raffinement non implemente.
- **Taux constants, non physiques.** `ionization_rate = 0.3`, `collision_rate = 0.5` sont des constantes de
  demonstration, sans calibration sur une section efficace ou une temperature reelle.
- **Ions et neutres isothermes.** Pas d'equation d'energie pour ces especes (`cs2` fixe) ; seuls les
  electrons portent une energie (Euler compressible).
- **Pas de magnetisation, geometrie cartesienne, domaine periodique.** Aucun champ `B`, aucune derive
  ExB ici (contrairement au diocotron) ; le seul couplage de champ est electrostatique via Poisson.
- **Grille grossiere, run court.** `48 x 48`, 20 pas : suffisant pour exercer le couplage, insuffisant
  pour une etude de convergence ou un taux de croissance quantitatif.

En resume : le cas valide que **les briques de couplage se composent et tournent en conservant la masse**,
rien de plus. Il ne doit pas etre presente comme une simulation de plasma physiquement calibree.

---

## 17. Tests/CI associes

- **Manifeste** (`cases_manifest.toml`) :

  ```toml
  [[case]]
  path = "plasma/run.py"
  category = "validation"
  ci = true
  needs = []
  desc = "Trois especes (e + i + n) : Poisson + ionisation + collision."
  ```

- **CI** (`.github/workflows/ci.yml`) : sur push/PR, la CI clone `adc_cpp`, construit le module `_adc`
  (`cmake ... -DADC_BUILD_PYTHON=ON`), installe `adc_cases` en editable, puis lance **uniquement** les cas
  `ci = true`, dont `plasma/run.py`. Un `assert` qui casse fait sortir le cas en erreur -> **CI rouge**. Ce
  cas est donc lui-meme un test de non-regression du couplage multi-especes (Poisson de systeme +
  ionisation + collision).
- **Tests de bindings** : la conservation isolee de la quantite de mouvement de la collision est verifiee
  dans le test des bindings de `adc_cpp` (pas ici : dans ce cas le champ electrique agit aussi sur les
  ions, donc la quantite de mouvement des ions n'est pas conservee isolement). Ce cas verifie le couplage
  **assemble**, pas chaque brique isolement.
