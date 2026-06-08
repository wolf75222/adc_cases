# diocotron_dsl : le diocotron ecrit en formules, prouve bit-identique au natif

Le modele diocotron (derive E x B d'une densite scalaire, fond neutralisant) ecrit
ENTIEREMENT en formules symboliques (`adc.dsl.Model`) au lieu de briques C++ nommees, puis
prouve **bit-identique** a la composition native `adc_cases.models.diocotron` sur la meme grille,
la meme condition initiale et le meme nombre de pas. La physique n'est PAS re-derivee ici :
elle l'est dans [`../diocotron/`](../diocotron/) (mecanisme de l'instabilite, taux de croissance,
relation de dispersion). Ce cas verifie une seule chose : que les formules DSL reproduisent
EXACTEMENT les conventions des briques du coeur, au point que les deux chemins produisent le meme
etat **au bit pres** (`np.array_equal`, aucune tolerance).

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` (`cases_manifest.toml`, `diocotron_dsl/run.py`, `ci = true`, `needs = ["cxx"]`) |
| Entrees | grille $96^2$, $L=1$, **periodique** ; CI bande mode 2 `band_density(amp=1, width=0.05, mode=2, disp=0.02)` ; $B_0=1$, $\alpha=1$ ; fond ionique $n_{i0}=\overline{n_e}=1.088623$ (moyenne de la CI, solubilite du Poisson periodique) ; 60 pas, CFL 0.4 ; minmod + Rusanov, SSPRK2, Poisson `geometric_mg` |
| Sorties | densite finale $n$ des DEUX chemins (natif, DSL) ; backend retenu ; 1 figure (3 panneaux) dans `figures/` + `figures/provenance.json` |
| Invariants garantis | les `assert` de `run.py:194-209` : `np.array_equal(d_dsl, d_natif)` (bit) ; `t_dsl == t_natif` (bit) ; `m_dsl == m_natif` (bit) ; `mass_drift < 1e-6` ; `amp_final > amp_initial` |
| PROUVE | **egalite bit du chemin complet** : $\max\lvert n_{\mathrm{DSL}}-n_{\mathrm{natif}}\rvert=0.000\times10^{0}$ apres 60 pas, `np.array_equal` True ; temps et masse identiques au bit ($t=6.213869$, $m=1.0032746734\times10^{4}$, les deux). L'identite tient parce que les formules DSL emettent les MEMES expressions ponctuelles que `ExBVelocity` et `BackgroundDensity` (table section 3), compilees dans le MEME assembleur par cellule |
| NE PROUVE PAS | **ce n'est PAS une reproduction publiee**, ni une validation du taux de croissance (cela vit dans [`../diocotron/`](../diocotron/), categorie `reproduction`). Aucun nombre n'est confronte a un article. L'egalite bit prouve que le chemin DSL ne devie pas du chemin natif ; elle ne dit RIEN de la justesse physique des deux (un bug commun aux deux briques resterait invisible). Le backend natif `production` (`add_native_block`) **echoue** sur cette plateforme (ABI : `_adc` bati contre des en-tetes != `include/`) : le run nominal passe par `aot` (host-marshale, numerique identique au natif, verifie). L'egalite bit du chemin `production` n'est donc PAS exercee ici |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, backend `aot` (apres echec `production`), $96^2$, ~6 s 1 coeur CPU ; `figures/provenance.json` |

A la fin tu sauras : quelles conventions du coeur le DSL doit reproduire pour que l'egalite bit
tienne, comment cette egalite est verifiee (`np.array_equal`, pas de tolerance) et ce qu'une carte
d'ecart non noire trahirait.

---

## 1. Physique : liee, pas recopiee

Le mecanisme (anneau/bande de charge, rotation differentielle, instabilite de Kelvin-Helmholtz
d'un anneau de vorticite, taux $\gamma_l$, relation de dispersion) est derive dans
[`../diocotron/`](../diocotron/README.md), sections 1, 4 et 5. Ici la CI est une **bande**
horizontale (`band_density`, mode azimutal 2), la variante periodique minimale du meme cas (cf.
[`../diocotron/band_instability.py`](../diocotron/band_instability.py)) : pas de paroi conductrice,
domaine periodique, ce qui rend le Poisson de systeme soluble sans geometrie cartesienne en marches
d'escalier. Le seul role de la physique ici est de fournir une dynamique non triviale (l'amplitude
croit, `run.py:209`) sur laquelle l'egalite bit a un sens : deux chemins qui resteraient identiquement
nuls seraient une egalite vide.

---

## 2. Les deux chemins, et qui compile quoi

Le cas construit le **meme** `adc.System(n=96, L=1, periodic=True)` (`run.py:115`, `make_system`)
deux fois, avec le meme Poisson (`set_poisson(rhs="charge_density", solver="geometric_mg")`),
la meme densite initiale et 60 pas `step_cfl(0.4)`. Seul le **bloc** differe :

| Chemin | Construction du bloc | Ligne `run.py` |
|---|---|---|
| natif (oracle) | `add_block("ne", model=models.diocotron(B0, alpha, n_i0), spatial=Spatial(minmod=True), time=Explicit())` | `run.py:122-123` (`run_native`) |
| DSL | `add_equation("ne", model=compiled, spatial=FiniteVolume(limiter="minmod", riemann="rusanov"), time=Explicit())` | `run.py:155-157` (`run_dsl`) |

`models.diocotron` (`adc_cases/models.py:18-25`) est `adc.Model(state=Scalar, transport=ExB(B0),
source=NoSource, elliptic=BackgroundDensity(alpha, n0=n_i0))` : quatre briques C++ nommees.
`compiled` est le **meme modele ecrit en expressions** (`diocotron_dsl_model`, `run.py:68-101`),
emis en C++ par `adc.dsl`, compile en `.so` et charge comme bloc. Pour que `add_equation` et
`add_block` empruntent le meme assembleur par cellule, il suffit que les expressions DSL emettent
les memes fonctions ponctuelles que les briques (`flux`, `eigenvalues`, `rhs` elliptique).

### Table 3 couches (qui calcule quoi, chemin DSL)

| Ligne `run.py` | Couche | Ce qui se passe |
|---|---|---|
| `sim.add_equation("ne", model=compiled, spatial=FiniteVolume(...), time=Explicit())` (`run.py:155-157`) | Python **compose** | choix du schema (MUSCL minmod + Rusanov) et de l'integrateur (SSPRK2), strictement les memes que le natif (`Spatial(minmod=True)`, `run.py:123`) |
| `m.flux(...)` / `m.eigenvalues(...)` / `m.elliptic_rhs(...)` (`run.py:88,90,98`) compiles par `model.compile(..., backend=cand)` (`run.py:151`) | les EXPRESSIONS DSL **figent** la physique | le DSL emet en C++ les memes expressions ponctuelles que `ExBVelocity` / `BackgroundDensity` (table section 3) ; ces expressions remplacent la brique nommee |
| `assemble_rhs<minmod, Rusanov>` + Poisson de systeme (`GeometricMG`) | noyau **par cellule** (device) | le MEME assembleur que `add_block` : `add_equation` aiguille sur `add_native_block` (production) ou `add_compiled_block` (aot), sans callback Python dans le hot path |

C'est le point de tout le cas : la couche du milieu change de FORME (expressions vs brique nommee)
sans changer le RESULTAT, parce que la couche du bas est identique.

---

## 3. Les conventions du coeur, reproduites en formules (justifie PROUVE)

L'egalite bit ne tient QUE si chaque expression DSL est le sosie exact de la fonction ponctuelle de
la brique. Voici la correspondance, ancree dans les en-tetes du coeur et dans `run.py`.

### Transport E x B (`include/adc/physics/hyperbolic.hpp`, struct `ExBVelocity`)

La brique native (`hyperbolic.hpp:27-59`) definit la vitesse de derive, le flux et le spectre :

```cpp
ADC_HD Real velocity(const Aux& a, int dir) const {            // hyperbolic.hpp:31-33
  return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
}
f[0] = u[0] * velocity(a, dir);                                // flux : hyperbolic.hpp:36
e[0] = velocity(a, dir);                                       // eigenvalue : hyperbolic.hpp:46
```

Les formules DSL (`diocotron_dsl_model`, `run.py:83-94`) reproduisent CHAQUE ligne :

| Convention du coeur | Brique (`hyperbolic.hpp`) | Formule DSL (`run.py`) |
|---|---|---|
| vitesse $v=(-\partial_y\phi/B_0,\ \partial_x\phi/B_0)$ | `velocity` l.31-33 : `(-grad_y/B0, grad_x/B0)` | `vx = (-grad_y)/B0`, `vy = grad_x/B0` (`run.py:84-85`) |
| flux $f = n\,v(\mathrm{dir})$ | `flux` l.34-38 : `u[0]*velocity` | `m.flux(x=[n*vx], y=[n*vy])` (`run.py:88`) |
| valeur propre (1 onde) $= v(\mathrm{dir})$ | `eigenvalues` l.44-48 : `velocity` | `m.eigenvalues(x=[vx], y=[vy])` (`run.py:90`) |
| variable conservative unique $n$ (role Density), prim = cons | `conservative_vars`/`to_primitive` l.49-58 : identite | `m.conservative_vars("n")`, `m.primitive_vars(n=n)`, `m.conservative_from([n])` (`run.py:75,93-94`) |

Les champs auxiliaires `phi`/`grad_x`/`grad_y` lus par le flux sont declares cote DSL par
`m.aux("phi")`, `m.aux("grad_x")`, `m.aux("grad_y")` (`run.py:79-81`) : ils nomment les emplacements
du canal `adc::Aux` que le coeur remplit avec le potentiel et son gradient, les memes membres
`a.grad_x`/`a.grad_y` que lit `velocity` (`hyperbolic.hpp:32`). `phi` est declare pour completer le
contrat mais le flux ne lit que le gradient (`velocity` n'utilise que `grad_x`/`grad_y`), exactement
comme la brique.

### Second membre elliptique (`include/adc/physics/elliptic.hpp`, struct `BackgroundDensity`)

```cpp
ADC_HD Real rhs(const State& u) const { return alpha * (u[0] - n0); }   // elliptic.hpp:34-36
```

Formule DSL (`run.py:98`) : `m.elliptic_rhs(ALPHA * (n - n_i0))`. Meme expression
$\alpha\,(n - n_{i0})$, meme role (fond neutralisant, RHS a moyenne nulle sur domaine periodique
grace au choix $n_{i0}=\overline{n_e}$, `run.py:173`). Le bloc se couple au Poisson de systeme via
`set_poisson(rhs="charge_density")` (`run.py:158`), l'alias generique de la somme des seconds
membres elliptiques de chaque bloc (ici l'unique `elliptic_rhs`).

### Source

`models.diocotron` utilise `adc.NoSource` ; cote DSL, `diocotron_dsl_model` n'appelle aucun
`m.source(...)`, ce que `m.check()` (`run.py:100`) accepte (source optionnelle). Pas de terme
source des deux cotes. `m.check()` verifie que toute variable referencee par
`flux`/`eigenvalues`/`elliptic_rhs` est declaree (conservative ou aux) : c'est le garde-fou qui
empeche d'emettre un C++ referencant un symbole fantome.

---

## 4. Comment l'egalite bit est verifiee, et ce qu'une divergence trahirait

`run.py:183-196` lance les deux chemins sur la MEME configuration et compare l'etat final :

```python
dn, tn, mn = run_native(ne0, n_i0, n_steps)                    # oracle natif (run.py:183)
dd, td, md, backend = run_dsl(ne0, n_i0, n_steps)              # chemin DSL (run.py:184)
max_abs = float(np.max(np.abs(dd - dn)))                       # run.py:191
identical = bool(np.array_equal(dd, dn))                       # run.py:192
assert identical, "...une formule DSL diverge d'une brique du coeur..."   # run.py:194-196
```

- `np.array_equal(dd, dn)` exige l'**egalite element par element au bit** : aucune tolerance, aucun
  `isclose`. C'est l'observable juste pour ce cas : les deux chemins executent le MEME assembleur
  flottant dans le MEME ordre d'operations, donc tout ecart non nul signalerait que les expressions
  emises different (une convention de signe, un facteur $1/B_0$ manquant, un $n_{i0}$ oublie), pas
  un simple bruit d'arrondi.
- `assert td == tn` (`run.py:206`) et `assert md == mn` (`run.py:207`) verrouillent aussi le temps
  et la masse au bit : le DSL ne doit pas seulement finir au meme etat, mais y arriver par la meme
  sequence de pas (meme `step_cfl`, meme `dt` a chaque pas).
- **Ce qu'une divergence trahirait** : si un seul element de `dd - dn` etait non nul, la cause
  serait une formule DSL deviant d'une brique du coeur. Exemples cibles : `vx = grad_y/B0` (signe
  inverse de `velocity`), `vy = grad_x` (facteur $1/B_0$ omis), ou `elliptic_rhs(ALPHA*n)` (fond
  $n_{i0}$ oublie, qui casserait la moyenne nulle du RHS periodique). Chacune deplacerait l'etat et
  ferait apparaitre une tache sur la carte de la section 5.

La masse est ensuite controlee dans l'absolu pour le seul chemin DSL : `mass_drift = relative_drift(
md, mass0)` puis `assert mass_drift < 1e-6` (`run.py:208`). La tolerance $10^{-6}$ est lache : la
derive mesuree vaut $1.813\times10^{-16}$ (provenance), au niveau machine, car le flux E x B est a
divergence nulle (conservation exacte au flottant pres). Elle borne la masse loin du signal sans
exiger l'egalite bit (deja couverte par `md == mn`).

---

## 5. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (meme configuration que `run.py`), versionnees avec
`figures/provenance.json`. Commande exacte en section 6.

### `equivalence_heatmap.png` : trois panneaux (natif, DSL, ecart)

![Trois panneaux cote a cote : densite finale natif, densite finale DSL (memes bandes ondulees), et la carte d'ecart entierement noire (max 0)](figures/equivalence_heatmap.png)

On montre les **deux champs reels** (panneaux 1 et 2) PUIS leur ecart (panneau 3), plutot qu'un
carre noir seul qui aurait l'air vide ou casse.

- **PROUVE / visible** : les panneaux natif et DSL sont la **MEME** bande ondulee (mode azimutal 2,
  deux creux), densite de $1.0$ a $\approx 1.98$ ($\sigma\approx 0.23$, donc un champ **structure**,
  pas uniforme). Le troisieme panneau ($\lvert n_{\mathrm{DSL}}-n_{\mathrm{natif}}\rvert$, echelle
  fixee $[0,\,10^{-15}]$) est **identiquement noir** : $\max=0.0\times10^{0}$, `np.array_equal` True
  (asserte `run.py:194`). L'echelle au niveau machine garantit qu'un **seul** pixel different
  ressortirait ; il n'y en a aucun. Le residu n'est pas "petit", il est **exactement nul** : meme
  tableau bit pour bit.
- **SUGGERE (non assere)** : l'amplitude du mode a cru d'un facteur $1.5212$ ($amp_{0}=6.778\times
  10^{-2}\to amp_{final}=1.031\times10^{-1}$, `run.py:202-203`) ; seul $amp_{final}>amp_{0}$ est
  asserte (`run.py:209`). La phase non lineaire d'enroulement n'est pas atteinte sur 60 pas.
- **NON MONTRE** : l'egalite bit ne dit RIEN de la justesse physique. Un bug present dans
  `ExBVelocity` ET reproduit fidelement par la formule DSL donnerait aussi une carte noire. Elle
  prouve la non-deviation DSL/natif, pas la correction du modele (validee ailleurs,
  [`../diocotron/`](../diocotron/)).

---

## 6. Reproduire (justifie 14 de la checklist : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/diocotron_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~6 s (compile la .so au 1er run)
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 figures + provenance.json
```

Prerequis : `numpy`, un compilateur C++20 (`needs = ["cxx"]` : le DSL emet et compile une `.so`),
et le module `adc` importe **avec le meme interpreteur** que celui qui l'a compile (suffixe ABI
`cpython-312`). Le `.so` DSL est ecrit sous `out/diocotron_dsl/` via `case_output_dir`
(`run.py:136`), repertoire git-ignore : aucun artefact jetable dans l'arbre source.

Sortie attendue de `run.py` (capturee, machine de dev macOS arm64) :

```
backend 'production' indisponible (RuntimeError), essai suivant
backend DSL retenu : 'aot'
natif : t = 6.213869, masse = 1.0032746734e+04
DSL   : t = 6.213869, masse = 1.0032746734e+04
max|DSL - natif| = 0.000e+00   bit-identique = True
amplitude : initiale 6.777566e-02 -> finale 1.031025e-01 (facteur 1.5212)
derive de masse relative (DSL) = 1.813e-16
OK diocotron_dsl (equivalence DSL <-> natif bit-identique, backend 'aot')
```

Le backend natif `production` echoue ici par une cle d'ABI : `_adc` a ete bati contre des en-tetes
differents de `include/` (`run.py:140-147`), donc `model.compile(backend="production")` puis
`add_native_block` levent un `RuntimeError` explicite (jamais d'UB silencieux). Le `try`/`except` du
`for cand in ("production", "aot")` (`run.py:149-164`) rejoue alors tout en `aot`
(`add_compiled_block`, host-marshale, numerique identique au natif) : c'est ce chemin que le run
nominal exerce. **Caveat plateforme** : le verdict `OK`, l'egalite bit ($\max=0$) et l'ordre de
grandeur (temps $\approx 6.2$, masse $\approx 10^{4}$) sont stables ; le backend retenu depend de la
compatibilite ABI du module `_adc` charge (`production` quand `_adc` est bati contre `include/`,
`aot` sinon), et les derniers chiffres de $t$/masse varient avec la BLAS et l'ordre de sommation
(cf. `figures/provenance.json`). Sur un module `production`-compatible, le run prendrait le chemin
natif et l'egalite bit `add_native_block` serait alors exercee.

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | les deux chemins (natif vs DSL), egalite bit par `assert` (`np.array_equal`, temps, masse) |
| `make_figures.py` | rejoue la config, ecrit `equivalence_heatmap.png` (3 panneaux) + `provenance.json` |
| `figures/equivalence_heatmap.png` | 3 panneaux : densite natif, densite DSL, ecart (noir, max 0) |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, backend, resolution, $\max\lvert d\rvert$, temps, masses, amplitude |
| [`../diocotron/`](../diocotron/) | physique du parent : mecanisme, taux $\gamma_l$, relation de dispersion (NON recopiee ici) |
| `adc_cpp/include/adc/physics/hyperbolic.hpp` | brique `ExBVelocity` reproduite par les formules DSL |
| `adc_cpp/include/adc/physics/elliptic.hpp` | brique `BackgroundDensity` reproduite par `elliptic_rhs` |
