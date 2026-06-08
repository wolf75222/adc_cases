# tutorial : le diocotron ecrit de trois facons equivalentes (helper, briques, formules)

Ce cas est un **tutoriel executable**. Il prend une seule physique (le diocotron : une densite
electronique scalaire transportee par derive $E \times B$, avec un fond ionique neutralisant) et la
construit de **trois manieres** avec l'API generale d'`adc`, sans jamais se reposer sur une "classe
specialisee" qui cacherait la composition :

1. **helper specialise** : `adc_cases.models.diocotron(...)` -- l'oracle "tout fait" ;
2. **briques natives** : `adc.Model(state, transport, source, elliptic)` reconstruit a la main ;
3. **formules (DSL)** : `adc.dsl.Model(...)`, ou la physique est ecrite en expressions symboliques.

La lecon tient en une phrase : *un "modele nomme" n'est qu'une composition de briques generiques ; on
peut l'ecrire en briques OU en formules, c'est interchangeable et numeriquement identique*. Le cas le
**prouve** (`np.array_equal` sur les trois sorties) puis trace la croissance de l'instabilite. C'est
le mirroir, cote `adc_cases`, du tutoriel Sphinx d'adc_cpp (`docs/sphinx/getting_started/tutorial.md`).

## Contrat

| Champ | Valeur |
|---|---|
| **Categorie** (manifeste) | `tutoriel` (la variante CI `tutorial/equivalence.py` est `validation`) |
| **Entrees** | aucune ; tout est code (grille $96^2$, bande mode $l=2$, 60 pas CFL 0.4) |
| **Sorties** | `figures/tutorial_growth.png`, `figures/tutorial.gif`, `figures/tutorial_bricks_vs_dsl.png`, `figures/provenance.json` ; stdout `OK tutorial` |
| **Invariants** | masse conservee (transport advectif, domaine periodique) ; amplitude finale > initiale |
| **PROUVE** | les **trois** constructions (helper / briques / formules DSL) donnent un etat final **bit-identique** (`np.array_equal`, donc `max|ecart| = 0`, aucune tolerance) ; la masse derive de $< 10^{-6}$ (relatif) ; l'amplitude L2 de la perturbation **croit** (facteur ~1.5 sur 60 pas) |
| **NE PROUVE PAS** | ce n'est **pas** une reproduction d'un taux de croissance publie (cf. cas `diocotron` pour la confrontation a arXiv:2510.11808) ; le schema (NoSlope + Rusanov, ordre 1) est volontairement dissipatif et **sous-estimerait** un taux mesure ; 60 pas ne menent pas l'instabilite jusqu'aux vortex satures (on voit le **debut** de l'enroulement, pas l'etat non lineaire final) |
| **Provenance** | `figures/provenance.json` (SHA adc_cpp + adc_cases, backend DSL retenu, resolution, commande) |

Chaque section ci-dessous nomme la clause qu'elle justifie.

---

## 1. Physique : le diocotron en deux equations (justifie PROUVE, Invariants)

Le diocotron est l'instabilite d'une colonne (ici d'une **bande**) de charge non neutre dans un champ
magnetique uniforme $B_0 \hat{z}$. Les electrons derivent a la vitesse $E \times B$ :

$$ \mathbf{v} = \frac{\mathbf{E} \times \mathbf{B}}{B_0^2}
            = \frac{1}{B_0}\left(-\partial_y \phi,\; \partial_x \phi\right), $$

ou $\mathbf{E} = -\nabla \phi$. Cette vitesse est **a divergence nulle**
($\partial_x v_x + \partial_y v_y = (-\partial_x \partial_y \phi + \partial_y \partial_x \phi)/B_0 = 0$) :
le transport est une **advection incompressible**, donc la densite est simplement convectee (son
maximum et sa masse sont conserves a la discretisation pres). La densite obeit a une loi de
conservation, et le potentiel a un Poisson dont le second membre est la densite de charge
neutralisee :

$$ \partial_t n + \nabla\cdot(n\,\mathbf{v}) = 0, \qquad
   \nabla^2 \phi = \alpha\,(n - n_{i0}). $$

$n_{i0}$ est le fond ionique uniforme (immobile) qui neutralise la charge moyenne : sur un domaine
**periodique**, le second membre doit etre a moyenne nulle pour que Poisson soit soluble, d'ou le
choix $n_{i0} = \langle n \rangle$ (la moyenne de la densite initiale, `run.py` `main`).

**Mecanisme de l'instabilite.** La bande porte un saut de densite ; le cisaillement de la vitesse de
derive de part et d'autre de la bande est une **instabilite de Kelvin-Helmholtz** du champ de
vorticite (ici $n$ joue le role de la vorticite, $\phi$ celui de la fonction de courant). Une
perturbation au mode $l=2$ (deux longueurs d'onde le long de $x$) s'amplifie et **enroule** la bande
en oeil-de-chat. La condition initiale est posee par `band_density(n, L, amp=1, width=0.05, mode=2,
disp=0.02)` (`adc_cases/common/initial_conditions.py`) :

$$ n(x,y) = 1 + \exp\!\left(-\frac{(y - y_0(x))^2}{w^2}\right), \qquad
   y_0(x) = \tfrac{L}{2} + \mathrm{disp}\cdot\cos\!\left(\frac{2\pi\,l\,x}{L}\right). $$

Le plancher 1 est neutralise par $n_{i0}$ ; la gaussienne d'amplitude 1 est la bande ; le `disp` sur
$y_0$ est la graine du mode $l=2$.

---

## 2. Les trois fronts (le coeur du tutoriel ; justifie PROUVE)

Les trois constructions different **uniquement** par la facon de decrire le modele ; la grille, le
Poisson, la condition initiale, le schema (minmod + Rusanov) et l'integrateur (SSPRK2) sont
**identiques** (fonction `make_system` + `set_poisson` + `set_density` communes).

### Front 1 -- le helper specialise (l'oracle)

```python
models.diocotron(B0=B0, alpha=ALPHA, n_i0=n_i0)
```

`adc_cases.models.diocotron` (`adc_cases/models.py`) est une fonction d'une ligne qui renvoie un
`adc.Model`. Ce n'est **pas** une classe C++ dediee : son corps EST la composition de briques du
front 2. On l'utilise comme reference et on prouve plus bas que les trois coincident.

### Front 2 -- les briques natives, reconstruites a la main

`diocotron_from_bricks` (`run.py`) compose les **quatre briques de role** que le coeur sait assembler.
Chaque ligne choisit une brique pour un role :

```python
adc.Model(
    state=adc.Scalar(),                                   # 1 variable conservative : la densite n
    transport=adc.ExB(B0=B0),                             # flux d'advection f = n v, v = (-dy phi, dx phi)/B0
    source=adc.NoSource(),                                # aucune source par cellule (scalaire pur)
    elliptic=adc.BackgroundDensity(alpha=ALPHA, n0=n_i0), # rhs de Poisson = alpha (n - n0)
)
```

- `adc.Scalar()` declare l'etat : **une** variable conservative (la densite). Le coeur l'exige
  cohérente avec un transport scalaire (`ExB`) -- une incoherence etat/transport leve une erreur.
- `adc.ExB(B0=B0)` est le transport : il pose le flux $f = n\,\mathbf{v}$ avec la vitesse $E \times B$
  ci-dessus. C'est la **convention exacte** de la struct C++ `ExBVelocity`
  (`adc_cpp/include/adc/physics/hyperbolic.hpp`).
- `adc.NoSource()` : pas de terme source cellule-local (un scalaire advecte n'a ni force ni travail).
- `adc.BackgroundDensity(alpha, n0)` est le second membre elliptique : il pose
  $\text{rhs} = \alpha\,(n - n_0)$, convention de la struct C++ `BackgroundDensity`
  (`adc_cpp/include/adc/physics/elliptic.hpp`).

Le bloc est attache par `add_block` (`add_bricks_block`), qui prend une `ModelSpec` **native** :

```python
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
```

`adc.Spatial(minmod=True)` = volumes finis, limiteur minmod, flux de Rusanov (defaut), reconstruction
conservative. `adc.Explicit()` = SSPRK2, un sous-pas.

### Front 3 -- les memes physiques en formules (DSL)

`diocotron_from_dsl` (`run.py`) ecrit la **meme** physique sans aucune brique nommee : on declare des
variables et des champs, puis on **ecrit les formules** comme des expressions symboliques. `adc.dsl`
en genere du C++, le compile en `.so` et l'installe via `add_equation`.

```python
m = dsl.Model("tutorial_diocotron")
(n,) = m.conservative_vars("n")      # 1 variable conservative, role canonique Density
m.aux("phi")                         # potentiel (contrat aux ; non lu par le flux)
grad_x = m.aux("grad_x")             # composantes de grad phi, fournies par le solveur elliptique
grad_y = m.aux("grad_y")
vx = (-grad_y) / B0                  # derive E x B : v = (-dy phi, dx phi)/B0 (div v = 0)
vy = grad_x / B0
m.flux(x=[n * vx], y=[n * vy])       # flux physique d'advection f = n v(dir)
m.eigenvalues(x=[vx], y=[vy])        # 1 onde : la vitesse de derive (borne Rusanov / CFL)
m.primitive_vars(n=n)                # scalaire : primitif = conservatif (layout [n])
m.conservative_from([n])             # inversion triviale prim -> cons
m.elliptic_rhs(ALPHA * (n - n_i0))   # couplage Poisson : alpha (n - n_i0)
m.check()                            # toute variable referencee est-elle declaree ?
```

Ligne a ligne, c'est la **traduction symbolique** des deux briques `ExBVelocity` et
`BackgroundDensity` : `m.flux` reproduit $f = n\,\mathbf{v}$, `m.eigenvalues` donne la borne d'onde
pour Rusanov/CFL, `m.elliptic_rhs` reproduit $\alpha(n - n_{i0})$. `conservative_vars`/`aux`/`flux`/
`eigenvalues`/`elliptic_rhs` sont detaillees dans la
[reference DSL](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/reference/dsl_reference.md).

Le modele est compile puis branche par `add_equation` (`add_dsl_block`), qui aiguille le
`CompiledModel` selon son backend :

```python
compiled, backend = compile_dsl(diocotron_from_dsl(n_i0))   # production -> aot en repli
sim.add_equation("ne", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
```

`compile_dsl` prefere le backend **production** (chemin natif zero-copie, meme moteur que `add_block`)
et retombe sur **aot** (numeriquement identique) si la cle ABI du module ne correspond pas aux
en-tetes locaux. Le backend retenu est affiche et consigne dans `provenance.json` (sur cette
plateforme : `production`).

---

## 3. Pourquoi les trois coincident, et ce qu'une divergence trahirait (justifie PROUVE)

```python
assert np.array_equal(final_helper, final_bricks)   # helper == briques
assert np.array_equal(final_bricks, final_dsl)       # briques == DSL
```

- **helper == briques** est attendu **par construction** : `models.diocotron` *est* le `adc.Model(...)`
  du front 2 (memes arguments). L'assert le verifie quand meme : si un jour le helper deviait (autre
  brique, autre defaut), le test casserait. C'est la demonstration que "un modele nomme = une
  composition de briques".
- **briques == DSL** est le resultat **non trivial** : deux chemins de generation differents (briques
  pre-compilees vs C++ genere depuis des formules) produisent le **meme** noyau numerique. L'egalite
  est **bit-a-bit** (`np.array_equal`, pas `allclose`) parce que les formules DSL reproduisent
  exactement les conventions du coeur : meme vitesse $E\times B$, meme flux de Rusanov, meme borne
  d'onde, meme second membre. Une divergence (meme $10^{-15}$) trahirait une formule fausse : un signe
  inverse dans $\mathbf{v}$, une borne d'onde differente (Rusanov dependant de `eigenvalues`), ou un
  $n_{i0}$ different.

La sortie observee : `max|briques - DSL| = 0.000e+00`, backend `production`. C'est ce que montre la
figure `tutorial_bricks_vs_dsl.png` (section 5).

---

## 4. La boucle d'integration et la mesure d'amplitude (justifie Invariants)

`run_capture` (`run.py`) avance par pas CFL et echantillonne :

```python
for k in range(n_steps + 1):
    d = np.asarray(sim.density("ne")).copy()
    if k % every == 0:
        frames.append(d); times.append(sim.time()); amps.append(perturbation_amplitude(d))
    if k < n_steps:
        sim.step_cfl(0.4)
```

`sim.step_cfl(0.4)` choisit le pas stable $\mathrm{d}t = 0.4\,h / \max|\mathbf{v}|$ (le facteur 0.4
est le nombre de Courant). `perturbation_amplitude` mesure l'ecart-type de la densite par rapport a sa
moyenne **le long de $x$** :

```python
base = density.mean(axis=1, keepdims=True)    # moyenne en x (la bande non perturbee est uniforme en x)
delta = density - base
return float(np.sqrt(np.mean(delta * delta)))  # norme L2 de la perturbation
```

La bande initiale est uniforme le long de $x$ a $y$ fixe ; ce qui s'en ecarte porte l'instabilite.
L'amplitude croit (assert `ampf > amp0`). La masse, elle, est conservee a la precision machine
(`relative_drift(m_b, ne0.sum()) < 1e-6` ; observe ~$10^{-16}$) car l'advection $E\times B$ est
incompressible et le schema volumes finis est conservatif.

---

## 5. Figures (generees par `run.py`, dans `figures/`)

### `tutorial_bricks_vs_dsl.png` : la preuve visuelle de l'equivalence (justifie PROUVE)

![Etats finals briques vs DSL, max|ecart| = 0](figures/tutorial_bricks_vs_dsl.png)

Les deux panneaux sont l'etat final, a gauche par les **briques** natives, a droite par les
**formules** DSL.

- **PROUVE** : le titre affiche `max|briques - DSL| = 0e+00` -- c'est le `np.array_equal` de la
  section 3 rendu visible. Les deux cartes sont pixel pour pixel identiques.
- **SUGGERE** (non assere) : a l'oeil, la bande mode $l=2$ commence a s'epaissir aux cretes et a
  s'amincir aux noeuds : le debut de l'enroulement Kelvin-Helmholtz. Aucun assert ne mesure cette
  forme.
- **NON MONTRE** : la figure ne dit rien d'un troisieme code de reference ni d'un taux de croissance ;
  elle prouve l'identite briques/DSL, pas la justesse physique absolue (cf. cas `diocotron`).

### `tutorial_growth.png` : croissance de l'instabilite + carte finale (justifie Invariants)

![Amplitude L2 vs temps (semilog) et densite finale](figures/tutorial_growth.png)

A gauche, l'amplitude L2 de la perturbation en fonction du temps (echelle semilog) ; a droite, la
densite finale $n_e$.

- **PROUVE** : la courbe est **monotone croissante** apres un court transitoire ($t \lesssim 1$), de
  $6.8\times10^{-2}$ a $1.0\times10^{-1}$ sur 60 pas (facteur ~1.5) -- c'est l'assert `ampf > amp0`.
- **SUGGERE** (non assere) : la portion quasi rectiligne en semilog ($t \in [2, 6]$) suggere une
  croissance **exponentielle** $\sim e^{\gamma t}$, signature attendue d'une instabilite lineaire ;
  mais on ne **mesure** pas $\gamma$ ici (le cas `diocotron` le fait et le confronte au papier).
- **NON MONTRE** : la carte de droite montre le **debut** de l'enroulement (bande $l=2$ epaissie), pas
  les vortex satures de l'etat non lineaire ; 60 pas a $96^2$ s'arretent avant la saturation.

### `tutorial.gif` : l'evolution de la densite

![Animation : la bande mode l=2 s'enroule](figures/tutorial.gif)

- **PROUVE / visible** : l'animation est la sortie reelle du solveur (front 2) ; la bande $l=2$ se
  deforme continument. C'est le meme champ que la carte finale, anime.
- **NON MONTRE** : pas de comparaison cote a cote uniforme/AMR ici (voir le cas `diocotron_amr`), pas
  d'etat sature.

---

## 6. Reproduire (justifie la checklist : commande + cout)

```bash
# depuis la racine du depot adc_cases, avec le module adc sur le PYTHONPATH
PYTHONPATH=<build>/python python3 tutorial/run.py          # tutoriel complet + figures
PYTHONPATH=<build>/python python3 tutorial/equivalence.py  # smoke CI (equivalence seule, sans figures)
```

`run.py` compile un `.so` DSL (backend `production` ; ~quelques secondes au premier appel, mis en
cache ensuite), lance trois runs de 60 pas a $96^2$, ecrit trois figures + `provenance.json`. Cout :
quelques secondes (hors premiere compilation). `equivalence.py` fait moins de pas ($64^2$, 30 pas),
sans figures : c'est la variante **CI** (`needs = ["cxx"]`) qui verrouille l'equivalence des trois
fronts a chaque commit.

---

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | tutoriel complet : 3 constructions du diocotron, assert d'equivalence, figures + provenance |
| `equivalence.py` | smoke CI : reutilise les constructeurs de `run.py`, assert d'equivalence seule (sans figures) |
| `figures/tutorial_growth.png` | amplitude L2 vs temps (semilog) + densite finale |
| `figures/tutorial.gif` | evolution de la densite (enroulement) |
| `figures/tutorial_bricks_vs_dsl.png` | etats finals briques vs DSL, `max\|ecart\| = 0` |
| `figures/provenance.json` | SHA, backend DSL retenu, resolution, commande |

Voir aussi : le cas `diocotron_dsl` (equivalence DSL/natif en 3 panneaux + heatmap d'ecart), le cas
`diocotron` (reproduction du taux de croissance, confrontation arXiv:2510.11808), le cas
`diocotron_amr` (le meme diocotron sur AMR), et le tutoriel Sphinx d'adc_cpp
(`docs/sphinx/getting_started/tutorial.md`) dont ce cas est le mirroir cote application.
