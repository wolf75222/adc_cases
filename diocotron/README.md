# Diocotron : benchmark de normalisation du diocotron reduit (Petri), arXiv:2510.11808

Benchmark de **normalisation** du **diocotron reduit** (limite de derive `E x B`) de la
Section 5.3 de Hoffart, Maier, Shadid, Tomas, *Structure-preserving finite-element
approximations of the magnetic Euler-Poisson equations*
([arXiv:2510.11808](https://arxiv.org/abs/2510.11808)), realise **avec le solveur `adc`**
(la lib `adc_cpp` via ses bindings Python), pas avec leur code ni un code tiers.

Honnetete totale sur le perimetre :

- Le modele numerique est **reduit** : un scalaire de densite transporte par derive `E x B`
  (briques `Scalar` + `ExB` + `BackgroundDensity`). **Ce n'est PAS** le systeme Euler-Poisson
  magnetise complet `(rho, rho u, rho v)` du papier. C'est la *limite de derive* que le papier
  vise, pas le systeme integral.
- La validation porte sur la **normalisation analytique** : la cible (probleme aux valeurs
  propres de Petri, resolu en numpy) est retrouvee a 3 chiffres. C'est un **benchmark de
  normalisation**, pas une reproduction du schema elements-finis du papier.
- Le solveur volumes-finis `adc` **sous-predit** le taux de croissance a resolution moderee
  (l=3 : -22 %, l=4 : -27 %, l=5 : -5 % a n=192, MUSCL minmod). Voir section 16.

Le pendant Euler-Poisson **complet** (etat conservatif, etage Schur, geometrie suspecte) est un
cas distinct, marque `reproduction-candidate` PENDING dans le manifeste :
[`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/). Le present cas ne le remplace pas.

---

## 1. Objectif du cas

Mesurer, avec le solveur `adc`, le **taux de croissance de l'instabilite diocotron** d'une
colonne creuse (anneau de charge) et le confronter a la relation de dispersion analytique
(probleme aux valeurs propres radial de Petri), elle-meme retrouvee a 3 chiffres des cibles du
papier. L'instabilite diocotron est une instabilite de cisaillement de la derive `E x B` d'un
plasma non neutre : une perturbation azimutale de mode `l` croit exponentiellement, l'anneau
developpe `l` lobes qui s'enroulent.

Deux scripts, deux roles distincts (manifeste `cases_manifest.toml`) :

| Script | Categorie | CI | Role |
|---|---|---|---|
| [`run.py`](run.py) | `reproduction` | non (long, `needs=["matplotlib"]`) | Benchmark de normalisation complet : analytique de Petri (numpy) + mesure `adc` modes 3/4/5 + figures + gif. **LONG**. |
| [`band_instability.py`](band_instability.py) | `validation` | oui | Variante periodique minimale : verifie conservation de masse + croissance de l'instabilite par `assert`. Leger, sans figures. |

Ce README documente les deux. `run.py` n'est **pas** relance ici (long) ; il est decrit depuis
le code. `band_instability.py` a ete execute (section 12).

---

## 2. Equations

### Modele numerique (ce que `adc` integre reellement) : diocotron reduit `E x B`

Une seule variable scalaire, la densite de charge `n_e(x, y, t)`, transportee par la vitesse
de derive `E x B` issue du potentiel electrostatique `phi` :

```
d_t n_e + div( v n_e ) = 0,          v = (E x B)/B0^2 = (1/B0) (-d_y phi, d_x phi)
-Laplacien(phi) = alpha (n_e - n_i0)
```

Le champ de vitesse `E x B` est a **divergence nulle** : le transport est purement advectif,
donc la **masse totale est conservee exactement** (invariant verifie, section 11). `n_i0` est un
fond neutralisant (ions immobiles) ; `alpha` est le couplage Poisson. Conditions sur `phi` :

- `run.py` (anneau) : **Dirichlet** sur une **paroi conductrice circulaire** (embedded boundary)
  de rayon `Rwall = 0.40`, domaine carre `[0,L]^2` non periodique.
- `band_instability.py` (bande) : domaine **periodique**, Poisson exige un RHS a moyenne nulle,
  d'ou le fond `n_i0 = moyenne(n_e)` (compatibilite de Fredholm).

### Cible analytique : probleme aux valeurs propres de Petri

Resolu en numpy (`run.py:diocotron_eigenvalue`), reimplemente d'apres Petri
(arXiv:astro-ph/0611936) / Davidson-Felice :

```
omega L_m phi = m Omega L_m phi + q_m phi,   phi(0) = phi(R_w) = 0
Omega(r) = -(1/r^2) integrale_0^r rho(r') r' dr'        (rotation azimutale)
q_m      = (m / r) d_r rho                              (gradient radial)
L_m      = operateur radial de Laplace azimutal (mode m)
```

On forme `M = L^{-1} A`, le taux est `gamma = max Im(omega)`, normalise par
`omega_D = rho_bar / (2 pi)` (facteur `2 pi / rho_bar` dans le code).

---

## 3. Modele physique

Plasma non neutre confine par un champ magnetique uniforme `B0 = 1` selon `z`. Le plasma ne
voit que sa propre derive `E x B` (pas d'inertie, pas de pression : c'est la limite de derive
`omega_d << omega_p << omega_c` visee par le papier). La configuration initiale est une **colonne
creuse** (anneau de densite, `run.py`) ou une **bande** (`band_instability.py`). Le cisaillement
de la rotation differentielle `Omega(r)` rend ces configurations lineairement instables :
chaque mode azimutal `l` croit en `exp(gamma_l t)` jusqu'a saturation non lineaire (enroulement
en lobes).

Le modele est assemble cote application par `adc_cases.models.diocotron` :

```python
# adc_cases/models.py
def diocotron(B0=1.0, alpha=1.0, n_i0=0.0):
    return adc.Model(
        state=adc.Scalar(),                              # 1 variable : la densite
        transport=adc.ExB(B0=B0),                        # flux de derive E x B
        source=adc.NoSource(),                           # aucune source locale
        elliptic=adc.BackgroundDensity(alpha=alpha, n0=n_i0),  # RHS Poisson = alpha (n - n0)
    )
```

`adc_cpp` ne connait aucun "scenario diocotron" : il ne fournit que des briques generiques
(`Scalar`, `ExB`, `BackgroundDensity`...). Le nom "diocotron" vit cote application.

---

## 4. Methode numerique

| Aspect | Choix |
|---|---|
| Discretisation spatiale | Volumes finis, grille cartesienne uniforme a centres de cellules |
| Reconstruction | MUSCL avec limiteur **minmod** (`adc.Spatial(minmod=True)`) -> ordre 2 |
| Flux numerique | **Rusanov** (Lax-Friedrichs local) |
| Integration temporelle | **Explicite SSPRK2** (`adc.Explicit()` par defaut) |
| Pas de temps | `dt = CFL * dx / v_max`, **CFL = 0.4** (`sim.step_cfl(0.4)`) |
| Couplage Poisson | resolu **une fois par sous-pas** (chaque etage SSPRK recalcule `phi`) |
| Solveur Poisson | multigrille geometrique (`solver="geometric_mg"`) |
| Conditions Poisson | Dirichlet + paroi circulaire (`run.py`) / periodique (`band_instability.py`) |

`sweep.py` / `SWEEP_RESULTS.md` montrent qu'un ordre plus eleve est atteignable depuis le chemin
natif (`adc.Spatial(limiter="weno5")` = WENO5-Z 5 points + `adc.Explicit(method="ssprk3")`), mais
le cas nominal reste **minmod + Rusanov + SSPRK2**.

---

## 5. Architecture ADC utilisee

« Python compose, le C++ calcule. » Tout le scenario est assemble depuis Python via l'API de
composition par blocs ; aucun solveur C++ dedie au diocotron.

```python
sim = adc.System(n=192, L=1.0, periodic=False)
sim.add_block("ne",
              model=models.diocotron(B0=1.0, alpha=1.0, n_i0=0.0),  # briques Scalar+ExB+BackgroundDensity
              spatial=adc.Spatial(minmod=True),                     # MUSCL minmod (ordre 2)
              time=adc.Explicit())                                  # SSPRK2
sim.set_poisson(rhs="charge_density", solver="geometric_mg",
                bc="dirichlet", wall="circle", wall_radius=0.40)    # paroi conductrice circulaire
sim.set_density("ne", ring_numpy)                                   # CI anneau (numpy)
dt = sim.step_cfl(0.4)                                              # transport + Poisson = C++
```

Briques `adc_cpp` exercees : `Scalar`, `ExB`, `BackgroundDensity`, `NoSource`, le multigrille
geometrique de Poisson et la **paroi conductrice embedded-boundary** (cercle Dirichlet). Cote
diagnostic, Python lit `sim.potential()`, `sim.density("ne")`, `sim.mass("ne")`, `sim.time()`.

---

## 6. Carte des fichiers

```
diocotron/
  README.md             ce fichier
  run.py                benchmark de normalisation complet (analytique + adc + figures + gif). LONG, hors CI.
  band_instability.py   variante periodique minimale (asserts masse + croissance). CI.
  sweep.py              balayage ordre x resolution x mode (etude, hors manifeste).
  SWEEP_RESULTS.md      donnees brutes du balayage (O1/O2/O5, n=128..512).
  figures/              assets committes (sorties de reference d'un run.py)
    dispersion.png        gamma vs mode l : analytique + adc + cibles papier
    amplitude.png         |c_l|(t) en log (croissance exponentielle)
    snapshots.png         4 instantanes de densite (mode l=4)
    diocotron.gif         evolution de la densite (mode l=4)
```

Paquet partage `adc_cases/` (briques et CI mutualisees) :

```
adc_cases/
  models.py                         models.diocotron(...) : composition des briques
  common/initial_conditions.py      ring_density(...) [run.py], band_density(...) [band_instability.py]
  common/checks.py                  relative_drift(...) (conservation de masse)
  common/grid.py                    meshgrid_xy(...) (convention field[j, i])
  common/io.py                      case_output_dir(...) -> out/diocotron/ (gitignore)
```

> Note : `run.py` ecrit ses figures dans `out/diocotron/` (hors source, gitignore), PAS dans
> `figures/`. Le dossier `figures/` contient des assets committes (sorties de reference).

---

## 7. Prerequis

- Module `adc` compile et sur le `PYTHONPATH` (build `adc_cpp` avec `-DADC_BUILD_PYTHON=ON`).
- `numpy` (les deux scripts).
- `matplotlib` (uniquement `run.py`, pour figures + gif ; backend `Agg`). `band_instability.py`
  n'en a **pas** besoin.
- `adc_cases` importable (installe, ou son depot sur le `PYTHONPATH` ; les deux scripts gerent le
  fallback `sys.path` automatiquement).
- Aucun compilateur C++ requis au runtime (le cas n'utilise que des briques natives ; pas de
  `needs=["cxx"]`).

---

## 8. Commande exacte

Variante legere (CI, executee ici) :

```bash
cd /private/tmp/adc_cases-readmes/diocotron
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 band_instability.py
```

Benchmark complet (LONG, figures + gif, **non relance ici**) :

```bash
cd /private/tmp/adc_cases-readmes/diocotron
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
# figures et gif ecrits dans out/diocotron/
```

Recette de build (depuis `adc_cpp`) :

```bash
cd ../adc_cpp && cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc -j
```

---

## 9. Explication du code par etapes

### `band_instability.py` (validation CI)

1. **CI bande** : `band_density(n=96, L=1.0, amp=1.0, width=0.05, mode=2, disp=0.02)` construit
   une bande horizontale gaussienne perturbee sinusoidalement le long de `x` (mode azimutal 2).
2. **Fond neutralisant** : `n_i0 = moyenne(ne0)`, indispensable pour la solubilite de Poisson
   periodique (RHS a moyenne nulle).
3. **Composition** : `adc.System(n=96, periodic=True)` + `add_block("ne", models.diocotron(...),
   minmod, Explicit)` + `set_poisson(rhs="charge_density", solver="geometric_mg")` (periodique,
   sans paroi).
4. **Reference** : `mass0 = sim.mass("ne")`, `amp0 = perturbation_amplitude(...)` ou l'amplitude
   est la norme L2 de la deviation de la densite a sa moyenne en `x` (partie non axisymetrique).
5. **Boucle** : 120 pas a CFL=0.4. A chaque pas, `assert` que la masse est conservee
   (`relative_drift < 1e-6`) ; toutes les 10 iterations, impression de `t`, amplitude, masse.
6. **Verdict** : `assert amp_last > amp0` (l'instabilite a fait croitre la perturbation).

### `run.py` (benchmark de normalisation complet)

1. **Analytique** (`diocotron_eigenvalue`, numpy pur) : assemble l'operateur radial `L_m`,
   la rotation `Omega(r)` (integrale cumulee de `rho r`), forme `M = L^{-1} A`, prend
   `gamma = max Im(eig(M))`, normalise par `2 pi / rho_bar`. Geometrie `a:b:Rw = 6:8:16`,
   anneau net `w = 0.05`, `N = 2000` points (le taux normalise est invariant d'echelle).
2. **Numerique** (`measure_growth`, par mode `l` in {3,4,5}) :
   - `make_ring_system` : `adc.System(n=192, periodic=False)` + bloc diocotron + Poisson
     Dirichlet paroi circulaire `Rwall=0.40` + CI anneau (`ring_density`, `delta=0.01`).
   - boucle `nsteps=900` a CFL=0.4 ; a chaque pas, `mode_l_amplitude(phi, ...)` echantillonne
     `phi` sur un cercle au rayon median par interpolation bilineaire (`bilinear_on_circle`)
     puis prend le coef de Fourier azimutal du mode `l` (FFT).
   - `fit_linear_phase` : pente de `log|c_l|` sur la phase de croissance (de `1.3 a0` a `0.85`
     du pic), `gamma_norm = pente * 2 pi / rho_bar`.
3. **Figures** : `dispersion.png` (gamma vs l : analytique + adc + cibles papier),
   `amplitude.png` (`|c_l|(t)` en log), puis `run_evolution(l=4, delta=0.1)` ->
   `diocotron.gif` + `snapshots.png`.

---

## 10. Conditions initiales

Les CI vivent dans `adc_cases/common/initial_conditions.py` (numpy, convention `field[j, i]`).

### Anneau (colonne creuse) -- `run.py`

```python
# ring_density(n, L=1.0, r0=0.15, r1=0.20, mode=l, delta=0.01, floor=1e-3)
r  = hypot(X - L/2, Y - L/2)
th = arctan2(Y - L/2, X - L/2)
ne = floor partout                                 # ~1e-3 hors anneau
ne[anneau] = 1 - delta + delta * sin(mode * th)    # ~1 + perturbation azimutale dans [r0, r1]
```

Anneau `r0:r1:Rwall = 0.15:0.20:0.40` (ratios `6:8:16`), centre du domaine `(L/2, L/2)`,
perturbation faible `delta=0.01` pour la mesure de taux, `delta=0.1` pour le gif (visible).

### Bande -- `band_instability.py`

```python
# band_density(n, L=1.0, amp=1.0, width=0.05, mode=2, disp=0.02, floor=1.0)
y0 = L/2 + disp * cos(2 pi mode x / L)
ne = floor + amp * exp(-(y - y0)^2 / width^2)      # bande horizontale ondulee (mode 2)
```

Domaine periodique, fond ionique `n_i0 = moyenne(ne)`.

---

## 11. Invariants et assertions

`band_instability.py` (le seul des deux a faire des `assert`) :

| Invariant | Verification | Tolerance |
|---|---|---|
| Conservation de la masse | `relative_drift(mass, mass0) < 1e-6` a chaque pas | `1e-6` |
| Croissance de l'instabilite | `amp_last > amp0` en fin de run | stricte |

La conservation exacte de la masse est garantie par construction : le champ `E x B` est a
divergence nulle, donc le transport est conservatif. Dans le run capture, la masse reste
constante a `1.00327467e+04` sur les 120 pas (derive ~ 1e-13, soit ~7 ordres sous la tolerance).

`run.py` ne fait **pas** d'assert de validation (c'est un script de figures) ; il imprime
`OK repro_paper_2510_11808` et garde-fou seulement contre les NaN de `phi` (arret de boucle).

---

## 12. Sorties attendues

### `band_instability.py` (execute, ~0.5 s wall)

```
=== Demo diocotron : instabilite de derive E x B ===
grille n = 96 x 96, dx = 1.041667e-02
fond ionique n_i0       = 1.088623e+00  (moyenne de ne, periodique)
masse initiale          = 1.003275e+04
amplitude initiale      = 6.777566e-02

 pas            t      amplitude             mass
  10     1.079402   6.901051e-02   1.00327467e+04
  ...
 120    11.720626   1.293996e-01   1.00327467e+04

amplitude finale = 1.293996e-01  (initiale 6.777566e-02)
facteur de croissance = 1.9092
OK diocotron
```

Diagnostic cle : masse constante (1.00327467e+04 du pas 10 au pas 120), facteur de croissance
de l'amplitude **1.9092** (> 1, instabilite observee). Exit 0, ligne finale `OK diocotron`.

### `run.py` (non relance ici -- sorties decrites)

Console : tableau analytique (gamma par mode), tableau numerique mode 3/4/5 (gamma_num vs
analytique vs papier, ecart en %), puis le log d'evolution du gif (`OK repro_paper_2510_11808`).
Fichiers : `dispersion.png`, `amplitude.png`, `diocotron.gif`, `snapshots.png` dans
`out/diocotron/`. Le dossier `figures/` contient des copies de reference committees.

---

## 13. Generation figures/GIF

Produits **uniquement par `run.py`** (backend matplotlib `Agg`), ecrits dans `out/diocotron/` :

- `dispersion.png` : `gamma` vs mode azimutal `l`. Trois series : analytique (Petri, numpy),
  mesure `adc` (modes 3/4/5), cibles du papier.
- `amplitude.png` : `|c_l|(t)` (mode `l` de `phi`) en echelle semilog ; la phase lineaire montre
  la croissance exponentielle.
- `diocotron.gif` : evolution de la densite `n_e` (mode `l=4`, `delta=0.1`), 60 frames,
  `PillowWriter(fps=12)`. L'anneau developpe 4 lobes qui s'enroulent.
- `snapshots.png` : 4 instantanes de densite (`t` croissant) du meme run.

`band_instability.py` ne produit **aucune** figure (sortie texte seule).

---

## 14. Backends reellement supportes

- **CPU** : oui (chemin nominal, valide ici). Toutes les briques (`Scalar`, `ExB`,
  `BackgroundDensity`) et le multigrille geometrique de Poisson tournent en C++ host.
- **GPU** : non exerce par ces scripts. Le cas n'instancie aucun chemin device ; il ne demande
  pas non plus de compilation C++ (`needs=[]` pour `band_instability.py`, `["matplotlib"]` pour
  `run.py`).
- **MPI / multi-box** : non. Grille uniforme mono-bloc (le pendant AMR multi-patch est le cas
  distinct [`../diocotron_amr/`](../diocotron_amr/)).

Pile validee ici : build `adc_cpp/build-master/python`, Python 3.12 (anaconda3), macOS arm64.

---

## 15. Cout approximatif

- `band_instability.py` (n=96, 120 pas, CFL=0.4) : **~0.5 s wall** mesure
  (1.93 s user / 404 % CPU multi-thread du multigrille), trace memoire negligeable. Convient a la
  CI.
- `run.py` : **LONG**, hors CI. Trois mesures de taux a n=192 x 900 pas (avec Poisson resolu a
  chaque sous-pas SSPRK), plus un run d'evolution de 60 frames (720 pas) pour le gif, plus le
  rendu matplotlib. Plusieurs minutes (non chronometre ici, non relance par consigne).

---

## 16. Limites et differences avec les references

Honnetete sur le perimetre (consigne et manifeste) :

- **Modele reduit, pas Euler-Poisson complet.** Le cas integre un **scalaire `E x B`**
  (`Scalar` + `ExB` + `BackgroundDensity`), pas le systeme conservatif `(rho, rho u, rho v)` du
  papier ni son etage Schur. C'est la *limite de derive* visee par la Section 5.3, pas le systeme
  integral. Le pendant complet est [`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/)
  (`reproduction-candidate` PENDING, baseline cartesienne loin du papier).
- **Ce qui est reellement valide : la normalisation analytique.** La cible (valeurs propres de
  Petri, resolue en numpy) est retrouvee a **3 chiffres** : l'analytique du script donne
  `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687` (cf. README/SWEEP), pic correct a
  `l=4` (mode le plus instable). C'est un **benchmark de normalisation oracle**, pas la
  reproduction du schema elements-finis du papier.

  > Detail de tracabilite : les *cibles papier* codees en dur dans `run.py`
  > (`PAPER = {3: 0.772, 4: 0.911, 5: 0.683}`) different au 3e chiffre des valeurs reproduites par
  > l'analytique numpy (`0.772 / 0.912 / 0.687`) ; l'ecart est de l'ordre du millieme (transcription
  > vs resolution numpy), sans incidence sur le verdict.

- **Le volumes-finis `adc` SOUS-PREDIT le taux** a resolution moderee. A **n=192, MUSCL minmod**
  (configuration nominale), ecart mesure vs analytique :

  | mode `l` | analytique (Petri numpy) | `adc` (n=192, minmod) | ecart |
  |---|---|---|---|
  | 3 | 0.772 | 0.599 | **-22 %** |
  | 4 | 0.912 | 0.662 | **-27 %** |
  | 5 | 0.687 | 0.652 | **-5 %** |

  Effet de diffusion numerique de l'ordre modere. `SWEEP_RESULTS.md` montre que monter la
  resolution et l'ordre (vanleer, puis WENO5-Z + SSPRK3) reduit l'ecart sans le fermer
  completement ; il subsiste un residu `l`-dependant candidat a un **plancher structurel du bord
  d'anneau cartesien** (cf. memoire `project_adc_cpp_overshoot`, `docs/DIOCOTRON_GROWTH_RATE.md`).
  Le cas **capture bien l'instabilite** (croissance exponentielle, bon classement des modes,
  `l=4` dominant) ; la quantification fine releve de la resolution et de l'ordre.
- **Geometrie.** L'anneau tourne en ratios `0.15:0.20:0.40` (`L=1`) avec paroi conductrice
  circulaire Dirichlet ; l'analytique tourne aux memes ratios `6:8:16` mais a une echelle ou le
  lissage represente un anneau net. Le taux normalise etant invariant d'echelle, les deux sont
  comparables.

Ce cas ne doit donc **pas** etre presente comme une "reproduction Hoffart complete" : c'est un
**benchmark de normalisation du modele reduit `E x B`**, oracle Petri reproduit a 3 chiffres,
volumes-finis sous-predictif (-22 / -27 / -5 %).

---

## 17. Tests/CI associes

- **CI** : seul [`band_instability.py`](band_instability.py) tourne en CI
  (`cases_manifest.toml` -> `category="validation"`, `ci=true`, `needs=[]`). La CI
  (`.github/workflows/ci.yml`) ne lance que les cas legers `ci=true`. Les `assert` (masse
  conservee a `1e-6`, croissance de l'amplitude) rendent la CI rouge en cas de regression.
- **Hors CI** : [`run.py`](run.py) (`category="reproduction"`, `ci=false`, `needs=["matplotlib"]`)
  est long (figures + gif), lance a la main.
- **Etudes annexes** (hors manifeste, a la main) : [`sweep.py`](sweep.py) /
  [`SWEEP_RESULTS.md`](SWEEP_RESULTS.md) balaient ordre x resolution x mode (O1/O2/O5,
  n=128..512) pour quantifier la diffusion vs le plancher structurel.
- **Cas freres** dans le manifeste : `diocotron_amr/run.py` (meme physique sur AMR multi-patch,
  `validation`, CI), `diocotron_dsl/run.py` (diocotron ecrit entierement en formules DSL,
  etat bit-identique au natif, `validation`, CI, `needs=["cxx"]`).
