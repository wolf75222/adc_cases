# Diocotron Euler-Poisson magnetise (Hoffart et al., arXiv:2510.11808) -- DSL ADC

> CLASSIFICATION (manifeste) : `check_model.py` = **validation** (oracle analytique, CI).
> `run.py` = **reproduction-candidate** (PAS une reproduction). La reproduction
> quantitative des taux de croissance du papier (0.772 / 0.911 / 0.683) n'est
> **PAS** etablie : la table de validation reste PENDING/MEASURED avec des erreurs
> de -82 a -95 % sur la baseline cartesienne. `run.py` est LONG et hors CI ; ne pas
> le lancer pour valider.

Ce cas ecrit le systeme Euler-Poisson magnetise COMPLET (continuite + quantite de
mouvement + force de Lorentz, fermeture barotrope) avec le mini-DSL `adc.dsl.Model`,
compile le modele symbolique en C++ (backend `production`), puis le pousse dans les
volumes finis d'ADC. Aucun callback Python n'est evalue par cellule.

Honnetete (lire avant tout) : la baseline cartesienne carre + mur de Poisson
circulaire mesure ~ -82 a -95 % d'erreur (n=256 ET n=384, donc PAS un probleme de
resolution). Le suspect numero un est la GEOMETRIE (boite carree cartesienne avec
un anneau circulaire impose seulement par le mur du Poisson : le bord d'anneau
diffuse), pas le splitting. Restent aussi des ecarts de methode : splitting de
**Lie** (et non Strang), et resolution de Gauss **une fois par pas**. Voir
`adc_cpp/docs/HOFFART_FIDELITY.md` et `adc_cpp/docs/HOFFART_STEP_SEQUENCE.md`.

---

## 1. Objectif du cas

Documenter et instrumenter une **tentative** de reproduction du test diocotron de
la section 5.3 de Hoffart et al. (arXiv:2510.11808), en ecrivant le systeme
Euler-Poisson magnetise complet entierement en formules symboliques (`adc.dsl.Model`)
plutot qu'avec une brique native. Deux objets distincts vivent ici :

- `check_model.py` (validation, CI) : un **oracle analytique** qui verifie par
  `assert` que le modele symbolique genere le bon flux, la bonne source
  electrostatique/Lorentz, les bonnes valeurs propres et le bon second membre de
  Poisson. C'est le seul morceau du cas qui tourne en CI.
- `run.py` (reproduction-candidate, hors CI) : un harnais de **mesure
  pre-enregistre** qui fait croitre le diocotron, suit l'amplitude du mode
  azimutal et fitte un taux de croissance, pour le comparer aux cibles du papier.
  Ce harnais n'invente aucun nombre : tant qu'un run n'a pas tourne, la cellule
  est marquee `PENDING`.

L'objectif n'est PAS de declarer la reproduction faite ; il est de fournir un
chemin honnete (PDE verbatim, fenetres de fit verbatim, enregistrement trace) qui
permet de mesurer l'ecart au papier sans tricher.

## 2. Equations

Systeme Euler-Poisson magnetise, fermeture barotrope isotherme `p = theta rho`
(l'equation d'energie de l'Euler complet n'est PAS evoluee ; l'etat conservatif
est exactement `(rho, rho*u, rho*v)`) :

```text
d_t rho + div(m)                         = 0
d_t m   + div(m m^T / rho + p I)         = -rho grad(phi) + m x Omega
-Delta(phi)                              = alpha rho
p = theta rho
```

avec `m = (rho*u, rho*v)` la quantite de mouvement et `Omega = omega e_z`. En 2D,
le produit vectoriel de Lorentz donne `m x Omega = (omega*m_y, -omega*m_x)`.

Note de convention : ADC resout `Delta(phi) = rhs`, alors que le papier ecrit
`-Delta(phi) = alpha rho`. Le modele fixe donc `elliptic_rhs = -alpha*rho`
(cf. `model.py`, `m.elliptic_rhs(-alpha * rho)`).

## 3. Modele physique

Diocotron : une couche de charge (electrons) en anneau, derivant sous E x B dans un
fort champ magnetique uniforme, devient instable et developpe des lobes azimutaux.

Parametres du papier (Section 5.3), encodes dans `PaperParameters` (`model.py`) :

```text
rayon du disque       R   = 16
anneau                r0  = 6,  r1 = 8
rho_min / rho_max     = 1e-6 / 1
beta                  = 1e6
alpha = beta^2/rho_max = 1e12
omega (= |Omega|)     = beta^2 = 1e12
perturbation          delta = 0.1
modes                 l = 3, 4, 5
temps final           tf = 10
cibles gamma          l=3:0.772  l=4:0.911  l=5:0.683
```

Temperature / fermeture : le papier definit `p = theta rho` mais ne donne pas de
valeur numerique pour `theta`. Ce cas utilise donc par defaut la **limite froide
`theta = 0`** (vitesse du son nulle, flux de pression nul) et enregistre la valeur
choisie dans `metadata.json`. `--temperature` permet d'en tester une autre.

## 4. Methode numerique

Chemin de reference `system-schur` (le plus proche du papier) :

- **Volumes finis** `adc.FiniteVolume`, reconstruction **WENO5(-Z)**
  (`limiter="weno5"`), flux de **Rusanov** (`riemann="rusanov"`), variables
  **conservatives** (`variables="conservative"`).
- **Stage hyperbolique** explicite **SSPRK3** (`adc.Explicit(ssprk3=True)`).
- **Stage source** electrostatique/Lorentz par **Schur condense**
  `adc.CondensedSchur(theta=0.5, alpha=...)`, branche via `adc.Split(...)`.
- **Poisson** : `set_poisson(rhs="composite", solver="geometric_mg",
  bc="dirichlet", wall="circle", wall_radius=R)` : multigrille geometrique, mur
  de Dirichlet sur un cercle R=16 embarque dans la grille carree.
- Vitesse de derive initiale calculee a partir de la premiere resolution de
  Poisson (etat de drift du papier).

Limites de methode (honnetes, non corrigees ici) :

- ADC applique transport **puis** source en composition de **Lie** (premier
  ordre, Godunov), alors que le papier utilise **Strang**.
- La force/Poisson (Gauss) est re-resolue **une fois par pas**, pas a chaque
  sous-etage.
- La geometrie reste une **boite carree cartesienne** ; l'anneau circulaire n'est
  impose que par le mur du Poisson, ce qui diffuse le bord d'anneau (suspect
  principal de l'ecart au papier).

Chemin alternatif `amr-imex` (experimental, voir Section 16) : meme PDE, mais
source IMEX **backward-Euler cell-local** (CondensedSchur non implemente sur AMR),
sur `adc.AmrSystem` (AMR dynamique, Kokkos, MPI).

## 5. Architecture ADC utilisee

Briques ADC reellement invoquees (toutes verifiees presentes dans le binding
`adc` / `adc/dsl.py`) :

| Brique ADC | Role dans le cas | Ou |
|---|---|---|
| `adc.dsl.Model` | modele symbolique (vars, flux, valeurs propres, source, elliptic_rhs) | `model.py` |
| `Model.compile(backend="production", target=...)` | genere + compile le `.so` C++ | `run.py:compile_model` |
| `adc.System` | grille uniforme, Poisson, stepping | `run.py:build_uniform` |
| `adc.AmrSystem` | grille AMR (chemin experimental) | `run.py:build_amr` |
| `adc.FiniteVolume(limiter,riemann,variables)` | discretisation spatiale WENO5 + Rusanov | `run.py` |
| `adc.Split(hyperbolic, source)` | composition transport/source | `run.py:build_uniform` |
| `adc.Explicit(ssprk3=True)` | integrateur hyperbolique | `run.py:build_uniform` |
| `adc.CondensedSchur(theta, alpha)` | stage source condense par Schur | `run.py:build_uniform` |
| `adc.IMEX(substeps)` | stage temporel AMR | `run.py:build_amr` |
| `System.set_poisson(...)` | Poisson MG Dirichlet, mur cercle | `run.py` |
| `System.set_magnetic_field(B_z)` | B_z uniforme (= omega) requis par CondensedSchur | `run.py:build_uniform` |
| `System.set_primitive_state` / `set_density` | conditions initiales | `run.py` |
| `System.solve_fields` / `step` / `time` / `potential` / `density` | execution + sorties | `run.py` |

Le DSL `Model` expose ici : `conservative_vars`, `primitive`, `primitive_vars`,
`conservative_from`, `flux(x=,y=)`, `eigenvalues(x=,y=)`, `aux`, `param`,
`source`, `elliptic_rhs`, `check`, `compile` (tous confirmes dans
`adc_cpp/.../adc/dsl.py`).

Variante de source (cf. `magnetic_euler_poisson_model(..., source=)`) :

- `source="schur"` : la source DSL locale est **nulle** (`[0, 0, 0]`) ; c'est
  `CondensedSchur` qui porte tout le stage electrostatique/Lorentz. Sinon il
  serait avance deux fois. Cible de compilation `target="system"`.
- `source="local"` : la source complete `[-rho*grad_x + omega*my,
  -rho*grad_y - omega*mx]` est emise dans le `.so` ; utilisee par le chemin AMR
  (IMEX cell-local). Cible `target="amr_system"`.

Note : le paquet partage `adc_cases/models.py` definit un helper `euler_poisson`
(brique native, base sur un signe attractif/repulsif) ; ce cas **ne l'utilise
pas** : il construit son propre modele symbolique magnetise dans `model.py`.

## 6. Carte des fichiers

```text
hoffart_euler_poisson_dsl/
  README.md            ce fichier
  model.py             modele symbolique adc.dsl + PaperParameters, IC, drift
  check_model.py       oracle analytique CI (assert flux/source/eigen/rhs)
  run.py               harnais de reproduction-candidate (LONG, hors CI)
  results.py           emetteur d'enregistrements de mesure (pur Python, self-test CI)
  NORMALIZATION.md     etude de normalisation 2pi/rhobar du diocotron REDUIT (autre modele)
  diag/
    diag_polar_omega.py  diagnostic du chemin polaire ExB REDUIT (hors manifeste)
    petri_eigenvalue.py  spectre analytique Petri (hors manifeste)
    petri_eigenvalue.md  notes du spectre
```

Dependances du paquet partage : `adc_cases/common/io.py` (`case_output_dir`,
sorties sous `<depot>/out/`). Le reste de `common/` (grid, initial_conditions,
checks, native) n'est pas importe par ce cas.

## 7. Prerequis

- Module `adc` construit (binding C++/pybind11) : ici
  `adc_cpp/build-master/python`. Le chemin AMR/MPI vise un build Kokkos
  (`build-kokkos/python`).
- `needs = ["matplotlib"]` (manifeste) pour `run.py` : figures, panneaux
  schlieren et GIF (`matplotlib`, `PillowWriter`).
- Un compilateur C++ (C++20) disponible pour `Model.compile(backend="production")`
  (generation + compilation du `.so` a la volee).
- `numpy`. MPI (`mpirun`) seulement pour le chemin `amr-imex` multi-rang.
- `check_model.py` n'a besoin que de `numpy` + `adc.dsl` (pas de build complet,
  pas de matplotlib).

## 8. Commande exacte

Oracle analytique (leger, CI, A LANCER pour valider le modele) :

```bash
cd /private/tmp/adc_cases-readmes/hoffart_euler_poisson_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 check_model.py
```

Auto-test du harnais de mesure (pur Python, CI) :

```bash
PYTHONPATH=/private/tmp/adc_cases-readmes \
  /opt/homebrew/anaconda3/bin/python3.12 hoffart_euler_poisson_dsl/results.py
```

Reproduction-candidate `system-schur` (LONG, hors CI -- NE PAS lancer pour valider) :

```bash
cd /chemin/vers/adc_cases
PYTHONPATH=/chemin/vers/adc_cpp/build-master/python python hoffart_euler_poisson_dsl/run.py \
    --engine system-schur --n 384 --t-end 10 --dt 1e-3
```

Fumee de compilation + execution courte :

```bash
python hoffart_euler_poisson_dsl/run.py --quick      # n=48, t_end=0.02, mode 3
python hoffart_euler_poisson_dsl/run.py --compile-only
```

Chemin AMR/Kokkos/MPI (experimental, exige l'aveu explicite) :

```bash
PYTHONPATH=/chemin/vers/adc_cpp/build-kokkos/python \
  mpirun -np 4 python hoffart_euler_poisson_dsl/run.py \
    --engine amr-imex --acknowledge-amr-approximation \
    --n 192 --t-end 10 --dt 1e-3 --distribute-coarse
```

Diagnostic de normalisation du modele REDUIT (autre modele, voir Section 16) :

```bash
PYTHONPATH=/chemin/vers/adc_cpp/build-master/python \
  python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py 128
```

## 9. Explication du code par etapes

`model.py` (modele symbolique) :

1. `PaperParameters` : dataclass figee des parametres du papier ; proprietes
   derivees `length = 2R`, `alpha = beta^2/rho_max`, `omega = beta^2`.
2. `magnetic_euler_poisson_model(params, source)` : construit `dsl.Model`,
   declare les conservatives `(rho, rho_u, rho_v)`, les primitives
   `u = mx/rho`, `v = my/rho`, `p = theta*rho`, pose le flux et les valeurs
   propres `(u-c, u, u+c)` / `(v-c, v, v+c)` avec `c = sqrt(theta)`, declare les
   aux `phi, grad_x, grad_y`, choisit la source (nulle si `schur`, complete si
   `local`), fixe `elliptic_rhs = -alpha*rho`, puis `check()`.
3. `paper_initial_density(n, mode, params)` : densite en anneau (eq. 35),
   `rho_min` partout, anneau `[r0,r1]` perturbe par `1 - delta + delta*sin(l*angle)`.
4. `drift_velocity_from_potential(phi, params)` : derive E x B initiale
   `u = -d_y phi / omega`, `v = +d_x phi / omega`, mise a zero hors du disque R.

`run.py` (harnais) :

1. `verify_paper_windows(PAPER_FIT_WINDOWS)` au demarrage : **verrou
   pre-enregistre** qui leve si les fenetres de fit ne sont pas exactement les
   fenetres verbatim du papier (Fig. 5.4). Empeche toute fenetre adaptative dans
   la comparaison du modele complet.
2. `compile_model` : compile le `.so` une fois, sous verrou `flock` (sûr en
   multi-rang MPI : les autres rangs reutilisent le cache DSL).
3. `build_uniform` (engine `system-schur`) : `adc.System`, `set_poisson`,
   `set_magnetic_field(omega)` (requis avant le stage CondensedSchur),
   `add_equation` (FiniteVolume + Split SSPRK3/CondensedSchur). Initialise
   `rho` a vitesse nulle, `solve_fields`, calcule la derive depuis le potentiel,
   re-pose l'etat primitif avec `(u0,v0)`, `solve_fields`.
4. `build_amr` (engine `amr-imex`) : `adc.AmrSystem`, `add_equation`
   (FiniteVolume + IMEX), `set_poisson`, `set_refinement`, `set_density`
   (densite seule -- d'ou la quantite de mouvement initiale nulle).
5. `run_mode` : boucle `sim.step(dt)` jusqu'a `t_end`, echantillonne
   l'amplitude du mode tous les `sample_every` pas, collecte snapshots et frames
   GIF aux instants du papier, leve `FloatingPointError` si phi/amplitude non
   finis.
6. `sample_circle` + `mode_amplitude` : interpole phi sur le cercle interne
   `r0`, prend la FFT azimutale, renvoie `2*|c_l|`.
7. `fit_growth` : pente de `log(amplitude)` sur la **fenetre verbatim** du mode.
8. `write_summary` : ecrit `growth_rates.csv/.png`, `metadata.json`, et un
   **enregistrement de mesure** par mode via `results.build_record` / `write_records`.

`results.py` (mesure honnete, pur Python) : `verify_paper_windows`,
`engine_label` (refuse tout moteur inconnu, ne melange jamais le label reduit),
`err_pct`, `build_record`, `write_records`. Marque `PENDING` toute valeur non
mesuree. Self-test `python results.py`.

## 10. Conditions initiales

- Densite (`paper_initial_density`) : `rho = rho_min = 1e-6` partout ;
  dans l'anneau `r0 <= r <= r1`, `rho = rho_max*(1 - delta + delta*sin(l*theta))`
  avec `delta = 0.1`, perturbation azimutale du mode `l`.
- Vitesse (chemin `system-schur` uniquement) : derive E x B initiale calculee a
  partir de la **premiere** resolution de Poisson :
  `u0 = -d_y phi / omega`, `v0 = +d_x phi / omega`, annulee hors du disque R=16.
  C'est l'etat de drift du papier.
- Chemin `amr-imex` : seule la densite est initialisee (`set_density`) ; la
  facade AMR n'expose pas l'etat conservatif/primitif complet, donc la quantite
  de mouvement **demarre a zero** et relaxe vers la derive (difference connue).

## 11. Invariants et assertions

- `check_model.py` (oracle CI) : `np.testing.assert_allclose` que le flux x/y, la
  source `[0, -rho*gx + omega*my, -rho*gy - omega*mx]`, le `elliptic_rhs`
  (`= -alpha*rho`) et les vitesses d'onde max `> 0` coincident avec les formules
  analytiques (parametres test `beta=3, theta=0.25`).
- `run.py` : `verify_paper_windows` (AssertionError si une fenetre adaptative se
  glisse) ; garde MPI (`system-schur` interdit multi-rang) ; aveu obligatoire
  `--acknowledge-amr-approximation` pour `amr-imex` ; `FloatingPointError` si phi
  ou amplitude deviennent non finis ; `RuntimeError` si `max_steps` depasse.
- `results.py` : self-test (fenetres verbatim, labels moteur, `err_pct` exact,
  PENDING preserve, round-trip CSV/JSON, le facteur 2pi du chemin reduit ne fuit
  pas).

Note : ce cas ne verifie PAS une conservation globale ; l'invariant central est
l'egalite analytique du modele genere (CI) et le verrou des fenetres de fit.

## 12. Sorties attendues

- `check_model.py` :
  `OK Hoffart DSL: flux, Lorentz/electric source, eigenvalues and Poisson rhs`
  (verifie, EXIT=0).
- `results.py` :
  `OK results.py: fenetres verbatim, labels moteur, err_pct, record brut, PENDING, IO`.
- `run.py` : sous `out/hoffart_euler_poisson_dsl_<engine>/`, pour chaque mode
  `amplitude.csv` + `amplitude.png` (log + courbe theorique du papier + fenetre de
  fit), un panneau schlieren 3x3 aux instants du papier, un GIF
  `diocotron_l<l>.gif`, plus au niveau du cas `growth_rates.csv/.png`,
  `metadata.json` et `measurement_record.csv/.json`.

Resultat de mesure **honnete** : les taux de croissance bruts mesures sur la
baseline cartesienne sont LOIN du papier (-82 a -95 %), voir Section 16. Ne pas
attendre une concordance.

## 13. Generation figures/GIF

Tout dans `write_mode_outputs` / `write_summary` (`run.py`), backend matplotlib
`Agg` :

- `amplitude.png` : `|c_l(t)|/|c_l(0)|` en semilog, courbe `exp(gamma_papier t)`
  ancree au milieu de la fenetre, bande bleue = fenetre de fit verbatim.
- `snapshots.png` : panneau 3x3 d'images **schlieren** (`log1p(20*|grad rho|)`)
  masquees hors du disque R, aux fractions de temps du papier.
- `diocotron_l<l>.gif` : animation schlieren via `FuncAnimation` + `PillowWriter`
  (12 fps), `--gif-frames` (defaut 80), desactivable par `--no-gif`.
- `growth_rates.png` : gamma numerique vs gamma papier par mode.

## 14. Backends reellement supportes

- `--engine system-schur` (defaut) : `adc.System`, **mono-rang** (le runner leve
  si `mpi_size > 1`). Backend etiquete `kokkos-serial`. C'est le chemin de
  reference vers une discussion de fidelite.
- `--engine amr-imex` : `adc.AmrSystem` avec AMR dynamique + Kokkos + MPI
  (`mpirun -np N`), backend `kokkos-serial` ou `kokkos-mpi-<N>`. **Experimental**
  (voir Section 16), exige `--acknowledge-amr-approximation`.
- Modele C++ compile a la volee (`backend="production"`, `target="system"` ou
  `"amr_system"`) : exige un compilateur C++20. `target="amr_system"` n'existe
  qu'en backend `production` (seul chemin `.so` AMR natif).
- `check_model.py` et `results.py` : pur CPU/Python, aucun build complet requis
  (results.py n'importe meme pas `adc`).

## 15. Cout approximatif

- `check_model.py` : << 1 s (mesure : sortie immediate, EXIT=0). 2x2 cellules
  analytiques, aucun build du coeur.
- `results.py` : << 1 s (self-test pur Python).
- `--quick` (n=48, t_end=0.02, mode 3) : domine par la compilation du `.so`
  (quelques secondes a quelques dizaines de secondes selon la toolchain), puis
  ~20 pas. Fumee, pas une mesure.
- `run.py` complet (`--n 384 --t-end 10 --dt 1e-3`, 3 modes) : **LONG**. ~10000
  pas de temps par mode sur une grille 384x384, plus un Poisson MG par pas et la
  generation des figures/GIF. C'est pourquoi il est hors CI ; ne pas le lancer
  pour valider.

(Cout de `run.py` non mesure ici, conformement a la consigne : reproduction LONG,
non lancee.)

## 16. Limites et differences avec les references

Limites du chemin `system-schur` (reference) face a Hoffart et al. :

1. **Reproduction quantitative PAS etablie.** Baseline cartesienne MEASURED :
   l=3 -95 %, l=4 -94/-93 %, l=5 -82 % (n=256 et n=384). L'erreur **n'ameliore
   pas** de n=256 a n=384 : ce n'est donc PAS un probleme de resolution.
2. **Geometrie suspecte (suspect principal).** Boite carree cartesienne + anneau
   circulaire impose seulement par le mur du Poisson : le transport reste sur la
   grille carree et le bord d'anneau diffuse. Le taux analytique ne depend que de
   `l, omega_d, r0, r1, R`, donc l'ecart n'est pas le splitting.
3. **Splitting de Lie, pas Strang.** Transport puis source, premier ordre.
4. **Poisson une fois par pas** (et non a chaque sous-etage).
5. `theta` non donne par le papier : limite froide `theta=0` par defaut.

Limites supplementaires du chemin `amr-imex` (experimental, jamais une
reproduction) :

1. `CondensedSchur` non implemente sur `AmrSystem` -> source IMEX backward-Euler
   cell-local.
2. La facade AMR n'initialise que la densite -> quantite de mouvement initiale
   **nulle** (pas l'etat de drift du papier).
3. Mur circulaire impose par le Poisson, transport toujours cartesien.

**A propos de `NORMALIZATION.md` / `diag/diag_polar_omega.py` (a ne PAS
confondre).** Cette etude valide un facteur de normalisation global `2pi/rhobar`
sur un chemin **polaire ExB scalaire**, qui resout un **modele REDUIT et
DIFFERENT** (derive ExB scalaire, type Petri, **sans** quantite de mouvement),
PAS le systeme Euler-Poisson complet ci-dessus. Sur ce modele reduit, seul `l=4`
colle exactement (l=3 +26 %, l=5 oscillant). Ce facteur `2pi/rhobar` appartient
**UNIQUEMENT** au chemin reduit (`engine = reduced-ExB`) et n'est **jamais**
applique au modele complet (`engine = full-system-schur`, pente BRUTE). Ce n'est
donc ni un contre-exemple ni une reproduction du modele complet.

Reference de fidelite detaillee : `adc_cpp/docs/HOFFART_FIDELITY.md`,
`adc_cpp/docs/HOFFART_STEP_SEQUENCE.md`.

Table de validation (verbatim, aucune cellule inventee ; `system-schur` =
modele complet pente BRUTE sans 2pi ; `reduced-ExB` = chemin reduit
`NORMALIZATION.md`, NE PAS confondre ; `amr-imex` reste experimental) :

| mode | n | gamma_numeric | gamma_paper | err_pct | fit_window | engine | dt | splitting | schur |
|------|-----|---------------|-------------|---------|--------------------|--------------|--------|------------------|----------------------|
| 3 | 256 | 0.0372  | 0.772 | -95.2   | [0.40,0.70] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 4 | 256 | 0.0489  | 0.911 | -94.6   | [0.60,0.75] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 5 | 256 | 0.1211  | 0.683 | -82.3   | [1.15,1.35] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 3 | 384 | 0.0385  | 0.772 | -95.0   | [0.40,0.70] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 4 | 384 | 0.0613  | 0.911 | -93.3   | [0.60,0.75] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 5 | 384 | 0.1257  | 0.683 | -81.6   | [1.15,1.35] paper  | full-system-schur (cart-square) | 1e-3 | Lie | CondensedSchur t=0.5 |
| 3 | 512 | PENDING | 0.772 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 4 | 512 | PENDING | 0.911 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 5 | 512 | PENDING | 0.683 | PENDING | PENDING            | system-schur | 1e-3   | Lie              | CondensedSchur t=0.5 |
| 3 | 128 | 0.9712  | 0.772 | +25.8   | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 4 | 128 | 0.9127  | 0.911 | +0.2    | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 5 | 128 | 0.4820  | 0.683 | -29.4   | [2.12, 12.58]      | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 3 | 192 | 0.9713  | 0.772 | +25.8   | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 4 | 192 | 0.9100  | 0.911 | -0.1    | full exp window    | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 5 | 192 | 0.8658  | 0.683 | +26.8   | [2.12, 5.96]       | reduced-ExB  | CFL 0.4| n/a (ExB explic) | none (NoSource)      |
| 3 | 192 | PENDING | 0.772 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |
| 4 | 192 | PENDING | 0.911 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |
| 5 | 192 | PENDING | 0.683 | PENDING | PENDING            | amr-imex     | 1e-3   | Lie (IMEX local) | none (IMEX local)    |

Tant que les cellules `system-schur` (n=512) ne sont pas remplies, aucune
affirmation que le modele complet reproduit le papier n'est permise.

## 17. Tests/CI associes

Selon `cases_manifest.toml` :

- `hoffart_euler_poisson_dsl/check_model.py` : categorie **validation**,
  `ci = true`, `needs = []`. C'est l'**oracle analytique** lance en CI (verifie
  flux, source Lorentz/electrique, valeurs propres et rhs de Poisson par assert).
- `hoffart_euler_poisson_dsl/run.py` : categorie **reproduction-candidate**,
  `ci = false`, `needs = ["matplotlib"]`. **Hors CI** (LONG, vise arXiv:2510.11808,
  table PENDING/MEASURED). Baseline cartesienne loin du papier, geometrie suspecte
  (cf. `adc_cpp/docs/HOFFART_FIDELITY.md`).

La CI ne lance que les cas legers `ci = true` ; `results.py` se self-teste aussi
en pur Python (`python results.py`) sans compiler le coeur. Les scripts `diag/`
ne sont PAS des cas du manifeste : ils se lancent a la main (etudes de
normalisation / spectre, hors CI).
