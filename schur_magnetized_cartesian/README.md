# Cas `schur_magnetized_cartesian` : effet TEMPOREL du complement de Schur

Etude de TIMING (balayage du plus grand pas de temps stable) mesurant l'effet de
l'etage source condense par Schur (`adc.CondensedSchur`, pile #118-128) sur un
fluide isotherme magnetise CARTESIEN RAIDE, face a l'integration EXPLICITE de la
meme source. Script : [`run.py`](run.py).

Categorie manifeste : `experimental` (`ci = false`, `needs = ["cxx"]`). C'est un
PROTOTYPE de mesure, pas une reproduction d'un resultat publie : aucune affirmation
de fidelite a un papier n'est faite ici. Le pendant qui VISE le papier Hoffart
(arXiv:2510.11808) est le cas separe `hoffart_euler_poisson_dsl` (et reste
lui-meme `reproduction-candidate` PENDING).


## 1. Objectif du cas

Quantifier le gain en pas de temps que procure l'etage source condense par Schur
sur la raideur cyclotronique d'un fluide magnetise. Sur le MEME modele isotherme
magnetise (memes equations, flux, valeurs propres, Poisson que le cas
`magnetic_isothermal_dsl`), on compare deux traitements de la source de Lorentz :

- variante `local` (« explicit ») : la force de Lorentz `m x Omega` est emise dans
  le C++ genere par le DSL et avancee EXPLICITEMENT apres le transport ;
- variante `schur` : la source locale est nulle ; l'etage electrostatique/Lorentz
  est avance IMPLICITEMENT par l'etage condense `CondensedSchur`.

Pour chaque variante, on cherche par balayage geometrique le plus grand `dt`
stable jusqu'a `t_end`, et on rapporte `dt`, le produit `dt * omega_c` et le gain.
Resultat attendu : l'explicite plafonne a `dt * omega_c ~ O(1)` (borne de la
rotation cyclotronique explicite) ; le Schur va bien au-dela (le pas n'est plus
limite que par le transport hyperbolique).


## 2. Equations

Memes equations que `magnetic_isothermal_dsl` (fluide d'Euler isotherme,
electrostatique + Lorentz, fermeture `p = cs2 rho`) :

    d_t rho + div(m)                              = 0
    d_t m   + div(m m^T / rho + cs2 rho I)        = q rho (-grad phi) + m x Omega
    Delta phi                                     = q rho

avec `m = (rho u, rho v)`, `Omega = omega_c e_z`, `omega_c = B_z`. La rotation de
Lorentz `m x Omega` se projette en 2D en `(+B_z my, -B_z mx)` sur les deux
composantes de quantite de mouvement : elle fait TOURNER la quantite de mouvement
sans modifier la masse ni l'energie cinetique. La charge `q = -1` est cuite dans
le modele ; son facteur global est absorbe dans `alpha` / `B_z`.

Le terme RAIDE est cette rotation cyclotronique. Integree EXPLICITEMENT elle
impose la borne de stabilite `dt * omega_c < O(1)` : a `omega_c` grand, le pas
explicite s'effondre. L'etage `CondensedSchur` assemble et resout l'operateur
condense `A = I + theta^2 dt^2 alpha rho B^{-1}` (B portant la rotation de
Lorentz) et avance la source implicitement, supprimant cette borne.


## 3. Modele physique

Fluide d'Euler isotherme magnetise, ecrit UNE SEULE FOIS en `adc.dsl.Model`
(fonction `magnetized_model(local_source, cs2)`), instancie en deux VARIANTES qui
partagent flux / valeurs propres / Poisson et ne different QUE par leur source :

- `local_source=True` (tag `local`) : `m.source([0, q rho (-gx) + bz my,
  q rho (-gy) - bz mx])` : electrostatique `q rho E` (E = -grad phi) + Lorentz
  `(+B_z my, -B_z mx)`. C'est la variante EXPLICITE de reference.
- `local_source=False` (tag `schur`) : `m.source([0*rho, 0*mx, 0*my])` : source
  LOCALE NULLE. L'etage condense `CondensedSchur` porte la source complete ; la
  laisser localement serait l'avancer deux fois.

Champs auxiliaires lus par le modele : `grad_x`, `grad_y` (gradient du potentiel)
et `B_z` (canal `adc::Aux` etendu, indice canonique au-dela de phi/grad). `B_z`
est peuple depuis Python par `sim.set_magnetic_field(omega_c * ones(n, n))` : ici
un CHAMP CONSTANT `omega_c` partout.

Parametres physiques (defauts) : `cs2 = 1e-4`, `q = -1`, `alpha = 1`,
`omega_c (B_z) = 1000`. La vitesse du son est prise LENTE (`cs2 = 1e-4`) a dessein :
le pas de transport explicite `~ h / cs` reste large devant `1 / omega_c`, ce qui
ISOLE la raideur de la SOURCE (le transport n'est pas limitant).


## 4. Methode numerique

- Spatial : volumes finis `adc.FiniteVolume(limiter="minmod", riemann="rusanov",
  variables="conservative")` (limiteur minmod, flux de Riemann Rusanov,
  reconstruction sur variables conservatives).
- Temporel (transport) : `adc.Explicit()`. Sur le backend AOT effectivement
  utilise, cela correspond a SSPRK2 (l'ABI AOT n'expose PAS SSPRK3 — cf. section
  14). Le schema RK du transport n'influe PAS sur la conclusion temporelle : le
  facteur mesure vient de la SOURCE, pas du transport.
- Poisson : `sim.set_poisson(rhs="charge_density", solver="geometric_mg",
  bc="periodic")` (multigrille geometrique, conditions periodiques).
- Etage source (variante `schur`) : `CondensedSchurSourceStepper` (#126), branche
  via le hook prive `sim._s.set_source_stage("plasma", "electrostatic_lorentz",
  theta, alpha)`. C'est l'integration IMPLICITE condensee de la source, jouee
  APRES le transport. Deux valeurs de `theta` sont mesurees : `theta = 0.5`
  (Crank-Nicolson, marginalement stable pour la rotation) et `theta = 1.0` (Euler
  retrograde, inconditionnellement stable).
- Domaine : grille `n x n` periodique, `L = 1` (defaut `n = 16`, soit `h = 0.0625`).

Mesure de stabilite (`is_stable`) : une simulation est dite stable a un `dt`
donne si la densite reste finie, bornee (`|rho|max <= 1e3`) et quasi-positive
(`rho.min >= -1e-2`) a chaque pas jusqu'a `t_end`. `largest_stable_dt` balaie un
quadrillage geometrique `dt = 10^(e/4)` pour `e` de -16 a +4 (quart de decade,
plafonne a `dt_max = 0.5`) et retient le plus grand `dt` stable.


## 5. Architecture ADC utilisee

- DSL declaratif : `adc.dsl.Model` ecrit la physique en expressions symboliques
  (`conservative_vars`, `primitive`, `aux`, `param`, `flux`, `eigenvalues`,
  `source`, `primitive_vars`, `conservative_from`, `elliptic_rhs`, `check`), puis
  `.compile(<so>, include, backend="aot")` genere le C++, le compile et produit
  un bloc compile installable.
- `adc.System(n, L, periodic=True)` : maillage cartesien periodique.
- `sim.add_equation("plasma", model=compiled, spatial=..., time=...)` : installe
  le bloc DSL compile ; `add_equation` elargit le canal `aux` partage pour porter
  `B_z`.
- `sim.set_poisson(...)`, `sim.set_magnetic_field(...)`,
  `sim.set_primitive_state("plasma", rho=, u=, v=)`, `sim.solve_fields()`,
  `sim.step(dt)`, `sim.density("plasma")` : pilotage standard du System.
- HOOK PRIVE : `sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta,
  alpha)`. C'est l'ABI C++ directe de l'etage condense. Le descripteur
  haut-niveau `adc.CondensedSchur` (utilise via `adc.Split`) n'est PAS employe ici
  (voir section 14) ; on appelle le binding sous-jacent, qui execute le MEME C++
  (`CondensedSchurSourceStepper`).
- Utilitaires `adc_cases` : seuls `common.io.case_output_dir` (repertoire de
  sortie `out/<cas>/`, hors source) et `common.native.adc_include` (localise les
  en-tetes du coeur pour la compilation DSL) sont importes. Le cas n'utilise NI
  `models.py`, NI `recipes.py`, NI les autres modules `common/`.


## 6. Carte des fichiers

| Fichier | Role |
| --- | --- |
| `schur_magnetized_cartesian/run.py` | script unique : modele DSL `local`/`schur`, build, mesure de `dt` stable, sortie console + CSV |
| `schur_magnetized_cartesian/README.md` | ce document |
| `adc_cases/common/io.py` | `case_output_dir(CASE)` -> `out/schur_magnetized_cartesian/` |
| `adc_cases/common/native.py` | `adc_include()` -> dossier `include/` du coeur adc_cpp pour le codegen DSL |
| `cases_manifest.toml` | declare le cas `experimental`, `ci = false`, `needs = ["cxx"]` |
| `out/schur_magnetized_cartesian/schurmag_local.so` | C++ DSL compile (variante explicite), genere a l'execution |
| `out/schur_magnetized_cartesian/schurmag_schur.so` | C++ DSL compile (variante Schur, source locale nulle), genere a l'execution |
| `out/schur_magnetized_cartesian/dt_stable.csv` | table des resultats (ecrite avec `--csv`) |

Il n'y a PAS de `check_model.py` ni de `band_instability.py` dans ce cas.


## 7. Prerequis

- Le module C++ `adc` (bindings pybind11 d'adc_cpp) sur le `PYTHONPATH` — ici le
  build `build-master/python`.
- Le paquet `adc_cases` importable (depot sur le `PYTHONPATH`, ou installe).
- Un compilateur C++20 (`needs = ["cxx"]`) : le DSL compile les deux modeles a la
  volee en backend `aot` au lancement. `numpy` est requis.
- Les en-tetes du coeur adc_cpp accessibles via `adc_include()` (priorite a
  `$ADC_INCLUDE`).


## 8. Commande exacte

NE PAS lancer en CI (long, experimental). Pour l'executer a la main :

    cd /private/tmp/adc_cases-readmes/schur_magnetized_cartesian \
      && PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-readmes \
         /opt/homebrew/anaconda3/bin/python3.12 run.py --csv

Forme generique :

    PYTHONPATH=<adc_cpp>/build-master/python:<depot> python run.py --csv

Options (toutes optionnelles) :

| Option | Defaut | Effet |
| --- | --- | --- |
| `--n` | 16 | taille de grille `n x n` |
| `--L` | 1.0 | longueur du domaine |
| `--omega-c` | 1e3 | frequence cyclotron `B_z` (terme raide de Lorentz) |
| `--cs2` | 1e-4 | carre de la vitesse du son (petit => transport non limitant) |
| `--alpha` | 1.0 | couplage electrostatique de l'etage source condense |
| `--t-end` | 1.0 | temps final du balayage |
| `--csv` | (off) | ecrit `out/schur_magnetized_cartesian/dt_stable.csv` |


## 9. Explication du code par etapes

1. `magnetized_model(local_source, cs2)` construit le `dsl.Model` (tag `local` ou
   `schur`) : variables conservatives `rho, rho_u, rho_v`, primitives `u, v`, aux
   `grad_x/grad_y/B_z`, params `cs2/charge`, flux isotherme, valeurs propres
   `u +- cs` / `v +- cs`, puis la source (complete si `local_source`, NULLE sinon),
   le layout primitif, l'inverse prim->cons, le rhs elliptique `q rho`, et
   `m.check()`.
2. `main()` appelle `adc_include()` et `case_output_dir(CASE)`, puis COMPILE les
   deux modeles en `aot` : `schurmag_local.so` (explicite) et `schurmag_schur.so`
   (source nulle).
3. Affiche le pas de transport approche (`0.5 h / sqrt(cs2 + 0.5)`) et la borne
   explicite de la source `1/omega_c`, pour contextualiser le balayage.
4. `largest_stable_dt(...)` est appele trois fois : variante `local` (`schur=False,
   theta=0.5`), variante `schur` (`schur=True, theta=0.5`), variante `schur`
   (`schur=True, theta=1.0`).
5. `build(...)` (appele a chaque essai de `dt`) : cree le `System` periodique,
   `set_poisson(charge_density, geometric_mg, periodic)`, `add_equation("plasma",
   ...)`, `set_magnetic_field(omega_c)` ; SI `schur`, branche
   `sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta, alpha)` ;
   fixe l'etat primitif initial et `solve_fields()`.
6. `is_stable(...)` integre `ceil(t_end/dt)` pas (au moins 2) et controle a chaque
   pas la finitude, la borne et la positivite de la densite.
7. La sortie console tabule `dt_stable` et `dt * omega_c` pour chaque variante,
   puis le gain Schur/explicite. Avec `--csv`, ecrit
   `dt_stable.csv` (colonnes `method, dt_stable, dt_times_omega_c,
   gain_over_explicit`).


## 10. Conditions initiales

`initial_state(n)` : densite perturbee par un cosinus le long de x et vitesse
oblique constante.

    x    = (arange(n) + 0.5) / n
    rho0 = 1 + 0.05 * cos(2 pi x)        (constant en y)
    u0 = v0 = 0.5                        (vitesse oblique partout)

L'etat est fixe en variables PRIMITIVES via `sim.set_primitive_state("plasma",
rho=rho0, u=u0, v=v0)`. La vitesse oblique `u = v = 0.5` rend la rotation de
Lorentz active des le depart (les deux composantes de quantite de mouvement sont
non nulles). Champ magnetique constant `B_z = omega_c` partout.


## 11. Invariants et assertions

Ce cas N'UTILISE PAS d'`assert` : ce n'est pas un test de validation mais une
MESURE de stabilite. Le critere de stabilite est implicite dans `is_stable` : a un
`dt` donne, la simulation est ACCEPTEE si, a chaque pas jusqu'a `t_end`, la
densite verifie

    np.isfinite(rho).all()  ET  |rho|.max() <= 1e3  ET  rho.min() >= -1e-2

Sinon le `dt` est rejete. Aucune autre quantite (masse, energie) n'est verifiee :
l'objectif est le seuil de stabilite temporelle, pas la conservation physique.


## 12. Sorties attendues

Sortie console (un tableau + un gain). A `omega_c = 1000`, `cs2 = 1e-4`,
`alpha = 1`, `n = 16`, `L = 1` (h = 0.0625), `t_end = 1.0`, transport
minmod/Rusanov :

    transport-limited dt ~ 6.2e-02 ; source-limited explicit dt ~ 1/omega_c = 1.0e-03

    methode                                  dt_stable     dt*omega_c
    explicit (Lorentz explicite)              3.162e-04           0.32
    schur theta=0.5 (Crank-Nicolson)          1.778e-01         177.83
    schur theta=1.0 (Euler retrograde)        3.162e-01         316.23

    gain en pas de temps du Schur sur l'explicite :
      theta=0.5 -> 562x ; theta=1.0 -> 1000x

Lecture honnete de ces chiffres :

- L'explicite plafonne a `dt * omega_c ~ 0.3` : c'est la borne de stabilite de la
  rotation cyclotronique explicite (`dt * omega_c < O(1)`).
- Le Schur reste stable a `dt * omega_c` de 178 (theta=0.5) a 316 (theta=1.0),
  bien au-dela de la borne explicite : la contrainte cyclotronique est retiree. Le
  pas Schur (~ 0.18 a 0.32) approche le pas de transport (~ 0.06), preuve que c'est
  desormais le TRANSPORT, plus la source, qui limite.
- `theta = 1.0` (Euler retrograde, inconditionnellement stable pour la rotation)
  gagne davantage que `theta = 0.5` (Crank-Nicolson, marginalement stable), comme
  attendu.

Ces valeurs proviennent du README precedent (executions anterieures sur la meme
configuration) ; elles N'ONT PAS ete re-capturees lors de la redaction de ce
document (le balayage est long et n'a pas ete relance). Le `dt_stable` etant issu
d'un balayage geometrique a quart de decade, c'est une borne discrete (pas une
valeur fine) ; les chiffres exacts peuvent varier avec la plateforme, le
compilateur et les options.


## 13. Generation figures/GIF

Aucune. Ce cas ne produit ni figure ni GIF : sa seule sortie fichier est
`out/schur_magnetized_cartesian/dt_stable.csv` (table des `dt` stables, ecrite
uniquement avec `--csv`). Il n'a pas besoin de `matplotlib`.


## 14. Backends reellement supportes

- Backend DSL `aot` (host-marshale) : SEUL backend utilise par le cas. Il supporte
  `set_source_stage` et la force de Lorentz via `B_z`. Le chemin AOT n'expose que
  l'integrateur explicite SSPRK2 pour le transport (pas SSPRK3) — sans incidence
  sur la conclusion, qui porte sur l'etage SOURCE.
- Backend DSL `production` (natif zero-copie) : ECHOUE au `dlopen` sur macOS arm64
  avec ce build (espace de noms a deux niveaux). Le cas ne l'emploie donc pas.
- `adc.Split(adc.Explicit, adc.CondensedSchur)` (le descripteur haut-niveau du
  splitting Schur) n'est cable que par le chemin natif `production` : l'ABI AOT
  (`.so`) ne transporte PAS l'integrateur SSPRK3 qu'attend `adc.Split`. Le cas
   branche donc l'etage condense DIRECTEMENT via le hook prive
  `sim._s.set_source_stage("plasma", "electrostatic_lorentz", theta, alpha)`, qui
  execute le MEME C++ (`CondensedSchurSourceStepper`, #126) que ce que produirait
  `adc.Split` cote production. Le docstring de `run.py` mentionne encore
  `adc.Split(Explicit, CondensedSchur)` comme INTENTION/cible (#118-128) ; le code
  reel passe par le hook prive — c'est ce dernier qui fait foi.


## 15. Cout approximatif

Le balayage construit et integre une simulation pour CHAQUE `dt` teste, sur trois
variantes (explicite, Schur theta=0.5, Schur theta=1.0). A petit `dt` stable, le
nombre de pas `ceil(t_end/dt)` est grand (p.ex. `t_end=1`, `dt~3e-4` => ~3300 pas
par essai), repete sur tous les paliers du quadrillage geometrique. Sur une grille
`16 x 16` cela reste modeste (quelques minutes au plus, dominees par les essais a
petit `dt`), mais le cas est NON trivial : il est marque `experimental` / hors CI
et n'a pas ete relance lors de la redaction (cout exact non re-mesure). Le cout
croit avec `n^2`, avec `t_end` et avec `omega_c` (l'explicite, force a un `dt`
~ `1/omega_c`, fait d'autant plus de pas).


## 16. Limites et differences avec les references

- CE N'EST PAS une reproduction d'un resultat publie. Le cas est `experimental` :
  il MESURE une propriete de l'integrateur (gain en `dt` du Schur sur la raideur
  cyclotronique), il ne reproduit aucune figure ni table de papier. Toute fidelite
  a Hoffart (arXiv:2510.11808) releve du cas separe `hoffart_euler_poisson_dsl`
  (lui-meme `reproduction-candidate` PENDING, baseline cartesienne loin du papier).
- Geometrie CARTESIENNE. Le chemin polaire (diocotron) est explicite-only ; cette
  mesure du Schur est donc realisee en cartesien, sur un fluide magnetise raide
  jouet (densite cosinus + vitesse oblique), pas sur une configuration physique de
  reference.
- Cas RAIDE FABRIQUE : `cs2 = 1e-4` est choisi petit a dessein pour rendre le
  transport non limitant et exposer la raideur de la source. Les chiffres de gain
  (562x, 1000x) sont propres a CE point de fonctionnement ; ils ne se transposent
  pas tels quels a un cas ou le transport limiterait deja le pas.
- `dt_stable` est une borne DISCRETE (balayage a quart de decade), pas un seuil
  fin. Le critere de stabilite est heuristique (finitude + bornes + positivite sur
  la densite), pas une analyse spectrale.
- Backend AOT (SSPRK2) uniquement sur cette plateforme ; le chemin
  `production`/`adc.Split` n'est pas exerce ici (cf. section 14).


## 17. Tests/CI associes

- Manifeste : `cases_manifest.toml` declare ce cas `category = "experimental"`,
  `ci = false`, `needs = ["cxx"]`. Il N'EST PAS lance en CI (long et prototype de
  mesure) ; il se lance a la main.
- Modele apparente : le cas `magnetic_isothermal_dsl` (`category = "validation"`,
  `ci = true`, `needs = ["cxx"]`) valide les MEMES equations isothermes magnetisees
  (parite inter-backend, oracle analytique du terme de Lorentz, conservation de la
  masse, rotation effective). C'est lui qui garantit la correction du modele
  partage ; le present cas n'ajoute que la MESURE temporelle de l'etage Schur.
- Aucun `assert` dans ce cas : pas de critere de reussite automatique, seulement
  une table de mesures (`dt_stable.csv`).
