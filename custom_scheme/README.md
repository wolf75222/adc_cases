# custom_scheme : ecrire son propre schema autour du solveur elliptique de adc

Un tutoriel : on ecrit le transport diocotron (reconstruction, flux upwind, integrateur SSPRK2)
entierement en numpy, cote Python. La densite vit dans un tableau numpy ; `adc` ne sert que
d'oracle de Poisson : a chaque sous-pas on lui remet la densite courante et on lui demande le
potentiel self-consistant `phi`. Aucune brique de transport de la lib n'est employee dans la boucle.
But pedagogique : montrer comment greffer une methode numerique maison sur le solveur elliptique
couteux, sans le reimplementer.

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `tutoriel` (`cases_manifest.toml`, `custom_scheme/run.py`, `ci = true`, `needs = []`) |
| Entrees | grille $96^2$, $L=1$, periodique ; CI bande gaussienne mode 4 `band_density(96, L, amp=1, width=0.05, mode=4, disp=0.02)` ($n=1+e^{-(y-y_0)^2/w^2}$, $y_0=0.5L+0.02\cos(8\pi x/L)$) ; $B_0=1$, $\alpha=1$, fond neutralisant $n_{i0}=\langle n\rangle$ ; CFL $=0.4$, 200 pas SSPRK2 |
| Sorties | densite numpy $n$ (lue/ecrite en numpy), potentiel $\phi$ rendu par `sim.potential()` ; figures `figures/density_evolution.png`, `figures/phi_evolution.png`, `figures/diagnostics.png` + `figures/provenance.json` |
| Invariants garantis | les `assert` de `run.py` : `assert_finite(n, ...)` a chaque pas (`run.py:94`) ; `assert drel < 1e-12` (masse, `run.py:101`) ; `assert moved > 1e-3` (la bande a evolue, `run.py:102`) ; `assert |phi|_max > 1e-8` (Poisson actif, `run.py:83`) |
| PROUVE | (1) le schema numpy conserve la masse a $2.040\times10^{-16}$ relatif sur 200 pas (flux upwind en forme flux, domaine periodique) ; (2) le couplage Poisson est actif : `adc` rend $\|phi\|_\infty=6.12\times10^{-3}\neq0$ ; (3) la dynamique est non triviale : $\max|n-n_0|=3.28\times10^{-1}$ |
| NE PROUVE PAS | demontre une capacite d'API, ne valide aucun resultat physique publie. Le schema maison (upwind ordre 1 + differences centrees pour $\nabla\phi$) n'est pas le schema valide du cas [`diocotron`](../diocotron/) (MUSCL minmod + Rusanov) : il est plus diffusif et aucun taux de croissance n'est mesure ni compare. Aucun assert ne teste la physique (pas de $\gamma_l$, pas d'oracle analytique). La conservation et la finitude sont des proprietes du schema, pas une reproduction. La CI est une bande (mode 4), pas l'anneau du benchmark : elle cisaille mais ne developpe pas l'instabilite annulaire du papier |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, backend natif serie (Poisson `geometric_mg`), $96^2$, ~1.2 s 1 coeur CPU ; `figures/provenance.json` |

A la fin tu sauras : ou tracer la frontiere Python / lib quand on ecrit son propre schema (Python
fait tout le transport, `adc` fait Poisson), comment ecrire un upwind conservatif et un SSPRK2 en
numpy, pourquoi la masse est conservee a la precision machine, et ce que ce tutoriel ne valide pas.

---

## 1. Ce que demontre ce cas (justifie PROUVE : capacite d'API)

La physique du diocotron (derive E x B, rotation differentielle, instabilite de cisaillement) est
derivee une fois pour toutes dans [`../diocotron/`](../diocotron/) ; ce tutoriel ne la rejoue pas.
Son objet est l'architecture : qui calcule quoi quand on veut son propre integrateur.

Le diocotron resout
$$\partial_t n + \nabla\cdot(n\mathbf{v}) = 0,\qquad \mathbf{v} = \frac{1}{B_0}(-\partial_y\phi,\ \partial_x\phi),\qquad \nabla^2\phi = \alpha\,(n - n_{i0}).$$
Le transport (premiere et deuxieme equations) est explicite et bon marche : une fois $\phi$
connu, c'est de l'advection a vitesse donnee. L'elliptique (Poisson, troisieme equation) est
global et couteux : un multigrille sur tout le domaine, refait a chaque sous-etage parce que
$\mathbf{v}$ depend de $\phi(n)$ qui depend de l'etat courant $n$. Ce cas met le transport en numpy
et garde Poisson dans `adc`. C'est le pendant spatial de l'integrateur temporel Python de
`adc.integrate`.

---

## 2. Les equations et qui les calcule (table 3 couches adaptee : Python fait tout le transport)

Dans les autres cas a briques, la couche du milieu est une brique de transport C++ figee
(`ExBVelocity`, `CompressibleFlux`). Ici la couche transport remonte cote Python : `adc` n'occupe
plus que la couche elliptique. La table reflete ce deplacement.

| Ligne `run.py` | Couche | Ce qui se passe |
|---|---|---|
| `drift(...)` + `divergence_upwind(...)` + boucle SSPRK2 (`run.py:34-50`, `run.py:88-93`) | Python calcule le transport | reconstruction $\nabla\phi$ (differences centrees), vitesse ExB, flux upwind conservatif, integrateur SSPRK2. Tout le hot path d'advection est en numpy |
| `poisson_oracle` -> `set_density` + `solve_fields` + `potential` (`run.py:53-57`) | adc = oracle de Poisson (le seul appel a la lib) | la lib resout $\nabla^2\phi=\alpha(n-n_{i0})$ et rend $\phi$. Un bloc `models.diocotron` est ajoute uniquement pour fournir le second membre elliptique `BackgroundDensity` |
| `assemble`... non employe ; seul le Poisson de systeme (`GeometricMG`) tourne (`run.py:78` `solver="geometric_mg"`) | noyau par cellule (device) | le multigrille de Poisson. Le flux ExB du coeur (`ExBVelocity::flux`) n'est jamais appele : le transport ne passe pas par la lib |

La brique `models.diocotron` est ajoutee (`run.py:77`) pour que `adc.System` ait un second membre
elliptique a assembler ; sa partie transport (`adc.ExB`, c.-a-d. `ExBVelocity` dans
`include/adc/physics/hyperbolic.hpp:27`) reste inerte parce que `run.py` n'appelle jamais
`advance`/`step` : on ne demande que `solve_fields`. Le reste de la composition (`adc.Scalar`,
`adc.NoSource`) est present mais non sollicite dans la boucle.

---

## 3. Le schema, fonction par fonction (justifie : ancrage reel)

`run.py` se lit du haut vers le bas. On glose les fonctions porteuses ; la plomberie (import,
fallback `sys.path`, `run.py:23-31`) est liee, pas paraphrasee.

### 3.1 La frontiere : `poisson_oracle` (`run.py:53-57`), le seul appel a la lib

```python
def poisson_oracle(sim, n):
    sim.set_density("ne", n)        # ecrit la densite numpy dans le bloc adc (run.py:55)
    sim.solve_fields()              # adc resout lap phi = alpha (n - n_i0) (run.py:56)
    return sim.potential()          # rend phi (n, n) au format numpy (run.py:57)
```
- `set_density("ne", n)` copie le tableau numpy $n$ dans l'unique bloc du `System`. `solve_fields`
  declenche le multigrille de Poisson de systeme (second membre = somme des briques elliptiques, ici
  l'unique `BackgroundDensity(alpha=1, n0=n_i0)` figee dans `elliptic.hpp:31-37` :
  `alpha*(u[0]-n0)`). `potential()` lit $\phi$ et le renvoie en numpy. Ces trois lignes sont
  l'integralite du contact avec `adc` : tout ce qui suit est du numpy pur. (`set_density`,
  `solve_fields`, `potential` sont des methodes de la facade compilee, exposees via
  `System.__getattr__`, `adc/__init__.py:1116-1117`.)

### 3.2 La vitesse ExB : `drift` (`run.py:34-38`)

```python
def drift(phi, dx, B0):
    dphidx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)   # d phi/dx centre
    dphidy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)   # d phi/dy centre
    return -dphidy / B0, dphidx / B0   # (vx, vy) = (-d_y phi, d_x phi)/B0 (run.py:38)
```
- Reconstruction de $\nabla\phi$ par differences centrees periodiques (`np.roll` referme le
  domaine). La convention de grille est `phi[j, i]` (cf. `adc_cases/common/grid.py`) : `axis=1` =
  colonne = $x$, `axis=0` = ligne = $y$. La vitesse ExB $\mathbf{v}=(-\partial_y\phi,\partial_x\phi)/B_0$
  reproduit en numpy exactement la formule figee par `ExBVelocity::velocity`
  (`hyperbolic.hpp:31-33` : `(dir==0) ? -grad_y/B0 : grad_x/B0`). Cette vitesse est a divergence
  nulle (c'est le rotationnel de $\phi$), ce qui rend le transport conservatif.

### 3.3 Le flux upwind conservatif : `divergence_upwind` (`run.py:41-50`)

```python
vxr = 0.5 * (vx + np.roll(vx, -1, axis=1))                    # vitesse a l'interface i+1/2
fxr = np.where(vxr > 0, n, np.roll(n, -1, axis=1)) * vxr      # flux upwind en i+1/2 (run.py:45)
fxl = np.roll(fxr, 1, axis=1)                                 # flux en i-1/2 = decalage de fxr
...
return -((fxr - fxl) + (fyr - fyl)) / dx                      # -div(n v) (run.py:50)
```
- Forme flux : on calcule le flux $f_{i+1/2}=n^{\text{amont}}\,v_{i+1/2}$ a chaque interface, et
  la divergence est la difference de flux entrant/sortant. `np.where(vxr>0, n, roll(n,-1))` choisit
  l'etat amont (upwind ordre 1) selon le signe de la vitesse a l'interface. Point cle pour la
  conservation : le flux en $i-1/2$ est exactement le flux en $i+1/2$ de la cellule de gauche
  (`fxl = np.roll(fxr, 1)`). Le flux sortant d'une cellule est donc, au signe pres, le flux entrant
  de sa voisine : la somme de toutes les divergences se telescope a zero sur un domaine periodique.
  C'est ce qui donne la conservation a la precision machine (section 4).

### 3.4 Le residu et l'integrateur : `rhs` + boucle SSPRK2 (`run.py:60-65`, `run.py:88-93`)

```python
def rhs(sim, n, dx, B0):
    phi = poisson_oracle(sim, n)       # Poisson par adc (run.py:62)
    vx, vy = drift(phi, dx, B0)        # vitesse ExB en numpy
    speed = float(np.hypot(vx, vy).max())
    return divergence_upwind(n, vx, vy, dx), speed   # -div(n v) + speed pour le pas adaptatif
```
```python
for step in range(nsteps):
    r1, speed = rhs(sim, n, dx, B0)
    dt = cfl * dx / max(speed, 1e-12)        # CFL : dt = 0.4 dx / max|v| (run.py:90)
    n1 = n + dt * r1                          # etage 1 d'Euler explicite (run.py:91)
    r2, _ = rhs(sim, n1, dx, B0)              # 2e evaluation : Poisson refait sur n1 (run.py:92)
    n = 0.5 * n + 0.5 * (n1 + dt * r2)        # SSPRK2 (Heun fort-stable) (run.py:93)
    assert_finite(n, "densite au pas %d" % step)
```
- SSPRK2 (Heun) ecrit a la main : etage predicteur $n_1=n+\Delta t\,R(n)$, puis correcteur
  $n^{+}=\tfrac12 n+\tfrac12(n_1+\Delta t\,R(n_1))$. La forme convexe $\tfrac12 n+\tfrac12(\dots)$
  est la propriete strong-stability-preserving : chaque etage est une combinaison convexe
  d'updates Euler explicites, donc l'integrateur n'ajoute pas d'oscillation que l'upwind ne controle
  deja. Le Poisson est refait a chaque etage (`rhs` appelle `poisson_oracle` deux fois par pas) :
  la vitesse de l'etage correcteur utilise le $\phi$ de l'etat predit $n_1$, pas celui de $n$. Le pas
  $\Delta t$ est adaptatif via la CFL sur la vitesse ExB max courante.

`assert_finite` (`adc_cases/common/checks.py:29-32`) leve `AssertionError` si un NaN/Inf apparait :
garde-fou contre une divergence du schema maison.

---

## 4. Pourquoi la masse est conservee a la precision machine (justifie PROUVE 1 et les tolerances)

La conservation n'est pas un hasard : elle suit de deux proprietes du schema, independantes l'une de
l'autre.

1. Forme flux + periodicite -> telescopage exact. La masse totale est
   $M=\sum_{j,i} n_{j,i}\,dx^2$. Son increment par pas Euler est
   $\Delta M = dx^2\sum_{j,i}(-\mathrm{div}\,f)_{j,i}\,\Delta t = -dx\,\Delta t\sum_{j,i}\big[(f^x_{i+1/2}-f^x_{i-1/2})+(\dots)\big]$.
   Comme `fxl = np.roll(fxr, 1)` (section 3.3), la somme des differences de flux est telescopique sur
   un domaine periodique : elle vaut exactement zero (chaque flux d'interface est compte une fois
   $+$ et une fois $-$). En arithmetique exacte $\Delta M=0$.
2. SSPRK2 preserve la conservation. $n^{+}=\tfrac12 n+\tfrac12(n_1+\Delta t R(n_1))$ est une
   combinaison affine d'etats dont chacun a la meme masse (les deux etages sont conservatifs) ;
   une moyenne convexe d'etats de masse $M$ a la masse $M$.

La seule derive observee vient de l'ordre de sommation flottant de `n.sum()` et du multigrille de
Poisson, pas d'un defaut du schema. D'ou les tolerances :

| Tolerance | Valeur | Pourquoi cette valeur |
|---|---|---|
| `drel < 1e-12` (`run.py:101`) | $10^{-12}$ | Borne basse : le schema est exactement conservatif, seule l'arithmetique flottante derive ($\sim10^{-16}$ relatif par accumulation sur $96^2$ cellules et 200 pas). Borne haute : tout $>10^{-12}$ trahirait une fuite reelle (flux mal apparies, condition de bord non periodique). Mesure : $2.040\times10^{-16}$, ~4 ordres sous la tolerance |
| `moved > 1e-3` (`run.py:102`) | $10^{-3}$ | Seuil de non-trivialite : si le schema etait inerte (vitesse nulle, ou Poisson rendant $\phi=0$), $\max|n-n_0|$ resterait au bruit machine. $10^{-3}$ est tres au-dessus du bruit et tres en-dessous de l'amplitude reelle ($3.28\times10^{-1}$ mesure) : il garantit que le transport bouge vraiment la bande |
| `|phi|_max > 1e-8` (`run.py:83`) | $10^{-8}$ | Le Poisson est actif ssi $\phi\neq0$. Mesure : $6.12\times10^{-3}$, ~6 ordres au-dessus : le couplage elliptique n'est pas un no-op |

---

## 5. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (memes parametres et memes fonctions numpy que `run.py`),
versionnees avec `figures/provenance.json`. Commande exacte en section 6. La frontiere Python / lib
est exactement celle du tutoriel : la densite (fig. 1) est calculee en numpy, $|phi|$ (fig. 2) est
rendu par `adc.potential()`. Les 4 instantanes sont pris aux pas $\{0, 40, 100, 200\}$, soit
$t\in\{0,\ 4.39,\ 11.28,\ 23.03\}$ avec le pas adaptatif CFL.

### `density_evolution.png` : la densite advectee par le transport numpy

![Densite n a 4 instants : bande mode 4 qui se cisaille et s'aplatit](figures/density_evolution.png)

- PROUVE / mesure (asserte `run.py:102`) : la bande evolue ($\max|n-n_0|=3.28\times10^{-1}$,
  $\gg 10^{-3}$). L'ondulation initiale (mode 4 du `disp=0.02`) est entrainee par la rotation
  differentielle E x B : les 4 bosses du panneau $t=0$ se cisaillent et s'etirent en filaments fins,
  puis la bande s'aplatit autour de $y=0.5$ ($t=23$).
- PROUVE (asserte `run.py:101`) : malgre cette deformation, la masse totale est conservee a
  $2.04\times10^{-16}$ relatif : le transport numpy est strictement conservatif (section 4).
- SUGGERE (non assere) : la formation de filaments fins evoque un cisaillement de type
  Kelvin-Helmholtz, mais aucun taux n'est mesure et la CI est une bande, pas l'anneau du benchmark
  diocotron.
- NON MONTRE : ce schema upwind ordre 1 est plus diffusif que le MUSCL minmod du cas
  [`diocotron`](../diocotron/) ; la comparaison de fidelite n'est pas faite ici.

### `phi_evolution.png` : le potentiel resolu par adc (le seul role de la lib)

![|phi| a 4 instants : potentiel self-consistant suivant la bande de charge](figures/phi_evolution.png)

- PROUVE (asserte `run.py:83`) : $\phi\neq0$, $\|phi\|_\infty=6.12\times10^{-3}$ a $t=0$,
  $6.01\times10^{-3}$ a $t=23$ : le Poisson de `adc` est actif a chaque sous-pas. Le potentiel est
  concentre sur la bande de charge (jaune) et change de structure de part et d'autre (bandes sombres
  en $y\approx0.27$ et $y\approx0.73$, le fond neutralisant $n_{i0}$ rendant le second membre de
  moyenne nulle, condition de compatibilite du Laplacien periodique).
- SUGGERE : $\phi$ se lisse au cours du temps (les bosses azimutales visibles a $t=0$ s'effacent
  a $t=23$) en suivant l'aplatissement de la densite : le couplage est self-consistant. Non assere
  (aucun test sur la forme de $\phi$).
- NON MONTRE : la magnitude absolue de $\phi$ n'est comparee a rien (pas d'unites physiques
  calibrees, $\alpha=1$ sans dimension) ; seule sa non-nullite est testee.

### `diagnostics.png` : conservation, evolution, couplage en fonction du temps

![Trois series temporelles : derive de masse, max|dn|, |phi|_max et vitesse ExB](figures/diagnostics.png)

- PROUVE : (gauche) la derive de masse relative reste collee au bruit machine
  ($\sim10^{-16}$, plusieurs ordres sous le trait rouge de tolerance $10^{-12}$) sur les 200 pas ;
  (milieu) $\max|n-n_0|$ franchit le seuil $10^{-3}$ des les premiers pas, culmine $\approx0.48$ puis
  oscille autour de $0.33$ ; (droite) $\|phi\|_\infty$ ($\sim6\times10^{-3}$) et la vitesse ExB max
  ($\sim3.5\times10^{-2}$) restent quasi constantes : le couplage Poisson est actif et stable du
  debut a la fin.
- SUGGERE : la lente decroissance de la vitesse ExB max (de $3.81\times10^{-2}$ a
  $3.54\times10^{-2}$) accompagne le lissage diffusif de la bande ; non testee.
- NON MONTRE : aucune comparaison a une solution de reference ou a un taux analytique ; ces
  series diagnostiquent la mecanique du schema (conservatif, actif, non trivial), pas la fidelite
  physique.

---

## 6. Reproduire (justifie : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/custom_scheme
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~1.2 s
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 3 figures + provenance.json
```

Prerequis : `numpy` (et `matplotlib` pour les figures), module `adc` compile et importe avec le
meme interpreteur que celui qui l'a compile (suffixe ABI `cpython-312`). Le premier chemin du
`PYTHONPATH` fournit le module C++ ; le second rend `adc_cases` importable sans installation (le cas
a aussi un fallback `sys.path`, `run.py:23-31`).

Sortie attendue de `run.py` (capturee, machine de dev macOS arm64) :

```
== custom_scheme : transport diocotron 100 % Python, Poisson par adc ==
  |phi|_max initial = 6.124932e-03  (Poisson de adc actif)
  derive de masse relative = 2.040e-16  (flux upwind conservatif)
  evolution max|dn|        = 3.280e-01  (dynamique non triviale)
Schema spatial + temporel ecrit en Python ; adc ne fait que Poisson.
OK custom_scheme
```

Cout : ~1.2 s temps mur (import numpy inclus), 200 pas SSPRK2 a $96^2$ avec 2 resolutions
multigrille de Poisson par pas (une par etage), soit 400 appels a `solve_fields`. Caveat
plateforme : les ordres de grandeur (masse $\sim10^{-16}$, $\|phi\|_\infty\sim6\times10^{-3}$,
$\max|dn|\sim0.33$) et le verdict `OK` sont stables d'une plateforme a l'autre ; les derniers
chiffres varient avec la BLAS, le multigrille et l'ordre de sommation (cf. `figures/provenance.json`).

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | le cas : transport ExB en numpy (`drift`, `divergence_upwind`, SSPRK2), Poisson par `adc` (`poisson_oracle`), asserts (masse, evolution, couplage) |
| `make_figures.py` | rejoue la physique en instrumentant l'evolution ; ecrit les 3 figures + `provenance.json` |
| `figures/*.png` | `density_evolution.png`, `phi_evolution.png`, `diagnostics.png` (versionnees, regenerees en place) |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, backend, resolution, nombres mesures (masse, $\|phi\|$, vitesse, $\max|dn|$) |
| `../diocotron/` | la physique diocotron complete (mecanisme, oracle analytique, taux $\gamma_l$) que ce tutoriel ne rejoue pas |
| `../adc_cases/models.py` | `diocotron(B0, alpha, n_i0)` = composition de briques natives (transport `ExB` inerte ici, elliptique `BackgroundDensity` seule utilisee) |
| `../adc_cases/common/initial_conditions.py` | `band_density` (la CI bande gaussienne mode 4) |
