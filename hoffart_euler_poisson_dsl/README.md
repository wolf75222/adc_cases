# hoffart_euler_poisson_dsl

Reproduction du cas test diocotron magnétique de Hoffart, Maier, Shadid, Tomas,
*Structure-preserving finite element approximations of the magnetic Euler-Poisson equations*
(arXiv:2510.11808, section 5.3), avec le cœur volumes finis `adc_cpp` piloté en Python par `adc_cases`.

Le modèle Euler-Poisson isotherme magnétisé complet (continuité, quantité de mouvement avec force de
Lorentz, Poisson) est écrit une seule fois en DSL symbolique, compilé en C++, puis avancé par un
splitting de Strang (SSPRK3 + étage source à complément de Schur). Mesuré dans les bonnes unités, le
chemin volumes finis cartésien reproduit les taux de croissance du papier à moins de 10 %, et converge
vers eux quand on raffine la grille.

Pour la théorie complète (physique, dérivation du facteur de normalisation, schéma) voir `TUTORIAL.md`.

## Le rollup diocotron, mode l=4

![Animation du rollup diocotron l=4](figures/diocotron_l4.gif)

L'anneau d'électrons perturbé au mode 4 se déforme en carré, puis s'enroule en quatre vortex, comme
dans la figure 5.2 du papier. Animations des trois modes : `figures/diocotron_l3.gif`,
`figures/diocotron_l4.gif`, `figures/diocotron_l5.gif`.

## Résultat

Taux de croissance du modèle complet `system-schur` (n=96, fenêtres papier mappées en temps de
simulation, conversion `gamma_paper = gamma_raw_sim * 2pi/rhobar`) :

| mode l | gamma_raw_sim | gamma_paper (×2π) | cible papier | erreur |
|---|---|---|---|---|
| 3 | 0.1117 | 0.702 | 0.772 | −9.1 % |
| 4 | 0.1423 | 0.894 | 0.911 | −1.9 % |
| 5 | 0.1087 | 0.683 | 0.683 | +0.04 % |

L'erreur décroît avec la résolution : à n=256 les trois modes tombent sous 1 % (voir la figure de
convergence plus bas). Le « déficit −95 % » des versions antérieures de ce cas était un artefact de
métrologie, expliqué dans la section « la leçon » et dans `RESULTS_SYSTEM_SCHUR.md`.

## Figures

Snapshots schlieren de la densité, palette du papier (disque blanc, extérieur ardoise, colormap Blues),
aux fractions de temps `0.01, 1/8, ..., 7/8, t_f`. Le nombre de vortex égale le mode.

Mode l=3 (figure 5.1 du papier) : triangle, puis trois bras, puis trois vortex.

![Snapshots l=3](figures/snapshots_l3.png)

Mode l=4 (figure 5.2) : carré, puis quatre vortex.

![Snapshots l=4](figures/snapshots_l4.png)

Mode l=5 (figure 5.3) : pentagone, étoile à cinq branches, puis cinq vortex en couronne.

![Snapshots l=5](figures/snapshots_l5.png)

Taux de croissance, style figure 5.4. Panneaux (a,b,c) : amplitude `|c_l(t)|/|c_l(0)|` en échelle log,
la courbe suit la pente papier (tirets rouges) dans la fenêtre de fit mappée, puis sature. Panneau (d) :
`gamma_l` contre le mode, pour le papier, le modèle complet et la dérive ExB réduite.

![Taux de croissance](figures/growth_rate.png)

Convergence en résolution : l'erreur relative au papier tend vers zéro quand n croît.

![Convergence](figures/convergence.png)

## Le modèle en code : `model.py`

### Les paramètres du papier

```python
@dataclass(frozen=True)
class PaperParameters:
    radius: float = 16.0          # R : rayon du disque (paroi de Poisson)
    ring_inner: float = 6.0       # r0 : bord intérieur de l'anneau
    ring_outer: float = 8.0       # r1 : bord extérieur
    rho_min: float = 1.0e-6       # densité de fond
    rho_max: float = 1.0          # densité dans l'anneau
    beta: float = 1.0e6           # paramètre d'échelle magnétique
    perturbation: float = 0.1     # delta : amplitude du sin(l*theta)
    temperature: float = 0.0      # theta de la fermeture p = theta*rho (limite froide)

    @property
    def alpha(self):
        return self.beta * self.beta / self.rho_max   # alpha = beta^2 / rho_max = 1e12

    @property
    def omega(self):
        return self.beta * self.beta                  # omega = beta^2 = 1e12 (= |Omega| = B_z)
```

Les sept scalaires `r0, r1, R, rho_min, rho_max, beta, delta` sont ceux de la section 5.3, valeur pour
valeur. Les deux propriétés dérivent `alpha` (couplage de Poisson) et `omega` (champ magnétique). Le
point qui décide de toute la mesure : `alpha/omega = 1/rho_max = 1`. Les deux `1e12` se simplifient dans
la vitesse de dérive `v = grad(phi)/omega` avec `-Delta phi = alpha*rho`, si bien que le champ qui
advecte la densité ne dépend pas de `beta`. Le modèle complet advecte donc la densité avec le même champ
qu'une dérive ExB normalisée. La section « la leçon » s'appuie sur ce fait.

### Le modèle symbolique

```python
m = dsl.Model("hoffart_magnetic_euler_poisson_%s" % source)
rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v",
                                  roles=["Density", "MomentumX", "MomentumY"])
u = m.primitive("u", mx / rho)                 # vitesses primitives u = m_x/rho
v = m.primitive("v", my / rho)
pressure = m.primitive("p", params.temperature * rho)   # p = theta*rho
m.primitive_vars(rho, u, v)
m.conservative_from([rho, rho * u, rho * v])   # (rho, rho*u, rho*v) <-> (rho, u, v)
```

- `conservative_vars` déclare les trois inconnues conservatives : densité et les deux composantes de la
  quantité de mouvement. Pas d'équation d'énergie (le modèle est barotrope, conformément à l'annexe A
  du papier).
- `primitive` définit les variables physiques `u, v, p` à partir des conservatives.

```python
m.flux(x=[mx, mx * u + pressure, mx * v],
       y=[my, my * u, my * v + pressure])
sound_speed = dsl.sqrt(params.temperature)
m.eigenvalues(x=[u - sound_speed, u, u + sound_speed],
              y=[v - sound_speed, v, v + sound_speed])
```

- `flux` est le flux hyperbolique d'Euler : transport de masse `m`, tenseur `m m^T/rho + p I` pour la
  quantité de mouvement.
- `eigenvalues` donne les vitesses d'onde `u ± c`, `u` pour le solveur de Riemann. En limite froide
  (`theta = 0`) la vitesse du son est nulle et les trois valeurs propres dégénèrent en `u`.

```python
m.aux("phi")                  # potentiel electrostatique, rempli par le solveur de champ
grad_x = m.aux("grad_x")      # gradient de phi (force electrique)
grad_y = m.aux("grad_y")

if source == "local":         # variante AMR : source emise dans le C++ genere
    omega = m.param("omega", params.omega)
    m.source([0.0 * rho,
              -rho * grad_x + omega * my,     # -rho dphi/dx + omega*m_y
              -rho * grad_y - omega * mx])     # -rho dphi/dy - omega*m_x
else:                          # variante schur : source nulle ici
    m.source([0.0 * rho, 0.0 * mx, 0.0 * my])

alpha = m.param("alpha", params.alpha)
m.elliptic_rhs(-alpha * rho)   # ADC resout Delta(phi) = rhs ; le papier veut -Delta phi = alpha rho
m.check()
```

- La source couple la force électrique `-rho grad(phi)` et la force de Lorentz `m × Omega = (omega m_y,
  -omega m_x)`. Deux variantes : `local` émet cette source dans le modèle compilé (chemin AMR) ; `schur`
  la laisse à zéro car l'étage `CondensedSchur` la prend en charge implicitement (chemin de référence).
- `elliptic_rhs(-alpha*rho)` pose la loi de Gauss. Le signe est négatif parce que le solveur résout
  `Delta(phi) = rhs`, et le papier veut `-Delta phi = alpha rho`.
- `m.check()` valide la cohérence du modèle. `check_model.py` compare ensuite le modèle compilé aux
  formules à la main sur 2×2 cellules : résidu exactement nul (`figures/oracle_residual.png`).

### La densité et la dérive initiales

```python
def paper_initial_density(n, mode, params=None):
    rho = np.full((n, n), params.rho_min)
    ring = (radius >= params.ring_inner) & (radius <= params.ring_outer)
    rho[ring] = params.rho_max * (1.0 - params.perturbation
                                  + params.perturbation * np.sin(mode * angle[ring]))
    return rho

def drift_velocity_from_potential(phi, params=None):
    grad_y, grad_x = np.gradient(phi, h, h, edge_order=2)
    u = -grad_y / params.omega     # dérive ExB : v0 = -(grad phi x Omega)/|Omega|^2
    v = grad_x / params.omega
    return u, v
```

- `paper_initial_density` est l'équation (35) : fond `rho_min`, anneau `rho_max(1 - delta + delta
  sin(l theta))` entre `r0` et `r1`.
- `drift_velocity_from_potential` donne la vitesse initiale `E×B`. Le facteur `1/omega` vient du produit
  vectoriel `1/|Omega|^2` en 2D, et `omega = |Omega|`.

## Le run en code : `run.py`

### Assemblage du chemin de référence

```python
def build_uniform(compiled, rho, params, geometry="square"):
    sim = adc.System(n=n, L=params.length, periodic=False)
    sim.set_poisson(rhs="composite", solver="geometric_mg",
                    bc="dirichlet", wall="circle", wall_radius=params.radius)
    sim.set_magnetic_field(params.omega * np.ones_like(rho))   # B_z avant l'etage Schur
    sim.add_equation("electrons", model=compiled,
        spatial=adc.FiniteVolume(limiter="weno5", riemann="rusanov", variables="conservative"),
        time=adc.Strang(hyperbolic=adc.Explicit(method="ssprk3"),
                        source=adc.CondensedSchur(theta=0.5, alpha=params.alpha)))
    zeros = np.zeros_like(rho)
    sim.set_primitive_state("electrons", rho=rho, u=zeros, v=zeros)
    sim.solve_fields()                                          # premier Poisson : phi a partir de rho
    u0, v0 = drift_velocity_from_potential(np.asarray(sim.potential()), params)
    sim.set_primitive_state("electrons", rho=rho, u=u0, v=v0)   # on installe la derive du papier
    sim.solve_fields()                                          # deuxieme passe, etat coherent
    return sim
```

Ligne à ligne :

- `adc.System(n, L, periodic=False)` : grille carrée `n×n`, côté `L = 2R = 32`, bords non périodiques.
- `set_poisson(..., wall="circle", wall_radius=R)` : Poisson par multigrille géométrique, Dirichlet
  homogène sur la paroi circulaire de rayon `R`. Le disque du papier est approché par cette paroi
  embarquée dans la grille carrée.
- `set_magnetic_field(omega * ones)` : champ `B_z = omega` uniforme. À poser avant l'étage Schur, qui en
  a besoin.
- `add_equation(...)` installe le schéma : volumes finis WENO5-Z avec flux de Rusanov en variables
  conservatives, et un splitting de Strang. La partie hyperbolique est intégrée en SSPRK3 ; l'étage
  source est un complément de Schur à `theta = 1/2` (Crank-Nicolson), avec le couplage `alpha`.
- Les deux appels `set_primitive_state` puis `solve_fields` sont la relaxation à deux passes du papier :
  on pose la densité, on résout `phi`, on en déduit la dérive `v0`, on réinstalle l'état avec `v0`, puis
  on résout `phi` une seconde fois pour partir d'un état cohérent.

### L'étage source à complément de Schur

`adc.CondensedSchur(theta=0.5, alpha=...)` avance la source implicitement. La force de Lorentz s'inverse
par un éliminateur 2×2

```
B^-1 = 1/(1+w^2) * [[1, w], [-w, 1]],   w = theta * dt * B_z,
```

et l'opérateur elliptique condensé est `A = I + c rho B^-1` avec `c = theta^2 dt^2 alpha`. On résout `A`
pour `phi^{n+theta}` (BiCGStab préconditionné multigrille), puis on reconstruit la quantité de mouvement
`v^{n+theta} = B^-1 (v^n - theta dt grad phi^{n+theta})`. Ce mécanisme franchit les échelles de temps
cyclotron et plasma sans les résoudre, ce qui est l'intérêt du schéma du papier.

### La mesure paper-faithful (correctif T3)

```python
def fit_growth(times, amplitudes, mode, rhobar=1.0):
    lo, hi = paper_to_sim_time_window(PAPER_FIT_WINDOWS[mode], rhobar)   # fenetre MAPPEE en temps sim
    mask = (times >= lo) & (times <= hi) & (amplitudes > 0.0)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    return float(np.polyfit(times[mask], np.log(amplitudes[mask]), 1)[0])  # pente = gamma_raw_sim
```

Dans `results.py` :

```python
def paper_to_sim_time_window(window_paper, rhobar=1.0):
    scale = 2.0 * math.pi / rhobar          # t_sim = (2pi/rhobar) * t_paper
    lo, hi = window_paper
    return (lo * scale, hi * scale)

def gamma_to_paper_units(gamma_raw_sim, rhobar=1.0):
    return gamma_raw_sim * (2.0 * math.pi / rhobar)   # gamma_paper = gamma_raw_sim * 2pi/rhobar
```

- `fit_growth` ajuste une droite sur `log|c_l|` dans la fenêtre papier, mais convertie en temps de
  simulation par `paper_to_sim_time_window`. Le solveur tourne dans l'horloge ExB-naturelle ; le papier
  rapporte dans l'horloge `omega_d` cyclique, où un tour vaut `2pi` radians. Sans cette conversion, la
  fenêtre papier appliquée au temps de simulation tombe dans le transitoire, ce qui produisait le
  « déficit −95 % ».
- `gamma_to_paper_units` applique le même facteur `2pi/rhobar` au taux brut. Chaque enregistrement
  (`measurement_record.json`) porte les deux nombres côte à côte, `gamma_raw_sim` et `gamma_paper_units`.

## La leçon : pourquoi le « déficit −95 % » était un artefact

Le solveur produisait le bon résultat depuis le début. La comparaison au papier était fausse sur deux
points, tous deux liés au facteur `2pi`.

1. Horloge. Le taux brut était comparé aux cibles sans le facteur `2pi/rhobar`. Ce facteur est la
   conversion entre fréquence cyclique (celle du papier, `omega_d = 1`) et fréquence angulaire (un tour =
   `2pi` radians), vérifiée mode par mode contre la valeur propre analytique de Davidson par
   `diag/petri_eigenvalue.py`.
2. Fenêtre. Les fenêtres de fit du papier sont en temps papier, mais étaient appliquées au temps de
   simulation. La fenêtre `[0.40, 0.70]` du mode 3 correspond à `t_sim ∈ [2.51, 4.40]`, pas à
   `[0.40, 0.70]` ; appliquée telle quelle, elle mesure le transitoire.

Décomposition du déficit du mode 3 (`0.0312 → 0.772`, facteur 24.7) : fenêtre 3.20, puis `2pi = 6.28`,
puis résidu de grille cart contre polaire 1.23. Le produit `3.20 × 6.28 × 1.23` vaut 24.7, le déficit
observé. Seul le résidu de grille est physique, et il tend vers zéro avec la résolution. Détail dans
`T2_NORMALIZATION_AUDIT.md` et `RESULTS_SYSTEM_SCHUR.md`.

## Reproduire

Construire le module `adc` (voir le README de `adc_cpp`), puis :

```bash
export PYTHONPATH=<adc_cpp>/build/python

# oracle analytique (sans run) : le modele compile == les formules a la main, et les cibles papier
python check_model.py
python diag/petri_eigenvalue.py

# table des taux (modele complet, mesure paper-faithful). t-end >= 8.5 (fenetre mappee du mode 5)
python run.py --engine system-schur --n 96 --t-end 10 --modes 3 4 5 --dt 2e-3 --no-gif

# audit de normalisation + convergence
python diag/diag_normalization_audit.py 128
python diag/convergence_reduced.py

# figures style papier (snapshots, GIF, taux, convergence)
python diag/make_paper_figures.py 3 4 5 --out figures
```

## Carte des fichiers

- `model.py` : le modèle Euler-Poisson magnétisé en DSL, les paramètres du papier, la densité et la
  dérive initiales.
- `run.py` : assemblage du System, mesure paper-faithful (fenêtres mappées, conversion `2pi/rhobar`),
  sorties (amplitude, snapshots, GIF, table des taux, métadonnées).
- `results.py` : émetteur d'enregistrements de mesure (CSV et JSON), helpers `paper_to_sim_time_window`
  et `gamma_to_paper_units`, auto-test pur Python.
- `check_model.py` : oracle analytique, comparé bit-à-bit au modèle compilé.
- `diag/petri_eigenvalue.py` : la valeur propre analytique de Davidson (cibles et origine du `2pi`).
- `diag/diag_normalization_audit.py` : l'audit dimensionnel (échelles, candidats, décomposition de la
  fenêtre).
- `diag/convergence_reduced.py` : la convergence en résolution.
- `diag/make_paper_figures.py` : le générateur des figures et GIF.
- `diag/diag_polar_omega.py` : le chemin polaire réduit ExB, qui valide la normalisation `2pi/rhobar`.
- `run_polar.py` : le modèle complet sur grille polaire (chemin séparé, qui diverge encore).
- `TUTORIAL.md` : le tutoriel complet, de la physique au code.
- `RESULTS_SYSTEM_SCHUR.md`, `T2_NORMALIZATION_AUDIT.md`, `NORMALIZATION.md` : la table des taux, l'audit
  T2, le code T3, la convergence, et la normalisation du chemin polaire réduit.
- `figures/provenance.json` : la provenance de chaque figure.
