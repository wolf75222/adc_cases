# Reproduire le diocotron magnétique de Hoffart et al. avec adc : tutoriel

Ce document explique, de la physique au code, comment reproduire le cas test diocotron de la
section 5.3 de Hoffart, Maier, Shadid, Tomas, *Structure-preserving finite element approximations
of the magnetic Euler-Poisson equations* (arXiv:2510.11808), avec `adc_cpp` (cœur volumes finis C++)
et `adc_cases` (scénarios Python). On obtient les taux de croissance du papier à moins de 10 %
(et à moins de 1 % en montant la résolution), et les figures de rollup dans le même style.

La leçon centrale du cas est métrologique : le solveur produisait le bon résultat
depuis le début ; c'est la façon de le mesurer (horloge et fenêtre de fit) qui faussait la comparaison.
La section 5 détaille ce point, qui est le vrai contenu pédagogique de ce cas.

---

## 1. La physique : instabilité diocotron d'une colonne creuse

Une colonne d'électrons non neutre, plongée dans un champ magnétique axial uniforme, tourne sous l'effet
de sa propre dérive `E×B`. Quand la densité a une forme d'anneau (creuse au centre), les deux bords de
l'anneau, intérieur et extérieur, portent des sauts de densité de signes opposés. Ces deux interfaces se
couplent par le champ électrique perturbé et s'amplifient mutuellement : c'est le mécanisme de
Kelvin-Helmholtz appliqué à la rotation `E×B`, qui porte ici le nom d'instabilité diocotron. Le mode
azimutal `ℓ` croît exponentiellement, puis l'anneau se replie en `ℓ` vortex.

Le papier travaille dans la limite de dérive magnétique : le champ magnétique est si fort que les
échelles de temps cyclotron (`ω_c`) et plasma (`ω_p`) sont des ordres de grandeur plus rapides que la
dérive lente (`ω_d`). Le schéma numérique doit franchir ces échelles rapides sans les résoudre, tout en
capturant la dérive lente. C'est ce que permet l'étage source implicite (section 4).

---

## 2. Le modèle et ses équations

Le modèle est l'Euler-Poisson isotherme magnétisé en 2D. Inconnues conservatives : densité `ρ` et
quantité de mouvement `m = ρ u`. Avec `Ω = ω e_z` (champ magnétique transverse, `m × Ω = (ω m_y, -ω m_x)`) :

$$\partial_t \rho + \nabla\cdot m = 0,$$
$$\partial_t m + \nabla\cdot\!\left(\tfrac{m\,m^\top}{\rho} + p\,I\right) = -\rho\,\nabla\phi + m\times\Omega,$$
$$-\Delta\phi = \alpha\,\rho,\qquad p = \theta\,\rho.$$

Fermeture isotherme `p = θρ`, vitesse du son `c = √θ`. La limite froide `θ = 0` rend le son négligeable
devant la dérive ; c'est le défaut du cas. La loi de Gauss `-Δφ = αρ` fixe le potentiel électrostatique,
et la force `-ρ∇φ` plus la force de Lorentz `m×Ω` forment l'étage source.

Le code écrit ce modèle une fois en DSL symbolique (`model.py`). Le bloc compilé est vérifié bit-à-bit
contre les formules à la main par `check_model.py` (résidu exactement nul sur 2×2 cellules), donc le
modèle généré n'est jamais en cause ; voir `figures/oracle_residual.png`.

### Paramètres du papier (section 5.3)

| grandeur | valeur | rôle |
|---|---|---|
| `r0, r1, R` | 6, 8, 16 | rayons intérieur/extérieur de l'anneau, paroi du disque |
| `ρ_min, ρ_max` | 1e-6, 1 | densités de fond et de l'anneau |
| `β` | 1e6 | paramètre d'échelle magnétique |
| `δ` | 0.1 | amplitude de la perturbation `sin(ℓθ)` |
| `α = β²/ρ_max` | 1e12 | couplage de Poisson |
| `ω = β²` | 1e12 | champ `B_z` (`= |Ω| = ω_c`) |
| `ℓ` | 3, 4, 5 | modes azimutaux testés |
| `t_f` | 10 | temps final, en périodes diocotron `T_d` |

Densité initiale (équation 35 du papier) : `ρ0 = ρ_min` hors de l'anneau, et
`ρ0 = ρ_max (1 - δ + δ sin(ℓθ))` pour `r0 < r < r1`. Vitesse initiale = dérive `E×B` du potentiel initial :
`u0 = -∂_y φ0 / ω`, `v0 = ∂_x φ0 / ω`, avec `-Δφ0 = αρ0`.

### La simplification qui décide tout : `α/ω = 1`

Avec `α = β²/ρ_max` et `ω = β²`, le rapport vaut `α/ω = 1/ρ_max = 1`. En posant `φ = α φ̃` (donc
`-Δφ̃ = ρ`), la vitesse de dérive devient

$$v = \frac{\nabla\phi}{\omega} = \frac{\alpha}{\omega}\,\nabla\tilde\phi = \nabla\tilde\phi .$$

Les deux `1e12` se simplifient : le champ qui advecte `ρ` ne dépend plus de `β`. C'est exactement la
dérive `E×B` normalisée (`B = 1`, charge `= 1`). Conséquence pratique : le modèle complet et un simple
transport de scalaire passif `ρ` par `v = ∇φ̃` advectent la densité avec le même champ. On le vérifie
numériquement (section 5) : les deux donnent le même taux à environ 2 % près.

---

## 3. La théorie linéaire et l'origine du facteur 2π

Sous la perturbation `exp(i(ℓ arctan(x2/x1) - ω_ℓ t))`, le potentiel est dominé par son mode `ℓ`, qui
croît comme `exp(γ_ℓ t)` avec `γ_ℓ = Im(ω_ℓ)`. Pour la colonne creuse top-hat, la théorie de Davidson
(réf. [13] du papier) donne un problème aux valeurs propres 2×2 sur les déplacements des deux bords. Les
fréquences de rotation de dérive d'équilibre valent, dans l'anneau,

$$\omega_E(r) = \frac{W_d}{2}\left(1 - \frac{r_0^2}{r^2}\right),\qquad W_d = 2\pi\,\omega_d .$$

Le couplage des deux bords (auto- et inter-couplage géométriques, paroi de Dirichlet en `R`) produit une
paire de valeurs propres complexes conjuguées : l'instabilité. La résolution donne, pour la géométrie
6:8:16,

$$\gamma_3 \approx 0.772,\quad \gamma_4 \approx 0.911,\quad \gamma_5 \approx 0.683 .$$

Le script `diag/petri_eigenvalue.py` reconstruit cette matrice et retrouve ces trois cibles à moins de
0.5 %. C'est l'oracle analytique de comparaison.

**Où est le 2π.** Le papier définit `ω_d := ρ_max α/|Ω| = 1` comme fréquence cyclique (un tour par
période `T_d = 1/ω_d = 1`). La relation de dispersion, elle, manipule une fréquence angulaire :
un tour vaut `2π` radians, donc `W_d = 2π ω_d`. Le `2π` est cette conversion cyclique vers angulaire,
pas un ajustement. Preuve par le script Petri : avec `W_d = 2π ω_d` il reproduit les cibles ; avec
`W_d = ω_d = 1` il rend exactement les cibles divisées par `2π`. Comme `Im(ω)` est linéaire en `W_d`, le
rapport est exactement `2π` sur les trois modes.

La conséquence est l'équation à retenir pour la mesure :

$$\boxed{\;\gamma_{\text{papier}} = \gamma_{\text{brut,sim}}\;\frac{2\pi}{\bar\rho}\;,\qquad \bar\rho=\rho_{\max}=1\;}$$

Le solveur numérique tourne dans l'horloge `E×B` naturelle ; le papier rapporte dans l'horloge `ω_d`
cyclique. Le facteur `2π/ρ̄` convertit l'une vers l'autre. Et puisque `α/ω = 1` (section 2.3), il
s'applique au modèle complet comme au transport réduit : les deux partagent la même horloge de dérive.

---

## 4. Le schéma numérique

Le chemin de référence (`--engine system-schur`) est un splitting d'opérateurs entre l'Euler hyperbolique
et la source électrostatique/Lorentz.

**Partie hyperbolique.** Volumes finis cellule-centrés sur grille carrée `L = 2R`, reconstruction WENO5-Z,
flux de Rusanov, variables conservatives. Intégration en temps SSPRK3 (Runge-Kutta fort-stable d'ordre 3,
peu dissipatif).

**Splitting de Strang.** Le pas complet est `H(dt/2) ; S(dt) ; H(dt/2)` : demi-transport, source pleine,
demi-transport. C'est le splitting symétrique d'ordre 2 du papier (`adc.Strang(hyperbolic=Explicit("ssprk3"),
source=CondensedSchur(theta=0.5))`, livré par adc_cpp #230).

**Étage source à complément de Schur.** La source couple potentiel, quantité de mouvement et champ
magnétique à des échelles de temps qui s'étalent sur de nombreux ordres de grandeur. La résoudre
explicitement imposerait `dt ~ 1/ω_c`. Le papier (et le code) la traitent implicitement par un complément
de Schur de PDE, qui se ramène à un Poisson généralisé. En θ-schéma (`θ = 1/2`, Crank-Nicolson),
l'éliminateur de Lorentz est l'inverse 2×2

$$B^{-1} = \frac{1}{1+w^2}\begin{pmatrix} 1 & w \\ -w & 1\end{pmatrix},\qquad w = \theta\,dt\,B_z,$$

et l'opérateur elliptique condensé est `A = I + c\,ρ\,B^{-1}` avec `c = θ² dt² α`. On résout
`A` pour `φ^{n+θ}` (BiCGStab préconditionné par multigrille géométrique), puis on reconstruit la quantité
de mouvement `v^{n+θ} = B^{-1}(v^n - θ dt ∇φ^{n+θ})`. Ce mécanisme franchit les échelles rapides sans les
résoudre, ce qui est tout l'intérêt du schéma.

Tout ceci a été audité ligne à ligne contre le papier (signe de Poisson, `m×Ω`, `B^{-1}`, `c = θ²dt²α`) :
voir `adc_cpp/docs/HOFFART_FIDELITY.md`. Aucun écart de câblage.

---

## 5. La métrologie : pourquoi le « déficit −95 % » était un artefact

Pendant longtemps le cas affichait un déficit de −82 à −95 % sur les trois modes. Le solveur n'était pas
en cause. Deux erreurs de mesure se cumulaient.

**Erreur d'horloge.** Le taux brut `γ_brut` était comparé directement aux cibles, sans le facteur
`2π/ρ̄` de la section 3. À lui seul, ce facteur vaut `2π ≈ 6.28`.

**Erreur de fenêtre.** La pente était ajustée dans les fenêtres de fit du papier (`[0.40,0.70]` pour
`ℓ=3`, etc.), mais appliquées au temps de simulation brut. Or ces fenêtres sont en temps papier,
et `t_sim = (2π/ρ̄) t_papier`. La fenêtre papier `[0.40,0.70]` correspond donc à `t_sim ∈ [2.51,4.40]`,
pas à `[0.40,0.70]` en temps sim. Appliquée au temps sim, elle tombe dans le transitoire précoce, où le
taux n'a pas encore atteint sa valeur exponentielle.

Mesuré sur un même run réduit (n=128), fenêtre papier appliquée au temps sim contre fenêtre établie :

| ℓ | fenêtre papier sur temps sim | fenêtre établie [3,12] | rapport |
|---|---|---|---|
| 3 | 0.0312 | 0.0998 | 3.20 |
| 4 | 0.0943 | 0.1135 | 1.20 |
| 5 | 0.1056 | 0.1137 | 1.08 |

Le rapport 3.20 pour `ℓ=3` est le « résidu ~3× » : c'est la fenêtre, pas une échelle physique manquante.
La décomposition complète du déficit `ℓ=3` ferme exactement :

$$0.0312 \;\xrightarrow{\text{fenêtre }3.20\times}\; 0.0998 \;\xrightarrow{\;2\pi\,=\,6.28\times\;}\; 0.627 \;\xrightarrow{\text{grille }1.23\times}\; 0.772,\qquad 3.20\times6.28\times1.23 = 24.7 .$$

Le facteur grille final (~20 % à `ℓ=3`, n=96) est la seule part physique. Il vient de la discrétisation
cartésienne du bord d'anneau, et il tend vers zéro quand on raffine (section 7).

### Le correctif dans le code (T3)

`run.py` et `results.py` font désormais la mesure correcte, sans rien cacher du brut :

- `paper_to_sim_time_window(window_paper, rhobar)` mappe la fenêtre papier en temps sim ;
- `gamma_to_paper_units(gamma_raw_sim, rhobar)` applique `× 2π/ρ̄` ;
- `fit_growth` ajuste la fenêtre mappée (avant : la fenêtre brute, d'où l'artefact) ;
- chaque enregistrement porte `gamma_raw_sim` et `gamma_paper_units`, plus les deux fenêtres et `ρ̄`.

Le brut reste disponible pour la reproductibilité ; le nombre comparable au papier est `gamma_paper_units`.

---

## 6. Reproduire, étape par étape

### 6.1 Construire adc_cpp (module Python)

```bash
cd adc_cpp
cmake -B build -G Ninja -DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3)
ninja -C build _adc
export PYTHONPATH=$PWD/build/python
```

### 6.2 Vérifier le modèle (oracle analytique, sans run)

```bash
python adc_cases/hoffart_euler_poisson_dsl/check_model.py      # résidu DSL vs analytique = 0
python adc_cases/hoffart_euler_poisson_dsl/diag/petri_eigenvalue.py   # cibles 0.772/0.911/0.683 a <0.5%
```

### 6.3 Mesurer les taux (chemin fidèle, mesure paper-faithful)

```bash
python adc_cases/hoffart_euler_poisson_dsl/run.py --engine system-schur \
    --n 96 --t-end 10 --modes 3 4 5 --dt 2e-3 --no-gif
```

Le `t-end` doit valoir au moins 8.5 : la fenêtre mappée de `ℓ=5` est `[7.23,8.48]`. La sortie
`growth_rates.csv` porte `mode, gamma_raw_sim, gamma_paper_units, gamma_paper, relative_error_percent`.
Résultat (n=96) :

| ℓ | gamma_raw_sim | gamma_paper_units | papier | erreur |
|---|---|---|---|---|
| 3 | 0.1117 | 0.702 | 0.772 | −9.1 % |
| 4 | 0.1423 | 0.894 | 0.911 | −1.9 % |
| 5 | 0.1087 | 0.683 | 0.683 | +0.04 % |

### 6.4 L'audit de normalisation et la convergence

```bash
python adc_cases/hoffart_euler_poisson_dsl/diag/diag_normalization_audit.py 128   # échelles + décompo fenêtre
python adc_cases/hoffart_euler_poisson_dsl/diag/convergence_reduced.py            # erreur vs n
```

### 6.5 Les figures style papier

```bash
python adc_cases/hoffart_euler_poisson_dsl/diag/make_paper_figures.py 3 4 5 \
    --out adc_cases/hoffart_euler_poisson_dsl/figures
```

---

## 7. Lecture des figures

### 7.1 Snapshots du rollup (style Fig. 5.1 à 5.3)

Densité advectée par la dérive `E×B` normalisée (le champ que le modèle complet advecte, section 2.3),
n=128, jusqu'à `t_f = 10` périodes diocotron. Palette du papier : disque blanc, extérieur ardoise,
schlieren `|∇ρ|` en colormap `Blues`.

`ℓ = 3` (Fig. 5.1) : l'anneau se déforme en triangle, puis trois bras s'enroulent en trois vortex, puis
filamentation.

![Mode l=3, snapshots](figures/snapshots_l3.png)

`ℓ = 4` (Fig. 5.2) : déformation carrée, puis quatre vortex, le motif le plus net des trois.

![Mode l=4, snapshots](figures/snapshots_l4.png)

`ℓ = 5` (Fig. 5.3) : pentagone, étoile à cinq branches, puis cinq vortex en couronne.

![Mode l=5, snapshots](figures/snapshots_l5.png)

Le nombre de vortex égale le mode `ℓ` dans chaque cas, et la chronologie suit celle du papier. La
résolution (n=128) est bien plus grossière que la référence du papier (`r=9`, ~12,5 millions de degrés de
liberté), donc les filaments tardifs sont moins fins, mais la dynamique est la même.

### 7.2 Animations

Les `diocotron_l{3,4,5}.gif` animent la même évolution, dans les mêmes couleurs. On y voit la rotation de
l'anneau, la croissance du mode, puis le repliement en vortex et l'étirement des filaments.

### 7.3 Taux de croissance (style Fig. 5.4)

![Taux de croissance](figures/growth_rate.png)

Panneaux (a,b,c) : `|c_ℓ(t)|/|c_ℓ(0)|` en échelle log contre `t/t_f`. La courbe bleue suit la pente
papier (tirets rouges) dans la fenêtre de fit mappée (bande bleue), puis sature vers `t/t_f ~ 5`. La
saturation n'est pas une instabilité numérique : l'amplitude ne croît plus que faiblement et reste finie
bien au-delà ; c'est la fin du régime linéaire. Panneau (d) : `γ_ℓ` contre le mode, pour le papier, le
modèle complet (T3) et la dérive réduite. Les trois courbes se suivent à environ 10 % près à cette
résolution.

### 7.4 Convergence

![Convergence](figures/convergence.png)

Erreur relative au papier contre `n`, de 64 à 256. L'écart décroît vers zéro : `ℓ=3` passe de 13.7 % à
0.6 %, `ℓ=4` de 13.8 % à 0.2 %, `ℓ=5` reste sous le pour-cent. À n=256 les trois modes sont à moins de
1 %. Cela confirme que le résidu basse-résolution était la discrétisation du bord d'anneau cartésien, pas
un verrou : la reproduction converge vers le papier.

---

## 8. Ce que ce cas démontre, et ses limites

Le chemin volumes finis cartésien uniforme, Strang SSPRK3, source à complément de Schur, reproduit les
taux diocotron du papier quand la mesure est faite dans les bonnes unités, et converge vers eux en
résolution. Le verrou n'était pas le cœur `adc_cpp` mais la métrologie du cas (`run.py`/`results.py`).

Limites à garder en tête. La reproduction est partielle au sens où, à résolution finie, un résidu de
grille subsiste (il décroît avec `n`). Le mode `ℓ=5` est sensible à la fenêtre de fit (±27 à 29 % selon la
fenêtre), donc son accord à moins de 1 % est en partie favorable ; les modes les plus robustes sont
`ℓ=4` puis `ℓ=3`. Les snapshots utilisent la dérive `E×B` réduite, qui partage le champ d'advection du
modèle complet mais n'en est pas une exécution complète ; les taux quantitatifs, eux, viennent du modèle
complet `system-schur`. Le chemin polaire du modèle complet diverge encore (non-positivité au bord
d'anneau) et fait l'objet d'un chantier séparé.

Pour aller plus loin : monter la résolution (n=192, 256) resserre encore le résidu ; porter la mesure
paper-faithful vers les chemins AMR/MPI/GPU est une extension de performance, plus un correctif physique.

### Références internes

- `model.py`, `check_model.py` : le modèle DSL et son oracle bit-exact.
- `diag/petri_eigenvalue.py` : la valeur propre analytique (cibles + origine du 2π).
- `RESULTS_SYSTEM_SCHUR.md` : la table complète, l'audit T2, le code T3, la convergence.
- `T2_NORMALIZATION_AUDIT.md` : l'audit dimensionnel détaillé.
- `NORMALIZATION.md` : la normalisation `2π/ρ̄` sur le chemin polaire réduit.
- `adc_cpp/docs/HOFFART_FIDELITY.md` : l'audit ligne à ligne contre le papier.
