# T2 — audit de normalisation du chemin `system-schur`

Ce document clôt la question ouverte par `RESULTS_SYSTEM_SCHUR.md` (sections 7ter / conclusion) :
le déficit −95 % du run hoffart `system-schur` (`gamma_raw ≈ 0.032`, fenêtres papier,
`alpha = omega = 1e12`) ne vient pas de la géométrie cartésienne ; il se décompose entièrement en
facteurs dimensionnels dérivés *à l'avance* (pas ajustés après coup), mesurés par
`diag/diag_normalization_audit.py`.

Résultat en une ligne : le déficit l=3 (×24.7 entre 0.0312 et 0.772) est le produit de trois facteurs :

    déficit = (fenêtre 3.20×) × (T_d = 2 pi = 6.28×) × (résidu grille cart vs polaire 1.23×) = 24.7×

Les deux premiers sont de la métrologie/normalisation (récupérables) ; seul le troisième (~20 %) est
une vraie différence physique de grille. Ce n'est donc pas de la métrologie pure (le 2 pi est réel mais un
résidu ~20 % subsiste), ni une limitation géométrique fondamentale : la reproduction
cartésienne est atteignable.

---

## 1. La clé dimensionnelle : `alpha/omega = 1`, le `1e12` se simplifie

`model.py` fixe (params papier, `rho_max = 1`) :

    alpha = beta^2 / rho_max = 1e12          (charge du Poisson : -Delta phi = alpha rho)
    omega = beta^2           = 1e12          (= |Omega| = omega_c, le champ B_z)

La vitesse de dérive du run complet (`build_uniform` → `drift_velocity_from_potential`) est
`v = grad(phi)/omega` avec `-Delta phi = alpha rho`. En posant `phi = alpha phi~` (donc
`-Delta phi~ = rho`) :

    v = (alpha/omega) grad(phi~) ,   alpha/omega = 1/rho_max = 1
    => v == grad(phi~) == EXACTEMENT la dérive ExB NORMALISEE (alpha = 1, B = 1).

Le `1e12` de `alpha` et le `1e12` de `omega` se simplifient dans le transport. Le run complet
(`alpha = omega = 1e12`) et le réduit ExB normalisé (`B0 = 1`, `charge = 1`, le chemin validé de
`diag_polar_omega.py`) advectent `rho` avec le même champ de vitesse, dans les mêmes unités de
temps de simulation. `gamma_raw` est donc directement comparable entre les deux, et la seule
différence possible entre `0.032` (run complet) et `~0.10` (réduit cart, RESULTS §7ter) est la
fenêtre de fit, pas une échelle physique.

> Confirmation croisée : le réduit ExB normalisé, fitté dans la fenêtre papier l=3 `[0.40,0.70]`
> *appliquée en temps de simulation*, donne `gamma_raw = 0.0312`, à 3 % de la mesure du run complet
> (RESULTS §1, n=128 : `0.0321`). L'équivalence full == réduit (RESULTS §7) est donc reconfirmée ici par
> les nombres bruts eux-mêmes.

## 2. Les échelles diocotron

Avec `rho_max = 1` :

| grandeur | définition | valeur |
|---|---|---|
| `omega_c = \|Omega\|` | `beta^2` | `1e12` (cyclotron, échelle RAPIDE) |
| `omega_d` | `rho_max * alpha / \|Omega\|` | **`1`** (diocotron/dérive, échelle LENTE) |
| `T_d` | `2 pi / omega_d` | **`2 pi ≈ 6.283`** (période diocotron) |
| `alpha/omega` | seule combinaison a-dimensionnée | **`1`** (`= 1/rho_max`) |

`omega_d = rho_max (beta^2/rho_max) / beta^2 = 1` : le `beta^2` se simplifie, la dynamique lente vit en
unités O(1). `T_d = 2 pi` est le facteur `2 pi` du dépôt (`NORMALIZATION.md`,
`diag_polar_omega.py:35`) : c'est la période diocotron, pas un fudge.

## 3. Les candidats de scaling s'effondrent tous sur `× 2 pi`

Les quatre candidats demandés (T2), appliqués au `gamma_raw` établi (l=4, fenêtre `[3,12]`,
n=128, `gamma_raw = 0.1135`) :

| candidat | formule | justification dimensionnelle | valeur |
|---|---|---|---|
| c1 | `gamma_raw * 2 pi` | conversion temps sim → temps papier via `T_d` | **0.7132** |
| c2 | `gamma_raw * 2 pi * (alpha/omega)` | `alpha/omega = 1` → **identique à c1** | 0.7132 |
| c3 | `gamma_raw / omega_d` | `omega_d = 1` → **no-op** | 0.1135 |
| c4 | `gamma_raw * T_d` | `T_d = 2 pi` → **identique à c1** | 0.7132 |
| — | cible papier l=4 | — | 0.9110 |

Conclusion §3 : tous les candidats dimensionnellement honnêtes s'effondrent sur `gamma_raw * 2 pi`
(car `alpha/omega = 1`, `omega_d = 1`, `T_d = 2 pi`). Il n'existe aucun facteur ~3 supplémentaire au
niveau dimensionnel. `c1` donne `0.713`, soit ~22 % sous le papier (0.911), exactement le résidu
grille cart-vs-polaire de RESULTS §7ter (cart ×2π `0.72` vs polaire `0.90`). Le `× 2 pi` est donc le
seul facteur de normalisation légitime, et il n'est pas le verrou : appliqué au `gamma_raw` établi
il reproduit à ~20 %.

## 4. Le « résidu ~3× » est la fenêtre de fit (mesuré)

`run.py:fit_growth` masque le temps de simulation directement avec `PAPER_FIT_WINDOWS`
(`[0.40,0.70]`, …). Mais `temps_papier = T_d × temps_sim` : la fenêtre papier appliquée en temps sim
tombe dans le transitoire (taux encore en rampe, cf. RESULTS §3 : taux local 0.03→0.11 sur
`t ∈ [0.5, 2.5]`), pas dans l'exponentielle établie. Mesure (`diag_normalization_audit.py`, n=128,
même run, deux fenêtres) :

| l | fenêtre papier (sim) | `gamma_raw` (papier) | fenêtre établie `[3,12]` | `gamma_raw` (établi) | **ratio établi/papier** |
|---|---|---|---|---|---|
| 3 | `[0.40,0.70]` | **0.0312** | `[3.0,12.0]` | **0.0998** | **3.20** |
| 4 | `[0.60,0.75]` | 0.0943 | `[3.0,12.0]` | 0.1135 | 1.20 |
| 5 | `[1.15,1.35]` | 0.1056 | `[3.0,12.0]` | 0.1137 | 1.08 |

Le ratio 3.20 (l=3) est le « résidu ~3× au-delà du 2 pi ». C'est un effet de fenêtre, pas une
échelle manquante : la fenêtre papier l=3 est la plus précoce (`[0.40,0.70]`), donc la plus enfoncée
dans le transitoire → facteur le plus grand. Pour l=4 / l=5 les fenêtres papier sont plus tardives
(`[0.60,0.75]`, `[1.15,1.35]`) → le facteur fenêtre tombe à 1.20 / 1.08. C'est pourquoi le déficit
était maximal en l=3 (−95.5 %) et moindre en l=5 (−83 %).

## 5. Décomposition complète du déficit (l=3) : elle ferme exactement

| facteur | de → vers | valeur | nature |
|---|---|---|---|
| fenêtre de fit | `gamma_raw` papier `0.0312` → établi `0.0998` | **3.20×** | métrologie (run.py fitte le transitoire) |
| `T_d = 2 pi` | `0.0998` → `0.627` | **6.28×** | métrologie (période diocotron) |
| grille cart vs polaire | `0.627` → papier `0.772` | **1.23× (~20 %)** | physique (seul résidu NON métrologique) |
| **produit** | `0.0312` → `0.772` | **24.7×** | == déficit −95.5 % observé |

`3.20 × 6.28 × 1.23 = 24.7` : la décomposition ferme exactement le déficit l=3 mesuré
(`0.772 / 0.0312 = 24.7`).

## 6. Verdict T2

- Le déficit ne vient pas de la géométrie cartésienne. Le `T_d = 2 pi` et le facteur fenêtre sont de la
  normalisation/métrologie récupérable (≈ 20× des ~24.7×). Reproduction cartésienne atteignable.
- Ce n'est pas de la métrologie pure non plus : après les deux facteurs `2 pi` (temps + fenêtre),
  il reste un résidu ~20 % (grille cart `0.72` vs polaire `0.90` ×2π) qui est une vraie différence
  physique de discrétisation azimutale, pas un facteur cosmétique.
- Aucun facteur dimensionnel ~3 n'existe : `alpha/omega = 1`, `omega_d = 1`, `T_d = 2 pi`, tous les
  candidats s'effondrent sur `× 2 pi`. Le « résidu 3× » était la fenêtre de fit, maintenant
  quantifié (ratio 3.20 pour l=3, §4).

### Implication actionnable pour `run.py` → **FAIT (T3)**
La mesure du chemin `system-schur` fittait la fenêtre papier en temps de simulation, donc dans le
transitoire. T3 (juin 2026) corrige ceci dans le code : `run.py:fit_growth` fitte désormais la
fenêtre papier mappée (`fenêtre_sim = 2 pi/rhobar × fenêtre_papier`) et `results.py` reporte les
deux `gamma_raw_sim` et `gamma_paper_units = gamma_raw_sim × 2 pi/rhobar` (le brut est conservé pour la
reproductibilité). Helpers : `paper_to_sim_time_window`, `gamma_to_paper_units`.

### 7. Vérification directe sur le full system-schur (pas le proxy réduit)
Les §4-5 utilisent le réduit ExB comme proxy (justifié par `α/ω=1`). T3 mesure le vrai full
system-schur (Strang ssprk3 + CondensedSchur, drift-seedé) avec les fenêtres mappées (n=96,
t_end=10) :

| l | fenêtre sim mappée | `gamma_raw_sim` | `gamma_paper_units` (×2π) | papier | erreur |
|---|---|---|---|---|---|
| 3 | [2.513,4.398] | 0.1117 | **0.702** | 0.772 | **−9.1 %** |
| 4 | [3.770,4.712] | 0.1423 | **0.894** | 0.911 | **−1.9 %** |
| 5 | [7.226,8.482] | 0.1087 | **0.683** | 0.683 | **+0.04 %** |

Le full reproduit le papier à −9 / −2 / +0 % avec les fenêtres mappées (mieux que la fenêtre établie
`[3,9]` : l=5 passe de +13 % à +0.04 %, sa fenêtre tardive captant la même phase que le papier). Le full
suit le réduit à ~2 % en fenêtre établie (le proxy était valide). Caveats (revue adversariale) : le
2 π est exact/mode-indépendant (Petri <0.5 %) ; le résidu ~0-9 % est grille/résolution(n=96)/roll-off de
fenêtre (pas de plateau scale-free, lissage WENO5 ≠ saturation) ; l=5 est sensible à la fenêtre
(±27-29 %), donc son +0.04 % est en partie fortuit, mener avec l=3/l=4. Détail : `RESULTS_SYSTEM_SCHUR.md`
section 9.

## Reproduire

```bash
PYTHONPATH=<adc_cpp>/build-master/python \
    python hoffart_euler_poisson_dsl/diag/diag_normalization_audit.py 128
```

Sortie : les échelles dimensionnelles, le tableau fenêtre-papier vs établie par mode (le ratio est le
facteur fenêtre) et l'effondrement des 4 candidats sur `× 2 pi`. Voir aussi `NORMALIZATION.md` (chemin
polaire validé) et `RESULTS_SYSTEM_SCHUR.md` §7ter (renversement géométrie).
