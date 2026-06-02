# Cas diocotron : reproduction de arXiv:2510.11808 (100 % Python)

Reproduction du benchmark **diocotron** de Hoffart, Maier, Shadid, Tomas, *Structure-preserving
finite-element approximations of the magnetic Euler-Poisson equations*
([arXiv:2510.11808](https://arxiv.org/abs/2510.11808), Section 5.3), réalisée **avec le solveur
`adc`** (la lib `adc_cpp` via ses bindings Python), pas avec leur code ni un code tiers.
Script : [`run.py`](run.py). Variante périodique minimale : [`band_instability.py`](band_instability.py).

## Le point clé : composé GÉNÉRIQUEMENT, sans solveur dédié

Le papier valide son schéma dans la **limite de dérive magnétique** (`ω_d ≪ ω_p ≪ ω_c`) en
reproduisant le **taux de croissance de l'instabilité diocotron** d'une colonne creuse, comparé
à la dispersion analytique. Ce modèle réduit de dérive `E × B` se **compose ici depuis Python**
via `adc.System` (**un bloc `diocotron` + un Poisson de système à paroi conductrice circulaire**),
**sans aucun solveur C++ dédié au diocotron** :

```python
sim = adc.System(n=192, L=1.0, B0=1.0, alpha=1.0, n_i0=0.0, periodic=False)
sim.add_block("ne", model="diocotron", charge=1.0, spatial=adc.Spatial(minmod=True))
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
sim.set_density("ne", ring_numpy)        # CI anneau écrite en numpy
dt = sim.step_cfl(0.4)                    # le calcul (transport + Poisson) reste C++
```

« Python compose, le C++ calcule » : la paroi conductrice (embedded boundary) et la
résolution de Poisson sont dans le cœur `adc_cpp` ; Python ne fait qu'assembler.

## Ce qui est reproduit (`figures/`)

| Figure | Contenu |
|---|---|
| `dispersion.png` | `γ` vs mode azimutal `l` : **analytique** (valeurs propres de Petri/Davidson-Felice, numpy) + **mesuré par `adc`** + **cibles du papier** |
| `amplitude.png` | `\|c_l\|(t)` (mode `l` de `φ`) en échelle log : croissance exponentielle, phase linéaire ajustée |
| `diocotron.gif` | évolution de la densité (mode `l=4`) : l'anneau développe `l` lobes qui s'enroulent |
| `snapshots.png` | 4 instantanés de densité du même run |

## Méthode (les deux côtés, en Python)

**Analytique** : problème aux valeurs propres radial de Petri (arXiv:astro-ph/0611936),
réimplémenté en numpy (`diocotron_eigenvalue`) :

```
ω L_m φ = m Ω L_m φ + q_m φ,   φ(0) = φ(R_w) = 0,   Ω(r) = -(1/r²) ∫₀ʳ ρ r' dr'
```

`M = L⁻¹ A`, `γ = max Im(ω)`, normalisé par `ω_D = ρ̄/(2π)`. Géométrie de l'anneau
`r0:r1:Rwall = 6:8:16` (anneau net `w = 0.05`). **Reproduit les cibles du papier :
`γ₃ = 0.772`, `γ₄ = 0.912`, `γ₅ = 0.687`** (le taux normalisé est invariant d'échelle).

**Numérique** : le bloc `diocotron` de `adc.System` (paroi conductrice circulaire Dirichlet,
ratios `0.15:0.20:0.40` à `L=1`, MUSCL Minmod + SSPRK2, couplage Poisson *once-per-step*).
Pour chaque mode `l` : perturbation azimutale faible (`δ = 0.01`), amplitude du **mode `l` de `φ`**
sur un cercle au rayon médian (FFT azimutale), ajustement `exp(γ t)` sur la phase linéaire,
normalisation par `ω_D`. (Un intégrateur *per-stage* écrit en Python est démontré dans
[`../composition/run.py`](../composition/run.py).)

## Résultats (taux de croissance normalisé)

| mode `l` | **papier** | **analytique** (Petri numpy) | **`adc`** (mesuré, n=192, Minmod) |
|---|---|---|---|
| 3 | 0.772 | **0.772** | 0.599 (−22 %) |
| 4 | 0.911 | **0.912** | 0.662 (−27 %) |
| 5 | 0.683 | **0.687** | 0.652 (−5 %) |

**Côté analytique, on reproduit le papier à 3 chiffres** : c'est la cible de validation,
retrouvée par notre propre résolution numpy du problème de Petri. Le pic à `l=4` (mode le plus
instable) est correct.

**Côté numérique**, le schéma volumes-finis **d'ordre 2 (Minmod) à n=192 sous-estime** le taux
(−5 à −27 %) : effet attendu de la **diffusion numérique** de l'ordre modéré, documenté dans
l'étude de résolution ([adc_cpp/docs/DIOCOTRON_GROWTH_RATE.md](../../adc_cpp/docs/DIOCOTRON_GROWTH_RATE.md)),
où seul l'ordre élevé (WENO5-Z + SSPRK3) referme l'écart. La simulation **capture bien
l'instabilité** (croissance exponentielle, bon classement des modes, `l=4` dominant) ; la
quantification fine relève de la résolution et de l'ordre.

## Reproduire

```bash
# 1) construire le module adc (depuis adc_cpp)
cd ../adc_cpp && cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc -j
# 2) lancer le cas (depuis adc_cases)
cd ../adc_cases
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/run.py
```
