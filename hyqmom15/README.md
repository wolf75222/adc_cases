# hyqmom15 — Vlasov–Poisson 2D à 15 moments (fermeture HyQMOM)

Modèle cinétique 2D : on transporte les moments en vitesse `M_pq = ∫ f v_x^p v_y^q dv`
d'ordre p+q ≤ 4 (15 composantes) de l'équation de Vlasov, couplés au Poisson du système.
Pour chaque moment,

```
∂t M_pq + ∂x M_{p+1,q} + ∂y M_{p,q+1} = q/m (p Ex M_{p-1,q} + q Ey M_{p,q-1})
                                        + Ωc (p M_{p-1,q+1} − q M_{p+1,q-1})
```

Le flux du dernier ordre fait apparaître des moments d'ordre 5 absents du vecteur d'état :
c'est le problème de fermeture. La fermeture HyQMOM (Bryngelson, Fox & Laurent 2025,
hal-05398171) exprime les six moments standardisés d'ordre 5 en fonction des ordres
inférieurs et rend le système hyperbolique.

État (ordre des composantes partagé avec la référence MATLAB RIEMOM2D) :

```
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
```

## Démarrage rapide

```bash
# depuis la racine d'adc_cases, module adc construit (PYTHONPATH vers build-*/python)
python hyqmom15/run.py            # flux vs goldens MATLAB + oracle gaussien
python hyqmom15/run_waves.py      # vitesses d'onde exactes vs goldens
python hyqmom15/run_crossing.py   # sources E/B, rotation de Larmor, croisement de jets
python hyqmom15/run_diocotron.py  # Vlasov-Poisson complet : anneau diocotron
python hyqmom15/run_relaxation.py # projection de réalisabilité + crossing Ma=20
```

## Composer le modèle

Tout passe par `build_moment_model` ([model.py](model.py)), qui délègue l'algèbre des
moments au générateur générique `adc.moments` d'adc_cpp. La seule physique écrite ici est
la fermeture — un callable qui reçoit les moments standardisés et rend ceux d'ordre 5 :

```python
from model import build_moment_model, hyqmom_closure

m = build_moment_model(
    closure=hyqmom_closure,   # la physique : S (ordres 2-4) -> S50..S05
    exact_speeds=True,        # vitesses d'onde par valeurs propres du jacobien (HLL fidèle)
    with_sources=True,        # sources électriques (lit grad phi) + magnétique (omega_c)
    debye=0.04,               # couplage Poisson : Delta phi = (M00 - rho_background)/debye^2
    rho_background=rho_bg,    # fond neutralisant = moyenne de M00 (obligatoire en périodique)
    omega_p=25.0,             # borne le pas de temps de la source
)
compiled = m.compile(so_path, include_dir, backend="aot")
sim = adc.System(n=128, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="fft")
```

Écrire un autre système de moments = fournir un autre callable de fermeture (même
contrat) et, si on veut les vitesses exactes par sous-blocs, une partition du jacobien
(`HYQMOM_BLOCKS` pour celui-ci).

### Options utiles

| Option | Effet |
|---|---|
| `robust=True` | planchers lisses sur M00, C20, C02 (divisions et racines protégées). Le défaut `False` reproduit le MATLAB, qui ne protège rien. |
| `exact_speeds=False` | borne de vitesse `u ± 3·√C` au lieu des valeurs propres exactes. Suffit pour démarrer en Rusanov ; des états réalisables la dépassent (vérifié par run.py), donc jamais pour HLL. |
| `solver=` (drivers) | `fft` (direct périodique), `fft_spectral` (symbole continu, exact sur les sinusoïdes), `geometric_mg` (général, requis en MPI). |

## Réalisabilité : le point qui pique

Un vecteur de moments doit rester celui d'une distribution positive (réalisabilité,
testée par la plus petite valeur propre de la matrice `p2p2`). Le schéma ne la préserve
pas : sans correction, l'état dérive hors de l'ensemble réalisable, les valeurs propres
du jacobien explosent et le pas CFL s'effondre (mesuré : dt ÷ 200 sur un run diocotron).

La parade est la projection `relaxation15` ([relaxation.py](relaxation.py)), appliquée à
chaque pas : clamps des moments standardisés puis relaxation vers une cible réalisable.
Le portage suit le MATLAB branche à branche ; le test « valeurs propres complexes »
utilise le jacobien autodiff du modèle. Usage par champ :

```python
from relaxation import make_corner_eigs, relax_field
fn = make_corner_eigs()
U = relax_field(U, lamin=1e-12, Ma=4.0, corner_eigs=fn)   # (15, ny, nx) -> projeté
```

C'est une boucle Python par cellule : adapté à la validation et aux runs modérés,
pas aux campagnes GPU (chemin compilé à venir côté adc_cpp).

## Validation

Les références sont générées en exécutant le vrai code MATLAB (RIEMOM2D) sous Octave —
jamais re-transcrites :

```bash
python3 gen_states.py
octave --no-gui --path /chemin/vers/RIEMOM2D golden_gen.m        # flux + valeurs propres
octave --no-gui --path /chemin/vers/RIEMOM2D golden_hll_gen.m    # trajectoire HLL (crossing)
octave --no-gui --path /chemin/vers/RIEMOM2D golden_relax_gen.m  # relaxation15 (5 branches)
```

Ce que les drivers garantissent, chiffres en CI :

- flux ≡ `Flux_closure15_2D.m` à 1e-12 sur 10 états (gaussiennes, mélanges, quasi-dégénéré),
  et fermeture exacte sur les gaussiennes (oracle d'Isserlis indépendant) ;
- vitesses d'onde ≡ `eigenvalues15_2D.m` à ~1e-11 (l'état quasi-dégénéré est jugé à
  l'aune de son conditionnement, mesuré dans le test) ;
- sources ≡ les équations explicites du document de référence à 1e-14 ; rotation de
  Larmor ≡ analytique ;
- trajectoire : en rejouant les pas de temps du golden HLL avec `time='euler'`, l'écart
  L2 au MATLAB après 20 pas est ~1e-16 (le schéma MATLAB — split additif + Euler — est
  algébriquement l'Euler non-splitté) ; en ssprk2 l'écart est 4 %, c'est l'ordre 2 ;
- Poisson : φ ≡ analytique sur sinusoïde (1e-14 en `fft_spectral`), champ E de la source
  ≡ −∇φ centré à 1e-16, checkpoint/restart bit-identique ;
- `relaxation15` ≡ Octave à 4e-14 sur 12 états couvrant les 5 branches ; à Ma=20 le
  contraste projeté/nu se mesure en réalisabilité (13 % vs 52 % de cellules violées).

## Limites

- La fermeture est exacte sur les gaussiennes mais le schéma ne préserve pas la
  réalisabilité : runs longs ⇒ projeter (section ci-dessus).
- `riemann="hllc"`/`"roe"` indisponibles : pas d'onde de contact ni d'eigenstructure
  fermée pour ce système.
- La collision BGK du MATLAB (`collision15.m`) n'est pas portée.
- Taux de croissance diocotron vs un golden MATLAB long : campagne dédiée, hors CI.
