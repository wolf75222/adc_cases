# hyqmom15 : modele 2D a 15 moments (fermeture HyQMOM), flux valide contre RIEMOM2D

## 0. Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` |
| Entrees | 10 etats figes `golden/golden_states.csv` (4 gaussiennes dont correlee et haut Mach ~ 20, 5 melanges discrets dont un quasi-degenere C20 ~ 1e-6, 1 gaussienne anisotrope C20/C02 = 100) ; aucun maillage : validation ponctuelle du flux. |
| Sorties | aucun fichier produit (asserts) ; `.so` production et aot compiles dans `out/hyqmom15/`. |
| Invariants garantis | (1) `eval_flux` == `Flux_closure15_2D.m` sur 10 etats x {Fx, Fy}, rtol 1e-12 + atol 1e-13 x echelle de l'etat (`run.py:89-101`) ; (2) les 6 entrees d'ordre 5 == moments bruts exacts d'Isserlis sur 4 gaussiennes, rtol 1e-12 (`run.py:104-118`) ; (3) les 20 recopies d'ordre <= 4 bit-identiques a U, `np.array_equal` (`run.py:121-130`) ; (4) `check_model` passe sur les 10 etats realisables, compilation AOT chronometree SANS assert mural (`run.py:132-150`) ; (5) contraste degenere C20 = 0 : flux `bit_match` divergent ET flux `robust` fini (`run.py:153-168`) ; (6) anti-derive goldens (`run.py:70-78`) + borne k*sqrt(C) confrontee aux vraies vitesses de `golden_vp.csv` : gaussiennes a sqrt(6)*sqrt(C) exactement, au moins un melange DEPASSE la borne (`run.py:170-202`). |
| Prouve | la transcription DSL du pipeline de fermeture (M -> C -> S -> closureS5 -> C5 -> M5) est correcte au sens du code MATLAB de reference, execute reellement (Octave) et non re-transcrit ; la fermeture est exacte sur les gaussiennes ; le modele compile et se lie par les chemins production et aot. |
| Ne prouve pas | la stabilite d'une evolution temporelle (aucun pas de temps ici : drivers ADC-84/85) ; les vitesses d'onde HLL exactes (ADC-87/88 ; `golden_vp.csv` ne sert ici qu'a ENCADRER la borne bring-up, pas a valider un calcul de vitesses du modele) ; la SURETE de la borne bring-up `|u| + 3 sqrt(C)` hors gaussiennes : l'invariant (6) DEMONTRE qu'elle est depassee par des melanges realisables (pire ratio 3.29 sur le jeu, non borne pres de la frontiere de realisabilite) -- demarrage Rusanov uniquement, jamais production ; le mode `robust` au-dela de la finitude (les planchers n'existent pas dans le MATLAB) ; l'execution du `.so` dans un `System` (compilation et chargement seulement, pas de pas de temps) ; les chemins device GPU/Kokkos et MPI. |
| Provenance | RIEMOM2D `0f2a196`, GNU Octave 11.3.0 (aarch64-darwin), adc_cpp `4bb7cec`, backends production + aot, macOS ; cout : ~10 s (compilations comprises). |

## 1. Physique : pourquoi 15 moments et une fermeture

Le systeme est la hierarchie des moments cartesiens d'ordre <= 4 de l'equation de Vlasov 2D
(document maths `main.pdf`, eq. 1.2) : pour chaque moment $M_{pq} = \int f\, v_x^p v_y^q\, dv$,

$$\partial_t M_{pq} + \partial_x M_{p+1,q} + \partial_y M_{p,q+1} = \text{sources},$$

le flux du moment d'ordre maximal fait donc apparaitre des moments d'ordre 5 qui ne sont pas
dans le vecteur d'etat : c'est le probleme de fermeture. La fermeture HyQMOM (Bryngelson, Fox
et Laurent 2025, hal-05398171, eq. 1.10-1.12 du document) exprime les six moments standardises
d'ordre 5 en fonction des moments d'ordre inferieur et rend le systeme globalement hyperbolique
(valeurs propres reelles). Justifie la clause Prouve (1) : ce cas verifie la transcription de
cette fermeture, pas sa physique.

## 2. Equations et table des couches

Etat (ordre du document et du MATLAB, 0-based) :

```
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
```

Flux : `Fx = [M10 M20 M30 M40 M50 M11 M21 M31 M41 M12 M22 M32 M13 M23 M14]` et
`Fy = [M01 M11 M21 M31 M41 M02 M12 M22 M32 M03 M13 M23 M04 M14 M05]`. 20 entrees sur 30
recopient une composante de U (justifie l'invariant (3)) ; les 6 moments d'ordre 5 distincts
(M50, M41, M32, M23, M14, M05) sont reconstruits par la fermeture.

| Ligne | Couche | Ce qui se passe |
|---|---|---|
| `run.py:215` `build_moment_model()` | Python compose | construction du modele symbolique, fermeture choisie par callable |
| `model.py:89-203` (48 `m.primitive(...)` + `m.flux(...)`) | expressions DSL compilees | le pipeline M -> C -> S -> closureS5 -> C5 -> M5 fige en C++ genere |
| `m.compile(..., backend="aot")` (`run.py:136-150`) | brique compilee | flux evalue par cellule sans callback Python |

## 3. Code : le pipeline de fermeture, fonction par fonction

Tout est dans [model.py](model.py) ; chaque etage est un let-binding `m.primitive(nom, expr)`,
donc une variable locale nommee du C++ genere, et les formules aval ne referencent que des
feuilles (codegen lineaire en la taille du pipeline, 48 primitives, construction < 0.01 s).

- `model.py:102-107` : vitesses moyennes `ux = M10/M00`, `uy = M01/M00` et leurs puissances.
- `model.py:110-133` : les 12 moments centres C20..C04 (transcription de `M4toC4.m`, reecrits
  en `ux`/`uy` ; algebriquement identiques, d'ou la tolerance d'arrondi rtol 1e-12 du golden et
  non l'egalite bit).
- `model.py:136-148` : standardisation `S_ij = C_ij / (C20^(i/2) C02^(j/2))` (`M2CS4_15.m`),
  via `sC20 = sqrt(C20)` et des produits entiers (jamais de `pow` fractionnaire).
- `model.py:61-78` `hyqmom_closure` : transcription LITTERALE de `closureS5.m` (forme
  polynomiale). Attention de transcription : les variantes `Moments5.m` et `S5_2D.m` du depot
  MATLAB different sur S32/S23 ; le chemin de flux de reference appelle `closureS5`, c'est elle
  qui est transcrite et le golden (1) detecterait toute derive.
- `model.py:160-165` : de-standardisation `C_ij = S_ij sC20^i sC02^j` (`Flux_closure15_2D.m`
  lignes 55-62).
- `model.py:169-186` : moments bruts d'ordre 5 (`C5toM5.m`, seules les 6 entrees d'ordre 5 ;
  les entrees d'ordre <= 4 du round-trip MATLAB se simplifient exactement en les composantes
  de U, assemblees comme recopies directes en `model.py:187-198`).
- borne de vitesse bring-up `u +- 3 sqrt(C)` (`m.eigenvalues`, fin de `build_moment_model`) :
  les vraies vitesses (eigenvalues15_2D, `golden_vp.csv`) valent EXACTEMENT `u +- sqrt(6) sqrt(C)`
  sur une gaussienne (k = 3 couvre, marge ~22 %) mais des melanges asymetriques realisables
  depassent k = 3 (pire ratio 3.29 sur le jeu golden) -- voir Ne prouve pas et l'invariant (6).
- Mode `robust=True` (`model.py:96-98`) : plancher lisse `max(x, eps) = ((x+eps)+|x-eps|)/2`
  sur M00, C20, C02. Hors MATLAB (qui ne protege rien : division par M00 et sqrt(C20)
  inconditionnels, `closureS5.m` test p2p2 commente) ; smoke de finitude seulement.

## 4. Maths : les deux oracles independants

Golden MATLAB : les goldens sont produits par `golden_gen.m` qui EXECUTE le depot de reference
(Octave, `--path RIEMOM2D`), pas par une re-transcription Python (une re-transcription
partagerait ses fautes avec le modele et ne prouverait rien). Commande exacte :

```
python3 gen_states.py
octave --no-gui --path /chemin/vers/RIEMOM2D golden_gen.m
```

Oracle gaussien : pour une gaussienne 2D de covariance $[[C_{20}, C_{11}], [C_{11}, C_{02}]]$,
tout moment centre d'ordre impair est nul (Isserlis), donc les six $C$ d'ordre 5 sont nuls et
les moments bruts $M_{pq}$ d'ordre 5 ont la forme fermee du binome
$M_{pq} = \rho \sum_{ij} \binom{p}{i}\binom{q}{j} u_x^{p-i} u_y^{q-j} C_{ij}$
(`model.py:238-254`, calcule sans le pipeline). La fermeture HyQMOM est exacte sur ce cas :
avec $S_{30} = S_{21} = S_{12} = S_{03} = 0$, chaque formule de `closureS5` s'annule terme a
terme, y compris pour $C_{11} \neq 0$ (etat correle n. 3). L'oracle verifie donc le pipeline
complet de bout en bout sur une famille a 6 parametres, independamment du MATLAB.

Realisabilite des etats de test : les melanges discrets $f = \sum_k w_k \delta(v - v_k)$
(`model.py:257-266`) sont des distributions, leurs moments sont realisables par construction ;
c'est ainsi qu'on obtient des etats fortement asymetriques (S30 != 0) et le quasi-degenere
(trois points resserres a 1e-3 en vx : C20 ~ 1e-6, test de cancellation des sqrt).

## 5. Limites et suite

La validation est ponctuelle (flux en un etat) : rien ici ne fait avancer un pas de temps, ne
resout Poisson, ni ne calcule de vitesses d'onde HLL. Suite prevue (epic ADC-81) : sources et
driver crossing (ADC-84), Poisson et diocotron (ADC-85), vitesses signees generiques sans
primitive p (ADC-83), vitesses exactes par jacobienne autodiff + eig par blocs (ADC-87/88,
`golden/golden_vp.csv` deja produit ici par `eigenvalues15_2D(M, 1)` chemin flagsym=1).
