# euler_poisson : Euler compressible couple a Poisson, attractif vs repulsif

Deux runs Euler-Poisson 2D identiques au signe du couplage pres : `sign=+1` (auto-gravite,
attractif) et `sign=-1` (charge d'espace, repulsif). Le cas verifie par `assert` trois invariants
structurels (masse conservee, impulsion nette nulle, derive d'energie de signes opposes) et expose
une prediction falsifiable de la linearisation : la derive d'energie suit $|dE|\propto\epsilon^2$.
Ce n'est pas une reproduction d'un resultat publie.

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` (`cases_manifest.toml`, `euler_poisson/run.py`, `ci = true`, `needs = []`) |
| Entrees | grille $64^2$, $L=1$, periodique ; CI $\rho=\rho_0(1+\epsilon\cos(2\pi x/L))$, $\epsilon=0.01$, repos ($v=0$, $E=\rho/(\gamma-1)$) ; $\gamma=1.4$, $\rho_0=1$, $4\pi G=1$ (sans unites), $dt=0.004$, 20 pas ; van Leer + HLLC + SSPRK2, Poisson `geometric_mg` |
| Sorties | etat `(4,n,n)=[\rho,\rho u,\rho v,E]` lu par `get_state("gas")` ; diagnostics globaux $E_{tot}=U[3].\mathrm{sum}()$, $p_x=U[1].\mathrm{sum}()$, $p_y=U[2].\mathrm{sum}()$ ; 3 figures dans `figures/` + `figures/provenance.json` |
| Invariants garantis | les 3 `assert` de `run.py` : masse `max_rel_mass < TOL_MASS=1e-9` ; impulsion `max_mom < TOL_MOM=1e-8` ; contraste `assert_opposite_sign(dE_grav, dE_plas, min_mag=TOL_DE=1e-5)` puis `dE_grav<0` et `dE_plas>0` (`run.py:150-180`) |
| PROUVE | masse conservee a $2.6\times10^{-14}$ relatif (les deux runs) ; impulsion nette $8.9\times10^{-16}$ ; $dE_{grav}=-5.857667\times10^{-4}<0$ et $dE_{plas}=+6.137105\times10^{-4}>0$ (signes opposes, magnitude $\gg$ TOL_DE) ; la pente $|dE|$ vs $\epsilon$ vaut 2.000 (figure 2) |
| NE PROUVE PAS | pas une reproduction publiee : aucun nombre n'est confronte a un article (ni effondrement de Jeans, ni benchmark plasma). Le signe physique de $dE$ n'est pas deductible du travail $\int\rho\,v\cdot g$ (qui est positif des deux cotes, section 4.3) : il se lit sur l'assert qui passe. $E_{tot}=U[3].\mathrm{sum}()$ est l'energie fluide seule (sans potentiel de champ), c'est un proxy de signe, pas une integrale physique calibree. Regime quasi-lineaire ($\epsilon=0.01$, 20 pas) : aucune dynamique non lineaire |
| Provenance | adc_cpp `01873299`, adc_cases `7c7a3403`, backend natif serie, $64^2$, ~0.3 s 1 coeur CPU ; `figures/provenance.json` |

A la fin tu sauras : pourquoi le meme code donne deux signes de $dE$ opposes (mecanisme), pourquoi
le travail de la force ne suffit pas a predire ce signe (le paradoxe de l'energie fluide), quelle
est la prediction quantitative testable ($|dE|\propto\epsilon^2$), et ce que chaque assert
etablit reellement.

---

## 1. Le mecanisme physique (justifie PROUVE : contraste de signe)

Un fluide compressible au repos, de densite $\rho=\rho_0(1+\epsilon\cos kx)$ avec $k=2\pi/L$, cree
son propre champ de force via Poisson. Trois ingredients enchaines :

1. Champ propre. La perturbation de densite resout l'equation de Poisson de systeme
   $\nabla^2\phi = \mathrm{sign}\cdot 4\pi G\,(\rho-\rho_0)$ (brique `GravityCoupling`). Le fond
   $\rho_0$ rend le second membre de moyenne nulle, condition de compatibilite du Laplacien
   periodique.
2. Force. Le fluide ressent $g=-\nabla\phi$ (brique `GravityForce`), qui pousse la quantite de
   mouvement : $\partial_t(\rho v)=\dots+\rho g$.
3. Reponse opposee selon le signe. Pour la gravite ($\mathrm{sign}=+1$), la surdensite creuse
   un puits de potentiel et $g$ pointe vers la crete (attractif). Pour le plasma
   ($\mathrm{sign}=-1$), le second membre change de signe, $\phi$ s'inverse, et $g$ pointe loin
   de la crete (repulsif). Les deux runs partent du meme etat au repos ; seul le signe du
   couplage les separe.

Ce que ce mecanisme ne dit pas tout seul : dans quel sens va l'energie fluide $E_{tot}$. On verra
section 4 que le travail de la force est positif des deux cotes, et que le signe de $dE_{tot}$
vient d'un autre canal (la compression/detente). C'est pour cela que le cas mesure et assere le
signe au lieu de le deduire d'une formule de cours.

Le systeme reduit ici est l'Euler-Poisson electrostatique non magnetise ; le systeme complet
magnetise (force de Lorentz, etage Schur) est traite par
[`hoffart_euler_poisson_dsl`](../hoffart_euler_poisson_dsl/). Ce cas ne couvre pas le couplage
magnetique.

---

## 2. Les equations et qui les calcule (justifie : la physique est figee en C++)

Etat conservatif par cellule, 4 composantes : $U=(\rho,\rho u,\rho v,E)$.

| Bloc | Equation | Brique `adc` |
|---|---|---|
| Transport | $\partial_t\rho+\nabla\cdot(\rho v)=0$, $\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v+pI)=\rho g$, $\partial_t E+\nabla\cdot((E+p)v)=\rho\,v\cdot g$ | `CompressibleFlux` |
| Etat / EOS | $p=(\gamma-1)(E-\tfrac12\rho|v|^2)$, $\gamma=1.4$ | `FluidState(kind="compressible")` |
| Source | $g=-\nabla\phi$ ; $s[1]=\rho g_x$, $s[2]=\rho g_y$, $s[3]=\rho_u g_x+\rho_v g_y$ | `GravityForce` |
| Elliptique | $\nabla^2\phi=\mathrm{sign}\cdot 4\pi G\,(\rho-\rho_0)$ | `GravityCoupling(sign, four_pi_G, rho0)` |

C'est `adc_cases.models.euler_poisson(sign, gamma, four_pi_G, rho0)` (`models.py:48-55`), une
composition `adc.Model(state, transport, source, elliptic)`. Le mot "euler_poisson" vit dans
`adc_cases` ; cote coeur ce sont quatre briques generiques.

Table 3 couches "qui calcule quoi", chaque ligne pinnee a une ligne reelle :

| Ligne `run.py` | Couche | Ce qui se passe |
|---|---|---|
| `sim.add_block("gas", model=..., spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.Explicit())` (`run.py:94-98`) | Python compose | choix du modele, du schema (MUSCL van Leer + HLLC), de l'integrateur (SSPRK2) |
| `models.euler_poisson(sign=...)` -> `GravityForce` (`source.hpp:52-62`) / `GravityCoupling` (`elliptic.hpp:43-49`) | brique C++ fige | la formule exacte de la force ($g=-\nabla\phi$, travail sur l'energie a 4 variables) et du second membre ($\mathrm{sign}\cdot 4\pi G(\rho-\rho_0)$) |
| `assemble_rhs<VanLeer, HLLC>` + Poisson de systeme (`GeometricMG`) (`run.py:99` `set_poisson(...)`) | noyau par cellule (device) | le calcul effectif, sans callback Python dans le hot path |

La distinction natif/electrostatique : `euler_poisson` utilise `GravityForce`+`GravityCoupling`
(signe porte par la brique elliptique), pas `PotentialForce`+`ChargeDensity` (signe porte par la
charge $q$, utilises par `electron_euler`/`ion_isothermal` dans `models.py`). Les deux familles
empruntent le meme chemin numerique cote coeur (somme generique des briques elliptiques) ; seul
le second membre differe.

---

## 3. La chaine de signe effective, ligne par ligne (justifie 7 de la checklist : signe par le comportement)

Le guide interdit d'ecrire "$-\nabla^2\phi=+4\pi G(\rho-\rho_0)$ donc attractif" sans verifier : le
solveur Poisson a plusieurs couches de signe. Voici la chaine effective, telle qu'elle est codee :

1. Operateur Poisson : `poisson_operator.hpp` resout $\nabla^2\phi=f$ (le stencil ecrit
   $\mathrm{lap}=\sum/dx^2$ sans facteur, $\varepsilon=1$, cf. `elliptic_problem.hpp:24-25`). Donc
   $f>0$ tend a rendre $\phi$ convexe (un minimum local).
2. Second membre : `GravityCoupling::rhs` renvoie `sign * four_pi_G * (u[0] - rho0)`
   (`elliptic.hpp:47`). Pour la gravite ($\mathrm{sign}=+1$), une surdensite ($\rho>\rho_0$) donne
   $f>0$, donc $\phi$ a un puits sous la crete.
3. Stockage du champ : le coupler stocke $aux=(\phi,+\partial_x\phi,+\partial_y\phi)$,
   convention `GradSign::Plus` (`elliptic_problem.hpp:35-39`). C'est-a-dire `aux.grad_x = +d\phi/dx`.
4. Force : `GravityForce::apply` pose `gx = -a.grad_x` (`source.hpp:55`), soit
   $g=-\nabla\phi$. Sous la crete (puits de $\phi$), $g$ pointe vers la crete : attractif.

Cette chaine donne le signe attendu, mais elle empile 4 conventions ; on ne s'y fie pas comme
preuve. La reference est l'assert qui passe : `run.py:177` impose `dE_grav < 0`, `run.py:179`
impose `dE_plas > 0`, et le run sort `OK` (section 6). C'est cela le signe physique du cas, pas la
derivation ci-dessus, qui ne sert qu'a montrer que le code est coherent.

---

## 4. Maths : pourquoi le signe de $dE$ ne se lit pas sur le travail (justifie NE PROUVE PAS)

### 4.1 Ce que $E_{tot}$ est reellement

`run.py:85` : `E_tot = U[3].sum()`. La composante 3 de l'etat est l'energie fluide totale
$E=\tfrac12\rho|v|^2+\dfrac{p}{\gamma-1}$ (cinetique + interne). Elle n'inclut pas l'energie
potentielle du champ $\tfrac12\int\rho\phi$. $E_{tot}$ n'est donc pas une quantite conservee du
systeme couple : c'est l'energie d'un des deux reservoirs (le fluide), et le champ peut lui en
donner ou lui en prendre. Mesure a $t=0$ (repos) : $E_{tot}=1.024\times10^4$, purement interne
(KE=0).

### 4.2 La decomposition mesuree (le coeur du paradoxe)

On instrumente $E_{tot}=KE+E_{int}$ sur les 20 pas, $\epsilon=0.01$ (script de diagnostic, memes
parametres que `run.py`) :

| sign | $dKE$ | $dE_{int}$ | $dE_{tot}=dKE+dE_{int}$ |
|---|---|---|---|
| $+1$ (gravite) | $+2.189\times10^{-2}$ | $-2.247\times10^{-2}$ | $-5.858\times10^{-4}$ |
| $-1$ (plasma) | $+2.412\times10^{-2}$ | $-2.351\times10^{-2}$ | $+6.137\times10^{-4}$ |

Lecture : l'energie cinetique augmente des deux cotes ($dKE>0$). La force, partant du repos,
accelere le fluide quel que soit son signe : c'est le travail $\int\rho\,v\cdot g\,dt$, et il est
positif gravite comme plasma. Le signe de $dE_{tot}$ n'est donc pas celui du travail : c'est le
residu d'une quasi-annulation entre $dKE\sim+2\times10^{-2}$ et $dE_{int}\sim-2\times10^{-2}$
(le fluide se comprime/detend differemment selon le sens de la force). Ce residu vaut $\sim6\times
10^{-4}$ et c'est lui dont le signe distingue gravite de plasma.

### 4.3 Preuve que le travail est positif des deux cotes (verifiee par sympy)

Au premier ordre, $\rho\approx\rho_0=1$, $\phi$ resout $\nabla^2\phi=\mathrm{sign}\cdot\epsilon\cos
kx$, soit $\phi=-\dfrac{\mathrm{sign}\cdot\epsilon}{k^2}\cos kx$, donc

$$g=-\partial_x\phi=-\frac{\mathrm{sign}\cdot\epsilon}{k}\sin kx,\qquad g^2=\frac{\epsilon^2}{k^2}\sin^2 kx.$$

Partant du repos, $v\approx g\,t$ aux temps courts, et la puissance volumique
$\rho\,v\cdot g\approx t\,g^2\ge 0$ est independante du signe ($g^2$ ne voit pas
$\mathrm{sign}$). Le travail cumule $\int_0^T\!\!\int\rho\,v\cdot g\,dt$ est donc strictement positif
pour $\mathrm{sign}=+1$ et $\mathrm{sign}=-1$. Ecrire "le travail $v\cdot g$ est negatif d'ou
$dE<0$" est faux. Le cas le sait : il assere $dE_{grav}<0$ comme un fait mesure, pas comme
une consequence du travail.

### 4.4 La prediction falsifiable : $|dE|\propto\epsilon^2$ (justifie PROUVE : pente 2)

A $\epsilon=0$, le second membre $f=\mathrm{sign}\cdot 4\pi G(\rho-\rho_0)$ est identiquement nul,
la force est nulle, et $dE=0$ exactement (mesure : $dE_{grav}(\epsilon{=}0)=dE_{plas}
(\epsilon{=}0)=0.0$, bit-machine). Pres de $\epsilon=0$, la force $g\propto\epsilon$ (4.3), la
vitesse $v\propto\epsilon$, donc chaque canal energetique en $v\cdot g$ ou $v^2$ est en
$\epsilon^2$. La linearisation predit donc $|dE|\propto\epsilon^2$ : doubler $\epsilon$ quadruple
$|dE|$. C'est verifiable (figure 2) et transforme un assert booleen en courbe de convergence.

Ce qu'une pente differente trahirait : pente $\approx 1$ = terme lineaire parasite (fond $\rho_0$
mal soustrait dans le second membre) ; pente $> 2$ aux grands $\epsilon$ = entree de la dynamique
non lineaire (compression d'amplitude finie). On mesure 2.000 sur $\epsilon\in[0.005,0.08]$ : le
regime est purement quadratique sur cette plage.

---

## 5. Code, fonction par fonction (justifie : ancrage reel)

`run.py` se lit du haut vers le bas. On glose les lignes porteuses ; la plomberie (import,
fallback `sys.path`) est liee, pas paraphrasee.

Condition initiale `initial_density()` (`run.py:75-79`) :

```python
x = (np.arange(N) + 0.5) * L / N                     # centres de cellules (run.py:77)
xx, _ = np.meshgrid(x, x, indexing="ij")
return RHO0 * (1.0 + EPS * np.cos(2.0 * np.pi * xx / L))   # rho = rho0 (1 + eps cos kx) (run.py:79)
```
- Perturbation cosinus de mode 1 selon $x$, invariante en $y$, amplitude $\epsilon=0.01$ autour
  de $\rho_0=1$. `set_density("gas", rho)` (`run.py:100`) ecrit $\rho$ sur la composante 0, met
  $v=0$ et $E=\rho/(\gamma-1)$ (repos thermique).

Diagnostics globaux `energy_and_momentum(sim)` (`run.py:82-85`) :

```python
U = sim.get_state("gas")                             # (4, n, n) = [rho, rho u, rho v, E]
return U[3].sum(), U[1].sum(), U[2].sum()            # E_tot, p_x, p_y (run.py:85)
```
- Sommes sur cellules des composantes conservatives, sans poids $dx^2$. Ce sont des proxys :
  suffisants pour un invariant relatif (conservation de masse) et un invariant de signe
  (contraste), insuffisants comme integrale physique absolue. Le cas ne teste que relatif et signe.

Boucle d'integration `run_case(sign, label)` (`run.py:88-137`) :

```python
sim = adc.System(n=N, L=L, periodic=True)            # run.py:93
sim.add_block("gas", model=models.euler_poisson(sign=sign, ...), ...)   # run.py:94-98
sim.set_poisson(rhs="charge_density", solver="geometric_mg")           # run.py:99
sim.set_density("gas", initial_density())            # run.py:100
mass0 = sim.mass("gas")                               # masse de reference (run.py:103)
for step in range(1, NSTEPS + 1):
    sim.advance(DT, 1)                               # 1 pas SSPRK2 + Poisson par etage (run.py:115)
    rel_mass = relative_drift(m, mass0)             # |m - mass0| / |mass0| (run.py:119)
    max_rel_mass = max(max_rel_mass, rel_mass)      # pire derive sur tous les pas (run.py:120)
    max_mom = max(max_mom, abs(px), abs(py))        # pire impulsion (run.py:121)
```
- `rhs="charge_density"` est l'alias generique du second membre compose (somme des briques
  elliptiques de chaque bloc ; ici l'unique `GravityCoupling`). `relative_drift`
  (`common/checks.py:11-13`) protege le denominateur contre zero. `mass("gas")` renvoie la somme de
  $\rho$ : $64\times64\times1=4096$, conforme a la sortie.

Verification `main()` (`run.py:140-183`) :

```python
assert res["max_rel_mass"] < TOL_MASS               # masse conservee (run.py:150)
assert res["max_mom"] < TOL_MOM                     # impulsion nette nulle (run.py:155)
dE_grav = grav["energy_final"] - grav["energy0"]    # run.py:160
dE_plas = plas["energy_final"] - plas["energy0"]    # run.py:161
assert_opposite_sign(dE_grav, dE_plas, min_mag=TOL_DE, ...)   # signes opposes + magnitude (run.py:174)
assert dE_grav < 0.0                                # gravite attractive (run.py:177)
assert dE_plas > 0.0                                # plasma repulsif (run.py:179)
```
- `assert_opposite_sign` (`common/checks.py:43-54`) exige `|a| > min_mag AND |b| > min_mag` puis
  `a*b < 0` : on ne valide pas trivialement deux quasi-zeros de produit negatif par accident.

---

## 6. Les tolerances, justifiees par un ordre de grandeur (justifie 8 de la checklist)

| Tolerance | Valeur | Pourquoi cette valeur |
|---|---|---|
| `TOL_MASS` | $10^{-9}$ | Le schema volumes finis est conservatif : la masse est un invariant exact, la seule derive est l'arithmetique flottante. Mesure : $2.6\times10^{-14}$ relatif, soit ~5 ordres sous la tolerance (`run.py:58`) |
| `TOL_MOM` | $10^{-8}$ | La force de Poisson derive d'un potentiel periodique : sa somme spatiale est nulle, elle n'injecte aucune impulsion. Mesure : $8.9\times10^{-16}$, ~8 ordres sous la tolerance (`run.py:59`) |
| `TOL_DE` | $10^{-5}$ | Borne basse : $dE=0$ exactement a $\epsilon=0$ (verifie). Borne haute : la magnitude physique attendue est $\sim6\times10^{-4}$ ($\epsilon=0.01$, 20 pas, `run.py:60-62`). $10^{-5}$ se situe entre le bruit (0) et le signal ($6\times10^{-4}$) : il rejette un signe non significatif sans rejeter le signal reel (`run.py:63`) |

---

## 7. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (memes parametres que `run.py`), versionnees avec
`figures/provenance.json`. Commande exacte en section 9.

### `energy_vs_t.png` : le contraste de signe

![E_tot(t) - E_tot(0) pour gravite (descend) et plasma (monte), cote a cote](figures/energy_vs_t.png)

- PROUVE (asserte `run.py:177-180`) : $E_{tot}$ diminue pour la gravite
  ($dE_{grav}=-5.858\times10^{-4}$) et augmente pour le plasma ($dE_{plas}=+6.137\times10^{-4}$).
  Les deux courbes partent de 0 (meme etat au repos) et divergent en sens opposes : signes
  strictement opposes, magnitudes $\gg$ TOL_DE.
- SUGGERE (non assere) : la quasi-symetrie miroir des deux courbes (gravite et plasma ont des
  $|dE|$ proches a ~5 %) est visible mais aucun assert ne la verifie ; elle n'est pas exacte (la
  reponse compressible n'est pas lineaire en $\mathrm{sign}$).
- NON MONTRE : ce graphe ne dit rien du travail de la force (positif des deux cotes, section
  4.3) ; le titre rappelle que le signe est asserte, pas deduit de $v\cdot g$.

### `de_vs_eps.png` : la prediction $|dE|\propto\epsilon^2$

![|dE| vs epsilon en log-log : droites de pente 2 pour gravite et plasma, superposees a la reference](figures/de_vs_eps.png)

- PROUVE : sur $\epsilon\in\{0.005,0.01,0.02,0.04,0.08\}$, la regression log-log donne une pente
  2.000 (gravite 1.99998, plasma 1.99998), confondue avec la droite de reference $\propto
  \epsilon^2$. Doubler $\epsilon$ quadruple $|dE|$ : $1.46\times10^{-4}\to5.86\times10^{-4}\to
  2.34\times10^{-3}$, chaque pas $\times 4$. La linearisation (section 4.4) est confirmee.
- SUGGERE : gravite et plasma ont des $|dE|$ tres proches (les deux droites se chevauchent) ;
  l'asymetrie est du second ordre et non testee.
- NON MONTRE : aucune deviation de pente n'apparait jusqu'a $\epsilon=0.08$, donc on ne voit pas
  l'entree du regime non lineaire (qui donnerait pente $>2$ aux plus grands $\epsilon$). Le controle
  $dE(\epsilon{=}0)=0.0$ (bit-machine) borne la tolerance par le bas mais n'est pas sur l'axe log.

### `density_map.png` : la perturbation reste 1D

![Cartes de densite : CI, gravite finale, plasma finale, perturbation en bandes verticales](figures/density_map.png)

- PROUVE / mesure : la perturbation reste 1D selon $x$ (ecart-type en $y$ : $3.8\times
  10^{-16}$, bit-machine) ; aucune structure transverse n'apparait. L'amplitude max-min passe de
  $2.00\times10^{-2}$ (CI) a $1.76\times10^{-2}$ (gravite) et $1.75\times10^{-2}$ (plasma) : les
  deux s'aplatissent legerement sur 20 pas.
- NON MONTRE : a $\epsilon=0.01$, le contraste gravite/plasma n'est pas visible a l'oeil sur
  la densite (les deux panneaux sont quasi identiques) ; le contraste de signe vit dans l'energie
  integree $dE\sim6\times10^{-4}$, pas dans la carte de densite. Aucun effondrement ni formation de
  structure : le regime est quasi-lineaire et a horizon court.

---

## 8. Ce que l'invariant ne capture pas (analyse honnete des limites)

- Pas une reproduction publiee. Categorie `validation` : on teste des invariants structurels
  (conservation, signe, pente $\epsilon^2$), pas une courbe d'article. Ne pas presenter comme
  effondrement de Jeans, formation de structure ou benchmark plasma.
- $E_{tot}$ est un proxy, pas l'energie totale du systeme couple. C'est l'energie fluide
  (cinetique+interne) sans le potentiel de champ $\tfrac12\int\rho\phi$ ; son signe distingue les
  regimes, sa valeur absolue n'est comparee a rien. Les diagnostics sont des sommes sur cellules
  sans poids $dx^2$ : seuls le relatif (masse) et le signe (energie) sont significatifs.
- $4\pi G=1$, $\rho_0=1$, sans unites. Choix de lisibilite, pas un calibrage gravitationnel.
- Regime quasi-lineaire, horizon court ($\epsilon=0.01$, 20 pas, $dt=0.004$, $t_{fin}=0.08$) : on
  observe la tendance energetique (travail + compression), pas de dynamique non lineaire.
- Domaine periodique homogene : c'est ce qui garantit l'impulsion nette nulle et la
  compatibilite du Poisson (second membre de moyenne nulle grace au fond $\rho_0$). Des parois ou un
  second membre non centre casseraient ces deux invariants.

---

## 9. Reproduire (justifie 14 de la checklist : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/euler_poisson
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~0.3 s
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 3 figures + provenance.json
```

Prerequis : `numpy` (et `matplotlib` pour les figures, hors `needs` du cas lui-meme), module `adc`
compile et importe avec le meme interpreteur que celui qui l'a compile (suffixe ABI
`cpython-312`). Le premier chemin du `PYTHONPATH` fournit le module C++ ; le second rend `adc_cases`
importable sans installation (le cas a aussi un fallback `sys.path`, `run.py:48-53`).

Sortie attendue de `run.py` (capturee, machine de dev macOS arm64) :

```
Contraste energetique (attractif vs repulsif) :
  dE GRAVITE = -5.857667e-04   dE PLASMA = +6.137105e-04
  -> signes opposes (gravite dE<0, plasma dE>0), magnitudes > 1e-05 : OK
OK euler_poisson
```

avec `max derive masse relative = 2.598e-14` (GRAVITE) / `2.098e-14` (PLASMA), `max |p| =
8.882e-16`. Cout : ~0.3 s temps mur (import numpy inclus), 2 runs $\times$ 20 pas $\times$ grille
$64^2$ + un Poisson multigrille par etage. Caveat plateforme : les signes, l'ordre de grandeur
($\sim6\times10^{-4}$), la pente (2.000) et le verdict `OK` sont stables d'une plateforme a l'autre ;
les derniers chiffres de $dE$ varient avec la BLAS et l'ordre de sommation (cf.
`figures/provenance.json`).

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | le cas : 2 runs (signe $\pm$), invariants par `assert` (masse, impulsion, contraste de signe) |
| `make_figures.py` | re-joue la physique + balayage $\epsilon$ ; ecrit les 3 figures + `provenance.json` |
| `figures/*.png` | `energy_vs_t.png`, `de_vs_eps.png`, `density_map.png` (versionnees, regenerees en place) |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, backend, resolution, nombres mesures ($dE$, pentes, derives) |
| `../adc_cases/models.py` | `euler_poisson(sign,...)` = composition des 4 briques natives (`l.48-55`) |
| `../adc_cases/common/checks.py` | `relative_drift`, `assert_opposite_sign` (utilises par le cas) |
