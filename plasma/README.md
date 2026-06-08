# plasma : trois especes (e + i + n) couplees par Poisson, ionisation et collision

Validation de la plomberie de couplage multi-especes de `adc.System` : trois fluides (electrons
Euler compressible, ions et neutres isothermes) partagent un Poisson de systeme et echangent de la
masse par ionisation ($n_g \to n_i + n_e$) et de l'impulsion par friction ion-neutre. Le cas ne
reproduit aucun plasma publie : il mesure trois invariants structurels (champ non nul, masse
$n_i + n_g$ conservee au transfert, densites finies et positives) et les assere. La prediction
falsifiable est un invariant exact, pas un nombre physique cible.

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` (`cases_manifest.toml`, `ci = true`, `needs = []`). Ce n'est pas une reproduction publiee : on verifie des invariants, pas une courbe d'un papier. |
| Entrees | grille $48^2$, $L=1$, periodique ; CI : $n_e = 1 + 0.05\cos(2\pi x)$ (faible separation de charge), $n_i = n_g = 1$, toutes au repos ; $q_e=-1$, $q_i=+1$, $q_g=0$ ; $\gamma_e=5/3$, $c_s^2=1$ ; $k_{ion}=0.3$, $k_{col}=0.5$ ; 20 macro-pas a CFL $0.3$ |
| Sorties | $\lvert\phi\rvert_{max}$ initial ; masses $M_i, M_g$ avant/apres ; min des trois densites ; 3 figures de diagnostic dans `figures/` + `figures/provenance.json` |
| Invariants garantis | les 4 `assert` de `run.py:66-69` : (1) $\lvert\phi\rvert_{max}>10^{-8}$ ; (2) $M_g$ baisse et $M_i$ monte (chacun $>10^{-6}$) ; (3) derive relative de $M_i+M_g$ sous $10^{-7}$ ; (4) densites finies et $>0$ |
| Prouve | (1) le Poisson de systeme est actif : $\lvert\phi\rvert_{max}=1.266\times10^{-3}$ ; (2) l'ionisation transfere la masse du neutre vers l'ion : $M_g\!:2304\to2237.32$, $M_i\!:2304\to2370.68$ ; (3) ce transfert conserve $M_i+M_g$ a $2.37\times10^{-15}$ (precision machine) ; (4) les trois densites restent finies et $>0$ (min $e=0.986$, $i=1.028$, $n=0.970$) |
| Ne prouve pas | l'ionisation n'agit que sur la densite (comp 0) : le transfert de quantite de mouvement et d'energie des particules creees est une simplification du coeur (`system.cpp:719-733`), aucun assert ne le teste. La friction neglige l'echauffement (`system.cpp:754-757`). Aucune energie totale, aucun taux de croissance, aucune section efficace physique : $k_{ion}, k_{col}$ sont des constantes de demonstration. Pas de magnetisation, pas de derive ExB (cf. [`../diocotron/`](../diocotron/)). $48^2$/20 pas : aucune convergence |
| Provenance | adc_cpp `01873299`, adc_cases `7c7a3403`, backend natif serie, $48^2$, ~0.2 s 1 coeur CPU ; `figures/provenance.json` |

A la fin tu sauras : pourquoi un Poisson de systeme couple trois fluides charges differemment,
pourquoi l'ionisation conserve $n_i+n_g$ exactement (le terme source est antisymetrique a la
precision machine), comment la friction conserve l'impulsion du couple ion-neutre, et ce que le
modele ne capture pas (momentum/energie des particules ionisees, echauffement de friction).

---

## 1. Le mecanisme physique

Trois fluides occupent le meme carre periodique. Ce qui les lie n'est pas le transport (chaque
espece advecte sa propre densite), mais trois couplages appliques apres le transport :

1. **Champ self-consistant.** Electrons ($q_e=-1$) et ions ($q_i=+1$) sont des sources du potentiel
   par $-\nabla^2\phi = f = q_e n_e + q_i n_i$. Les neutres ($q_g=0$) n'y entrent pas. A l'instant
   initial, $n_e$ est module ($1+0.05\cos 2\pi x$) et $n_i$ uniforme : $f=-n_e+n_i=-0.05\cos 2\pi x$
   n'est pas nul, donc $\phi$ non plus. C'est la separation de charge qui allume le Poisson.

2. **Ionisation $n_g \to n_i + n_e$.** Un neutre heurte un electron, perd un electron et devient un
   ion ; il y a desormais un ion et un electron de plus, un neutre de moins. Le taux local est
   $r = k_{ion}\,n_e\,n_g$ (proportionnel a la densite des deux reactifs). La masse passe du
   reservoir neutre au reservoir ionise : $n_g$ descend, $n_i$ et $n_e$ montent du meme montant.

3. **Friction ion-neutre.** Les ions et les neutres s'echangent de l'impulsion par collisions :
   une force $\mathbf{F}=k_{col}(\mathbf{u}_i-\mathbf{u}_g)$ freine l'espece rapide et accelere la
   lente, opposee sur chaque fluide. C'est un transfert interne : l'impulsion totale du couple
   ion-neutre est conservee par la friction seule.

Le coeur de ce cas est le couplage 2 : son invariant ($n_i+n_g$ conserve) est exact parce que
le terme source est ecrit antisymetrique (ce qui sort de $n_g$ entre dans $n_i$, voir section 4).
C'est cet invariant que les figures de la section 6 confrontent. Le couplage 3 est cable et
actif mais son invariant d'impulsion n'est pas asseré ici (justifie en section 7).

---

## 2. Les equations et qui les calcule

Trois fluides sur $[0,L]^2$ periodique. Transport (chaque espece) puis couplages (operator-split) :

| Espece | Transport | Source champ | Couplages subis |
|---|---|---|---|
| electrons (Euler, $\gamma=5/3$) | $\partial_t U_e + \nabla\cdot F(U_e) = 0$, $U_e=(\rho_e,\rho_e\mathbf{u}_e,E_e)$ | $\tfrac{q_e}{m}\rho_e\mathbf{E}$ (+ travail) | ionisation (gain de densite) |
| ions (isotherme, $c_s^2=1$) | $\partial_t U_i + \nabla\cdot F(U_i) = 0$, $U_i=(\rho_i,\rho_i\mathbf{u}_i)$ | $\tfrac{q_i}{m}\rho_i\mathbf{E}$ | ionisation (gain), friction |
| neutres (isotherme, $c_s^2=1$) | $\partial_t U_g + \nabla\cdot F(U_g) = 0$, $U_g=(\rho_g,\rho_g\mathbf{u}_g)$ | aucune ($q_g=0$) | ionisation (perte), friction |

Couplage elliptique partage : $-\nabla^2\phi = q_e n_e + q_i n_i$, $\mathbf{E}=-\nabla\phi$,
periodique. Chaque espece est un modele nomme cote application (`adc_cases.models`), compose de
briques generiques par `adc.Model(state, transport, source, elliptic)` :

| Espece | `models.*` (`models.py`) | Etat | Transport | Source | Elliptique |
|---|---|---|---|---|---|
| electrons | `electron_euler` (l.28-35) | `FluidState(compressible, gamma=5/3)` | `CompressibleFlux` | `PotentialForce(q=-1)` | `ChargeDensity(q=-1)` |
| ions | `ion_isothermal` (l.38-45) | `FluidState(isothermal, cs2=1)` | `IsothermalFlux` | `PotentialForce(q=+1)` | `ChargeDensity(q=+1)` |
| neutres | `neutral_isothermal` (l.69-77) | `FluidState(isothermal, cs2=1)` | `IsothermalFlux` | `NoSource` | `ChargeDensity(q=0)` |

`ChargeDensity(q=0)` est presente mais nulle sur les neutres : ils sont declares au Poisson avec
un poids zero, donc n'y contribuent pas. Qui calcule quoi (table 3 couches, ancree sur les lignes
reelles de `recipes.plasma`, `recipes.py:32-51`, declenchee par `run.py:43-44`) :

| Ligne | Couche | Ce qui se passe |
|---|---|---|
| `recipes.plasma(sim, ne, ni, ng, ...)` (`run.py:43`) -> `add_block` x3 + `set_poisson` + `add_ionization` + `add_collision` (`recipes.py:37-47`) | Python compose | choix des 3 modeles, des schemas (electrons HLLC+vanleer+primitif ; ions/neutres minmod), du Poisson de systeme, des 2 couplages |
| `models.electron_euler/ion_isothermal/neutral_isothermal` -> briques `CompressibleFlux` / `IsothermalFlux` / `PotentialForce` / `ChargeDensity` (`include/adc/physics/*.hpp`) | brique C++ fige | la convention exacte du flux, de la force $q\rho\mathbf{E}$, du second membre de Poisson $\sum_b q_b n_b$ |
| `assemble_rhs<Limiter,Flux>` par bloc + `GeometricMG` (Poisson) + foncteurs ionisation/collision (`system.cpp:723-733`, `758-767`) | noyau par cellule (device) | le transport reel et les deux couplages, sans callback Python dans le hot path |

Le mot "plasma" vit dans `recipes.py`, jamais cote coeur : c'est une composition de briques
generiques, pas un scenario code en dur.

---

## 3. La prediction falsifiable : l'invariant $n_i + n_g = \text{cste}$

Ce cas etant `validation`, sa prediction n'est pas un taux mais un invariant exact : tout neutre
ionise devient exactement un ion, donc

$$\frac{d}{dt}\big(M_i + M_g\big) = 0, \qquad M_s \equiv \sum_{\text{cell}} n_s .$$

La derivation (section 4) montre pourquoi : le terme source est antisymetrique entre $n_i$ et
$n_g$ a la precision machine. L'artefact qui confronte cette prediction est `ionization.png`
(section 6) : la courbe noire $M_i+M_g$ doit etre plate, et la derive relative doit plafonner
sous la tolerance $10^{-7}$ de l'assert. C'est la clause Prouve (3). La clause Prouve (2) (sens du
transfert) et Prouve (1) (Poisson actif) sont confrontees par les memes figures. La clause
Ne prouve pas (momentum/energie ignores) est justifiee en section 4 (ce que le foncteur n'ecrit pas)
et en section 7.

---

## 4. Maths : pourquoi l'ionisation conserve $n_i+n_g$ et pas l'energie

### 4.1 Le terme source d'ionisation est antisymetrique par construction

L'ionisation est appliquee en operator-split apres le transport. Le foncteur C++ (`system.cpp:723-733`)
calcule, par cellule, un seul scalaire $\delta n = \Delta t\,k_{ion}\,n_e\,n_g$ puis le distribue :

```cpp
const Real dn = dt * k * ue(i, j, de) * ug(i, j, dg);   // delta_n = dt k n_e n_g  (system.cpp:728)
ug(i, j, dg) -= dn;                                      // neutre : -delta_n      (system.cpp:729)
ui(i, j, di) += dn;                                      // ion    : +delta_n      (system.cpp:730)
ue(i, j, de) += dn;                                      // electron: +delta_n     (system.cpp:731)
```

- `de`, `di`, `dg` sont les indices de la composante densite de chaque bloc, resolus par role
  (`role_index(..., Density, 0)`, `system.cpp:716-718`) : un bloc qui range sa densite ailleurs que
  l'indice 0 reste correctement couple.
- $n_g$ perd exactement ce que $n_i$ gagne : le meme $\delta n$ est soustrait a l'un et ajoute a
  l'autre. La somme cellule par cellule $n_i+n_g$ est donc invariante a l'arithmetique flottante
  pres ; en sommant sur toutes les cellules, $M_i+M_g$ est conservee. C'est l'origine du
  $2.37\times10^{-15}$ mesure (somme de $48^2$ annulations flottantes, pas un zero exact).
- $n_e$ gagne aussi $\delta n$ (l'electron arrache n'est pas detruit) : $n_e$ et $n_i$ croissent
  du meme montant. C'est verifiable : les masses finales electrons et ions sont identiques a
  $10^{-13}$ pres ($2370.677033292462$ vs $2370.677033292496$, `provenance.json`). $M_e$ n'est donc
  pas conservee (les electrons sont crees, pas advectes seulement) ; seul le couple $M_i+M_g$ l'est.

### 4.2 Ce que le foncteur n'ecrit pas (la simplification, clause Ne prouve pas)

Les trois lignes ci-dessus touchent uniquement la composante densite (comp 0). Elles n'ecrivent
ni la quantite de mouvement (comp 1, 2) ni l'energie (comp 3 des electrons). Physiquement, un ion
cree devrait naitre avec la quantite de mouvement du neutre dont il provient, et l'electron arrache
emporte une energie ; ici rien de tout cela n'est transfere. Le commentaire du coeur le dit :
"le transfert de quantite de mouvement / energie (especes fluides) est un raffinement ulterieur"
(`system.cpp:721-722`). Consequence concrete : l'energie totale n'est ni definie ni controlee, et
aucun assert ne porte dessus. La conservation que l'on assere (`drel < 1e-7`) est uniquement une
conservation de masse $n_i+n_g$, pas de momentum ni d'energie.

### 4.3 La friction conserve l'impulsion du couple, pas l'energie

Le foncteur de collision (`system.cpp:758-767`) calcule la force de friction par cellule et l'oppose
sur chaque espece :

```cpp
const Real fx = dt * k * (ua(i,j,mxa)/ua(i,j,da) - ub(i,j,mxb)/ub(i,j,db));  // dt k (u_a - u_b)
ua(i, j, mxa) -= fx;  ub(i, j, mxb) += fx;                                   // opposee (system.cpp:763)
```

- `mxa`, `mxb` (et `mya`, `myb`) sont les composantes quantite de mouvement $\rho u$ resolues
  par role (`system.cpp:748-753`) ; `da`, `db` les densites (pour reconstruire $u=\rho u/\rho$).
- La force est $\mathbf{F}=k_{col}(\mathbf{u}_a-\mathbf{u}_b)$, retiree a $a$, ajoutee a $b$ : la
  somme $\rho_a\mathbf{u}_a + \rho_b\mathbf{u}_b$ change de $-fx + fx = 0$. L'impulsion totale du
  couple ion-neutre est conservee par la friction. Mesure : $\Delta(P_x^{ion}+P_x^{neutre})$ final
  $=-1.7\times10^{-17}$, le zero machine (figure `ionization.png`, panneau 3).
- L'echauffement par friction (la chaleur dissipee, $\propto k_{col}|\mathbf{u}_a-\mathbf{u}_b|^2$)
  n'est pas rendu a l'energie : "l'echauffement par friction (energie) est un raffinement
  ulterieur" (`system.cpp:756-757`). C'est coherent pour des especes isothermes (sans equation
  d'energie), mais c'est une simplification a nommer.

### 4.4 Pourquoi la tolerance $10^{-7}$

`assert drel < 1e-7` (`run.py:68`) se situe entre deux echelles : le bruit de l'antisymetrie
flottante (mesure $2.37\times10^{-15}$, soit la somme de $48^2$ annulations) et toute violation
structurelle qui trahirait un bug de distribution (si le foncteur soustrayait a $n_g$ autre
chose qu'il n'ajoute a $n_i$, la derive serait de l'ordre de la fraction ionisee, $\sim 3\times10^{-2}$
ici). $10^{-7}$ est largement au-dessus du bruit et tres en-dessous d'une fuite de masse reelle :
sept ordres de grandeur de marge. La tolerance n'est pas posee, elle separe deux regimes mesures.

---

## 5. Conditions initiales (`run.py:38-44`)

Posees en numpy (la physique du scenario vit cote application, jamais en C++ par cas) :

```python
n, L = 48, 1.0
x  = (np.arange(n) + 0.5) / n                                     # centres de cellules le long de x
ne = 1.0 + 0.05 * np.cos(2 * PI * x)[None, :] * np.ones((n, n))   # electrons modules le long de x (run.py:40)
recipes.plasma(sim, ne=ne, ni=np.ones((n, n)), ng=np.ones((n, n)),
               ionization_rate=0.3, collision_rate=0.5)           # ions/neutres uniformes (run.py:43-44)
```

- **Electrons** : $1+0.05\cos(2\pi x)$, modulation 5 % le long de $x$, constante en $y$. C'est
  l'unique source de non-trivialite du Poisson : $f=-n_e+n_i=-0.05\cos 2\pi x$.
- **Ions, neutres** : uniformes a $1$. Tous les fluides demarrent au repos : `set_density` ne
  pose que la densite, le reste de l'etat conservatif est complete au repos par le modele du bloc.
- Convention de grille (`adc_cases.common.grid`) : `field[j, i]`, centre $x=(i+0.5)/n\,L$. La
  modulation ne depend que de la colonne $i$ (axe $x$), d'ou des cartes finales **striees en $x$**
  (section 6).

L'avance : `sim.step_cfl(0.3)` x20 (`run.py:53-54`), $dt$ choisi a CFL $0.3$ a chaque macro-pas.
Le temps final mesure est $t=0.0965$ (`provenance.json`). Schemas : electrons
`Spatial(vanleer, hllc, primitive)` (recon primitive = positivite de $\rho,p$ pour Euler) ;
ions/neutres `Spatial(minmod)` (flux rusanov par defaut). Integrateur SSPRK2 par defaut.

---

## 6. Figures (diagnostic, `figures/`, generees par `make_figures.py`)

`make_figures.py` re-joue les memes CI, recette, nombre de pas et CFL que `run.py`, mais
instrumente la boucle pour enregistrer l'historique. Commande en section 8.

### `densities.png` : densites moyennes des trois especes vs t

![Densites moyennes e/i/n vs t : ions et electrons montent, neutres descendent](figures/densities.png)

- **Prouve** (clause 2) : la densite moyenne des ions monte ($\bar n_i\!:1\to1.0289$) et celle des
  neutres descend ($\bar n_g\!:1\to0.9711$) de maniere exactement opposee : l'ionisation vide le
  reservoir neutre dans le reservoir ionise. Pentes initiales egales et de signe oppose (asserte par
  $M_g<M_{g,0}$ et $M_i>M_{i,0}$, `run.py:67`).
- **Suggéré (non assere)** : la courbe electron (bleu) est invisible, masquee sous la courbe ion
  (rouge) : $\bar n_e=\bar n_i$ a $10^{-13}$ pres (section 4.1, $n_e$ et $n_i$ gagnent le meme
  $\delta n$). Visible a l'oeil, mais aucun assert ne compare $M_e$ a $M_i$.
- **Non montré** : la courbure tres legere (le taux $k n_e n_g$ depend de $n_e n_g$ qui evolue) ;
  sur $t<0.1$ et une fraction ionisee de 3 %, l'evolution reste quasi lineaire. Pas de saturation
  (le neutre n'est pas epuise).

### `ionization.png` : bilan d'ionisation, conservation et impulsion

![Trois panneaux : transfert n_g vers n_i, derive de masse, impulsion totale](figures/ionization.png)

- **Prouve** (clause 3), panneau gauche : $M_i$ (rouge) et $M_g$ (vert) divergent en miroir, mais
  leur somme $M_i+M_g$ (noir) est rigoureusement plate a $4608=2\times2304$. Le transfert ne
  cree ni ne detruit de masse $i\!+\!g$.
- **Prouve** (clause 3), panneau central : la derive relative de $M_i+M_g$ plafonne autour de
  $10^{-15}$ (precision machine), huit ordres de grandeur sous la tolerance d'assert $10^{-7}$
  (ligne grise) : l'antisymetrie du foncteur (section 4.1) tient au bit pres. C'est l'observable qui
  Prouve l'invariant, pas seulement le rend plausible.
- **Suggéré / Non montré**, panneau droit : la variation d'impulsion totale du couple ion-neutre
  reste au zero machine ($\sim10^{-17}$). La friction conserve cette impulsion (section 4.3) ;
  ici elle est de toute facon quasi nulle car les vitesses partent de zero et restent faibles. Le
  panneau suggere la conservation mais ne la prouve pas (aucun assert sur l'impulsion dans ce
  cas ; voir section 7) : a vitesse nulle, c'est un test peu exigeant.

### `density_map.png` : cartes de densite a l'etat final

![Cartes 2D n_e, n_i, n_g a t final : striees en x, ions/neutres modules par l'ionisation](figures/density_map.png)

- **Prouve** (clause 4) : les trois cartes sont finies et partout positives (min $e=0.986$,
  $i=1.028$, $n=0.970$ ; asserte `run.py:69`). Aucun creux negatif, le primitif electron et le minmod
  isotherme tiennent la positivite.
- **Suggéré (non assere)** : ions et neutres, partis uniformes, ont developpe une modulation en
  $x$ qui copie le motif electron (ion : maximum la ou $n_e$ est dense ; neutre : photographie
  negative). Cause : le taux local $k\,n_e\,n_g$ est proportionnel a $n_e$, donc on ionise plus la ou
  les electrons sont denses. C'est une consequence directe (et correcte) du couplage, mais aucun
  assert ne la verifie : signature, pas garantie.
- **Non montré** : aucune dynamique en $y$ (CI invariante en $y$, advection au repos) ; aucune
  structure non lineaire (run trop court, gradients trop faibles).

---

## 7. Ce que les invariants ne capturent pas

L'oracle de ce cas est la plomberie, pas une physique de reference. Les ecarts au "vrai" plasma
sont structurels et assumes :

1. **Ionisation sans momentum ni energie** (section 4.2). Le foncteur n'ecrit que la densite. Un ion
   cree devrait heriter de l'impulsion du neutre source et l'electron d'une energie ; ici les
   particules creees apparaissent au repos thermodynamique du fluide receveur. La conservation
   asseree ne porte donc que sur $n_i+n_g$.

2. **Friction sans echauffement** (section 4.3). La chaleur de friction n'est pas rendue a l'energie.
   Coherent pour des especes isothermes, mais ce serait faux si on activait l'energie des ions.

3. **Impulsion de collision non asseree ici.** Le panneau droit de `ionization.png` montre une
   impulsion au zero machine, mais (a) les vitesses partent de zero, donc le test est peu exigeant,
   et (b) le champ $\mathbf{E}$ agit aussi sur les ions ($q_i\rho_i\mathbf{E}$), si bien que
   l'impulsion des ions seuls n'est pas conservee : seule la friction, isolement, conserve le couple.
   La conservation d'impulsion de la friction pure est verifiee a part dans le test des bindings de
   `adc_cpp`, pas dans ce cas assemble.

4. **Taux non physiques.** $k_{ion}=0.3$, $k_{col}=0.5$ sont des constantes de demonstration, sans
   section efficace ni dependance en temperature. La fraction ionisee de 3 % en $t=0.0965$ n'a pas de
   sens calibre.

5. **Pas de magnetisation, geometrie cartesienne, run court.** Aucun champ $B$, aucune derive ExB
   (pour cela voir [`../diocotron/`](../diocotron/), la limite de derive d'une densite scalaire) ;
   $48^2$/20 pas exerce le couplage, ne mesure aucune convergence.

---

## 8. Reproduire

Le cas (asserts, ~0.2 s) :

```bash
cd /private/tmp/adc_cases-deeptut/plasma
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py
```

Sortie attendue (deterministe, re-execute a l'identique ; les derniers chiffres varient avec la BLAS
et l'ordre de sommation flottante, mais signes et ordres de grandeur sont stables) :

```
== plasma : electrons + ions + neutres (Poisson + ionisation + collision) ==
  |phi|_max = 1.266e-03  (Poisson de systeme actif)
  ionisation : n_i 2304.0000 -> 2370.6770,  n_g 2304.0000 -> 2237.3230,  (n_i+n_g) drel = 2.37e-15
  densites   : min e=9.862e-01 i=1.028e+00 n=9.698e-01 (toutes finies et positives : True)
OK plasma
```

Les figures de diagnostic (re-joue la physique, ~0.5 s, ecrit `figures/*.png` + `provenance.json`) :

```bash
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py
```

Prerequis : `numpy`, `matplotlib` (figures uniquement ; le cas `run.py` n'a besoin que de `numpy`),
et le module `adc` compile, importe **avec le meme interpreteur** que celui qui l'a compile (suffixe
ABI `cpython-3XY`). En CI, seul `run.py` tourne (`category="validation"`, `ci=true`, `needs=[]`) ;
les figures sont hors CI.

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | le cas : pose les CI, cable `recipes.plasma`, 20 pas, 4 asserts (sec. 4, 6) |
| `make_figures.py` | re-joue la physique instrumentee, trace les 3 figures + `provenance.json` |
| `figures/*.png`, `figures/provenance.json` | diagnostics du tutoriel (hors CI) + nombres mesures |
| `../adc_cases/recipes.py` (`plasma`) | recette systeme : 3 blocs + Poisson + ionisation + collision |
| `../adc_cases/models.py` | modeles d'espece : `electron_euler`, `ion_isothermal`, `neutral_isothermal` |
| `../adc_cases/common/checks.py` (`relative_drift`) | derive relative protegee, invariant de masse |
