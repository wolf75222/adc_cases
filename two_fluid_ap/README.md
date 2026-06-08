# two_fluid_ap : bi-fluide isotherme raide, asymptotic-preserving

Un plasma a deux fluides (electrons + ions) couples au champ electrique de Poisson est integre
par un schema IMEX dont le terme raide (la relaxation a la quasi-neutralite, d'echelle la frequence
plasma $\omega_{pe}$) est traite implicitement. La propriete testee est l'asymptotic-preserving
(AP) : le pas stable ne s'effondre pas quand la raideur $s=\Delta t\,\omega_{pe}$ croit, alors qu'un
schema explicite est limite a $s\lesssim 1$ et explose au-dela. L'integrateur AP a quitte le coeur
`adc_cpp` (ce n'est pas une brique composable `adc.System`) : c'est un scenario C++ sur mesure
(`two_fluid_ap.hpp` + `_two_fluid_ap.cpp`), compile a la volee via `ctypes`. Ce n'est pas une
reproduction d'un resultat publie.

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` (`cases_manifest.toml`, `two_fluid_ap/run.py`, `ci = true`, `needs = ["cxx"]`) |
| Entrees | grille $64^2$, $L=2\pi$, periodique ; CI $n_e=1+\epsilon\cos(kx+ky)$, $k=2\pi/L$, $\epsilon=10^{-3}$, $n_i=1$, $m_s=0$ ; isotherme $c_e^2=1$, $c_i^2=0.04$ ; $z_e=-1$, $z_i=+1$, $n_0=1$. Run 1 raide : $\omega_{pe}=10^3$, $\omega_{pi}=20$, $\Delta t=5\times10^{-3}$, 200 pas, $s=\Delta t\,\omega_{pe}=5$. Run 2 magnetise : $\omega_{ce}=4$, $\omega_{ci}=0.2$, $\Delta t=10^{-2}$, 100 pas |
| Sorties | diagnostics imprimes (`max_dev`, `max_charge`, `mass_e`), ligne finale `OK two_fluid_ap` ; 2 figures + `figures/provenance.json` (via `make_figures.py`) ; lib JIT dans `out/two_fluid_ap/build/` |
| Invariants garantis | les `assert` de `run.py` : finitude (`np.isfinite`, run 1+2) ; `max_dev < 0.1` et `max_charge < 0.1` (run 1, `run.py:199-200`) ; `mass_rel < 1e-7` (run 1+2, `run.py:202`, `run.py:235`) |
| Prouve | a $s=5$ (run 1), schema AP fini et borne : $\max\lvert n_e-1\rvert=5.325\times10^{-7}$, $\max\lvert n_i-n_e\rvert=6.698\times10^{-11}$, masse $e$ conservee a $2.276\times10^{-14}$ relatif ; run 2 magnetise masse $e$ a $1.665\times10^{-14}$. Prediction AP falsifiable (figure 1) : la deviation AP plateaute a $5.41\times10^{-7}$ pour $s\in[1,50]$, tandis que l'explicite est fini pour $s\le1.0$ et NaN des $s\ge1.2$ |
| Ne prouve pas | pas une reproduction publiee : aucun nombre confronte a un article. Les `assert` de `run.py` testent des bornes ($<0.1$, $<10^{-7}$), pas l'ordre AP : le contraste AP/explicite est mesure par `make_figures.py`, pas asserte. `mass_e=4096` est une somme sans poids $dx^2$ (proxy de conservation relative, pas une masse physique). Le diagnostic C++ `tfap_max_dev` est non fiable sur un champ explose (`fmax` sur NaN rend $0.0$, section 6) : l'explosion explicite est detectee cote Python par `np.isfinite`. Regime quasi-lineaire ($\epsilon=10^{-3}$, schema spatial bas ordre, fond $n_0=1$ constant) ; backend valide = CPU serie seul (portabilite GPU non exercee ici) |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, scenario C++ JIT `TwoFluidAP2D<GeometricMG>` (Apple clang 21, C++20), CPU serie, $64^2$ ; run.py ~3.6 s (cache a jour) / ~5.5 s (1ere compilation) ; `figures/provenance.json` |

A la fin tu sauras : ce qu'est la raideur d'un plasma et pourquoi un schema explicite y est limite a
$\Delta t\,\omega_{pe}\lesssim1$ (mecanisme), comment la reformulation AP de l'elliptique leve cette
borne (la derivation $\beta_0=\Delta t^2(\omega_{pe}^2+\omega_{pi}^2)$), quelle est la prediction
falsifiable (deviation bornee quand $s\to\infty$, explicite NaN), et pourquoi le solveur vit dans
`adc_cases` au lieu du coeur `adc_cpp`.

---

## 1. Le mecanisme physique : la raideur du plasma (justifie Prouve : AP borne)

Deux fluides isothermes charges, electrons (densite $n_e$, charge $z_e=-1$) et ions ($n_i$,
$z_i=+1$), partagent un champ electrique auto-consistant. Trois ingredients enchaines, dont le
dernier est la source de raideur :

1. **Transport isotherme.** Chaque espece advecte sa densite et sa quantite de mouvement
   $m_s=(m_{s,x},m_{s,y})$ avec une pression $p_s=c_s^2 n_s$ (pas d'equation d'energie) :
   $\partial_t n_s+\nabla\cdot m_s=0$, $\partial_t m_s+\nabla\cdot(m_s\otimes m_s/n_s+c_s^2 n_s I)=z_s n_s E$.
2. **Champ propre.** L'ecart de densite cree $\phi$ par Poisson, $\nabla^2\phi=n_e-n_i$, et
   $E=-\nabla\phi$. Une separation de charge $n_e\ne n_i$ engendre un champ qui rappelle les
   especes l'une vers l'autre.
3. **Relaxation raide a la quasi-neutralite.** Ce rappel oscille a la frequence plasma
   $\omega_{pe}=\sqrt{4\pi n_0 e^2/m_e}$ (ici $\omega_{pe}$ est un parametre direct). Plus le plasma
   est dense, plus $\omega_{pe}$ est grand, plus la relaxation est rapide devant l'echelle
   d'interet (le transport, d'echelle $c_s/L$). C'est une echelle de temps raide : un schema
   explicite doit la resoudre, donc $\Delta t\,\omega_{pe}\lesssim1$, meme si la physique lente ne
   l'exige pas.

La propriete AP consiste a traiter le canal raide (2)-(3) implicitement, de sorte que le pas
$\Delta t$ soit fixe par le transport lent (1) et non par $\omega_{pe}$. Dans la limite
asymptotique $s=\Delta t\,\omega_{pe}\to\infty$ (plasma infiniment raide, ou pas de temps grand), le
schema doit rester stable et converger vers la solution quasi-neutre $n_e\approx n_i$. C'est
ce que la figure 1 mesure.

Ce modele est le bi-fluide isotherme electrostatique. Le couplage magnetique (rotation cyclotron,
run 2) est ajoute en option ; la version magnetisee non raide assemblee par composition vit dans
[`magnetic_isothermal_dsl`](../magnetic_isothermal_dsl/). Ce cas-ci ne couvre pas un champ $B$
auto-consistant : $B_z$ est uniforme et impose.

---

## 2. Les equations et qui les calcule (justifie : la physique est figee en C++, hors coeur)

Etat conservatif par espece, 3 composantes : $U_s=(n_s,m_{s,x},m_{s,y})$.

| Bloc | Equation | Kernel C++ (`two_fluid_ap.hpp`) |
|---|---|---|
| Transport (quantite de mvt) | $\partial_t m_s+\nabla\cdot(m_s\otimes m_s/n_s+c_s^2 n_s I)=0$ (predicteur) | `tfap_mstar` (Rusanov scinde, `:44-74`) |
| Transport (continuite) | $\partial_t n_s+\nabla\cdot m_s=0$ | `tfap_div_update` centre (`:77-86`) |
| Elliptique AP | $\nabla^2\phi=(n_e^*-n_i^*)/(1+\beta_0)$, $\beta_0=\Delta t^2(\omega_{pe}^2+\omega_{pi}^2)$ | RHS `:258-266` + `ell.solve()` |
| Force (terme raide) | $m_s^{n+1}=m_s^*+\Delta t\,z_s\,\omega_{ps}^2\,E$ (implicite) | `tfap_lorentz` (`:162-170`) |
| Force magnetisee | push de Boris (demi-E, rotation $B$, demi-E) | `tfap_boris` (`:179-193`) |

Ce cas n'appelle aucun scenario du coeur : la physique deux-fluides AP est ecrite ici, et
n'emprunte au coeur que des briques generiques (maillage, elliptique, parallele). Table 3 couches
"qui calcule quoi", chaque ligne pinnee a une ligne reelle :

| Ligne | Couche | Ce qui se passe |
|---|---|---|
| `TwoFluidAP(lib, n=.., omega_pe=.., stabilize=True)` (`run.py:173`, `:212`) | Python pilote | choix des parametres physiques + du flag AP ; lit l'etat via `ctypes` |
| `TwoFluidAP2D<GeometricMG>` instancie par `Solver` (`_two_fluid_ap.cpp:32-43`) | scenario C++ fige | l'integrateur AP complet : split IMEX, Poisson reformule, Boris. C'est le `.cpp` compile JIT |
| `for_each_cell(dom, [=] ADC_HD(i,j){...})` (`two_fluid_ap.hpp`, chaque kernel) | noyau par cellule (device-clean) | le calcul reel : flux Rusanov, divergence, push, sans callback Python dans le hot path |

La couche du milieu n'est pas une brique nommee `models.two_fluid_ap` : le solveur a quitte le coeur
parce que sa stabilisation couple $\Delta t$ dans l'elliptique (section 4), ce que la composition
`adc.System` bloc-a-bloc ne sait pas exprimer. La justification est en tete de `two_fluid_ap.hpp:5-9`
et `run.py:5-12` (`TwoFluidAP` "remplace l'ancien echappatoire interne `adc._adc._TwoFluidAP` retire
du coeur").

---

## 3. Pourquoi le solveur est compile a la volee (justifie : ancrage reel, hors-coeur)

`run.py` ne charge aucun binding C++ du cas : il compile `_two_fluid_ap.cpp` en `.dylib`/`.so` et la
charge par `ctypes`, exactement comme le JIT du DSL.

Build et chargement, `_build_lib` (`run.py:69-78`) :

```python
sources = [os.path.join(HERE, "_two_fluid_ap.cpp"), os.path.join(HERE, "two_fluid_ap.hpp")]
lib_path = native.build_shared("two_fluid_ap", sources)        # cache hors source, cle d'ABI
return native.load_symbols(lib_path, TFAP_SYMBOLS)             # 12 symboles tfap_* verifies
```

- `native.build_shared` (`common/native.py:104-144`) compile avec `-shared -fPIC -std=c++20 -O2 -I
  <adc_cpp/include>` et met la lib dans `out/two_fluid_ap/build/` (jamais a cote du `.cpp`,
  conforme a la note du manifeste). La lib est indexee par une cle d'ABI = hash du compilateur,
  des flags, des sources, et de la signature de l'arbre d'en-tetes du coeur : si un `.hpp` du
  coeur change, la cle change et la lib est recompilee. On ne recharge jamais une lib perimee.
- `native.load_symbols` (`common/native.py:147-164`) verifie que les 12 symboles `tfap_*`
  (`run.py:61-65`) existent : un symbole manquant leve une `RuntimeError` explicite au chargement,
  pas un `AttributeError` opaque au premier appel.

L'ABI C est minimale : `tfap_create(n, L, cse2, csi2, omega_pe, omega_pi, stabilize, eps,
upwind_continuity, omega_ce, omega_ci)` -> handle opaque, puis `tfap_step`/`tfap_advance` et les
diagnostics (`_two_fluid_ap.cpp:78-115`). Le flag `stabilize` (4e argument du `Solver`,
`_two_fluid_ap.cpp:36-38`) est le commutateur AP : `true`=$\beta_0$ actif, `false`=explicite. C'est
lui que `make_figures.py` bascule pour le contraste de la section 7.

---

## 4. Maths : la reformulation AP de l'elliptique (justifie Prouve : pas stable non effondre)

### 4.1 D'ou vient la borne explicite

Le canal raide est la boucle force-charge : la force $z_s n_s E$ avec $E=-\nabla\phi$ et
$\nabla^2\phi=n_e-n_i$. Linearise autour de $n_0=1$, une separation de charge
$\delta n=n_e-n_i$ oscille comme un oscillateur harmonique de pulsation
$\omega_p^2=\omega_{pe}^2+\omega_{pi}^2$ (les deux especes rappellent en parallele). Un schema
explicite sur un oscillateur de pulsation $\omega_p$ est stable si $\Delta t\,\omega_p\lesssim1$ :
au-dela, l'amplitude croit a chaque pas et diverge. Comme $\omega_{pe}\gg\omega_{pi}$ ici
($10^3$ vs $20$), la borne est essentiellement $s=\Delta t\,\omega_{pe}\lesssim1$. La figure 1 la
mesure : l'explicite est fini jusqu'a $s=1.0$ et NaN des $s=1.2$.

### 4.2 Le truc AP : absorber le terme raide dans le Poisson

Au lieu de resoudre $\nabla^2\phi=n_e^*-n_i^*$ puis d'appliquer la force explicitement (ce qui
re-introduit la borne), on rend la force implicite. Schematiquement, pour le predicteur de densite
apres le push implicite,

$$n_s^{n+1}=n_s^*-\Delta t\,\nabla\cdot m_s^{n+1},\qquad m_s^{n+1}=m_s^*+\Delta t\,z_s\,\omega_{ps}^2\,E,\qquad E=-\nabla\phi.$$

En reportant le push dans la divergence et en utilisant $\nabla\cdot(n_s\nabla\phi)\approx n_0\nabla^2\phi$
($n_0=1$), la contrainte $\nabla^2\phi=n_e^{n+1}-n_i^{n+1}$ devient, apres regroupement des termes en
$\nabla^2\phi$ :

$$\big(1+\Delta t^2(\omega_{pe}^2+\omega_{pi}^2)\big)\,\nabla^2\phi=n_e^*-n_i^*\quad\Longrightarrow\quad\boxed{\nabla^2\phi=\dfrac{n_e^*-n_i^*}{1+\beta_0}},\quad\beta_0=\Delta t^2(\omega_{pe}^2+\omega_{pi}^2).$$

Le pas de temps $\Delta t$ apparait dans le membre de droite de l'elliptique : c'est exactement
ce que la composition `adc.System` ne sait pas exprimer (un bloc ne connait pas $\Delta t$ a
l'assemblage du Poisson), d'ou le solveur sur mesure. Chaque symbole pointe sa ligne :

```cpp
const Real beta0 = stabilize ? dt * dt * (ce + ci) : Real(0);   // ce=wpe^2, ci=wpi^2 (hpp:258)
const Real inv = Real(1) / (Real(1) + beta0);                   // facteur AP (hpp:262)
r(i, j, 0) = (ne(i, j, 0) - ni(i, j, 0)) * inv;                 // RHS Poisson reformule (hpp:264)
```

- `ce`, `ci` sont $\omega_{pe}^2$, $\omega_{pi}^2$, caches dans le solveur (`two_fluid_ap.hpp:201`,
  `:215-216`). A $s=5$ : $\beta_0=\Delta t^2\omega_{pe}^2\approx(5\times10^{-3}\cdot10^3)^2=25$, donc
  $1/(1+\beta_0)\approx 0.038$ : le RHS de charge est divise par 26, ce qui borne la reponse.
- `stabilize` decide $\beta_0$ vs $0$. A $\beta_0=0$ on retombe sur le Poisson nu + force explicite :
  c'est le schema explicite de la figure 1.

### 4.3 Ce que l'AP preserve, et ce que la figure teste

Dans la limite $s\to\infty$, $\beta_0\to\infty$ et $1/(1+\beta_0)\to0$ : le RHS de Poisson est
ecrase, le champ ne sur-reagit plus, et le systeme relaxe vers la solution quasi-neutre
$n_e\approx n_i\approx n_0$ au lieu d'osciller. La prediction falsifiable est donc : la deviation
$\max\lvert n_e-1\rvert$ doit rester bornee (et meme plateauer) quand $s\to\infty$, alors qu'un
schema explicite diverge des $s\gtrsim1$. Ce qu'une mesure differente trahirait : une deviation AP qui
croit avec $s$ signalerait une stabilisation incorrecte ($\beta_0$ mal forme, ou le push non
reellement implicite) ; un explicite qui survit a $s\gg1$ signalerait que le canal raide n'est
pas active (couplage $\omega_{ps}^2$ nul). On mesure (section 7) un plateau AP a $5.41\times10^{-7}$
et un explicite NaN des $s=1.2$ : la propriete AP tient.

---

## 5. Code du scenario, kernel par kernel (justifie : ancrage reel)

Un pas `TwoFluidAP2D::step(dt, stabilize)` (`two_fluid_ap.hpp:239-290`) est un split IMEX. Ordre reel :

1. **Ghosts periodiques** : `fill_boundary(e/ion, dom, per)` (`:245-246`).
2. **Predicteur quantite de mvt** `m*` (`tfap_mstar`, `:247-248`) : flux d'Euler isotherme par Rusanov
   (local Lax-Friedrichs) dimensionnellement scinde, vitesse d'onde $a=\lvert u\rvert+c_s$,
   $F_{xx}=m_x^2/n+c^2 n$, $F_{yy}=m_y^2/n+c^2 n$. Lit $n,m_x,m_y$ avec 1 ghost.
3. **Predicteur densite** `n*` (`tfap_div_update`, `:254-257`) : $n-\Delta t\,\nabla\cdot m^*$,
   divergence centree ordre 2 (defaut `upwind_continuity=false`, le seul utilise par `run.py`).
4. **Poisson AP** : RHS $(n_e^*-n_i^*)/(1+\beta_0)$ (`:258-266`) puis `ell.solve()` (`:267`).
5. **Champ** $E=-\nabla\phi$ (`tfap_efield`, `:269`).
6. **Push implicite (terme raide)** : non magnetise `tfap_lorentz` $m^{n+1}=m^*+\Delta t\,z\,\omega_{ps}^2 E$
   (`:274-275`) ; magnetise push de Boris symetrique `tfap_boris` (`:271-272`).
7. **Correcteur densite** $n^{n+1}=n-\Delta t\,\nabla\cdot m^{n+1}$ (`:285-286`) + recopie de
   $(m_x,m_y)$ dans l'etat (`copy_mom`, `:288-289`).

Le push de Boris (`tfap_boris`, `:179-193`) est exact pour la rotation : demi-impulsion electrique,
rotation $B$ complete d'angle $\theta=z\,\omega_c\,\Delta t$, seconde demi-impulsion. Il reproduit
exactement la derive $E\times B$ et conserve $\lvert m\rvert$ sous $B$ seul, sans croissance
seculaire ; quand $\omega_c=0$ il se reduit a `tfap_lorentz` (commentaire `:174-178`). C'est ce qui
rend le run 2 stable sans limite $\omega_c\,\Delta t$.

Device-clean : tous les kernels passent par `for_each_cell` avec lambdas `ADC_HD`, $\lvert
x\rvert$/max/minmod via ternaires (`tfap::ab/mx2/mm2`, `:35-39` : `std::fabs`/`std::fmax` ne sont pas
device-safe), $\cos$/$\sin$/$\sqrt$ calcules cote hote pour les champs uniformes. La facade compile
donc telle quelle pour GPU si on passe les flags adequats ; ce cas ne fixe aucun flag de backend
(CPU serie ici).

**Conditions initiales** `TwoFluidAP2D::init(eps)` (`two_fluid_ap.hpp:227-237`), boucle hote :

```cpp
const Real k = 2 * pi / L;                                          // mode 1 diagonal (hpp:231)
ae(i, j, 0) = Real(1) + eps * std::cos(k * x_cell(i) + k * y_cell(j));  // n_e = 1 + eps cos(kx+ky)
ai(i, j, 0) = Real(1);                                              // n_i = 1 (fond uniforme)
```

- La perturbation ne porte que sur $n_e$ ; la charge nette initiale $n_i-n_e=-\epsilon\cos(\cdot)$
  est d'ordre $\epsilon=10^{-3}$. C'est cette separation de charge que la dynamique raide relaxe.
  $m_s=0$ (repos). Le pilote `TwoFluidAP` passe le defaut `eps=1e-3` (`run.py:115`).

---

## 6. Diagnostics et leur fiabilite (justifie Ne prouve pas : proxys)

Les diagnostics C++ (`_two_fluid_ap.cpp`) :

- `tfap_mass_e/i` = `adc::sum(.., 0)` : somme de $n$ sur les cellules, sans poids $dx^2$. Vaut
  $4096=64\times64\times1$ (fond 1, perturbation $\cos$ de moyenne nulle). C'est un proxy de
  conservation relative, pas une masse physique calibree.
- `tfap_max_charge` = $\max\lvert n_i-n_e\rvert$, `tfap_max_dev` = $\max\lvert n_e-1\rvert$
  (`:54-73`), precedes de `device_fence()` (barriere host/device, memoire unifiee GPU).

Piege important (justifie la clause Ne prouve pas) : `tfap_max_dev` fait `std::fmax` sur le champ
et propage mal les NaN. Verifie : pour le schema explicite a $s=5$ (champ entierement NaN, 4096
cellules), `tfap_max_dev()` rend `0.0` et `tfap_max_charge()` rend `0.0`. Un `0.0` du
diagnostic C++ ne prouve donc pas que le schema est stable. `make_figures.py` ne s'y fie pas : il lit
le champ via `density_e()`/`density_i()` cote Python et teste `np.isfinite` (`make_figures.py`,
`_field_diag`). C'est cette lecture cote champ, insensible au NaN, qui distingue "borne" de "explose".

Les `assert` de `run.py` ne sont pas pieges parce qu'ils ne testent que le schema AP
(`stabilize=True`, jamais NaN) : `np.isfinite(...)` (`run.py:195-197`), `max_dev<0.1` et
`max_charge<0.1` (`run.py:199-200`), `mass_rel<1e-7` (`run.py:202`). Ce sont des bornes larges, pas
un test de l'ordre AP.

---

## 7. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (meme solveur JIT que `run.py`), versionnees avec
`figures/provenance.json`. Commande exacte en section 9.

### `ap_vs_explicit.png` : la propriete AP

![max|n_e-1| vs raideur s : AP plateaute, explicite NaN des s>=1.2](figures/ap_vs_explicit.png)

A $\Delta t=5\times10^{-3}$ et horizon 200 pas fixes, on balaie $s=\Delta t\,\omega_{pe}$ en variant
$\omega_{pe}$ (avec $\omega_{pi}=0.02\,\omega_{pe}$, ratio du run 1), pour le schema AP
(`stabilize=True`) et explicite (`stabilize=False`).

- **Prouve** (mesure cote champ, `np.isfinite`) : la deviation AP (bleu) reste bornee sur tout le
  balayage et plateaute a $5.41\times10^{-7}$ pour $s\in[1,50]$ (valeurs : $s=5\to5.325\times10^{-7}$,
  $s=10\to5.390\times10^{-7}$, $s=50\to5.414\times10^{-7}$). Le schema explicite (rouge) suit l'AP
  tant que $s\le1.0$ ($s=1.0\to5.416\times10^{-7}$) puis devient NaN des $s=1.2$ (croix rouges) :
  la borne explicite $s=\Delta t\,\omega_{pe}\approx1$ (trait gris) est exactement celle predite en
  4.1. C'est la propriete asymptotic-preserving : le pas stable ne s'effondre pas quand $s\to\infty$.
- **Suggéré (non assere)** : la deviation AP decroit puis plateaute quand $s$ croit (de
  $5.9\times10^{-4}$ a $s=0.05$ vers $5.4\times10^{-7}$) : la limite quasi-neutre est de mieux en
  mieux approchee a forte raideur. C'est coherent avec l'AP (la stabilisation ecrase le RHS de
  charge), mais aucun assert ne teste la monotonie ni la valeur du plateau.
- **Non montré** : aucun `assert` de `run.py` ne teste ce contraste (les asserts ne touchent que le
  schema AP). La figure ne montre pas pourquoi l'explicite diverge (croissance pas-a-pas de
  l'oscillation de charge) : on observe l'etat NaN final, pas la trajectoire de l'instabilite.

### `final_state.png` : l'etat quasi-neutre des deux fluides

![n_e, n_i, charge nette n_i-n_e a s=5 : bandes diagonales, charge ~6e-11](figures/final_state.png)

Etat final du run raide de reference ($s=5$, AP, 200 pas).

- **Prouve / mesure** : $n_e$ et $n_i$ valent tous deux $1+5.3\times10^{-7}\cos(kx+ky)$ (bandes
  diagonales, suivant la CI) : le plasma est quasi-neutre, les deux especes ont relaxe vers le
  meme profil malgre $s=5$ (un explicite aurait deja NaN). La charge nette $n_i-n_e$ est d'ordre
  $6.7\times10^{-11}$ : trois ordres sous la deviation, et bien sous la separation de charge
  initiale $\epsilon=10^{-3}$. La quasi-neutralite est imposee, pas supposee.
- **Suggéré** : la charge nette porte une texture en damier d'echelle grille (au niveau
  $\sim6\times10^{-11}$) : c'est le bruit dispersif de la continuite centree (dissipation nulle)
  a amplitude residuelle, plausible a l'oeil mais non quantifie par un assert.
- **Non montré** : a $\epsilon=10^{-3}$ et schema bas ordre, aucune dynamique non lineaire ni
  separation de charge macroscopique. La carte ne dit rien du schema explicite (qui n'a pas d'etat
  final fini a $s=5$).

---

## 8. Les tolerances, justifiees par un ordre de grandeur (justifie 8 de la checklist)

| Tolerance | Valeur | Pourquoi cette valeur |
|---|---|---|
| `max_dev < 0.1`, `max_charge < 0.1` (`run.py:199-200`) | $0.1$ | Borne large de quasi-neutralite : la perturbation initiale est $\epsilon=10^{-3}$ et la deviation AP mesuree est $5.3\times10^{-7}$, soit ~5 ordres sous $0.1$. La tolerance rejette une explosion (deviation $O(1)$ ou NaN) sans rejeter le signal physique ; elle n'est pas un test de l'ordre AP |
| `mass_rel < 1e-7` (`run.py:202`, `:235`) | $10^{-7}$ | Le schema est conservatif en masse (continuite en divergence) : la seule derive est l'arithmetique flottante. Mesure : $2.276\times10^{-14}$ (run 1) / $1.665\times10^{-14}$ (run 2), ~7 ordres sous la tolerance, au niveau du bruit IEEE754 sur une somme de 4096 termes |
| `np.isfinite(...)` (`run.py:195-197`, `:232-233`) | (booleen) | Garde-fou minimal : un grand pas qui explose donne NaN/Inf. C'est le seul test cote `run.py` qui detecterait directement une perte de stabilite AP (mais seul le schema AP est exerce ici, jamais l'explicite) |

---

## 9. Reproduire (justifie 14 de la checklist : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/two_fluid_ap
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~3.6 s (cache a jour)
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 figures + provenance.json
```

Prerequis : `numpy` (et `matplotlib` pour les figures, hors `needs` du cas), un compilateur
C++20 (`needs=["cxx"]` : le concept `EllipticSolver` et `static_assert` l'exigent), et les
en-tetes du coeur `adc_cpp/include/` localises par `native.adc_include()` (via `$ADC_INCLUDE`, sinon
depuis le paquet `adc`, sinon `../adc_cpp/include`). Le module `adc` n'est utilise que pour localiser
les en-tetes ; le calcul AP ne passe pas par les bindings pybind11.

Sortie attendue de `run.py` (capturee, macOS arm64, Apple clang 21) :

```
[run 1 - raide, non magnetise]
  dt=5.000e-03  nsteps=200  dt*omega_pe=5.0  (explicite EXPLOSERAIT)
  max_dev()    = 5.325451e-07   (ecart a la quasi-neutralite)
  max_charge() = 6.697598e-11   (charge nette locale)
  mass_e: 4.096000e+03 -> 4.096000e+03   (err. relative 2.276e-14)
[run 2 - raide magnetise]
  max_dev()    = 9.447867e-04   max_charge() = 7.732753e-04
  mass_e: 4.096000e+03 -> 4.096000e+03   (err. relative 1.665e-14)
OK two_fluid_ap
```

Cout : ~3.6 s temps mur (cache de build a jour), ~5.5 s au premier appel (compilation de la
`.dylib` incluse). La recompilation n'est declenchee que si la cle d'ABI change (compilateur, flags,
sources, ou en-tetes du coeur). Artefact hors source (gitignore) :
`out/two_fluid_ap/build/_two_fluid_ap.dylib` (+ `.abikey`). Caveat plateforme : les signes,
l'ordre de grandeur des deviations ($\sim5\times10^{-7}$), la borne explicite ($s\approx1$) et le
verdict `OK` sont stables ; les derniers chiffres varient avec l'ordre de sommation et le solveur
multigrille (cf. `figures/provenance.json`).

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | pilote Python : build JIT (`_build_lib`), `ctypes` (`_bind`), classe `TwoFluidAP`, 2 runs, asserts |
| `two_fluid_ap.hpp` | physique AP : kernels Rusanov/MUSCL/Boris, Poisson reformule, `TwoFluidAP2D<Elliptic>` |
| `_two_fluid_ap.cpp` | ABI `extern "C"` (`tfap_create`/`step`/`advance`/diagnostics) + instancie `<GeometricMG>` |
| `make_figures.py` | balaie la raideur (AP vs explicite) + etat final ; ecrit les 2 figures + `provenance.json` |
| `figures/ap_vs_explicit.png`, `final_state.png` | assets versionnes, regeneres en place |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, compilateur, balayage de raideur, plateau AP mesure |
| `../adc_cases/common/native.py` | `build_shared` (cache hors source + cle d'ABI), `load_symbols` |
