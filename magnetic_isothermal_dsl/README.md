# magnetic_isothermal_dsl : fluide isotherme magnetise ecrit en formules, valide sans oracle natif

Un fluide d'Euler ISOTHERME (fermeture $p=c_s^2\rho$) couple a Poisson, avec une force de Lorentz
pilotee par un champ $B_z$ constant lu dans le canal auxiliaire ETENDU (indice 3). Toute la physique
est DECLAREE en expressions symboliques (`adc.dsl.Model`) ; le DSL genere le C++, le compile en `.so`
et l'installe via `add_equation(...)`. Particularite : AUCUN modele natif de reference n'existe pour
ce modele (pas de brique nommee "magnetic_isothermal" cote coeur). La correction n'est donc PAS
prouvee contre un oracle natif. Elle l'est par (1) parite inter-backend production/aot quand les deux
se lient, et (2) un oracle de Lorentz ANALYTIQUE en numpy. Sur macOS, le backend `production` ne se
lie pas (ABI des en-tetes du module pre-construit) : seul `aot` est exerce, et c'est (2) qui porte la
preuve. La physique de la force de Lorentz et la fermeture isotherme ne sont pas re-derivees ici :
elles renvoient aux briques du coeur et au cas magnetise complet
[`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/).

## Contrat

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | `validation` (`cases_manifest.toml`, `magnetic_isothermal_dsl/run.py`, `ci = true`, `needs = ["cxx"]`) |
| Entrees | grille $32^2$, $L=1$, **periodique** ; CI $\rho=1+0.05\cos(2\pi x)$, $m_x=0.3\rho$ ($u=0.3$), $m_y=0$ ; $c_s^2=1$, $q=-1$, $B_z=2$ (constant) ; schema minmod + Rusanov, SSPRK2, 40 pas a CFL$=0.4$ |
| Sorties | etat `(3,n,n)=[\rho,m_x,m_y]` via `get_state("plasma")` ; `eval_rhs("plasma")` (residu local) ; 2 figures dans `figures/` + `figures/provenance.json` ; les `.so` DSL sous `out/magnetic_isothermal_dsl/` |
| Invariants garantis | les `assert` de `run.py` : oracle Lorentz `err_x == 0 and err_y == 0` (`run.py:217`) et canal densite `max\|dR[0]\| == 0` (`run.py:221`) ; `lor_contrib > 0` (`run.py:222`) ; parite inter-backend `np.array_equal` SI $\geq 2$ backends (`run.py:196`) ; masse `drift < 1e-9` (`run.py:240`) ; rotation `\|\langle m_y\rangle\| > 1e-6` (`run.py:242`) |
| PROUVE | le terme magnetique compile vaut EXACTEMENT $(B_z m_y,\,-B_z m_x)$ : `err_x = err_y = 0.000e+00` (egalite bit, numpy) ; $B_z$ ne touche jamais la densite (`dR[0]==0`) ; il est non nul ($\max\|dR\|=6.299\times10^{-1}$) ; la masse derive de $2.887\times10^{-15}$ ; la quantite de mouvement moyenne TOURNE de $\langle m\rangle=(0.3,0)\to(0.2162,-0.2080)$, angle $-43.88^\circ$, a comparer a $\omega_c t=-43.88^\circ$ |
| NE PROUVE PAS | **pas une reproduction publiee** et **pas de parite DSL-vs-natif** ici (aucune brique native "magnetic_isothermal" n'existe ; le `MagneticLorentzForce` du coeur n'est pas branche en Python dans ce cas). Sur macOS le backend `production` ne se lie pas (ABI en-tetes) : la **parite inter-backend est SAUTEE**, un seul chemin (`aot`) est verifie. L'oracle ne teste QUE le terme magnetique (difference $B_z\!=\!B_0$ moins $B_z\!=\!0$), pas le flux ni l'electrostatique. Regime EXPLICITE (pas le Schur condense raide) ; $B_z$ uniforme |
| Provenance | adc_cpp `01873299`, adc_cases `a9541ba4`, backend DSL `aot` (production non lie), $32^2$, ~19 s temps mur (2 figures, recompile la `.so` deux fois), macOS arm64 ; `figures/provenance.json` |

A la fin tu sauras : quelles conventions du coeur les formules DSL reproduisent (table ancree
`physics/*.hpp`), comment la correction est etablie SANS oracle natif (oracle analytique + parite
inter-backend), pourquoi `production` echoue sur cette plateforme, et ce qu'une divergence de
l'oracle trahirait.

---

## 1. Renvoi : la physique n'est pas re-derivee ici

C'est un cas d'EQUIVALENCE DSL. La derivation du systeme Euler-isotherme magnetise (flux, fermeture,
force de Lorentz, etage source) appartient au cas magnetise complet
[`../hoffart_euler_poisson_dsl/`](../hoffart_euler_poisson_dsl/) (vise arXiv:2510.11808, Schur
condense). Le mecanisme de la rotation cyclotron est celui de toute force $q\,v\times B$ : $v\times B$
est PERPENDICULAIRE a $v$, donc la force ne fait pas de travail ($F\cdot v=0$) et la quantite de
mouvement TOURNE a la pulsation cyclotron sans changer de module. Ce cas ne re-derive pas cela : il
VERIFIE que le terme compile depuis les formules a la bonne forme. Le coeur du tutoriel est donc la
table des conventions (section 2) et le protocole de preuve sans oracle natif (sections 3-4).

Equations resolues (variables conservatives $U=(\rho, m_x, m_y)$, $m=\rho v$) :

$$\partial_t\rho+\nabla\cdot(\rho v)=0,\qquad
\partial_t m+\nabla\cdot(\rho v\otimes v + p\,I)=q\rho\,E + q\,v\times B_z\hat z,\qquad p=c_s^2\rho,$$

avec $E=-\nabla\phi$, $-\nabla^2\phi=\text{(densite de charge)}=q\rho$ couple au Poisson du systeme.
Projete en 2D, $v\times B_z\hat z$ donne $(+B_z m_y,\,-B_z m_x)$ sur la quantite de mouvement, et $0$
sur l'energie (absente ici : modele isotherme, 3 variables).

---

## 2. Les conventions du coeur reproduites, ancrees `include/adc/physics/*.hpp`

Le DSL ne nomme aucune brique : il REDECLARE leurs formules. L'egalite bit (quand 2 backends se
lient) et l'oracle analytique ne tiennent que parce que chaque formule reproduit EXACTEMENT la
convention C++ correspondante. La couche du milieu d'un cas DSL n'est pas une brique nommee mais les
EXPRESSIONS que `adc.dsl` compile :

| Ligne `run.py` (expression DSL) | Convention du coeur reproduite | Formule |
|---|---|---|
| `m.flux(x=[mx, mx*u + cs2*rho, mx*v], y=[my, my*u, my*v + cs2*rho])` (`run.py:102-103`) | `IsothermalFlux::flux` (`hyperbolic.hpp:132-141`) | $F_x=(m_x,\,m_x u + c_s^2\rho,\,m_x v)$, $F_y=(m_y,\,m_y u,\,m_y v + c_s^2\rho)$ |
| `m.eigenvalues(x=[u-cs,u,u+cs], y=[v-cs,v,v+cs])` (`run.py:105`) | `IsothermalFlux::eigenvalues` (`hyperbolic.hpp:165-174`) | $(v_n-c_s,\,v_n,\,v_n+c_s)$, $c_s=\sqrt{c_s^2}$ |
| `q*rho*(-gx)` / `q*rho*(-gy)` (`run.py:110-111`) | `PotentialForce::apply` (`source.hpp:36-43`) | $s_1=q\rho E_x$, $E_x=-\,$`grad_x` ; idem $s_2$ |
| `+ bz*my` / `- bz*mx` (`run.py:110-111`) | `MagneticLorentzForce::apply` (`source.hpp:84-93`) | $s_1=q_{om}B_z m_y$, $s_2=-q_{om}B_z m_x$, $s_3=0$ (travail nul) |
| `m.elliptic_rhs(q*rho)` (`run.py:118`) | `ChargeDensity::rhs` (`elliptic.hpp:19-25`) | $f=q\rho$ (second membre du Poisson de systeme) |

Trois subtilites de convention, verifiees sur le code, pas plaquees :

- **Signe de la force electrostatique.** `PotentialForce` (`source.hpp:37`) pose `Ex = -a.grad_x` puis
  `s[1] = qom*u[0]*Ex`. La formule DSL ecrit `q*rho*(-gx)` : meme signe, $q\rho(-\partial_x\phi)$. Ici
  `q = -1` (charge electronique, comme la brique).
- **Le canal aux ETENDU.** `B_z` est la composante canonique 3 de `adc::Aux` ; `MagneticLorentzForce`
  declare `n_aux = 4` (`source.hpp:82`) pour que `load_aux` la remplisse. Cote DSL, `m.aux("B_z")`
  (`run.py:96`) declare la 4e composante ; `add_equation` elargit le canal partage et
  `sim.set_magnetic_field(B0*ones)` (`run.py:146`) la peuple. C'est ce que les deux autres
  demonstrateurs DSL (mono-espece, multi-espece) ne couvraient pas : une SOURCE qui lit au-dela du
  contrat de base $\phi/\nabla\phi$ (indices 0/1/2).
- **Travail nul du terme magnetique.** `MagneticLorentzForce` laisse `s[3]` a $0$ meme a 4 variables
  (`source.hpp:91`) car $v\times B\perp v$. Le modele isotherme n'a que 3 variables (pas d'energie) :
  la question ne se pose pas ici, mais la rotation a module conserve (figure 2) en est la consequence
  directe.

$q_{om}=q/m$ dans la brique ; le cas pose $q_{om}=q=-1$ (masse absorbee). Donc la pulsation cyclotron
effective est $\omega_c=q_{om}B_z=q B_z=(-1)(2)=-2$, signe negatif (giration horaire).

---

## 3. La prediction falsifiable : egalite bit + oracle analytique (justifie PROUVE / NE PROUVE PAS)

Le cas calcule la prediction par DEUX voies independantes, parce qu'il n'a PAS d'oracle natif :

**(A) Parite inter-backend** (`run.py:188-200`). Si `production` ET `aot` se lient, leurs `eval_rhs`
sont confrontes par `np.array_equal` : `assert np.array_equal(r_b, r_ref)` (`run.py:196`), sans
tolerance. Les deux backends inlinent le MEME chemin de production sur le MEME modele genere ; toute
divergence trahirait un non-determinisme du codegen ou une difference de marshaling host. Sur macOS,
`production` ne se lie pas (section 5) : cette voie est SAUTEE et le `run.py` l'imprime explicitement
(`run.py:199-200`).

**(B) Oracle Lorentz analytique** (`run.py:202-222`). On lie le modele DEUX fois : $B_z=B_0=2$ et
$B_z=0$. Flux et electrostatique sont identiques entre les deux runs ; la SEULE difference est le
terme magnetique. Donc le residu

$$\Delta R=\texttt{eval\_rhs}(B_z{=}B_0)-\texttt{eval\_rhs}(B_z{=}0)$$

doit valoir EXACTEMENT, canal par canal, la forme analytique calculee en numpy :

```python
lorentz_x = B0 * my0     # +B_z m_y sur la qte de mvt x (run.py:206)
lorentz_y = -B0 * mx0    # -B_z m_x sur la qte de mvt y (run.py:207)
dR = eval_rhs(B0) - eval_rhs(0)
err_x = max|dR[1] - lorentz_x|   # attendu 0 (run.py:211)
err_y = max|dR[2] - lorentz_y|   # attendu 0 (run.py:212)
assert err_x == 0.0 and err_y == 0.0          # egalite bit (run.py:217)
assert max|dR[0]| == 0.0                        # densite jamais touchee (run.py:221)
assert lor_contrib > 0.0                        # B_z bien lu, terme non nul (run.py:222)
```

Mesure (backend `aot`) : `err_x = 0.000e+00`, `err_y = 0.000e+00`, `max|dR| = 6.299e-01`. Comme
$m_y(0)=0$ partout (CI), $\texttt{lorentz\_x}=B_0 m_y(0)\equiv 0$ et le canal $m_x$ de $\Delta R$ est
identiquement nul ; tout le terme magnetique vit dans $\Delta R[2]=-B_0 m_x(0)$, dans
$[-0.6299,-0.5701]$ (car $m_x(0)=0.3\rho_0$, $\rho_0=1\pm0.05$). L'egalite est SANS tolerance : le
codegen lit `B_z` au bon indice et applique la bonne forme.

Pourquoi $== 0.0$ exactement et non $\sim10^{-16}$ : $\Delta R$ soustrait deux `eval_rhs` qui ne
different QUE par le terme magnetique ; les contributions flux et electrostatique, IDENTIQUES entre
les deux runs, s'annulent bit-a-bit, et le terme magnetique restant est la MEME expression
`bz*my - ...` que la forme analytique (meme ordre d'operations flottantes). Aucun arrondi residuel.

Les deux autres tolerances ne sont pas des egalites bit mais des bornes justifiees par un ordre de
grandeur. `drift < 1e-9` (`run.py:240`) : le schema volumes finis est conservatif, la masse est un
invariant exact et la seule derive est l'arithmetique flottante ; mesure $2.887\times10^{-15}$, ~6
ordres sous la borne. `|<m_y>| > 1e-6` (`run.py:242`) : borne BASSE separant le bruit machine du
signal physique ; $m_y(0)=0$ exactement, donc toute valeur au-dessus de $10^{-6}$ ne peut venir que
du terme de Lorentz ; mesure $|\langle m_y\rangle|=0.208$, ~5 ordres au-dessus de la borne.

---

## 4. Ce qu'une divergence trahirait

L'oracle est un test de NON-REGRESSION du codegen DSL sur le canal aux etendu. Une valeur non nulle
de chaque assert pointe une faute precise :

- `err_x != 0` ou `err_y != 0` : le terme compile n'est pas $(B_z m_y, -B_z m_x)$. Causes : mauvais
  indice aux lu (`B_z` confondu avec `phi`/`grad`), signe inverse ($+B_z m_x$ au lieu de $-B_z m_x$),
  ou $q_{om}$ applique deux fois.
- `max|dR[0]| != 0` : le terme magnetique a contamine la densite, ce qui est physiquement impossible
  (Lorentz n'agit que sur la quantite de mouvement). Trahirait un melange de composantes au codegen.
- `lor_contrib == 0` : `B_z` n'est PAS lu (canal aux non elargi, `set_magnetic_field` sans effet, ou
  `m.aux("B_z")` oublie) ; le terme est partout nul et le modele est sans Lorentz.
- parite `np.array_equal` False (quand 2 backends) : non-determinisme entre `production` (natif
  zero-copie) et `aot` (host-marshale) sur le MEME modele : marshaling fautif ou ordre de sommation
  divergent. Ici non teste (un seul backend).

---

## 5. Pourquoi `production` ne se lie pas sur macOS

`bind_backends` (`run.py:151-169`) tente `production` puis `aot` et ne garde que ceux effectivement
LIES. La sortie reelle :

```
backend 'production' indisponible (RuntimeError), essai suivant
backends DSL lies : 'aot'
parite inter-backend SAUTEE (un seul backend lie ... 'aot') ; correction prouvee par l'oracle ...
```

La cause N'EST PAS le namespace a deux niveaux du dlopen : c'est une **incompatibilite d'ABI des
en-tetes**. Le loader natif (`add_native_block`) verifie que la signature des en-tetes contre
lesquels la `.so` est compilee correspond a celle du module `_adc` deja charge. Message exact capture :

```
add_native_block : ABI incompatible -- cle du loader 'compiler=Apple LLVM 21.0.0;std=202302L;
headers=079c02c0...' != cle du module 'compiler=Apple LLVM 21.0.0;std=202302L;headers=f8273719...'.
```

Le module `build-master` est PRE-CONSTRUIT : sa signature d'en-tetes (`f8273719...`) differe de
l'arbre `include/` courant (`079c02c0...`) que la `.so` DSL embarque. La compilation `production`
REUSSIT ; c'est le BRANCHEMENT qui rejette le loader. Le backend `aot` n'a pas cette garde (pas de cle
d'ABI verifiee) et reste fonctionnel. Avec un module `_adc` rebati contre les memes en-tetes que
`include/`, `production` se lierait et la parite inter-backend serait alors exercee. Comportement
identique a celui documente pour [`../diocotron_dsl/`](../diocotron_dsl/).

Consequence honnete : sur cette plateforme la preuve repose ENTIEREMENT sur la voie (B) (oracle
analytique) plus les invariants masse/rotation (section 6). C'est suffisant pour ce qu'on affirme (le
terme magnetique a la bonne forme et est exerce), pas pour une parite de chemins.

---

## 6. Figures (generees par `make_figures.py`, dans `figures/`)

Generees par `python make_figures.py` (memes parametres et meme modele DSL que `run.py`),
versionnees avec `figures/provenance.json`. Commande exacte en section 8.

### `lorentz_oracle.png` : le residu DSL confronte a la forme analytique

![Trois cartes du residu err_rho, err_mx, err_my, toutes blanches a l'echelle eps machine](figures/lorentz_oracle.png)

- **PROUVE** (asserte `run.py:217,221`) : les trois cartes du residu ($\Delta R_\rho-0$,
  $\Delta R_{m_x}-B_0 m_y$, $\Delta R_{m_y}-(-B_0 m_x)$) sont IDENTIQUEMENT au centre neutre (blanc),
  `max|.| = 0.0e+00` partout. L'echelle de couleur est ancree a $\pm\epsilon_{\text{mach}}=2.22\times
  10^{-16}$ : tout pixel non nul (au-dela du dernier bit) SATURERAIT en bleu ou rouge. Aucun ne
  sature : le terme magnetique compile egale la forme numpy au bit pres, et la densite (panneau
  gauche) n'est jamais touchee.
- **SUGGERE (non assere)** : rien. L'egalite est exacte, pas approchee ; il n'y a pas de structure a
  lire au-dela du zero.
- **NON MONTRE** : la figure ne couvre QUE la difference $B_z\!=\!B_0$ moins $B_z\!=\!0$, donc le seul
  terme magnetique. Elle ne teste ni le flux isotherme, ni l'electrostatique, ni la parite
  inter-backend (un seul backend lie). Un residu sur le flux passerait inapercu ici.

### `cyclotron_trajectory.png` : la rotation de Lorentz a $\omega_c t$

![A gauche la trajectoire de (m_x, m_y) sur le cercle cyclotron ; a droite le module conserve](figures/cyclotron_trajectory.png)

- **PROUVE / mesure** (asserte `run.py:240,242`) : partant de $\langle m\rangle=(0.3,0)$ (purement
  longitudinal, $m_y=0$), la quantite de mouvement moyenne TOURNE vers $(0.2162,-0.2080)$ apres 40
  pas ($t=0.3829$). L'angle final mesure $-43.88^\circ$ coincide avec la prediction cyclotron
  $\omega_c t=(-2)(0.3829)=-43.88^\circ$ (rapport $1.00006$). Le module $|\langle m\rangle|$ (panneau
  droit) est conserve : derive relative $-1.3\times10^{-7}$ sur l'horizon. La masse derive de
  $2.887\times10^{-15}$. Comme $m_y(0)=0$, toute composante transverse apparue ($\langle m_y\rangle=
  -0.2080$) vient EXCLUSIVEMENT du terme de Lorentz : la physique magnetique est exercee.
- **SUGGERE (non assere)** : le tres leger ecart au cercle analytique (rapport $1.00006$) et la
  legere variation du module ($\sim10^{-7}$) sont la signature de la discretisation en temps finie
  (SSPRK2, $\omega_c\,dt$ non infinitesimal) et de la dynamique de pression/Poisson superposee a la
  rotation ; aucun assert ne quantifie cet ecart.
- **NON MONTRE** : pas de regime raide (le Schur condense de
  [`../schur_magnetized_cartesian/`](../schur_magnetized_cartesian/) n'est pas teste) ; horizon court
  (40 pas, moins d'un quart de tour) ; pas de comparaison a une trajectoire publiee.

---

## 7. Limites (ce que ce cas ne capture pas)

- **Pas de parite DSL-vs-natif.** Contrairement a [`../diocotron_dsl/`](../diocotron_dsl/) qui
  confronte le DSL a `models.diocotron` (briques natives), il n'existe AUCUN modele natif assemble
  "magnetic_isothermal" branche cote Python ici. La brique `MagneticLorentzForce` existe
  (`source.hpp`) mais n'est pas composee en oracle dans ce cas : la reference est analytique, pas
  native.
- **Parite inter-backend conditionnelle.** Elle n'est exercee que si $\geq 2$ backends se lient ; sur
  macOS (module pre-construit) un seul (`aot`) se lie, la voie est sautee.
- **L'oracle ne teste qu'un terme.** Par construction (difference de deux runs), il isole le terme
  magnetique ; le flux isotherme et l'electrostatique sont identiques entre les deux et s'annulent.
  Leur correction repose sur la fidelite des formules aux conventions du coeur (table section 2),
  non sur cet oracle.
- **Regime explicite, $B_z$ uniforme, horizon court.** Pas de raideur cyclotron (Schur), champ
  magnetique constant, 40 pas. On observe la rotation et l'exactitude du terme, pas une dynamique
  longue ni un benchmark publie.

---

## 8. Reproduire (justifie 14 de la checklist : commande + cout mesure)

```bash
cd /private/tmp/adc_cases-deeptut/magnetic_isothermal_dsl
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 run.py            # le cas : asserts, ~2 s
PYTHONPATH=/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp/build-master/python:/private/tmp/adc_cases-deeptut \
  /opt/homebrew/anaconda3/bin/python3.12 make_figures.py   # 2 figures + provenance.json, ~19 s
```

Prerequis : `numpy`, **compilateur C++20** (`needs = ["cxx"]` : le DSL compile une `.so` a la volee),
les en-tetes du coeur adc_cpp accessibles (`$ADC_INCLUDE` sinon defaut), et `matplotlib` pour les
figures. Le module `adc` doit etre importe avec le MEME interpreteur que celui qui l'a compile
(suffixe ABI `cpython-312`). Le premier chemin du `PYTHONPATH` fournit le module C++, le second rend
`adc_cases` importable (le cas a aussi un fallback `sys.path`, `run.py:63-67`).

Sortie attendue de `run.py` (capturee, macOS arm64) :

```
backend 'production' indisponible (RuntimeError), essai suivant
backends DSL lies : 'aot'
parite inter-backend SAUTEE (un seul backend lie sur cette plateforme : 'aot') ; ...
oracle Lorentz ['aot'] : err_x = 0.000e+00, err_y = 0.000e+00, max|dR| = 6.299e-01
apres 40 pas (backend 'aot') : t = 0.382939, derive de masse = 2.887e-15
qte de mvt transverse moyenne : initiale 0.000e+00 -> finale -2.080e-01 (rotation de Lorentz)
OK magnetic_isothermal_dsl (Lorentz exerce, B_z = 2.0 pilote depuis Python, backends 'aot')
```

Cout : `run.py` ~2 s (compile 4 `.so` : production+aot pour $B_z\!=\!B_0$ et $B_z\!=\!0$, puis 40 pas
$32^2$) ; `make_figures.py` ~19 s (recompile les `.so` et avance la trajectoire). **Caveat
plateforme** : les egalites exactes (`err_x = err_y = 0`, `dR[0]=0`), les signes et l'ordre de
grandeur ($\max|dR|\sim0.63$, angle $\sim-44^\circ$) sont stables ; le backend reellement lie depend
de la coherence ABI des en-tetes du module (`aot` si pre-construit, `production`+parite si rebati) ;
les derniers chiffres de $t$ et de la masse varient avec la BLAS et l'ordre de sommation (cf.
`figures/provenance.json`).

## Carte des fichiers

| Fichier | Role |
|---|---|
| `run.py` | le cas : modele DSL magnetise, oracle Lorentz analytique, parite inter-backend (si 2), invariants masse/rotation |
| `make_figures.py` | rejoue la physique ; ecrit `lorentz_oracle.png`, `cyclotron_trajectory.png` + `provenance.json` |
| `figures/*.png` | assets versionnes, regeneres en place |
| `figures/provenance.json` | SHA adc_cpp/adc_cases, backend lie, resolution, nombres mesures (err, $\max|dR|$, angle, derives) |
| `../adc_cpp/include/adc/physics/source.hpp` | briques `PotentialForce` + `MagneticLorentzForce` (conventions reproduites par le DSL) |
| `../adc_cpp/include/adc/physics/hyperbolic.hpp` | brique `IsothermalFlux` (flux + valeurs propres reproduits) |
| `../adc_cpp/include/adc/physics/elliptic.hpp` | brique `ChargeDensity` ($f=q\rho$) |
| `../hoffart_euler_poisson_dsl/` | systeme Euler-Poisson magnetise complet (physique de reference, non re-derivee ici) |
| `../diocotron_dsl/` | demonstrateur DSL avec parite DSL-vs-natif (que ce cas n'a pas) |
