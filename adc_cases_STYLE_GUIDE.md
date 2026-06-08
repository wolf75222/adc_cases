# Guide de style : tutoriels de cas adc_cases

## 1. Principe directeur

On vise un README qui rend un cas reproductible et compris par un lecteur sans connaissance prealable : il doit pouvoir relancer le cas, lire le code reel ligne par ligne, suivre les derivations math sans trou, comprendre le mecanisme physique, et savoir exactement ce qui est prouve et ce qui ne l'est pas. On refuse trois choses : le bullshit promotionnel (mots-creux, ton commercial), la sur-vente (presenter une validation comme une reproduction, sur-lire une figure de diagnostic), et la redondance (re-dire au chapitre 12 ce que disait le chapitre 1). Chaque phrase porte un fait, un nombre, un symbole, un signe ou une raison verifiable ; sinon elle saute. L'honnetete sur les limites n'est pas une concession, c'est le coeur du contrat : la clause `Ne prouve pas` est aussi detaillee que les garanties positives.

## 2. Structure canonique d'un README de cas

Le README s'ouvre toujours par le bloc Contrat (section 2.0), puis enchaine les sections de fond. Chaque section de fond doit nommer la clause du contrat qu'elle justifie ; une section qui ne justifie aucune clause est coupee.

### 2.0 Bloc Contrat (obligatoire, en tete, tous types)

Un tableau dense, le plus dur a remplir de bullshit. Colonnes / lignes :

| Champ | Contenu |
|---|---|
| Categorie (manifeste) | la valeur exacte de `cases_manifest.toml` (`validation` / `tutoriel` / `reproduction` / `reproduction-candidate` / `experimental`). Elle dicte le ton de Ne prouve pas (voir 2.x). |
| Entrees | grille (n, L, periodique ?), CI, parametres physiques avec valeurs et unites (ou "sans unites", ex. `four_pi_G=1`). |
| Sorties | etat(s) lus, diagnostics calcules, figures/CSV produits et leur emplacement. |
| Invariants garantis | la liste des `assert` reels du `run.py`, chacun avec sa tolerance. |
| Prouve | ce qu'un `assert` etablit reellement (ex. signe de dE, masse conservee a 1e-9, egalite bit). |
| Ne prouve pas | aussi detaille que Prouve : nommer le proxy, le regime, ce qui n'est pas teste (voir 2.x). |
| Provenance | SHA adc_cpp + adc_cases, backend, resolution, cout mesure, plateforme. |

La regle de couplage : chaque section en aval ecrit en clair "justifie la clause X du contrat". Si une clause Prouve/Ne prouve pas n'est justifiee par aucune section, soit on ecrit la section, soit on retire la clause.

### 2.x Le ton de Ne prouve pas est dicte par la categorie

- `reproduction` (ex. `diocotron/run.py`) : dire ce qui est reproduit (oracle analytique == papier a 3 chiffres) et ce qui ne l'est pas (taux FV sous-estime de -22 a -27 %). Pas de pending : la reproduction est etablie pour la partie annoncee.
- `reproduction-candidate` (ex. `hoffart_euler_poisson_dsl/run.py`) : doit ecrire pending. La table de validation est explicitement non etablie ; interdiction de presenter comme reproduit.
- `validation` (ex. `euler_poisson/run.py`) : doit dire "ce n'est pas une reproduction publiee". On verifie des invariants, pas une courbe du papier.
- `tutoriel` (ex. `composition/run.py`) : la clause dit "demontre une capacite d'API, ne valide aucun resultat physique publie".
- `experimental` (ex. `schur_magnetized_cartesian/run.py`, `dsl_euler/run.py`) : doit signaler prototype / chemin non finalise (DSL interprete, mesure de timing dependante plateforme).

### Sections de fond, par type de cas

L'ossature commune (apres le contrat) :

1. Physique : le mecanisme (section 5).
2. Equations resolues + table des briques (3 couches, section 3).
3. Maths / derivation de la prediction (section 4).
4. Code, fonction par fonction, ancre lignes (section 3).
5. Conditions initiales.
6. Figures, generees et analysees (section 6).
7. Analyse honnete des ecarts / limites.
8. Provenance, commande exacte, cout (re-detaille).

La prediction falsifiable est enoncee des le contrat (section 0) et l'artefact qui la confronte est produit en section 6. La nature de la prediction glisse selon le type, et l'ossature s'adapte :

- Reproduction physique (`diocotron`, `hoffart_*`, `plasma`, `euler_poisson` cote contraste) : prediction = un nombre physique (taux de croissance, signe et magnitude de dE). Garder toutes les sections ; la section 7 (analyse de l'ecart au papier/a l'analytique) est le coeur.

- Validation d'invariant (`euler_poisson`, `multispecies`, `two_euler`, `plasma`, `diocotron_amr`) : prediction = un invariant structurel (masse conservee, impulsion nette nulle, signes opposes). La derivation math (4) explique pourquoi l'invariant tient (potentiel periodique -> somme de force nulle). Section physique allegee, section "pourquoi cet invariant" renforcee. La section 7 devient "ce que l'invariant ne capture pas".

- Equivalence DSL-vs-natif (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`) : prediction = egalite de chemins bit-a-bit (`np.array_equal`). ne jamais reproduire la physique deja derivee dans le cas-parent : lier (`../diocotron/`), pas copier. Les sections 1/4/5 se reduisent a un renvoi ; le coeur devient "quelles conventions du coeur sont reproduites" (table ExBVelocity / BackgroundDensity ancree `include/adc/physics/*.hpp`) et "comment l'egalite bit est verifiee et ce qu'une divergence trahirait".

- Etude de timing (`schur_magnetized_cartesian`) : prediction = un facteur de gain (pas stable Schur / pas stable explicite) et la borne `dt*omega_c`. Le coeur est la methodologie de mesure (`largest_stable_dt`, balayage geometrique, critere de stabilite densite finie/bornee/positive) et les caveats plateforme (backend AOT, `set_source_stage` au lieu de `adc.Split`). Pas de figure physique : un tableau methode/dt_stable/gain.

- Prototype (`dsl_euler`, chemins `experimental`) : prediction = "le chemin declaratif produit un etat fini et coherent", pas un nombre cible. Dire franchement que c'est un prototype interprete, non en CI, et ce qui manque pour le promouvoir.

## 3. Comment traiter le code

Regles dures :

- Ancrage reel : citer `run.py:NN` (intervalle de lignes) et le nom exact (`run_case`, `diocotron_eigenvalue`, `largest_stable_dt`, `assert_opposite_sign`, `TOL_DE`). Une affirmation theorique sans ligne qui l'implemente est coupee. Aucune ligne non triviale sans son justificatif.
- Ne jamais paraphraser une ligne triviale (`import numpy as np`, `sys.path.insert`). On glose uniquement les lignes porteuses de physique ou d'algorithme.
- Inline vs lien : on montre inline les fonctions physique-cle (le flux, les valeurs propres, l'assemblage de l'operateur, la mesure du diagnostic) en blocs de 5 a 15 lignes du `run.py` reel, suivis de puces qui expliquent chaque variable non triviale (`rho`, `Om`, `Lmat`, `Q`, `dE_grav`). On lie (sans copier) la plomberie : le bloc try/except import `adc_cases`, la machinerie de fallback backend, l'argparse.
- Granularite selon le role : une fonction physique-cle (ex. `mode_l_amplitude`, `diocotron_eigenvalue`, `magnetized_model`) merite le commentaire ligne par ligne ; une fonction de plomberie (ex. `make_system` qui ne fait que `adc.System(...)`) merite une phrase.
- Table 3 couches "qui calcule quoi", obligatoire pour les cas a briques. Trois lignes, chacune pinnee a une ligne reelle de `run.py` :

| Ligne run.py | Couche | Ce qui se passe |
|---|---|---|
| `add_block(...)` / `add_equation(...)` | Python compose et diagnostique | choix du modele, du schema, de l'integrateur ; lecture de l'etat |
| `models.euler_poisson(...)` / brique `ExBVelocity` / `BackgroundDensity` | brique C++ compilee | le choix physique fige (flux, valeurs propres, RHS elliptique) |
| `assemble_rhs<Limiter,Flux>`, Newton local, Poisson de systeme | noyau par cellule (device) | le calcul reel, sans callback Python dans le hot path |

Pour un cas DSL, la couche du milieu n'est plus une brique nommee mais les expressions (`m.flux(...)`, `m.eigenvalues(...)`, `m.elliptic_rhs(...)`) que `adc.dsl` compile ; ancrer la table sur ces appels.

## 4. Comment traiter les maths

- Deriver, ne pas assener : pour une prediction falsifiable, on montre les etapes. Exemple obligatoire pour `diocotron` : passer de la linearisation $\phi'=\hat\phi(r)e^{i(m\theta-\omega t)}$ au probleme aux valeurs propres $\omega\mathcal{L}_m\hat\phi=(m\Omega\mathcal{L}_m+Q)\hat\phi$, puis a la forme standard $\omega\hat\phi=\mathcal{L}_m^{-1}(\dots)\hat\phi=M\hat\phi$, et dire que `eigvals(M)` rend le spectre. Chaque symbole de la formule pointe la ligne qui le calcule (`Om` = $\Omega(r)$ ligne 110, `Q` = $\frac{m}{r}\frac{dn_0}{dr}$ ligne 134, `Lmat` = $\mathcal{L}_m$ lignes 120-125).
- Admettre proprement : ce qui n'est pas re-derivable en quelques lignes (convention exacte du papier, normalisation $\times 2\pi/\bar\rho$) est cite avec sa source, pas reconstruit a la main.
- Notation : LaTeX GitHub, `$...$` en ligne et `$$...$$` en bloc. Accents francais OK. pas d'em-dash (U+2014) ; utiliser deux-points, parentheses ou points.
- La prediction quantitative falsifiable est privilegiee. Pour `euler_poisson`, la vraie prediction testable de la linearisation est $|dE|\propto\epsilon^2$ : un graphe log-log de $|dE|$ vs $\epsilon$ doit avoir une pente 2 ; doubler $\epsilon$ quadruple $|dE|$. C'est verifiable et transforme un assert booleen en courbe de convergence. L'enoncer, et dire ce qu'une pente differente trahirait (pente ~1 = terme lineaire parasite, fond `rho0` mal soustrait ; pente > 2 aux grands $\epsilon$ = entree non lineaire).
- Verifier le signe par le comportement, jamais par une convention de manuel plaquee. Le solveur Poisson (`poisson_operator.hpp`) a plusieurs couches de signe plus un `GradSign` en post-traitement. Ecrire "$-\nabla^2\phi=+4\pi G(\rho-\rho_0)$ donc gravite attractive" sans verifier est faux (peut donner une repulsion). Le signe physique se lit sur l'assert qui passe : pour `euler_poisson`, `run.py:177-180` impose `dE_grav < 0` (attractif) et `dE_plas > 0` (repulsif) ; c'est CA la reference, pas une formule de cours.
- Nommer les paradoxes, ne pas fabriquer la derivation. Pour `euler_poisson`, $E_{tot}=U[3].sum()$ est l'energie fluide seule (pas de potentiel de champ) et elle diminue pour la gravite meme si $v\cdot g>0$. Enoncer la tension ouvertement et l'attribuer a la convention de couplage, sans manufacturer un theoreme encadre. Un signe encadre faux est pire qu'un report honnete.

## 5. Comment traiter la physique

- Le mecanisme avant le resultat. Pour `diocotron` : la rotation differentielle $\Omega(r)=-\frac{1}{r^2}\int_0^r n_e r'dr'$ cree un cisaillement, le cisaillement est une instabilite de Kelvin-Helmholtz d'un anneau de vorticite ($n_e$ joue la vorticite, $\phi$ la fonction de courant), donc l'anneau developpe $l$ lobes qui s'enroulent. Le taux $\gamma_l$ vient après, comme consequence quantifiee.
- Relier le modele reduit au modele complet, explicitement. `diocotron` resout la limite de derive E x B ; le systeme Euler-Poisson magnetise complet est `hoffart_euler_poisson_dsl`. Dire "ce cas ne reproduit que la limite de derive, pas le systeme complet", avec le lien.
- Honnetete sur ce qui est modelise : nommer les simplifications (une seule variable conservee, pas de quantite de mouvement ni d'energie pour le diocotron ; `four_pi_G=1` sans unites ; regime quasi-lineaire $\epsilon=0.01$, 20 pas, pas d'effondrement de Jeans pour `euler_poisson`).

## 6. figures : quelles figures, comment les generer, comment les analyser

### Tableau par type de cas

| Type | Figures de diagnostic a generer | Ce qu'on y lit |
|---|---|---|
| Reproduction physique (taux) | `dispersion.png` (gamma vs mode : analytique + points papier + mesures adc) ; `amplitude.png` (semilog $|c_l|(t)$, droite = exponentielle) ; `snapshots.png` + `*.gif` (enroulement non lineaire) | classement des modes, ecart mesure/analytique, signature visuelle a $l$ lobes |
| Validation d'invariant | conservation vs t (masse, impulsion en echelle absolue) ; contraste energetique (dE des deux runs cote a cote) ; convergence $|dE|$ vs $\epsilon$ en log-log (pente attendue 2) ; carte 2D de la perturbation | l'invariant tient a la tolerance ; le signe est franc et au-dessus du bruit ; la pente confirme le regime |
| Equivalence DSL-vs-natif | heatmap $|state_{dsl}-state_{natif}|$ qui doit etre identiquement noire ; histogramme du residu plafonnant a ~1e-15 (ou exactement 0) | un seul pixel non noir = echec ; le residu au niveau machine est l'observable qui prouve le determinisme |
| Multi-especes / couple | masses par espece vs t (chacune plate) ; carte de densite par espece ; potentiel couple $|\phi|$ | conservation par espece, couplage Poisson actif |
| Uniforme vs AMR (`diocotron_amr`) | comparaison cote a cote uniforme/AMR du meme diagnostic ; carte des patches | le reflux conservatif preserve l'invariant ; l'AMR suit la meme dynamique |
| Timing (`schur`) | dt_stable vs methode (barres ou tableau) ; dt*omega_c ; gain | la source explicite s'effondre quand omega_c grandit ; le Schur leve la borne |

### Convention de generation

- Assets de reproduction versionnes : seuls les cas `reproduction` committent leurs figures dans `<cas>/figures/` avec un `figures/provenance.json` (champs reels : `adc_cpp_sha`, `adc_cases_sha`, `backend`, `resolution`, `nsteps_growth`, `cfl`, `python`, et les nombres mesures comme `gamma_num_mesure`).
- Diagnostics transitoires : tout le reste ecrit sous `out/<cas>/` via `case_output_dir(<cas>)` (cf. `adc_cases/common/io.py`), repertoire git-ignore. Ne jamais ecrire un diagnostic jetable dans l'arbre source.
- Workflow d'un cas sans figure aujourd'hui : dire explicitement quelle figure generer (run + plot + commit), avec la commande exacte et l'emplacement (`out/<cas>/` pour explorer, `<cas>/figures/` seulement si le cas devient une reproduction versionnee).

### Regles d'analyse

- Chaque figure est embed (`![alt parlant](figures/xxx.png)`) puis suivie de 2 a 4 phrases qui interpretent ce qu'elle montre physiquement. Jamais une legende creuse ("voici la densite").
- Partitionner la lecture en Prouve / Suggéré / Non montré :
  - Prouve : ce qu'un assert teste (signes opposes de dE, masse plate, egalite bit).
  - Suggéré (non assere) : ce qui est plausible a l'oeil mais non teste (ex. la symetrie miroir ~5 % gravite/plasma est visible mais aucun assert ne la verifie ; le dire comme suggestion).
  - Non montré : ce que la figure ne couvre pas (pas de dynamique non lineaire sur 20 pas ; pas d'effondrement de Jeans).
- Lecture diagnostique, pas decorative. Une pente != 2 sur $|dE|$ vs $\epsilon$ trahit (pente ~1 = lineaire parasite, pente > 2 = non lineaire). Sur une heatmap d'equivalence, dire ce qu'une tache non noire signalerait (une formule DSL qui diverge d'une brique du coeur, ex. mauvaise convention de signe dans `eigenvalues` ou `elliptic_rhs`).
- Provenance sur chaque nombre cite : SHA, backend, resolution, cout mesuré (pas estime), plus le caveat plateforme : les signes et l'ordre de grandeur sont stables, les derniers chiffres varient avec la bibliotheque BLAS et l'ordre de sommation. Citer les vrais nombres du run (les `gamma_num_mesure` du `provenance.json`, pas des valeurs inventees).

## 7. Anti-bullshit : regles dures

- Test de suppression : si retirer une phrase ne fait perdre ni fait, ni nombre, ni symbole, ni signe, ni raison, elle saute.
- zero recap. Pas de section "En conclusion" qui re-dit l'intro. L'exemplaire `diocotron` actuel a des sections 8 (Architecture) et 12 (Limites) qui re-disent les sections 1 a 7 ; ce budget est reclame pour la derivation et l'analyse de figures, a longueur quasi constante. Un fait, a une seule altitude.
- Une affirmation theorique sans la ligne de code qui l'implemente est coupee.
- Une tolerance est une clause justifiee par un ordre de grandeur, jamais une constante posee. `TOL_DE=1e-5` se situe entre le bruit machine (dE = 0 exactement a $\epsilon=0$) et la magnitude physique attendue ~6e-4 (`run.py:60-62`) : ecrire ce ratio. `TOL_MASS=1e-9` car le schema est conservatif et la derive vient de l'arithmetique flottante. Chaque tolerance a son "pourquoi".
- Toujours distinguer Prouve (par un assert) de Suggéré (rendu plausible par une figure). Garde-fou anti-sur-interpretation, a appliquer aux 15 cas.
- Pour les cas-enfants DSL : lier, ne pas copier la physique du parent.

Liste noire (mots/tournures interdits), avec correction :

- "puissant / seamless / leverage / robuste (decoratif) / elegant" -> supprimer ou remplacer par le fait. Avant : "adc compose de maniere puissante et elegante." Apres : "adc.System compose un bloc par `add_block`, chaque bloc fige son schema en C++ a l'ajout."
- Regle de trois decorative -> couper la triade vide. Avant : "rapide, fiable et extensible." Apres : "~60 s sur un coeur CPU (3 modes x 900 pas a $192^2$)."
- Hedging vide ("il convient de noter que", "d'une certaine maniere", "globalement") -> supprimer.
- Ton promotionnel ("ce cas met en lumiere la richesse du solveur") -> remplacer par l'enonce de ce qui est teste. Apres : "ce cas verifie par assert : masse conservee a 1e-9, impulsion nette < 1e-8, signes de dE opposes."
- Sur-vente de categorie. Avant : "reproduction du benchmark Hoffart." Apres (si `reproduction-candidate`) : "vise arXiv:2510.11808 ; reproduction quantitative pending (table de validation non etablie)."
- Adverbe d'emphase vide ("clairement", "evidemment", "notamment" en ouverture) -> supprimer.

## 8. Longueur et densite

Cible par type (texte hors blocs de code) :

- Reproduction physique : 350 a 550 lignes. C'est le plus long ; la derivation et l'analyse des 3 a 4 figures le justifient.
- Validation d'invariant : 180 a 320 lignes. Centre sur "pourquoi l'invariant tient" et la prediction $\epsilon^2$.
- Equivalence DSL-vs-natif : 120 a 220 lignes. Court par construction : la physique est liee au parent, le coeur est la table de conventions et l'egalite bit.
- Timing : 150 a 250 lignes. La methodologie de mesure et les caveats plateforme dominent.
- Prototype : 100 a 180 lignes. On dit ce que c'est, ce qui manque, on ne sur-construit pas.

Test de densite : aucune phrase n'est supprimable sans perte de fait, nombre, symbole, signe ou raison. Si une section depasse la cible, c'est presque toujours un recap a couper, pas du fond a ajouter.

## 9. Checklist de validation d'un README

Binaire, a cocher avant acceptation :

1. [ ] Le bloc Contrat est en tete, avec la categorie exacte du manifeste.
2. [ ] La clause Ne prouve pas est aussi detaillee que Prouve et son ton suit la categorie (pending si `reproduction-candidate`, "pas une repro publiee" si `validation`, prototype si `experimental`).
3. [ ] Chaque section de fond nomme la clause du contrat qu'elle justifie ; aucune clause orpheline.
4. [ ] Une prediction falsifiable est enoncee dans le contrat et un artefact (figure/assert/tableau) la confronte.
5. [ ] Chaque affirmation theorique pointe une ligne reelle (`run.py:NN`) et un nom reel ; aucune ligne triviale n'est paraphrasee.
6. [ ] La table 3 couches (Python compose / brique fige / noyau par cellule) est presente pour un cas a briques, chaque ligne pinnee a une ligne reelle.
7. [ ] Les signes physiques sont verifies par le comportement asserte, pas par une convention de manuel plaquee.
8. [ ] Chaque tolerance est justifiee par un ordre de grandeur (ratio bruit / magnitude physique).
9. [ ] Chaque figure est embed et suivie de 2 a 4 phrases d'analyse, partitionnees Prouve / Suggéré / Non montré.
10. [ ] Les diagnostics transitoires vont dans `out/<cas>/` (via `case_output_dir`) ; seules les figures de reproduction versionnees sont dans `<cas>/figures/` avec `provenance.json`.
11. [ ] Chaque nombre cite a sa provenance (SHA, backend, resolution, cout mesure) et le caveat plateforme (signes/ordre de grandeur stables, derniers chiffres variables).
12. [ ] Aucune section de recap ; aucun fait dit deux fois a deux altitudes.
13. [ ] Aucun mot de la liste noire ; le test de suppression passe sur chaque phrase.
14. [ ] La commande exacte de lancement, les prerequis et le cout mesure sont donnes.
15. [ ] Pour un cas-enfant DSL : la physique du parent est liee, pas copiee ; le coeur est la table de conventions du coeur (`include/adc/physics/*.hpp`) et l'egalite bit (`np.array_equal`).