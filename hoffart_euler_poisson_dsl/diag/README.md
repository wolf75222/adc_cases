# diag/ — diagnostics

Scripts de diagnostic, **hors manifeste** (ce ne sont pas des cas) : ils éclairent la
normalisation, le spectre analytique, la convergence, et génèrent les figures. Lancés à la main
depuis la racine du cas.

| Fichier | Rôle |
|---|---|
| `petri_eigenvalue.py` (+ `petri_eigenvalue.md`) | Valeur propre analytique de Davidson/Petri : dérive les cibles `0.772 / 0.911 / 0.683` et l'origine du facteur 2π (numpy seul, aucune dépendance au moteur `adc`). |
| `diag_normalization_audit.py` | Audit dimensionnel exécutable (échelles, candidats de normalisation, décomposition de la fenêtre). Support de [`../docs/T2_NORMALIZATION_AUDIT.md`](../docs/T2_NORMALIZATION_AUDIT.md). |
| `diag_polar_omega.py` | Chemin polaire **réduit** E×B scalaire : valide la normalisation `2π/rhobar` (récupère l=4 exact). |
| `convergence_reduced.py` | Convergence en résolution : l'erreur relative au papier tend vers 0 quand `n` croît. |
| `make_paper_figures.py` | Générateur des figures et GIF style papier (snapshots schlieren, taux de croissance, animations du rollup). |

Voir le [README du cas](../README.md).
