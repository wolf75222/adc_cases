# docs/ — notes d'analyse et d'audit

Documentation d'approfondissement du cas `hoffart_euler_poisson_dsl`. Ces fichiers ne sont pas
exécutables : ils consignent le raisonnement, les audits de normalisation et les résultats
détaillés. La synthèse lisible est dans le [README du cas](../README.md).

| Fichier | Contenu |
|---|---|
| `NORMALIZATION.md` | La normalisation `2π/rhobar` du chemin polaire réduit E×B : origine du facteur 2π, validation l=4 exacte du chemin polaire réduit. |
| `T2_NORMALIZATION_AUDIT.md` | Audit dimensionnel détaillé : échelles, candidats de normalisation, décomposition `fenêtre × 2π × résidu de grille`. Clôt la question « géométrie vs métrologie » (le déficit était de la métrologie, pas la géométrie). |
| `RESULTS_SYSTEM_SCHUR.md` | Journal complet : table des taux `system-schur`, audit T2, code T3, convergence, et l'historique des renversements (déficit « −95 % » → reproduit à moins de 10 %). |
