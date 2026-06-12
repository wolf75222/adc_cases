# tests/ — garde-fous du cas

Tests **build-free** (ils installent un faux module `adc` minimal) qui verrouillent les contrats
d'assemblage et de signe **sans lancer la simulation lourde**. Chacun ajoute la racine du cas
(`..`) au `sys.path` pour importer `model` / `run` / `run_polar`.

Lancement (depuis la racine du dépôt, comme la CI) :
`PYTHONPATH=<adc_cpp>/build/python python3 hoffart_euler_poisson_dsl/tests/<test>.py`

| Fichier | Catégorie | Vérifie |
|---|---|---|
| `test_polar_assembly.py` | **validation (CI)** | L'ordre des appels façade du chemin polaire (`run_polar.py`) : Poisson polaire/Dirichlet → champ magnétique **avant** l'étage Schur → WENO5+Rusanov+SSPRK3+CondensedSchur → densité top-hat annulaire → équilibre rotatif stationnaire. 16 assertions. |
| `test_signs.py` | garde-fou | Les conventions de signe du modèle DSL (force électrique/Lorentz, RHS de Poisson `-alpha*rho`). 6 assertions. |
| `test_geometry_flag.py` | garde-fou | Le flag `--geometry {square,staircase}` de `run.py` (chemin AMR-IMEX ; nécessite un build AMR, donc hors CI légère). |
