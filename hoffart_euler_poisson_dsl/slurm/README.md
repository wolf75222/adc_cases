# slurm/ — campagnes ROMEO

Scripts batch SLURM pour les campagnes de mesure sur ROMEO (URCA, partition x64cpu). Ils pilotent
`run.py` / `run_polar.py` (à la racine du cas) ; leurs chemins internes sont **absolus**
(`${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl/...`), donc inchangés par ce déplacement.

| Fichier | Lance | But |
|---|---|---|
| `campaign_geometry.sbatch` | `run.py --geometry {square,staircase}` | Discriminant géométrie : le bord d'anneau cartésien est-il le verrou du taux mesuré ? |
| `campaign_polar.sbatch` | `run_polar.py` (frozen-equilibrium) | Chemin complet sur grille polaire (anneau résolu), l=3,4,5, nr=ntheta=256, t_end=10. |

Voir le [README du cas](../README.md), section « Performance et passage à l'échelle ».
