#!/usr/bin/env bash
# =============================================================================
# campaign_polar.sbatch -- modele COMPLET Euler-Poisson sur GRILLE POLAIRE.
#
# But : lancer le chemin run_polar.py (modele complet isotherme magnetise +
# etage Schur condense POLAIRE) sur l'ANNEAU resolu pour l = 3, 4, 5, a
# nr = ntheta = 256, t_end = 10. La direction radiale est portee par un axe de
# grille (bords d'anneau resolus) : c'est la levee du verrou des bords d'anneau
# cartesiens (cf. docs/HOFFART_GEOMETRY_VERDICT.md, ou le chemin cartesien
# plafonne a -82/-95% INDEPENDAMMENT de la resolution).
#
# Pile numerique : IsothermalFluxPolar (#209) + WENO5-Z + Rusanov + SSPRK3 +
# PolarCondensedSchurSourceStepper (#215, theta=0.5) + Poisson polaire direct
# (FFT-theta + Thomas-r, Dirichlet phi=0 aux parois radiales). Init : top-hat
# annulaire + perturbation sin(l theta) + EQUILIBRE ROTATIF (v_r=0, v_theta(r)
# racine du bilan radial centrifuge/pression/electrique/Lorentz) du premier solve
# Poisson. cs2 = 1e-4 (petit mais NON nul) pour la stricte hyperbolicite polaire
# (le papier autorise theta >= 0).
#
# BILAN DISCRET BIEN POSE -- OPTION (c), --frozen-equilibrium (defaut ON) :
# le modele complet n'est PAS discretement bien pose a la raideur du papier
# (B_z = omega = 1e12) ; l'etage hyperbolique (WENO5 + Rusanov au saut top-hat de
# l'anneau) et l'etage Schur n'annulent le bilan electrique/Lorentz QU'AU CONTINU,
# jamais au discret -> derive parasite O(1) axisymetrique -> NaN ~ t 0.02, quel que
# soit dt (l'IC d'equilibre rotatif et la Newton #23 etaient des NO-OP : jacobienne
# rho*B_z ~ 1e12 -> correction ~ 1e-12). On PRECALCULE donc le residu d'equilibre
# GELE R_eq = step(U_eq) - U_eq UNE FOIS sur l'anneau axisymetrique (perturbation=0),
# puis on SOUSTRAIT R_eq a chaque pas (U <- step(U) - R_eq) : U_eq devient un POINT
# FIXE DISCRET EXACT, la derive O(1) est annulee a la precision machine, seule la
# perturbation O(delta) (mode physique + petit residu O(delta)) evolue. PAS FIXE
# OBLIGATOIRE (dt = 1e-3) : le --cfl adaptatif est BRISE pour ce modele (vitesse
# estimee -> 0 sur l'equilibre quasi-stationnaire -> dt ~ 1e28 -> NaN instantane).
# Observable : phi sur le cercle r=r0 (colonne native phi[:, i_r0]), FFT-theta,
# fenetres de fit VERBATIM du papier -> growth_rates.csv (pente BRUTE).
#
# MONO-RANG OBLIGATOIRE : l'etage Schur condense polaire = boite unique couvrant
# l'anneau (solveur direct) ; n_ranks>1 leve. -> 1 GPU, ntasks=1.
#
# Cible : backend Kokkos GH200 -> armgpu, 1 GPU. Compte / projet ROMEO :
# r250127. Cf. skill romeo-hpc. On REUTILISE le build aarch64 du chemin
# cartesien (build-c20 / build-py) : run_polar.py n'ajoute AUCUN symbole C++
# (briques polaires deja sur adc_cpp master), seulement du Python -- aucun
# rebuild necessaire si le module adc est deja construit avec le chemin polaire.
# =============================================================================
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=hoffart_polar
#SBATCH --error=hoffart_polar.%J.err
#SBATCH --output=hoffart_polar.%J.out

set -euo pipefail

# --- Environnement ARM/GPU ---------------------------------------------------
romeo_load_armgpu_env
# Pinner la MEME version d'interpreteur que celle ayant compile le module adc
# (suffixe ABI cpython-3XY). Adapter au build adc_cpp aarch64.
spack load python@3.11.9

# --- Localisation du module adc et du depot adc_cases ------------------------
# ADC_PY_BUILD : dossier contenant le paquet adc/ (typiquement <adc_cpp>/build-py/python,
# build aarch64 dans un job armgpu ; cmake casse sur les noeuds de calcul, voir
# reference adc_cpp ROMEO). ADC_CASES_ROOT : racine du depot adc_cases (ce fichier
# vit dans adc_cases/hoffart_euler_poisson_dsl/). Surchargeables via l'environnement.
: "${ADC_PY_BUILD:=$HOME/adc_cpp/build-py/python}"
: "${ADC_CASES_ROOT:=$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." && pwd)}"
export PYTHONPATH="${ADC_PY_BUILD}:${ADC_CASES_ROOT}:${PYTHONPATH:-}"

# Verifie tot que le module charge ET que le chemin polaire est present (echec
# lisible plutot que panne opaque a l'assemblage).
python3 -c "import adc; adc.System(mesh=adc.PolarMesh(r_min=2.0, r_max=16.0, nr=8, ntheta=8)); print('adc + PolarMesh OK depuis', getattr(adc, '__file__', '?'))"

# --- Auto-test de STATIONARITE A LA PRECISION MACHINE (frozen-eq) AVANT campagne -
# Avec --frozen-equilibrium (ON), --check-equilibrium exige que U_eq soit un POINT
# FIXE DISCRET EXACT de la carte step()-R_eq : max_n ||U^n - U_eq||_inf reste sous
# C eps_mach ||U_eq||_inf sur >= 200 pas (et NON le critere laxiste base-amplitude,
# normalise par le fond O(1), qui MASQUAIT la derive parasite O(1)). C'est la PREUVE
# que la soustraction du residu gele rend l'equilibre stationnaire ; si elle echoue,
# on s'arrete ici plutot que de gaspiller 8 h de campagne. Pas FIXE dt = 1e-3.
echo "==============================================================="
echo "[campaign_polar] auto-test stationarite PRECISION MACHINE (frozen-eq, perturbation=0)"
echo "==============================================================="
srun --ntasks=1 python3 "${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl/run_polar.py" \
  --check-equilibrium --frozen-equilibrium \
  --nr 128 --ntheta 128 --r-min 2.0 --dt 1e-3 --cs2 1e-4 \
  --max-steps-check 200 --check-modes 1 2 3 4 5

# --- Espace de travail scratch (le + rapide ; pas de sauvegarde) -------------
WORKDIR="/scratch_p/${USER}/${SLURM_JOBID}"
mkdir -p "${WORKDIR}"
export ADC_CASES_OUT="${WORKDIR}/out"
mkdir -p "${ADC_CASES_OUT}"

RUN="${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl/run_polar.py"

# --- Campagne polaire l = 3, 4, 5 sur l'anneau resolu ------------------------
# nr = ntheta = 256 (annulus n~256 sur un seul GH200). t_end = 10, dt FIXE = 1e-3.
# --frozen-equilibrium (option c) : R_eq = step(U_eq) - U_eq precalcule UNE FOIS,
# soustrait a chaque pas -> bilan discret bien pose (pas de NaN). PAS FIXE
# OBLIGATOIRE (le --cfl adaptatif est brise pour ce modele : il est d'ailleurs
# ignore en mode frozen). MONO-RANG : srun avec 1 seule tache (l'etage Schur
# polaire refuse n_ranks>1).
echo "==============================================================="
echo "[campaign_polar] modele complet POLAIRE (frozen-eq) : l=3,4,5 nr=ntheta=256 t_end=10"
echo "==============================================================="
srun --ntasks=1 python3 "${RUN}" \
  --modes 3 4 5 \
  --frozen-equilibrium \
  --nr 256 \
  --ntheta 256 \
  --r-min 2.0 \
  --t-end 10 \
  --dt 1e-3 \
  --cs2 1e-4

# --- Recapitulatif : le growth_rates.csv polaire -----------------------------
POLAR_CSV="${ADC_CASES_OUT}/hoffart_euler_poisson_dsl_polar_schur/growth_rates.csv"
echo "==============================================================="
echo "[campaign_polar] growth_rates (polaire) : ${POLAR_CSV}"
if [ -f "${POLAR_CSV}" ]; then
  cat "${POLAR_CSV}"
else
  echo "(absent : la run a echoue ou n'a pas ecrit de growth_rates.csv)"
fi

echo "[campaign_polar] sorties completes sous ${ADC_CASES_OUT}"
echo "[campaign_polar] PENSER a recopier ${ADC_CASES_OUT} hors scratch (aucune sauvegarde GPFS)."
