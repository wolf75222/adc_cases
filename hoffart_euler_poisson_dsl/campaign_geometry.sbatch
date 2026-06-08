#!/usr/bin/env bash
# =============================================================================
# campaign_geometry.sbatch -- discriminant GEOMETRIE pour le cas hoffart.
#
# But : mesurer combien le SEUL masque de domaine disque en escalier (contrat T2
# #216, deja merge sur adc_cpp master) recupere du taux de croissance diocotron,
# AVANT la geometrie cut-cell complete. On lance le chemin de reference complet
# (system-schur, WENO5-Z + SSPRK3 + CondensedSchur theta=0.5) pour l = 3, 4, 5 a
# n = 256 et t_end = 10, dans DEUX geometries :
#
#   square     : transport FV sur le carre cartesien complet (cote L = 2R) ;
#                comportement historique, baseline bit-identique.
#   staircase  : set_disc_domain(L/2, L/2, R) materialise le masque escalier ;
#                le transport FV est confine au disque via le masque T2.
#
# Chaque geometrie ecrit son propre growth_rates.csv (sous-dossiers de sortie
# distincts), donc on peut comparer gamma(l) square vs staircase directement.
#
# AVERTISSEMENT (followup integrateur) : sur adc_cpp master HEAD ~24d01f8,
# set_disc_domain CONSTRUIT le masque mais ne le BRANCHE PAS dans System::step().
# Le transport passe par assemble_rhs<> (block_builder.hpp:55,104), pas par
# assemble_rhs_masked<> (spatial_operator.hpp:621, appele seulement par les tests).
# Tant que ce routage moteur n'est pas cable, la run 'staircase' POSE le contrat
# geometrie cote cas mais reste numeriquement identique a 'square'. Une fois le
# routage en place cote adc_cpp, ce meme sbatch mesure le gain SANS modification.
#
# Cible system-schur : mono-rang, backend Kokkos GH200 -> armgpu, 1 GPU.
# Compte / projet ROMEO : rmdraux / r250127. Cf. skill romeo-hpc.
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
#SBATCH --job-name=hoffart_geom
#SBATCH --error=hoffart_geom.%J.err
#SBATCH --output=hoffart_geom.%J.out

set -euo pipefail

# --- Environnement ARM/GPU ---------------------------------------------------
romeo_load_armgpu_env
# Adapter la version a l'interpreteur ayant compile le module adc (suffixe ABI
# cpython-3XY). Pinner ici la MEME version que le build adc_cpp.
spack load python@3.11.9

# --- Localisation du module adc et du depot adc_cases ------------------------
# ADC_PY_BUILD : dossier contenant le paquet adc/ (typiquement <adc_cpp>/build-py/python),
# construit pour aarch64 dans un job armgpu (cmake casse sur les noeuds de calcul ; voir
# reference adc_cpp ROMEO). ADC_CASES_ROOT : racine du depot adc_cases (ce fichier vit
# dans adc_cases/hoffart_euler_poisson_dsl/). On laisse l'integrateur surcharger via
# l'environnement ; sinon on devine depuis le repertoire de soumission.
: "${ADC_PY_BUILD:=$HOME/adc_cpp/build-py/python}"
: "${ADC_CASES_ROOT:=$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." && pwd)}"
export PYTHONPATH="${ADC_PY_BUILD}:${ADC_CASES_ROOT}:${PYTHONPATH:-}"

# Verifie tot que le module charge (echec lisible plutot que panne opaque).
python3 -c "import adc; print('adc charge depuis', getattr(adc, '__file__', '?'))"

# --- Espace de travail scratch (le + rapide ; pas de sauvegarde) -------------
WORKDIR="/scratch_p/${USER}/${SLURM_JOBID}"
mkdir -p "${WORKDIR}"
# Toutes les sorties des deux geometries sous ce scratch (chaque geometrie cree
# son propre sous-dossier hoffart_euler_poisson_dsl_system_schur[_staircase]).
export ADC_CASES_OUT="${WORKDIR}/out"
mkdir -p "${ADC_CASES_OUT}"

RUN="${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl/run.py"

# --- Parametres communs aux deux geometries ----------------------------------
COMMON_ARGS=(
  --engine system-schur
  --modes 3 4 5
  --n 256
  --t-end 10
  --dt 1e-3
  --no-gif
)

# --- Baseline carre puis disque escalier -------------------------------------
for GEOM in square staircase; do
  echo "==============================================================="
  echo "[campaign_geometry] geometrie=${GEOM} : l=3,4,5 n=256 t_end=10"
  echo "==============================================================="
  srun python3 "${RUN}" "${COMMON_ARGS[@]}" --geometry "${GEOM}"
done

# --- Recapitulatif : les deux growth_rates.csv cote a cote -------------------
echo "==============================================================="
echo "[campaign_geometry] growth_rates par geometrie :"
SQUARE_CSV="${ADC_CASES_OUT}/hoffart_euler_poisson_dsl_system_schur/growth_rates.csv"
STAIR_CSV="${ADC_CASES_OUT}/hoffart_euler_poisson_dsl_system_schur_staircase/growth_rates.csv"
for label in "square:${SQUARE_CSV}" "staircase:${STAIR_CSV}"; do
  name="${label%%:*}"; path="${label#*:}"
  echo "--- ${name} (${path}) ---"
  if [ -f "${path}" ]; then
    cat "${path}"
  else
    echo "(absent : la run ${name} a echoue ou n'a pas ecrit de growth_rates.csv)"
  fi
done

echo "[campaign_geometry] sorties completes sous ${ADC_CASES_OUT}"
echo "[campaign_geometry] PENSER a recopier ${ADC_CASES_OUT} hors scratch (aucune sauvegarde GPFS)."
