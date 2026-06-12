#!/usr/bin/env bash
# =============================================================================
# campaign_figures_n512.sbatch -- figures haute resolution du rollup REEL (ADC-79).
#
# But : dumps npz du modele complet system-schur (build_real, minmod) a n=512
# jusqu'a t_f = 10 T_d, UN MODE PAR JOB (le solveur est quasi sequentiel --
# scaling threads ~1.6x a 24 ; le gain ROMEO est le parallelisme ENTRE jobs).
# Le RENDU (schlieren/GIF) se fait en local sur les npz rapatries : aucun
# matplotlib ici (diag/run_real_npz.py, sans rendu).
#
#   sbatch campaign_figures_n512.sbatch <mode>          # mode = 3 | 4 | 5
#   sbatch --export=ALL,N=768 campaign_figures_n512.sbatch 3   # surcharge n
#
# PREREQUIS (une fois, sur le LOGIN x64 -- meme archi que les noeuds x64cpu) :
#   1. module _adc x86_64 construit (ninja _adc dans <adc_cpp>/build-pyomp) ;
#   2. .so DSL precompile :  python3 diag/run_real_npz.py --mode 3 --precompile-only
#      (les jobs reutilisent le cache out_dsl/, pas de g++ concurrent par job).
# Compte / projet ROMEO : rmdraux / r250127. Cf. skill romeo-hpc + ADC-79.
# =============================================================================
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --job-name=hoffart_n512
#SBATCH --error=hoffart_n512.%J.err
#SBATCH --output=hoffart_n512.%J.out

set -euo pipefail

MODE="${1:?usage: sbatch campaign_figures_n512.sbatch <mode 3|4|5>}"
: "${N:=512}"
: "${LIMITER:=minmod}"

# --- Environnement x86_64 (AMD EPYC) -----------------------------------------
romeo_load_x64cpu_env
# numpy spack construit contre LE python du module (_adc cpython-38 <- python@3.8.19 /zcpokjg).
# Hash py-numpy@1.24.4^python@3.8.19 sur l'install 2025 ; re-resoudre via
# `spack find -dl py-numpy` si l'install change.
spack load /g3c5rqu
# Interpreteur = CELUI qui a compile le module _adc (suffixe ABI cpython-3XY) :
# build-pyomp est configure contre le python-venv spack ci-dessous ; surcharger
# ADC_PYTHON si le build change.
: "${ADC_PYTHON:=$HOME/adc_cpp/build-pyomp/.adc_python}"
if [[ -f "${ADC_PYTHON}" ]]; then ADC_PYTHON="$(cat "${ADC_PYTHON}")"; fi
# libstdc++ du gcc qui a compile le module (gcc-14 : GLIBCXX_3.4.31). numpy (compile gcc-11)
# s'importe AVANT _adc et charge SA libstdc++ dans le process -> _adc la reutilise et echoue
# (GLIBCXX_3.4.31 absent). LD_PRELOAD de la libstdc++ gcc-14 (retro-compatible) regle l'ordre.
GXX_LIB64="$(dirname "$(dirname "$(grep '^CMAKE_CXX_COMPILER:' "$HOME/adc_cpp/build-pyomp/CMakeCache.txt" | head -1 | cut -d= -f2 | tr -d '[:space:]')")")/lib64"
export LD_PRELOAD="${GXX_LIB64}/libstdc++.so.6:${LD_PRELOAD:-}"

: "${ADC_PY_BUILD:=$HOME/adc_cpp/build-pyomp/python}"
: "${ADC_CASES_ROOT:=$(cd "${SLURM_SUBMIT_DIR:-$PWD}/../.." && pwd)}"
CASE="${ADC_CASES_ROOT}/hoffart_euler_poisson_dsl"
export PYTHONPATH="${ADC_PY_BUILD}:${ADC_CASES_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ADC_DSL_CACHE="${CASE}/out_dsl"    # cache .so DSL precompile sur le login

"${ADC_PYTHON}" -c "import adc; print('adc charge depuis', getattr(adc, '__file__', '?'))"

# --- Scratch (le + rapide ; pas de sauvegarde -> rapatrier les npz ensuite) --
WORKDIR="/scratch_p/${USER}/${SLURM_JOBID}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

srun "${ADC_PYTHON}" "${CASE}/diag/run_real_npz.py" \
  --mode "${MODE}" --n "${N}" --limiter "${LIMITER}" \
  --out "${WORKDIR}/npz_l${MODE}_n${N}"

echo "[campaign_figures] npz dans ${WORKDIR}/npz_l${MODE}_n${N} (rapatrier puis rendre en local)"
