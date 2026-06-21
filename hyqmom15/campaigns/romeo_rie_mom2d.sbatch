#!/bin/bash
#SBATCH --job-name=hyqmom15_campaign
#SBATCH --partition=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=hyqmom15_campaign_%j.out

# ROMEO campaign over the five RieMOM2D_Electrostatic_periodic cases (ADC-376).
# Submit AFTER the _adc x64cpu module is built and installed (see the adc_cpp
# ROMEO build recipe). The Octave/Matlab speedup baseline is produced separately
# (octave_matlab.py) and passed via --matlab-times.
#
#   sbatch hyqmom15/campaigns/romeo_rie_mom2d.sbatch
#
# Override the defaults via environment variables:
#   ADC_CASES=/path/to/adc_cases  PYTHONPATH=/path/to/_adc  OUT=/scratch/.../campaign
set -e

ADC_CASES="${ADC_CASES:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-$ADC_CASES/out/hyqmom15_campaign}"
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
MATLAB_TIMES="${MATLAB_TIMES:-}"

cd "$ADC_CASES"
export OMP_NUM_THREADS="$THREADS"

ARGS=(--full --out "$OUT" --threads "$THREADS")
if [ -n "$MATLAB_TIMES" ]; then
  ARGS+=(--matlab-times "$MATLAB_TIMES")
fi

python3 hyqmom15/campaigns/romeo_rie_mom2d.py "${ARGS[@]}"
