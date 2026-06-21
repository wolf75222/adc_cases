#!/bin/bash
#SBATCH --job-name=hyqmom15_campaign
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out

# ROMEO campaign over the five RieMOM2D_Electrostatic_periodic cases (ADC-376).
#
# PREREQUISITE: _adc rebuilt from the CURRENT adc_cpp master (the ADC-368 ROE hook
# is required for fluid_wave; the stock build-mpik may be pre-ROE). Rebuild via the
# adc_cpp ROMEO recipe (cmake -S adc_cpp -B build-mpik && make _adc -j32).
#
#   sbatch hyqmom15/campaigns/romeo_rie_mom2d.sbatch
#
# The env below mirrors the proven build recipe so backend="production" can compile
# the model .so at runtime (ADC_KOKKOS_ROOT / ADC_CXX / ADC_INCLUDE). Spack paths
# are environment-specific -- re-check them against the current ROMEO install if a
# load fails. Octave speedup + matplotlib figures run separately (see README).
set -e
source /etc/profile
romeo_load_x64cpu_env

GCC14=/apps/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/gcc-11.4.1/gcc-14.2.0-fmrcj5jrubquulzxeqmhucxqunhodrxp
LIBMD=/gpfs/app/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/aocc-4.2.0/libmd-1.1.0-w6vvjrgurkf3gtni3j7lfuou6zguo5rz/lib
PY=/apps/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/aocc-5.0.0/python-3.11.9-esusxhdmlaeiclb4vnytlldmf4ijzkvv/bin/python3.11
export LD_LIBRARY_PATH=$GCC14/lib64:$LIBMD:$LD_LIBRARY_PATH
export ADC_KOKKOS_ROOT=$HOME/kokkos-x64-pic
export ADC_INCLUDE=$HOME/adc_cpp/include
export ADC_CXX=$GCC14/bin/g++
export OMPI_CXX=$GCC14/bin/g++
export PYTHONPATH=$HOME/adc_cpp/build-mpik/python:$HOME/adc_cases

ADC_CASES="${ADC_CASES:-$HOME/adc_cases}"
OUT="${OUT:-/scratch_p/$USER/hyqmom15_campaign_$SLURM_JOBID}"
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
mkdir -p "$OUT"
cd "$ADC_CASES"
export OMP_NUM_THREADS="$THREADS"

# 1) run the 5 cases (snapshots + run_meta + realizability monitoring + synthesis)
$PY hyqmom15/campaigns/romeo_rie_mom2d.py --full --out "$OUT" --threads "$THREADS"
# 2) consolidate each case into HDF5 (skipped if h5py is absent)
$PY hyqmom15/campaigns/export_h5.py "$OUT"
# 3) per-case analysis report (config + realizability + figures + h5)
$PY hyqmom15/campaigns/make_rapport.py "$OUT"
# 4) ParaView time series per case (<case>.pvd + .vti) -- open the .pvd in ParaView
$PY hyqmom15/campaigns/to_paraview.py "$OUT"
echo "campaign + HDF5 + rapport.md + ParaView (.pvd) written under: $OUT"
