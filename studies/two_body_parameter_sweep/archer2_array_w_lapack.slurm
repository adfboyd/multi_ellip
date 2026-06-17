#!/bin/bash --login

#SBATCH --job-name=me_2body
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

#SBATCH --account=e643
#SBATCH --partition=standard
#SBATCH --qos=standard

set -euo pipefail

export PROJ=${PROJ:-e643}
export WORK=${WORK:-/work/${PROJ}/${PROJ}/shared}
export WORK_A=${WORK_A:-/work/${PROJ}/${PROJ}/adfboyd}
export ME_DIR=${ME_DIR:-${WORK_A}/multi_ellip}
export CARGO_HOME=${CARGO_HOME:-${WORK}/rust/}
export RUSTUP_HOME=${RUSTUP_HOME:-${WORK}/rust/}
export PATH=${WORK}/rust/bin:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-128}}

module load PrgEnv-gnu
module load cray-libsci

MANIFEST=${MANIFEST:-${ME_DIR}/studies/two_body_parameter_sweep/two_body_parameter_sweep_manifest.csv}
BINARY=${BINARY:-${ME_DIR}/target/release/multi_ellip}
RERUN=${RERUN:-0}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted as a SLURM array job." >&2
    exit 2
fi

cd "${ME_DIR}"

row=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR == idx + 1 { print; exit }' "${MANIFEST}")
if [[ -z "${row}" ]]; then
    echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

IFS=',' read -r name shape_name rho energy_ratio separation repeat ndiv dt tend input_rel output_rel <<< "${row}"

input_path="${ME_DIR}/${input_rel}"
output_path="${ME_DIR}/${output_rel}"
output_dir=$(dirname "${output_path}")
run_log="${output_dir}/run.log"
complete_file="${output_dir}/multiple_body_complete.dat"

mkdir -p "${output_dir}"
lfs setstripe -c -1 "${output_dir}" 2>/dev/null || true

expected_rows=$(python3 - <<PY
dt = float("${dt}")
tend = float("${tend}")
print(round(tend / dt) + 2)
PY
)

if [[ "${RERUN}" != "1" && -f "${complete_file}" ]]; then
    actual_rows=$(wc -l < "${complete_file}")
    if [[ "${actual_rows}" -ge "${expected_rows}" ]]; then
        echo "Skipping ${name}: existing output has ${actual_rows}/${expected_rows} rows."
        exit 0
    fi
fi

echo "ARCHER2 two-body sweep task"
echo "  task:       ${SLURM_ARRAY_TASK_ID}"
echo "  name:       ${name}"
echo "  input:      ${input_path}"
echo "  output_dir: ${output_dir}"
echo "  binary:     ${BINARY}"
echo "  rerun:      ${RERUN}"
echo "  rayon:      ${RAYON_NUM_THREADS}"
echo

"${BINARY}" "${input_path}" "${output_dir}/" > "${run_log}" 2>&1
