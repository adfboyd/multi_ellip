#!/bin/bash --login

#SBATCH --job-name=me_2body_serial
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

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
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-0}
PROGRESS_LOG=${PROGRESS_LOG:-${ME_DIR}/studies/two_body_parameter_sweep/archer2_serial_progress.log}

cd "${ME_DIR}"

echo "ARCHER2 serial two-body sweep"
echo "  job:         ${SLURM_JOB_ID:-manual}"
echo "  manifest:    ${MANIFEST}"
echo "  binary:      ${BINARY}"
echo "  rerun:       ${RERUN}"
echo "  start_index: ${START_INDEX}"
echo "  end_index:   ${END_INDEX:-all}"
echo "  max_cases:   ${MAX_CASES}"
echo "  rayon:       ${RAYON_NUM_THREADS}"
echo

echo "Building release binary with LAPACK..."
cargo build --release --features lapack,lapack-system
echo "Build complete."
echo

n_cases=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n_cases}" -le 0 ]]; then
    echo "No cases found in ${MANIFEST}" >&2
    exit 2
fi

if [[ -z "${END_INDEX}" ]]; then
    END_INDEX="${n_cases}"
fi

mkdir -p "$(dirname "${PROGRESS_LOG}")"
{
    echo
    echo "==== serial sweep job ${SLURM_JOB_ID:-manual} started $(date -Is) ===="
    echo "manifest=${MANIFEST}"
    echo "range=${START_INDEX}-${END_INDEX}, max_cases=${MAX_CASES}, rerun=${RERUN}"
} >> "${PROGRESS_LOG}"

ran_cases=0
skipped_cases=0
failed_cases=0

for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
    if [[ "${idx}" -gt "${n_cases}" ]]; then
        break
    fi
    if [[ "${MAX_CASES}" -gt 0 && "${ran_cases}" -ge "${MAX_CASES}" ]]; then
        echo "Reached MAX_CASES=${MAX_CASES}; stopping cleanly."
        break
    fi

    row=$(awk -v idx="${idx}" 'NR == idx + 1 { print; exit }' "${MANIFEST}")
    if [[ -z "${row}" ]]; then
        echo "No manifest row for index ${idx}; stopping." >&2
        break
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
            echo "[$(date -Is)] skip ${idx}/${n_cases} ${name}: ${actual_rows}/${expected_rows} rows"
            echo "skip,index=${idx},name=${name},rows=${actual_rows}/${expected_rows},time=$(date -Is)" >> "${PROGRESS_LOG}"
            skipped_cases=$((skipped_cases + 1))
            continue
        fi
    fi

    echo "[$(date -Is)] run ${idx}/${n_cases} ${name}"
    echo "run,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"

    if "${BINARY}" "${input_path}" "${output_dir}/" > "${run_log}" 2>&1; then
        ran_cases=$((ran_cases + 1))
        echo "done,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"
    else
        failed_cases=$((failed_cases + 1))
        echo "fail,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"
        echo "Case failed: ${idx} ${name}. See ${run_log}" >&2
        exit 1
    fi
done

{
    echo "summary,ran=${ran_cases},skipped=${skipped_cases},failed=${failed_cases},time=$(date -Is)"
    echo "==== serial sweep job ${SLURM_JOB_ID:-manual} finished $(date -Is) ===="
} >> "${PROGRESS_LOG}"

echo
echo "Serial sweep summary:"
echo "  ran:     ${ran_cases}"
echo "  skipped: ${skipped_cases}"
echo "  failed:  ${failed_cases}"
echo "  log:     ${PROGRESS_LOG}"
