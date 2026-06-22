#!/bin/bash --login

#SBATCH --job-name=me_mb_conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
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
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}

module load PrgEnv-gnu
module load cray-libsci

MANIFEST=${MANIFEST:-${ME_DIR}/studies/multibody_convergence_sweep/manifest.csv}
BINARY=${BINARY:-${ME_DIR}/target/release/multi_ellip}
RERUN=${RERUN:-0}
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-0}
CASES_CONCURRENT=${CASES_CONCURRENT:-32}
THREADS_PER_CASE=${THREADS_PER_CASE:-8}
PROGRESS_LOG=${PROGRESS_LOG:-${ME_DIR}/studies/multibody_convergence_sweep/archer2_packed_progress.log}

cd "${ME_DIR}"

if [[ $((CASES_CONCURRENT * THREADS_PER_CASE)) -gt "${SLURM_CPUS_PER_TASK:-256}" ]]; then
    echo "Requested CASES_CONCURRENT*THREADS_PER_CASE exceeds allocated CPUs." >&2
    echo "  cases_concurrent=${CASES_CONCURRENT}" >&2
    echo "  threads_per_case=${THREADS_PER_CASE}" >&2
    echo "  cpus=${SLURM_CPUS_PER_TASK:-256}" >&2
    exit 2
fi

echo "ARCHER2 packed multibody convergence sweep"
echo "  job:              ${SLURM_JOB_ID:-manual}"
echo "  manifest:         ${MANIFEST}"
echo "  binary:           ${BINARY}"
echo "  rerun:            ${RERUN}"
echo "  start_index:      ${START_INDEX}"
echo "  end_index:        ${END_INDEX:-all}"
echo "  max_cases:        ${MAX_CASES}"
echo "  cases_concurrent: ${CASES_CONCURRENT}"
echo "  threads_per_case: ${THREADS_PER_CASE}"
echo "  omp_threads:      ${OMP_NUM_THREADS}"
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
    echo "==== packed multibody convergence job ${SLURM_JOB_ID:-manual} started $(date -Is) ===="
    echo "manifest=${MANIFEST}"
    echo "range=${START_INDEX}-${END_INDEX}, max_cases=${MAX_CASES}, rerun=${RERUN}"
    echo "cases_concurrent=${CASES_CONCURRENT}, threads_per_case=${THREADS_PER_CASE}"
} >> "${PROGRESS_LOG}"

manifest_field() {
    local idx="$1"
    python3 - "$MANIFEST" "$idx" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
idx = int(sys.argv[2])
with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if idx < 1 or idx > len(rows):
    sys.exit(1)
row = rows[idx - 1]
print("\t".join([row["name"], row["input"], row["output"], row["dt"], row["tend"]]))
PY
}

wait_one_batch() {
    local pid
    local status
    local failed=0
    for pid in "$@"; do
        if wait "${pid}"; then
            :
        else
            status=$?
            echo "Background case process ${pid} failed with status ${status}" >&2
            failed=1
        fi
    done
    return "${failed}"
}

running=0
launched_cases=0
skipped_cases=0
failed_cases=0
status_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/me_mb_conv_status.XXXXXX")
pids=()

for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
    if [[ "${idx}" -gt "${n_cases}" ]]; then
        break
    fi
    if [[ "${MAX_CASES}" -gt 0 && "${launched_cases}" -ge "${MAX_CASES}" ]]; then
        echo "Reached MAX_CASES=${MAX_CASES}; stopping after active cases finish."
        break
    fi

    row=$(manifest_field "${idx}" || true)
    if [[ -z "${row}" ]]; then
        echo "No manifest row for index ${idx}; stopping." >&2
        break
    fi

    IFS=$'\t' read -r name input_rel output_rel dt tend <<< "${row}"
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

    echo "[$(date -Is)] launch ${idx}/${n_cases} ${name}"
    echo "launch,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"

    (
        export RAYON_NUM_THREADS="${THREADS_PER_CASE}"
        if "${BINARY}" "${input_path}" "${output_dir}/" > "${run_log}" 2>&1; then
            echo "done,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"
            touch "${status_dir}/${idx}.ok"
        else
            echo "fail,index=${idx},name=${name},time=$(date -Is)" >> "${PROGRESS_LOG}"
            touch "${status_dir}/${idx}.fail"
            exit 1
        fi
    ) &
    pids+=("$!")
    running=$((running + 1))
    launched_cases=$((launched_cases + 1))

    if [[ "${running}" -ge "${CASES_CONCURRENT}" ]]; then
        if ! wait_one_batch "${pids[@]}"; then
            failed_cases=$((failed_cases + 1))
            echo "At least one packed case failed; stopping. See run.log files and ${PROGRESS_LOG}" >&2
            exit 1
        fi
        pids=()
        running=0
    fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
    if ! wait_one_batch "${pids[@]}"; then
        failed_cases=$((failed_cases + 1))
        echo "At least one packed case failed; stopping. See run.log files and ${PROGRESS_LOG}" >&2
        exit 1
    fi
fi

ok_count=$(find "${status_dir}" -name '*.ok' -type f 2>/dev/null | wc -l)
fail_count=$(find "${status_dir}" -name '*.fail' -type f 2>/dev/null | wc -l)
failed_cases=$((failed_cases + fail_count))
rm -rf "${status_dir}"

{
    echo "summary,launched=${launched_cases},completed=${ok_count},skipped=${skipped_cases},failed=${failed_cases},time=$(date -Is)"
    echo "==== packed multibody convergence job ${SLURM_JOB_ID:-manual} finished $(date -Is) ===="
} >> "${PROGRESS_LOG}"

echo
echo "Packed sweep summary:"
echo "  launched:  ${launched_cases}"
echo "  completed: ${ok_count}"
echo "  skipped:   ${skipped_cases}"
echo "  failed:    ${failed_cases}"
echo "  log:       ${PROGRESS_LOG}"
