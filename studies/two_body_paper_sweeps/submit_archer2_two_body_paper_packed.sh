#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ME_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SETUP="${ME_DIR}/studies/two_body_paper_sweeps/setup_two_body_paper_sweeps.py"
PACKED_SLURM="${ME_DIR}/studies/two_body_paper_sweeps/archer2_packed_w_lapack.slurm"

SHAPE=${SHAPE:-spheroid_1_0p7_0p7}
MANIFEST="${ME_DIR}/studies/two_body_paper_sweeps/manifest_${SHAPE}.csv"

RERUN=${RERUN:-0}
PARTITION=${PARTITION:-standard}
QOS=${QOS:-standard}
WALLTIME=${WALLTIME:-24:00:00}
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-0}
CASES_CONCURRENT=${CASES_CONCURRENT:-32}
THREADS_PER_CASE=${THREADS_PER_CASE:-8}
PROGRESS_LOG=${PROGRESS_LOG:-${ME_DIR}/studies/two_body_paper_sweeps/archer2_${SHAPE}_progress.log}

cd "${ME_DIR}"

python3 "${SETUP}" --shape "${SHAPE}" --write --portable-manifest

n_cases=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n_cases}" -le 0 ]]; then
    echo "No cases found in ${MANIFEST}" >&2
    exit 2
fi

job_id=$(sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --time="${WALLTIME}" \
    --export=ALL,ME_DIR="${ME_DIR}",MANIFEST="${MANIFEST}",RERUN="${RERUN}",START_INDEX="${START_INDEX}",END_INDEX="${END_INDEX}",MAX_CASES="${MAX_CASES}",CASES_CONCURRENT="${CASES_CONCURRENT}",THREADS_PER_CASE="${THREADS_PER_CASE}",PROGRESS_LOG="${PROGRESS_LOG}" \
    "${PACKED_SLURM}")

echo "Prepared ${n_cases} two-body paper cases for ${SHAPE}."
echo "Packed job: ${job_id}"
echo
echo "Submitted with:"
echo "  shape:            ${SHAPE}"
echo "  partition:        ${PARTITION}"
echo "  qos:              ${QOS}"
echo "  cases_concurrent: ${CASES_CONCURRENT}"
echo "  threads_per_case: ${THREADS_PER_CASE}"
echo "  total threads:    $((CASES_CONCURRENT * THREADS_PER_CASE))"
echo "  walltime:         ${WALLTIME}"
echo
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROGRESS_LOG}"
echo "  sacct -j ${job_id} --format=JobID,JobName,State,Elapsed,ExitCode"
echo
echo "Resume/requeue by running this same command again; completed cases are skipped."
echo
echo "Useful options:"
echo "  SHAPE=ellipsoid_1_0p8_0p6 ${0}"
echo "  SHAPE=all ${0}"
echo "  MAX_CASES=8 WALLTIME=01:00:00 ${0}"
echo "  CASES_CONCURRENT=16 THREADS_PER_CASE=16 ${0}"
echo "  START_INDEX=25 ${0}"
echo "  RERUN=1 ${0}"
