#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ME_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SETUP="${ME_DIR}/studies/multibody_convergence_sweep/setup_multibody_convergence_sweep.py"
MANIFEST="${ME_DIR}/studies/multibody_convergence_sweep/manifest.csv"
PACKED_SLURM="${ME_DIR}/studies/multibody_convergence_sweep/archer2_packed_w_lapack.slurm"

RERUN=${RERUN:-0}
PARTITION=${PARTITION:-standard}
QOS=${QOS:-standard}
WALLTIME=${WALLTIME:-24:00:00}
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-0}
CASES_CONCURRENT=${CASES_CONCURRENT:-32}
THREADS_PER_CASE=${THREADS_PER_CASE:-8}

cd "${ME_DIR}"

python3 "${SETUP}" --write --portable-manifest

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
    --export=ALL,ME_DIR="${ME_DIR}",MANIFEST="${MANIFEST}",RERUN="${RERUN}",START_INDEX="${START_INDEX}",END_INDEX="${END_INDEX}",MAX_CASES="${MAX_CASES}",CASES_CONCURRENT="${CASES_CONCURRENT}",THREADS_PER_CASE="${THREADS_PER_CASE}" \
    "${PACKED_SLURM}")

echo "Prepared ${n_cases} multibody convergence cases."
echo "Packed job: ${job_id}"
echo
echo "Submitted with:"
echo "  partition:        ${PARTITION}"
echo "  qos:              ${QOS}"
echo "  cases_concurrent: ${CASES_CONCURRENT}"
echo "  threads_per_case: ${THREADS_PER_CASE}"
echo "  total threads:    $((CASES_CONCURRENT * THREADS_PER_CASE))"
echo "  walltime:         ${WALLTIME}"
echo
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  tail -f ${ME_DIR}/studies/multibody_convergence_sweep/archer2_packed_progress.log"
echo "  sacct -j ${job_id} --format=JobID,JobName,State,Elapsed,ExitCode"
echo
echo "Resume/requeue by running this same command again; completed cases are skipped."
echo
echo "Useful options:"
echo "  WALLTIME=12:00:00 ${0}"
echo "  MAX_CASES=8 ${0}"
echo "  CASES_CONCURRENT=16 THREADS_PER_CASE=16 ${0}"
echo "  START_INDEX=25 ${0}"
echo "  RERUN=1 ${0}"
