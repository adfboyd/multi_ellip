#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ME_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
MANIFEST="${ME_DIR}/studies/two_body_parameter_sweep/two_body_parameter_sweep_manifest.csv"
SETUP="${ME_DIR}/studies/two_body_parameter_sweep/setup_two_body_parameter_sweep.py"
SERIAL_SLURM="${ME_DIR}/studies/two_body_parameter_sweep/archer2_serial_w_lapack.slurm"

SHORT_PARTITION=${SHORT_PARTITION:-standard}
SHORT_QOS=${SHORT_QOS:-short}
WALLTIME=${WALLTIME:-00:20:00}
RERUN=${RERUN:-0}
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-1}
PROGRESS_LOG=${PROGRESS_LOG:-${ME_DIR}/studies/two_body_parameter_sweep/archer2_short_test_progress.log}

cd "${ME_DIR}"

python3 "${SETUP}" --write --portable-manifest

n_cases=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n_cases}" -le 0 ]]; then
    echo "No cases found in ${MANIFEST}" >&2
    exit 2
fi

job_id=$(sbatch \
    --parsable \
    --partition="${SHORT_PARTITION}" \
    --qos="${SHORT_QOS}" \
    --time="${WALLTIME}" \
    --job-name=me_2body_short \
    --export=ALL,ME_DIR="${ME_DIR}",MANIFEST="${MANIFEST}",RERUN="${RERUN}",START_INDEX="${START_INDEX}",END_INDEX="${END_INDEX}",MAX_CASES="${MAX_CASES}",PROGRESS_LOG="${PROGRESS_LOG}" \
    "${SERIAL_SLURM}")

echo "Prepared ${n_cases} cases."
echo "Short test job: ${job_id}"
echo
echo "Submitted with:"
echo "  partition: ${SHORT_PARTITION}"
echo "  qos:       ${SHORT_QOS}"
echo "  walltime:  ${WALLTIME}"
echo "  max_cases: ${MAX_CASES}"
echo
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROGRESS_LOG}"
echo "  sacct -j ${job_id} --format=JobID,JobName,Partition,QOS,State,Elapsed,ExitCode"
echo
echo "Useful options:"
echo "  START_INDEX=5 ${0}"
echo "  MAX_CASES=2 ${0}"
echo "  SHORT_PARTITION=standard SHORT_QOS=short ${0}"
