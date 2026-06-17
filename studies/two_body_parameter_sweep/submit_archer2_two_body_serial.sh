#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ME_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
MANIFEST="${ME_DIR}/studies/two_body_parameter_sweep/two_body_parameter_sweep_manifest.csv"
SETUP="${ME_DIR}/studies/two_body_parameter_sweep/setup_two_body_parameter_sweep.py"
SERIAL_SLURM="${ME_DIR}/studies/two_body_parameter_sweep/archer2_serial_w_lapack.slurm"

RERUN=${RERUN:-0}
WALLTIME=${WALLTIME:-24:00:00}
START_INDEX=${START_INDEX:-1}
END_INDEX=${END_INDEX:-}
MAX_CASES=${MAX_CASES:-0}

cd "${ME_DIR}"

python3 "${SETUP}" --write --portable-manifest

n_cases=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n_cases}" -le 0 ]]; then
    echo "No cases found in ${MANIFEST}" >&2
    exit 2
fi

job_id=$(sbatch \
    --parsable \
    --time="${WALLTIME}" \
    --export=ALL,ME_DIR="${ME_DIR}",MANIFEST="${MANIFEST}",RERUN="${RERUN}",START_INDEX="${START_INDEX}",END_INDEX="${END_INDEX}",MAX_CASES="${MAX_CASES}" \
    "${SERIAL_SLURM}")

echo "Prepared ${n_cases} cases."
echo "Serial job: ${job_id}"
echo
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  tail -f ${ME_DIR}/studies/two_body_parameter_sweep/archer2_serial_progress.log"
echo "  sacct -j ${job_id} --format=JobID,JobName,State,Elapsed,ExitCode"
echo
echo "Resume/requeue by running this same command again; completed cases are skipped."
echo
echo "Useful options:"
echo "  WALLTIME=12:00:00 ${0}"
echo "  START_INDEX=25 ${0}"
echo "  MAX_CASES=10 ${0}"
echo "  RERUN=1 ${0}"
