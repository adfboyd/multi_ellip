#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ME_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
MANIFEST="${ME_DIR}/studies/two_body_parameter_sweep/two_body_parameter_sweep_manifest.csv"
SETUP="${ME_DIR}/studies/two_body_parameter_sweep/setup_two_body_parameter_sweep.py"
BUILD_SLURM="${ME_DIR}/studies/two_body_parameter_sweep/archer2_build_w_lapack.slurm"
ARRAY_SLURM="${ME_DIR}/studies/two_body_parameter_sweep/archer2_array_w_lapack.slurm"

MAX_CONCURRENT=${MAX_CONCURRENT:-4}
RERUN=${RERUN:-0}

cd "${ME_DIR}"

python3 "${SETUP}" --write --portable-manifest

n_cases=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n_cases}" -le 0 ]]; then
    echo "No cases found in ${MANIFEST}" >&2
    exit 2
fi

build_job=$(sbatch --parsable --export=ALL,ME_DIR="${ME_DIR}" "${BUILD_SLURM}")
array_job=$(sbatch \
    --parsable \
    --dependency=afterok:${build_job} \
    --array=1-${n_cases}%${MAX_CONCURRENT} \
    --export=ALL,ME_DIR="${ME_DIR}",MANIFEST="${MANIFEST}",RERUN="${RERUN}" \
    "${ARRAY_SLURM}")

echo "Prepared ${n_cases} cases."
echo "Build job: ${build_job}"
echo "Array job: ${array_job}"
echo
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  sacct -j ${array_job} --format=JobID,JobName,State,Elapsed,ExitCode"
echo
echo "To change array concurrency:"
echo "  MAX_CONCURRENT=8 ${0}"
echo "To force rerun existing outputs:"
echo "  RERUN=1 ${0}"
