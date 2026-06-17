#!/bin/bash --login

#SBATCH --job-name=me_build
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

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

module load PrgEnv-gnu
module load cray-libsci

cd "${ME_DIR}"
cargo build --release --features lapack,lapack-system
