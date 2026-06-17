#!/bin/bash --login

#SBATCH --job-name=multi_ellip
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

#SBATCH --account=e643
#SBATCH --partition=standard
#SBATCH --qos=standard

export PROJ=e643
export WORK=/work/${PROJ}/${PROJ}/shared
export WORK_A=/work/${PROJ}/${PROJ}/adfboyd
#export X3D=${WORK}/shared/Incompact3d_Stock/Incompact3d/xcompact3d
export ME_DIR=${WORK_A}/multi_ellip
#  export OMP_NUM_THREADS=1
export CARGO_HOME=${WORK}/rust/
export RUSTUP_HOME=${WORK}/rust/
export PATH=${WORK}/rust/bin:$PATH
#export LD_LIBRARY_PATH=${WORK}/rust/bin:$LD_LIBRARY_PATH
module load PrgEnv-gnu
module load cray-libsci
cd ${ME_DIR}

# mkdir -p ${RUNDIR}
# lfs setstripe -c -1 ${RUNDIR}
# cd ${RUNDIR}

cargo build --release --features lapack,lapack-system 
cargo run --release --features lapack,lapack-system -- ${SLURM_SUBMIT_DIR}/input.txt ${SLURM_SUBMIT_DIR}/output/ > ${SLURM_SUBMIT_DIR}/me_out.log
wait