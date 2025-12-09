#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --job-name=pack_results
#SBATCH --output=pack_results_%A_%a.out
#SBATCH --array=0

# activate venv
source /lustre/gale/stf218/scratch/emin/ncclvenv/bin/activate

# set misc env vars
export LD_LIBRARY_PATH=/lustre/gale/stf218/scratch/emin/aws-ofi-nccl-1.14.0/lib:$LD_LIBRARY_PATH  # enable aws-ofi-nccl
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3   # can improve performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1       # on large systems, this nccl setting has been found to improve performance
export HF_HOME="/lustre/gale/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/gale/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/gale/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/gale/stf218/scratch/emin/pytorch_kernel_cache"
export MPLCONFIGDIR="/lustre/gale/stf218/scratch/emin/mplconfigdir"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

srun csc pack-results

echo "Done"