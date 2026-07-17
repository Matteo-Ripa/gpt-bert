#!/bin/bash

# Launch script used by slurm scripts, don't invoke directly.

module purge
module load "virtualenv/20.26.2-GCCcore-13.3.0"
source /mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/envs/babylm_NEW/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u "$@"