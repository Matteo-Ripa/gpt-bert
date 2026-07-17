#!/bin/bash
#SBATCH -A NAISS2025-5-551
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A40:4
#SBATCH --ntasks-per-node=4
#SBATCH -p alvis
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=bert-%j.out

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9856
export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE= $WORLD_SIZE"

export NCCL_SOCKET_IFNAME=ib0,ib1

CONTAINER=/apps/containers/PyTorch/PyTorch-NGC-latest.sif

set -euo pipefail

CMD="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/pretraining/train_ACLM_new1.py \
  --name ltg_128_4k_64_eleni_15 \
  --hybrid_numerator 1 \
  --hybrid_denominator 1 \
  --tokenizer_path /mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/tokenizers/tokenizer_10M_4k.json \
  --config_file /mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/configs/small.json"


echo $CMD
echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    singularity exec \
    "$CONTAINER" \
    ./pretraining/launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"