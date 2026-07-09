#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-tvkl-3k
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.err

set -euo pipefail

export REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
export RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup
export DATA_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
export CACHE_DIR=${DATA_ROOT}/target_cache_10k

mkdir -p "${RUN_ROOT}/logs"
mkdir -p /mnt/vast/runs/andy/hf_home
mkdir -p /mnt/vast/runs/andy/torch_home
mkdir -p /mnt/vast/runs/andy/triton_cache
mkdir -p /mnt/vast/runs/andy/xdg_cache

export HF_HOME=/mnt/vast/runs/andy/hf_home
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=/mnt/vast/runs/andy/torch_home
export TRITON_CACHE_DIR=/mnt/vast/runs/andy/triton_cache/${SLURM_JOB_ID}
export XDG_CACHE_HOME=/mnt/vast/runs/andy/xdg_cache
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29861}
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc '
set -euo pipefail
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NNODES}"

python train.py \
    --config config/eagle3/eagle3_ministral3_3b_e2e_tv_from_kl.py \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "data.num_workers=2" \
    --opts "train.local_batch_size=1" \
    --opts "train.global_batch_size=512" \
    --opts "train.max_train_steps=3000" \
    --opts "logging.checkpointing_steps=3000" \
    --opts "logging.save_only_checkpointing_steps=25" \
    --opts "logging.keep_last_checkpoints=8"
'
