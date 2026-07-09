#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-tv-smoke
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup
DATA_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
CACHE_DIR=${DATA_ROOT}/smoke_target_cache

mkdir -p "${RUN_ROOT}/logs"
mkdir -p /mnt/vast/runs/andy/hf_home
mkdir -p /mnt/vast/runs/andy/torch_home
mkdir -p /mnt/vast/runs/andy/triton_cache
mkdir -p /mnt/vast/runs/andy/xdg_cache

cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export HF_HOME=/mnt/vast/runs/andy/hf_home
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=/mnt/vast/runs/andy/torch_home
export TRITON_CACHE_DIR=/mnt/vast/runs/andy/triton_cache/${SLURM_JOB_ID}
export XDG_CACHE_HOME=/mnt/vast/runs/andy/xdg_cache
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29701}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python train.py \
    --config config/eagle3/eagle3_ministral3_3b_e2e_tv.py \
    --opts "exp_name=eagle3_ttt7_ministral3_3b_e2e_tv_smoke" \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "data.num_workers=0" \
    --opts "train.local_batch_size=1" \
    --opts "train.global_batch_size=1" \
    --opts "train.max_train_steps=2" \
    --opts "logging.checkpointing_steps=1" \
    --opts "logging.keep_last_checkpoints=2"
