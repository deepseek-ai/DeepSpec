#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-train10k
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
CACHE_DIR=${RUN_ROOT}/target_cache_10k

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29661}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python train.py \
    --config config/eagle3/eagle3_ministral3_3b.py \
    --opts "exp_name=eagle3_ttt7_ministral3_3b_10k_3000steps" \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "data.num_workers=2" \
    --opts "train.local_batch_size=1" \
    --opts "train.global_batch_size=512" \
    --opts "train.max_train_steps=3000" \
    --opts "logging.checkpointing_steps=3000" \
    --opts "logging.save_only_checkpointing_steps=10" \
    --opts "logging.keep_last_checkpoints=8"
