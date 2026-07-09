#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-gen10k
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%A_%a.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DATA_DIR=${RUN_ROOT}/data_10k
NUM_SHARDS=8
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}

mkdir -p "${RUN_ROOT}/logs" "${DATA_DIR}/regen_shards"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export USE_HUB_KERNELS=0

python scripts/data/generate_train_data_transformers.py \
    --model mistralai/Ministral-3-3B-Instruct-2512 \
    --input-file-path "${DATA_DIR}/perfectblend_train_10k.jsonl" \
    --output-file-path "${DATA_DIR}/regen_shards/perfectblend_train_10k_regen_shard_${SHARD_INDEX}.jsonl" \
    --error-file-path "${DATA_DIR}/regen_shards/perfectblend_train_10k_regen_shard_${SHARD_INDEX}_error.jsonl" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --max-new-tokens 4096 \
    --resume
