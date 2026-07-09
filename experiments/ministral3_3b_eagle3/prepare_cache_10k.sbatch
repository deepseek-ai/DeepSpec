#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-cache10k
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DATA_DIR=${RUN_ROOT}/data_10k
CACHE_DIR=${RUN_ROOT}/target_cache_10k

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29651}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python scripts/data/prepare_target_cache.py \
    --config config/eagle3/eagle3_ministral3_3b.py \
    --train-data-path "${DATA_DIR}/perfectblend_train_10k_regen.jsonl" \
    --output-dir "${CACHE_DIR}" \
    --local-batch-size 1 \
    --num-workers 2 \
    --max-shard-bytes 17179869184
