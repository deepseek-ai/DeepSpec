#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-cache-smoke
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
CACHE_DIR=${RUN_ROOT}/smoke_target_cache

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29631}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python scripts/data/prepare_target_cache.py \
    --config config/eagle3/eagle3_ministral3_3b.py \
    --train-data-path experiments/ministral3_3b_eagle3/smoke_train.jsonl \
    --output-dir "${CACHE_DIR}" \
    --min-loss-tokens 1 \
    --local-batch-size 1 \
    --num-workers 0 \
    --max-shard-bytes 1073741824
