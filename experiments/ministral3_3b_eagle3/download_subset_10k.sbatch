#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-subset10k
#SBATCH --partition=cpu
#SBATCH --qos=research
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DATA_DIR=${RUN_ROOT}/data_10k

mkdir -p "${RUN_ROOT}/logs" "${DATA_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python scripts/data/download_streaming_subset.py \
    --sample-size 10000 \
    --output-path "${DATA_DIR}/perfectblend_train_10k.jsonl" \
    --skip-existing
