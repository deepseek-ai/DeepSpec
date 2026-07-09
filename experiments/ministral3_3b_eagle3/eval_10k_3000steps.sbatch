#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-eval10k
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DRAFT_DIR=${HOME}/checkpoints/deepspec/eagle3_ttt7_ministral3_3b_10k_3000steps/step_latest
TB_DIR=${HOME}/tensorboard/deepspec/eagle3_ttt7_ministral3_3b_10k_3000steps_eval

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29671}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python eval.py \
    --target_name_or_path mistralai/Ministral-3-3B-Instruct-2512 \
    --draft_name_or_path "${DRAFT_DIR}" \
    --tensorboard-dir "${TB_DIR}" \
    --output-json "${RUN_ROOT}/eval_10k_step3000_metrics.json" \
    --step 3000
