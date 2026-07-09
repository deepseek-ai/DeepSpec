#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-evaltask
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --array=1-8%2
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%A_%a.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DRAFT_DIR=${HOME}/checkpoints/deepspec/eagle3_ttt7_ministral3_3b_10k_3000steps/step_latest
TB_ROOT=${HOME}/tensorboard/deepspec/eagle3_ttt7_ministral3_3b_10k_3000steps_eval_tasks

TASK_NAMES=(
    gsm8k
    math500
    aime25
    humaneval
    mbpp
    livecodebench
    mt-bench
    alpaca
    arena-hard-v2
)
TASK_COUNTS=(500 500 30 164 256 500 80 500 500)

TASK_NAME=${TASK_NAMES[${SLURM_ARRAY_TASK_ID}]}
TASK_COUNT=${TASK_COUNTS[${SLURM_ARRAY_TASK_ID}]}
SAFE_TASK_NAME=${TASK_NAME//[^a-zA-Z0-9]/_}

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29681}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python eval.py \
    --target_name_or_path mistralai/Ministral-3-3B-Instruct-2512 \
    --draft_name_or_path "${DRAFT_DIR}" \
    --tensorboard-dir "${TB_ROOT}/${SAFE_TASK_NAME}" \
    --output-json "${RUN_ROOT}/eval_10k_step3000_${SAFE_TASK_NAME}_metrics.json" \
    --step 3000 \
    --task "${TASK_NAME}:${TASK_COUNT}"
