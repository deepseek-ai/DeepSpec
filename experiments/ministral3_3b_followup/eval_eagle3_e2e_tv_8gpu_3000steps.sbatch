#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-tv8-eval
#SBATCH --partition=h100
#SBATCH --qos=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/vast/home/andy/code/DeepSpec-ministral-repro-20260702-151809
RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_followup
DRAFT_DIR=/mnt/vast/runs/andy/deepspec_checkpoints/deepspec/eagle3_ttt7_ministral3_3b_e2e_tv_8gpu/step_latest
TB_DIR=/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/eagle3_ttt7_ministral3_3b_e2e_tv_8gpu_eval

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
export MASTER_PORT=${MASTER_PORT:-29781}
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=0

python eval.py \
    --target_name_or_path mistralai/Ministral-3-3B-Instruct-2512 \
    --draft_name_or_path "${DRAFT_DIR}" \
    --tensorboard-dir "${TB_DIR}" \
    --output-json "${RUN_ROOT}/eval_eagle3_e2e_tv_8gpu_step3000_metrics.json" \
    --step 3000
