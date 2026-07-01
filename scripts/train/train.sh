#!/usr/bin/env bash
set -euo pipefail

# Prepare the desired Python and accelerator runtime before running this script.
# train.py spawns one worker per visible accelerator.
if [[ "${DEEPSPEC_DEVICE:-}" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

config_path=${config_path:-config/dspark/dspark_gemma4_12b.py}
target_model_path=${target_model_path:-google/gemma-4-12B-it}
target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/gemma4_target_cache}
target_layer_ids=${target_layer_ids:-}
global_batch_size=${global_batch_size:-512}
max_train_steps=${max_train_steps:-}
data_max_length=${data_max_length:-4096}
logging_steps=${logging_steps:-10}
checkpointing_steps=${checkpointing_steps:-3000}
exp_name=${exp_name:-}

cmd=(
    python train.py
    --config "${config_path}"
    --opts "model.target_model_name_or_path=${target_model_path}"
    --opts "data.target_cache_path=${target_cache_dir}"
    --opts "data.max_length=${data_max_length}"
    --opts "train.global_batch_size=${global_batch_size}"
    --opts "logging.logging_steps=${logging_steps}"
    --opts "logging.checkpointing_steps=${checkpointing_steps}"
)

if [[ -n "${target_layer_ids}" ]]; then
    cmd+=(--opts "model.target_layer_ids=${target_layer_ids}")
fi
if [[ -n "${max_train_steps}" ]]; then
    cmd+=(--opts "train.max_train_steps=${max_train_steps}")
fi
if [[ -n "${exp_name}" ]]; then
    cmd+=(--opts "exp_name=${exp_name}")
fi

"${cmd[@]}"
