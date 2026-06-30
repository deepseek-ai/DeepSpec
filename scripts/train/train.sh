#!/usr/bin/env bash

# Local launch mirrors the repo's node launcher, not standard
# torchrun semantics. train.py spawns one worker per visible GPU by itself.
# Here RANK/WORLD_SIZE mean node_rank/node_count, so WORLD_SIZE=1 is a
# single-node local run; total GPU workers come from CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# Available public configs:
## dflash
#   config/dflash/dflash_gemma4_12b.py
#   config/dflash/dflash_qwen3_4b.py
#   config/dflash/dflash_qwen3_8b.py
#   config/dflash/dflash_qwen3_14b.py
## dspark
#   config/dspark/dspark_gemma4_12b.py
#   config/dspark/dspark_qwen3_4b.py
#   config/dspark/dspark_qwen3_8b.py
#   config/dspark/dspark_qwen3_14b.py
## eagle3
#   config/eagle3/eagle3_gemma4_12b.py
#   config/eagle3/eagle3_qwen3_4b.py
#   config/eagle3/eagle3_qwen3_8b.py
#   config/eagle3/eagle3_qwen3_14b.py

# The training config. Defaults to the cached DSpark Qwen3-4B config; override it by
# passing a config path as the first argument, e.g. to use online training:
#   bash scripts/train/train.sh config/dspark/dspark_qwen3_4b_online.py
config_path=${1:-config/dspark/dspark_qwen3_4b.py}

target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}

# --opts overrides any config field by dotted key path: --opts "<key.path>=<value>".
# Values are parsed as Python scalars (int/float/bool/str). Repeat the flag to set
# multiple fields, e.g.:
#   --opts "data.target_cache_path=${target_cache_dir}" \
#   --opts "train.lr=3e-4" \
#   --opts "train.local_batch_size=2"
#
# local_batch_size is the per-GPU micro-batch size. Raise it to better utilize GPUs
# with more memory (e.g. 4 or 8 on 80GB cards), or keep it at 1 if you hit OOM.
# Override it without editing the config via:
#   --opts "train.local_batch_size=4"
# data.target_cache_path is ignored by online configs (data.online_target=True), which
# stream conversations from data.train_data_paths instead of reading a target cache.
python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${target_cache_dir}"
