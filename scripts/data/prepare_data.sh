#!/usr/bin/env bash
set -euo pipefail

model_path=${model_path:-Qwen/Qwen3-4B}
config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}

dataset_name=${dataset_name:-mlabonne/open-perfectblend}
test_size=${test_size:-0.05}
train_split_path=${train_split_path:-train_datasets/perfectblend_train.jsonl}
eval_data_dir=${eval_data_dir:-eval_datasets}

train_data_path=${train_data_path:-train_datasets/qwen3_4b/perfectblend_train_regen.jsonl}
cache_dir=${cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}

server_host=${server_host:-127.0.0.1}
visible_gpus=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a visible_gpu_ids <<< "${visible_gpus}"
num_workers=${num_workers:-${#visible_gpu_ids[@]}}
start_port=${start_port:-30000}
concurrency=${concurrency:-32}
temperature=${temperature:-0.7}
top_p=${top_p:-0.8}
top_k=${top_k:-20}
min_p=${min_p:-0}
max_tokens=${max_tokens:-4096}
local_batch_size=${local_batch_size:-16}

if (( num_workers < 1 )); then
    echo "num_workers must be at least 1" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${visible_gpus}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

echo "Step 1/3: downloading and splitting ${dataset_name}"
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: generating ${model_path} train data: ${train_data_path}"
echo "Start sglang first with: bash scripts/data/launch_sglang_server.sh"
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

echo "Stop sglang before Step 3 if it is using the same GPUs."
echo "Step 3/3: preparing ${model_path} target cache: ${cache_dir}"
python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size "${local_batch_size}"
