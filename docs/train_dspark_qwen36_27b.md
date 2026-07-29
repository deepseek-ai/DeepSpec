# Training a DSpark Drafter for Qwen3.6-27B

The pipeline has 3 stages: **data preparation** → **target cache** → **training**. Below are the commands adapted for `Qwen/Qwen3.6-27B`.

## Prerequisites

```bash
pip install -r requirements.txt
pip install "sglang[all]"  # for data generation server
```

## Step 1: Download and Split Prompt Data

```bash
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

## Step 2: Regenerate Assistant Answers with Qwen3.6-27B

The 27B model needs tensor parallelism (TP). Each SGLang worker below uses 2 GPUs (`--tp 2`), so 8 GPUs gives 4 workers.

### Terminal 1 — Launch SGLang servers

```bash
#!/usr/bin/env bash
model_path=Qwen/Qwen3.6-27B
num_workers=4
start_port=30000
log_dir=logs/sglang_qwen36_27b

mkdir -p "${log_dir}"

for ((i = 0; i < num_workers; i++)); do
    port=$((start_port + i))
    gpu_start=$((i * 2))
    gpu_end=$((gpu_start + 1))
    log_file="${log_dir}/worker_${i}_port_${port}.log"

    echo "Starting worker ${i} on GPUs ${gpu_start},${gpu_end} port=${port}"
    CUDA_VISIBLE_DEVICES=${gpu_start},${gpu_end} sglang serve \
        --model-path "${model_path}" \
        --host 0.0.0.0 \
        --port "${port}" \
        --tp 2 \
        --dtype bfloat16 \
        --mem-fraction-static 0.9 \
        > "${log_file}" 2>&1 &
done
wait
```

### Terminal 2 — Regenerate data

```bash
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3.6-27B \
    --server-address \
        127.0.0.1:30000 \
        127.0.0.1:30001 \
        127.0.0.1:30002 \
        127.0.0.1:30003 \
    --concurrency 16 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen36_27b/perfectblend_train_regen.jsonl
```

Stop SGLang before Step 3 if sharing GPUs.

## Step 3: Prepare Target Cache

This runs the 27B target model on all training data and saves intermediate hidden states.

> **Storage warning:** With 5 target layers and hidden_size=3456 on the full dataset, the cache can be very large (hundreds of TB). Reduce the training set or use fewer `target_layer_ids` if storage is limited.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_27b.py \
    --train-data-path train_datasets/qwen36_27b/perfectblend_train_regen.jsonl \
    --output-dir ${HOME}/.cache/deepspec/qwen36_27b_target_cache \
    --local-batch-size 8
```

Use `--local-batch-size 4` or lower if you hit OOM — the 27B model is larger.

## Step 4: Train the DSpark Drafter

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_27b.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen36_27b_target_cache"
```

For multi-node training, set `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` environment variables. You may also want to switch sharding:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_27b.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen36_27b_target_cache" \
    --opts "train.sharding_strategy=full_shard"
```

## Step 5: Evaluate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3.6-27B \
    --draft_name_or_path ${HOME}/checkpoints/deepspec/dspark_block7_qwen3_27b/step_latest
```

## Key Differences from Smaller Qwen3 Models

| Parameter | Qwen3-4B/8B | Qwen3-14B | Qwen3.6-27B |
|-----------|-------------|-----------|-------------|
| `num_hidden_layers` | 36 | 40 | **64** |
| `target_layer_ids` | [1,9,17,25,33] | [1,10,19,28,37] | **[1,16,31,46,61]** |
| SGLang TP | 1 | 1 | **2** (needs 2 GPUs per server) |
| `local_batch_size` (cache) | 16 | 16 | **8** (lower due to memory) |

The draft model itself remains small (5 layers) regardless of target size — only the target cache preparation and serving require more GPU memory.
