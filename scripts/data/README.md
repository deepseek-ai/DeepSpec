# Data Preparation

This directory contains an example data preparation pipeline using `Qwen/Qwen3-4B` as the target model.

DeepSpec trains draft models against a target model. The data pipeline does three things:

1. download and split prompt data,
2. regenerate assistant answers with the target model,
3. precompute the target cache used by training.

The example below targets `Qwen/Qwen3-4B`, but the same pipeline applies to other models (e.g. Gemma). To switch targets, change the model name (`--model` / `model_path`) and adjust the sampling parameters (`--temperature`, `--top-p`, `--top-k` and `--min-p`) to match the recommended generation settings for that model. Output paths in the examples reference `qwen3_4b`; rename them as needed.

The wrapper script [prepare_data.sh](./prepare_data.sh) records the default settings. The individual Python scripts are also documented below for users who want to run each stage manually.

## Outputs

Default outputs:

```text
train_datasets/perfectblend_train.jsonl
train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
~/.cache/deepspec/qwen3_4b_target_cache
```

The example scripts assume a single machine with eight visible GPUs by default. For fewer GPUs, set `CUDA_VISIBLE_DEVICES` before running the shell scripts; `num_workers` is derived from the visible GPU list unless you override it explicitly.

## Step 1: Download And Split Data

The source dataset is `mlabonne/open-perfectblend`. The train split is written as JSONL, and the held-out user turns are written under `eval_datasets/`.

```bash
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

This produces:

```text
train_datasets/perfectblend_train.jsonl
eval_datasets/perfectblend.jsonl
```

## Step 2: Regenerate Answers With Qwen3-4B

This step serves the target model and regenerates assistant answers against it. Any OpenAI-compatible inference engine works (SGLang, vLLM, TGI, etc.) — the example below uses [SGLang](https://github.com/sgl-project/sglang), but you can swap in whatever engine you prefer as long as it exposes an OpenAI-compatible `/v1` endpoint. SGLang is not in `requirements.txt`; install it separately, e.g. `pip install "sglang[all]"`.

Start local sglang servers in one terminal:

```bash
bash scripts/data/launch_sglang_server.sh
```

By default this starts eight `Qwen/Qwen3-4B` workers on ports `30000` to `30007` and writes logs to:

```text
logs/sglang_qwen3_4b/
```

For fewer GPUs, pass the device list to the launcher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/data/launch_sglang_server.sh
```

This starts four workers on ports `30000` to `30003`. Use the same `CUDA_VISIBLE_DEVICES`, `num_workers`, and `start_port` values when running `prepare_data.sh` so the data generation step connects to the right servers.

In another terminal, regenerate the assistant answers:

```bash
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address \
        127.0.0.1:30000 \
        127.0.0.1:30001 \
        127.0.0.1:30002 \
        127.0.0.1:30003 \
        127.0.0.1:30004 \
        127.0.0.1:30005 \
        127.0.0.1:30006 \
        127.0.0.1:30007 \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

This produces:

```text
train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

If any samples fail, the script writes them to:

```text
train_datasets/qwen3_4b/perfectblend_train_regen_error.jsonl
```

Stop the sglang servers before the next step if they are using the same GPUs.

## Step 3: Prepare Target Cache

The training loop reads a precomputed target cache instead of repeatedly running the target model. Prepare it with:

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
    --output-dir ${HOME}/.cache/deepspec/qwen3_4b_target_cache \
    --local-batch-size 16
```

> **Storage warning:** The target cache stores per-token hidden states for the
> full training set and can be very large. With the default `Qwen/Qwen3-4B`
> setting it takes roughly **38 TB** of disk. Make sure the `--output-dir`
> filesystem has enough free space (scaling with dataset size, sequence length,
> and target hidden dimension) before running this step. If storage is limited,
> use a smaller training set and/or reduce `model.target_layer_ids` in the config
> (fewer captured layers means proportionally less cache).

This produces the cache consumed by [scripts/train/train.sh](../train/train.sh):

```text
~/.cache/deepspec/qwen3_4b_target_cache
```

## Wrapper Script

The wrapper script combines the default public commands:

```bash
bash scripts/data/prepare_data.sh
```

Use the manual commands above if you want to stop and restart services between stages, change sampling parameters, use fewer GPUs, or inspect intermediate outputs.
