#!/usr/bin/env bash
#SBATCH --job-name=dspec-min3-merge10k
#SBATCH --partition=cpu
#SBATCH --qos=research
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.out
#SBATCH --error=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/logs/%x-%j.err

set -euo pipefail

RUN_ROOT=/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3
DATA_DIR=${RUN_ROOT}/data_10k
OUT=${DATA_DIR}/perfectblend_train_10k_regen.jsonl

mkdir -p "${RUN_ROOT}/logs"
rm -f "${OUT}"
for shard in $(seq 0 7); do
    shard_path=${DATA_DIR}/regen_shards/perfectblend_train_10k_regen_shard_${shard}.jsonl
    test -s "${shard_path}"
    cat "${shard_path}" >> "${OUT}"
done

wc -l "${OUT}"
