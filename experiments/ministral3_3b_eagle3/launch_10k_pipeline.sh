#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

download_id=$(sbatch --parsable "${HERE}/download_subset_10k.sbatch")
generate_id=$(sbatch --parsable --dependency=afterok:"${download_id}" "${HERE}/generate_subset_10k.sbatch")
merge_id=$(sbatch --parsable --dependency=afterok:"${generate_id}" "${HERE}/merge_subset_10k.sbatch")
cache_id=$(sbatch --parsable --dependency=afterok:"${merge_id}" "${HERE}/prepare_cache_10k.sbatch")
train_id=$(sbatch --parsable --dependency=afterok:"${cache_id}" "${HERE}/train_10k_3000steps.sbatch")
eval_id=$(sbatch --parsable --dependency=afterok:"${train_id}" "${HERE}/eval_10k_3000steps.sbatch")

cat <<EOF
download ${download_id}
generate ${generate_id}
merge    ${merge_id}
cache    ${cache_id}
train    ${train_id}
eval     ${eval_id}
EOF
