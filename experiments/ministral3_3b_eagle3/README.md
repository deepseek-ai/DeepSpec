# Ministral3 EAGLE3 Reproduction

This directory tracks the DeepSpec-side reproduction attempt for
`mistralai/Ministral-3-3B-Instruct-2512`.

The smoke scripts are intentionally tiny:

- `prepare_cache_smoke.sbatch` builds a four-sample target cache.
- `train_smoke.sbatch` trains the EAGLE3 draft for two optimizer steps against
  that cache.

They validate model loading, target hidden-state extraction, tokenizer/loss-mask
parsing, and the EAGLE3 training loop before launching a full Open-PerfectBlend
run.

The first reproduction pipeline uses a 10k Open-PerfectBlend subset generated
by the Ministral3 target, then trains for 3000 optimizer steps at DeepSpec's
EAGLE3 global batch size of 512:

- `download_subset_10k.sbatch` streams and normalizes the source subset.
- `generate_subset_10k.sbatch` regenerates assistant turns with Ministral3.
- `merge_subset_10k.sbatch` combines the eight regeneration shards.
- `prepare_cache_10k.sbatch` builds the target hidden-state cache.
- `train_10k_3000steps.sbatch` trains
  `eagle3_ttt7_ministral3_3b_10k_3000steps`.
- `eval_10k_3000steps.sbatch` evaluates the resulting `step_latest`.

`launch_10k_pipeline.sh` submits those jobs with `afterok` dependencies. Outputs
live under `/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3`, checkpoints under
`~/checkpoints/deepspec`, and TensorBoard logs under `~/tensorboard/deepspec`.
