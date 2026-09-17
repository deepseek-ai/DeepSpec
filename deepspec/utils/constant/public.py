import os

## cache dir
CACHE_DIR = os.path.expanduser("~/.cache/deepspec")

## model_name_or_path
QWEN_3_4B = "Qwen/Qwen3-4B"
QWEN_3_8B = "Qwen/Qwen3-8B"
QWEN_3_14B = "Qwen/Qwen3-14B"
GEMMA_4_12B = "google/gemma-4-12B-it"
BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")

## auto eval
auto_eval_command = None