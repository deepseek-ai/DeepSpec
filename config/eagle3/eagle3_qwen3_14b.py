import os

from deepspec.trainer import Qwen3Eagle3Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_14B


project_name = "deepspec"
exp_name = "eagle3_ttt7_qwen3_14b"
seed = 0

model = dict(
    target_model_name_or_path=QWEN_3_14B,
    target_layer_ids=[1, 10, 19, 28, 37],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=Qwen3Eagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        project_name,
        exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
