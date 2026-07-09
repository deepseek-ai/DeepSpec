import os

from deepspec.trainer import Ministral3Eagle3Trainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "eagle3_ttt7_ministral3_3b"
seed = 0

model = dict(
    target_model_name_or_path="mistralai/Ministral-3-3B-Instruct-2512",
    target_layer_ids=[1, 7, 13, 19, 24],
    ttt_length=7,
    step_loss_decay=0.8,
    loss_type="soft_ce",
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=Ministral3Eagle3Trainer,
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
    save_only_checkpointing_steps=None,
    keep_last_checkpoints=None,
)

data = dict(
    target_cache_path=None,
    chat_template="ministral3",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
