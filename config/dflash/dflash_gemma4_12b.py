import os

from deepspec.trainer import Gemma4DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, GEMMA_4_12B


project_name = "deepspec"
exp_name = "dflash_block7_gemma4_12b"
seed = 42

model = dict(
    target_model_name_or_path=GEMMA_4_12B,
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[5, 17, 29, 41, 46],
    mask_token_id=4,
    num_anchors=512,

    # Disable markov head.
    markov_rank=0,

    # Disable confidence head.
    confidence_head_alpha=0.0,

    # CE-only loss.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Gemma4DSparkTrainer,
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
    chat_template="gemma4",
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
