import os

from config.eagle3.eagle3_ministral3_3b import *  # noqa: F403

BASE_TB_DIR = "/mnt/vast/runs/andy/deepspec_tensorboard"
BASE_CKPT_DIR = "/mnt/vast/runs/andy/deepspec_checkpoints"
KL_DRAFT_CHECKPOINT = (
    "/mnt/vast/home/andy/checkpoints/deepspec/"
    "eagle3_ttt7_ministral3_3b_10k_3000steps/step_latest"
)

exp_name = "eagle3_ttt7_ministral3_3b_e2e_tv_from_kl"
model = dict(model)  # noqa: F405
model["loss_type"] = "e2e_tv"
model["init_draft_model_name_or_path"] = KL_DRAFT_CHECKPOINT


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
