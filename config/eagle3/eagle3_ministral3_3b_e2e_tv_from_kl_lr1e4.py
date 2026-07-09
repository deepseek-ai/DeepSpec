from config.eagle3.eagle3_ministral3_3b_e2e_tv_from_kl import *  # noqa: F403

exp_name = "eagle3_ttt7_ministral3_3b_e2e_tv_from_kl_lr1e4"
train = dict(train)  # noqa: F405
train["lr"] = 1.0e-4
