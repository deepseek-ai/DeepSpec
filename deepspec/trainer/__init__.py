from .base_trainer import BaseTrainer
from .dspark_trainer import Gemma4DSparkTrainer, Qwen3DSparkTrainer


def __getattr__(name):
    if name in {"Gemma4Eagle3Trainer", "Qwen3Eagle3Trainer"}:
        from .eagle3_trainer import Gemma4Eagle3Trainer, Qwen3Eagle3Trainer

        return {
            "Gemma4Eagle3Trainer": Gemma4Eagle3Trainer,
            "Qwen3Eagle3Trainer": Qwen3Eagle3Trainer,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BaseTrainer",
    "Gemma4Eagle3Trainer",
    "Gemma4DSparkTrainer",
    "Qwen3Eagle3Trainer",
    "Qwen3DSparkTrainer",
]
