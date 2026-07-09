import os

os.environ["USE_HUB_KERNELS"] = "0"

from .dspark import (
    DSparkForwardOutput,
    Gemma4DSparkModel,
    Ministral3DSparkModel,
    Qwen3DSparkModel,
)
from .eagle3 import Gemma4Eagle3Model, Qwen3Eagle3Model

__all__ = [
    "DSparkForwardOutput",
    "Gemma4Eagle3Model",
    "Gemma4DSparkModel",
    "Ministral3DSparkModel",
    "Qwen3Eagle3Model",
    "Qwen3DSparkModel",
]
