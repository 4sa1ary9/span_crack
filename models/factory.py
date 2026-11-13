from __future__ import annotations

import torch.nn as nn

from models.segformer_base import TCSSegFormer


def build_model(model_config: dict) -> nn.Module:
    variant = model_config.get("variant", "tcs_seg")
    arch_config = model_config.get("arch", {}) or {}

    return TCSSegFormer(
        pretrained=model_config.get("pretrained", True),
        model_type=model_config.get("type", "b0"),
        variant=variant,
        arch_config=arch_config,
    )
