"""Remove only the unreachable original head from a trained TCS inference model."""
import copy

from torch import nn
from models.factory import build_model


def compact_inference_model(model_config, state_dict):
    if model_config.get("variant") != "tcs_seg":
        raise ValueError("Compact export is only defined for the TCS decoder path")
    config = copy.deepcopy(model_config)
    config["pretrained"] = False  # all trained tensors are loaded below
    model = build_model(config)
    # Load strictly BEFORE removing anything: corrupt/mismatched source
    # checkpoints must never masquerade as a successful compact conversion.
    model.load_state_dict(state_dict, strict=True)
    model.base_model.decode_head = nn.Identity()
    return model.eval(), config


def load_compact_bundle(bundle):
    if bundle.get("format") != "tcs_compact_v1":
        raise ValueError("Unrecognized compact checkpoint format")
    config = copy.deepcopy(bundle["model_config"])
    if config.get("variant") != "tcs_seg" or config.get("pretrained") is not False:
        raise ValueError("Invalid compact model config")
    model = build_model(config)
    model.base_model.decode_head = nn.Identity()
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.eval()
