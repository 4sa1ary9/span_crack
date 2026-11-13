from __future__ import annotations

import torch
from thop import clever_format, profile

from models.factory import build_model
from utils.config import CONFIG_ROOT, load_yaml_config, parse_config_args, require_section, validate_model_variant


DEFAULT_CONFIG_PATH = CONFIG_ROOT / "tcs_seg" / "flops.yaml"


def run(config_path) -> None:
    config = load_yaml_config(config_path)
    experiment_config = config.get("experiment", {})
    model_config = require_section(config, "model")
    input_config = require_section(config, "input")
    variant = model_config.get("variant", "tcs_seg")
    config_family = validate_model_variant(config, model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = experiment_config.get("name", "flops_experiment")
    print(f"Experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Using config: {config['_meta']['config_path']}")
    if config_family:
        print(f"Config family: {config_family} -> model.variant={variant}")

    image_size = input_config.get("image_size", 512)
    flops_model_config = dict(model_config)
    flops_model_config["pretrained"] = False
    model = build_model(flops_model_config).to(device)
    dummy_input = torch.randn(
        input_config.get("batch_size", 1),
        input_config.get("channels", 3),
        image_size,
        image_size,
        device=device,
    )
    flops, _profiled_params = profile(model, inputs=(dummy_input,))
    total_params = sum(parameter.numel() for parameter in model.parameters())
    flops, params = clever_format([flops, total_params], "%.3f")
    print(f"{variant} -> Params: {params}, FLOPs: {flops}")


def main() -> None:
    args = parse_config_args(
        "Calculate model FLOPs from a profiling-specific YAML config.",
        default_config_path=DEFAULT_CONFIG_PATH,
    )
    run(args.config)


if __name__ == "__main__":
    main()
