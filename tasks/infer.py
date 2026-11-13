from __future__ import annotations

import torch
import cv2

from models.factory import build_model
from utils.config import (
    CONFIG_ROOT,
    load_yaml_config,
    parse_config_args,
    require_section,
    resolve_path,
    validate_model_variant,
)
from utils.dataloader import CrackDataset, DataLoader, get_transforms
from utils.output_manager import build_prediction_path, get_experiment_root


DEFAULT_CONFIG_PATH = CONFIG_ROOT / "tcs_seg" / "infer" / "deepcrack.yaml"


def run(config_path) -> None:
    config = load_yaml_config(config_path)
    experiment_config = config.get("experiment", {})
    model_config = require_section(config, "model")
    data_config = require_section(config, "data")
    inference_config = require_section(config, "inference")
    variant = model_config.get("variant", "tcs_seg")
    config_family = validate_model_variant(config, model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = experiment_config.get("name", "infer_experiment")
    print(f"Experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Using config: {config['_meta']['config_path']}")
    if config_family:
        print(f"Config family: {config_family} -> model.variant={variant}")

    checkpoint_path = resolve_path(model_config["checkpoint"])
    image_dir = resolve_path(data_config["image_dir"])
    threshold = inference_config.get("threshold", 0.5)
    experiment_root = get_experiment_root(inference_config, experiment_name)
    prediction_prefix = inference_config.get("prediction_prefix", experiment_name)

    infer_model_config = dict(model_config)
    infer_model_config["pretrained"] = False
    model = build_model(infer_model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    infer_dataset = CrackDataset(
        image_dir=image_dir,
        transform=get_transforms(mode="infer"),
        mode="infer",
        image_extensions=data_config.get("image_extensions"),
    )
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
    )
    if len(infer_dataset) == 0:
        raise FileNotFoundError(f"No input images found in: {image_dir}")

    print(f"Found {len(infer_dataset)} image(s) for inference.")
    with torch.no_grad():
        for batch in infer_loader:
            image_name = batch["image_name"][0]
            input_tensor = batch["image"].to(device)
            logits = model(input_tensor)
            # >= 0.5 matches the evaluation protocol (metrics use probability
            # >= threshold) so qualitative maps and quantitative stats agree.
            pred_mask = (torch.sigmoid(logits) >= threshold).float().squeeze().cpu().numpy() * 255
            pred_mask = pred_mask.astype("uint8")
            save_path = build_prediction_path(experiment_root, prediction_prefix, image_name)
            cv2.imwrite(str(save_path), pred_mask)
            print(f"Prediction saved to: {save_path}")


def main() -> None:
    args = parse_config_args(
        "Run crack segmentation inference from an inference-specific YAML config.",
        default_config_path=DEFAULT_CONFIG_PATH,
    )
    run(args.config)


if __name__ == "__main__":
    main()
