from __future__ import annotations

import os

import torch

from models.factory import build_model
from utils.checkpoints import resolve_checkpoint_override
from utils.config import (
    build_config_parser,
    CONFIG_ROOT,
    load_yaml_config,
    require_section,
    resolve_path,
    validate_model_variant,
)
from utils.dataloader import CrackDataset, DataLoader, get_transforms
from utils.metrics import evaluate_fixed_threshold, evaluate_threshold_sweep
from utils.output_manager import build_metrics_path, get_experiment_root, save_json_report


os.environ["HF_HUB_OFFLINE"] = "1"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "tcs_seg" / "eval" / "deepcrack.yaml"
DEFAULT_FIXED_THRESHOLD = 0.5
DEFAULT_THRESHOLD_PROBES = (0.5, 0.9)


def _resolve_fixed_threshold(evaluation_config: dict) -> tuple[float, str]:
    """Return the threshold used for fixed-threshold metrics.

    Do not fall back to the test-set best threshold. Using the test-set best
    threshold for the fixed-F1 column makes fixed F1 indistinguishable from an
    ODS-style result and weakens comparison fairness.
    """
    if evaluation_config.get("fixed_threshold") is None:
        return DEFAULT_FIXED_THRESHOLD, "default_0.5"
    return float(evaluation_config["fixed_threshold"]), "config.fixed_threshold"


def _resolve_threshold_probes(evaluation_config: dict, fixed_threshold: float) -> list[float]:
    raw_values = evaluation_config.get("threshold_probes", DEFAULT_THRESHOLD_PROBES)
    if isinstance(raw_values, (int, float)):
        raw_values = [raw_values]

    values = [float(threshold) for threshold in raw_values]
    values.append(float(fixed_threshold))

    deduped: list[float] = []
    for threshold in values:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold_probes values must be in [0, 1], got {threshold}.")
        if not any(abs(threshold - existing) < 1e-12 for existing in deduped):
            deduped.append(threshold)
    return sorted(deduped)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def run(config_path, checkpoint_override: str | None = None) -> None:
    config = load_yaml_config(config_path)
    experiment_config = config.get("experiment", {})
    model_config = require_section(config, "model")
    data_config = require_section(config, "data")
    evaluation_config = require_section(config, "evaluation")
    output_config = require_section(config, "output")

    variant = model_config.get("variant", "tcs_seg")
    config_family = validate_model_variant(config, model_config)
    fixed_threshold, fixed_threshold_source = _resolve_fixed_threshold(evaluation_config)
    threshold_probes = _resolve_threshold_probes(evaluation_config, fixed_threshold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = experiment_config.get("name", "eval_experiment")
    print(f"Experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Using config: {config['_meta']['config_path']}")
    if config_family:
        print(f"Config family: {config_family} -> model.variant={variant}")
    print(f"Fixed threshold: {fixed_threshold:.4f} ({fixed_threshold_source})")
    print("Threshold probes: " + ", ".join(f"{threshold:.4f}" for threshold in threshold_probes))

    test_image_dir = resolve_path(data_config["image_dir"])
    test_mask_dir = resolve_path(data_config["mask_dir"])
    checkpoint_resolution = resolve_checkpoint_override(model_config["checkpoint"], checkpoint_override)
    checkpoint_path = checkpoint_resolution.path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data_name = experiment_config.get("dataset", test_image_dir.parent.name)
    print(f"Checkpoint: {checkpoint_path} ({checkpoint_resolution.source})")

    test_dataset = CrackDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=get_transforms(mode="eval"),
        mode="eval",
        mask_suffix=data_config.get("mask_suffix", ".png"),
        image_extensions=data_config.get("image_extensions"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_config.get("batch_size", 1),
        shuffle=False,
    )

    eval_model_config = dict(model_config)
    eval_model_config["pretrained"] = False
    model = build_model(eval_model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    sweep_metrics = evaluate_threshold_sweep(
        model,
        test_loader,
        device,
        threshold_min=evaluation_config.get("threshold_min", 0.01),
        threshold_max=evaluation_config.get("threshold_max", 0.99),
        threshold_steps=evaluation_config.get("threshold_steps", 99),
        desc="EvalSweep(ODS/OIS)",
    )
    best_threshold = float(sweep_metrics["best_threshold"])

    fixed_metrics = evaluate_fixed_threshold(
        model,
        test_loader,
        device,
        threshold=fixed_threshold,
        desc="EvalFixed(@fixed_th)",
        include_topology_metrics=True,
    )
    threshold_probe_metrics = {}
    for probe_threshold in threshold_probes:
        if abs(probe_threshold - fixed_threshold) < 1e-12:
            probe_metrics = dict(fixed_metrics)
            probe_metrics.pop("cldice", None)
        else:
            probe_metrics = evaluate_fixed_threshold(
                model,
                test_loader,
                device,
                threshold=probe_threshold,
                desc=f"EvalProbe(@{probe_threshold:g})",
                include_topology_metrics=False,
            )
        threshold_probe_metrics[_threshold_key(probe_threshold)] = probe_metrics

    print("\n" + "=" * 60)
    print(f"Evaluation report for dataset: {data_name}")
    print("=" * 60)
    print("Protocol:")
    print("  Positive class:             crack foreground")
    print("  Fixed F1/P/R/IoU:           dataset-level foreground metrics")
    print("  ODS:                        best dataset-level foreground F1 over threshold sweep")
    print("  OIS:                        best per-image foreground F1, averaged over images")
    print("-" * 60)
    print(f"ODS:                         {sweep_metrics['ods']:.4f}")
    print(f"OIS:                         {sweep_metrics['ois']:.4f}")
    print(f"Best Threshold for ODS:       {best_threshold:.4f}")
    print(f"Best P for ODS:               {sweep_metrics['best_precision']:.4f}")
    print(f"Best R for ODS:               {sweep_metrics['best_recall']:.4f}")
    print("-" * 60)
    print(f"Fixed Threshold:              {fixed_metrics['threshold']:.4f} ({fixed_threshold_source})")
    print(f"P (@fixed_th):                {fixed_metrics['precision']:.4f}")
    print(f"R (@fixed_th):                {fixed_metrics['recall']:.4f}")
    print(f"F1 (@fixed_th):               {fixed_metrics['f1']:.4f}")
    print(f"mIoU (@fixed_th):             {fixed_metrics['miou']:.4f}")
    print(f"Crack IoU (@fixed_th):        {fixed_metrics['crack_iou']:.4f}")
    print(f"Background IoU (@fixed_th):   {fixed_metrics['bg_iou']:.4f}")
    print(f"clDice (@fixed_th):           {fixed_metrics.get('cldice', 0.0):.4f}")
    print("-" * 60)
    for threshold_key, probe_metrics in threshold_probe_metrics.items():
        print(f"Probe F1 (@{threshold_key}):              {probe_metrics['f1']:.4f}")
    print("=" * 60)

    experiment_root = get_experiment_root(output_config, experiment_name)
    report_prefix = output_config.get("report_prefix", f"eval_{experiment_name}")
    report_path = build_metrics_path(experiment_root, report_prefix)
    report_payload = {
        "experiment_name": experiment_name,
        "run_id": experiment_config.get("run_id", experiment_name),
        "attempt": experiment_config.get("attempt"),
        "seed": experiment_config.get("seed"),
        "config_path": str(config["_meta"]["config_path"]),
        "dataset": data_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_source": checkpoint_resolution.to_report(),
        "model_type": model_config.get("type", "b0"),
        "model_variant": variant,
        "protocol": {
            "positive_class": "crack_foreground",
            "prediction_binarization": "probability >= threshold",
            "ground_truth_binarization": "mask > 0.5",
            "primary_metric": "fixed_threshold_dataset_global_foreground_f1",
            "fixed_threshold": float(fixed_threshold),
            "fixed_aggregation": "dataset_global",
            "fixed_metrics": "dataset-global foreground crack metrics at fixed_threshold",
            "sweep_usage": "diagnostic_only",
            "ods": "diagnostic best dataset-level foreground crack F1 over one global threshold sweep",
            "ois": "best per-image foreground crack F1, averaged over images",
            "miou": "mean of foreground crack IoU and background IoU",
            "aggregation_note": "Primary fixed F1 uses dataset-global TP/FP/FN aggregation for the crack foreground class.",
            "fixed_threshold_source": fixed_threshold_source,
        },
        "metrics": {
            "sweep": {
                "ods": float(sweep_metrics["ods"]),
                "ois": float(sweep_metrics["ois"]),
                "best_f1": float(sweep_metrics["best_f1"]),
                "best_threshold": best_threshold,
                "best_precision": float(sweep_metrics["best_precision"]),
                "best_recall": float(sweep_metrics["best_recall"]),
                "threshold_min": float(sweep_metrics["threshold_min"]),
                "threshold_max": float(sweep_metrics["threshold_max"]),
                "threshold_steps": int(sweep_metrics["threshold_steps"]),
            },
            "fixed": {
                "threshold": float(fixed_metrics["threshold"]),
                "threshold_source": fixed_threshold_source,
                "aggregation": "dataset_global",
                "positive_class": "crack_foreground",
                "precision": float(fixed_metrics["precision"]),
                "recall": float(fixed_metrics["recall"]),
                "f1": float(fixed_metrics["f1"]),
                "miou": float(fixed_metrics["miou"]),
                "cldice": float(fixed_metrics.get("cldice", 0.0)),
                "crack_iou": float(fixed_metrics["crack_iou"]),
                "bg_iou": float(fixed_metrics["bg_iou"]),
                "tp": float(fixed_metrics["tp"]),
                "fp": float(fixed_metrics["fp"]),
                "fn": float(fixed_metrics["fn"]),
                "tn": float(fixed_metrics["tn"]),
            },
            "threshold_probes": {
                threshold_key: {
                    "threshold": float(probe_metrics["threshold"]),
                    "aggregation": "dataset_global",
                    "positive_class": "crack_foreground",
                    "precision": float(probe_metrics["precision"]),
                    "recall": float(probe_metrics["recall"]),
                    "f1": float(probe_metrics["f1"]),
                    "miou": float(probe_metrics["miou"]),
                    "crack_iou": float(probe_metrics["crack_iou"]),
                    "bg_iou": float(probe_metrics["bg_iou"]),
                    "tp": float(probe_metrics["tp"]),
                    "fp": float(probe_metrics["fp"]),
                    "fn": float(probe_metrics["fn"]),
                    "tn": float(probe_metrics["tn"]),
                }
                for threshold_key, probe_metrics in threshold_probe_metrics.items()
            },
        },
    }
    save_json_report(report_path, report_payload)
    print(f"Metrics report saved to: {report_path}")


def main() -> None:
    parser = build_config_parser(
        "Evaluate FT-Seg from an evaluation-specific YAML config.",
        default_config_path=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional eval checkpoint override: best, latest, or a checkpoint path.",
    )
    args = parser.parse_args()
    run(args.config, checkpoint_override=args.checkpoint)


if __name__ == "__main__":
    main()
