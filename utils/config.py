from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from utils.project import PROJECT_ROOT

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to load YAML experiment configs.") from exc


CONFIG_ROOT = PROJECT_ROOT / "configs"
CONFIG_FAMILY_MODEL_VARIANTS = {
    # Study registries under outputs/ (including topology_v3) explicitly set
    # model.variant and intentionally do not introduce new model families.
    "unet_light": "unet_light",
    "deeplabv3_resnet50": "deeplabv3_resnet50",
    "pure": "pure_segformer",
    "tcs_seg": "tcs_seg",
    "ab_tcs_no_tsm": "tcs_seg",
    "ab_tcs_no_waf": "tcs_seg",
    "ab_tcs_no_cal": "tcs_seg",
    "ab_tcs_tsm_only": "tcs_seg",
    "ab_tcs_waf_only": "tcs_seg",
    "ab_tcs_cal_only": "tcs_seg",
    "ab_tcs_tsm_waf": "tcs_seg",
    "ab_tcs_tsm_cal": "tcs_seg",
    "ab_tcs_waf_cal": "tcs_seg",
}


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_config_parser(
    description: str,
    default_config_path: str | Path | None = None,
) -> argparse.ArgumentParser:
    """Build a shared parser for config-driven entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_config_path(default_config_path)) if default_config_path else None,
        required=default_config_path is None,
        help="Path to the YAML config file.",
    )
    return parser


def parse_config_args(
    description: str,
    default_config_path: str | Path | None = None,
) -> argparse.Namespace:
    """Parse CLI args for config-driven entrypoints."""
    parser = build_config_parser(description, default_config_path=default_config_path)
    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got: {type(config).__name__}")

    config["_meta"] = {
        "config_path": resolved_path,
        "project_root": PROJECT_ROOT,
    }
    return config


def require_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise KeyError(f"Missing or invalid config section: '{section_name}'")
    return section


def get_config_family(config: Mapping[str, Any]) -> str | None:
    config_path = config.get("_meta", {}).get("config_path")
    if config_path is None:
        return None

    try:
        relative_parts = Path(config_path).resolve().relative_to(CONFIG_ROOT).parts
    except ValueError:
        return None

    return relative_parts[0] if relative_parts else None


def validate_model_variant(config: Mapping[str, Any], model_config: Mapping[str, Any]) -> str | None:
    """Validate config-family naming against the internal model variant.

    The user-facing config family (for example ``pure``) is intentionally
    separate from the implementation variant (for example ``pure_segformer``).
    This guard catches accidentally evaluating a checkpoint with the wrong
    model structure.
    """
    family = get_config_family(config)
    if family is None:
        return None

    expected_variant = CONFIG_FAMILY_MODEL_VARIANTS.get(family)
    if expected_variant is None:
        return family

    actual_variant = model_config.get("variant")
    if actual_variant != expected_variant:
        raise ValueError(
            f"Config family '{family}' expects model.variant='{expected_variant}', "
            f"got '{actual_variant}'."
        )

    return family


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_training_loss(loss: Mapping[str, Any]) -> None:
    """Reject ambiguous published-loss controls before spending a training run."""
    import math
    for name in ("main_weight", "author_cldice_weight", "skeleton_recall_weight", "scnp_weight"):
        weight = float(loss.get(name, 1.0 if name == "main_weight" else 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(loss.get("main_weight", 1.0)) == 0 and float(loss.get("scnp_weight", 0.0)) <= 0:
        raise ValueError("Disabling the raw main loss requires an SCNP replacement loss")
    if float(loss.get("author_cldice_weight", 0)) > 0 and float(loss.get("soft_cldice_weight", 0)) > 0:
        raise ValueError("Choose author clDice or legacy clDice explicitly, not both")
    window = int(loss.get("scnp_window", 3))
    if window <= 0 or window % 2 == 0:
        raise ValueError("SCNP requires a positive odd neighborhood size")
