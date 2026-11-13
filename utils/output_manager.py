from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import resolve_path
from utils.project import ensure_directory


def create_timestamp() -> str:
    """Create a filesystem-friendly timestamp for experiment artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_root(output_config, experiment_name: str) -> Path:
    """Resolve and create the root output directory for an experiment."""
    root_dir = output_config.get("root_dir", f"outputs/tcs_seg/adhoc/{experiment_name}")
    resolved_root = resolve_path(root_dir)
    ensure_directory(resolved_root)
    return resolved_root


def build_archive_checkpoint_path(root_dir: Path) -> Path:
    """Build a per-run archive checkpoint path under the experiment checkpoint directory."""
    checkpoint_dir = ensure_directory(root_dir / "checkpoints")
    return checkpoint_dir / f"best_{create_timestamp()}.pth"


def build_best_checkpoint_path(root_dir: Path) -> Path:
    """Build the stable global-best checkpoint path under the experiment checkpoint directory."""
    checkpoint_dir = ensure_directory(root_dir / "checkpoints")
    return checkpoint_dir / "best.pth"


def build_latest_checkpoint_path(root_dir: Path) -> Path:
    """Build the stable latest-run-best checkpoint path under the experiment checkpoint directory."""
    checkpoint_dir = ensure_directory(root_dir / "checkpoints")
    return checkpoint_dir / "latest.pth"


def build_prediction_path(root_dir: Path, prediction_prefix: str, image_name: str) -> Path:
    """Build the output path for a predicted mask under the predictions directory."""
    prediction_dir = ensure_directory(root_dir / "predictions")
    image_stem = Path(image_name).stem
    return prediction_dir / f"{prediction_prefix}_{image_stem}.png"


def build_metrics_path(root_dir: Path, report_prefix: str) -> Path:
    """Build the output path for a metrics report under the metrics directory."""
    metrics_dir = ensure_directory(root_dir / "metrics")
    return metrics_dir / f"{report_prefix}_{create_timestamp()}.json"


def build_log_path(root_dir: Path, filename: str) -> Path:
    """Build the output path for logs or config snapshots."""
    logs_dir = ensure_directory(root_dir / "logs")
    return logs_dir / filename


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON experiment report."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
