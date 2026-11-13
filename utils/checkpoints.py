from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.config import resolve_path


CHECKPOINT_SLOTS = {"best", "latest"}


@dataclass(frozen=True)
class CheckpointResolution:
    path: Path
    source: str
    value: str
    config_checkpoint: Path
    is_override: bool
    slot: str | None = None

    def to_report(self) -> dict[str, str | bool | None]:
        return {
            "source": self.source,
            "value": self.value,
            "config_checkpoint": str(self.config_checkpoint),
            "resolved_checkpoint": str(self.path),
            "is_override": self.is_override,
            "slot": self.slot,
        }


def resolve_checkpoint_override(
    config_checkpoint: str | Path,
    checkpoint_override: str | Path | None = None,
) -> CheckpointResolution:
    """Resolve eval checkpoint selection from config, slot override, or path override."""
    config_path = resolve_path(config_checkpoint)
    if checkpoint_override is None or str(checkpoint_override).strip() == "":
        return CheckpointResolution(
            path=config_path,
            source="config",
            value=str(config_checkpoint),
            config_checkpoint=config_path,
            is_override=False,
            slot=None,
        )

    override_value = str(checkpoint_override)
    if override_value in CHECKPOINT_SLOTS:
        return CheckpointResolution(
            path=config_path.parent / f"{override_value}.pth",
            source="slot_override",
            value=override_value,
            config_checkpoint=config_path,
            is_override=True,
            slot=override_value,
        )

    return CheckpointResolution(
        path=resolve_path(override_value),
        source="path_override",
        value=override_value,
        config_checkpoint=config_path,
        is_override=True,
        slot=None,
    )
