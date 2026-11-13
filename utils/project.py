from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_project_root_to_path() -> Path:
    """Ensure the project root is importable when a script is run directly."""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)
    return PROJECT_ROOT


def resolve_project_path(*parts: PathLike) -> Path:
    """Build an absolute path from the project root."""
    path = PROJECT_ROOT
    for part in parts:
        path /= Path(part)
    return path


def ensure_directory(path: PathLike) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
