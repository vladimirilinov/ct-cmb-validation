from __future__ import annotations

import os
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file. If PyYAML is unavailable, return an empty dict
    so scripts can fall back to built-in defaults.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    except FileNotFoundError:
        return {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
